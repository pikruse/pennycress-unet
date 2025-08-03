# import necessary libraries
import os, sys, glob, argparse
import subprocess
from pathlib import Path
import wandb
import socket
import math
import time
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

from argparse import ArgumentParser
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from PIL import Image
from importlib import reload # when you make changes to a .py, force reload imports
from DGXutils import GetFileNames, GetLowestGPU

# make path start at root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.GetLR import get_lr
import utils.Train as Train
import utils.Plot as Plot
import utils.WeightedCrossEntropy as WeightedCrossEntropy
import utils.BuildUNet as BuildUNet
import utils.TileGenerator as TG
import utils.DistanceMap as DistanceMap

os.environ["MIOPEN_FIND_MODE"] = "NORMAL"

#### Helper Setup ###

def extract_master_addr():
    try:
        # Use scontrol to get the hostname of the first node
        nodelist = os.environ["SLURM_NODELIST"]
        node = subprocess.check_output(
            ["scontrol", "show", "hostname", nodelist]
        ).decode().splitlines()[0]
        return node
    except Exception as e:
        print(f"[WARN] Failed to extract master address from SLURM_NODELIST: {e}")
        return "localhost"


def setup_ddp():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    os.environ.setdefault("MASTER_PORT", "6000")
    os.environ.setdefault("MASTER_ADDR", subprocess.check_output(
        ["scontrol", "show", "hostname", os.environ["SLURM_NODELIST"]]
    ).decode().splitlines()[0])

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Since each process sees only one GPU, we always use cuda:0
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    print(f"[Rank {rank}] Initialized on device {device}")
    return device, 0, rank, world_size

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num_iters", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()

def cleanup_ddp():
    """clean up torch distributed backend"""
    dist.destroy_process_group()

def is_main(rank) -> bool:
    """check if current process is main"""
    return rank == 0

def main():
    print(f"[Rank {os.environ.get('SLURM_PROCID', '?')}] Available devices: {torch.cuda.device_count()}")
    args = parse_args()
    device, local_rank, rank, world_size = setup_ddp()
    print(f"[Rank {os.environ.get('SLURM_PROCID')}] Visible CUDA devices: {torch.cuda.device_count()}")

    #### model init ####
    model_kwargs = {
        'layer_sizes': [32, 64, 128, 256, 512],
        'in_channels': 3,
        'out_channels': 4,
        'conv_per_block': 3,
        'dropout_rate': 0.1,
        'hidden_activation': torch.nn.GELU(),
        'output_activation': None
    }
    unet = BuildUNet.UNet(**model_kwargs).to(device)
    model = torch.nn.parallel.DistributedDataParallel(unet,
                                                      device_ids=[0]) 

    #### Data Loading ####
    # define options
    img_path = 'data/train/train_images_by_pod/'
    mask_path = 'data/train/train_masks_by_pod/'

    # load images and masks into list
    img_names = GetFileNames(img_path,
                            extension=".png")
    mask_names = GetFileNames(mask_path,
                                extension=".png")

    pennycress_images = []
    pennycress_masks = []
    n_pad = 128   # padding for images

    if is_main(rank):
        print(f'loading {len(img_names)} images and masks from {img_path} and {mask_path}')
    for img_name in mask_names:
        # load image
        image = np.array(Image.open(img_path + img_name))
        image = (image[:, :, :3] / 255.0) # normalize image
        image = np.pad(image, ((n_pad, n_pad), (n_pad, n_pad), (0, 0)), 'edge') # pad image w/ values on edge
        pennycress_images.append(image)

        # load mask
        mask = np.array(Image.open(mask_path + img_name))
        mask = (mask / 255.0) # normalize mask
        mask = np.pad(mask, ((n_pad, n_pad), (n_pad, n_pad), (0, 0)), 'constant', constant_values=0) # pad mask w/ constant 0 value
        pennycress_masks.append(mask)

    # split masks into wing and pod and seed
    wings = [m[:, :, 0] > 0.5 for m in pennycress_masks] # take red channel and booleanize
    envelopes = [m[:, :, 1:].sum(-1) > 0.5 for m in pennycress_masks] # "... blue ..."
    seeds = [m[:, :, 2] > 0.5 for m in pennycress_masks] # "... green ..."

    # create list for multiclass masks
    multiclass_masks = []

    # add additional channel to pennycress masks for one-hot encoding
    for mask in pennycress_masks:
        bg = mask.sum(-1) == 0 # booleanize background
        mask = np.concatenate([bg.reshape(*bg.shape, 1), mask], axis=-1) # add background channel
        multiclass_masks.append(mask)
    if is_main(rank):
        print(f'loaded {len(pennycress_images)} images and masks')
        print(f'Creating data generators...')
    #### Data Generation ####
    # options
    images = pennycress_images
    masks = multiclass_masks
    tile_size = 128
    train_prop = 0.8
    distance_weights = True

    #create train/val splits by image
    p = np.random.permutation(len(images))
    train_idx = p[:int(train_prop*len(images))]
    val_idx = p[int(train_prop*len(images)):]

    # instantiate tilegenerator class
    train_generator = TG.TileGenerator(
        images=[images[i] for i in train_idx],
        masks=[masks[i] for i in train_idx], 
        tile_size=tile_size, 
        split='train',
        n_pad = n_pad,
        distance_weights=True,
        border_weight=5
        )

    val_generator = TG.TileGenerator(
        images=[images[i] for i in val_idx],
        masks=[masks[i] for i in val_idx], 
        tile_size=tile_size, 
        split='val',
        n_pad = n_pad,
        distance_weights=True,
        border_weight=5
        )

    # init ddp samplers
    train_sampler = DistributedSampler(train_generator, shuffle=True)
    val_sampler = DistributedSampler(val_generator, shuffle=False)
    if is_main(rank):
        print(f'created train and val generators.') 
        print(f'Starting training...')
    #### Train Model ####
    # define our loss function, optimizer
    reload(WeightedCrossEntropy)
    if distance_weights:
        loss_function = WeightedCrossEntropy.WeightedCrossEntropy(device=device)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)

    # log options
    chckpnt_path = 'checkpoints/checkpoint_{0}_up.pt'

    # lr options
    warmup_iters = 1000
    lr_decay_iters = 90000
    max_lr = 1e-3
    min_lr = 1e-5
    max_iters = 150000
    batches_per_eval = 1000
    eval_interval = 1000
    early_stop = 50

   # non-customizable options
    best_val_loss = None # initialize best validation loss
    last_improved = 0 # start early stopping counter
    iter_num = 0 # initialize iteration counter
    t0 = time.time() # start timer

    # training loop
    # refresh log
    if is_main(rank):
        wandb.init(
            project="pennycress-unet",
            entity=os.getenv("WANDB_ENTITY"),
        )

    # keep training until break
    while True:
        #
        # checkpoint
        #
        # set epoch for sampler
        train_sampler.set_epoch(iter_num)

        # shuffle dataloaders
        train_loader = DataLoader(
            train_generator, 
            batch_size=args.batch_size,
            pin_memory=True,
            sampler=train_sampler
        )

        val_loader = DataLoader(
            val_generator, 
            batch_size=args.batch_size,
            pin_memory=True,
            sampler=val_sampler)

        # estimate loss
        if is_main(rank):
            model.eval()
            with torch.no_grad():
                train_loss, val_loss = 0, 0
                with tqdm(total=batches_per_eval, desc=' Eval', disable=not is_main(rank)) as pbar:
                    for (xbt, ybt), (xbv, ybv) in zip(train_loader, val_loader):
                        xbt, ybt = xbt.to(device), ybt.to(device)
                        xbv, ybv = xbv.to(device), ybv.to(device)
                        train_loss += loss_function(model(xbt), ybt).item()
                        val_loss += loss_function(model(xbv), ybv).item()
                        pbar.update(1)  
                        if pbar.n == pbar.total:
                            break
                train_loss /= batches_per_eval
                val_loss /= batches_per_eval
            model.train()

            # wandb logging
            if is_main(rank):
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "iter_num": iter_num
                })

            # checkpoint model
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'kwargs': model_kwargs,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'train_ids': train_idx,
                    'val_ids': val_idx,
                }
                torch.save(checkpoint, chckpnt_path.format(iter_num))

            # book keeping
            if best_val_loss is None:
                best_val_loss = val_loss

            if iter_num > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    last_improved = 0
                    print(f'*** validation loss improved: {best_val_loss:.4e} ***')
                else:
                    last_improved += 1
                    print(f'validation has not improved in {last_improved} steps')
                if last_improved > early_stop:
                    print()
                    print(f'*** no improvement for {early_stop} steps, stopping ***')
                    break
        
        # --------
        # backprop
        # --------

        # iterate over batches
        with tqdm(total=eval_interval, desc='Train', disable=not is_main(rank)) as pbar:
            for xb, yb in train_loader:

                # update the model
                xb, yb = xb.to(device), yb.to(device)

                loss = loss_function(model(xb), yb)

                if torch.isnan(loss):
                    print('loss is NaN, stopping')
                    break
                
                # apply learning rate schedule
                lr = get_lr(it = iter_num,
                            warmup_iters = warmup_iters, 
                            lr_decay_iters = lr_decay_iters, 
                            max_lr = max_lr, 
                            min_lr = min_lr)
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # update wandb
                if is_main(rank):
                    wandb.log({
                        "train_loss": loss.item(),
                        "lr": lr,
                        "iter_num": iter_num
                    })

                # update book keeping
                pbar.update(1)
                iter_num += 1
                if pbar.n == pbar.total:
                    break

        # break once hitting max_iters
        if iter_num > max_iters:
            print(f'maximum iterations reached: {max_iters}')
            break
        pbar.close()
    cleanup_ddp()
        
if __name__ == "__main__":
    main()