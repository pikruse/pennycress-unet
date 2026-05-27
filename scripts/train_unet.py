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
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    
    return torch.device("cuda:0"), local_rank, rank, world_size

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--distance_weights",
        choices=("none", "up", "down"),
        default="up",
        help="Pixel weighting mode: none disables distance weights, up weights seed borders up, down weights seed borders down.",
    )
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num_iters", type=int, default=50_000)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--batches_per_eval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--checkpoint_dir", type=Path, default=None)
    return p.parse_args()

def cleanup_ddp():
    """clean up torch distributed backend"""
    dist.destroy_process_group()

def is_main(rank) -> bool:
    """check if current process is main"""
    return rank == 0

def make_shared_split(num_images, train_prop, seed, rank):
    split = [None]
    if is_main(rank):
        rng = np.random.default_rng(seed)
        p = rng.permutation(num_images)
        n_train = int(train_prop * num_images)
        split[0] = (p[:n_train], p[n_train:])
    dist.broadcast_object_list(split, src=0)
    train_idx, val_idx = split[0]
    return np.asarray(train_idx), np.asarray(val_idx)

def reduce_eval_losses(train_loss_sum, train_count, val_loss_sum, val_count, device):
    totals = torch.tensor(
        [train_loss_sum, train_count, val_loss_sum, val_count],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    train_loss = totals[0] / totals[1].clamp_min(1)
    val_loss = totals[2] / totals[3].clamp_min(1)
    return train_loss.item(), val_loss.item()

def reduce_scalar_mean(value, device):
    value = value.detach().to(device=device, dtype=torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value.item()

def make_loader(dataset, batch_size, sampler, num_workers):
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": True,
        "sampler": sampler,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)

def distance_weight_options(mode):
    if mode == "none":
        return False, None
    if mode == "up":
        return True, 5
    if mode == "down":
        return True, 0.5
    raise ValueError(f"Unsupported distance weight mode: {mode}")

def default_checkpoint_dir(mode):
    return Path(f"checkpoints/unet_{mode}")

def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    model_kwargs,
    iter_num,
    best_val_loss,
    train_idx,
    val_idx,
    distance_weights,
    border_weight,
):
    checkpoint = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'kwargs': model_kwargs,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'train_ids': train_idx,
        'val_ids': val_idx,
        'distance_weights': distance_weights,
        'border_weight': border_weight,
    }
    checkpoint_path = checkpoint_dir / f"checkpoint_{iter_num:06d}.pt"
    torch.save(checkpoint, checkpoint_path)

def log_wandb(metrics, step):
    metrics = dict(metrics)
    metrics["iter_num"] = step
    metrics["optimizer_step"] = step
    wandb.log(metrics, step=step)

def compact_number(value):
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)

def wandb_run_name(model_name, args, world_size, max_lr):
    return "_".join([
        model_name,
        f"dw-{args.distance_weights}",
        f"bs{args.batch_size}",
        f"gpus{world_size}",
        f"iters{compact_number(args.num_iters)}",
        f"lr{max_lr:g}",
        f"seed{args.seed}",
    ])

def wandb_run_tags(model_name, args, world_size):
    return [
        model_name,
        f"distance_weights:{args.distance_weights}",
        f"batch_size:{args.batch_size}",
        f"gpus:{world_size}",
        f"seed:{args.seed}",
    ]

def main():
    print(f"[Rank {os.environ.get('SLURM_PROCID', '?')}] Available devices: {torch.cuda.device_count()}")
    args = parse_args()
    device, local_rank, rank, world_size = setup_ddp()
    print(f"[Rank {os.environ.get('SLURM_PROCID')}] Visible CUDA devices: {torch.cuda.device_count()}")
    use_distance_weights, border_weight = distance_weight_options(args.distance_weights)
    if is_main(rank):
        print(f"distance_weights={args.distance_weights}")

    # checkpoint setup
    CHECKPOINT_DIR = args.checkpoint_dir or default_checkpoint_dir(args.distance_weights)
    if rank == 0:
        CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

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

    # create one train/val split on rank 0 and share it with every DDP process
    train_idx, val_idx = make_shared_split(len(images), train_prop, args.seed, rank)

    # instantiate tilegenerator class
    train_generator = TG.TileGenerator(
        images=[images[i] for i in train_idx],
        masks=[masks[i] for i in train_idx], 
        tile_size=tile_size, 
        split='train',
        n_pad = n_pad,
        distance_weights=use_distance_weights,
        border_weight=border_weight
        )

    val_generator = TG.TileGenerator(
        images=[images[i] for i in val_idx],
        masks=[masks[i] for i in val_idx], 
        tile_size=tile_size, 
        split='val',
        n_pad = n_pad,
        distance_weights=use_distance_weights,
        border_weight=border_weight
        )

    # init ddp samplers
    train_sampler = DistributedSampler(train_generator, shuffle=True)
    val_sampler = DistributedSampler(val_generator, shuffle=False)
    if is_main(rank):
        print(f'created train and val generators.') 
        print(f'Starting training...')

    train_loader = make_loader(train_generator, args.batch_size, train_sampler, args.num_workers)
    val_loader = make_loader(val_generator, args.batch_size, val_sampler, args.num_workers)

    #### Train Model ####
    # define our loss function, optimizer
    reload(WeightedCrossEntropy)
    if use_distance_weights:
        loss_function = WeightedCrossEntropy.WeightedCrossEntropy(device=device)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # lr options
    warmup_iters = 1000
    lr_decay_iters = 90000
    max_lr = 1e-3
    min_lr = 1e-5
    max_iters = args.num_iters
    batches_per_eval = args.batches_per_eval
    eval_interval = args.eval_interval
    early_stop = 50

   # non-customizable options
    best_val_loss = None # initialize best validation loss
    last_improved = 0 # start early stopping counter
    iter_num = 0 # initialize iteration counter
    t0 = time.time() # start timer

    # training loop
    # refresh log
    if is_main(rank):
        run_name = wandb_run_name("unet", args, world_size, max_lr)
        wandb.init(
            project="pennycress-unet",
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            group=f"unet_dw-{args.distance_weights}",
            tags=wandb_run_tags("unet", args, world_size),
            config={
                "model_name": "unet",
                "distance_weights": args.distance_weights,
                "border_weight": border_weight,
                "checkpoint_dir": str(CHECKPOINT_DIR),
                "batch_size": args.batch_size,
                "world_size": world_size,
                "num_iters": args.num_iters,
                "eval_interval": args.eval_interval,
                "batches_per_eval": args.batches_per_eval,
                "log_interval": args.log_interval,
                "num_workers": args.num_workers,
                "seed": args.seed,
                "warmup_iters": warmup_iters,
                "lr_decay_iters": lr_decay_iters,
                "max_lr": max_lr,
                "min_lr": min_lr,
            },
        )
        wandb.define_metric("iter_num")
        wandb.define_metric("*", step_metric="iter_num")

    # keep training until break
    while True:
        #
        # checkpoint
        #
        # set epoch for sampler
        train_sampler.set_epoch(iter_num)

        # estimate loss. All ranks must participate because this is a DDP model.
        model.eval()
        with torch.no_grad():
            train_loss_sum, val_loss_sum = 0.0, 0.0
            train_count, val_count = 0, 0
            eval_batches = 0
            with tqdm(total=batches_per_eval, desc=' Eval', disable=not is_main(rank)) as pbar:
                for (xbt, ybt), (xbv, ybv) in zip(train_loader, val_loader):
                    xbt, ybt = xbt.to(device, non_blocking=True), ybt.to(device, non_blocking=True)
                    xbv, ybv = xbv.to(device, non_blocking=True), ybv.to(device, non_blocking=True)
                    train_loss_sum += loss_function(model(xbt), ybt).item()
                    val_loss_sum += loss_function(model(xbv), ybv).item()
                    train_count += 1
                    val_count += 1
                    eval_batches += 1
                    pbar.update(1)
                    if eval_batches >= batches_per_eval:
                        break
            train_loss, val_loss = reduce_eval_losses(
                train_loss_sum,
                train_count,
                val_loss_sum,
                val_count,
                device,
            )
        model.train()

        # wandb logging
        if is_main(rank):
            log_wandb(
                {
                    "eval/train_loss": train_loss,
                    "eval/val_loss": val_loss,
                    "val_loss": val_loss,
                },
                step=iter_num,
            )

        # checkpoint model
        if rank == 0 and iter_num > 0:
            save_checkpoint(
                CHECKPOINT_DIR,
                model,
                optimizer,
                model_kwargs,
                iter_num,
                best_val_loss,
                train_idx,
                val_idx,
                args.distance_weights,
                border_weight,
            )

        # book keeping
        if best_val_loss is None:
            best_val_loss = val_loss

        should_stop = False
        if iter_num > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                last_improved = 0
                if is_main(rank):
                    print(f'*** validation loss improved: {best_val_loss:.4e} ***')
            else:
                last_improved += 1
                if is_main(rank):
                    print(f'validation has not improved in {last_improved} steps')
            if last_improved > early_stop:
                if is_main(rank):
                    print()
                    print(f'*** no improvement for {early_stop} steps, stopping ***')
                should_stop = True

        if should_stop:
            break
        
        # --------
        # backprop
        # --------

        # iterate over batches
        hit_nan = False
        train_batches = 0
        with tqdm(total=eval_interval, desc='Train', disable=not is_main(rank)) as pbar:
            for xb, yb in train_loader:

                # update the model
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

                loss = loss_function(model(xb), yb)

                nan_flag = torch.tensor(int(torch.isnan(loss).item()), device=device)
                dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
                if nan_flag.item():
                    if is_main(rank):
                        print('loss is NaN, stopping')
                    hit_nan = True
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
                next_iter_num = iter_num + 1

                # update wandb
                if next_iter_num == 1 or next_iter_num % args.log_interval == 0 or next_iter_num >= max_iters:
                    reduced_loss = reduce_scalar_mean(loss, device)
                else:
                    reduced_loss = None
                if is_main(rank) and reduced_loss is not None:
                    log_wandb(
                        {
                            "train/loss": reduced_loss,
                            "train_loss": reduced_loss,
                            "train/lr": lr,
                            "lr": lr,
                        },
                        step=next_iter_num,
                    )

                # update book keeping
                pbar.update(1)
                train_batches += 1
                iter_num = next_iter_num
                if train_batches >= eval_interval or iter_num >= max_iters:
                    break

        if hit_nan:
            break

        # break once hitting max_iters
        if iter_num >= max_iters:
            if rank == 0:
                save_checkpoint(
                    CHECKPOINT_DIR,
                    model,
                    optimizer,
                    model_kwargs,
                    iter_num,
                    best_val_loss,
                    train_idx,
                    val_idx,
                    args.distance_weights,
                    border_weight,
                )
                print(f'maximum iterations reached: {max_iters}')
            break
        pbar.close()
    if is_main(rank):
        wandb.finish()
    cleanup_ddp()
        
if __name__ == "__main__":
    main()
