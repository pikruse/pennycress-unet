import argparse
import os
import subprocess
import sys
import time
from importlib import reload
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from DGXutils import GetFileNames
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.GetLR import get_lr
from utils.Sam3Custom import Sam3Custom
import utils.TileGenerator as TG
import utils.WeightedCrossEntropy as WeightedCrossEntropy

os.environ["MIOPEN_FIND_MODE"] = "NORMAL"


def extract_master_addr():
    try:
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
        world_size=world_size,
    )

    return torch.device("cuda:0"), local_rank, rank, world_size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--distance_weights",
        choices=("none", "up", "down"),
        default="none",
        help="Pixel weighting mode: none disables distance weights, up weights seed borders up, down weights seed borders down.",
    )
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num_iters", type=int, default=150_000)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--batches_per_eval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--checkpoint_dir", type=Path, default=None)
    p.add_argument(
        "--pretrained_model_name",
        default=os.getenv("SAM3_MODEL_NAME", "facebook/sam3"),
    )
    p.add_argument("--encoder_image_size", type=int, default=128)
    p.add_argument("--decoder_channels", type=int, default=256)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true")
    return p.parse_args()


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank) -> bool:
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
    return Path(f"checkpoints/sam3_{mode}")


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
        "architecture": "sam3",
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "kwargs": model_kwargs,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "train_ids": train_idx,
        "val_ids": val_idx,
        "distance_weights": distance_weights,
        "border_weight": border_weight,
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


def pretrained_model_slug(pretrained_model_name):
    return pretrained_model_name.split("/")[-1].replace("-", "_")


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
        f"lora:{not args.no_lora}",
    ]


def main():
    print(f"[Rank {os.environ.get('SLURM_PROCID', '?')}] Available devices: {torch.cuda.device_count()}")
    args = parse_args()
    device, local_rank, rank, world_size = setup_ddp()
    print(f"[Rank {os.environ.get('SLURM_PROCID')}] Visible CUDA devices: {torch.cuda.device_count()}")
    use_distance_weights, border_weight = distance_weight_options(args.distance_weights)
    if is_main(rank):
        print(f"distance_weights={args.distance_weights}")

    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir(args.distance_weights)
    if rank == 0:
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model_kwargs = {
        "pretrained_model_name": args.pretrained_model_name,
        "num_labels": 3,
        "pretrained": True,
        "ignore_mismatched_sizes": True,
        "local_files_only": os.getenv("HF_HUB_OFFLINE", "0") == "1",
        "use_lora": not args.no_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "encoder_image_size": args.encoder_image_size,
        "decoder_channels": args.decoder_channels,
    }
    model = Sam3Custom(**model_kwargs).to(device)
    trainable_params, total_params = model.trainable_parameter_counts()
    if is_main(rank):
        print(
            f"SAM3 parameters: trainable={trainable_params:,} "
            f"total={total_params:,} trainable_pct={100 * trainable_params / total_params:.3f}%"
        )
    model_kwargs["config_dict"] = model.config_dict
    model_kwargs["pretrained"] = False
    model_kwargs["local_files_only"] = True
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])

    img_path = "data/train/train_images_by_pod/"
    mask_path = "data/train/train_masks_by_pod/"

    img_names = GetFileNames(img_path, extension=".png")
    mask_names = GetFileNames(mask_path, extension=".png")

    pennycress_images = []
    pennycress_masks = []
    n_pad = 128

    if is_main(rank):
        print(f"loading {len(img_names)} images and masks from {img_path} and {mask_path}")
    for img_name in mask_names:
        image = np.array(Image.open(img_path + img_name))
        image = image[:, :, :3] / 255.0
        image = np.pad(image, ((n_pad, n_pad), (n_pad, n_pad), (0, 0)), "edge")
        pennycress_images.append(image)

        mask = np.array(Image.open(mask_path + img_name))
        mask = mask / 255.0
        mask = np.pad(mask, ((n_pad, n_pad), (n_pad, n_pad), (0, 0)), "constant", constant_values=0)
        pennycress_masks.append(mask)

    multiclass_masks = []
    for mask in pennycress_masks:
        bg = mask.sum(-1) == 0
        mask = np.concatenate([bg.reshape(*bg.shape, 1), mask], axis=-1)
        multiclass_masks.append(mask)

    if is_main(rank):
        print(f"loaded {len(pennycress_images)} images and masks")
        print("Creating data generators...")

    images = pennycress_images
    masks = multiclass_masks
    tile_size = 128
    train_prop = 0.8

    train_idx, val_idx = make_shared_split(len(images), train_prop, args.seed, rank)

    train_generator = TG.TileGenerator(
        images=[images[i] for i in train_idx],
        masks=[masks[i] for i in train_idx],
        tile_size=tile_size,
        split="train",
        n_pad=n_pad,
        distance_weights=use_distance_weights,
        border_weight=border_weight,
    )
    val_generator = TG.TileGenerator(
        images=[images[i] for i in val_idx],
        masks=[masks[i] for i in val_idx],
        tile_size=tile_size,
        split="val",
        n_pad=n_pad,
        distance_weights=use_distance_weights,
        border_weight=border_weight,
    )

    train_sampler = DistributedSampler(train_generator, shuffle=True)
    val_sampler = DistributedSampler(val_generator, shuffle=False)
    if is_main(rank):
        print("created train and val generators.")
        print("Starting training...")

    train_loader = make_loader(train_generator, args.batch_size, train_sampler, args.num_workers)
    val_loader = make_loader(val_generator, args.batch_size, val_sampler, args.num_workers)

    reload(WeightedCrossEntropy)
    if use_distance_weights:
        loss_function = WeightedCrossEntropy.WeightedCrossEntropy(device=device)
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("SAM3 model has no trainable parameters.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=0.001)

    warmup_iters = 1000
    lr_decay_iters = 90000
    max_lr = 1e-3
    min_lr = 1e-5
    max_iters = args.num_iters
    batches_per_eval = args.batches_per_eval
    eval_interval = args.eval_interval
    early_stop = 50

    best_val_loss = None
    last_improved = 0
    iter_num = 0
    t0 = time.time()

    if is_main(rank):
        model_name = f"sam3_{pretrained_model_slug(args.pretrained_model_name)}"
        run_name = wandb_run_name(model_name, args, world_size, max_lr)
        wandb.init(
            project="pennycress-unet",
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            group=f"{model_name}_dw-{args.distance_weights}",
            tags=wandb_run_tags(model_name, args, world_size),
            config={
                "model_name": model_name,
                "pretrained_model_name": args.pretrained_model_name,
                "distance_weights": args.distance_weights,
                "border_weight": border_weight,
                "loss_type": "dense_cross_entropy",
                "checkpoint_dir": str(checkpoint_dir),
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
                "use_lora": not args.no_lora,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "encoder_image_size": args.encoder_image_size,
                "decoder_channels": args.decoder_channels,
                "trainable_params": trainable_params,
                "total_params": total_params,
            },
        )
        wandb.define_metric("iter_num")
        wandb.define_metric("*", step_metric="iter_num")

    while True:
        train_sampler.set_epoch(iter_num)

        model.eval()
        with torch.no_grad():
            train_loss_sum, val_loss_sum = 0.0, 0.0
            train_count, val_count = 0, 0
            eval_batches = 0
            with tqdm(total=batches_per_eval, desc=" Eval", disable=not is_main(rank)) as pbar:
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

        if is_main(rank):
            log_wandb(
                {
                    "eval/train_loss": train_loss,
                    "eval/val_loss": val_loss,
                    "val_loss": val_loss,
                },
                step=iter_num,
            )

        if rank == 0 and iter_num > 0:
            save_checkpoint(
                checkpoint_dir,
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

        if best_val_loss is None:
            best_val_loss = val_loss

        should_stop = False
        if iter_num > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                last_improved = 0
                if is_main(rank):
                    print(f"*** validation loss improved: {best_val_loss:.4e} ***")
            else:
                last_improved += 1
                if is_main(rank):
                    print(f"validation has not improved in {last_improved} steps")
            if last_improved > early_stop:
                if is_main(rank):
                    print()
                    print(f"*** no improvement for {early_stop} steps, stopping ***")
                should_stop = True

        if should_stop:
            break

        hit_nan = False
        train_batches = 0
        with tqdm(total=eval_interval, desc="Train", disable=not is_main(rank)) as pbar:
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                loss = loss_function(model(xb), yb)

                nan_flag = torch.tensor(int(torch.isnan(loss).item()), device=device)
                dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
                if nan_flag.item():
                    if is_main(rank):
                        print("loss is NaN, stopping")
                    hit_nan = True
                    break

                lr = get_lr(
                    it=iter_num,
                    warmup_iters=warmup_iters,
                    lr_decay_iters=lr_decay_iters,
                    max_lr=max_lr,
                    min_lr=min_lr,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                next_iter_num = iter_num + 1

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

                pbar.update(1)
                train_batches += 1
                iter_num = next_iter_num
                if train_batches >= eval_interval or iter_num >= max_iters:
                    break

        if hit_nan:
            break

        if iter_num >= max_iters:
            if rank == 0:
                save_checkpoint(
                    checkpoint_dir,
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
                elapsed_hours = (time.time() - t0) / 3600.0
                print(f"maximum iterations reached: {max_iters} after {elapsed_hours:.2f}h")
            break
        pbar.close()

    if is_main(rank):
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    main()
