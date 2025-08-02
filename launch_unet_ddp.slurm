#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J pennycress-unet-ddp 
#SBATCH -N 2
#SBATCH -t 30:00
#SBATCH -p extended 
#SBATCH -q debug
#SBATCH -C nvme
#SBATCH --gres=gpu:8
#SBATCH -o logs/%x.out # Out Path
#SBATCH -e logs/%x.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

set -eo pipefail

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-cv/

export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

# miopen settings (needed for AMD GPUs)
export PYTORCH_ROCM_ARCH=gfx90a
export MIOPEN_FIND_MODE="NORMAL"
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_DEBUG_CONV_DIRECT=1
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DISABLE_CACHE=1  # Use only if the cache is corrupted and can't be used
export HIP_LAUNCH_BLOCKING=1

srun --ntasks-per-node=8 --gpus-per-node=8 --nodes=2 \
    --export=ALL \
    python -u scripts/train_unet_ddp.py --batch_size 64 --num_iters 100_000    
