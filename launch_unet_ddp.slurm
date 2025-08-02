#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J pennycress-unet-ddp 
#SBATCH -N 1
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

export PYTORCH_ROCM_ARCH=gfx90a
export MIOPEN_FIND_MODE="NORMAL"
export MIOPEN_DEBUG_CONV_DIRECT=1

# debug miopen
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_ENABLE_LOGGING_ENV=1
export MIOPEN_LOG_LEVEL=5

# Skip problematic kernels or fallback to more stable algorithms
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_DEBUG_CONV_DIRECT=1
# You can also try disabling Winograd convolutions:
export MIOPEN_DEBUG_CONV_WINOGRAD=0

# Prevent crashing on autotune write errors
export MIOPEN_DISABLE_CACHE=1  # Use only if the cache is corrupted and can't be used

# Optional: Enable HIP/ROCm debugging info
export HCC_DEBUG=1
export HIP_LAUNCH_BLOCKING=1

srun python -u scripts/train_unet_ddp.py --batch_size 64 --num_iters 100_000    
