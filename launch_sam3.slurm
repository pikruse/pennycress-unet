#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J sam3
#SBATCH -N 1 
#SBATCH -t 14:00:00
#SBATCH -p extended
#SBATCH -C nvme
#SBATCH -o logs/%x-%j.out # Out Path
#SBATCH -e logs/%x-%j.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

## settings, environment, and module loads ###
set -eo pipefail

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-cv/

### wandb settings ### 
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_ENTITY=${WANDB_ENTITY:-jail-ai}
export WANDB_HTTP_TIMEOUT=90

### pretrained model cache settings ###
export MODEL_CACHE_ROOT="${PWD}/nnu_net/cache/model_weights"
export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${MODEL_CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${MODEL_CACHE_ROOT}/xdg"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME/hub/checkpoints" "$XDG_CACHE_HOME"

python - <<'PY'
import os

from utils.sam3Custom import sam3Custom

model_name = os.getenv("SAM3_MODEL_NAME, "facebook/sam3")
model = sam3Custom(
    pretrained_model_name=model_name,
    num_labels=3,
    pretrained=True,
    ignore_mismatched_sizes=True,
    peft=True
)
num_params = sum(p.numel() for p in model.parameters())
print(f"Warmed SAM3 weights from {model_name}; parameters={num_params:,}")
PY

export HF_HUB_OFFLINE=1

### network / rccl settings ###
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=hsn0
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1

### miopen settings ###
export MIOPEN_USER_DB_PATH="${PWD}/nnu_net/cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=FAST
export MIOPEN_FIND_ENFORCE=NONE

### srun ###
NGPUS=$(($SLURM_NNODES * 8))
DISTANCE_WEIGHTS=${DISTANCE_WEIGHTS:-none}
MASK2FORMER_MODEL_NAME=${MASK2FORMER_MODEL_NAME:-facebook/sam3}
BATCH_SIZE=${BATCH_SIZE:-1}
if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    CHECKPOINT_DIR="checkpoints/sam3_${DISTANCE_WEIGHTS}"
fi

srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train_sam3.py \
        --batch_size "${BATCH_SIZE}" \
        --distance_weights "${DISTANCE_WEIGHTS}" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --pretrained_model_name "${SAM3_MODEL_NAME}" \
        --num_iters 150_000 \
        --batches_per_eval 100 \
        --log_interval 10 \
        --num_workers 4
