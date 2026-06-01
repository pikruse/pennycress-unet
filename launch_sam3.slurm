#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J sam3
#SBATCH -N 1 
#SBATCH -t 10:00:00
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

### optional local secrets ###
HF_ENV_FILE=${HF_ENV_FILE:-.env}
if [[ -f "$HF_ENV_FILE" ]]; then
    set -a
    source "$HF_ENV_FILE"
    set +a
fi

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Put HF_TOKEN=hf_... in ${HF_ENV_FILE}, or export HF_TOKEN before sbatch." >&2
    exit 1
fi

export SAM3_MODEL_NAME=${SAM3_MODEL_NAME:-facebook/sam3}
export SAM3_ENCODER_IMAGE_SIZE=${SAM3_ENCODER_IMAGE_SIZE:-128}
export SAM3_DECODER_CHANNELS=${SAM3_DECODER_CHANNELS:-256}
export SAM3_LORA_R=${SAM3_LORA_R:-8}
export SAM3_LORA_ALPHA=${SAM3_LORA_ALPHA:-16}
export SAM3_LORA_DROPOUT=${SAM3_LORA_DROPOUT:-0.05}
export BATCH_SIZE=${BATCH_SIZE:-1}

python - <<'PY'
import importlib.util
import os

import transformers

if not hasattr(transformers, "Sam3Model"):
    raise RuntimeError(
        "The active environment does not expose transformers.Sam3Model. "
        "Install a Transformers release with SAM3 support before launching this job."
    )
if importlib.util.find_spec("peft") is None:
    raise RuntimeError("The active environment is missing peft, which is required for SAM3 LoRA.")

from utils.Sam3Custom import Sam3Custom

model_name = os.getenv("SAM3_MODEL_NAME", "facebook/sam3")
model = Sam3Custom(
    pretrained_model_name=model_name,
    num_labels=3,
    pretrained=True,
    ignore_mismatched_sizes=True,
    use_lora=True,
    encoder_image_size=int(os.getenv("SAM3_ENCODER_IMAGE_SIZE", "128")),
    decoder_channels=int(os.getenv("SAM3_DECODER_CHANNELS", "256")),
    lora_r=int(os.getenv("SAM3_LORA_R", "8")),
    lora_alpha=int(os.getenv("SAM3_LORA_ALPHA", "16")),
    lora_dropout=float(os.getenv("SAM3_LORA_DROPOUT", "0.05")),
)
num_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"Warmed SAM3 weights from {model_name}; "
    f"parameters={num_params:,}; trainable={trainable_params:,}"
)
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
        --encoder_image_size "${SAM3_ENCODER_IMAGE_SIZE}" \
        --decoder_channels "${SAM3_DECODER_CHANNELS}" \
        --lora_r "${SAM3_LORA_R}" \
        --lora_alpha "${SAM3_LORA_ALPHA}" \
        --lora_dropout "${SAM3_LORA_DROPOUT}" \
        --num_iters 150_000 \
        --batches_per_eval 100 \
        --log_interval 10 \
        --num_workers 4
