#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J model-infer
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -eo pipefail

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-cv/

export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_ENTITY=${WANDB_ENTITY:-jail-ai}
export WANDB_HTTP_TIMEOUT=90

export MODEL_CACHE_ROOT="${PWD}/nnu_net/cache/model_weights"
export HF_HOME="${MODEL_CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${MODEL_CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${MODEL_CACHE_ROOT}/xdg"
export HF_HUB_OFFLINE=1

export MIOPEN_USER_DB_PATH="${PWD}/nnu_net/cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=FAST
export MIOPEN_FIND_ENFORCE=NONE

MODEL=${MODEL:-auto}
DISTANCE_WEIGHTS=${DISTANCE_WEIGHTS:-up}
PREDICTION_FORMAT=${PREDICTION_FORMAT:-auto}
PRED_PREFIX=${PRED_PREFIX:-pred_}
MEASURE_PODS=${MEASURE_PODS:-0}
SKIP_INFERENCE=${SKIP_INFERENCE:-0}
SKIP_EVALUATION=${SKIP_EVALUATION:-0}

if [[ -z "${MASK_DIR:-}" ]]; then
    if [[ "$MODEL" == "nnunet" || "$MODEL" == "nnUNet" || "$MODEL" == "nnunetv2" ]]; then
        MASK_DIR="nnu_net/nnUNet_raw/Dataset501_PennycressPods/labelsTs"
    else
        MASK_DIR="data/test/test_masks_preproc"
    fi
fi

MODEL_ARGS=(
    --model "$MODEL"
    --distance-weights "$DISTANCE_WEIGHTS"
    --prediction-format "$PREDICTION_FORMAT"
    --pred-prefix "$PRED_PREFIX"
)

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
elif [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    CHECKPOINT_ARGS=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

PATH_ARGS=(
    --image-dir "${IMAGE_DIR:-data/test/test_images}"
    --mask-dir "$MASK_DIR"
    --pod-mask-dir "${POD_MASK_DIR:-data/test/test_masks_by_pod}"
)
if [[ -n "${IMAGE_NAME:-}" ]]; then
    PATH_ARGS+=(--image-name "$IMAGE_NAME")
fi
if [[ -n "${PREDICTION_DIR:-}" ]]; then
    PATH_ARGS+=(--prediction-dir "$PREDICTION_DIR")
fi
if [[ -n "${POD_PREDICTION_DIR:-}" ]]; then
    PATH_ARGS+=(--pod-prediction-dir "$POD_PREDICTION_DIR")
fi
if [[ -n "${RGB_OUTPUT_DIR:-}" ]]; then
    PATH_ARGS+=(--rgb-output-dir "$RGB_OUTPUT_DIR")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    PATH_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

MEASURE_ARGS=()
if [[ "$MEASURE_PODS" == "1" || "$MEASURE_PODS" == "true" || "$MEASURE_PODS" == "TRUE" ]]; then
    MEASURE_ARGS=(--measure-pods)
fi

SKIP_ARGS=()
if [[ "$SKIP_INFERENCE" == "1" || "$SKIP_INFERENCE" == "true" || "$SKIP_INFERENCE" == "TRUE" ]]; then
    SKIP_ARGS+=(--skip-inference)
fi
if [[ "$SKIP_EVALUATION" == "1" || "$SKIP_EVALUATION" == "true" || "$SKIP_EVALUATION" == "TRUE" ]]; then
    SKIP_ARGS+=(--skip-evaluation)
fi

srun -N 1 \
    -n 1 \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/inference.py \
        "${MODEL_ARGS[@]}" \
        "${CHECKPOINT_ARGS[@]}" \
        "${PATH_ARGS[@]}" \
        --device cuda \
        "${SKIP_ARGS[@]}" \
        "${MEASURE_ARGS[@]}"
