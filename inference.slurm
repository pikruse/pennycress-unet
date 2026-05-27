#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J deeplabv3+-infer
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q debug
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

DISTANCE_WEIGHTS=${DISTANCE_WEIGHTS:-up}
RUN_LABEL=${RUN_LABEL:-deeplabv3_plus_${DISTANCE_WEIGHTS}}

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    CHECKPOINT_DIR="checkpoints/deeplabv3_plus_${DISTANCE_WEIGHTS}"
fi

PREDICTION_DIR=${PREDICTION_DIR:-data/test/${RUN_LABEL}_test_predictions}
POD_PREDICTION_DIR=${POD_PREDICTION_DIR:-data/test/${RUN_LABEL}_test_predictions_by_pod}
OUTPUT_DIR=${OUTPUT_DIR:-data/test/${RUN_LABEL}_test_metrics}
MEASURE_PODS=${MEASURE_PODS:-0}

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
else
    CHECKPOINT_ARGS=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

MEASURE_ARGS=()
if [[ "$MEASURE_PODS" == "1" || "$MEASURE_PODS" == "true" || "$MEASURE_PODS" == "TRUE" ]]; then
    MEASURE_ARGS=(--measure-pods)
fi

srun -N 1 \
    -n 1 \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/inference.py \
        "${CHECKPOINT_ARGS[@]}" \
        --image-dir data/test/test_images \
        --mask-dir data/test/test_masks_preproc \
        --prediction-dir "$PREDICTION_DIR" \
        --pod-prediction-dir "$POD_PREDICTION_DIR" \
        --pod-mask-dir data/test/test_masks_by_pod \
        --output-dir "$OUTPUT_DIR" \
        --device cuda \
        "${MEASURE_ARGS[@]}"
