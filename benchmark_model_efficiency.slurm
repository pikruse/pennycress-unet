#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J model-bench
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p extended
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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME/hub/checkpoints" "$XDG_CACHE_HOME"

export MIOPEN_USER_DB_PATH="${PWD}/nnu_net/cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export MIOPEN_FIND_ENFORCE=${MIOPEN_FIND_ENFORCE:-NONE}

TEST_ROOT=${TEST_ROOT:-data/test}
IMAGE_DIR=${IMAGE_DIR:-${TEST_ROOT}/test_images}
MASK_DIR=${MASK_DIR:-${TEST_ROOT}/test_masks_preproc}
OUTPUT=${OUTPUT:-${TEST_ROOT}/model_efficiency_summary.csv}
PER_IMAGE_OUTPUT=${PER_IMAGE_OUTPUT:-${TEST_ROOT}/model_efficiency_per_image.csv}
DEVICE=${DEVICE:-cuda}
APPEND=${APPEND:-1}
SHOW_PROGRESS=${SHOW_PROGRESS:-0}
STRICT=${STRICT:-0}

BENCHMARK_ARGS=(
    --test-root "$TEST_ROOT"
    --image-dir "$IMAGE_DIR"
    --mask-dir "$MASK_DIR"
    --output "$OUTPUT"
    --per-image-output "$PER_IMAGE_OUTPUT"
    --device "$DEVICE"
)

if [[ -n "${MODEL_RESULTS:-}" ]]; then
    read -r -a MODEL_RESULT_ARGS <<< "$MODEL_RESULTS"
    BENCHMARK_ARGS+=(--models "${MODEL_RESULT_ARGS[@]}")
fi

if [[ -n "${MAX_IMAGES:-}" ]]; then
    BENCHMARK_ARGS+=(--max-images "$MAX_IMAGES")
fi

if [[ -n "${IMAGE_NAME:-}" ]]; then
    while IFS= read -r image_name; do
        [[ -n "$image_name" ]] && BENCHMARK_ARGS+=(--image-name "$image_name")
    done <<< "$IMAGE_NAME"
fi

if [[ "$APPEND" == "1" || "$APPEND" == "true" || "$APPEND" == "TRUE" ]]; then
    BENCHMARK_ARGS+=(--append)
fi

if [[ "$SHOW_PROGRESS" == "1" || "$SHOW_PROGRESS" == "true" || "$SHOW_PROGRESS" == "TRUE" ]]; then
    BENCHMARK_ARGS+=(--show-progress)
fi

if [[ "$STRICT" == "1" || "$STRICT" == "true" || "$STRICT" == "TRUE" ]]; then
    BENCHMARK_ARGS+=(--strict)
fi

if [[ -n "${EXTRA_BENCHMARK_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "$EXTRA_BENCHMARK_ARGS"
    BENCHMARK_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Writing summary to: $OUTPUT"
echo "Writing per-image timings to: $PER_IMAGE_OUTPUT"
echo "MODEL_RESULTS=${MODEL_RESULTS:-<all checkpoint-backed results>}"
echo "MAX_IMAGES=${MAX_IMAGES:-<all selected images>}"
echo "APPEND=$APPEND"
echo "SHOW_PROGRESS=$SHOW_PROGRESS"
echo "STRICT=$STRICT"

srun -N 1 \
    -n 1 \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/benchmark_model_efficiency.py "${BENCHMARK_ARGS[@]}"
