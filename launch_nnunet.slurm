#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J pennycress-nnunet
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH -C nvme
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err
#SBATCH --open-mode=truncate

set -eo pipefail

### settings, environment, and module loads ###
module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-cv/

### network / proxy settings ###
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=${MASTER_PORT:-6000}
export NCCL_SOCKET_IFNAME=hsn0
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1

### miopen settings ###
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH" && mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=NORMAL

### nnU-Net settings ###
export nnUNet_raw="${PWD}/nnu_net/nnUNet_raw"
export nnUNet_preprocessed="${PWD}/nnu_net/nnUNet_preprocessed"
export nnUNet_results="${PWD}/nnu_net/nnUNet_results"
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-12}

DATASET_ID=${DATASET_ID:-501}
DATASET_NAME=${DATASET_NAME:-PennycressPods}
CONFIGURATION=${CONFIGURATION:-2d}
FOLD=${FOLD:-0}
NUM_GPUS=${NUM_GPUS:-1}
NNUNET_EXTRA_ARGS=${NNUNET_EXTRA_ARGS:---npz}

read -r -a EXTRA_ARGS <<< "$NNUNET_EXTRA_ARGS"

mkdir -p logs "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

if ! command -v nnUNetv2_plan_and_preprocess >/dev/null 2>&1; then
    echo "nnUNetv2_plan_and_preprocess was not found. Install nnU-Net v2 in this environment first:"
    echo "  pip install nnunetv2"
    exit 1
fi

python -u scripts/prepare_nnunet_dataset.py \
    --dataset-id "$DATASET_ID" \
    --dataset-name "$DATASET_NAME" \
    --overwrite

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

srun -N 1 \
    -n 1 \
    --gpus-per-task="$NUM_GPUS" \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD" "${EXTRA_ARGS[@]}" -num_gpus "$NUM_GPUS"
