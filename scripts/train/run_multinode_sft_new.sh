#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

set -euo pipefail

# ============================================================
# Multi-node VERL SFT for MinerU2.5 on Atlas 800T A2 / Ascend 910B
#
# Usage:
#   node-0:
#     MASTER_ADDR=10.10.10.10 NODE_RANK=0 TRAIN_IFACE=bond0 \
#       MODEL_PATH=/path/to/model TRAIN_FILES='["/path/train/part-00000.parquet"]' \
#       bash scripts/train/run_multinode_sft.sh
#
#   node-1:
#     MASTER_ADDR=10.10.10.10 NODE_RANK=1 TRAIN_IFACE=bond0 \
#       MODEL_PATH=/path/to/model TRAIN_FILES='["/path/train/part-00000.parquet"]' \
#       bash scripts/train/run_multinode_sft.sh
#
# Notes:
#   - MASTER_ADDR must be the host-plane IP of node-0, not an NPU RoCE IP.
#   - TRAIN_IFACE should be the host-plane NIC used for torchrun/HCCL bootstrap.
#   - MODEL_PATH, TRAIN_FILES/VAL_FILES, checkpoint paths, and source-reference
#     OCR_DATA_ROOT must resolve identically in every container.
#   - TRAIN_FILES and VAL_FILES are passed to Hydra as raw values. For sharded
#     views, pass an explicit Hydra list such as '["/data/view/train/part-00000.parquet"]'.
# ============================================================


# ---------------------------
# 1. Repository context
# ---------------------------

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"


# ---------------------------
# 2. Cluster topology
# ---------------------------

export NNODES="${NNODES:-2}"
export NPUS_PER_NODE="${NPUS_PER_NODE:-8}"

export MASTER_ADDR="${MASTER_ADDR:?Please set MASTER_ADDR to node-0 host-plane IP}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export NODE_RANK="${NODE_RANK:?Please set NODE_RANK=0 on node-0 and NODE_RANK=1 on node-1}"
export TRAIN_IFACE="${TRAIN_IFACE:?Please set TRAIN_IFACE, e.g. bond0 / eth0 / enp189s0f0}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"


# ---------------------------
# 3. Ascend runtime environment
# ---------------------------

export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${TRAIN_IFACE}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${TRAIN_IFACE}}"

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"

export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true

# Helps reduce NPU memory fragmentation.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# Conservative CPU thread setting for dataloader/tokenization.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ulimit -n 32768 || true
ulimit -l unlimited || true


# ---------------------------
# 4. Paths and data
# ---------------------------

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
TRAIN_FILES="${TRAIN_FILES:-${TRAIN_FILE:-}}"
if [[ -z "${TRAIN_FILES}" ]]; then
  echo "[ERROR] TRAIN_FILES is required. TRAIN_FILE is accepted as a compatibility alias." >&2
  exit 1
fi
VAL_FILES="${VAL_FILES:-${VAL_FILE:-null}}"
OCR_DATA_ROOT="${OCR_DATA_ROOT:-}"

CKPTS_DIR="${CKPTS_DIR:-${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/mineru25-sft-multinode}}"

mkdir -p "${CKPTS_DIR}"


# ---------------------------
# 5. Stable default hyperparameters
# ---------------------------

# Global batch across all data-parallel ranks.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-64}}"

# Per-NPU micro batch. Keep 1 for stable VLM SFT.
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"

# SFT max sequence length after tokenization.
MAX_LENGTH="${MAX_LENGTH:-4096}"

# Token-based dynamic batch budget per NPU.
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"

# Sequence parallel.
# Recommended:
#   SP_SIZE=1 for normal crop-level SFT
#   SP_SIZE=2 for long page/table/diagram samples
SP_SIZE="${SP_SIZE:-1}"

# FSDP shard group.
# -1 means use all ranks as one FSDP group.
# If cross-node communication is the bottleneck, try FSDP_SIZE="${NPUS_PER_NODE}".
FSDP_SIZE="${FSDP_SIZE:--1}"

# Training length.
# Use null for epoch-based training.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"

# Optimizer.
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"

# Data loader and debugging controls.
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-8}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
IGNORE_INPUT_IDS_MISMATCH="${IGNORE_INPUT_IDS_MISMATCH:-False}"
FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-False}
USE_OCR_DATASET="${USE_OCR_DATASET:-True}"

# Checkpoint / validation.
SAVE_FREQ="${SAVE_FREQ:-500}"
TEST_FREQ="${TEST_FREQ:-500}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-5}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"

# Logging.
WANDB_PROJECT="${WANDB_PROJECT:-${PROJECT_NAME:-mineru25_trim_pt}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mineru25_a2x${NNODES}_fsdp2_bf16_gbs${GLOBAL_BATCH_SIZE}_sp${SP_SIZE}}"
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console", "wandb"]'}

# For stability on Ascend. Enable later only after baseline is stable.
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"

if [[ "${USE_OCR_DATASET}" == "True" ]]; then
  OCR_DATASET_ARGS=(+data.image_key=runtime_images
    +data.data_root="${OCR_DATA_ROOT}"
    data.custom_cls.path="${PROJECT_ROOT}/tools/data_management/runtime/verl_multimodal_dataset.py"
    data.custom_cls.name=OcrMultiTurnSFTDataset)
else
  OCR_DATASET_ARGS=(data.image_key=images)
fi


# ---------------------------
# 6. Basic checks
# ---------------------------

echo "========== Cluster =========="
echo "NNODES=${NNODES}"
echo "NPUS_PER_NODE=${NPUS_PER_NODE}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "TRAIN_IFACE=${TRAIN_IFACE}"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"

echo "========== Paths =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "VAL_FILES=${VAL_FILES}"
echo "OCR_DATA_ROOT=${OCR_DATA_ROOT}"
echo "CKPTS_DIR=${CKPTS_DIR}"

echo "========== Hyperparameters =========="
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU}"
echo "SP_SIZE=${SP_SIZE}"
echo "FSDP_SIZE=${FSDP_SIZE}"
echo "LR=${LR}"
echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
echo "DATA_NUM_WORKERS=${DATA_NUM_WORKERS}"
echo "TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}"
echo "VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}"
echo "FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER}"
echo "USE_OCR_DATASET=${USE_OCR_DATASET}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

python - <<'PY'
import importlib
import torch

print("torch:", torch.__version__)

try:
    import torch_npu
    print("torch_npu:", torch_npu.__version__)
    print("npu count:", torch.npu.device_count())
except Exception as e:
    print("[ERROR] torch_npu not available:", repr(e))
    raise

importlib.import_module("verl_plugins.trainers.sft_trainer")
print("OCR SFT trainer import: OK")
PY


# ---------------------------
# 7. Launch SFT
# ---------------------------

set -x

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPUS_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m verl_plugins.trainers.sft_trainer \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${GLOBAL_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=right \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  data.balance_dp_token=True \
  data.num_workers="${DATA_NUM_WORKERS}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.messages_key=messages \
  "${OCR_DATASET_ARGS[@]}" \
  +model.freeze_vision_tower="${FREEZE_VISION_TOWER}" \
  data.tools_key=tools \
  data.enable_thinking_default=none \
  data.ignore_input_ids_mismatch="${IGNORE_INPUT_IDS_MISMATCH}" \
  model.path="${MODEL_PATH}" \
  model.trust_remote_code=True \
  model.enable_gradient_checkpointing=True \
  model.enable_activation_offload=False \
  model.use_remove_padding=True \
  model.use_liger=False \
  model.use_fused_kernels=False \
  engine.strategy=fsdp2 \
  engine.fsdp_size="${FSDP_SIZE}" \
  engine.reshard_after_forward=True \
  engine.param_offload=False \
  engine.optimizer_offload=False \
  engine.offload_policy=False \
  engine.dtype=bfloat16 \
  engine.model_dtype=bfloat16 \
  engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
  engine.use_torch_compile="${USE_TORCH_COMPILE}" \
  optim.optimizer=AdamW \
  optim.optimizer_impl=torch.optim \
  optim.lr="${LR}" \
  'optim.betas=[0.9,0.95]' \
  optim.weight_decay="${WEIGHT_DECAY}" \
  optim.lr_warmup_steps_ratio="${WARMUP_RATIO}" \
  optim.lr_scheduler_type=cosine \
  optim.clip_grad="${CLIP_GRAD}" \
  checkpoint.save_contents='["model","optimizer","extra"]' \
  checkpoint.load_contents='["model","optimizer","extra"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  trainer.device=npu \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${NPUS_PER_NODE}" \
  "$@"
