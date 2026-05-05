#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
TRAIN_FILE=${TRAIN_FILE:?TRAIN_FILE is required}
VAL_FILE=${VAL_FILE:-null}
CKPTS_DIR=${CKPTS_DIR:-"${PROJECT_ROOT}/checkpoints/sft-fsdp"}
WANDB_PROJECT=${WANDB_PROJECT:-ocr-vlm-training}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft-fsdp}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
OCR_DATA_ROOT=${OCR_DATA_ROOT:-}
FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-False}
USE_OCR_DATASET="${USE_OCR_DATASET:-True}"

if [[ "${USE_OCR_DATASET}" == "True" ]]; then
  OCR_DATASET_ARGS=(+data.image_key=runtime_images
    +data.data_root="${OCR_DATA_ROOT}"
    data.custom_cls.path="${PROJECT_ROOT}/tools/data_management/runtime/verl_multimodal_dataset.py"
    data.custom_cls.name=OcrMultiTurnSFTDataset)
else
  OCR_DATASET_ARGS=(data.image_key=images)
fi

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPUS_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  -m verl_plugins.trainers.sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.messages_key=messages \
  "${OCR_DATASET_ARGS[@]}" \
  +model.freeze_vision_tower="${FREEZE_VISION_TOWER}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-1}" \
  data.max_length="${MAX_LENGTH:-4096}" \
  data.pad_mode=no_padding \
  data.truncation=error \
  model.path="${MODEL_PATH}" \
  model.use_remove_padding=True \
  engine=fsdp \
  optim=fsdp \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${NPUS_PER_NODE}" \
  trainer.device=npu \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode="${RESUME_MODE:-auto}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.validate_only="${VALIDATE_ONLY:-False}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}" \
  checkpoint.save_contents='["model","optimizer","extra"]' \
  "$@"
