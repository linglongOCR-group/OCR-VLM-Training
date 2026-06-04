#!/bin/bash

set -Eeuo pipefail

# ----------------------------------------------------------------------
# Logging: capture all stdout + stderr to logs/YYMMDD-HHmm.log
# ----------------------------------------------------------------------
mkdir -p logs
LOG_FILE="logs/$(date +%y%m%d-%H%M).log"
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'rc=$?; echo "[ERROR] aborted (exit $rc) near line $LINENO" >&2' ERR

# ----------------------------------------------------------------------
# Provenance: record the full environment and this script's source
# ----------------------------------------------------------------------
SELF="${BASH_SOURCE[0]:-$0}"

echo "================ run start: $(date '+%F %T %Z') ================"
echo "host=$HOSTNAME  log=$LOG_FILE  script=$SELF"

echo "---------------- environment (env | sort) ---------------------"
env | sort
echo "---------------- script source: $SELF -------------------------"
cat -- "$SELF"
echo "---------------- end provenance -------------------------------"

# ----------------------------------------------------------------------
# Derive NODE_RANK from the hostname's trailing digit (e.g. node3 -> 2)
# ----------------------------------------------------------------------
host_suffix="${HOSTNAME: -1}"
if ! [[ "$host_suffix" =~ ^[0-9]$ ]]; then
    echo "[ERROR] HOSTNAME '$HOSTNAME' does not end in a digit; cannot derive NODE_RANK." >&2
    exit 1
fi
export NODE_RANK=$(( host_suffix - 1 ))
export OCR_DATA_ROOT=/mnt/sas-server-0/DataMgmt

# ----------------------------------------------------------------------
# Resolve the training NIC on the 192.168.10.0/24 fabric
# ----------------------------------------------------------------------
get_train_iface() {
    ip -o -4 addr show | awk '$4 ~ /^192\.168\.10\./ {print $2}' | head -n 1
}
export TRAIN_IFACE="$(get_train_iface)"
if [[ -z "$TRAIN_IFACE" ]]; then
    echo "[ERROR] no interface on 192.168.10.0/24; cannot set TRAIN_IFACE." >&2
    exit 1
fi

echo "NODE_RANK=$NODE_RANK  TRAIN_IFACE=$TRAIN_IFACE"
echo "==============================================================="

env \
  MODEL_PATH=/mnt/sas-server-0/Models/saved-models/MinerU2.5-Trim-Pretrain/mineru25_trim_0521_bs1024/global_step_17544/merged \
  TRAIN_FILES=/mnt/sas-server-0/DataMgmt/views/unirec40m_mineru_train_nested_reference/train \
  VAL_FILES=/mnt/sas-server-0/DataMgmt/views/mineru25_pro_omnidocbench_v15_val_nested_reference_ar180/val \
  CKPT_DIR=/mnt/sas-server-0/Models/saved-models/MinerU2.5-Trim-Pretrain/mineru25_trim_ptv4_ftv1_0529 \
  PROJECT_NAME=mineru25_trim_ptv4_ftv1_0529 \
  MAX_CKPT_TO_KEEP=10 \
  MAX_LENGTH=16384 \
  GLOBAL_BATCH_SIZE=3072 \
  MICRO_BATCH_SIZE_PER_GPU=128 \
  LR=1e-5 \
  FREEZE_VISION_TOWER=True \
  DATA_NUM_WORKERS=4 \
  RESUME_MODE=auto \
  SAVE_FREQ=100 \
  TEST_FREQ=500 \
  NNODES=3 \
  MASTER_ADDR=192.168.10.22 \
bash scripts/train/run_multinode_sft_new.sh \
  engine.param_offload=false \
  engine.optimizer_offload=false

echo "================ run done: $(date '+%F %T %Z') ================"
