#!/usr/bin/env bash
set -euo pipefail

RESUME_FROM_PATH=${RESUME_FROM_PATH:?RESUME_FROM_PATH is required}
export RESUME_MODE=resume_path
export RESUME_FROM_PATH

case "${TRAINING_MODE:-grpo}" in
  grpo)
    exec bash scripts/train/run_grpo_fsdp.sh "$@"
    ;;
  sft)
    exec bash scripts/train/run_sft.sh "$@"
    ;;
  *)
    echo "TRAINING_MODE must be grpo or sft" >&2
    exit 2
    ;;
esac
