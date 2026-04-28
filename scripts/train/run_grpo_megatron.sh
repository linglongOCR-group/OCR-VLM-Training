#!/usr/bin/env bash
set -euo pipefail

echo "Megatron GRPO is provided as a template. Validate mbridge/Megatron dependencies before use." >&2
BACKEND_CONFIG=${BACKEND_CONFIG:-configs/train/verl/rl/grpo_megatron_vllm.yaml}
exec bash scripts/train/run_grpo_fsdp.sh --config-name=ppo_megatron_trainer model_engine=megatron "$@"
