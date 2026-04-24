#!/usr/bin/env bash
set -euo pipefail

echo "MindSpeedLLM GRPO is provided as a template. Validate model-specific llm_kwargs before use." >&2
exec bash scripts/train/run_grpo_fsdp.sh model_engine=mindspeed "$@"
