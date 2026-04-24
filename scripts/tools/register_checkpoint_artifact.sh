#!/usr/bin/env bash
set -euo pipefail

python -m src.callbacks.save_and_eval register-checkpoint "$@"
