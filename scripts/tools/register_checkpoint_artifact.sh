#!/usr/bin/env bash
set -euo pipefail

python -m verl_plugins.callbacks.save_and_eval register-checkpoint "$@"
