#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PROJECT_ROOT:-/workspace}:${PYTHONPATH:-}"
exec "$@"
