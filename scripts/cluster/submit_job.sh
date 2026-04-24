#!/usr/bin/env bash
set -euo pipefail

RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}

ray job submit --address="${RAY_DASHBOARD_ADDRESS}" --working-dir="${PROJECT_ROOT}" -- "$@"
