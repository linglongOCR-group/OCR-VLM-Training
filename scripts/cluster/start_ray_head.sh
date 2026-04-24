#!/usr/bin/env bash
set -euo pipefail

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
NODE_IP_ADDRESS=${NODE_IP_ADDRESS:-}

ARGS=(--head --port="${RAY_PORT}" --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" --num-gpus="${NPUS_PER_NODE}")
if [ -n "${NODE_IP_ADDRESS}" ]; then
  ARGS+=(--node-ip-address="${NODE_IP_ADDRESS}")
fi

ray start "${ARGS[@]}"
