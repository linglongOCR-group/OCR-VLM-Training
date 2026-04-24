#!/usr/bin/env bash
set -euo pipefail

if [ -z "${RAY_HEAD_ADDRESS:-}" ]; then
  echo "RAY_HEAD_ADDRESS must be set, for example head-node:6379" >&2
  exit 2
fi

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
ray start --address="${RAY_HEAD_ADDRESS}" --num-gpus="${NPUS_PER_NODE}"
