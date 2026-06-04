#!/usr/bin/env bash
set -Eeuo pipefail

# Default: sync current directory to the same absolute path on these hosts.
SRC_DIR="$(pwd -P)"

# Change these to your real target machines.
TARGET_HOSTS=(
  "gpu-server-00"
  "gpu-server-01"
  "gpu-server-03"
)

SSH_USER="${SSH_USER:-root}"

# Set DELETE=1 if you want target dirs to exactly mirror source dirs.
DELETE="${DELETE:-0}"

# Set DRY_RUN=1 to preview changes.
DRY_RUN="${DRY_RUN:-0}"

RSYNC_EXCLUDES=(
  "--exclude=.git/"
  "--exclude=.venv/"
  "--exclude=venv/"
  "--exclude=__pycache__/"
  "--exclude=.pytest_cache/"
  "--exclude=.mypy_cache/"
  "--exclude=.ruff_cache/"
  "--exclude=wandb/"
  "--exclude=runs/"
  "--exclude=outputs/"
  "--exclude=checkpoints/"
  "--exclude=*.log"
  "--exclude=*.tmp"
)

RSYNC_OPTS=(
  -az
  --human-readable
  --info=progress2,stats2
  --partial
  --mkpath
)

if [[ "$DELETE" == "1" ]]; then
  RSYNC_OPTS+=(--delete)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

echo "Source: ${SRC_DIR}"
echo "Target path: ${SRC_DIR}"
echo "SSH user: ${SSH_USER}"
echo "DELETE=${DELETE}"
echo "DRY_RUN=${DRY_RUN}"
echo

sync_one_host() {
  local host="$1"

  echo "========== Syncing to ${host} =========="

  rsync "${RSYNC_OPTS[@]}" \
    "${RSYNC_EXCLUDES[@]}" \
    -e "ssh -T -o Compression=no" \
    "${SRC_DIR}/" \
    "${SSH_USER}@${host}:${SRC_DIR}/"

  echo "========== Done: ${host} =========="
  echo
}

for host in "${TARGET_HOSTS[@]}"; do
  sync_one_host "$host" &
done

wait

echo "All sync jobs completed."
