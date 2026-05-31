---
name: training-ops
description: Safely deploy the current OCR-VLM-Training working tree and launch SFT or GRPO on multi-node Atlas 800T A2 clusters through host SSH and docker exec.
---

# OCR-VLM Training Ops

Use this skill when deploying or launching OCR-VLM training on Atlas 800T A2 nodes.

## Safety Rules

- Use `scripts/trainops --run <run.yaml> preflight` before deployment or launch.
- Treat `run.nodes` order as the active run topology: SFT `NODE_RANK` is derived from this order, while GRPO/Ray uses the selected head address and does not need `NODE_RANK`.
- Package the current working tree with `scripts/trainops --run <run.yaml> package`; review the package manifest before deploying when excludes changed.
- Do not start SSHD inside containers. All container commands go through host SSH plus `docker exec`.
- Assume `verl-vlm-grpo` containers already exist and are running unless the user explicitly expands scope.
- Do not delete old releases or stop processes without an explicit target and a dry-run/status check.
- Treat concurrency findings as warnings in v1; surface them to the user before launch.

## Standard Flow

1. Inspect the run config and inventory:
   `scripts/trainops --run configs/ops/runs/<run>.yaml inventory`
2. Run preflight:
   `scripts/trainops --run configs/ops/runs/<run>.yaml preflight`
3. Package the dirty working tree:
   `scripts/trainops --run configs/ops/runs/<run>.yaml package`
4. Deploy:
   - Shared path: `scripts/trainops --run <run.yaml> deploy shared`
   - Container tmp: `scripts/trainops --run <run.yaml> deploy tmp`
5. For GRPO, start Ray:
   - `scripts/trainops --run <run.yaml> ray start-head`
   - `scripts/trainops --run <run.yaml> ray start-worker`
6. Launch:
   - GRPO: `scripts/trainops --run <run.yaml> launch grpo`
   - SFT: `scripts/trainops --run <run.yaml> launch sft`
7. Monitor:
   - `scripts/trainops --run <run.yaml> status`
   - `scripts/trainops --run <run.yaml> logs`
8. Stop only with an explicit target:
   - `scripts/trainops --run <run.yaml> stop --target ray`
   - `scripts/trainops --run <run.yaml> stop --target grpo`
   - `scripts/trainops --run <run.yaml> stop --target sft`
9. Cleanup starts as dry-run:
   `scripts/trainops --run <run.yaml> cleanup --release-root <path> --current-link <path>`

## Evidence To Collect

- Package manifest with branch, commit SHA, dirty status, release ID, timestamp, source path, target nodes, and checksum.
- Deployment manifest with per-node host/container verification.
- Launch metadata with effective env, extra args, target nodes, and command log paths.
- Status metadata and logs under the run ops state directory.
