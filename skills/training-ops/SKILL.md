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
- Re-check NPU occupancy immediately before assigning single-node jobs. `npu-smi` may show HBM held by non-training processes such as VLLM even when no `sft_trainer` or `torchrun` process is present; do not treat such a node as free for an 8-NPU SFT run unless the user explicitly clears or authorizes it.
- For multiple independent single-node SFT ablations, create one run YAML per node/run with distinct `run.id`, `release_id`, `EXPERIMENT_NAME`, `master_port`, checkpoint directory, and log path. Set exactly one selected node so `NNODES=1` and `NODE_RANK=0` are derived correctly.
- Before launching inside a container, verify the container can see every shared path, not just the host: checkpoint, `OCR_DATA_ROOT`, finalized `TRAIN_FILES`, `VAL_FILES`, and output directory. If the default container lacks the required mounts, use an explicitly mounted dedicated container or inventory override rather than assuming host-visible `/mnt/sas-server-*` paths are visible in Docker.
- For background SFT launch, inspect the generated launch stderr/log immediately. If a launcher constructs `nohup VAR=value ...`, it will fail because `nohup` treats the first `VAR=value` token as the executable. Launch via `bash -lc 'export VAR=...; cd <release>; nohup bash scripts/train/run_multinode_sft_new.sh ... > training-sft-rank0.log 2>&1 &'` or fix the launcher to wrap env assignments in a shell. In manual fallback launches, export every required env var before `cd`/`nohup`; writing `VAR=value cd $PROJECT_ROOT && ...` leaves `MASTER_ADDR`, `TRAIN_FILES`, etc. unset for the script.
- Preserve W&B logging when the reference/baseline training script uses it. Do not silently downgrade to `trainer.logger=["console"]`; if auth is missing, either set `WANDB_MODE=offline` while keeping `trainer.logger=["console", "wandb"]`, or ask/record that W&B auth must be fixed before online launch. When the user confirms W&B auth is configured, explicitly `unset WANDB_MODE`, set/export `TRAINER_LOGGER='["console", "wandb"]'`, and verify the log shows W&B login/sync plus a run URL.
- If `deploy tmp` fails with empty stdout/stderr, first verify whether the package is visible inside the target container and whether the release directory already exists. Prefer extracting to a new unique release directory for recovery; do not remove or replace an existing release directory without explicit user approval.
- If a release directory is replaced while a training process from that release is still running, the process may keep writing to deleted-but-open log/script file descriptors. Do not conclude the job stopped just because the new `training-sft-rank0.log` is missing or stale; inspect `/proc/<launcher_pid>/fd/1` or `/proc/<rank0_pid>/fd/1`, record the PID, and avoid relaunching duplicates while the matching `torchrun`/trainer processes are alive.

## Standard Flow

1. Inspect the run config and inventory:
   `scripts/trainops --run configs/ops/runs/<run>.yaml inventory`
2. Run preflight:
   `scripts/trainops --run configs/ops/runs/<run>.yaml preflight`
3. For data-backed SFT, verify the training view path is finalized and validated before launch. A view directory ending in `.tmp` (for example `train.tmp`) means the build is still in progress; packaging/deployment may proceed, but launch must wait for the final `train/` path and successful `docds validate-view`.
4. Package the dirty working tree:
   `scripts/trainops --run configs/ops/runs/<run>.yaml package`
5. Deploy:
   - Shared path: `scripts/trainops --run <run.yaml> deploy shared`
   - Container tmp: `scripts/trainops --run <run.yaml> deploy tmp`
6. For GRPO, start Ray:
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
- For SFT startup verification, collect both process evidence (`torchrun` plus one worker per NPU) and log evidence that the script echoed `MODEL_PATH`, `TRAIN_FILES`, `VAL_FILES`, `GLOBAL_BATCH_SIZE`, detected `torch_npu`, and reported the expected NPU count. Continue monitoring until checkpoint/data loading and the first training iteration are visible. If W&B is expected, verification is not complete until `trainer.logger=["console", "wandb"]` is present in the effective process args/log and the log shows W&B syncing with a run URL.

## References

- `references/single-node-sft-ablation.md`: notes for single-node SFT ablations, container mount checks, manual background launch fallback, and deleted-log-fd recovery after replacing a live release.
