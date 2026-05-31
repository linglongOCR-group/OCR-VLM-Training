# Tasks

## 1. CLI and Package Structure

- [x] 1.1 Create `tools/training_ops/` package with module skeletons from the design
- [x] 1.2 Add `scripts/trainops` wrapper that invokes the Python CLI
- [x] 1.3 Implement top-level argparse command routing for inventory, preflight, package, deploy, exec, ray, launch, status, logs, stop, and cleanup
- [x] 1.4 Add repo-local `skills/training-ops/SKILL.md` with the safe operational workflow

## 2. Configuration and Inventory

- [x] 2.1 Implement YAML loading for separate inventory and one-profile-per-file run configs
- [x] 2.2 Implement node selection, head-node resolution, rank ordering, and per-node container defaults
- [x] 2.3 Validate required fields for shared deployment, tmp deployment, SFT launch, and GRPO launch
- [x] 2.4 Render effective launch environment for GRPO and SFT, including derived topology variables
- [x] 2.5 Add example inventory and run config files under `configs/ops/`

## 3. Ops State and Metadata

- [x] 3.1 Implement run ops state directory creation under configurable root or `runs/training-ops/<run_id>/`
- [x] 3.2 Implement immutable snapshots of inventory config, run config, and effective environment
- [x] 3.3 Implement JSON manifest writers for package, deployment, run metadata, command records, and status
- [x] 3.4 Implement sanitized command rendering that preserves audit value without leaking unnecessary shell noise

## 4. Packaging

- [x] 4.1 Implement current-working-tree archive creation including uncommitted tracked and untracked files
- [x] 4.2 Implement default exclude rules for `.git`, environments, caches, checkpoints, datasets, logs, W&B runs, outputs, ops runs, and large archives
- [x] 4.3 Implement configurable extra include/exclude rules
- [x] 4.4 Compute artifact SHA256 checksum and write package manifest
- [x] 4.5 Detect existing package artifacts and reuse them when checksum and manifest match

## 5. Command Execution

- [x] 5.1 Implement local, SSH, container `docker exec`, dry-run, and fake executor interfaces
- [x] 5.2 Record stdout/stderr logs and command metadata for every executor call
- [x] 5.3 Implement `exec` subcommand for explicit host or container command execution
- [x] 5.4 Ensure container commands always route through host SSH and `docker exec` rather than container SSHD

## 6. Preflight

- [x] 6.1 Check SSH access on every selected host
- [x] 6.2 Check configured container is running on every selected host
- [x] 6.3 Check NPU visibility and required Python/runtime imports inside each container
- [x] 6.4 Check configured model, train, validation, checkpoint, OCR data, shared, and temporary paths as applicable
- [x] 6.5 Detect active Ray or training processes and report warning-only concurrency findings

## 7. Deployment Method A: Shared Filesystem

- [x] 7.1 Extract package into immutable `release_root/<release_id>/`
- [x] 7.2 Write release-local and ops-state deployment manifests
- [x] 7.3 Verify release checksum and release path visibility on every selected host
- [x] 7.4 Verify the unified absolute release path is visible inside every selected container
- [x] 7.5 Atomically update the configured `current` symlink after verification
- [x] 7.6 Reuse matching existing releases and reject conflicting release IDs

## 8. Deployment Method B: Temporary Paths

- [x] 8.1 Implement SSH tar streaming directly into `docker exec` as the default tmp deployment path
- [x] 8.2 Verify available space, extraction success, checksum marker, and container path visibility on every selected node
- [x] 8.3 Support explicit host `/tmp` extraction mode with host and container visibility verification
- [x] 8.4 Add explicit small/debug fallback mode for `docker cp` without making it the default
- [x] 8.5 Reuse matching existing tmp releases and reject conflicting release IDs

## 9. Ray Operations

- [x] 9.1 Implement `ray start-head` command builder using `scripts/cluster/start_ray_head.sh`
- [x] 9.2 Implement `ray start-worker` command builder using `scripts/cluster/start_ray_worker.sh`
- [x] 9.3 Derive and record `RAY_HEAD_ADDRESS` and `RAY_ADDRESS` values from inventory and run config
- [x] 9.4 Implement Ray status collection across selected containers
- [x] 9.5 Implement explicit Ray stop operation with dry-run support

## 10. Training Launch

- [x] 10.1 Implement `launch grpo` command builder for `scripts/train/run_grpo_fsdp.sh` on the head container
- [x] 10.2 Implement `launch sft` command builder for `scripts/train/run_multinode_sft_new.sh` on every selected container
- [x] 10.3 Support foreground debugging and background launch modes with logs redirected to ops state
- [x] 10.4 Record launch metadata before process execution
- [x] 10.5 Ensure GRPO and SFT launch commands set `PROJECT_ROOT`, `PYTHONPATH`, model/data/checkpoint paths, topology vars, and extra script arguments explicitly

## 11. Status, Logs, Stop, and Cleanup

- [x] 11.1 Implement status command for release, Ray, training process, warnings, and recent log state
- [x] 11.2 Implement logs command for run, node, and command-label filtering
- [x] 11.3 Implement stop command requiring explicit target such as Ray, GRPO, SFT, or run ID
- [x] 11.4 Implement cleanup planning with dry-run default
- [x] 11.5 Prevent cleanup from deleting the active shared release by default

## 12. Tests and Validation

- [x] 12.1 Add unit tests for YAML loading, validation, inventory resolution, and effective env rendering
- [x] 12.2 Add unit tests for package excludes, checksums, and manifest contents
- [x] 12.3 Add unit tests for shared and tmp deployment command construction with fake executors
- [x] 12.4 Add unit tests for Ray, SFT, GRPO, status, logs, stop, and cleanup command builders
- [x] 12.5 Add dry-run integration tests that do not require live Atlas nodes
- [x] 12.6 Add optional live smoke test hooks gated by explicit environment/config flags
- [x] 12.7 Run `pytest` for the affected tests
- [x] 12.8 Run `openspec validate --all --strict`
