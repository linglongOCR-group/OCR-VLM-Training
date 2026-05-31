# Design: Training Ops Deployment

## Context

The repository already provides VERL-backed launch scripts for OCR-VLM SFT and GRPO on Ascend/NPU infrastructure: `scripts/train/run_multinode_sft_new.sh`, `scripts/train/run_grpo_fsdp.sh`, and Ray helper scripts under `scripts/cluster/`. Current operational gaps are outside the trainer: packaging the dirty working tree, deploying the same code revision to all Atlas 800T A2 nodes and containers, verifying path visibility, recording manifests, and launching repeatable multi-node runs.

The proposed behavior is additive. Training remains owned by the existing scripts and VERL. The new layer orchestrates code deployment and remote execution around those scripts.

## Goals / Non-Goals

**Goals:**

- Provide a repo-local agent skill and CLI for repeated Atlas 800T A2 deployment and training operations.
- Use YAML as the source of truth, with separate reusable inventory files and one run profile per file.
- Package the current working tree, including uncommitted local changes, while excluding large generated artifacts.
- Support shared-filesystem deployment and `/tmp` deployment through SSH tar streaming into existing containers.
- Verify host and container visibility before training.
- Record package, deployment, command, run, status, and launch metadata in a separate ops state directory.
- Launch GRPO through `scripts/train/run_grpo_fsdp.sh` and SFT through `scripts/train/run_multinode_sft_new.sh`.

**Non-Goals:**

- Starting SSHD inside containers.
- Creating, restarting, or upgrading Docker containers.
- Replacing Ray, VERL, torchrun, vLLM, or the existing training scripts.
- Enforcing hard locks for concurrent operations in the first version.
- Managing dataset creation or checkpoint lifecycle beyond path verification and environment recording.

## Decisions

### Decision: Repo-local Python CLI plus agent skill

The implementation will add `tools/training_ops/`, a thin `scripts/trainops` wrapper, and `skills/training-ops/SKILL.md`.

Rationale:

- Python is already used for structured repo tooling and tests.
- A CLI package can validate YAML, build manifests, normalize command execution, and be tested with fake executors.
- The skill captures the operator workflow for agents without embedding operational policy in shell snippets.

Alternatives considered:

- Extending `sync_codebase.sh`: faster to write, but weak for config validation, metadata, checksums, and testability.
- External operator package: more reusable across repos, but less direct for packaging the current dirty working tree.

### Decision: YAML inventory plus one run profile per file

Inventory YAML describes cluster topology, SSH user, containers, shared roots, and per-node rank/IP/interface. Run YAML references one inventory and contains one active deployment/training profile.

Rationale:

- Separate inventory avoids duplicating stable cluster topology.
- One profile per run file gives clean audit trails and simple config snapshots.
- YAML handles nested host lists and environment blocks more readably than TOML for this use case.

### Decision: Package first, deploy from the package

The package command creates a tar archive from the current working tree and writes a manifest before deployment.

Rationale:

- The exact source artifact can be checksummed once and verified everywhere.
- Dirty and untracked changes are intentionally included.
- Exclude rules are centrally auditable.

The default excludes will cover `.git/`, virtual environments, Python caches, test caches, checkpoints, datasets, `wandb/`, `runs/`, `outputs/`, logs, temporary files, and large archive artifacts. Exclude rules must be visible in the package manifest.

### Decision: Method A uses immutable releases and an atomic current symlink

Shared deployment extracts into `release_root/<release_id>/`, writes a manifest, verifies host/container visibility, then atomically updates `current_link`.

Rationale:

- Immutable releases make rollback possible by repointing the symlink.
- A unified absolute path keeps `PROJECT_ROOT` identical across nodes and containers.
- Verification catches mount mismatches before launch.

### Decision: Method B defaults to SSH tar streaming into `docker exec`

Temporary deployment streams the package through host SSH into the existing `verl-vlm-grpo` container and extracts under a configured container path such as `/tmp/ocr-vlm-training/releases/<release_id>`.

Rationale:

- It works whether or not host `/tmp` is bind-mounted into the container.
- It avoids starting SSHD in containers.
- It avoids `docker cp` except for explicit small/debug fallback use.

### Decision: Central executor records every command

All remote operations go through a command executor abstraction. Host commands use SSH. Container commands use host SSH plus `docker exec <container> bash -lc`.

Each command record includes run ID, release ID, node, host, rank, container, command label, sanitized command, timestamps, exit code, stdout/stderr paths, and selected environment.

Rationale:

- Command construction and logging are high-risk cross-cutting concerns.
- A fake executor allows dry-run and unit/integration tests without live Atlas nodes.

### Decision: Integrate with existing training scripts through environment rendering

Launch commands set explicit environment variables and call the existing scripts. GRPO launches on the head node and uses Ray. SFT launches on all selected nodes with derived `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `TRAIN_IFACE`.

Rationale:

- Existing scripts already encode Ascend, VERL, and training-specific behavior.
- The ops layer should make launches reproducible, not fork trainer logic.

## Proposed Module Structure

```text
tools/training_ops/
  __init__.py
  cli.py
  config.py        # YAML loading, schema validation, effective env rendering
  inventory.py     # node selection, rank/head resolution
  package.py       # archive creation, exclude handling, checksum manifest
  executor.py      # local, SSH, docker-exec, dry-run/fake executor interfaces
  deploy.py        # shared and tmp deployment workflows
  ray.py           # Ray start/status/stop command builders
  launch.py        # SFT/GRPO launch command builders
  state.py         # ops state paths, metadata, command records, status records
  cleanup.py       # dry-run cleanup planning and explicit cleanup execution
  errors.py
```

## Data Flow / Control Flow

Typical GRPO flow:

```text
1. trainops preflight --run configs/ops/runs/grpo-smoke.yaml
2. trainops package --run configs/ops/runs/grpo-smoke.yaml
3. trainops deploy shared --run configs/ops/runs/grpo-smoke.yaml
4. trainops ray start-head --run configs/ops/runs/grpo-smoke.yaml
5. trainops ray start-worker --run configs/ops/runs/grpo-smoke.yaml
6. trainops launch grpo --run configs/ops/runs/grpo-smoke.yaml
7. trainops status --run configs/ops/runs/grpo-smoke.yaml
8. trainops logs --run configs/ops/runs/grpo-smoke.yaml
```

Typical SFT flow skips Ray unless configured and runs `launch sft` across all selected nodes.

## Risks / Trade-offs

- Remote stop commands can terminate unrelated work -> require explicit stop targets and prefer scoped run/process markers where available.
- Dirty-tree packaging can include unwanted files -> use conservative default excludes, allow explicit extra excludes, and make exclude rules part of the manifest.
- Shared filesystem mounts can differ between hosts and containers -> verify visibility on every host and inside every configured container before launch.
- Warning-only concurrency can permit overlapping runs -> detect active Ray/training/current-link conflicts and record warnings in status metadata.
- Container paths under `/tmp` are ephemeral -> record this clearly in deployment manifests and require redeployment after container restart.

## Migration Plan

This is additive. Existing scripts continue to work directly. Operators can adopt the new workflow by adding inventory/run YAML files and invoking `scripts/trainops`.

Rollback is operationally simple: stop using `scripts/trainops` and call the existing scripts directly. For shared deployments, rollback to an older release is done by repointing the `current` symlink after verifying the target release. No data migration is required.

## Open Questions

- Exact sample inventory and run config names will be selected during implementation.
- Live Atlas smoke tests depend on access to real hosts and should be gated by explicit environment/config flags.
