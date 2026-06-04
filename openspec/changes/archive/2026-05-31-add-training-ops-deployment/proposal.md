## Why

Multi-node Atlas 800T A2 training currently depends on manual code sync, ad hoc remote commands, and hand-managed Ray/SFT/GRPO launch steps. This change adds an auditable training-ops workflow that can repeatedly deploy the current working tree, verify container visibility, and launch existing training scripts without reimplementing VERL training.

## Scope

In scope: repo-local agent skill, repo-local CLI, YAML inventory/run configs, current-working-tree packaging, shared-filesystem deployment, `/tmp` container deployment through SSH tar streaming, preflight checks, Ray operations, SFT/GRPO launch wrappers, status/log/stop/cleanup commands, and run metadata.

Out of scope for the first version: creating or restarting Docker containers, starting SSHD inside containers, enforcing hard cluster locks, replacing training scripts, uploading checkpoints, or managing datasets/checkpoints beyond path verification.

## What Changes

- Add a repo-local `training-ops` agent skill that guides preflight, packaging, deployment, verification, Ray startup, launch, log collection, status checks, stop, and cleanup.
- Add a Python CLI exposed by a thin `scripts/trainops` wrapper.
- Add reusable cluster inventory YAML files and one-profile-per-file run YAML configs.
- Package the current working tree, including uncommitted tracked and untracked changes, while excluding large generated artifacts.
- Support Method A shared-path deployment with immutable releases and atomic `current` symlink updates.
- Support Method B `/tmp` deployment through SSH tar streaming into `docker exec`, with host `/tmp` extraction as an explicit option.
- Record package, deployment, run, command, and status metadata under a separate ops state directory.
- Integrate launches with `scripts/train/run_grpo_fsdp.sh` and `scripts/train/run_multinode_sft_new.sh`.

## Capabilities

### New Capabilities

- `training-ops-deployment`: Defines packaging, deployment, remote execution, Ray orchestration, training launch, audit metadata, status/log/stop/cleanup behavior for Atlas 800T A2 OCR-VLM training operations.

### Modified Capabilities

- None.

## Impact

Affected areas include `skills/`, `tools/`, `scripts/`, `configs/ops/`, tests, and documentation. The change depends on existing system tools on operator and hosts: `ssh`, `tar`, `sha256sum`, `docker`, `bash`, and the already-running `verl-vlm-grpo` containers. It does not change dataset contracts, VERL trainer internals, or checkpoint semantics.

## Risks

- Remote command execution can affect active training if commands are scoped poorly; v1 must default to read-only checks and explicit stop/cleanup targets.
- Packaging a dirty working tree can accidentally include large or sensitive local artifacts; exclude rules and manifest review are required.
- Shared path visibility may differ between host and container mounts; deployment verification must check both.
- Warning-only concurrency can permit overlapping launches; manifests and status checks must make the risk visible.

## Open Questions

- Exact default inventory and run YAML file names can be chosen during implementation.
- Live Atlas smoke tests require real host/container access and should be gated by explicit config or environment variables.
