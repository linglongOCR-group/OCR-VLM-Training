# Proposal: Checkpoint Lineage Tracking via WandB Artifacts

## Intent

Build a checkpoint lineage system that records every checkpoint's metadata and filesystem location as a WandB artifact, tracks derivation chains (which checkpoint was produced by which run, starting from which checkpoint), supports resolving WandB artifact IDs as model paths, handles multi-node path portability, and auto-registers seed models.

## Scope

In scope:
- Checkpoint-to-checkpoint lineage tracking via WandB artifacts
- Dataset-to-run lineage (which dataset was consumed by which run)
- Artifact-based model path resolution
- Multi-node path portability via `CKPT_ROOT`-relative paths
- Seed model auto-registration
- Library module (`CheckpointTracker`) callable from training code
- Graceful degradation when WandB is unavailable

Out of scope:
- Uploading checkpoint files to WandB cloud storage
- Config or code snapshot versioning as artifacts
- Artifact lifecycle management (retention, garbage collection)
- Cross-project artifact references
- Automated checkpoint cleanup or pruning

## Impact

- Affected capabilities: [[training-bootstrap]], [[checkpoint-tracking]]
- Affected code areas: `verl_plugins/callbacks/save_and_eval.py`, training entry scripts, SFT/GRPO launchers
- Affected users or operators: Training and infra engineers
- Compatibility impact: Additive. Existing filesystem-path-based behavior is unchanged.
- Migration impact: New `tracking` config group, new `CKPT_ROOT` environment variable, updated `CheckpointArtifactMetadata` dataclass

## Risks

- WandB offline/unreachable during training: graceful degradation required
- Path portability across nodes with different mount points: resolved by `CKPT_ROOT`-relative storage
- Multiple matches when searching artifacts by path: use most recent by creation time
- Backward compatibility: new metadata fields default to `None`/`False`

## Open Questions

- Whether SFT and GRPO should use separate or shared WandB projects
- Artifact naming and alias conventions beyond `latest`, `step-{N}`, `best`, `seed`
- Whether checkpoint registration happens synchronously or asynchronously after save
- Whether resume validation should run automatically after restore
