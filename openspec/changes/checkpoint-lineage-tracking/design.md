# Design: Checkpoint Lineage Tracking

## Technical Approach

Each checkpoint is a separate WandB artifact. Lineage is expressed through WandB's native consumption/production graph:
- `run.use_artifact(input)` marks an artifact as input
- `run.log_artifact(output)` marks an artifact as output

This produces a natural DAG: Checkpoint -> Run -> Checkpoint.

A Python library module (`CheckpointTracker`) is called explicitly by training code at defined lifecycle points: run start (resolve + register source), checkpoint save, and dataset registration.

## Architecture Decisions

### Decision: Artifact-per-Checkpoint

Rationale:
- A single versioned artifact per model line makes branching ambiguous (SFT from step 5 vs step 10)
- Run-centric tracking doesn't map individual checkpoints to their specific inputs when a run produces multiple checkpoints

Alternatives considered:
- One artifact per model line with versioned aliases: ambiguous branching
- Custom graph storage: unnecessary complexity given WandB's built-in lineage

### Decision: CKPT_ROOT-Relative Paths

Rationale:
- Multi-node setups mount the same checkpoint directory at different absolute paths
- Artifact metadata stores only relative paths; absolute paths are computed at runtime

Alternatives considered:
- Storing absolute paths: breaks portability across nodes
- Storing mount-specific symlinks: fragile and requires admin setup

### Decision: Library Module, Not Transparent Hooks

Rationale:
- Simpler to reason about than transparent interception
- Fits existing callback pattern in the codebase

### Decision: Seed Model Auto-Registration

Rationale:
- Eliminates need for a separate manual registration step
- Unknown filesystem paths are automatically registered with `is_seed=True`

## Data Flow / Control Flow

Typical training run:

```
1. wandb.init()
2. CheckpointTracker(project)
3. tracker.resolve_model_path(MODEL_PATH) -> (absolute_path, artifact_id)
4. tracker.register_source_model(run, path, artifact_id)
5. tracker.register_dataset(run, TRAIN_FILE, "train")
6. For each checkpoint save:
   tracker.register_checkpoint(run, ckpt_dir, step, mode, source_id)
7. run.finish()
```

Resumed run: same sequence. MODEL_PATH points to resume checkpoint. Lineage chain extends naturally.

## Artifact Data Model

### Checkpoint Artifact Metadata

| Field | Type | Description |
|-------|------|-------------|
| checkpoint_name | str | Logical name |
| checkpoint_dir_rel | str | Relative to CKPT_ROOT |
| checkpoint_root | str | CKPT_ROOT at save time (audit only) |
| global_step | int | Training step |
| training_mode | str | sft, grpo, dapo, gspo, ... |
| model_id | str | Base model identifier |
| config_hash | str | SHA-256 of config |
| git_commit | str | HEAD commit |
| source_artifact | str or None | Parent artifact ID |
| source_run_id | str or None | Parent run ID |
| is_seed | bool | True for untrained models |
| epoch | int or None | Epoch number |
| optimizer_state_included | bool | Default True |
| trainer_state_included | bool | Default True |
| resume_compatibility_version | str | Default "verl-bootstrap-v1" |
| validation_metrics | dict or None | Metrics at save time |
| save_timestamp | str or None | ISO 8601 UTC |

Key invariant: `checkpoint_dir_rel` is always stored; absolute paths are never stored in metadata.

### Dataset Artifact Metadata

| Field | Type | Description |
|-------|------|-------------|
| dataset_name | str | Name |
| dataset_path_rel | str | Relative to CKPT_ROOT |
| dataset_uri_rel | str | file:// relative path |
| format | str | "parquet" |
| split | str | train/val/test |
| row_count | int or None | Row count |
| dataset_version | str or None | Version string |

## Module Interface

```
verl_plugins/
  tracking/
    __init__.py
    checkpoint_tracker.py    # CheckpointTracker class
    artifacts.py             # Metadata dataclasses
    resolver.py              # resolve_model_path, find_artifact_by_path
    exceptions.py            # ArtifactNotFoundError
```

Core methods:
- `resolve_model_path(path_or_artifact) -> (absolute_path, artifact_id_or_none)`
- `register_source_model(run, model_path, source_artifact_id, metadata)`
- `register_seed_model(run, model_path, model_name, metadata) -> artifact_id`
- `register_checkpoint(run, checkpoint_dir, global_step, training_mode, source_artifact_id, aliases, extra_metadata) -> artifact_id`
- `register_dataset(run, dataset_path, split, dataset_name, metadata) -> artifact_id`
- `find_artifact_by_path(model_path) -> artifact_id_or_none`
- `trace_lineage(artifact_id) -> list of (artifact_id, training_mode, run_id)`

## Compatibility and Migration

- Additive changes only. Existing behavior with filesystem paths is unchanged.
- New `tracking` config group with `artifact_prefix` and `auto_register_seed`.
- New `CKPT_ROOT` environment variable (falls back to cwd with warning).
- Updated `CheckpointArtifactMetadata` with new fields that default to None/False.
- Training scripts add tracker initialization calls at defined points.

## Validation Strategy

Unit tests:
- Artifact ID resolution, raw path resolution, seed auto-registration, idempotent seed registration, checkpoint lineage, dataset registration, multiple checkpoints per run, relative path storage, WandB offline degradation, lineage tracing, path portability.

Integration tests:
- Full SFT run lineage, GRPO from SFT checkpoint, resume preserves lineage.
