# Checkpoint Tracking and Lineage Management Design Spec

## 1. Document Status

**Status:** Draft v0.1
**Date:** 2026-05-07
**Audience:** Training / infra engineers
**Depends on:** SPEC.md (bootstrap phase), `verl_plugins/callbacks/save_and_eval.py`

---

## 2. Objective

Build a checkpoint lineage system on top of WandB artifacts that:

1. Records every checkpoint's metadata and filesystem location as an artifact
2. Tracks the derivation chain: which checkpoint was produced by which run, starting from which checkpoint
3. Supports resolving a WandB artifact ID as the model path when launching a training job
4. Handles multi-node environments where checkpoint directories are mounted at different paths
5. Auto-registers initial (seed) models that have no prior training history
6. Extends to future training modes (DAPO, GSPO, etc.) without restructuring

---

## 3. Scope

### 3.1 In Scope

- Checkpoint-to-checkpoint lineage tracking via WandB artifacts
- Dataset-to-run lineage (which dataset was consumed by which run)
- Artifact-based model path resolution
- Multi-node path portability via `CKPT_ROOT`-relative paths
- Seed model auto-registration
- A library module (`CheckpointTracker`) callable from training code
- Graceful degradation when WandB is unavailable

### 3.2 Out of Scope

- Uploading checkpoint files to WandB cloud storage (reference-only policy unchanged)
- Config or code snapshot versioning as artifacts
- Artifact lifecycle management (retention, garbage collection)
- Cross-project artifact references
- Automated checkpoint cleanup or pruning
- Query UI beyond WandB's built-in artifact graph

---

## 4. Design Decisions

### 4.1 Artifact-per-Checkpoint

Each checkpoint is a separate WandB artifact. Lineage is expressed through WandB's native consumption/production graph:

- `run.use_artifact(input)` marks an artifact as an **input** to the run
- `run.log_artifact(output)` marks an artifact as an **output** of the run

This produces a natural DAG: Checkpoint → Run → Checkpoint.

Rationale: a single versioned artifact per model line makes branching ambiguous (SFT from step 5 vs step 10 of the same model), and run-centric tracking doesn't map individual checkpoints to their specific inputs when a run produces multiple checkpoints.

### 4.2 `CKPT_ROOT`-Relative Paths

On multi-node setups, the same checkpoint directory is mounted at different absolute paths:

```
Node 0: /mnt/nfs/shared/checkpoints/ocr-vlm/qwen25vl-sft/step-500
Node 1: /data/shared/checkpoints/ocr-vlm/qwen25vl-sft/step-500
```

Artifact metadata stores paths **relative to `CKPT_ROOT`**. Resolution reconstructs the absolute path by joining `CKPT_ROOT` with the relative path at runtime.

### 4.3 Seed Model Auto-Registration

When a training run starts with a raw filesystem path that matches no existing artifact, the system automatically registers it as a seed model artifact with `is_seed=True`. This eliminates the need for a separate manual registration step.

### 4.4 Library Module, Not Transparent Hooks

The tracker is a Python class that training code calls explicitly at defined points. This is simpler to reason about than transparent interception, and fits the existing callback pattern in the codebase.

---

## 5. Artifact Data Model

### 5.1 Artifact Types

| Type | Purpose |
|------|---------|
| `model-checkpoint` | A training checkpoint or seed model |
| `training-dataset` | A dataset consumed by a training run |

### 5.2 Artifact Naming Convention

```
{artifact_prefix}/{model-identifier}:{alias}
```

- **artifact_prefix** — groups artifacts by project or purpose (e.g., `ocr-vlm`). Defaults to the WandB project name.
- **model-identifier** — derived from the run's experiment name or explicitly configured. Must be URL-safe (lowercase, hyphens, no spaces).
- **alias** — WandB-managed version tags: `latest`, `step-{N}`, `best`, `seed`.

Examples:

```
ocr-vlm/qwen25vl-base:seed
ocr-vlm/qwen25vl-sft-phase1:step-500
ocr-vlm/qwen25vl-sft-phase1:latest
ocr-vlm/qwen25vl-grpo-v2:step-200
```

### 5.3 Checkpoint Artifact Metadata

```python
@dataclass
class CheckpointArtifactMetadata:
    checkpoint_name: str
    checkpoint_dir_rel: str               # relative to CKPT_ROOT
    checkpoint_root: str                  # CKPT_ROOT at save time (audit trail)
    global_step: int
    training_mode: str                    # "sft" | "grpo" | "dapo" | "gspo" | ...
    model_id: str                         # base model identifier
    config_hash: str                      # SHA-256 of config snapshot
    git_commit: str                       # HEAD commit hash
    source_artifact: str | None           # parent artifact ID (e.g., "ocr-vlm/base:seed")
    source_run_id: str | None             # WandB run ID that produced the source
    is_seed: bool                         # True for untrained base models
    epoch: int | None = None
    optimizer_state_included: bool = True
    trainer_state_included: bool = True
    resume_compatibility_version: str = "verl-bootstrap-v1"
    validation_metrics: dict[str, Any] | None = None
    save_timestamp: str | None = None     # ISO 8601 UTC, set at save time
```

Key invariants:
- `checkpoint_dir_rel` is always stored. `checkpoint_dir` (absolute) is never stored in metadata — it is computed at runtime from `CKPT_ROOT` + `checkpoint_dir_rel`.
- `checkpoint_root` is stored for auditing only; it is not used for resolution.
- For seed models: `source_artifact` is `None`, `source_run_id` is `None`, `is_seed` is `True`, `global_step` is `0`.
- For trained checkpoints: `is_seed` is `False`, `source_artifact` and `source_run_id` are required.

### 5.4 Dataset Artifact Metadata

```python
@dataclass
class DatasetArtifactMetadata:
    dataset_name: str
    dataset_path_rel: str                 # relative to CKPT_ROOT (or DATA_ROOT if set)
    dataset_uri_rel: str                  # file://<relative-path>
    format: str                           # "parquet"
    split: str                            # "train" | "val" | "test"
    row_count: int | None = None
    dataset_version: str | None = None
```

`dataset_path_rel` is relative to the same root as checkpoints (`CKPT_ROOT`), since datasets are typically stored on the same shared filesystem. If datasets live on a different mount, `DATA_ROOT` can be used instead and the same relative-path pattern applies.

---

## 6. Module Interface

### 6.1 File Layout

```
verl_plugins/
  tracking/
    __init__.py
    checkpoint_tracker.py    # CheckpointTracker class
    artifacts.py             # CheckpointArtifactMetadata, DatasetArtifactMetadata
    resolver.py              # resolve_model_path, find_artifact_by_path
    exceptions.py            # ArtifactNotFoundError
  callbacks/
    save_and_eval.py         # existing, updated with new metadata fields
```

### 6.2 `CheckpointTracker` Class

```python
class CheckpointTracker:
    def __init__(
        self,
        wandb_project: str,
        wandb_entity: str | None = None,
        artifact_prefix: str | None = None,
    ):
        """
        artifact_prefix defaults to wandb_project if not provided.
        """
        ...
```

The tracker does **not** own the WandB run. It receives the run object from training code.

### 6.3 Core Methods

#### `resolve_model_path(model_path_or_artifact) -> tuple[str, str | None]`

Called at run start. Accepts either a filesystem path or an artifact ID.

- **Artifact ID** (`name:alias` or `name:version`): resolves via `wandb.Api().artifact()`, reads `checkpoint_dir_rel` from metadata, joins with `CKPT_ROOT`, returns the absolute path.
- **Filesystem path**: returns it unchanged. The caller passes the result to `register_source_model()` which handles artifact matching.

Returns: `(resolved_absolute_path, artifact_id_or_None)`

#### `register_source_model(run, model_path, source_artifact_id=None, metadata=None)`

Called at run start after `resolve_model_path()`. Records the input model as a consumed artifact.

- If `source_artifact_id` is provided: calls `run.use_artifact(id)`.
- If `model_path` is a raw path with no artifact ID: calls `find_artifact_by_path()` to search for a matching artifact. If found, calls `run.use_artifact()`. If not found, calls `register_seed_model()`.
- Logs source model info to the run's config.

#### `register_seed_model(run, model_path, model_name, metadata=None) -> str`

Creates a `model-checkpoint` artifact for an initial model.

- Sets `is_seed=True`, `global_step=0`.
- Uses alias `seed`.
- Stores `checkpoint_dir_rel` computed from `model_path` relative to `CKPT_ROOT`.
- Returns the artifact ID.

When called automatically by `register_source_model()`, `model_name` is derived from the path: the directory name of `model_path`, slugified (lowercased, non-alphanumeric replaced with hyphens). For example, `/data/models/Qwen2.5-VL-7B-Instruct` → `qwen25-vl-7b-instruct`. If the caller wants a custom name, it can call `register_seed_model()` directly before `register_source_model()`.

#### `register_checkpoint(run, checkpoint_dir, global_step, training_mode, source_artifact_id=None, aliases=None, extra_metadata=None) -> str`

Called each time a checkpoint is saved.

- Computes `checkpoint_dir_rel` from `checkpoint_dir` relative to `CKPT_ROOT`.
- Creates a `model-checkpoint` artifact with full metadata.
- Records `source_artifact_id` in metadata.
- Calls `run.log_artifact()` with aliases `["latest", f"step-{global_step}"]`.
- Returns the artifact name for chaining (use as `source_artifact_id` in subsequent checkpoints within the same run).

#### `register_dataset(run, dataset_path, split, dataset_name=None, metadata=None) -> str`

Creates a `training-dataset` reference artifact and calls `run.use_artifact()`.

#### `find_artifact_by_path(model_path) -> str | None`

Searches all `model-checkpoint` artifacts in the project for one whose `checkpoint_dir_rel` matches the given path (after stripping `CKPT_ROOT`). Returns the artifact ID or `None`.

Matching is by exact resolved relative path. If multiple matches exist, the most recent (by creation time) is used.

### 6.4 Convenience Query Methods

#### `trace_lineage(artifact_id) -> list[tuple[str, str | None, str | None]]`

Walks the derivation chain from the given artifact back to the root seed model. Returns a list of `(artifact_id, training_mode, run_id)` tuples from child to root.

#### `list_descendants(artifact_id) -> list[str]`

Returns all artifact IDs that are transitively derived from the given artifact.

These methods are built on WandB's API — no custom storage required.

---

## 7. Lineage DAG

### 7.1 Construction

WandB builds the lineage graph from two calls per run:

- `run.use_artifact(input_artifact)` — input edge
- `run.log_artifact(output_artifact)` — output edge

No custom graph code is needed. The DAG is queryable via WandB's API and visible in the WandB UI.

### 7.2 Example

```
seed: ocr-vlm/qwen25vl-base:seed
  |
  +-- [SFT run-abc]  <- consumes qwen25vl-base:seed
  |     |
  |     +-- ocr-vlm/qwen25vl-sft-phase1:step-500
  |     +-- ocr-vlm/qwen25vl-sft-phase1:step-1000  (also :latest)
  |
  +-- [GRPO run-def]  <- consumes qwen25vl-sft-phase1:latest
        |
        +-- ocr-vlm/qwen25vl-grpo-v2:step-200
        +-- ocr-vlm/qwen25vl-grpo-v2:step-400  (also :latest)
```

### 7.3 Multiple Checkpoints in One Run

All checkpoints produced by a single run share the same `source_artifact` (the input to that run). Each is a separate artifact with its own `step-{N}` alias.

### 7.4 Resumed Runs

When a run resumes from checkpoint C:

- `resolve_model_path()` resolves C (either by artifact ID or raw path)
- `register_source_model()` records C as the input to the resumed run
- New checkpoints from the resumed run have `source_artifact` pointing to C

This correctly represents "the resumed run started from C."

---

## 8. Path Resolution for Multi-Node Environments

### 8.1 `CKPT_ROOT` Environment Variable

`CKPT_ROOT` defines the mount point of the shared checkpoint filesystem on each node. It is set per-node and may differ across nodes.

```bash
# Node 0
export CKPT_ROOT=/mnt/nfs/shared/checkpoints

# Node 1
export CKPT_ROOT=/data/shared/checkpoints
```

### 8.2 Path Storage

Artifact metadata stores only the relative path:

| Scenario | Stored in artifact | Computed at runtime |
|----------|-------------------|---------------------|
| Saving a checkpoint | `checkpoint_dir_rel = "ocr-vlm/qwen25vl-sft/step-500"` | Absolute path from `CKPT_ROOT` + rel |
| Resolving an artifact | Read `checkpoint_dir_rel` | `Path(CKPT_ROOT) / checkpoint_dir_rel` |

The absolute path is never stored. `checkpoint_root` is stored for auditing only.

### 8.3 Resolution Examples

| Input to `resolve_model_path` | Resolution |
|-------------------------------|-----------|
| `/mnt/nfs/shared/checkpoints/ocr-vlm/qwen25vl-sft/step-500` | Strip `CKPT_ROOT` → search artifacts by relative path → return absolute path |
| `ocr-vlm/qwen25vl-base:seed` | Fetch artifact → read `checkpoint_dir_rel` → join with `CKPT_ROOT` → return absolute path |
| `ocr-vlm/qwen25vl-sft-phase1:latest` | Fetch artifact → read `checkpoint_dir_rel` → join with `CKPT_ROOT` → return absolute path |
| `ocr-vlm/qwen25vl-grpo-v2:v3` | Fetch specific version → read `checkpoint_dir_rel` → join with `CKPT_ROOT` → return absolute path |

### 8.4 Fallback

If `CKPT_ROOT` is not set, the tracker falls back to `Path.cwd()` and logs a warning. This preserves backward compatibility for single-node setups where the full path is stable.

---

## 9. Call Sequence

### 9.1 Typical Training Run

```python
# 1. WandB init (done by VERL or training code)
run = wandb.init(project="ocr-vlm-training", ...)

# 2. Create tracker
tracker = CheckpointTracker(wandb_project="ocr-vlm-training")

# 3. Resolve model path
resolved_path, artifact_id = tracker.resolve_model_path(MODEL_PATH)

# 4. Register source model (auto-registers seed if needed)
tracker.register_source_model(run, resolved_path, source_artifact_id=artifact_id)

# 5. Register datasets
tracker.register_dataset(run, TRAIN_FILE, "train")
tracker.register_dataset(run, VAL_FILE, "val")

# 6. Training loop
source_id = artifact_id  # or the seed artifact ID if auto-registered
for step in training_steps:
    ...
    # On checkpoint save:
    source_id = tracker.register_checkpoint(
        run, ckpt_dir, global_step=step,
        training_mode="sft",
        source_artifact_id=source_id,
    )

# 7. WandB finish (done by training code)
run.finish()
```

### 9.2 Resumed Run

Same sequence. The `MODEL_PATH` (or `RESUME_FROM_PATH`) points to the resume checkpoint. The tracker resolves it and records it as the input, establishing the lineage from the resume checkpoint to the new outputs.

---

## 10. Error Handling

| Scenario | Behavior |
|----------|----------|
| WandB offline or unreachable | Warning logged. All artifact operations skipped. Training proceeds normally. |
| Invalid artifact ID in `resolve_model_path()` | Raises `ArtifactNotFoundError` with the ID and context. |
| `find_artifact_by_path()` finds multiple matches | Warning logged. Uses the most recent match by creation time. |
| `register_checkpoint()` called with nonexistent `checkpoint_dir` | Raises `FileNotFoundError`. This is a real error. |
| Artifact registration fails mid-save | Warning logged. Checkpoint is valid on disk; only metadata is lost. |
| Seed model registration for a path that already has an artifact | Skips registration. Uses the existing artifact. |
| `CKPT_ROOT` not set | Falls back to `Path.cwd()`. Logs a warning. |

---

## 11. Configuration

### 11.1 New Config Group: `tracking`

```yaml
# configs/train/verl/base/tracking.yaml
artifact_prefix: ${oc.env:ARTIFACT_PREFIX,null}
auto_register_seed: true
```

- `artifact_prefix`: defaults to the WandB project name if null.
- `auto_register_seed`: when true, unknown model paths are auto-registered as seed artifacts.

### 11.2 Updated Config: `checkpoint`

```yaml
# configs/train/verl/base/checkpoint.yaml
local_root: ${oc.env:CKPTS_DIR,checkpoints/ocr-vlm-bootstrap}
ckpt_root: ${oc.env:CKPT_ROOT,}
resume_mode: ${oc.env:RESUME_MODE,auto}
resume_from_path: ${oc.env:RESUME_FROM_PATH,null}
save_contents: ["model", "optimizer", "extra"]
register_wandb_reference: true
resume_compatibility_version: verl-bootstrap-v1
```

`ckpt_root` is new. The existing fields are unchanged.

### 11.3 Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `CKPT_ROOT` | Shared checkpoint filesystem mount point | Recommended |
| `ARTIFACT_PREFIX` | Artifact name prefix | Optional (defaults to WandB project) |
| `MODEL_PATH` | Model path or artifact ID | Yes (existing) |
| `CKPTS_DIR` | Local checkpoint output directory | Yes (existing) |

---

## 12. Changes to Existing Code

### 12.1 `verl_plugins/callbacks/save_and_eval.py`

- `CheckpointArtifactMetadata` gains new fields: `checkpoint_dir_rel`, `checkpoint_root`, `source_artifact`, `source_run_id`, `is_seed`.
- The absolute `checkpoint_dir` field is removed from the dataclass; it becomes a runtime-only value.
- `register_checkpoint_reference()` gains an optional `source_artifact_id` parameter.
- Backward-compatible: new fields default to `None` or `False`.

### 12.2 Training Scripts

- SFT trainer and GRPO entrypoints add tracker initialization calls at defined points.
- `MODEL_PATH` accepts artifact IDs in addition to filesystem paths.
- Changes are additive — existing behavior with filesystem paths is unchanged.

### 12.3 What Does Not Change

- Checkpoint saving logic (`ckpt_handler.save_checkpoint()`)
- Local filesystem storage policy
- Resume logic (reads from local path, now resolved via tracker)
- Hydra config structure (new fields are additive)

---

## 13. Testing

### 13.1 Unit Tests

| Test | Verifies |
|------|----------|
| `test_resolve_artifact_id` | Artifact ID → relative path → absolute path via CKPT_ROOT |
| `test_resolve_raw_path` | Raw path → strips CKPT_ROOT → matches existing artifact |
| `test_resolve_raw_path_no_match` | Raw path with no matching artifact → returns path unchanged |
| `test_register_seed_model` | Auto-registers unknown path as seed with `is_seed=True` |
| `test_register_seed_model_idempotent` | Second call for same path → reuses existing artifact |
| `test_register_checkpoint_lineage` | Checkpoint has correct `source_artifact` and `source_run_id` |
| `test_register_dataset` | Dataset artifact created with correct metadata |
| `test_multiple_checkpoints_one_run` | All checkpoints in a run share the same `source_artifact` |
| `test_relative_path_storage` | Metadata contains only relative paths, never absolute |
| `test_wandb_offline` | All operations degrade gracefully, training unaffected |
| `test_trace_lineage` | `trace_lineage()` returns correct chain from child to root seed |
| `test_path_portability` | Save with CKPT_ROOT=/a, resolve with CKPT_ROOT=/b → correct path |

### 13.2 Integration Tests

| Test | Verifies |
|------|----------|
| `test_sft_e2e_lineage` | Full SFT run: seed → training → checkpoint. Verify DAG in WandB. |
| `test_grpo_from_sft_checkpoint` | GRPO consuming SFT output. Verify cross-mode lineage. |
| `test_resume_preserves_lineage` | Resumed run produces checkpoints with correct source chain. |

Integration tests use `wandb.init(mode="disabled")` or a dedicated test project.

---

## 14. Future Extensions

These are explicitly out of scope but the design accommodates them:

- **DAPO / GSPO training modes**: add the mode string to `training_mode`. No structural changes.
- **Config snapshot artifacts**: a new artifact type (`training-config`) registered alongside checkpoints.
- **Cross-project artifact references**: extend `resolve_model_path()` to accept `entity/project/name:alias`.
- **Artifact lifecycle policies**: a separate module that queries artifact age and prunes old checkpoints.
