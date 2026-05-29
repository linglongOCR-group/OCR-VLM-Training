# Tasks

## 1. Library Module

- [ ] 1.1 Create `verl_plugins/tracking/` package with `__init__.py`
- [ ] 1.2 Implement `artifacts.py` with `CheckpointArtifactMetadata` and `DatasetArtifactMetadata` dataclasses
- [ ] 1.3 Implement `exceptions.py` with `ArtifactNotFoundError`
- [ ] 1.4 Implement `resolver.py` with `resolve_model_path` and `find_artifact_by_path`
- [ ] 1.5 Implement `checkpoint_tracker.py` with `CheckpointTracker` class and all core methods

## 2. Configuration

- [ ] 2.1 Create `configs/train/verl/base/tracking.yaml` with `artifact_prefix` and `auto_register_seed`
- [ ] 2.2 Update checkpoint config with `ckpt_root` field
- [ ] 2.3 Document `CKPT_ROOT` and `ARTIFACT_PREFIX` environment variables

## 3. Training Integration

- [ ] 3.1 Update SFT trainer to initialize tracker and call lifecycle methods
- [ ] 3.2 Update GRPO entrypoint to initialize tracker and call lifecycle methods
- [ ] 3.3 Update `verl_plugins/callbacks/save_and_eval.py` with new metadata fields
- [ ] 3.4 Remove absolute `checkpoint_dir` from dataclass; make it runtime-only

## 4. Error Handling

- [ ] 4.1 Implement graceful degradation when WandB is offline
- [ ] 4.2 Implement `CKPT_ROOT` fallback with warning
- [ ] 4.3 Handle multiple artifact matches by recency
- [ ] 4.4 Handle seed model idempotent registration

## 5. Testing

- [ ] 5.1 Write unit tests for all core methods (12 tests from design spec)
- [ ] 5.2 Write integration tests for SFT and GRPO lineage
- [ ] 5.3 Write resume lineage preservation test
- [ ] 5.4 Write path portability test (save with one CKPT_ROOT, resolve with another)

## 6. Validation

- [ ] 6.1 Run `openspec validate --all --strict`
- [ ] 6.2 Fix all validation errors
- [ ] 6.3 Verify no changes to existing checkpoint save/load/resume behavior
