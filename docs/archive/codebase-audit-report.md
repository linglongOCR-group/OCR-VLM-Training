# Codebase Audit Report

**Date:** 2026-05-27
**Scope:** Full codebase (~9,500 lines, 74 Python files)
**Branch:** `data-module`

---

## Executive Summary

The audit identified **152 issues** across 6 subsystems. The codebase has grown through rapid iteration and carries significant technical debt concentrated in three areas:

1. **Massive cross-adapter duplication** in `sources/adapters/` — the three adapters share ~400 lines of near-identical boilerplate (`_ShardWriter`, `ExportReport`, export pipeline scaffolding).
2. **Silent failures and data corruption risks** — missing dimensions silently fallback to `1x1`, corrupt images are swallowed, and schema inference performs unnecessary full I/O.
3. **Dead code accumulation** — at least 3 entire files, 15+ functions, and 6 identical stub files serve no purpose.

The highest-impact fix is extracting shared adapter infrastructure (ShardWriter, ExportReport, export pipeline) — this alone eliminates ~400 lines of duplication and prevents the three adapters from silently diverging.

---

## Issue Summary by Severity

| Severity | Count | Examples |
|----------|-------|---------|
| **Critical** (bug / data corruption) | 6 | `train_time` always 0, division by zero, silent bbox corruption |
| **High** (dead modules, export gaps) | 18 | 3 dead files, 6 dead stub files, duplicate `__all__` drops export |
| **Medium** (duplication, inconsistency) | 78 | Triply-duplicated ShardWriter, duplicated sampling logic, inconsistent defaults |
| **Low** (style, minor fragility) | 50 | Unused imports, minor naming, test patterns |

---

## Part 1: Critical Issues (Bugs & Data Corruption)

### C1. `train_time` is never updated — always prints `0.00s`
- **File:** `verl_plugins/trainers/sft_trainer.py:86,148`
- `train_time = 0` is initialized but never incremented. The final log prints a misleading `0.00s`.
- **Fix:** Implement actual timing with `time.monotonic()`, or remove the variable and log statement.

### C2. Potential division by zero in `_apply_image_filters`
- **File:** `tools/data_management/views/builder.py:686`
- `width / height` is computed without checking for zero dimensions. The validator (`validator.py:337`) guards against this, but the builder does not.
- **Fix:** Add `width == 0 or height == 0` guard.

### C3. Silent bbox corruption when page dimensions are missing
- **File:** `tools/data_management/serializers/layout_mineru.py:26-27,58-61`
- When `width`/`height` are missing, fallback to `1.0` produces garbage bounding boxes (e.g., `x / 1.0` clamped to 1000). `_to_grid` silently returns `0` for `size <= 0`.
- **Fix:** Remove the `or 1` fallback. Raise `ValueError` if dimensions cannot be determined.

### C4. `hybrid_message.py` uses `(1, 1)` sentinel bbox without documentation
- **File:** `tools/data_management/sources/adapters/hybrid_message.py:72,360,458`
- Non-layout tasks get `bbox=[0.0, 0.0, 1.0, 1.0]` with no metadata flag. Downstream consumers may misinterpret this.
- **Fix:** Add `metadata={"bbox_sentinel": True}` or use `None`.

### C5. Duplicate `__all__` in `runtime/__init__.py` silently drops `resolve_runtime_images`
- **File:** `tools/data_management/runtime/__init__.py:4,6`
- Two `__all__` assignments — the second overrides the first, dropping `resolve_runtime_images`.
- **Fix:** Remove the second `__all__` on line 6.

### C6. `lineage/resolver.py` double-nests `canonical_record` key
- **File:** `tools/data_management/lineage/resolver.py:22,31`
- `trace_view_record` wraps `trace_canonical_record`'s result, producing `{"canonical_record": {"canonical_record": row}}`.
- **Fix:** Return `row` directly from `trace_canonical_record`.

---

## Part 2: Dead Code (Modules, Functions, Files)

### Dead Modules (delete entirely)
| File | Lines | Reason |
|------|-------|--------|
| `tools/data_management/mineru_export.py` | 25 | Wrapper around `cli.py`, never imported |
| `tools/data_management/export_views.py` | 20 | Wrapper around `cli.py`, never imported |
| `tools/data_management/config/loader.py` | 12 | `load_config` never called; all consumers use `resolver.py` |

### Dead Functions
| Function | File:Line | Notes |
|----------|-----------|-------|
| `_uses_root_view_assets` | `views/builder.py:1001` | Always returns `False`; makes line 282 unreachable |
| `_asset_dirs_for_row` | `views/builder.py:1177` | Never called |
| `_iter_rows` | `views/validator.py:108` | Never called; different from the one in `validate_grpo_view.py` |
| `_chunked` | `sources/adapters/mineru.py:657` | Never called; inline chunking used instead |
| `_save_region_crop` | `sources/adapters/mineru.py:813` | Wrapper around `_RegionCropSaver`, never called |
| `_default_assets_dir` | `validate_grpo_view.py:167` | Never referenced |
| `ensure_dir` | `utils/io.py:10` | Never imported |
| `apply_npu_defaults` | `runtime/env.py` | Never called |
| `_export_sample` | `sources/adapters/mineru.py:365` | Trivial delegation to `_export_sample_records` |
| `render_prompt` | `prompts.py:79` | Only called internally by `resolve_prompt` |

### 6 Identical Reward Stub Files
All six contain exactly the same two lines (`from aggregate import reward; __all__ = ["reward"]`):
- `verl_plugins/rewards/table_reward.py`
- `verl_plugins/rewards/text_reward.py`
- `verl_plugins/rewards/formula_reward.py`
- `verl_plugins/rewards/diagram_reward.py`
- `verl_plugins/rewards/seal_reward.py`
- `verl_plugins/rewards/syntax_reward.py`

**Fix:** Delete all six. If per-type customization is planned, add files when the differentiated logic exists.

### Unused Abstract Method
- `scan_documents` in `base.py:16` — implemented by all adapters but never called. `PubTableSourceAdapter` returns `[]`.
- **Fix:** Remove from the base class and all implementations.

---

## Part 3: Architecture — Cross-Adapter Duplication

This is the single largest source of technical debt. The three source adapters (`mineru.py` 987 lines, `hybrid_message.py` 647 lines, `pubtable.py` 524 lines) share ~400 lines of near-identical code:

### A1. `_ShardWriter` — defined 3 times
- `mineru.py:926-987` (62 lines)
- `pubtable.py:476-524` (49 lines)
- `hybrid_message.py:618-648` (31 lines, stripped-down variant)

All three share the same buffer-and-flush-to-parquet pattern. The mineru and pubtable versions are nearly identical. The hybrid_message version lacks `id_column` dedup and `overwrite` cleanup.

**Fix:** Extract to `adapters/_shard_writer.py`. Parameterize the differences (id_column support, overwrite behavior).

### A2. `ExportReport` dataclass — defined 3 times
- `MinerUExportReport` (mineru.py:93) — adds `skipped_completed_samples`
- `PubTableExportReport` (pubtable.py:57) — adds unused `to_dict()`
- `HybridMessageExportReport` (hybrid_message.py:106)

All share the same 9 base fields. Only MinerU has one extra field.

**Fix:** Create a single `ExportReport` dataclass with `skipped_completed_samples: int = 0`.

### A3. `_drain_completed_exports` — byte-for-byte identical in 2 files
- `mineru.py:668-674`
- `pubtable.py:468-473`

**Fix:** Move to shared module.

### A4. Export pipeline boilerplate (~120 lines per adapter)
All three `export()` methods share:
- 7 writer initializations
- Iteration + write loop
- Writer close loop
- Report aggregation
- Manifest writing
- Progress finish

**Fix:** Extract a `_run_export_pipeline()` helper that accepts an iterable of export results plus configuration.

### A5. Hardcoded canonical path construction
Path strings like `"entities/documents"`, `"assets/files"`, `"records/{task}"`, and the `f"source={name}"` partition format appear dozens of times across all three adapters.

**Fix:** Define path helpers (e.g., `partition_path(root, entity, source_name)`, `region_crop_path(source_name, asset_id)`) in a shared module.

---

## Part 4: Duplication Within Views/Builder

### D1. Sampling logic duplicated across 3 methods
- `_compute_sample_key_sets` (builder.py:537-579)
- `_apply_samples` (builder.py:705-754)
- `_assign_splits` (builder.py:764)

All three parse the same `sample` config (count, level, sources, tasks, where, seed) and apply the same SHA-256 deterministic sort. The key-selection logic is nearly identical in all three.

**Fix:** Extract a `_parse_sample_config` helper and a `_select_keys` helper. Let the three methods delegate to the shared logic.

### D2. Split key derivation duplicated 3 times
`"record_id" if level == "record" else "page_id" if level == "page" else "document_id"` appears at lines 554, 719, and 764.

**Fix:** Extract `def _key_field_for_level(level: str) -> str`.

### D3. Filter predicate duplicated within `_apply_samples`
The same filter logic (checking sources, tasks, where) is written out twice (lines 726-733 and 743-750).

**Fix:** Extract into a `_record_matches_filter` helper.

---

## Part 5: Cross-Codebase Duplication

### X1. `levenshtein_distance` implemented in two places
- `tools/data_management/rewards/levenshtein.py:52-71`
- `verl_plugins/rewards/common.py`

Same algorithm, different variable naming.

**Fix:** Have `verl_plugins/rewards/common.py` import from the canonical location.

### X2. `_as_bool` defined identically in two files
- `verl_plugins/trainers/sft_trainer.py:226-235`
- `verl_plugins/trainers/sft_freeze.py:90-99`

**Fix:** Extract to a shared `_utils.py` or import across the two files.

### X3. Default prompt strings duplicated
- `_legacy_prompt_map` (prompts.py:50-76)
- `resolve_prompt` (prompts.py:93-101)

The same hardcoded strings appear in both.

**Fix:** Define a single `TASK_DEFAULTS` dict constant.

### X4. `_as_list` / `to_plain` duplication
- `schemas.to_plain` normalizes values for serialization
- `validate_grpo_view._as_list` does similar normalization with PyArrow-specific handling
- `test_data_pipeline._as_list` has yet another variant

**Fix:** Consolidate PyArrow-aware normalization into a single utility.

### X5. Test PNG image helpers duplicated across 3 test files
- `test_data_pipeline.py:27`
- `test_verl_runtime_dataset.py:137`
- `test_mineru_image_resolution.py:8`

**Fix:** Extract to `tests/conftest.py`.

---

## Part 6: Performance Issues

### P1. O(N*M) directory scan per sample in MinerU adapter
- **File:** `sources/adapters/mineru.py:913-923`
- `_prefixed_image_candidates` calls `iterdir()` + filter per sample. For N samples and M files, this is O(N*M).
- **Fix:** Cache a prefix-to-file mapping at export start.

### P2. Full JSON file loaded into memory in hybrid_message adapter
- **File:** `sources/adapters/hybrid_message.py:334-336`
- `json.load()` reads the entire data file. Called twice (export + scan_documents) with no caching.
- **Fix:** Load once and cache, or use streaming JSON parsing for large files.

### P3. Lineage resolver linear scan
- **File:** `lineage/resolver.py:17-30`
- `trace_view_record` and `trace_canonical_record` load every parquet file to find a single record. No index, no caching.
- **Fix:** Build an in-memory record-id-to-file index on first use.

### P4. Schema inference still performs full image I/O
- **File:** `views/builder.py:828-871`
- When `schema_inference=True`, the code still calls `_transform_and_encode` and `_read_image_file_bytes` before discarding the pixel data.
- **Fix:** Check `schema_inference` before I/O operations and produce placeholders immediately.

### P5. VerlRewardWrapper rebuilds registry per instance
- **File:** `runtime/verl_export.py:10`
- `default_reward_registry()` is called in `__init__` every time.
- **Fix:** Cache as a module-level singleton or class variable.

### P6. `_as_bool` Levenshtein is pure Python O(n*m)
- **File:** `tools/data_management/rewards/levenshtein.py:52-71`
- No C extension, no early termination. Called per prediction in RLHF scoring.
- **Fix:** Use `rapidfuzz` or `python-Levenshtein` in production.

### P7. New adapter instance created per sample in MinerU multiprocessing
- **File:** `sources/adapters/mineru.py:693`
- `_export_sample_for_worker` instantiates `MinerUSourceAdapter(options)` for every sample.
- **Fix:** Cache the adapter instance per worker process.

### P8. Double parquet read in canonical reader
- **File:** `canonical/reader.py:43-50`
- When `selected_asset_ids` is set, the file is read twice (once for IDs, once for full data).
- **Fix:** Read once, then filter.

---

## Part 7: Fragility & Silent Failures

### F1. Silent exception swallowing in image I/O
- `views/builder.py:1167-1174` — `_read_image_file_bytes` catches ALL exceptions, returns `None`
- `views/builder.py:1233-1236` — `_transform_and_encode` catches ALL exceptions, returns `None`
- Corrupt images, permission errors, and OOM are silently treated the same.

**Fix:** Narrow catches to `OSError`. Log warnings.

### F2. `json.dumps(..., default=str)` silently converts non-serializable types
- **File:** `utils/io.py:25`
- A stray `Path` or `None` gets silently stringified rather than raising an error.

**Fix:** Remove `default=str`. Handle expected types explicitly.

### F3. Crash-unsafe file lock in MinerU adapter
- **File:** `sources/adapters/mineru.py:751-768`
- `O_CREAT | O_EXCL` lock file persists if the process crashes (SIGKILL, OOM). No stale-lock detection.
- **Fix:** Write PID to lock file. Check if PID is alive on collision. Or use `fcntl.flock`/`filelock`.

### F4. `--overwrite` default is `True` with `store_true` action
- **File:** `cli.py:47`
- `add_argument("--overwrite", action="store_true", default=True)` — user can never turn overwrite off.
- **Fix:** Set `default=False` or add `--no-overwrite`.

### F5. Hardcoded `/verl/` absolute path in Hydra config
- **File:** `verl_plugins/trainers/sft_trainer.py:248`
- `config_path="/verl/verl/trainer/config"` breaks outside Docker.
- **Fix:** Resolve dynamically: `Path(verl.__file__).parent / "trainer" / "config"`.

### F6. `_resolve_source_image_path` returns non-existent path
- **File:** `sources/adapters/hybrid_message.py:491-499`
- After all resolution attempts, falls through to returning the original `image_path` even if it doesn't exist.
- **Fix:** Verify the chosen path exists. Raise clear error or log warning.

### F7. Mutable module-level globals for multiprocessing workers
- **File:** `views/builder.py:29-33`
- Five separate mutable globals (`_WORKER_BUILDER`, `_WORKER_CONFIG`, `_WORKER_CONTEXT`, `_WORKER_VIEW_NAME`, `_WORKER_STAGE`) passed to child workers. Adding a parameter requires updating three places.
- **Fix:** Bundle into a single dataclass/NamedTuple.

### F8. Fragile positional tuple unpacking for task parameters
- **File:** `views/validator.py:125-138`
- 11-element tuple unpacked positionally. Silent misassignment if order changes.
- **Fix:** Use `NamedTuple` or `@dataclass(frozen=True)`.

---

## Part 8: Inconsistency Issues

### I1. Version strings: `"1.0.0"` vs `"1.0"`
- mineru: `"1.0.0"`, pubtable: `"1.0"`, hybrid_message: `"1.0.0"`

### I2. `DEFAULT_IMAGE_EXTENSIONS` differ between adapters
- mineru: `(".jpg", ".jpeg", ".png", ".webp")`
- hybrid_message: `(".png", ".jpg", ".jpeg")` — missing `.webp`, different order

### I3. Serializer validation patterns differ
- `PlainTextSerializer` and `LatexPlainSerializer` use `if "X" not in target: raise ValueError`
- `EnhancedOTSLSerializer` uses `.get()` fallback chains
- Error messages lack serializer name

### I4. `RewardResult` lacks `validate()` method (all other schema dataclasses have one)
- **File:** `schemas.py:266-274`

### I5. `ViewRecord.to_dict` selectively removes `None` for `images` but not other nullable fields
- **File:** `schemas.py:260-262`

### I6. Pubtable `_ShardWriter` lazy-imports `pandas`; mineru imports at module level
- **File:** `pubtable.py:500` vs `mineru.py` top-level

### I7. `configured_source_registry` registers classes; others register instances
- **File:** `registry/configured.py:11-15` vs `:18-35`

### I8. `task_type` redundantly set twice in `compute_score`
- **File:** `verl_plugins/rewards/aggregate.py:43-44` (already set inside `normalized_levenshtein_reward`)

### I9. Inconsistent dataset mutation between `OcrRLHFDataset` and `OcrMultiTurnSFTDataset`
- **File:** `runtime/verl_multimodal_dataset.py:27` — RLHF mutates `example[self.image_key]`; SFT does not

---

## Part 9: Test Quality Issues

### T1. `object.__new__()` pattern used in 4+ test files
Bypasses `__init__`, making tests fragile to attribute changes. Files: `test_verl_runtime_dataset.py`, `test_sft_validation_flags.py`.
- **Fix:** Create factory helpers or use `@pytest.fixture`.

### T2. `try/except/else` instead of `pytest.raises` in 2 test files
- `test_data_pipeline.py:1283-1302`
- `test_checkpoint_artifacts.py:65-81`

### T3. Repeated view config dicts (15+ tests)
Nearly identical config dicts constructed in each test with minor variations.
- **Fix:** Extract `_base_view_config(**overrides)` helper.

### T4. Missing negative test coverage
- `test_rewards.py` — no tests for invalid inputs
- `test_otsl.py` — no rowspan/cross-merge roundtrip tests
- `test_progress.py` — no `finish()` test

### T5. Hardcoded shell script content checks
- `test_train_scripts.py:4-52` — exact substring matches against shell scripts, extremely brittle.

### T6. Hardcoded magic number `1036`
- `test_data_pipeline.py:599-622` — references MinerU's layout pre-processing without a named constant.

---

## Proposed Refactoring Plan

### Phase 1: Delete Dead Code (Low Risk, Immediate)
1. Delete `mineru_export.py`, `export_views.py`, `config/loader.py`
2. Delete 6 reward stub files in `verl_plugins/rewards/`
3. Delete dead functions: `_uses_root_view_assets`, `_asset_dirs_for_row`, `_iter_rows`, `_chunked`, `_save_region_crop`, `_default_assets_dir`, `ensure_dir`, `apply_npu_defaults`, `_export_sample`, `render_prompt` (make private)
4. Remove `scan_documents` from `SourceAdapter` base class and all implementations
5. Clean up `config/__init__.py` to remove loader re-exports

### Phase 2: Fix Critical Bugs (Low Risk, Immediate)
1. Fix duplicate `__all__` in `runtime/__init__.py` (C5)
2. Fix `train_time` measurement or remove it (C1)
3. Add division-by-zero guard in `_apply_image_filters` (C2)
4. Remove silent `1x1` dimension fallback in layout serializer (C3)
5. Fix double-nested `canonical_record` in lineage resolver (C6)
6. Fix `--overwrite` CLI default (F4)

### Phase 3: Extract Shared Adapter Infrastructure (Medium Risk, High Impact)
1. Create `adapters/_shard_writer.py` — unified `_ShardWriter`
2. Create `adapters/_export_types.py` — unified `ExportReport`
3. Create `adapters/_pipeline.py` — shared `_run_export_pipeline()`, `_drain_completed_exports()`
4. Create `adapters/_paths.py` — canonical path construction helpers
5. Refactor each adapter to use shared infrastructure
6. Align inconsistent defaults (version strings, image extensions)

### Phase 4: Reduce Duplication in Views/Builder (Medium Risk)
1. Extract `_key_field_for_level` helper
2. Unify sampling logic into shared helpers
3. Bundle worker globals into a dataclass
4. Convert task tuples to NamedTuple in validator
5. Move schema-inference check before image I/O

### Phase 5: Harden Error Handling (Medium Risk)
1. Narrow exception catches in image I/O to `OSError`
2. Add logging for caught exceptions
3. Remove `default=str` from `json.dumps` in `io.py`
4. Add stale-lock detection in MinerU file lock
5. Validate loaded JSON structure in hybrid_message adapter
6. Add `_resolve_source_image_path` existence check

### Phase 6: Fix Inconsistencies (Low Risk)
1. Standardize serializer validation pattern
2. Add `validate()` to `RewardResult`
3. Standardize `ViewRecord.to_dict` None handling
4. Move `pandas` import to module level in pubtable adapter
5. Remove redundant `task_type` assignment in `aggregate.py`
6. Resolve `/verl/` hardcoded path dynamically

### Phase 7: Test Quality Improvements (Low Risk)
1. Extract shared PNG helper to `conftest.py`
2. Create `_base_view_config` helper
3. Create factory helpers for `object.__new__()` test pattern
4. Replace `try/except/else` with `pytest.raises`
5. Add negative test cases for rewards and OTSL
6. Parameterize repeated test variants

---

## Estimated Impact

| Phase | Lines Removed | Lines Added (net) | Risk | Impact |
|-------|--------------|-------------------|------|--------|
| Phase 1: Dead code | ~300 | 0 | Minimal | Cleaner codebase |
| Phase 2: Bug fixes | ~10 | ~20 | Minimal | Correctness |
| Phase 3: Adapter infra | ~400 | ~150 | Medium | Maintainability |
| Phase 4: Builder cleanup | ~100 | ~40 | Medium | Maintainability |
| Phase 5: Error handling | ~20 | ~50 | Medium | Reliability |
| Phase 6: Consistency | ~30 | ~30 | Low | Consistency |
| Phase 7: Test quality | ~200 | ~100 | Low | Test reliability |
| **Total** | **~1,060** | **~390** | | **Net: -670 lines** |
