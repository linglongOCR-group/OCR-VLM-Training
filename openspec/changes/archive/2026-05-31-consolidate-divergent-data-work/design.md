# Design: Consolidate Divergent Data Module Work

## Difference Analysis

The divergent working copy is 22 commits behind `origin/data-module` and includes manual sync edits plus local work. The current branch already contains many old-tree changes byte-for-byte, including SFT/GRPO launch updates, several MinerU view configs, runtime image-column resolution, SFT trainer helpers, and most SFT validation tests.

Selected preservation candidates are behavior that is absent from the current branch and still aligned with the current architecture:
- `tools/data_management/deploy.py`, `scripts/data/deploy_view_to_nodes`, and `tests/test_dataset_deploy.py`.
- UniRec40M adapter behavior from `tools/data_management/sources/adapters/unirec.py`.
- UniRec40M source profiles and `unirec40m_mineru_train_nested_reference.yaml`.
- Missing view configs `mineru2.5_trim_pretrain_v3_fixed.yaml` and `mineru2.5_trim_pretrain_v4.yaml`.
- Reserved media token filtering and runtime escaping for assistant-side literal media tokens.

Discarded differences:
- Root `docs/*.md` files already archived under `docs/archive/` and represented in `openspec/specs/`.
- Old adapter-local `_ShardWriter` / report duplication, because the current branch has `tools/data_management/sources/adapters/_shared.py`.
- `.gitignore` changes that would hide tracked source configs.
- Generated `outputs/`, `__pycache__`, notebook edits, local `.codex` deletion, `HaoY-Formula-Labels.csv`, and temporary sync/debug files.
- Old `docds build-view --overwrite` default change, because the current branch's explicit overwrite behavior is safer.

## Technical Approach

Reimplement the selected behavior against current branch structure:

1. Add a UniRec adapter using the current `SourceAdapter` and shared adapter helpers. It should export document, page, region, page-render asset, region-crop asset, and task records using source-referenced image paths.
2. Register the UniRec adapter through `configs/data/processing.yaml` rather than hard-coded CLI wiring.
3. Add UniRec40M source profiles and view configs as configuration-only assets.
4. Add a deployment module and script wrapper that builds a dataset-relative manifest from view files plus parquet image references, then transfers that manifest via rsync or tar.
5. Add SFT media-token safeguards in the current view builder and runtime dataset wrapper without regressing schema stabilization or logging behavior that exists on the current branch.
6. Add tests that prove behavior in isolation and fit the current test suite style.

## Architecture Decisions

### Decision: Reimplement on Current Helpers

Rationale:
- The old working copy duplicated shard writers and export reports inside adapters.
- The current branch already extracted shared adapter helpers.
- Copying old adapter files would regress maintainability and revive duplication.

### Decision: Keep Deployment Separate from Training Launch

Rationale:
- Operators only need staged data plus `OCR_DATA_ROOT` and view parquet paths.
- The deployment task is a data-management concern, not a training launcher concern.
- This preserves the existing runtime contract.

### Decision: Default to Rsync, Add Tar as Opt-in

Rationale:
- Rsync is safer for incremental updates and retries.
- Tar streaming is faster for initial seeding but cannot efficiently skip already-transferred files.
- Keeping tar opt-in makes the speed/reliability tradeoff explicit.

### Decision: Filter Reserved Assistant Media Tokens at View Build

Rationale:
- MinerU/VERL processors treat reserved tokens as multimodal placeholders.
- Labels containing literal `<image>` or `<video>` can corrupt SFT message construction.
- Dropping these records during SFT view construction prevents invalid training samples, while runtime escaping protects legacy views that already contain such labels.

## Migration Plan

- Start from the current `dev/data-module` branch only.
- Reapply candidate behavior file-by-file, adapting imports and helper usage.
- Do not copy root docs or generated artifacts from the old tree.
- Validate each migrated feature with focused tests.

## Rollback

All changes are additive. Rollback can remove the new adapter/profile/view/deploy files and revert the small registration/runtime/view-builder changes without affecting existing datasets or training configs.

## Validation Strategy

- Run targeted tests for dataset deployment, UniRec export, SFT media-token filtering, and runtime media-token preservation.
- Run existing related tests for data pipeline, runtime dataset, SFT validation, train scripts, and image resolution.
- Run OpenSpec validation if the CLI is available.
