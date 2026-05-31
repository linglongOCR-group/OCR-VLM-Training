# Migration Analysis: Divergent Data Module Work

## Compared Inputs

- Current branch: `/tmp/ocr-vlm-tuning-dev-data-module` at `fee75cf85492854dc7970997d3e89bc8a7de2502`
- Divergent copy: `/home/byhou/projects/ocr-vlm-tuning` at `793326965688695826f6d1b50aac93c871b4f746`, branch `data-module`, behind `origin/data-module` by 22 commits

## Already Present in Current Branch

Byte-identical old-tree changes include:
- Training launch scripts: `run_grpo_fsdp.sh`, `run_sft.sh`, `run_multinode_sft.sh`, `run_multinode_sft_new.sh`
- View/source configs: HaoY, PubTable1M, several MinerU2.5 nested-reference configs
- Runtime/path helpers: `tools/data_management/paths.py`, `progress.py`, `runtime/image_columns.py`, `config/resolver.py`, canonical reader updates
- SFT trainer helpers and tests: `verl_plugins/trainers/*`, SFT freeze/position/validation tests, train script tests, MinerU image resolution test

## Obsolete or Manual-Sync Differences

- Old root docs `docs/dataset-data-spec.md`, `docs/dataset-process-module-spec.md`, and `docs/mineru-data-format.md` are byte-identical to current `docs/archive/` copies and are represented by OpenSpec specs.
- Old adapter edits that inline `_ShardWriter`, export reports, and multiprocessing helpers are superseded by current `tools/data_management/sources/adapters/_shared.py`.
- Old `.gitignore` would hide `configs/data/sources/*` except two profiles; current branch intentionally tracks source profiles.
- Old `docds build-view --overwrite` default changes behavior globally and should not be migrated without a separate operator-facing decision.

## Temporary or Artifact Differences

- Deleted/generated Hydra `outputs/` files.
- Bytecode under `__pycache__/`.
- `test.ipynb`, `HaoY-Formula-Labels.csv`, `sync_codebase.sh`, `.codex` deletion, and local permission-limited `extra-info/`.
- `docs/superpowers/` historical planning notes now represented by OpenSpec migration artifacts.

## Selected for Preservation

- UniRec40M source adapter and source profiles.
- UniRec40M MinerU2.5 nested-reference view config.
- Missing MinerU2.5 pretraining view configs: `mineru2.5_trim_pretrain_v3_fixed.yaml` and `mineru2.5_trim_pretrain_v4.yaml`.
- Deployment CLI for staging views and referenced assets to remote nodes with rsync/tar modes.
- SFT view-builder and runtime safeguards for reserved literal media tokens.

## Reimplementation Notes

- Do not copy adapter files wholesale; port UniRec behavior onto current shared helpers.
- Preserve current schema stabilization and nullable-field handling in `schemas.py`.
- Preserve current `NamedTuple` validation task structure in `views/validator.py`.
- Keep deployment as a standalone data script and leave training launch contracts unchanged.
