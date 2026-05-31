# Tasks

## 1. Difference Audit

- [x] 1.1 Record the old-tree file buckets: already present, obsolete/manual sync, temporary artifacts, and preservation candidates.
- [x] 1.2 Confirm archived docs in the old tree are represented by `docs/archive/` and `openspec/specs/`.
- [x] 1.3 Exclude generated outputs, bytecode, notebook scratch changes, local sync helpers, and root-level CSV artifacts.

## 2. UniRec40M Source Ingestion

- [x] 2.1 Add `tools/data_management/sources/adapters/unirec.py` adapted to current shared adapter helpers.
- [x] 2.2 Register the `unirec` source adapter in `configs/data/processing.yaml`.
- [x] 2.3 Add UniRec40M category source profiles under `configs/data/sources/`.
- [x] 2.4 Add unit tests for text cleanup, formula wrapper stripping, table classification, image path resolution, and canonical record emission.

## 3. View Configs and View Safety

- [x] 3.1 Add `configs/data/views/unirec40m_mineru_train_nested_reference.yaml`.
- [x] 3.2 Add the missing MinerU2.5 pretraining configs `mineru2.5_trim_pretrain_v3_fixed.yaml` and `mineru2.5_trim_pretrain_v4.yaml`.
- [x] 3.3 Add SFT view-builder filtering for labels containing reserved literal media tokens.
- [x] 3.4 Add runtime escaping/restoration for assistant-side literal media tokens in SFT dataset message construction.
- [x] 3.5 Add tests for SFT filtering and runtime preservation.
- [x] 3.6 Add plain table-text serialization for UniRec table labels.

## 4. View Deployment CLI

- [x] 4.1 Add `tools/data_management/deploy.py` using current path/progress helpers.
- [x] 4.2 Add `scripts/data/deploy_view_to_nodes` wrapper.
- [x] 4.3 Support manifest generation from view files plus `images` and `images_path` parquet columns.
- [x] 4.4 Support default rsync transfer with ssh/rsync options and remote mkdir.
- [x] 4.5 Support opt-in tar streaming transfer for initial seeding.
- [x] 4.6 Support dry-run, progress, parallel parquet scanning, asset existence checking, and target parsing from repeated args, comma list, or file.
- [x] 4.7 Print remote `OCR_DATA_ROOT` and view path overrides after manifest construction.
- [x] 4.8 Add focused deployment tests.

## 5. Validation

- [x] 5.1 Run `PYTHONPATH=. pytest tests/test_dataset_deploy.py -q`.
- [x] 5.2 Run `PYTHONPATH=. pytest tests/test_data_pipeline.py tests/test_verl_runtime_dataset.py -q`.
- [x] 5.3 Run relevant existing SFT/image/training tests: `tests/test_sft_validation_flags.py`, `tests/test_mineru_image_resolution.py`, `tests/test_sft_freeze_vision.py`, `tests/test_sft_position_ids_patch.py`, and `tests/test_train_scripts.py`.
- [x] 5.4 Run `scripts/data/deploy_view_to_nodes --help`.
- [x] 5.5 Run `openspec validate --all --strict` if the OpenSpec CLI is installed.
