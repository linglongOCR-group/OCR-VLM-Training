# Delta for View Construction

## ADDED Requirements

### Requirement: UniRec40M MinerU2.5 Nested-Reference View

The repository SHALL provide a MinerU2.5 SFT view configuration that combines UniRec40M text, formula, and table records with nested image references.

#### Scenario: Build UniRec40M training view

- GIVEN canonical UniRec40M category outputs exist under `canonical/`
- WHEN `docds build-view configs/data/views/unirec40m_mineru_train_nested_reference.yaml` runs
- THEN the generated view SHALL include text, formula, and table tasks from the configured UniRec40M sources
- AND the view SHALL store image references relative to `OCR_DATA_ROOT`
- AND the split policy SHALL produce a train-only view

### Requirement: Migrated MinerU2.5 Pretraining View Configs

The repository SHALL retain the missing curated MinerU2.5 pretraining view configurations that are not already present on the current branch.

#### Scenario: Build fixed v3 pretraining view

- GIVEN required canonical sources exist
- WHEN `docds build-view configs/data/views/mineru2.5_trim_pretrain_v3_fixed.yaml` runs
- THEN the view SHALL combine fixed Fintech, PubTable1M, HaoY, and DocBank records according to the config sampling policy

#### Scenario: Build v4 pretraining view

- GIVEN required canonical sources exist
- WHEN `docds build-view configs/data/views/mineru2.5_trim_pretrain_v4.yaml` runs
- THEN the view SHALL combine PubTable1M, capped Fintech, HaoY, DocBank, and MinerU Hybrid records according to the config sampling policy

### Requirement: Reserved Media Token Filtering for SFT Views

The SFT view builder SHALL drop records whose assistant label text contains reserved media placeholder tokens.

#### Scenario: Drop assistant label with literal media token

- GIVEN an SFT view record whose serialized target label contains `<image>` or `<video>`
- WHEN the view builder applies label filters
- THEN the record SHALL be excluded from the output view
- AND `stats.json` SHALL report the dropped count under `reserved_media_token_label`
