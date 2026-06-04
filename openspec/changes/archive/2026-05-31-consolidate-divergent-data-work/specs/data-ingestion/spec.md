# Delta for Data Ingestion

## ADDED Requirements

### Requirement: UniRec40M Nested JSONL Export

The source ingestion system SHALL export UniRec40M nested subset JSONL records into the canonical dataset layout.

#### Scenario: Export text recognition crop

- GIVEN a UniRec40M source profile with `adapter: unirec`
- AND a source root containing `<subset>/annotations/records.jsonl` and source images
- WHEN `docds export-source unirec --source-config <profile>` runs for text tasks
- THEN the system SHALL write canonical document, page, region, asset manifest, and text task records
- AND image assets SHALL reference the original dataset-relative image path rather than copying image bytes

#### Scenario: Export formula record

- GIVEN a UniRec40M record whose label is a standalone LaTeX formula
- WHEN the UniRec adapter exports formula tasks
- THEN the system SHALL write a formula task record whose target contains stripped LaTeX without display wrappers

#### Scenario: Export table record

- GIVEN a UniRec40M record with table hints in metadata or table-like label text
- WHEN the UniRec adapter exports table tasks
- THEN the system SHALL write a table task record with the cleaned table text target

### Requirement: UniRec40M Label Normalization

The UniRec adapter SHALL normalize source-specific text markers before writing canonical targets.

#### Scenario: Remove source line tokens

- GIVEN a UniRec label containing `<|ln|>`, `<|pn|>`, `<|sn|>`, or `<<<change_line_token_wrap>>>`
- WHEN the label is exported as text
- THEN those tokens SHALL be removed from the canonical target

#### Scenario: Convert inline math wrappers

- GIVEN a UniRec text label containing `\(...\)` inline math
- WHEN the label is exported as text
- THEN the inline math content SHALL be preserved using `$...$` delimiters

### Requirement: UniRec40M Category Profiles

The repository SHALL provide source profiles for the UniRec40M category shards selected for MinerU2.5 training.

#### Scenario: Resolve UniRec40M source profile

- GIVEN `configs/data/processing.yaml`
- WHEN a UniRec40M category profile is passed to `docds export-source`
- THEN the configured source registry SHALL resolve the `unirec` adapter class
- AND the profile SHALL resolve paths relative to `OCR_DATA_ROOT`
