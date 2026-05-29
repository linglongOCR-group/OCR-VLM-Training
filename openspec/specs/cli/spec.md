# Data CLI Specification

## Purpose
The `docds` command-line tool is the primary interface for the document dataset processing pipeline. It provides subcommands for source export, canonical validation, view building and validation, reward scoring, and lineage tracing.

## Requirements

### Requirement: Export Source Command
The system SHALL provide an `export-source` subcommand that reads a named source adapter configuration and writes canonical records to the canonical store.

#### Scenario: Export with required arguments
- GIVEN a valid source adapter name and a `--source-config` path
- WHEN `export-source {name} --source-config {path}` is invoked
- THEN the system SHALL resolve the source configuration, instantiate the adapter, and produce canonical parquet partitions

#### Scenario: Export rejects missing source config
- GIVEN an invocation of `export-source` without `--source-config`
- WHEN the command is parsed
- THEN the system SHALL exit with an error indicating `--source-config` is required

#### Scenario: Export with overwrite-partitions flag
- GIVEN an `export-source` invocation with `--overwrite-partitions`
- WHEN the adapter runs
- THEN the system SHALL overwrite existing partition data rather than skipping completed partitions

#### Scenario: Export with skip-errors flag
- GIVEN an `export-source` invocation with `--skip-errors`
- WHEN individual records fail during export
- THEN the system SHALL log the error and continue processing remaining records

### Requirement: Validate Canonical Command
The system SHALL provide a `validate-canonical` subcommand that checks canonical records for structural integrity.

#### Scenario: Validate all canonical records
- GIVEN a canonical root directory with parquet partitions
- WHEN `validate-canonical` is invoked without filters
- THEN the system SHALL scan all record parquet files and report duplicate record IDs or missing required columns

#### Scenario: Validate filtered by task and source
- GIVEN a canonical root directory
- WHEN `validate-canonical --task layout --source mineru_publaynet` is invoked
- THEN the system SHALL validate only parquet files matching that task and source partition

#### Scenario: Validate reports duplicate IDs
- GIVEN canonical records where two files contain the same `record_id`
- WHEN `validate-canonical` is invoked
- THEN the system SHALL raise a ValueError listing the duplicate IDs

### Requirement: Build View Command
The system SHALL provide a `build-view` subcommand that reads a view configuration YAML and produces split parquet files for training.

#### Scenario: Build view from config
- GIVEN a valid view configuration YAML with include rules and split policy
- WHEN `build-view {view.yaml}` is invoked
- THEN the system SHALL produce train/val/test split parquet files under the view root directory

#### Scenario: Build view with overwrite disabled
- GIVEN an existing view directory
- WHEN `build-view {view.yaml} --overwrite` is not set
- THEN the system SHALL not clear existing asset directories before building

#### Scenario: Build view with overwrite enabled
- GIVEN an existing view directory with assets
- WHEN `build-view {view.yaml} --overwrite` is set
- THEN the system SHALL remove existing asset directories before writing new data

#### Scenario: Build view outputs stats
- GIVEN a successful view build
- WHEN the build completes
- THEN the system SHALL write a `stats.json` file containing split counts, total records, and optional filter statistics

### Requirement: Validate View Command
The system SHALL provide a `validate-view` subcommand that checks view parquet files for correctness.

#### Scenario: Validate view structure
- GIVEN a view root directory with parquet files
- WHEN `validate-view {view_root}` is invoked
- THEN the system SHALL verify required columns exist, prompts are non-empty, labels are non-empty, and split columns match file placement

#### Scenario: Validate view detects split leakage
- GIVEN view parquet files where the same `document_id` appears in both train and val splits
- WHEN `validate-view` is invoked
- THEN the system SHALL raise a ValueError reporting the leaked documents

#### Scenario: Validate view with image checking
- GIVEN a view root with image path references
- WHEN `validate-view {view_root} --require-images` is invoked
- THEN the system SHALL verify each referenced image file exists on disk and is readable

### Requirement: Reward Smoke Test Command
The system SHALL provide a `reward-smoke-test` subcommand that validates reward scoring using ground truth as predictions.

#### Scenario: Smoke test with default limit
- GIVEN a view root containing RLVR records with reward payloads
- WHEN `reward-smoke-test --view {view_root}` is invoked
- THEN the system SHALL score up to 100 records using their ground truth labels as predictions and output aggregate statistics (min, mean, max score)

#### Scenario: Smoke test with custom limit
- GIVEN a view root with RLVR records
- WHEN `reward-smoke-test --view {view_root} --limit 10` is invoked
- THEN the system SHALL score at most 10 RLVR records

### Requirement: Score Predictions Command
The system SHALL provide a `score-predictions` subcommand that scores model prediction outputs against view ground truth.

#### Scenario: Score predictions end to end
- GIVEN a view root and a predictions parquet file containing an `id` column and a `prediction` column
- WHEN `score-predictions --view {view_root} --predictions {path} --output {path}` is invoked
- THEN the system SHALL compute reward scores for each prediction and write results to the output parquet file

#### Scenario: Score predictions requires all arguments
- GIVEN an invocation missing `--predictions` or `--output`
- WHEN the command is parsed
- THEN the system SHALL exit with an error indicating the required argument is missing

### Requirement: Trace Command
The system SHALL provide a `trace` subcommand that resolves lineage chains from view records to canonical records.

#### Scenario: Trace by view record ID
- GIVEN a view root and canonical root
- WHEN `trace --view-record-id {id}` is invoked
- THEN the system SHALL return a JSON payload containing the view record and its corresponding canonical record

#### Scenario: Trace by canonical record ID
- GIVEN a canonical root
- WHEN `trace --canonical-record-id {id}` is invoked
- THEN the system SHALL return a JSON payload containing the matching canonical record

#### Scenario: Trace requires an ID argument
- GIVEN an invocation of `trace` without `--view-record-id` or `--canonical-record-id`
- WHEN the command is parsed
- THEN the system SHALL exit with an error indicating one of the ID arguments is required

### Requirement: Config Resolution
Every subcommand SHALL accept an optional `--config` argument pointing to a processing configuration file that provides default paths and registry settings.

#### Scenario: Config provides canonical root default
- GIVEN a processing config that specifies `canonical_root`
- WHEN a subcommand is invoked with `--config` but without `--canonical-root`
- THEN the system SHALL use the canonical root from the processing config

#### Scenario: Explicit path overrides config
- GIVEN a processing config with a `canonical_root` value
- WHEN a subcommand is invoked with both `--config` and `--canonical-root`
- THEN the system SHALL use the explicitly provided `--canonical-root` value

### Requirement: Progress Reporting
Long-running subcommands SHALL support progress reporting via `--progress`, `--no-progress`, and `--quiet` flags.

#### Scenario: Progress enabled on TTY
- GIVEN a TTY-attached stderr stream
- WHEN a subcommand is invoked without explicit progress flags
- THEN the system SHALL enable progress reporting by default

#### Scenario: Quiet mode suppresses output
- GIVEN the `--quiet` flag
- WHEN a subcommand runs
- THEN the system SHALL suppress all progress output

### Requirement: Build View Schema Inference
The build-view command SHALL infer a stable parquet schema by sampling records before writing.

#### Scenario: Schema covers all column variants
- GIVEN view records that include both SFT-only and RLVR-only columns
- WHEN the schema is inferred
- THEN the resulting parquet schema SHALL include all column types present across the sampled records

### Requirement: Export Source Worker Configuration
The export-source command SHALL support parallel processing via `--num-workers`, `--worker-chunksize`, and `--max-in-flight` flags.

#### Scenario: Parallel export
- GIVEN `--num-workers 4`
- WHEN `export-source` runs
- THEN the system SHALL distribute work across 4 worker processes

#### Scenario: Invalid worker count
- GIVEN `--num-workers` set to 0 or a negative value
- WHEN the adapter options are validated
- THEN the system SHALL raise an error
