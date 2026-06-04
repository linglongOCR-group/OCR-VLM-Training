## ADDED Requirements

### Requirement: Build View Skip Failure Option
The `build-view` subcommand SHALL provide a `--skip-failure` option that allows per-record materialization failures to be skipped during view building.

#### Scenario: Build view fails fast by default
- **GIVEN** a view configuration that selects a canonical record that fails during materialization
- **WHEN** `docds build-view {view.yaml}` is invoked without `--skip-failure`
- **THEN** the command SHALL fail with the materialization error

#### Scenario: Build view continues with skip-failure
- **GIVEN** a view configuration that selects both valid records and a canonical record that fails during materialization
- **WHEN** `docds build-view {view.yaml} --skip-failure` is invoked
- **THEN** the command SHALL skip the failed record
- **AND** the command SHALL continue materializing remaining valid records
- **AND** the command SHALL write output view parquet files for the successful records

#### Scenario: Skip-failure reports skipped records
- **GIVEN** `docds build-view {view.yaml} --skip-failure` skips one or more records
- **WHEN** the command completes
- **THEN** progress or log output SHALL include diagnostics for skipped records containing source, task, record identity, and failure cause
- **AND** `stats.json` SHALL include skipped materialization failure counts
