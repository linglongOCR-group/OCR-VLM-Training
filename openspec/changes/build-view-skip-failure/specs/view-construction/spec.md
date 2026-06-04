## ADDED Requirements

### Requirement: Build View Materialization Failure Skipping
The ViewBuilder SHALL support opt-in skipping of per-record materialization failures during schema inference and final row writing.

#### Scenario: Schema inference skips failed sample records
- **GIVEN** skip-failure mode is enabled
- **AND** schema inference samples records where one record raises during materialization and another record materializes successfully
- **WHEN** the ViewBuilder infers the parquet schema
- **THEN** the ViewBuilder SHALL skip the failed sampled record
- **AND** the ViewBuilder SHALL infer the schema from successfully materialized sampled records

#### Scenario: Schema inference fails when every sampled record fails
- **GIVEN** skip-failure mode is enabled
- **AND** every sampled record raises during materialization
- **WHEN** the ViewBuilder infers the parquet schema
- **THEN** the ViewBuilder SHALL fail with an error explaining that no valid records were available for schema inference

#### Scenario: Final materialization skips failed records
- **GIVEN** skip-failure mode is enabled
- **AND** selected records include both valid records and records that raise during materialization
- **WHEN** the ViewBuilder writes output rows
- **THEN** it SHALL write rows for successfully materialized records
- **AND** it SHALL omit rows for failed records
- **AND** split counts SHALL reflect only successfully written rows

#### Scenario: Skip mode records failure diagnostics
- **GIVEN** skip-failure mode is enabled
- **AND** a record fails during schema inference or final materialization
- **WHEN** the ViewBuilder records the failure
- **THEN** diagnostics SHALL include the phase, record ID, source name, task, target format, target keys or preview, exception type, and exception message
- **AND** `stats.json` SHALL include aggregate failure counts and capped failure examples

### Requirement: Serializer-Aware Table Label Filtering
The ViewBuilder SHALL filter table records according to the selected table target serializer so records known to be unsupported by that serializer are excluded before materialization.

#### Scenario: Enhanced OTSL table filter rejects text-only target
- **GIVEN** a view configuration with `target_serialization.table: enhanced_otsl_v1`
- **AND** a canonical table record whose target contains `text` but no `enhanced_otsl`, `otsl`, or `html`
- **WHEN** the ViewBuilder applies label filters
- **THEN** the table record SHALL be excluded from the output view
- **AND** the exclusion SHALL be counted as an empty or unserializable label filter result

#### Scenario: Enhanced OTSL table filter keeps supported target
- **GIVEN** a view configuration with `target_serialization.table: enhanced_otsl_v1`
- **AND** a canonical table record whose target contains `enhanced_otsl`, `otsl`, or convertible `html`
- **WHEN** the ViewBuilder applies label filters
- **THEN** the table record SHALL remain eligible for materialization
