## Why

Large `docds build-view` jobs can run for hours over millions of canonical records and currently abort when a single record fails during view materialization. This blocks otherwise usable builds and gives operators too little context to quickly locate the bad source record.

## Scope

This change covers best-effort skipping for per-record materialization failures in `docds build-view`, plus structured diagnostics for failed records. It does not change source export behavior, canonical schemas, serializer strictness, or the default fail-fast build behavior.

## What Changes

- Add an explicit `docds build-view --skip-failure` option.
- Keep the default behavior fail-fast unless `--skip-failure` is provided.
- When skip-failure mode is enabled, skip records that fail during schema inference or final materialization and continue building remaining records.
- Log concise failure diagnostics containing the record identity, source, task, target format, target summary/content preview, and exception cause.
- Persist skipped failure counts and capped failure examples in `stats.json` for post-build auditing.
- Tighten view label filtering for table records so table targets are considered buildable only when compatible with the configured table serializer.
- Fail clearly if all candidate records needed for schema inference fail and a stable schema cannot be inferred.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `cli`: `build-view` gains the explicit `--skip-failure` operator control.
- `view-construction`: view materialization gains opt-in skipped failure reporting and serializer-aware table label filtering.

## Impact

Affected code includes `tools/data_management/cli.py`, `tools/data_management/views/builder.py`, and focused tests in `tests/test_data_pipeline.py`. The user-facing API change is one new CLI flag. Output views may contain fewer records only when skip-failure mode is explicitly enabled, and `stats.json` will expose that data loss.

## Risks

- Operators could overuse skip-failure mode and miss systemic data corruption. The design mitigates this by keeping fail-fast as the default and surfacing failure counts/examples in logs and stats.
- Multi-worker materialization needs deterministic aggregation of failure diagnostics without losing successful rows from the same batch.
- Schema inference must not skip so many records that output schema becomes invalid or incomplete.

## Open Questions

None. The approved design scopes skipping to per-record materialization failures and keeps earlier selection/export semantics unchanged.
