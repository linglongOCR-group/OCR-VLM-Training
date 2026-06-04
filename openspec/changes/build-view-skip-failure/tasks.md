## 1. Regression Tests

- [x] 1.1 Add a failing test showing `docds build-view` still fails by default when a selected record raises during materialization.
- [x] 1.2 Add a failing test showing `docds build-view --skip-failure` skips a serializer/materialization failure and writes valid remaining records.
- [x] 1.3 Add a failing test showing schema inference skips failed sampled records in skip-failure mode and fails clearly when every sampled record fails.
- [x] 1.4 Add a failing test showing skipped failure diagnostics appear in `stats.json` with counts and capped examples.
- [x] 1.5 Add a failing test showing `enhanced_otsl_v1` table filtering excludes text-only table targets before materialization.
- [x] 1.6 Add or extend a multi-worker build-view test so worker materialization failures are reported without dropping successful rows from the same batch.

## 2. Build-View CLI Surface

- [x] 2.1 Add `--skip-failure` to the `build-view` CLI parser.
- [x] 2.2 Pass the parsed skip-failure option into `ViewBuilder.build()`.
- [x] 2.3 Preserve existing fail-fast behavior when `--skip-failure` is not set.

## 3. Failure Diagnostics Model

- [x] 3.1 Add an internal structured representation for materialization failures.
- [x] 3.2 Include phase, record ID, source name, task, document/page/region IDs, target format, target keys, target preview, exception type, and exception message.
- [x] 3.3 Add preview truncation so diagnostics are useful without making logs or `stats.json` excessively large.
- [x] 3.4 Aggregate failure counts by phase, task, source, and exception type.

## 4. Materialization Skip Behavior

- [x] 4.1 Wrap schema-inference `_materialize()` calls with skip-failure handling.
- [x] 4.2 Fail schema inference with a clear error when skip-failure mode leaves no valid sampled rows.
- [x] 4.3 Wrap single-worker final materialization with skip-failure handling.
- [x] 4.4 Update multi-worker batch materialization to return successes and structured failures to the parent process.
- [x] 4.5 Ensure split counts and total record counts include only successfully written rows.
- [x] 4.6 Log concise skipped-record diagnostics through the existing progress/logging path.

## 5. Serializer-Aware Filtering

- [x] 5.1 Pass the selected target serialization mapping into label filtering.
- [x] 5.2 Update table label filtering so `enhanced_otsl_v1` accepts only `enhanced_otsl`, `otsl`, or convertible `html`.
- [x] 5.3 Preserve existing text, formula, reserved-media-token, and HTML emptiness filtering behavior.
- [x] 5.4 Count text-only `enhanced_otsl_v1` table targets as filtered unusable labels rather than materialization failures.

## 6. Verification and Failed Build Completion

- [x] 6.1 Run focused data pipeline tests for build-view filtering and skip-failure behavior.
- [x] 6.2 Run the broader relevant test target, such as `pytest tests/test_data_pipeline.py -q`.
- [x] 6.3 Run `openspec validate --all --strict`.
- [ ] 6.4 Rerun the previously failed `mineru25-trim-18layer-0604` build with `--skip-failure` when the dataset root is available.
- [ ] 6.5 Run `docds validate-view` on the rebuilt view with image checking when image reachability matters.
- [ ] 6.6 Inspect and report `stats.json` split counts, filtered counts, and skipped failure counts before treating the build as complete.
