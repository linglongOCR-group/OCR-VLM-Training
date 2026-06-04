## Context

`docds build-view` currently loads selected canonical records, applies configured excludes, sampling, label filters, and image filters, then infers a parquet schema from sampled materialized rows before writing all materialized rows. Both schema inference and final writing call the same per-record materialization path, which resolves prompts, serializers, image references or bytes, and optional reward payloads.

The current behavior is fail-fast: any serializer, image transform, image read, or record-shape error raised by `_materialize()` aborts the build. This is useful by default, but it is operationally expensive for large runs where a few malformed canonical records should not prevent a usable view from completing.

## Goals / Non-Goals

**Goals:**

- Add an explicit `--skip-failure` option for `docds build-view`.
- Keep fail-fast behavior as the default when the option is absent.
- Skip only per-record materialization failures in schema inference and final row writing.
- Emit actionable diagnostics for skipped records, including source, record identity, target summary, and exception cause.
- Persist skipped failure counts and capped examples in `stats.json`.
- Tighten table label filtering so records incompatible with the selected table serializer are filtered before materialization when possible.
- Support single-worker and multi-worker builds with the same external behavior.

**Non-Goals:**

- Do not add skip behavior to `export-source`, canonical validation, or view validation.
- Do not weaken serializer validation or make serializers accept invalid targets.
- Do not skip global configuration, selection, split assignment, or schema-unification errors.
- Do not automatically rebuild remote or historical failed views as part of the code change; provide the command path for operators to rerun them after implementation.

## Decisions

### Decision: Opt-in skip mode only

`--skip-failure` will be an explicit build-view CLI flag passed into `ViewBuilder.build()`. Without the flag, current fail-fast behavior remains unchanged.

Alternative considered: default to best-effort skipping. Rejected because silent data loss is dangerous for training datasets and would alter existing automation.

### Decision: Skip only materialization failures

The skip boundary will wrap `_materialize()` calls during schema inference and final writing. Failures before materialization, such as invalid config, unreadable canonical partitions, empty selected record sets, and split assignment errors, will remain fatal.

Alternative considered: broad best-effort wrapping across the whole builder. Rejected because it would make the resulting view harder to audit and could hide systemic configuration problems.

### Decision: Track structured failure records

The builder will collect failure diagnostics as structured data with fields such as `phase`, `record_id`, `source_name`, `task`, `document_id`, `page_id`, `region_id`, `target_format`, `target_keys`, `target_preview`, `exception_type`, and `message`. Logs will include concise versions of the same information. `stats.json` will include total failed records, phase/type/source/task counts, and a capped list of examples.

Alternative considered: print exception text only. Rejected because operators need enough lineage context to inspect or exclude the bad canonical record without rerunning with ad hoc instrumentation.

### Decision: Preserve schema inference safety

In skip mode, schema inference may skip failed sampled records. If no valid sampled row remains, the build will fail with a clear message because there is no reliable schema to write. Final materialization may then skip additional records and write only successful rows.

Alternative considered: synthesize a schema from static column definitions. Rejected for now because stage-specific and optional columns are already inferred from real materialized rows, and broad static schema work is outside this change.

### Decision: Make table label filtering serializer-aware

The current generic table label filter treats `target.text` as serializable, while `enhanced_otsl_v1` requires `enhanced_otsl`, `otsl`, or `html`. The filter should account for the selected table serializer so known-unserializable rows are counted as filtered labels instead of relying on skip-failure mode.

Alternative considered: leave filtering unchanged and rely only on `--skip-failure`. Rejected because avoidable bad rows should be filtered deterministically before materialization.

## Risks / Trade-offs

- **Risk: Operators miss large-scale data corruption** -> Keep skip mode opt-in, report counts in logs and stats, and expose examples for inspection.
- **Risk: Multi-worker errors lose record context** -> Return either successful rows or structured failure payloads from worker batches so the parent process can aggregate and log consistently.
- **Risk: Schema inference misses column variants after skipping failures** -> Preserve existing sampling behavior and only skip failed records; fail if all sampled records fail.
- **Risk: `stats.json` grows too large** -> Cap stored failure examples while retaining aggregate counts.

## Migration Plan

No data migration is required. Existing commands continue to fail fast. Operators who need best-effort recovery rerun failed builds with `--skip-failure`, then run `docds validate-view` and inspect `stats.json` failure counts before launching training.

Rollback is to omit `--skip-failure` or revert the code change. Views built with skip mode remain ordinary view directories but have lower row counts and failure diagnostics in `stats.json`.

## Open Questions

None.
