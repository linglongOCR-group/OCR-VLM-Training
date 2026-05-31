## Context

UniRec40M ingestion normalizes source labels before writing canonical text, formula, and table records. A previously built UniRec40M view showed inline formulas inside text or table labels as `\$...\$`, but MinerU annotation format expects unescaped `$...$` delimiters. Derived views can only be correct if canonical targets and any table/text serializers preserve the unescaped delimiter.

Current behavior to verify during implementation:

- UniRec standalone formula records strip display wrappers and store raw LaTeX in `target.latex`.
- UniRec text labels with `\(...\)` are intended to become `$...$`.
- Table records use cleaned text targets and may carry formula-bearing content through the same normalization path.

## Goals / Non-Goals

**Goals:**

- Guarantee UniRec text and table targets represent inline formulas as `$...$`.
- Add regression tests that fail on `\$...\$` output.
- Validate or rebuild affected UniRec40M canonical/view outputs so stale escaped labels are not reused.

**Non-Goals:**

- Change standalone formula task serialization.
- Change MinerU layout, OTSL table, prompt, reward, or VERL runtime behavior.
- Rewrite historical source annotations under `sources/`.

## Decisions

1. Normalize at UniRec ingestion before view construction.
   - Rationale: the source adapter owns UniRec-specific cleanup markers and formula wrapper conversion. Fixing only a view serializer would leave canonical data inconsistent and could miss future view configs.
   - Alternative considered: patch view labels during SFT view building. Rejected because it hides bad canonical targets and broadens the change to non-UniRec sources.

2. Test both text and table records.
   - Rationale: the reported bad output appears in text or table records, and table classification can route formula-bearing labels through a table target instead of a text target.
   - Alternative considered: only extend the existing text cleanup test. Rejected because it would not prove the table path is protected.

3. Treat materialized data as stale after the formatter fix.
   - Rationale: existing canonical/view Parquet files will not change until rebuilt. The implementation must include commands or evidence for regenerating the affected UniRec40M partitions or, at minimum, validating that the rebuilt view labels no longer contain `\$`.
   - Alternative considered: code-only fix. Rejected because the user found the issue in a previously built view, not just in unit behavior.

## Risks / Trade-offs

- Existing view data remains escaped until rebuilt -> include rebuild or validation steps in the implementation tasks.
- A broad string replacement could corrupt literal backslash-dollar text that is not a formula delimiter -> target UniRec formula wrapper normalization instead of global label replacement.
- Table heuristics may classify only some formula-bearing examples as tables -> add a synthetic table fixture with an inline formula and table metadata.

## Migration Plan

1. Patch UniRec label normalization and regression tests.
2. Run focused tests for UniRec adapter behavior.
3. Re-export affected UniRec40M canonical partitions or run a targeted smoke export.
4. Rebuild the affected UniRec40M view and validate that labels contain `$...$` and no `\$...\$` formula delimiters.

Rollback is code-level: revert the adapter/test change and rebuild the affected canonical/view outputs from the previous revision if needed.

## Open Questions

- Which concrete previously built UniRec40M view should be rebuilt first if more than one view was materialized from the affected canonical partitions?
