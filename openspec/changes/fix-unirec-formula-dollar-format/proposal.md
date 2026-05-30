## Why

Previously built UniRec40M views contain inline formulas in text or table labels formatted as `\$...\$`. MinerU annotation standards require `$...$` delimiters without backslash escaping, so the UniRec export and view path must preserve unescaped dollar delimiters.

## Scope

This change covers UniRec40M label normalization for formulas embedded in text and table records, plus tests and validation that generated canonical/view labels no longer contain escaped math delimiters.

## What Changes

- Ensure UniRec inline formula normalization emits `$...$`, not `\$...\$`, for text records.
- Apply the same unescaped inline formula rule to table records that preserve formula-bearing text.
- Add regression coverage for text and table labels containing inline formulas.
- Rebuild or validate affected UniRec40M view outputs after the fix so existing bad formatting is not silently carried forward.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `data-ingestion`: Tighten UniRec40M label normalization requirements so inline math delimiters in text and table targets are unescaped `$...$` according to MinerU annotation standards.

## Impact

- Affected code: `tools/data_management/sources/adapters/unirec.py` and related tests in `tests/test_data_pipeline.py`.
- Affected data: UniRec40M canonical text/table records and any derived SFT views built from those records.
- APIs/dependencies: No public API or dependency changes expected.

## Risks

- Existing materialized canonical/view data may remain stale until rebuilt.
- A narrow formatter fix could miss table-specific paths if table labels are normalized separately from text labels.

## Open Questions

- Which already materialized UniRec40M view partitions should be rebuilt after implementation, if multiple historical views exist?
