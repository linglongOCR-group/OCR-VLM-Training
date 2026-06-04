## 1. Regression Coverage

- [x] 1.1 Add or update a UniRec text-export test that fails if inline formula output contains `\$...\$` instead of `$...$`.
- [x] 1.2 Add or update a UniRec table-export test with table metadata and inline formula content, asserting `$...$` is preserved and `\$...\$` is absent.
- [x] 1.3 Run the focused UniRec adapter tests and confirm the new assertions fail before the implementation change.

## 2. UniRec Normalization Fix

- [x] 2.1 Locate the UniRec label normalization path that converts source inline formula wrappers for text and table records.
- [x] 2.2 Update the normalization logic so inline formulas emit unescaped `$...$` delimiters without applying a global replacement that could corrupt non-formula literal text.
- [x] 2.3 Confirm standalone formula task targets still strip display wrappers and store raw LaTeX unchanged.

## 3. Verification And Data Refresh

- [x] 3.1 Re-run the focused UniRec adapter tests and confirm they pass.
- [x] 3.2 Run the broader affected data-pipeline test subset covering UniRec export and view construction.
- [x] 3.3 Run a targeted UniRec40M smoke export or full affected canonical re-export with `OCR_DATA_ROOT=/mnt/sas-server-0/DataMgmt`.
- [x] 3.4 Rebuild the affected UniRec40M view and validate that assistant labels contain `$...$` formula delimiters and no `\$...\$` formula delimiters.
