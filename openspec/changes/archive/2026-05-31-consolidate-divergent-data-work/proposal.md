# Proposal: Consolidate Divergent Data Module Work

## Intent

Cleanly reapply the meaningful work from `/home/byhou/projects/ocr-vlm-tuning` onto the current `dev/data-module` branch without merging the old working tree wholesale. The migration preserves only additive dataset, deployment, and SFT-safety behavior that is absent from the current branch, while discarding stale manual sync edits, generated artifacts, and duplicated older implementations.

## Scope

In scope:
- UniRec40M canonical export support for nested subset JSONL records.
- UniRec40M source profiles and a nested-reference MinerU2.5 SFT view config.
- Missing curated SFT view configs from the divergent tree.
- View deployment CLI that stages a view and referenced assets to training nodes.
- SFT safeguards for literal `<image>` / `<video>` tokens in assistant labels.
- Tests for the migrated behavior.

Out of scope:
- Wholesale copying or merging `/home/byhou/projects/ocr-vlm-tuning`.
- Reverting current OpenSpec migration, shared adapter helpers, schema fixes, or validation improvements.
- Generated outputs, notebooks, bytecode, local sync scripts, CSV scratch files, and root-level archived docs.
- Changing the remote training contract beyond `OCR_DATA_ROOT` plus the staged view path.

## Impact

- Affected capabilities: [[data-ingestion]], [[view-construction]], [[cli]], [[training-bootstrap]]
- Affected code areas: `tools/data_management/sources/adapters/`, `tools/data_management/views/`, `tools/data_management/runtime/`, `tools/data_management/deploy.py`, `configs/data/`, `tests/`
- Affected users or operators: Dataset builders and multi-node training operators
- Compatibility impact: Additive. Existing adapters, view configs, and training launch behavior remain valid.
- Migration impact: Old-tree code must be adapted to the current branch's shared helper modules and OpenSpec-era layout instead of copied as-is.

## Risks

- UniRec40M classification heuristics may misclassify ambiguous formula/table labels.
- Deployment dry-runs still need to scan parquet metadata before printing transfer commands, which can take time for large views.
- Tar transfer mode is speed-first and non-incremental; operators must choose it intentionally.
- Media-token filtering can drop records that intentionally contain literal reserved tokens.

## Open Questions

- Whether remote free-space preflight should be included in this migration or tracked as a follow-up.
- Whether UniRec40M source profiles should remain one file per category shard or be collapsed into a single parameterized profile.
- Whether the two missing MinerU2.5 pretraining view configs are both still operationally needed.
