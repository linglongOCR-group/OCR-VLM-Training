# OpenSpec Migration Report

**Date:** 2026-05-29
**Branch:** `data-module`

## Summary

Migrated scattered project documentation (8 files across `docs/`, root, and `docs/superpowers/`) into a clean OpenSpec knowledge base with 11 capability specs and 1 proposed change. All artifacts pass `openspec validate --all --strict`. Original documents are preserved in place.

## New OpenSpec Structure

```
openspec/
├── config.yaml
├── specs/
│   ├── checkpoint-tracking/spec.md     (9 requirements)
│   ├── cli/spec.md                     (11 requirements)
│   ├── data-ingestion/spec.md          (15 requirements)
│   ├── dataset-architecture/spec.md    (19 requirements)
│   ├── lineage/spec.md                 (7 requirements)
│   ├── mineru-data-format/spec.md      (14 requirements)
│   ├── reward-system/spec.md           (10 requirements)
│   ├── target-serialization/spec.md    (11 requirements)
│   ├── training-bootstrap/spec.md      (16 requirements)
│   ├── validation/spec.md              (14 requirements)
│   └── view-construction/spec.md       (15 requirements)
└── changes/
    ├── checkpoint-lineage-tracking/
    │   ├── proposal.md
    │   ├── design.md
    │   ├── tasks.md
    │   └── specs/
    │       ├── training-bootstrap/spec.md  (delta)
    │       └── checkpoint-tracking/spec.md (delta)
    └── archive/
```

**Total:** 141 requirements across 11 specs, 1 change with 23 tasks.

## Source-to-Target Mapping

| Original file | New location | Action | Notes |
|---|---|---|---|
| `SPEC.md` | `openspec/specs/training-bootstrap/spec.md` | Rewritten | 647 lines of mixed scope/requirements → 16 focused behavior requirements with scenarios |
| `docs/dataset-data-spec.md` | `openspec/specs/dataset-architecture/spec.md` + `openspec/specs/view-construction/spec.md` | Split | 1882 lines → 19 architecture reqs + 15 view construction reqs |
| `docs/dataset-process-module-spec.md` | `openspec/specs/data-ingestion/spec.md` + `openspec/specs/reward-system/spec.md` + `openspec/specs/target-serialization/spec.md` + `openspec/specs/cli/spec.md` + `openspec/specs/validation/spec.md` | Split | 1718 lines → 5 focused specs |
| `docs/mineru-data-format.md` | `openspec/specs/mineru-data-format/spec.md` | Rewritten | 688 lines → 14 requirements covering MinerU2.5 data contract |
| `docs/superpowers/specs/2026-05-07-checkpoint-lineage-design.md` | `openspec/changes/checkpoint-lineage-tracking/` | Converted | Design spec → proposal.md + design.md + tasks.md + delta specs |
| `docs/codebase-audit-report.md` | `docs/archive/codebase-audit-report.md` | Preserved | Historical snapshot |
| `docs/legacy/completed-items.md` | `docs/archive/completed-items.md` | Preserved | Historical log |
| `README.md` | Kept in place | Cross-referenced | Operational instructions remain; behavior claims extracted into specs |

## Capability Inventory

| Capability | Spec path | Requirements | Source documents | Notes |
|---|---|---|---|---|
| dataset-architecture | `openspec/specs/dataset-architecture/` | 19 | dataset-data-spec.md (§1-10) | Three-layer architecture, entities, assets, coordinates, IDs, splits |
| data-ingestion | `openspec/specs/data-ingestion/` | 15 | dataset-process-module-spec.md (§4-6) | Source adapters, canonical writer, asset manager |
| view-construction | `openspec/specs/view-construction/` | 15 | dataset-data-spec.md (§10-15), dataset-process-module-spec.md (§9-10) | View builder, schemas, selection, splits, prompts |
| target-serialization | `openspec/specs/target-serialization/` | 11 | dataset-process-module-spec.md (§6) | Serializer adapters for layout, table, formula, text |
| reward-system | `openspec/specs/reward-system/` | 10 | dataset-process-module-spec.md (§7-8) | Reward adapters, Levenshtein, registry, payloads |
| training-bootstrap | `openspec/specs/training-bootstrap/` | 16 | SPEC.md (§1-16) | SFT/GRPO training, W&B, checkpoints, resume, config |
| checkpoint-tracking | `openspec/specs/checkpoint-tracking/` | 9 | SPEC.md (§7.4), README.md | Checkpoint save, storage, resume, metadata registration |
| cli | `openspec/specs/cli/` | 11 | dataset-process-module-spec.md (§13) | `docds` CLI commands |
| validation | `openspec/specs/validation/` | 14 | dataset-data-spec.md (§17), dataset-process-module-spec.md (§13) | Source/canonical/view/reward validation |
| lineage | `openspec/specs/lineage/` | 7 | dataset-data-spec.md (§15) | Record traceability from view back to source |
| mineru-data-format | `openspec/specs/mineru-data-format/` | 14 | mineru-data-format.md | MinerU2.5 I/O contract, OTSL grammar, training records |

## Ambiguities and Open Questions

- **Inferred behavior:** Several requirements in `training-bootstrap` were inferred from SPEC.md success criteria and deliverables sections rather than explicit "shall" statements. These are marked implicitly by being derived from Sections 10-11 (acceptance tests).
- **Future reward adapters (TEDS, CDM, IoU, graph, seal, unit tests):** Described in dataset-process-module-spec.md but not yet implemented. Not included as requirements in the reward-system spec. Documented as future extension points only.
- **Diagram and seal serializers:** Mentioned in dataset-process-module-spec.md but not implemented. Target-serialization spec covers only layout, table, formula, and text.
- **Exact parquet schema for SFT and GRPO views:** Listed as an open decision in SPEC.md Section 15. Current specs describe required columns but do not nail down the exact byte-level schema.
- **W&B project structure:** Whether SFT and GRPO use separate or shared WandB projects remains an open decision.
- **Checkpoint lineage:** The full artifact-based lineage system is captured as a proposed change (`openspec/changes/checkpoint-lineage-tracking/`) rather than a current capability spec.

## Preserved / Archived Documents

| Document | Location | Status |
|---|---|---|
| `README.md` | Root | Preserved in place. Operational instructions remain. |
| `SPEC.md` | `docs/archive/SPEC.md` | Archived. Content lives in `openspec/specs/training-bootstrap/`. |
| `docs/dataset-data-spec.md` | `docs/archive/dataset-data-spec.md` | Archived. Content lives in `openspec/specs/dataset-architecture/` and `openspec/specs/view-construction/`. |
| `docs/dataset-process-module-spec.md` | `docs/archive/dataset-process-module-spec.md` | Archived. Content lives in `openspec/specs/data-ingestion/`, `reward-system/`, `target-serialization/`, `cli/`, `validation/`. |
| `docs/mineru-data-format.md` | `docs/archive/mineru-data-format.md` | Archived. Content lives in `openspec/specs/mineru-data-format/`. |
| `docs/superpowers/specs/2026-05-07-checkpoint-lineage-design.md` | `docs/archive/checkpoint-lineage-design.md` | Archived. Content lives in `openspec/changes/checkpoint-lineage-tracking/`. |
| `docs/codebase-audit-report.md` | `docs/archive/codebase-audit-report.md` | Archived. Historical snapshot from 2026-05-27. |
| `docs/legacy/completed-items.md` | `docs/archive/completed-items.md` | Archived. Historical log from 2026-04-21. |

## Validation Results

```
$ openspec validate --all --strict

✓ change/checkpoint-lineage-tracking
✓ spec/checkpoint-tracking
✓ spec/cli
✓ spec/data-ingestion
✓ spec/dataset-architecture
✓ spec/lineage
✓ spec/mineru-data-format
✓ spec/reward-system
✓ spec/target-serialization
✓ spec/training-bootstrap
✓ spec/validation
✓ spec/view-construction
Totals: 12 passed, 0 failed (12 items)
```

## Review of Inferred Requirements

A systematic review was conducted comparing the three specs with the most inferred content (training-bootstrap, dataset-architecture, view-construction) against their original source documents. Key findings and corrections applied:

### Corrections Applied

| Spec | Issue | Fix |
|------|-------|-----|
| training-bootstrap | Checkpoint validity upgraded from source's "should" to SHALL | Restored SHOULD for individual metadata fields while keeping SHALL for the overall requirement |
| training-bootstrap | Implementation details (adapter class names, entry points, torchrun, config paths) presented as spec-level requirements | Replaced with generic descriptions; removed `OcrMultiTurnSFTDataset`, `OcrRLHFDataset`, `sft_trainer`, `main_ppo`, `configs/train/verl/base/` |
| dataset-architecture | Entity ID formats presented as mandatory | Changed to "SHOULD follow the recommended convention" |
| view-construction | SFT `messages` column presented as mandatory | Changed to SHOULD with migration note citing source's "Recommended additional" |
| view-construction | "SFT views SHALL NOT include reward columns" prohibition added | Removed - not stated in source |
| view-construction | SFT/RLVR directory separation presented as mandatory | Changed to SHOULD with migration note |
| view-construction | Invented error scenarios (missing template FileNotFoundError, missing image source error, empty include list warning) | Removed from spec |
| view-construction | Prescriptive stats.json format | Softened to "SHOULD contain per-split row counts" with migration note |
| view-construction | "Exactly one image column" constraint | Softened with migration note noting source does not explicitly mandate this |

### Remaining Known Limitations

- **training-bootstrap**: Non-functional requirements (Debuggability, Simplicity Over Completeness) from source Section 9 not captured as explicit requirements. Acceptance tests from source Section 11 not captured.
- **dataset-architecture**: ID format conventions remain as SHOULD but are currently implemented exactly as described. Practical defaults from source Section 20 not captured as requirements.
- **view-construction**: VERL integration data contract, reward payload store details, and several optional columns (`canonical_image_asset_id`, `view_image_asset_id`, `loss_mask_policy`, `difficulty`) not captured.

## Recommended Next Steps

1. **Review inferred requirements:** Several requirements were inferred from existing documentation rather than explicitly stated. A domain expert should review `training-bootstrap`, `dataset-architecture`, and `view-construction` specs for accuracy.
2. **Decide on original document retention:** The original `SPEC.md`, `docs/dataset-data-spec.md`, `docs/dataset-process-module-spec.md`, and `docs/mineru-data-format.md` are preserved in place. Consider archiving them once the OpenSpec specs are validated as accurate.
3. **Implement checkpoint-lineage-tracking change:** The proposal, design, and tasks are ready in `openspec/changes/checkpoint-lineage-tracking/`. This is a well-scoped change with clear deliverables.
4. **Add future reward adapters as changes:** TEDS, CDM, IoU, graph, seal, and unit-test rewards should each become a separate OpenSpec change when implementation is planned.
5. **Resolve open decisions from SPEC.md Section 15:** Exact parquet schemas, W&B project structure, artifact naming conventions, and checkpoint registration timing remain open.
6. **Update CLAUDE.md or AGENTS.md:** Add a pointer to `openspec/` so AI coding tools discover the specs automatically.
