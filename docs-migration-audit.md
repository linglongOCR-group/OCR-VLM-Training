# Documentation Migration Audit

**Date:** 2026-05-29
**Branch:** `data-module`

---

## Source Document Inventory

| Source file | Current role | Target location(s) | Action | Notes |
|---|---|---|---|---|
| `README.md` | Mixed: project overview + operational | Keep in place. Extract behavior claims into OpenSpec specs. | reference | Contains operational instructions (CLI, training, checkpoints) that overlap with SPEC.md and dataset specs |
| `SPEC.md` | Capability specification with formal requirements | `openspec/specs/training-bootstrap/spec.md` | rewrite | 647 lines. Sections 7-11 contain testable requirements with success criteria. Sections 1-6, 12-16 are context/scope/risks |
| `docs/dataset-data-spec.md` | Dataset architecture + entity schemas + view schemas | `openspec/specs/dataset-architecture/spec.md` + `openspec/specs/view-construction/spec.md` | split | 1882 lines. Sections 1-10 are architecture. Sections 11-15 are prompt/label/selection design. Sections 16-21 are metadata/defaults |
| `docs/dataset-process-module-spec.md` | Processing module + CLI + reward + serializer design | `openspec/specs/data-ingestion/spec.md` + `openspec/specs/reward-system/spec.md` + `openspec/specs/target-serialization/spec.md` + `openspec/specs/cli/spec.md` | split | 1718 lines. Sections 1-6 are module design. Section 7 is serializer design. Section 8 is reward design. Sections 9-13 are view schemas and CLI |
| `docs/mineru-data-format.md` | MinerU2.5 training data format contract | `openspec/specs/mineru-data-format/spec.md` | rewrite | 688 lines. Model I/O contract, chat format, task-specific output grammars, validation rules |
| `docs/superpowers/specs/2026-05-07-checkpoint-lineage-design.md` | Proposed design: checkpoint tracking via WandB | `openspec/changes/checkpoint-lineage-tracking/` | convert | 497 lines. Full design spec for a not-yet-implemented feature |
| `docs/codebase-audit-report.md` | Technical debt audit with fix recommendations | `docs/archive/codebase-audit-report.md` | preserve | Historical snapshot of codebase state on 2026-05-27 |
| `docs/legacy/completed-items.md` | Historical completion log | `docs/archive/completed-items.md` | preserve | Date: 2026-04-21. Records what was built in the initial phase |

## Capability Boundaries

| Capability | Source documents | Primary content | Spec priority |
|---|---|---|---|
| `dataset-architecture` | dataset-data-spec.md (§1-10) | Three-layer Source→Canonical→View, entity schemas, image assets, coordinates | High |
| `data-ingestion` | dataset-process-module-spec.md (§4-6) | Source adapters, canonical writer, asset manager, package layout | High |
| `view-construction` | dataset-data-spec.md (§10-15), dataset-process-module-spec.md (§9-10) | View builder, view schema, selection, splits, prompt/label generation | High |
| `target-serialization` | dataset-process-module-spec.md (§6) | Serializer adapters (layout, table, formula, text, diagram, seal) | Medium |
| `reward-system` | dataset-process-module-spec.md (§7-8) | Reward adapters, Levenshtein, future TEDS/CDM/IoU/graph/unit-test rewards | Medium |
| `training-bootstrap` | SPEC.md (§1-16) | Multi-node SFT/GRPO, W&B, checkpoints, resume, configuration | High |
| `checkpoint-tracking` | checkpoint-lineage-design.md | WandB artifact lineage, path resolution, multi-node portability | Proposed change |
| `cli` | dataset-process-module-spec.md (§13) | `docds` CLI commands for source, canonical, view, reward, lineage | Medium |
| `validation` | dataset-data-spec.md (§17), dataset-process-module-spec.md (§13) | Source/canonical/view validation checks | Medium |
| `lineage` | dataset-data-spec.md (§15) | Record traceability from view back to source | Low |
| `mineru-data-format` | mineru-data-format.md | MinerU2.5 I/O contract, OTSL grammar, training record format | High |

## Classification Summary

- **Current capability specifications** (rewrite into OpenSpec): SPEC.md, dataset-data-spec.md, dataset-process-module-spec.md, mineru-data-format.md
- **Proposed future changes** (convert to openspec/changes/): checkpoint-lineage-design.md
- **Technical design**: Mixed into capability specs and proposed change
- **Operational documentation**: README.md (CLI, training, checkpoint instructions)
- **Historical/obsolete**: codebase-audit-report.md, completed-items.md
