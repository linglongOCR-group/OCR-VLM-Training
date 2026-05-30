---
name: data-management
description: End-to-end dataset management for OCR/document VLM training repositories, including source discovery, dataset root inspection, source-to-canonical export, canonical validation, view construction/deployment for SFT/RLVR, reward-payload checks, lineage tracing, shard/count audits, and safe handling of large dataset roots. Use when Codex is asked to add, inspect, convert, validate, profile, deploy, or build datasets/views rather than only edit model training configs.
---

# Data Management

## Core Workflow

Treat dataset management as a lifecycle, not only as source-to-canonical conversion:

```text
Source -> Canonical -> View -> Runtime/Training -> Audit/Lineage
```

Start by locating the dataset root, source metadata, source annotations, existing canonical partitions, and view configs. Prefer inspecting real files over assuming layouts.

## Operating Rules

- Preserve raw source data. Do not rewrite or normalize files under `sources/` unless the user explicitly asks.
- Write canonical data under `canonical/` and model-ready materializations under `views/`.
- Keep canonical records model-independent. Put prompts, serialized labels, reward profiles, split assignment, and model-specific image transforms in views.
- For large datasets, use streaming/sharded writers. Do not accumulate full source, region, asset, or task-record tables in memory.
- Validate after every materialization step. A successful export command is not enough.
- Track lineage from view record to canonical task record, entity rows, image assets, and source annotations.
- Ask before overwriting existing source/canonical/view partitions if the requested operation is ambiguous.

## Task Map

- **Discover or onboard a source**: inspect `sources/<source_name>/README.md`, source metadata, raw assets, annotation format, and any exporter config. Confirm adapter, source image root, annotation root, supported tasks, and expected scale.
- **Export source to canonical**: choose the source adapter, select tasks, run a smoke export to `/tmp` for unfamiliar layouts, then run the full export with sharding and validate canonical output.
- **Validate canonical**: check required columns, task/source partitions, unique IDs, entity references, asset references, coordinate spaces, manifests, and source/task counts.
- **Build views**: inspect or create `view.yaml`, apply include/exclude filters by task/source, choose serializers, build SFT or RLVR parquet, then validate prompt/label/image/reward-payload fields.
- **Prepare RLVR**: ensure each RLVR row has `reward_profile_id` and either inline or sidecar payload. For initial bring-up, normalized Levenshtein payloads may mirror the serialized label.
- **Audit or debug data**: compare source counts, canonical counts, view split counts, shard sizes, missing images, skipped samples, and lineage traces.
- **Deploy views**: generate a dataset-relative transfer manifest, include view files plus referenced assets, optionally stat referenced images, then transfer with the repo deployment script.

## Repository-Specific Guidance

When working in `ocr-vlm-tuning`, prefer the root executable `scripts/data/docds`
and dataset-root resolution through `OCR_DATA_ROOT`.

Common commands:

```bash
OCR_DATA_ROOT=/home/byhou/datasets/ocr-training scripts/data/docds export-source mineru \
  --source-config configs/data/sources/docbank_mineru.yaml \
  --canonical-root /home/byhou/datasets/ocr-training/canonical \
  --tasks layout,table,formula,text

OCR_DATA_ROOT=/home/byhou/datasets/ocr-training scripts/data/docds validate-canonical \
  --canonical-root /home/byhou/datasets/ocr-training/canonical \
  --source DocBank_500K

OCR_DATA_ROOT=/home/byhou/datasets/ocr-training scripts/data/docds build-view \
  configs/data/views/<view_name>.yaml

OCR_DATA_ROOT=/home/byhou/datasets/ocr-training scripts/data/deploy_view_to_nodes \
  --view <view_name> \
  --remote-root /path/on/node/datasets/ocr-training \
  --node <host> \
  --check-assets
```

For MinerU outputs, inspect whether each sample uses `vlm`, `hybrid_auto`, or another annotation subdirectory. In this repo, `<stem_name>_model.json` may include both layout and region results; use it as the primary annotation source unless source metadata says otherwise.

For MinerU document bundles with `<stem_name>_origin.pdf`, source export must
render each PDF page into canonical `page_render` assets before building task
records. Parse `<stem_name>_model.json` as `list<list<object>>`; the outer list
is page order. Bbox values in `model.json` are normalized `[0, 1]` values from
the MinerU `[0, 1000]` grid divided by 1000, so scale them by the rendered page
width/height. Layout records reference the page render; recognition records
reference deterministic `region_crop` assets cut from that page render. Do not
use `content_list_v2` crop-image references as the canonical crop source unless
the profile explicitly declares that fallback.

For UniRec40M sources, the canonical adapter is `unirec` and source configs live
under `configs/data/sources/unirec40m_*.yaml`. The raw data is nested by subset
under `sources/UniRec40M_*/<subset>/annotations/records.jsonl` with crop images
referenced from each JSONL row; use `OCR_DATA_ROOT=/mnt/sas-server-0/DataMgmt`
when working against the real source tree. Export region records only unless the
user explicitly asks for layout. Classify each row from content and metadata
instead of trusting a source-level category: table hints are conservative,
standalone display formulas become `formula`, and all other non-empty labels are
`text`. Preserve the UniRec cleanup contract for `<|ln|>`, `<|pn|>`, `<|sn|>`,
`<<<change_line_token_wrap>>>`, inline `\(...\)`, and display formula wrappers.

For SFT views, literal `<image>` or `<video>` in assistant labels are reserved
media tokens, not safe text labels. Before blaming runtime media ingestion,
inspect the exact view row. Current view building filters such labels with the
`reserved_media_token_label` statistic, and the VERL runtime escapes literal
media tokens in non-user messages before parsing.

Use `image_policy.materialization.mode: nested_reference` when the training
runtime should receive Verl-style nested `images` references without copying
assets into the view. Use `source_reference` when view parquet should carry
dataset-relative source asset paths. In both cases, validate with
`validate-view --require-images` when image reachability matters.

## References

Read `references/ocr-vlm-data-management.md` when the task involves concrete OCR VLM dataset operations, expected layouts, validation commands, source/canonical/view schemas, UniRec40M exports, reserved media-token failures, or view deployment.
