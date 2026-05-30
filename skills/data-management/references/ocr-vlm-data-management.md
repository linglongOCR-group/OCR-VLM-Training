# OCR VLM Data Management Reference

## Default Dataset Root

Use `/home/byhou/datasets/ocr-training` when the user says "the dataset root" in
this project, unless they provide a concrete root such as
`/mnt/sas-server-0/DataMgmt`. The repo data tooling resolves this root from
`OCR_DATA_ROOT`.

Expected high-level layout:

```text
/home/byhou/datasets/ocr-training/
  sources/
    <source_name>/
  canonical/
    entities/
    records/
    assets/
    manifests/
  views/
    <view_name>/
  legacy/
```

## Source Inspection Checklist

Inspect before exporting:

- `sources/<source_name>/README.md`
- Source config/profile under `configs/data_profiles/` if present
- Raw image/PDF location and whether files exist
- Annotation root and sample subdirectory naming
- Annotation format and task coverage
- Existing canonical partitions for the same source
- Existing legacy output only for comparison, never as the new target

For MinerU sources:

- Count available annotation subdirectories, commonly `vlm` and `hybrid_auto`.
- Confirm the selected subdir has `<stem_name>_model.json`.
- Treat `<stem_name>_model.json` as containing layout and region results when present.
- Use `<stem_name>_middle.json` for page size and stable pixel bboxes when available.
- If `<stem_name>_origin.pdf` exists, render pages from that PDF during source export.
- Parse `<stem_name>_model.json` as `list<list<object>>`; the outer list is page order.
- Treat `model.json` bboxes in `[0, 1]` as the MinerU `[0, 1000]` grid values divided by 1000.
- Cut recognition crops from the rendered canonical page image, not from `content_list_v2` image references, unless an explicit source profile declares that fallback.

For UniRec40M sources:

- Use the `unirec` adapter and source configs under `configs/data/sources/unirec40m_*.yaml`.
- The real source root is normally `OCR_DATA_ROOT=/mnt/sas-server-0/DataMgmt`, with sources such as `sources/UniRec40M_english`.
- Each source is nested by subset and uses `<subset>/annotations/records.jsonl` plus source-referenced crop images.
- Export region records only unless layout is explicitly requested.
- Classify each record independently from content/metadata: conservative table hints, standalone display formula wrappers as `formula`, otherwise `text`.
- Preserve cleanup rules for `<|ln|>`, `<|pn|>`, `<|sn|>`, `<<<change_line_token_wrap>>>`, inline `\(...\)`, and display formula wrappers.

## Canonical Output Contract

Write source-specific partitions:

```text
canonical/entities/documents/source=<source_name>/part-*.parquet
canonical/entities/pages/source=<source_name>/part-*.parquet
canonical/entities/regions/source=<source_name>/part-*.parquet
canonical/records/<task>/source=<source_name>/part-*.parquet
canonical/assets/manifests/source=<source_name>/part-*.parquet
canonical/manifests/sources/<source_name>.json
```

Required task record fields:

```text
record_id, task, source_name, document_id, page_id, region_id,
image_asset_id, target, provenance, metadata, schema_version
```

Canonical targets must remain model-independent:

- `layout`: elements with pixel bboxes, labels, rotation, reading order
- `text`: text fields
- `table`: HTML, OTSL, enhanced OTSL, cells where available
- `formula`: LaTeX and display metadata
- `diagram`: graph/serialization fields where available
- `seal`: structured text segments where available

## Safe Export Pattern

1. Run a smoke export to `/tmp/<source>-canonical-smoke` with a small `--max-samples`.
2. Validate the smoke canonical output.
3. Inspect representative parquet rows and asset paths.
4. Run the full export with explicit `--canonical-root`, `--source-config`, and `--tasks`.
5. Validate the full canonical root by source.
6. Read the source manifest and report final counts.

Example:

```bash
OCR_DATA_ROOT=/home/byhou/datasets/ocr-training scripts/data/docds export-source mineru \
  --source-config configs/data/sources/docbank_mineru.yaml \
  --canonical-root /tmp/docbank-canonical-smoke \
  --max-samples 3 \
  --tasks layout,table,formula,text \
  --allow-unreadable-images

OCR_DATA_ROOT=/home/byhou/datasets/ocr-training scripts/data/docds validate-canonical \
  --canonical-root /tmp/docbank-canonical-smoke \
  --source DocBank_500K
```

Fintech one-document smoke shape:

```bash
OCR_DATA_ROOT=/mnt/sas-server-0/DataMgmt scripts/data/docds export-source mineru \
  --source-config /tmp/fintech_fixed_smoke_source.yaml \
  --canonical-root /mnt/sas-server-0/DataMgmt/canonical \
  --tasks layout,text,table \
  --max-samples 1 \
  --num-workers 1 \
  --worker-chunksize 1 \
  --max-in-flight 1 \
  --overwrite-partitions \
  --progress \
  --log-every 1
```

UniRec40M smoke shape:

```bash
OCR_DATA_ROOT=/mnt/sas-server-0/DataMgmt scripts/data/docds export-source unirec \
  --source-config configs/data/sources/unirec40m_education.yaml \
  --canonical-root /tmp/unirec40m_education_smoke \
  --max-samples 20 \
  --tasks text,formula,table \
  --overwrite-partitions \
  --progress \
  --log-every 10
```

## View Management

A view is model/training-stage specific. It owns:

- prompt strings
- serialized labels
- split assignment
- view image policy
- target serializer selection
- reward profile and payload fields for RLVR

For SFT, validate at least: `id`, `stage`, `task`, `image_path`, `prompt`, `label`, `split`, `canonical_record_id`.

For RLVR, additionally validate: `reward_profile_id`, `reward_payload` or sidecar pointer, and `answer_key` where applicable.

Image policy modes:

- `embedded`: embed or copy view assets for self-contained rows.
- `source_reference`: write dataset-relative paths to existing source/canonical assets.
- `nested_reference`: write Verl-style nested `images` objects that point at dataset-relative assets.

For SFT labels, treat literal `<image>` and `<video>` in assistant content as
reserved media tokens. If training crashes in VERL with a media-token parsing
failure, inspect the exact view row first; an OCR label may contain the token as
text. Current view building drops those labels and reports
`reserved_media_token_label`; current VERL SFT runtime escapes/restores literal
media tokens in non-user messages.

The current all-UniRec40M MinerU training view is:

```bash
OCR_DATA_ROOT=/mnt/sas-server-0/DataMgmt scripts/data/docds build-view \
  configs/data/views/unirec40m_mineru_train_nested_reference.yaml

OCR_DATA_ROOT=/mnt/sas-server-0/DataMgmt scripts/data/docds validate-view \
  unirec40m_mineru_train_nested_reference --require-images
```

## Validation And Audit Commands

Useful commands:

```bash
OCR_DATA_ROOT=<dataset_root> scripts/data/docds validate-canonical \
  --canonical-root <canonical_root> \
  --source <source_name>

OCR_DATA_ROOT=<dataset_root> scripts/data/docds validate-view <view_name_or_path>

OCR_DATA_ROOT=<dataset_root> scripts/data/docds reward-smoke-test \
  --view <view_root> \
  --limit 1000
```

Deploy a view and referenced assets to training nodes:

```bash
OCR_DATA_ROOT=<dataset_root> scripts/data/deploy_view_to_nodes \
  --view <view_name_or_path> \
  --remote-root <remote_dataset_root> \
  --node <ssh_target> \
  --manifest /tmp/<view>.files \
  --check-assets
```

The deployment manifest is dataset-relative and includes view files plus
referenced assets from `images_path` or nested `images`. Use `--transfer-mode tar`
for initial seeding when streaming is preferable to rsync file-list handling.

Count shards:

```bash
find <canonical_root> -path '*source=<source_name>/part-*.parquet' -type f | wc -l
```

Inspect manifest:

```bash
cat <canonical_root>/manifests/sources/<source_name>.json
```

## Failure Modes

- Missing raw page images: record asset paths and report the caveat if export runs with `--allow-unreadable-images`.
- Fintech/MinerU label-image mismatch: check whether page renders came from `_origin.pdf` and whether recognition crops were cut from those canonical page renders.
- Wrong MinerU subdir: low scanned/export counts often mean `mineru_subdir` points to the wrong annotation variant.
- Non-streaming exporter: patch before running full 500K-scale jobs.
- Partial output: do not validate or summarize until the export command exits and the source manifest exists.
- Mixed old/new layouts: keep `legacy/` outputs separate from current spec-aligned `canonical/`.
- UniRec source root drift: if export paths point under `/tmp` or the repo instead of the real data tree, check `OCR_DATA_ROOT` and the source config before editing adapter logic.
- Reserved media-token labels: literal `<image>` or `<video>` labels can masquerade as missing media at runtime; identify the exact row and preserve the builder/runtime guard.
- View deployment misses assets: build the deploy manifest from the dataset root and rerun with `--check-assets` before copying to nodes.
