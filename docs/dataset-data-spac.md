# Dataset Format Specification

Version: `v0.1`
Scope: Dataset architecture, canonical data format, image asset management, lineage tracking, and view generation for document parsing training.

---

## 1. Design Goal

This specification defines a practical dataset architecture for document parsing model training, especially for tasks such as:

* Layout detection
* Text recognition
* Table recognition
* Formula recognition
* Diagram recognition
* Seal recognition
* Other future document component extraction tasks

The design follows a three-layer structure:

```text
Source → Canonical → View
```

The core principle is:

> **Source is for preservation. Canonical is for normalized reusable truth. View is for model-specific training data.**

This design supports heterogeneous raw datasets, task-specific canonical records, deterministic image derivatives, model-specific prompt/label generation, and full lineage from training samples back to the original files.

This is consistent with modern document parsing pipelines where layout detection first identifies regions and semantic categories, and then category-specific content extraction is performed on cropped regions with specialized prompts .

---

# 2. Layer Overview

## 2.1 Source Layer

The Source layer stores raw datasets in their original form.

It owns:

* Original PDFs
* Original images
* Original annotations
* Source README
* Source metadata
* Export configuration
* Checksums of raw assets

The Source layer should be treated as mostly immutable.

It does **not** enforce a unified annotation format.

---

## 2.2 Canonical Layer

The Canonical layer stores normalized, model-independent dataset records.

It owns:

* Shared document/page/region entities
* Task-specific canonical records
* Deterministic reusable image derivatives
* Canonical asset manifests
* Dataset/task/source manifests
* Schema registry
* Lineage metadata

The Canonical layer should not store model-specific prompt strings or training labels such as MinerU-specific box strings. Those belong to the View layer.

---

## 2.3 View Layer

The View layer stores model-specific training-ready datasets.

It owns:

* Prompt-rendered training records
* Model-specific output labels
* Task/source/category sampling policies
* Train/validation/test splits
* Optional model-specific image derivatives
* Final Parquet files consumed by VERL or other training frameworks

The View layer is allowed to be redundant because it is a materialized training dataset.

---

# 3. Recommended Directory Structure

```text
datasets/
  sources/
    <source_name>/
      README.md
      source.yaml
      export.yaml
      raw/
        pdfs/
        images/
      annotations/
      checksums.jsonl

  canonical/
    registry/
      tasks.yaml
      element_types.yaml
      image_profiles.yaml
      schemas/
        document.schema.json
        page.schema.json
        region.schema.json
        layout.schema.json
        text.schema.json
        table.schema.json
        formula.schema.json
        diagram.schema.json
        seal.schema.json

    entities/
      documents/
        source=<source_name>/
          part-000.parquet
      pages/
        source=<source_name>/
          part-000.parquet
      regions/
        source=<source_name>/
          part-000.parquet

    records/
      layout/
        source=<source_name>/
          part-000.parquet
      text/
        source=<source_name>/
          part-000.parquet
      table/
        source=<source_name>/
          part-000.parquet
      formula/
        source=<source_name>/
          part-000.parquet
      diagram/
        source=<source_name>/
          part-000.parquet
      seal/
        source=<source_name>/
          part-000.parquet

    assets/
      pages/
        source=<source_name>/
          document_id=<document_id>/
            page_0000.png
      regions/
        source=<source_name>/
          task=<task_name>/
            document_id=<document_id>/
              page_0000_region_0001.png
      manifests/
        asset_manifest.parquet

    manifests/
      canonical.yaml
      sources/
        <source_name>.yaml
      tasks/
        layout.yaml
        text.yaml
        table.yaml
        formula.yaml
        diagram.yaml
        seal.yaml

    indexes/
      table/
        source=<source_name>/
          category=merged_cell_table.parquet

  views/
    <view_name>/
      README.md
      view.yaml
      train.parquet
      val.parquet
      test.parquet
      stats.json
      assets/
        layout/
        table/
        formula/
        text/
        diagram/
        seal/
      manifests/
        asset_manifest.parquet
```

---

# 4. Source Layer Specification

## 4.1 Purpose

The Source layer preserves the raw dataset and describes how it can be converted into canonical form.

Different datasets may have different annotation formats:

* COCO
* YOLO
* MinerU output
* OCR JSON
* HTML table annotations
* LaTeX formula annotations
* Custom XML/JSON
* Synthetic-generation metadata

The Source layer should not be forced into a uniform structure beyond a minimal contract.

---

## 4.2 Required Files

Each source dataset should contain:

```text
sources/<source_name>/
  README.md
  source.yaml
  export.yaml
  raw/
  annotations/
  checksums.jsonl
```

---

## 4.3 `README.md`

Human-facing documentation.

Recommended contents:

```text
# Dataset Name

## Description

## Original Source

## License

## Directory Layout

## Annotation Format

## Supported Tasks

## Known Issues

## Processing Owner

## Processing Date

## Export Notes
```

---

## 4.4 `source.yaml`

Machine-readable source metadata.

Example:

```yaml
name: DocBank
version: v1
description: Document layout dataset converted from original DocBank annotations.

owner: document-ai-team
created_at: "2026-04-24"
license: unknown
homepage: null

raw_format:
  document_type: image
  annotation_format: custom
  coordinate_system: pixel_xyxy
  origin: top_left

granularity:
  source_record_level: page
  contains_regions: true
  contains_recognition_targets: false

supported_tasks:
  - layout

element_types:
  - title
  - text
  - table
  - figure
  - formula

quality:
  status: raw
  known_issues:
    - "Some bounding boxes may overlap."
```

---

## 4.5 `export.yaml`

Defines how to export the source into canonical format.

Example:

```yaml
adapter: docbank_adapter
adapter_version: "1.0.0"

canonical_image_policy:
  page_render:
    enabled: true
    dpi: 200
    format: png
    color_mode: rgb
    background: white
    apply_pdf_rotation: true

  region_crop:
    enabled: true
    tasks:
      - text
      - table
      - formula
      - diagram
      - seal
    crop_from: canonical_page_image
    margin:
      type: relative
      value: 0.03
      max_pixels: 32
    clip_to_page: true
    format: png

  coordinate_policy:
    canonical_bbox_space: page_pixel_xyxy
    preserve_original_bbox: true
    save_transform: true

export_tasks:
  - layout
  - table
  - formula

split_policy:
  level: document
  strategy: predefined
  split_file: annotations/splits.json
```

---

# 5. Canonical Layer Specification

## 5.1 Design Principle

The Canonical layer should store **semantic, model-independent truth**.

It should not store training prompt formats directly.

For example, canonical layout records should store:

```json
{
  "bbox": [100, 200, 500, 700],
  "label": "table",
  "rotation": 0
}
```

not:

```text
<|box_start|>100 200 500 700<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>
```

The latter is a View-layer serialization.

This distinction is important because different models may use different target formats. For example, DocV3RExpert-1.2B uses MinerU-style OTSL for tables, while DocV3RExpert-2B uses HTML .

---

## 5.2 Canonical Layer Components

The Canonical layer contains four major parts:

```text
canonical/entities/
canonical/records/
canonical/assets/
canonical/manifests/
```

Their responsibilities are:

| Component    | Purpose                                   |
| ------------ | ----------------------------------------- |
| `entities/`  | Shared document/page/region indexes       |
| `records/`   | Task-specific canonical records           |
| `assets/`    | Deterministic reusable image derivatives  |
| `manifests/` | Source/task/canonical metadata            |
| `registry/`  | Shared schemas and task definitions       |
| `indexes/`   | Optional lightweight materialized indexes |

---

# 6. Canonical Entity Tables

Canonical entities define stable IDs and shared lineage.

There are three core entity tables:

```text
documents
pages
regions
```

They are physically partitioned by source:

```text
canonical/entities/documents/source=DocBank/
canonical/entities/pages/source=DocBank/
canonical/entities/regions/source=DocBank/
```

---

## 6.1 `documents.parquet`

One row per original document.

Required columns:

| Column               | Type        | Description                          |
| -------------------- | ----------- | ------------------------------------ |
| `document_id`        | string      | Stable canonical document ID         |
| `source_name`        | string      | Source dataset name                  |
| `source_document_id` | string      | Original document ID or filename     |
| `document_path`      | string      | Path to raw PDF/image                |
| `document_type`      | string      | `pdf`, `image`, etc.                 |
| `num_pages`          | int         | Number of pages                      |
| `language`           | string      | Optional language label              |
| `domain`             | string      | Optional domain, e.g. finance, legal |
| `checksum`           | string      | Raw document checksum                |
| `metadata`           | json/string | Additional source-specific metadata  |
| `schema_version`     | string      | Entity schema version                |

Example:

```json
{
  "document_id": "doc:DocBank:8f31c9",
  "source_name": "DocBank",
  "source_document_id": "doc_000001.pdf",
  "document_path": "sources/DocBank/raw/pdfs/doc_000001.pdf",
  "document_type": "pdf",
  "num_pages": 5,
  "language": "en",
  "domain": "academic",
  "checksum": "sha256:...",
  "metadata": {},
  "schema_version": "1.0.0"
}
```

---

## 6.2 `pages.parquet`

One row per document page.

Required columns:

| Column                | Type        | Description                      |
| --------------------- | ----------- | -------------------------------- |
| `page_id`             | string      | Stable canonical page ID         |
| `document_id`         | string      | Parent document ID               |
| `source_name`         | string      | Source dataset name              |
| `source_page_id`      | string      | Original source page ID          |
| `page_index`          | int         | Zero-based page index            |
| `page_image_asset_id` | string      | Canonical rendered page asset ID |
| `width`               | int         | Canonical page image width       |
| `height`              | int         | Canonical page image height      |
| `rotation`            | float       | Page-level rotation              |
| `attributes`          | json/string | Page-level attributes            |
| `schema_version`      | string      | Entity schema version            |

---

## 6.3 `regions.parquet`

One row per annotated or generated region.

Required columns:

| Column                 | Type        | Description                                      |
| ---------------------- | ----------- | ------------------------------------------------ |
| `region_id`            | string      | Stable region ID                                 |
| `page_id`              | string      | Parent page ID                                   |
| `document_id`          | string      | Parent document ID                               |
| `source_name`          | string      | Source dataset name                              |
| `source_annotation_id` | string      | Original annotation ID                           |
| `bbox`                 | list/array  | `[x1, y1, x2, y2]` in canonical page coordinates |
| `rotation`             | float       | Region rotation angle                            |
| `element_type`         | string      | `text`, `table`, `formula`, etc.                 |
| `category`             | string      | Optional fine-grained category                   |
| `subcategory`          | string      | Optional finer-grained category                  |
| `reading_order`        | int         | Optional reading order                           |
| `crop_asset_id`        | string      | Canonical crop asset ID                          |
| `quality_flags`        | list/string | Quality annotations                              |
| `metadata`             | json/string | Additional metadata                              |
| `schema_version`       | string      | Entity schema version                            |

---

# 7. Canonical Task Records

## 7.1 Partitioning Rule

Canonical task records should be physically partitioned by:

```text
task/source
```

Example:

```text
canonical/records/layout/source=DocBank/
canonical/records/table/source=DocBank/
canonical/records/formula/source=DocBank/
```

This provides intuitive dataset management.

For example, to use DocBank tables and formulas but not DocBank layout records, the view config can simply include:

```yaml
include:
  - task: table
    sources: [DocBank]
  - task: formula
    sources: [DocBank]
```

and omit:

```yaml
task: layout
source: DocBank
```

No unintuitive global filter is required.

---

## 7.2 Common Task Record Fields

All canonical task records should contain the following fields.

| Column           | Type        | Required | Description                               |
| ---------------- | ----------- | -------: | ----------------------------------------- |
| `record_id`      | string      |      Yes | Stable canonical task record ID           |
| `task`           | string      |      Yes | Task name                                 |
| `source_name`    | string      |      Yes | Source dataset name                       |
| `document_id`    | string      |      Yes | Document ID                               |
| `page_id`        | string      |      Yes | Page ID                                   |
| `region_id`      | string/null | Optional | Region ID for crop-level tasks            |
| `image_asset_id` | string      |      Yes | Canonical image asset used by this record |
| `target`         | json/string |      Yes | Canonical model-independent target        |
| `category`       | string/null | Optional | Fine-grained category                     |
| `subcategory`    | string/null | Optional | Finer-grained category                    |
| `language`       | string/null | Optional | Language                                  |
| `quality_flags`  | list/string | Optional | Quality labels                            |
| `provenance`     | json/string |      Yes | Source lineage                            |
| `metadata`       | json/string | Optional | Additional metadata                       |
| `schema_version` | string      |      Yes | Schema version                            |

---

## 7.3 Layout Record

Stored under:

```text
canonical/records/layout/source=<source_name>/
```

Example:

```json
{
  "record_id": "layout:DocBank:page_000001",
  "task": "layout",
  "source_name": "DocBank",
  "document_id": "doc:DocBank:8f31c9",
  "page_id": "page:DocBank:8f31c9:0000",
  "region_id": null,
  "image_asset_id": "asset:page_render:DocBank:8f31c9:0000",
  "target": {
    "coordinate_space": "canonical_page_pixel_xyxy",
    "elements": [
      {
        "region_id": "region:DocBank:8f31c9:0000:001",
        "bbox": [100, 120, 500, 180],
        "label": "title",
        "rotation": 0,
        "reading_order": 1
      },
      {
        "region_id": "region:DocBank:8f31c9:0000:002",
        "bbox": [80, 220, 900, 700],
        "label": "table",
        "rotation": 0,
        "reading_order": 2
      }
    ]
  },
  "schema_version": "1.0.0"
}
```

---

## 7.4 Text Record

Stored under:

```text
canonical/records/text/source=<source_name>/
```

Example:

```json
{
  "record_id": "text:DocBank:region_000001",
  "task": "text",
  "source_name": "DocBank",
  "document_id": "doc:DocBank:8f31c9",
  "page_id": "page:DocBank:8f31c9:0000",
  "region_id": "region:DocBank:8f31c9:0000:001",
  "image_asset_id": "asset:region_crop:DocBank:8f31c9:0000:001",
  "target": {
    "text": "Annual Financial Report",
    "normalization": "none"
  },
  "language": "en",
  "schema_version": "1.0.0"
}
```

---

## 7.5 Table Record

Stored under:

```text
canonical/records/table/source=<source_name>/
```

Canonical table records may contain multiple serializations.

Example:

```json
{
  "record_id": "table:DocBank:region_000002",
  "task": "table",
  "source_name": "DocBank",
  "document_id": "doc:DocBank:8f31c9",
  "page_id": "page:DocBank:8f31c9:0000",
  "region_id": "region:DocBank:8f31c9:0000:002",
  "image_asset_id": "asset:region_crop:DocBank:8f31c9:0000:002",
  "category": "merged_cell_table",
  "target": {
    "html": "<table>...</table>",
    "otsl": "<fcel>...</fcel>",
    "enhanced_otsl": "<fcel>...<nest_start>...</nest_end>",
    "cells": [
      {
        "row": 0,
        "col": 0,
        "rowspan": 1,
        "colspan": 2,
        "text": "Revenue"
      }
    ]
  },
  "schema_version": "1.0.0"
}
```

Enhanced OTSL is useful for nested tables. The referenced document parsing work extends OTSL using `<nest_start>` and `<nest_end>` to represent nested structures while preserving compatibility with standard OTSL .

---

## 7.6 Formula Record

Stored under:

```text
canonical/records/formula/source=<source_name>/
```

Example:

```json
{
  "record_id": "formula:source:region_000003",
  "task": "formula",
  "source_name": "FormulaSet",
  "document_id": "doc:FormulaSet:abc123",
  "page_id": "page:FormulaSet:abc123:0000",
  "region_id": "region:FormulaSet:abc123:0000:003",
  "image_asset_id": "asset:region_crop:FormulaSet:abc123:0000:003",
  "target": {
    "latex": "E = mc^2",
    "display_type": "inline"
  },
  "schema_version": "1.0.0"
}
```

---

## 7.7 Diagram Record

Stored under:

```text
canonical/records/diagram/source=<source_name>/
```

Canonical diagram records should not rely only on Mermaid strings. They should preserve graph-like semantic structure.

Example:

```json
{
  "record_id": "diagram:SyntheticDiagram:region_000004",
  "task": "diagram",
  "source_name": "SyntheticDiagram",
  "document_id": "doc:SyntheticDiagram:abc123",
  "page_id": "page:SyntheticDiagram:abc123:0000",
  "region_id": "region:SyntheticDiagram:abc123:0000:004",
  "image_asset_id": "asset:region_crop:SyntheticDiagram:abc123:0000:004",
  "target": {
    "nodes": [
      {
        "node_id": "A",
        "text": "Input",
        "bbox": [10, 20, 100, 60]
      },
      {
        "node_id": "B",
        "text": "Output",
        "bbox": [200, 20, 300, 60]
      }
    ],
    "edges": [
      {
        "source": "A",
        "target": "B",
        "type": "directed",
        "label": null
      }
    ],
    "groups": [],
    "blocks": [],
    "serializations": {
      "mermaid": "graph TD\nA[Input] --> B[Output]",
      "d_mermaid": "graph TD<nl>A[Input]<nl>B[Output]<nl>A-->B"
    }
  },
  "schema_version": "1.0.0"
}
```

D-Mermaid or other task-aware diagram formats may be generated in the View layer. This is useful because ordinary Mermaid may not fully express alignment, symmetry, grouping, and other document-specific relational structures .

---

## 7.8 Seal Record

Stored under:

```text
canonical/records/seal/source=<source_name>/
```

Example:

```json
{
  "record_id": "seal:SealSet:region_000005",
  "task": "seal",
  "source_name": "SealSet",
  "document_id": "doc:SealSet:abc123",
  "page_id": "page:SealSet:abc123:0000",
  "region_id": "region:SealSet:abc123:0000:005",
  "image_asset_id": "asset:region_crop:SealSet:abc123:0000:005",
  "target": {
    "horizontal_segments": [
      ["APPROVED"]
    ],
    "circular_segments": [
      ["GLOBAL", "FINANCE", "LIMITED"]
    ],
    "normalization": {
      "case": "upper",
      "remove_decorative_symbols": true,
      "rotation_aware": true
    }
  },
  "schema_version": "1.0.0"
}
```

---

# 8. Image Asset Specification

## 8.1 Core Rule

Image resources are first-class dataset assets.

The ownership rule is:

```text
Source layer:
  Owns raw images and PDFs.

Canonical layer:
  Owns deterministic reusable image derivatives.

View layer:
  Owns optional model-specific training image derivatives.

Runtime:
  Owns tensor conversion, random augmentation, and cheap declared transforms.
```

---

## 8.2 Asset Creation Timing

| Asset Type                   | Created During       |   Persisted | Notes                             |
| ---------------------------- | -------------------- | ----------: | --------------------------------- |
| Raw PDF/image                | Source ingestion     |         Yes | Original immutable asset          |
| Rendered page image          | `source → canonical` |         Yes | Shared by layout and region tasks |
| Canonical region crop        | `source → canonical` | Usually yes | Shared by recognition tasks       |
| Model-specific resized image | `canonical → view`   |    Optional | Controlled by `view.yaml`         |
| Random augmentation          | Runtime              |          No | Should not be persisted           |
| Tensor normalization         | Runtime              |          No | Framework-specific                |

---

## 8.3 Canonical Image Policy

Defined in:

```text
sources/<source_name>/export.yaml
```

or shared in:

```text
canonical/registry/image_profiles.yaml
```

Example:

```yaml
canonical_image_policy:
  page_render:
    enabled: true
    dpi: 200
    format: png
    color_mode: rgb
    background: white
    apply_pdf_rotation: true

  region_crop:
    enabled: true
    tasks:
      - text
      - table
      - formula
      - diagram
      - seal
    crop_from: canonical_page_image
    margin:
      type: relative
      value: 0.03
      max_pixels: 32
    clip_to_page: true
    format: png

  coordinate_policy:
    canonical_bbox_space: page_pixel_xyxy
    preserve_original_bbox: true
    save_transform: true
```

---

## 8.4 View Image Policy

Defined in:

```text
views/<view_name>/view.yaml
```

Example:

```yaml
image_policy:
  image_source:
    layout: canonical_page_image
    table: canonical_region_crop
    formula: canonical_region_crop
    text: canonical_region_crop

  transform:
    layout:
      resize:
        max_side: 2048
        keep_aspect_ratio: true
      pad:
        enabled: false

    table:
      resize:
        max_side: 1024
        keep_aspect_ratio: true
      pad:
        enabled: true
        pad_to_multiple: 28

  materialization:
    mode: embedded

  runtime_transforms:
    enabled: false
```

Supported materialization modes:

| Mode               | Meaning                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------- |
| `embedded`         | Store runtime images in Parquet as flat `images_bytes: list<binary>`                        |
| `source_reference` | Copy or transform selected images into `views/<view>/assets/` and reference filenames only  |

`embedded` is the default for portable VERL views. `image_path` remains a lineage/debug field and must not be required by the training runtime in embedded mode.

For `source_reference`, the view builder still reads from the canonical asset path, but the runtime values written to `images_path` are filenames only. The files themselves are materialized under `views/<view>/assets/`. If an image transform is configured, the transformed PNG is written; otherwise the original image bytes are copied with their original extension.

Large embedded-byte views should use explicit sharding:

```yaml
shard_policy:
  rows_per_shard: 128
```

This writes `train/part-00000.parquet`, `train/part-00001.parquet`, and so on.
For VERL SFT, pass the expanded shard file list to `data.train_files`; passing a
single very large nested binary Parquet can trigger pyarrow/pandas chunked-array
conversion errors.

---

## 8.5 Recommended Image Formats

| Layer                  | Asset Type             | Recommended Format       |
| ---------------------- | ---------------------- | ------------------------ |
| Source                 | Raw images/PDFs        | Original                 |
| Canonical              | Rendered pages         | PNG                      |
| Canonical              | Region crops           | PNG                      |
| View                   | Embedded runtime image bytes | PNG or high-quality WebP |
| View                   | Source-reference training images | PNG or high-quality WebP |

Default recommendation:

```yaml
canonical_image_policy:
  page_render:
    format: png
  region_crop:
    format: png

view_image_policy:
  materialization:
    mode: embedded
```

For OCR, table, formula, and diagram tasks, avoid low-quality JPEG by default.

---

## 8.6 Asset Manifest

Every generated image asset must be recorded in an asset manifest.

Path:

```text
canonical/assets/manifests/asset_manifest.parquet
views/<view_name>/manifests/asset_manifest.parquet
```

Recommended columns:

| Column                | Type        | Description                                             |
| --------------------- | ----------- | ------------------------------------------------------- |
| `asset_id`            | string      | Stable asset ID                                         |
| `asset_type`          | string      | `raw_image`, `page_render`, `region_crop`, `view_image` |
| `source_name`         | string      | Source dataset                                          |
| `document_id`         | string      | Parent document                                         |
| `page_id`             | string      | Parent page                                             |
| `region_id`           | string/null | Parent region if crop                                   |
| `task`                | string/null | Related task                                            |
| `path`                | string      | Relative asset path                                     |
| `width`               | int         | Image width                                             |
| `height`              | int         | Image height                                            |
| `format`              | string      | `png`, `jpg`, `webp`                                    |
| `checksum`            | string      | File checksum                                           |
| `parent_asset_id`     | string/null | Parent asset                                            |
| `transform_spec_hash` | string      | Hash of transform config                                |
| `coordinate_space`    | string      | Coordinate system                                       |
| `transform`           | json/string | Transform metadata                                      |

Example crop transform:

```json
{
  "operation": "crop",
  "from": "page_pixel_xyxy",
  "bbox": [100, 200, 500, 700],
  "margin_pixels": [16, 16, 16, 16],
  "clip_to_page": true,
  "output_size": [432, 532]
}
```

Example resize-and-pad transform:

```json
{
  "operation": "resize_pad",
  "input_size": [432, 532],
  "output_size": [1024, 1024],
  "scale": 1.9248,
  "pad_left": 96,
  "pad_top": 0,
  "coordinate_mapping": {
    "x_view": "x_canonical * scale + pad_left",
    "y_view": "y_canonical * scale + pad_top"
  }
}
```

---

# 9. Coordinate System Specification

## 9.1 Canonical Coordinate Rule

Canonical annotations must use canonical page coordinates:

```text
canonical_page_pixel_xyxy
```

Example:

```json
{
  "bbox": [100, 200, 500, 700],
  "coordinate_space": "canonical_page_pixel_xyxy"
}
```

---

## 9.2 View Coordinate Rule

View labels must match the actual image fed to the model.

If the view image is resized or padded, the corresponding target coordinates must be transformed.

Example:

Canonical bbox:

```text
[100, 200, 500, 700]
```

If the image is resized by `0.5`, the View-layer label should use:

```text
[50, 100, 250, 350]
```

For MinerU-style layout training:

```text
<|box_start|>50 100 250 350<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>
```

The transform must be recorded in the View asset manifest.

---

# 10. View Layer Specification

## 10.1 Purpose

The View layer materializes model-specific training samples.

It performs:

* Source/task/category selection
* Split construction
* Sampling and mixture balancing
* Prompt rendering
* Target serialization
* Optional model-specific image transformation
* Final Parquet export

---

## 10.2 View Directory

```text
views/<view_name>/
  README.md
  view.yaml
  train.parquet
  val.parquet
  test.parquet
  train/
    part-00000.parquet
    part-00001.parquet
  stats.json
  assets/
  manifests/
    asset_manifest.parquet
```

For small views, each split may be a single `<split>.parquet` file. For large
embedded-image views, prefer sharded split directories (`train/part-*.parquet`,
`val/part-*.parquet`, `test/part-*.parquet`) so VERL/pandas reads one bounded
file at a time.

---

## 10.3 `view.yaml`

Example:

```yaml
name: mineru25_sft_layout_table_formula_v1
created_at: "2026-04-24"

model_family: mineru2.5
training_stage: sft

include:
  - task: layout
    sources:
      - DocLayNet
      - M6Doc

  - task: table
    sources:
      - DocBank
      - PubTabNet
    where:
      category:
        in:
          - standard_table
          - merged_cell_table
          - borderless_table
      quality_flags:
        not_contains:
          - needs_review

  - task: formula
    sources:
      - DocBank
      - UniMER

exclude:
  - task: layout
    sources:
      - DocBank

task_mixture:
  layout: 0.4
  table: 0.3
  formula: 0.2
  text: 0.1

split_policy:
  level: document
  train_ratio: 0.98
  val_ratio: 0.01
  test_ratio: 0.01
  seed: 42

prompt_templates:
  layout: templates/mineru25/layout.jinja
  table: templates/mineru25/table.jinja
  formula: templates/mineru25/formula.jinja
  text: templates/mineru25/text.jinja

target_serialization:
  layout: mineru_box_string
  table: enhanced_otsl
  formula: latex
  text: plain_text

image_policy:
  image_source:
    layout: canonical_page_image
    table: canonical_region_crop
    formula: canonical_region_crop
    text: canonical_region_crop

  materialization:
    mode: embedded

lineage:
  canonical_snapshot: "canonical-20260424"
  code_version: "<git_sha>"
  config_hash: "<hash>"
```

---

## 10.4 View Parquet Schema

Recommended columns:

| Column                     | Type        | Required | Description                         |
| -------------------------- | ----------- | -------: | ----------------------------------- |
| `id`                       | string      |      Yes | Unique view record ID               |
| `task`                     | string      |      Yes | Task name                           |
| `image_path`               | string      |      Yes | Lineage/debug image path            |
| `images_bytes`             | list<binary>| Optional | Embedded runtime image bytes        |
| `images_path`              | list<string>| Optional | Source-reference asset filenames    |
| `messages`                 | list/null   | Optional | VERL SFT conversation               |
| `prompt`                   | string      |      Yes | Model input prompt                  |
| `label`                    | string      |      Yes | Target output string                |
| `source_name`              | string      |      Yes | Source dataset                      |
| `document_id`              | string      |      Yes | Canonical document ID               |
| `page_id`                  | string      |      Yes | Canonical page ID                   |
| `region_id`                | string/null | Optional | Canonical region ID                 |
| `canonical_record_id`      | string      |      Yes | Link to canonical task record       |
| `canonical_image_asset_id` | string      |      Yes | Link to canonical image asset       |
| `view_image_asset_id`      | string/null | Optional | Link to view-specific image asset   |
| `target_format`            | string      |      Yes | Serialized target format            |
| `prompt_template_id`       | string      |      Yes | Prompt template identifier          |
| `split`                    | string      |      Yes | `train`, `val`, or `test`           |
| `metadata`                 | json/string | Optional | Additional metadata                 |

Example:

```json
{
  "id": "view:mineru25_sft_v1:000001",
  "task": "table",
  "image_path": "canonical/assets/regions/source=DocBank/doc/page_region.png",
  "images_bytes": ["<binary PNG or WebP bytes>"],
  "images_path": null,
  "prompt": "<image>\nTable Recognition:",
  "label": "<fcel>Revenue<fcel>Amount<nl><fcel>2025<fcel>100",
  "source_name": "DocBank",
  "document_id": "doc:DocBank:8f31c9",
  "page_id": "page:DocBank:8f31c9:0000",
  "region_id": "region:DocBank:8f31c9:0000:002",
  "canonical_record_id": "table:DocBank:region_000002",
  "canonical_image_asset_id": "asset:region_crop:DocBank:8f31c9:0000:002",
  "view_image_asset_id": "asset:view_image:mineru25_sft_v1:000001",
  "target_format": "enhanced_otsl_v1",
  "prompt_template_id": "mineru25_table_v1",
  "split": "train",
  "metadata": {}
}
```

---

# 11. Prompt and Label Generation

## 11.1 Prompt Templates

Prompts should be rendered in the View layer.

Example MinerU-style prompts:

```text
<image>
Layout Detection:
```

```text
<image>
Table Recognition:
```

```text
<image>
Formula Recognition:
```

---

## 11.2 Target Serialization

Canonical targets may be serialized into different model-specific formats.

Examples:

| Task    | Canonical Target             | View Serialization                     |
| ------- | ---------------------------- | -------------------------------------- |
| Layout  | bbox + label + rotation JSON | MinerU box string                      |
| Table   | HTML / cells / OTSL          | OTSL or enhanced OTSL                  |
| Formula | LaTeX JSON field             | LaTeX string                           |
| Text    | Text field                   | Plain text                             |
| Diagram | Graph structure              | Mermaid / D-Mermaid / natural language |
| Seal    | Segment structure            | Rotation-aware normalized text         |

---

# 12. Dataset Selection and Filtering

## 12.1 Primary Selection

Primary selection should be based on:

```text
task + source
```

Example:

```yaml
include:
  - task: table
    sources:
      - DocBank

  - task: formula
    sources:
      - DocBank

  - task: layout
    sources:
      - DocLayNet
      - M6Doc
```

This avoids unintuitive global filtering.

---

## 12.2 Secondary Filtering

Secondary filtering can use metadata columns:

```yaml
where:
  category:
    in:
      - merged_cell_table
      - borderless_table
  language:
    in:
      - en
      - zh
  quality_flags:
    not_contains:
      - needs_review
      - invalid_bbox
```

---

## 12.3 Materialized Indexes

For frequently used subsets, lightweight indexes may be created:

```text
canonical/indexes/table/source=DocBank/category=merged_cell_table.parquet
```

These index files should contain only pointers:

| Column        |
| ------------- |
| `record_id`   |
| `task`        |
| `source_name` |
| `category`    |
| `page_id`     |
| `region_id`   |

They should not duplicate full canonical records.

---

# 13. Split Management

Splits should be assigned at the document level by default.

Rule:

> **Do not split records from the same document across train/val/test unless explicitly required.**

Recommended split levels:

| Level        | Use Case                                       |
| ------------ | ---------------------------------------------- |
| `document`   | Default                                        |
| `page`       | Acceptable for page-independent synthetic data |
| `record`     | Only for special cases                         |
| `predefined` | Use source-provided splits                     |

Example:

```yaml
split_policy:
  level: document
  strategy: random
  train_ratio: 0.98
  val_ratio: 0.01
  test_ratio: 0.01
  seed: 42
```

---

# 14. Stable ID Design

IDs should be deterministic whenever possible.

Recommended convention:

```text
document_id = doc:{source_name}:{document_hash}
page_id     = page:{source_name}:{document_hash}:{page_index}
region_id   = region:{source_name}:{document_hash}:{page_index}:{region_hash}
record_id   = {task}:{source_name}:{page_or_region_id}
asset_id    = asset:{asset_type}:{source_name}:{content_or_transform_hash}
view_id     = view:{view_name}:{canonical_record_id}:{template_hash}
```

Avoid random UUIDs unless unavoidable.

---

# 15. Lineage Tracking

Every View record should be traceable to:

```text
view record
  → canonical task record
    → canonical region/page/document
      → canonical image asset
        → source raw file
          → source annotation
```

Minimum required lineage fields:

| Layer            | Required Lineage                                                       |
| ---------------- | ---------------------------------------------------------------------- |
| Source           | raw path, annotation path, checksum                                    |
| Canonical entity | source name, source document/page/annotation ID                        |
| Canonical record | document ID, page ID, region ID, source annotation ID                  |
| Canonical asset  | parent asset ID, transform hash                                        |
| View record      | canonical record ID, prompt template ID, target format, image asset ID |

---

# 16. Metadata and Manifests

## 16.1 Avoid Metadata Duplication

Do not require README, YAML, and statistics files under every directory.

Use centralized manifests instead:

```text
canonical/manifests/canonical.yaml
canonical/manifests/sources/<source_name>.yaml
canonical/manifests/tasks/<task_name>.yaml
views/<view_name>/view.yaml
views/<view_name>/stats.json
```

---

## 16.2 Source Manifest Example

```yaml
source: DocBank
status: active

entities:
  documents:
    path: canonical/entities/documents/source=DocBank/
    num_records: 10000
  pages:
    path: canonical/entities/pages/source=DocBank/
    num_records: 50000
  regions:
    path: canonical/entities/regions/source=DocBank/
    num_records: 300000

tasks:
  layout:
    status: disabled
    path: canonical/records/layout/source=DocBank/
    num_records: 50000

  table:
    status: active
    path: canonical/records/table/source=DocBank/
    num_records: 18000

  formula:
    status: active
    path: canonical/records/formula/source=DocBank/
    num_records: 9000
```

---

## 16.3 Task Manifest Example

```yaml
task: table
schema: canonical/registry/schemas/table.schema.json

sources:
  DocBank:
    status: active
    path: canonical/records/table/source=DocBank/
    num_records: 18000
    categories:
      borderless_table: 3200
      merged_cell_table: 4100
      three_line_table: 7800

  PubTabNet:
    status: active
    path: canonical/records/table/source=PubTabNet/
    num_records: 500000
    categories:
      standard_table: 500000
```

---

# 17. Validation Requirements

Validation should be implemented as first-class tooling.

## 17.1 Source Validation

Check:

* Raw files exist
* Annotation files exist
* Checksums are valid
* Declared source format matches files
* Required metadata exists

Command:

```bash
dataset validate-source DocBank
```

---

## 17.2 Canonical Validation

Check:

* Unique IDs
* Valid document/page/region references
* Valid image asset references
* Valid bounding boxes
* Valid coordinate spaces
* Valid task schema
* Valid target serialization
* Valid quality flags
* No broken lineage

Command:

```bash
dataset validate-canonical --task table --source DocBank
```

---

## 17.3 View Validation

Check:

* Image paths exist
* Prompt is non-empty
* Label is non-empty
* Split is valid
* No train/val/test leakage
* Target format matches task
* Coordinate transforms are applied correctly
* View records point back to canonical records

Command:

```bash
dataset validate-view mineru25_sft_v1
```

---

# 18. Processing Module Design

Recommended package layout:

```text
dataset/
  sources/
    adapters/
      base.py
      docbank.py
      doclaynet.py
      mineru.py
      synthetic.py

  canonical/
    schemas/
    writer.py
    validator.py
    normalizer.py

  assets/
    renderer.py
    cropper.py
    transformer.py
    manifest.py

  views/
    builder.py
    sampler.py
    serializer.py
    templates/
      mineru25/
        layout.jinja
        table.jinja
        formula.jinja
        text.jinja
      qwen3vl/
        layout.jinja
        table_html.jinja

  lineage/
    resolver.py

  stats/
    profiler.py
    report.py

  cli.py
```

---

## 18.1 Source Adapter Interface

```python
class SourceAdapter:
    def scan_documents(self):
        ...

    def export_documents(self):
        ...

    def export_pages(self):
        ...

    def export_regions(self):
        ...

    def export_task_records(self, task: str):
        ...
```

Each adapter implements only the capabilities supported by the source dataset.

---

## 18.2 View Builder Interface

```python
class ViewBuilder:
    def load_partitions(self, config):
        ...

    def apply_selection(self, records, include, exclude):
        ...

    def sample_records(self, records, mixture):
        ...

    def prepare_image(self, record, image_policy):
        ...

    def render_prompt(self, record, template):
        ...

    def render_label(self, record, target_format):
        ...

    def write_parquet(self, records, output_path):
        ...
```

---

# 19. CLI Commands

Recommended commands:

```bash
# Source layer
dataset validate-source <source_name>

# Source → Canonical
dataset export-source <source_name> \
  --config sources/<source_name>/export.yaml

# Canonical validation
dataset validate-canonical
dataset validate-canonical --task table
dataset validate-canonical --task table --source DocBank

# Canonical statistics
dataset profile-canonical
dataset profile-canonical --task layout --source DocBank

# Build view
dataset build-view views/<view_name>/view.yaml

# Validate view
dataset validate-view <view_name>

# Report
dataset report-view <view_name>

# Trace lineage
dataset trace --view-record-id <id>
dataset trace --canonical-record-id <id>
```

---

# 20. Practical Defaults

For the first implementation, use the following defaults:

```yaml
canonical:
  partitioning:
    entities: source
    records: task/source

  image:
    page_render_format: png
    crop_format: png
    page_render_dpi: 200
    crop_margin_relative: 0.03
    crop_margin_max_pixels: 32

  coordinate_space:
    canonical: canonical_page_pixel_xyxy

  metadata:
    use_manifests: true
    avoid_directory_level_readme_duplication: true

view:
  output_format: parquet
  image_materialization: source_reference
  split_level: document

lineage:
  deterministic_ids: true
  require_asset_manifest: true
  require_canonical_record_id_in_view: true
```

---

# 21. Final Architecture Summary

The final dataset design is:

```text
Source Layer
  - Preserve raw PDFs/images and original annotations.
  - Store source metadata and export configuration.
  - Do not normalize aggressively.

Canonical Layer
  - Normalize into documents, pages, regions, and task records.
  - Keep task records physically partitioned by task/source.
  - Store deterministic page renders and region crops as canonical assets.
  - Store model-independent semantic targets.
  - Maintain manifests, schemas, and lineage.

View Layer
  - Select task/source/category subsets.
  - Apply sampling and split policies.
  - Render prompts and serialize labels for a specific model.
  - Optionally materialize model-specific image assets.
  - Export train/val/test Parquet files for training.

Runtime
  - Perform tensor conversion, normalization, random augmentation, and declared lightweight transforms only.
```

The most important final decisions are:

1. **Keep the three-layer architecture.**
2. **Keep task/source partitioning in the canonical layer.**
3. **Use shared entity indexes for documents, pages, and regions.**
4. **Treat image derivatives as first-class assets.**
5. **Create reusable deterministic image assets during `source → canonical`.**
6. **Create model-specific image assets during `canonical → view` only when configured.**
7. **Keep canonical targets model-independent.**
8. **Generate prompts and model-specific labels only in the View layer.**
9. **Use manifests and generated statistics instead of redundant metadata files in every directory.**
10. **Ensure every training record can be traced back to canonical records, canonical assets, and original source files.**
