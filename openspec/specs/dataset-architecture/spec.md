# Dataset Architecture Specification

## Purpose

Define the three-layer dataset architecture (Source -> Canonical -> View), entity schemas, image asset management, coordinate systems, ID conventions, and split policies used for OCR VLM training data.

## Requirements

### Requirement: Three-Layer Separation

The system SHALL organize all dataset artifacts into three distinct layers: Source, Canonical, and View.

#### Scenario: Layer ownership

- GIVEN a dataset processing pipeline
- WHEN data flows from raw sources through to training-ready files
- THEN the Source layer SHALL own raw PDFs, images, and original annotations
- AND the Canonical layer SHALL own normalized model-independent records and deterministic image derivatives
- AND the View layer SHALL own model-specific prompts, labels, splits, and optional image derivatives

### Requirement: Source Layer Immutability

The Source layer SHALL be treated as mostly immutable. The system SHALL NOT modify source files during canonical export or view construction.

#### Scenario: No modification of raw files

- GIVEN an exported source dataset
- WHEN the canonical export pipeline runs
- THEN the system SHALL NOT modify any files in the Source layer
- AND the system SHALL write all outputs to the Canonical layer

### Requirement: Canonical Model Independence

The Canonical layer SHALL store model-independent semantic truth.

#### Scenario: No model-specific formats in canonical records

- GIVEN a canonical task record
- WHEN the record is inspected
- THEN it SHALL NOT contain model-specific prompt strings or training labels (e.g., MinerU box strings, OTSL tokens)
- AND it SHALL contain only semantic, model-independent target data (e.g., bbox coordinates, labels, rotation angles as structured JSON)

### Requirement: Document Entity Schema

The system SHALL store one row per original document with required fields: `document_id`, `source_name`, `source_document_id`, `document_path`, `document_type`, `num_pages`, `language`, `domain`, `checksum`, `metadata`, `schema_version`.

#### Scenario: Document entity creation

- GIVEN a source dataset containing PDF documents
- WHEN the canonical export produces document entities
- THEN each entity SHALL have a stable `document_id`
- AND the ID SHOULD follow the recommended convention `doc:{source_name}:{document_hash}`
- AND `checksum` SHALL contain the raw document checksum
- AND `num_pages` SHALL reflect the actual page count

### Requirement: Page Entity Schema

The system SHALL store one row per document page with required fields: `page_id`, `document_id`, `source_name`, `source_page_id`, `page_index`, `page_image_asset_id`, `width`, `height`, `rotation`, `attributes`, `schema_version`.

#### Scenario: Page entity creation

- GIVEN a multi-page document
- WHEN the canonical export produces page entities
- THEN `page_id` SHOULD follow the recommended convention `page:{source_name}:{document_hash}:{page_index}`
- AND `page_index` SHALL be zero-based
- AND `width` and `height` SHALL correspond to the canonical rendered page image dimensions

### Requirement: Region Entity Schema

The system SHALL store one row per annotated or generated region with required fields: `region_id`, `page_id`, `document_id`, `source_name`, `source_annotation_id`, `bbox`, `rotation`, `element_type`, `category`, `subcategory`, `reading_order`, `crop_asset_id`, `quality_flags`, `metadata`, `schema_version`.

#### Scenario: Region entity with bounding box

- GIVEN a source annotation defining a table region on a page
- WHEN the canonical export produces a region entity
- THEN `bbox` SHALL be a list `[x1, y1, x2, y2]` in canonical page pixel coordinates
- AND `element_type` SHALL be one of the supported types (text, table, formula, etc.)
- AND `crop_asset_id` SHALL reference the canonical crop image asset

### Requirement: Task Record Partitioning

Canonical task records SHALL be physically partitioned by `task/source`.

#### Scenario: Partition-based selection

- GIVEN a view configuration that includes `task: table, sources: [DocBank]`
- WHEN the view builder loads canonical records
- THEN it SHALL read from `canonical/records/table/source=DocBank/`
- AND it SHALL NOT load records from other tasks or sources

### Requirement: Common Task Record Fields

All canonical task records SHALL contain: `record_id`, `task`, `source_name`, `document_id`, `page_id`, `region_id`, `image_asset_id`, `target`, `category`, `subcategory`, `language`, `quality_flags`, `provenance`, `metadata`, `schema_version`.

#### Scenario: Task record creation

- GIVEN a canonical table record
- WHEN the record is written to Parquet
- THEN `record_id` SHALL be stable and SHOULD follow the convention `{task}:{source_name}:{page_or_region_id}`
- AND `target` SHALL contain model-independent structured data (e.g., HTML, cells, OTSL as JSON fields)
- AND `provenance` SHALL include source lineage information

### Requirement: Image Asset First-Class Treatment

Image resources SHALL be treated as first-class dataset assets with manifests.

#### Scenario: Asset manifest recording

- GIVEN a canonical region crop image is generated
- WHEN the asset is written to disk
- THEN the system SHALL record an entry in the asset manifest with `asset_id`, `asset_type`, `parent_asset_id`, `transform_spec_hash`, `width`, `height`, `format`, `checksum`, and `path`

### Requirement: Canonical Coordinate System

Canonical annotations SHALL use `canonical_page_pixel_xyxy` coordinates.

#### Scenario: Bounding box in canonical space

- GIVEN a region annotation with pixel coordinates on the original page
- WHEN the canonical record is created
- THEN `bbox` SHALL be in `[x1, y1, x2, y2]` format relative to the canonical page image pixel space
- AND the coordinate space SHALL be documented in the target JSON

### Requirement: View Coordinate Transform

View labels SHALL use coordinates that match the actual image fed to the model.

#### Scenario: Layout target after image resize

- GIVEN a canonical bbox `[100, 200, 500, 700]` and a view image resized by factor 0.5
- WHEN the view serializer produces the layout label
- THEN the coordinates SHALL be transformed to `[50, 100, 250, 350]`
- AND the transform SHALL be recorded in the view asset manifest

### Requirement: Image Materialization Modes

The view builder SHALL support three image materialization modes: `embedded`, `source_reference`, and `nested_reference`.

#### Scenario: Embedded mode

- GIVEN a view configured with `materialization.mode: embedded`
- WHEN the view builder writes training records
- THEN each record SHALL contain `images_bytes` with flat binary image data (PNG or WebP)
- AND `image_path` SHALL remain as a lineage/debug field only

#### Scenario: Source reference mode

- GIVEN a view configured with `materialization.mode: source_reference`
- WHEN the view builder writes training records
- THEN each record SHALL contain `images_path` with dataset-root-relative paths
- AND identity images SHALL reference canonical/source asset paths
- AND transformed images SHALL be saved under `views/<view>/assets/`

#### Scenario: Nested reference mode

- GIVEN a view configured with `materialization.mode: nested_reference`
- WHEN the view builder writes training records
- THEN each record SHALL contain `images` with VERL-native nested image dicts
- AND paths SHALL follow the same dataset-root-relative policy as source_reference mode

### Requirement: Deterministic Stable IDs

Dataset IDs SHALL be deterministic whenever possible.

#### Scenario: ID generation consistency

- GIVEN the same source data and export configuration
- WHEN the canonical export runs twice
- THEN all generated IDs (document_id, page_id, region_id, record_id, asset_id) SHALL be identical across both runs

### Requirement: Document-Level Default Splitting

Splits SHALL be assigned at the document level by default.

#### Scenario: No cross-split document leakage

- GIVEN a split policy with `level: document`
- WHEN splits are assigned
- THEN all records from the same document SHALL be in the same split
- AND no document SHALL have records in both train and val

#### Scenario: Alternative split levels

- GIVEN a split policy with `level: page`
- WHEN splits are assigned
- THEN pages from the same document MAY appear in different splits
- AND each record's split assignment SHALL be deterministic given the same seed

### Requirement: Asset Creation Timing

Image assets SHALL be created at the correct layer boundary.

#### Scenario: Canonical image creation

- GIVEN a source-to-canonical export
- WHEN page renders and region crops are generated
- THEN these SHALL be created during `source -> canonical` and persisted as canonical assets

#### Scenario: View-specific images

- GIVEN a view configuration with image transforms
- WHEN the view builder materializes records
- THEN model-specific resized/padded images SHALL be created during `canonical -> view`
- AND random augmentation SHALL NOT be persisted

### Requirement: Sharded Output for Large Views

The view builder SHALL support sharded Parquet output for large embedded-image views.

#### Scenario: Sharded split output

- GIVEN a view configured with `shard_policy.rows_per_shard: 128`
- WHEN the view builder writes the training split
- THEN output SHALL be written as `train/part-00000.parquet`, `train/part-00001.parquet`, etc.
- AND each shard SHALL contain at most 128 rows

### Requirement: View Layer Redundancy

The View layer SHALL be allowed to be redundant because it is a materialized training dataset.

#### Scenario: View regeneration

- GIVEN an existing materialized view
- WHEN the view builder re-runs with the same configuration
- THEN the output SHALL be deterministic and replace the previous materialization

### Requirement: Manifest Centralization

The system SHALL use centralized manifests instead of requiring per-directory metadata files.

#### Scenario: Source manifest

- GIVEN an exported source dataset
- WHEN canonical export completes
- THEN the system SHALL write a source manifest to `canonical/manifests/sources/<source_name>.yaml`
- AND it SHALL NOT require README files in every subdirectory

### Requirement: Lineage Traceability

Every view record SHALL be traceable back to its source annotation through canonical records.

#### Scenario: End-to-end lineage chain

- GIVEN a view training record
- WHEN lineage is traced
- THEN the system SHALL resolve: view record -> canonical task record -> canonical region/page/document -> canonical image asset -> source raw file -> source annotation
