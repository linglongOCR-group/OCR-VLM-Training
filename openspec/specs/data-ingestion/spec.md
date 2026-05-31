# Data Ingestion

## Purpose

Define how source adapters, the canonical writer, and the asset manager cooperate to
convert heterogeneous raw datasets (COCO, YOLO, MinerU, OCR JSON, HTML tables, LaTeX,
custom XML/JSON, synthetic) into a normalized canonical dataset of entities, task
records, and image assets stored as partitioned Parquet with deterministic filenames and
source/task manifests.

---
## Requirements

### Requirement 1: Source Adapter Interface

Every source dataset MUST be ingested through a concrete `SourceAdapter` subclass that
exposes `scan_documents`, `export_documents`, `export_pages`, `export_regions`, and
`export_task_records(task)`. Each adapter SHALL declare a stable `name` and `version`.

#### Scenario: Exporting a single-source dataset

Given a `DocBankAdapter` registered under name `"DocBank"` at version `"1.0.0"`
And the raw DocBank files are present at the configured source root
When `export_task_records("table")` is called
Then the adapter yields canonical table records for every annotated region
And no record references serializer logic, reward profiles, or model prompts.

#### Scenario: Scanning before export

Given a `MinerUAdapter` with source root pointing to a directory of PDFs and JSON
When `scan_documents()` is called
Then it returns an iterable of document descriptors keyed by `document_id`
And each descriptor includes the raw file paths and metadata needed for subsequent
export calls.

---

### Requirement 2: Adapter Isolation from Downstream Concerns

Source adapters MUST NOT import or reference serializers, reward adapters, view
builders, prompt templates, or training-framework code. They SHALL produce only
canonical entities, canonical records, and canonical asset references.

#### Scenario: Verifying adapter purity

Given the source code of any registered adapter
When a static import analysis is performed
Then no import resolves to `serializers.*`, `rewards.*`, `views.*`, `runtime.*`, or
`prompt*`.

---

### Requirement 3: Heterogeneous Source Format Support

The ingestion pipeline SHALL support at least the following source formats: COCO JSON,
YOLO txt, MinerU JSON, OCR JSON, HTML tables, LaTeX source, custom XML/JSON, and
synthetic generators. Adding a new format MUST NOT require modifying the canonical
writer or asset manager.

#### Scenario: Registering a new source format

Given a new `SyntheticAdapter` that subclasses `SourceAdapter`
When the adapter class path is added to the `source_adapters` section of
`configs/processing.yaml`
Then `docds export-source Synthetic --tasks layout` succeeds without code changes to
the canonical writer or asset manager.

---

### Requirement 4: Canonical Writer Partitioned Output

The canonical writer MUST persist entities as partitioned Parquet files under
`canonical/entities/{entity_type}/source={name}/` and task records under
`canonical/records/{task}/source={name}/`. File names SHALL be deterministic
(e.g. `part-00000.parquet` based on a stable sort key).

#### Scenario: Writing page entities for two sources

Given DocBank pages and DocLayNet pages are exported in the same run
When `write_pages(records, source_name="DocBank")` and
`write_pages(records, source_name="DocLayNet")` are called
Then files appear at `canonical/entities/pages/source=DocBank/part-00000.parquet`
and `canonical/entities/pages/source=DocLayNet/part-00000.parquet`
And each partition contains only records from its respective source.

---

### Requirement 5: Overwrite Mode and Partition Isolation

The canonical writer MUST support an overwrite-per-partition mode. Writing one
source/task partition SHALL NOT modify or delete unrelated partitions.

#### Scenario: Re-exporting a single source

Given canonical records already exist for `source=DocBank` task `table` and
`source=PubTabNet` task `table`
When `docds export-source DocBank --tasks table --overwrite-partitions` is executed
Then only `canonical/records/table/source=DocBank/` is rewritten
And `canonical/records/table/source=PubTabNet/` remains unchanged.

---

### Requirement 6: Manifest Maintenance

The canonical writer SHALL maintain a source manifest and a task manifest listing
exported partitions, row counts, checksums, and adapter version. Manifests MUST be
updated only for partitions that were actually written during a run.

#### Scenario: Manifest reflects partial export

Given manifests list source `DocBank` (tasks: `table`, `layout`) and source `PubTabNet`
(task: `table`)
When only `DocBank` task `table` is re-exported
Then the source manifest entry for `DocBank` is updated with the new row count and
checksum for task `table`
And the entry for `DocBank` task `layout` is untouched
And the entry for `PubTabNet` task `table` is untouched.

---

### Requirement 7: Asset Manager - Page Renders

The asset manager MUST render source document pages into canonical page image assets.
Each rendered asset SHALL record: parent asset ID, transform specification hash, output
width and height, output path, file checksum, and coordinate mapping from source
coordinates to rendered pixel coordinates.

#### Scenario: Rendering a PDF page

Given a source document at page index 3 with a render profile specifying 200 DPI and
PNG output
When `render_page(document, page_index=3, profile)` is called
Then a PNG file is written under `canonical/assets/pages/source={name}/`
And the returned asset dict includes `parent_asset_id`, `transform_hash`, `width`,
`height`, `path`, `checksum`, and `coordinate_mapping`.

---

### Requirement 8: Asset Manager - Region Crops

The asset manager MUST produce region crop assets from page images using bounding box
coordinates. The crop asset SHALL inherit coordinate mapping from the parent page asset
so downstream transforms can trace back to source coordinates.

#### Scenario: Cropping a table region

Given a page asset with path `canonical/assets/pages/source=DocBank/000001.png`
And a bounding box `[100, 200, 500, 700]` in source page pixel coordinates
When `crop_region(page_asset, bbox, profile)` is called
Then a cropped image is written under `canonical/assets/regions/source=DocBank/`
And the crop asset dict records `parent_asset_id` equal to the page asset ID
And `coordinate_mapping` maps from crop pixel space back to source page pixel space.

---

### Requirement 9: Asset Manager - View-Specific Transforms

The asset manager SHALL support optional view-specific image transforms (resize, pad,
normalize) that produce derived assets. Each derived asset SHALL record the full
transform specification so that coordinate transformations can be replayed by serializers.

#### Scenario: Resizing a region crop for a view

Given a region crop asset of size 400x500 pixels
And a view transform profile specifying `scale: 0.5`
When `transform_for_view(crop_asset, profile)` is called
Then a resized image of size 200x250 pixels is produced
And the returned asset dict contains a `transform_hash` derived from the profile
And `coordinate_mapping` encodes the scale factor so that layout bboxes can be
transformed from canonical coordinates to view-image coordinates.

---

### Requirement 10: Asset Manifest

The asset manager SHALL write an asset manifest listing every asset with its ID, parent
ID, transform hash, dimensions, path, checksum, and coordinate mapping. The manifest
MUST be updated atomically after each export run.

#### Scenario: Inspecting assets after export

Given an export run produced 120 page renders and 340 region crops
When the export completes
Then `canonical/assets/manifest.json` lists 460 asset entries
And each entry includes `asset_id`, `parent_asset_id`, `transform_hash`, `width`,
`height`, `path`, `checksum`, and `coordinate_mapping`.

---

### Requirement 11: Task-Filtered Export

The export-source command MUST accept a `--tasks` flag that limits export to the
specified tasks. When `--tasks` is omitted, all tasks supported by the adapter SHALL be
exported.

#### Scenario: Exporting only layout and table tasks

Given a source adapter supports tasks `layout`, `table`, `formula`, and `text`
When `docds export-source DocBank --tasks layout,table` is executed
Then only `canonical/records/layout/source=DocBank/` and
`canonical/records/table/source=DocBank/` are written
And no records or manifests are produced for `formula` or `text`.

---

### Requirement 12: Malformed Annotation Error Handling

By default, a malformed annotation during export MUST halt the export with a descriptive
error. When `--skip-errors` is passed, malformed annotations SHALL be omitted from the
output shards and logged in the manifest under an `errors` section listing each skipped
record with the reason.

#### Scenario: Strict mode stops on bad annotation

Given a source containing one document with a corrupted bounding box (negative width)
And `--skip-errors` is not set
When `docds export-source DocBank --tasks layout` is executed
Then the export aborts with an error message identifying the document and annotation
And no partial Parquet files remain in the output directory.

#### Scenario: Skip-errors mode logs and continues

Given the same corrupted annotation
And `--skip-errors` is set
When `docds export-source DocBank --tasks layout` is executed
Then the export completes for all valid annotations
And the manifest `errors` section lists the skipped annotation with its `document_id`,
`reason`, and original raw data reference.

---

### Requirement 13: Unreadable Source Image Handling

When a source image is missing or permission-denied, the system SHALL respect an
`allow_unreadable_images` configuration flag. When `false` (default), unreadable images
MUST cause a hard error. When `true`, affected records SHALL be skipped and logged in
the manifest errors section.

#### Scenario: Missing source image with default settings

Given a source document references an image file that does not exist on disk
And `allow_unreadable_images` is `false` (default)
When export is attempted for that document
Then the export fails with a clear error naming the missing file.

#### Scenario: Missing source image with allow_unreadable_images enabled

Given the same missing image
And `allow_unreadable_images` is `true`
When export continues
Then all records referencing the missing image are skipped
And the manifest errors section lists each skipped record with the missing file path.

---

### Requirement 14: Dry-Run Support

The export-source command MUST support a `--dry-run` flag that validates source files,
enumerates the documents and tasks to be exported, and reports projected row counts
without writing any output files.

#### Scenario: Validating a large export before running

Given a source directory containing 10,000 documents
When `docds export-source DocBank --tasks table --dry-run` is executed
Then no Parquet files or image assets are written
And the command prints the number of documents scanned, the number of regions with
table annotations, and the projected task record count.

---

### Requirement 15: Deterministic and Reproducible Output

Given the same source data and the same adapter version, the canonical writer MUST
produce bit-identical Parquet files and checksums across repeated runs. Record ordering
within each partition SHALL be deterministic based on a stable sort key.

#### Scenario: Re-exporting produces identical output

Given the DocBank source data has not changed
And the adapter version is `"1.0.0"` for both runs
When `docds export-source DocBank --tasks table` is executed twice
Then the Parquet files from both runs are byte-identical
And the manifest checksums match.

### Requirement: UniRec40M Inline Formula Delimiters

The UniRec40M ingestion path SHALL normalize formulas embedded in text and table targets using MinerU-compatible unescaped `$...$` delimiters. It MUST NOT emit escaped delimiters such as `\$...\$` for inline formula annotations.

#### Scenario: Text record inline formula uses unescaped delimiters

- **GIVEN** a UniRec40M text label containing an inline formula source wrapper such as `\(x^2\)`
- **WHEN** the UniRec adapter exports the record as a canonical text task
- **THEN** the canonical text target SHALL contain `$x^2$`
- **AND** the canonical text target SHALL NOT contain `\$x^2\$`

#### Scenario: Table record inline formula uses unescaped delimiters

- **GIVEN** a UniRec40M table label containing an inline formula source wrapper such as `\(x^2\)`
- **WHEN** the UniRec adapter exports the record as a canonical table task
- **THEN** the canonical table target SHALL contain `$x^2$`
- **AND** the canonical table target SHALL NOT contain `\$x^2\$`

#### Scenario: Rebuilt view preserves unescaped delimiters

- **GIVEN** UniRec40M canonical text or table records whose targets contain inline formulas
- **WHEN** a UniRec40M SFT view is rebuilt from those canonical records
- **THEN** the assistant label SHALL preserve `$...$` inline formula delimiters
- **AND** the assistant label SHALL NOT contain `\$...\$` formula delimiters

### Requirement: UniRec40M Nested JSONL Export

The source ingestion system SHALL export UniRec40M nested subset JSONL records into the canonical dataset layout.

#### Scenario: Export text recognition crop

- GIVEN a UniRec40M source profile with `adapter: unirec`
- AND a source root containing `<subset>/annotations/records.jsonl` and source images
- WHEN `docds export-source unirec --source-config <profile>` runs for text tasks
- THEN the system SHALL write canonical document, page, region, asset manifest, and text task records
- AND image assets SHALL reference the original dataset-relative image path rather than copying image bytes

#### Scenario: Export formula record

- GIVEN a UniRec40M record whose label is a standalone LaTeX formula
- WHEN the UniRec adapter exports formula tasks
- THEN the system SHALL write a formula task record whose target contains stripped LaTeX without display wrappers

#### Scenario: Export table record

- GIVEN a UniRec40M record with table hints in metadata or table-like label text
- WHEN the UniRec adapter exports table tasks
- THEN the system SHALL write a table task record with the cleaned table text target

### Requirement: UniRec40M Label Normalization

The UniRec adapter SHALL normalize source-specific text markers before writing canonical targets.

#### Scenario: Remove source line tokens

- GIVEN a UniRec label containing `<|ln|>`, `<|pn|>`, `<|sn|>`, or `<<<change_line_token_wrap>>>`
- WHEN the label is exported as text
- THEN those tokens SHALL be removed from the canonical target

#### Scenario: Convert inline math wrappers

- GIVEN a UniRec text label containing `\(...\)` inline math
- WHEN the label is exported as text
- THEN the inline math content SHALL be preserved using `$...$` delimiters

### Requirement: UniRec40M Category Profiles

The repository SHALL provide source profiles for the UniRec40M category shards selected for MinerU2.5 training.

#### Scenario: Resolve UniRec40M source profile

- GIVEN `configs/data/processing.yaml`
- WHEN a UniRec40M category profile is passed to `docds export-source`
- THEN the configured source registry SHALL resolve the `unirec` adapter class
- AND the profile SHALL resolve paths relative to `OCR_DATA_ROOT`

