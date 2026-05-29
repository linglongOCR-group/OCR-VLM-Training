# MinerU2.5 Fine-Tuning Data Format Specification

## Purpose

This specification defines the data format contract for constructing SFT-style training datasets compatible with MinerU2.5 inference behavior. It covers the chat/message format, task-specific prompt and target grammars, annotation-to-training conversion rules, and validation requirements. The key design principle is that training targets SHALL be raw model output strings matching what the inference parser expects, not final post-processed output.

## Requirements

### Requirement: SFT Record Structure

Each training sample SHALL be one JSONL row containing a globally unique `id`, a `task` field, an `images` list, and a `messages` array in chat format with system, user, and assistant turns.

#### Scenario: Layout detection record

- **GIVEN** a document page image at `images/pages/doc001_p000.png`
- **WHEN** constructing a layout detection training sample
- **THEN** the record SHALL contain `"task": "layout"`, the full page image in the `images` list, and a `messages` array with system prompt `"You are a helpful assistant."`, a user turn containing the image followed by `"\nLayout Detection:"`, and an assistant turn containing the raw layout grammar string

#### Scenario: Crop-level recognition record

- **GIVEN** a cropped block image from a detected layout region
- **WHEN** constructing a text, formula, or table recognition sample
- **THEN** the record SHALL contain the appropriate `task` value, the crop image path in `images`, and a `messages` array where the user turn contains the crop image followed by the task-specific prompt string

### Requirement: Raw Training Target Contract

Assistant targets in training data SHALL be raw model output strings before post-processing, not final ContentBlock JSON or HTML output. The `messages[-1]["content"]` assistant turn SHALL contain exactly what the inference-time parser expects as input.

#### Scenario: Layout target is compact grammar, not JSON

- **WHEN** writing the assistant target for a layout detection record
- **THEN** the content SHALL be a concatenation of `<|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>block_type<|ref_end|><|rotate_*|>` records
- **AND** SHALL NOT be a ContentBlock JSON array or any other post-processed representation

#### Scenario: Table target is OTSL, not HTML

- **WHEN** writing the assistant target for a table recognition record
- **THEN** the content SHALL use OTSL tokens (`<fcel>`, `<ecel>`, `<lcel>`, `<ucel>`, `<xcel>`, `<nl>`)
- **AND** SHALL NOT contain HTML tags unless a non-standard training configuration explicitly overrides this

#### Scenario: Formula target is bare LaTeX

- **WHEN** writing the assistant target for a formula recognition record
- **THEN** the content SHALL be a LaTeX string without Markdown fences, without `$...$` wrapping, and without natural-language commentary

#### Scenario: Text target is plain recognized content

- **WHEN** writing the assistant target for a text recognition record
- **THEN** the content SHALL be only the recognized textual content
- **AND** SHALL NOT be wrapped in JSON, prefixed with explanatory prose, or otherwise decorated

### Requirement: Layout Detection Grammar

Layout detection raw output SHALL follow the grammar `<|box_start|>{x1} {y1} {x2} {y2}<|box_end|><|ref_start|>{block_type}<|ref_end|><|rotate_*|>` with integer coordinates on a 0-1000 logical grid, a valid block type, and a rotation token.

#### Scenario: Coordinate conversion from pixel annotations

- **GIVEN** a pixel-level bounding box `[x1_px, y1_px, x2_px, y2_px]` on a page of dimensions `width x height`
- **WHEN** converting to layout target coordinates
- **THEN** each coordinate SHALL be computed as `round(1000 * px_value / dimension)`, producing integers in `[0, 1000]`
- **AND** `x1 < x2` and `y1 < y2` SHALL hold
- **AND** values SHALL be clamped to `[0, 1000]` after rounding

#### Scenario: Block type validation

- **WHEN** specifying a block type in layout output
- **THEN** the type SHALL be one of: `text`, `title`, `table`, `equation`, `code`, `algorithm`, `aside_text`, `ref_text`, `phonetic`, `list_item`, `table_caption`, `image_caption`, `code_caption`, `table_footnote`, `image_footnote`, `header`, `footer`, `page_number`, `page_footnote`, `image`, `chart`, `list`, `image_block`, `equation_block`, `unknown`

#### Scenario: Rotation token usage

- **WHEN** writing layout output for a block with orientation angle
- **THEN** the rotation token SHALL be one of: `<|rotate_up|>` for 0 degrees, `<|rotate_right|>` for 90 degrees, `<|rotate_down|>` for 180 degrees, `<|rotate_left|>` for 270 degrees
- **AND** the rotation token SHALL be present in every layout record, not omitted

### Requirement: Task-Specific Prompt Mapping

Each recognition task SHALL use its designated prompt string appended after the image in the user turn, mapping block types to prompts deterministically.

#### Scenario: Prompt selection by block type

- **GIVEN** a cropped block with a known block type
- **WHEN** constructing the user turn
- **THEN** the prompt SHALL be selected as: table maps to `"\nTable Recognition:"`, equation maps to `"\nFormula Recognition:"`, image or chart maps to `"\nImage Analysis:"`, and all other text-like types map to `"\nText Recognition:"`
- **AND** layout detection SHALL use `"\nLayout Detection:"` with the full page image

#### Scenario: Layout uses full page image, not a crop

- **WHEN** constructing a layout detection record
- **THEN** the image SHALL be the full document page image
- **AND** SHALL NOT be a cropped region

### Requirement: Table OTSL Token Semantics

Table recognition training targets SHALL use OTSL tokens where `<fcel>` introduces a filled cell with content after the token, `<ecel>` marks an empty cell, `<lcel>` continues horizontally from the left cell, `<ucel>` continues vertically from the upper cell, `<xcel>` continues in both directions, and `<nl>` starts a new row.

#### Scenario: Simple table with filled cells

- **GIVEN** a visual table with headers `[Name, Amount]` and rows `[A, 10]` and `[B, 20]`
- **WHEN** generating the OTSL target
- **THEN** the output SHALL be `<fcel>Name<fcel>Amount<nl><fcel>A<fcel>10<nl><fcel>B<fcel>20<nl>`

#### Scenario: Table with merged cells

- **GIVEN** a table where a header spans two columns
- **WHEN** generating the OTSL target for the merged header
- **THEN** the output SHALL use `<fcel>` for the content cell followed by `<lcel>` for the horizontal continuation

#### Scenario: Table with empty cells

- **GIVEN** a table cell with no visible content
- **WHEN** generating the OTSL target for that cell
- **THEN** the output SHALL use `<ecel>` with no text following it

### Requirement: Image and Chart Block Handling

Image and chart blocks SHALL be optional in training data and skipped by default unless image analysis is explicitly enabled, matching the inference behavior where `image_analysis=True` is required for these block types.

#### Scenario: Omitting image/chart records by default

- **WHEN** constructing a training dataset for standard document OCR behavior
- **THEN** image and chart analysis records SHALL be omitted
- **AND** layout detection SHALL still detect and classify image/chart blocks with their spatial positions

#### Scenario: Including image/chart records when needed

- **WHEN** the target application requires figure description, chart parsing, or diagram extraction
- **THEN** image/chart records SHALL be included using the `"\nImage Analysis:"` prompt
- **AND** the assistant target SHALL be the analysis string the model is expected to produce

### Requirement: Canonical JSONL Field Schema

Each JSONL record SHALL include an `id`, `task`, `images` list, `messages` array, and optional `metadata` with document-level provenance. Optional `postprocess_target` fields MAY store the final expected output for validation purposes.

#### Scenario: Required fields present

- **WHEN** writing a JSONL training record
- **THEN** the record SHALL contain `"id"` (globally unique string), `"task"` (one of `layout`, `text`, `table`, `equation`, `image_analysis`, `chart_analysis`), `"images"` (list of relative paths), and `"messages"` (chat-format array with system, user, and assistant turns)

#### Scenario: Metadata provenance

- **WHEN** metadata is included
- **THEN** it SHOULD contain `doc_id`, `page_index`, `language`, `width`, and `height` fields to support filtering and debugging

#### Scenario: Postprocess target for validation

- **WHEN** a `postprocess_target` field is present for a table record
- **THEN** it SHALL contain the HTML representation of the table under `postprocess_target.content` with `postprocess_target.content_format: "html"`
- **AND** this field SHALL NOT be used as the training target itself

### Requirement: VERL Portable Format Modes

Training data SHALL support three portable format modes when materialized as Parquet views for VERL SFT: embedded bytes, source reference paths, and nested reference dicts.

#### Scenario: Embedded mode for self-contained shards

- **WHEN** using embedded mode
- **THEN** the Parquet SHALL contain an `images_bytes` column with binary PNG or WebP data
- **AND** `images_path` SHALL be null
- **AND** `image_path` MAY retain a canonical asset path for lineage and debugging

#### Scenario: Source reference mode for path-backed data

- **WHEN** using source reference mode
- **THEN** the Parquet SHALL contain `images_path` with dataset-root-relative paths referencing canonical or transformed assets
- **AND** identity runtime paths SHALL reference source assets relative to `OCR_DATA_ROOT`
- **AND** transformed images SHALL be saved under the view's own `assets/` directory

#### Scenario: Nested reference mode for VERL-native shape

- **WHEN** using nested reference mode
- **THEN** the Parquet SHALL keep VERL's original `images: [{"image": "..."}]` shape
- **AND** paths SHALL follow the same dataset-root-relative path policy as source reference mode

### Requirement: Sharded Output for Large Views

VERL portable Parquet views SHALL be sharded when the row count is large, with each shard being independently readable.

#### Scenario: Shard sizing

- **WHEN** materializing a large embedded-byte view for VERL SFT
- **THEN** the output SHALL be sharded with a default of 128 rows per shard
- **AND** shard files SHALL follow the naming pattern `train/part-NNNNN.parquet`
- **AND** the expanded shard file list SHALL be passable to VERL `data.train_files`

### Requirement: Task Mixture Proportions

Training datasets SHALL follow recommended mixture proportions to ensure balanced coverage across layout, text, table, and formula tasks.

#### Scenario: Computing mixture ratios

- **WHEN** assembling a training split
- **THEN** layout detection records SHALL comprise 25-35% of samples, text recognition 30-40%, table recognition 20-30%, and formula recognition 5-15%
- **AND** image/chart analysis records are optional and excluded from the percentage calculation by default

#### Scenario: Validating mixture balance

- **WHEN** validating a prepared dataset
- **THEN** a validator SHALL check that the per-task sample counts fall within the recommended ranges
- **AND** SHALL warn if any task deviates by more than 5 percentage points from the recommended range

### Requirement: Annotation-to-Training Conversion

Pixel-level layout annotations SHALL be converted to the 0-1000 coordinate grid, blocks SHALL be cropped to produce recognition records, and the correct prompt SHALL be assigned per block type.

#### Scenario: Layout annotation conversion

- **GIVEN** a page annotation with `bbox_px` pixel coordinates and `width`/`height` dimensions
- **WHEN** producing the layout training target
- **THEN** pixel coordinates SHALL be converted to the 0-1000 grid using `round(1000 * px / dimension)`
- **AND** the angle field SHALL be mapped to the corresponding rotation token

#### Scenario: Crop generation from block annotations

- **GIVEN** a detected block with pixel bbox and block type
- **WHEN** producing a content recognition record
- **THEN** the block SHALL be cropped from the original page image
- **AND** non-table crops SHALL be rotated to upright orientation
- **AND** the prompt SHALL be selected based on block type using the standard mapping

### Requirement: Content Validation Rules

Training data SHALL pass validation checks that verify prompt-task alignment, target format correctness, coordinate validity, and round-trip parsability before training.

#### Scenario: Layout validation

- **WHEN** validating a layout detection record
- **THEN** all box records SHALL match the `<|box_start|>...<|box_end|><|ref_start|>...<|ref_end|><|rotate_*|>` grammar
- **AND** all coordinates SHALL be integers in `[0, 1000]` with `x1 < x2` and `y1 < y2`
- **AND** all block types SHALL be in the supported block type list
- **AND** rotation tokens SHALL be present and valid

#### Scenario: Content target validation

- **WHEN** validating a content recognition record
- **THEN** the prompt SHALL match the block type
- **AND** text targets SHALL contain only recognized text with no JSON wrapping or explanatory prose
- **AND** formula targets SHALL be valid LaTeX strings with no Markdown fences
- **AND** table targets SHALL use only OTSL tokens and SHALL be convertible to valid HTML

#### Scenario: Round-trip validation

- **WHEN** performing round-trip validation on layout records
- **THEN** parsing the assistant target through the inference layout parser SHALL produce valid ContentBlock objects
- **AND** the parsed bbox values SHALL normalize to `[0, 1]` range
- **WHEN** performing round-trip validation on table records
- **THEN** parsing the OTSL target through the OTSL-to-HTML converter SHALL produce a non-empty HTML table

### Requirement: System Prompt Consistency

All training records SHALL use the system prompt `"You are a helpful assistant."` as the first message in the `messages` array, matching the MinerU2.5 inference default.

#### Scenario: System prompt in every record

- **WHEN** constructing any training record
- **THEN** the first element of `messages` SHALL be `{"role": "system", "content": "You are a helpful assistant."}`
- **AND** no other system prompt variant SHALL be used unless a non-standard configuration explicitly documents the deviation

### Requirement: Block Type Pragmatics

Certain block types have special handling rules that SHALL be followed to produce training data compatible with the MinerU2.5 post-processing pipeline.

#### Scenario: Avoiding inline_formula in training

- **WHEN** selecting block types for layout training data
- **THEN** `inline_formula` SHALL be avoided because the parser skips it during post-processing
- **AND** inline formula regions SHALL be absorbed into surrounding text blocks instead

#### Scenario: Preferring image over unknown

- **WHEN** labeling a visual block that contains an image or figure
- **THEN** the block type `image` SHALL be preferred over `unknown`
- **AND** `unknown` SHALL only be used when the content genuinely cannot be classified

#### Scenario: Restricting grouping block types

- **WHEN** labeling block types for standard training
- **THEN** `image_block` and `equation_block` SHALL NOT be used unless the downstream pipeline explicitly expects these grouping constructs
- **AND** `list_item` SHALL be treated as a text-like block type since post-processing normalizes it to text
