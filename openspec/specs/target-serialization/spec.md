# Target Serialization Specification

## Purpose

Target serializers convert canonical task-record targets into model-specific label
strings. Each serializer is a named, versioned adapter bound to a single task.
The view builder selects a serializer per task via the view configuration
(`target_serialization` mapping) and calls `serialize()` during view
materialization.
## Requirements
### Requirement: TargetSerializer Abstract Base Interface

Every serializer SHALL subclass `TargetSerializer` and implement the five members:
`name` (unique registry key, e.g. `mineru_layout_box_v1`), `version` (semantic
version), `task` (canonical task name), `serialize(canonical_record, context) -> str`,
and `validate(canonical_record)`. The base `validate` SHALL reject records whose
`task` field does not equal the serializer's `task`.

#### Scenario: Subclass exposes required attributes

- **GIVEN** a concrete serializer subclass of `TargetSerializer`
- **WHEN** its `name`, `version`, and `task` attributes are inspected
- **THEN** each attribute SHALL return a non-empty string

#### Scenario: Base validate rejects task mismatch

- **GIVEN** a serializer with `task = "layout"`
- **AND** a canonical record with `task = "table"`
- **WHEN** `validate(canonical_record)` is called
- **THEN** it SHALL raise `ValueError`

### Requirement: Serialization Context Contract

The `context` dict passed to `serialize()` SHALL contain the keys
`image_transform`, `coordinate_space`, `model_family`, `target_format`, `width`,
and `height`. Serializers that do not need image geometry MAY ignore the transform
keys but SHALL NOT raise on their presence.

#### Scenario: Context provides image geometry metadata

- **GIVEN** a context dict built by the view builder
- **WHEN** the layout serializer reads `context["image_transform"]`
- **THEN** it SHALL find `scale`, `pad_left`, `pad_top`, `output_width`, and
  `output_height` values

#### Scenario: Non-geometry serializer ignores transform keys

- **GIVEN** a text serializer receiving a context with `image_transform` populated
- **WHEN** `serialize()` is called
- **THEN** the serializer SHALL return the text label without error

### Requirement: Layout Serializer Coordinate Transformation

`MinerULayoutSerializer` (`mineru_layout_box_v1`) SHALL read each element's
`bbox` (pixel xyxy), apply the affine transform from
`context["image_transform"]` (scale by `scale_x`/`scale_y`, offset by
`pad_left`/`pad_top`), convert to an integer grid in `[0, 1000]` relative to the
output image dimensions using `floor(1000 * value / size)`, and emit one element
per region in MinerU box format.

#### Scenario: Single element with scale and padding

- **GIVEN** a canonical layout record whose target contains one element with
  `bbox: [100, 200, 500, 700]`, `label: "table"`, `rotation: 0`
- **AND** a context with `width: 1000`, `height: 1000`,
  `image_transform: {scale: 0.5, pad_left: 0, pad_top: 0}`
- **WHEN** the layout serializer calls `serialize()`
- **THEN** the output SHALL contain exactly one MinerU box string with grid
  coordinates derived from the halved pixel values

#### Scenario: Multiple elements preserve input order

- **GIVEN** a canonical layout record with three elements in the order title,
  text block, and figure
- **WHEN** the layout serializer calls `serialize()`
- **THEN** the output SHALL concatenate the three MinerU box strings in the same
  order with no separator between them

#### Scenario: Grid values are clamped to zero through one thousand

- **GIVEN** a layout element whose transformed pixel coordinates would map outside
  the `[0, 1000]` range
- **WHEN** the grid function is applied
- **THEN** each coordinate SHALL be clamped to `[0, 1000]` before emission

### Requirement: Layout Serializer Rotation Token Mapping

The layout serializer SHALL map rotation degrees to rotation tokens: 0 to
`<|rotate_up|>`, 90 to `<|rotate_right|>`, 180 to `<|rotate_down|>`, 270 to
`<|rotate_left|>`. Unknown or missing rotations SHALL default to
`<|rotate_up|>`.

#### Scenario: Element with two hundred seventy degree rotation

- **GIVEN** a layout element with `rotation: 270`
- **WHEN** the layout serializer produces the box string
- **THEN** the output SHALL include `<|rotate_left|>`

#### Scenario: Element with unknown rotation value

- **GIVEN** a layout element with `rotation: 45`
- **WHEN** the layout serializer produces the box string
- **THEN** the output SHALL include `<|rotate_up|>` as the default

### Requirement: Table Serializer Enhanced OTSL

`EnhancedOTSLSerializer` (`enhanced_otsl_v1`) SHALL produce an OTSL string from
the canonical table target using this fallback chain: if `target.enhanced_otsl`
is present return it directly; else if `target.otsl` is present return it
directly; else if `target.html` is present convert to OTSL via `html_to_otsl`.
If none of the three fields exist, the serializer SHALL raise `ValueError`.

#### Scenario: Serialization from pre-computed OTSL

- **GIVEN** a canonical table record whose target contains
  `otsl: "<fcel>X<fcel>Y<nl>"`
- **WHEN** the table serializer calls `serialize()`
- **THEN** it SHALL return `<fcel>X<fcel>Y<nl>` directly without HTML conversion

#### Scenario: Serialization from HTML source

- **GIVEN** a canonical table record whose target contains
  `html: "<table><tr><td>A</td><td>B</td></tr></table>"`
- **WHEN** the table serializer calls `serialize()`
- **THEN** it SHALL convert via `html_to_otsl` and return the OTSL string

#### Scenario: Missing all table fields raises error

- **GIVEN** a canonical table record whose target has no `enhanced_otsl`, `otsl`,
  or `html` fields
- **WHEN** the table serializer calls `serialize()`
- **THEN** it SHALL raise `ValueError`

#### Scenario: Table with horizontal merge uses lcel token

- **GIVEN** an HTML table with `colspan="2"` on one cell
- **WHEN** the table serializer converts to OTSL
- **THEN** the output SHALL use `<lcel>` for the spanned position and `<fcel>`
  with text for the origin cell

### Requirement: Formula Serializer LaTeX Pass-Through

`LatexPlainSerializer` (`latex_plain_v1`) SHALL extract `target.latex` and return
it as a string without transformation or validation of LaTeX syntax. If the
field is missing, it SHALL raise `ValueError`.

#### Scenario: LaTeX formula returned unchanged

- **GIVEN** a canonical formula record whose target contains
  `latex: "\\frac{a}{b}"`
- **WHEN** the formula serializer calls `serialize()`
- **THEN** it SHALL return `\frac{a}{b}`

#### Scenario: Missing LaTeX field raises error

- **GIVEN** a canonical formula record whose target has no `latex` field
- **WHEN** the formula serializer calls `serialize()`
- **THEN** it SHALL raise `ValueError`

### Requirement: Text Serializer Plain Text

`PlainTextSerializer` (`plain_text_v1`) SHALL extract `target.text` and return it
as a string with no normalization of whitespace, case, or Unicode. If the field
is missing, it SHALL raise `ValueError`.

#### Scenario: Plain text returned without normalization

- **GIVEN** a canonical text record whose target contains `text: "Hello world"`
- **WHEN** the text serializer calls `serialize()`
- **THEN** it SHALL return `Hello world`

#### Scenario: Missing text field raises error

- **GIVEN** a canonical text record whose target has no `text` field
- **WHEN** the text serializer calls `serialize()`
- **THEN** it SHALL raise `ValueError`

### Requirement: Registry and Discovery

All built-in serializers SHALL be registered in the default serializer registry
via `default_serializer_registry()`. The registry SHALL map serializer `name` to
a singleton `TargetSerializer` instance. The view builder SHALL resolve a
serializer by looking up the `target_serialization` mapping from the view config
and calling `registry.get(name)`.

#### Scenario: Registry lookup by serializer name

- **GIVEN** the default serializer registry
- **WHEN** `registry.get("enhanced_otsl_v1")` is called
- **THEN** it SHALL return a singleton `EnhancedOTSLSerializer` instance

#### Scenario: Registry lookup for unregistered name raises error

- **GIVEN** the default serializer registry
- **WHEN** `registry.get("nonexistent_serializer")` is called
- **THEN** it SHALL raise `KeyError`

### Requirement: Deterministic Output Ordering

Given the same `canonical_record` and `context`, `serialize()` SHALL return the
same string. Element ordering in layout output SHALL follow the order of
`target.elements`. Table OTSL output SHALL follow left-to-right, top-to-bottom
cell order.

#### Scenario: Repeated serialization produces identical output

- **GIVEN** a canonical layout record and a fixed context
- **WHEN** `serialize()` is called twice with the same inputs
- **THEN** both calls SHALL return byte-identical strings

#### Scenario: Layout elements follow input ordering

- **GIVEN** a canonical layout record with elements ordered title, text, figure
- **WHEN** the layout serializer calls `serialize()`
- **THEN** the output SHALL list title first, text second, and figure third

### Requirement: Null and Malformed Input Error Handling

`serialize()` SHALL raise `ValueError` -- not return an empty string, `None`, or
a placeholder -- when the canonical record's `task` does not match the
serializer's `task`, when required target fields are missing, or when layout
context lacks `width`/`height` with no fallback in `canonical_record.metadata`.

#### Scenario: Empty canonical target is rejected

- **GIVEN** a canonical record with `target: {}` containing no fields
- **WHEN** any serializer calls `serialize()`
- **THEN** it SHALL raise `ValueError` indicating the required field that is
  missing

#### Scenario: Layout with missing page dimensions raises error

- **GIVEN** a layout serializer and a context with no `width`, `height`, and no
  `metadata.width`/`metadata.height` on the record
- **WHEN** `serialize()` is called
- **THEN** it SHALL raise `ValueError` indicating that page dimensions are
  required

#### Scenario: Task mismatch rejection

- **GIVEN** a serializer with `task = "layout"`
- **AND** a canonical record with `task = "table"`
- **WHEN** `serialize()` is called
- **THEN** it SHALL raise `ValueError`

### Requirement: Version Stability

The `name` and `version` pair SHALL uniquely identify the output format. Once a
serializer version is in use by exported views, its output format SHALL NOT
change. Format changes SHALL require a new `name` or `version`. View records
SHALL carry `target_format` equal to the serializer `name`, enabling audits that
tie a label string back to the serialization code.

#### Scenario: Existing version output is invariant

- **GIVEN** a serializer version `1.0.0` that has been used in exported views
- **WHEN** a developer modifies the serialization logic
- **THEN** the change SHALL be released under a new version number

#### Scenario: View record carries target format for audit

- **GIVEN** a materialized view record produced by `mineru_layout_box_v1`
- **WHEN** the view record is inspected
- **THEN** its `target_format` field SHALL equal `mineru_layout_box_v1`

### Requirement: Plain Table Text Serialization

The target serialization system SHALL support table records whose canonical target contains plain recognized table text.

#### Scenario: Serialize UniRec table text

- GIVEN a canonical table record whose target contains `text: "项目\t金额\n收入\t100"`
- WHEN `table_text_v1` serializes the record
- THEN the serializer SHALL return the target text unchanged

#### Scenario: Reject missing table text

- GIVEN a canonical table record whose target lacks a `text` field
- WHEN `table_text_v1` serializes the record
- THEN the serializer SHALL fail with a clear error

