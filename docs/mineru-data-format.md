# MinerU2.5 Fine-tuning Dataset Specification

**Derived from `opendatalab/mineru-vl-utils` inference contract**

This spec describes how to construct SFT-style training data for MinerU2.5-compatible fine-tuning. It is based on the public `mineru-vl-utils` API and parser behavior, not on a released official training manifest. The key design is a **two-stage document parsing contract**: first predict page-level layout boxes and block types; then crop each region and run a task-specific recognition prompt. MinerU2.5 itself is described as using this decoupled strategy: global layout detection on downsampled page images, followed by content recognition on native-resolution crops. ([arXiv][1]) The uploaded DocV3R paper also describes the same operational pattern: task-specific prompts for layout, tables, formulas, diagrams, text blocks, seals, captions, and titles; layout detection outputs spatial regions and categories, and content extraction uses cropped regions with specialized prompts. 

---

## 1. Model I/O contract

### 1.1 Inference-level output

`mineru-vl-utils` exposes the final parsed result as a list of `ContentBlock` objects. Each block has:

```python
{
  "type": str,
  "bbox": [xmin, ymin, xmax, ymax],
  "angle": 0 | 90 | 180 | 270 | None,
  "content": str | None,
  "merge_prev": bool
}
```

The README defines `ContentBlock.type`, normalized `bbox`, `angle`, and `content`; `bbox` values are normalized to `[0, 1]`, and `content` is text for text blocks, HTML for tables, LaTeX for equations, and `None` for image blocks. ([GitHub][2]) Internally, the `ContentBlock` constructor validates the type, bbox range/order, angle options, string-or-`None` content, and `merge_prev` constraints. ([GitHub][3])

### 1.2 Training-level output

For fine-tuning, the assistant target should usually be the **raw model output string before post-processing**, not the final `ContentBlock` JSON. This is especially important for:

| Task                         | Prompt                   | Raw target                               | Final `ContentBlock.content`        |
| ---------------------------- | ------------------------ | ---------------------------------------- | ----------------------------------- |
| Layout detection             | `\nLayout Detection:`    | Compact bbox/type/rotation token grammar | Parsed into block list              |
| Text/title/caption/code/etc. | `\nText Recognition:`    | Plain text                               | Text string                         |
| Formula                      | `\nFormula Recognition:` | LaTeX string                             | LaTeX, often later wrapped          |
| Table                        | `\nTable Recognition:`   | OTSL token sequence                      | HTML table after conversion         |
| Image/chart analysis         | `\nImage Analysis:`      | Image/chart analysis string              | Parsed or converted by post-process |

The default prompt mapping in `mineru-vl-utils` is defined as: table → `\nTable Recognition:`, equation → `\nFormula Recognition:`, image/chart → `\nImage Analysis:`, default text → `\nText Recognition:`, and layout → `\nLayout Detection:`. ([GitHub][4])

---

## 2. Chat/message format

### 2.1 Canonical SFT record

Use one JSONL row per image-task pair. A single page will usually produce:

1. one **layout detection** record using the full page image;
2. multiple **content recognition** records, one per cropped region.

Recommended canonical format:

```json
{
  "id": "doc001_p000_layout",
  "task": "layout",
  "images": ["images/pages/doc001_p000.png"],
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "images/pages/doc001_p000.png"},
        {"type": "text", "text": "\nLayout Detection:"}
      ]
    },
    {
      "role": "assistant",
      "content": "<|box_start|>80 71 910 132<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|><|box_start|>95 160 905 762<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>"
    }
  ],
  "metadata": {
    "doc_id": "doc001",
    "page_index": 0,
    "width": 2480,
    "height": 3508,
    "language": "en"
  }
}
```

When these records are materialized as portable VERL view Parquet, the runtime image input should be embedded rather than path-based:

```json
{
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "<image>\nLayout Detection:"}
  ],
  "images": [{"bytes": "<binary PNG or WebP bytes>"}],
  "image_path": "canonical asset path kept only for lineage/debugging"
}
```

Path-backed runtime images remain valid only for explicit cached/path view modes.

`mineru-vl-utils` uses `"You are a helpful assistant."` as the default system prompt. ([GitHub][5]) For Hugging Face `transformers` inference, the client builds chat-template messages with a system message, image content, and text prompt; by default the image is placed before the text unless `<image>` appears explicitly in the prompt. ([GitHub][6])

---

## 3. Layout detection task

### 3.1 Input

Use the full document page image.

```text
\nLayout Detection:
```

At inference time, the utility resizes the page to `1036 × 1036` before layout prediction, but the raw output grammar uses integer coordinates on a `0..1000` logical coordinate grid. The parser then divides by `1000` to produce normalized `ContentBlock.bbox` values. ([GitHub][4])

### 3.2 Raw output grammar

The raw layout output is a concatenation of box records:

```text
<|box_start|>{x1} {y1} {x2} {y2}<|box_end|><|ref_start|>{block_type}<|ref_end|><|rotate_up|>
```

The parser regex expects:

```regex
<|box_start|>(\d+) (\d+) (\d+) (\d+)<|box_end|>
<|ref_start|>(\w+?)<|ref_end|>
(?:(<|rotate_(?:up|right|down|left)|>))?
```

([GitHub][4])

### 3.3 Coordinate rules

Convert pixel-level annotations into layout target coordinates as:

```python
x1_1000 = round(1000 * x1_px / page_width)
y1_1000 = round(1000 * y1_px / page_height)
x2_1000 = round(1000 * x2_px / page_width)
y2_1000 = round(1000 * y2_px / page_height)
```

Rules:

* Coordinates must be integers in `[0, 1000]`.
* `x1 < x2`, `y1 < y2`.
* Clamp after rounding if needed.
* Store final client-side bbox as `[x1/1000, y1/1000, x2/1000, y2/1000]`.

### 3.4 Rotation tokens

Use one of:

```text
<|rotate_up|>     -> 0 degrees
<|rotate_right|>  -> 90 degrees
<|rotate_down|>   -> 180 degrees
<|rotate_left|>   -> 270 degrees
```

The `ContentBlock` angle options are `{None, 0, 90, 180, 270}`. ([GitHub][3]) Although the regex allows the rotation token to be absent, training data should include it consistently.

### 3.5 Supported block types

Use the block types supported by the utility code:

```text
text
title
table
equation
code
algorithm
aside_text
ref_text
phonetic
list_item
table_caption
image_caption
code_caption
table_footnote
image_footnote
header
footer
page_number
page_footnote
image
chart
list
image_block
equation_block
unknown
```

These are the block types defined in `BlockType`. ([GitHub][3]) Practical notes:

* Prefer `image` over `unknown`; the parser maps `unknown` to `image`.
* The parser skips `inline_formula`; avoid it unless you intentionally want such boxes ignored.
* `list_item` is later normalized to text in post-processing.
* `image_block` and `equation_block` are block-level grouping constructs used by post-processing; do not use them unless your downstream pipeline expects them.

### 3.6 Layout target example

```text
<|box_start|>72 45 921 106<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|><|box_start|>79 130 924 242<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|><|box_start|>102 278 900 710<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|><|box_start|>111 736 894 811<|box_end|><|ref_start|>table_caption<|ref_end|><|rotate_up|>
```

---

## 4. Content recognition tasks

After layout detection, `two_step_extract()` crops each detected region, runs a second-stage prompt, and writes the recognition result into `block.content`. ([GitHub][4]) The utility chooses the prompt from the block type, skips certain block types, rotates non-table crops according to the detected angle, optionally masks table regions, and sends each crop to the VLM. ([GitHub][4])

### 4.1 Text-like blocks

Applies to:

```text
text, title, code, algorithm, aside_text, ref_text, phonetic,
list_item, table_caption, image_caption, code_caption,
table_footnote, image_footnote, header, footer, page_number, page_footnote
```

Default input prompt:

```text
\nText Recognition:
```

Raw assistant target:

```text
Quarterly Revenue Analysis
```

For multi-line text:

```text
The consolidated revenue increased by 12.4% year over year.
Operating margin remained stable despite higher infrastructure costs.
```

Rules:

* Output only the recognized textual content.
* Do not wrap the result in JSON.
* Do not add explanations such as “The text is…”.
* Preserve meaningful line breaks.
* Use Unicode text directly.

Example JSONL:

```json
{
  "id": "doc001_p000_b001_text",
  "task": "text",
  "images": ["images/crops/doc001_p000_b001.png"],
  "block_type": "text",
  "bbox": [0.079, 0.130, 0.924, 0.242],
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "images/crops/doc001_p000_b001.png"},
        {"type": "text", "text": "\nText Recognition:"}
      ]
    },
    {
      "role": "assistant",
      "content": "The consolidated revenue increased by 12.4% year over year.\nOperating margin remained stable despite higher infrastructure costs."
    }
  ]
}
```

---

### 4.2 Formula blocks

Applies to:

```text
equation
```

Default input prompt:

```text
\nFormula Recognition:
```

Raw assistant target:

```text
E = mc^2
```

More complex example:

```text
\mathcal{L}_{\mathrm{PPO}}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
```

Rules:

* Output LaTeX only.
* Do not add Markdown fences.
* Do not include natural-language commentary.
* Do not wrap with `$...$` unless your training/inference convention explicitly requires it.
* The post-processing pipeline may later wrap block equations as display equations.

---

### 4.3 Table blocks

Applies to:

```text
table
```

Default input prompt:

```text
\nTable Recognition:
```

Raw assistant target should be **OTSL**, not HTML, for vanilla MinerU2.5-style training. The post-processing code detects OTSL table tokens and converts the raw table output into HTML. ([GitHub][7]) The OTSL converter defines the core table tokens as `<nl>`, `<fcel>`, `<ecel>`, `<lcel>`, `<ucel>`, and `<xcel>`. ([GitHub][8])

#### OTSL token semantics

| Token    | Meaning                                                 |
| -------- | ------------------------------------------------------- |
| `<nl>`   | New table row                                           |
| `<fcel>` | Filled cell; text after the token is the cell content   |
| `<ecel>` | Empty cell                                              |
| `<lcel>` | Horizontal continuation from the left cell              |
| `<ucel>` | Vertical continuation from the upper cell               |
| `<xcel>` | Continuation in both horizontal and vertical directions |

The converter parses rows split by `<nl>`, creates cells for `<fcel>` and `<ecel>`, and uses `<lcel>`, `<ucel>`, and `<xcel>` to infer column/row spans. ([GitHub][8]) It then exports the parsed table as HTML. ([GitHub][8])

#### Simple table example

Visual table:

| Name | Amount |
| ---- | -----: |
| A    |     10 |
| B    |     20 |

Raw target:

```text
<fcel>Name<fcel>Amount<nl><fcel>A<fcel>10<nl><fcel>B<fcel>20<nl>
```

#### Empty-cell example

```text
<fcel>Metric<fcel>Q1<fcel>Q2<nl><fcel>Revenue<fcel>12.0<fcel>13.5<nl><fcel>Margin<ecel><fcel>18%<nl>
```

#### Merged-cell example

A header spanning two columns can be represented using a filled cell followed by a horizontal continuation:

```text
<fcel>Financial Summary<lcel><nl><fcel>Revenue<fcel>Profit<nl><fcel>120<fcel>31<nl>
```

#### Table JSONL example

```json
{
  "id": "doc001_p000_b002_table",
  "task": "table",
  "images": ["images/crops/doc001_p000_b002.png"],
  "block_type": "table",
  "bbox": [0.102, 0.278, 0.900, 0.710],
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "images/crops/doc001_p000_b002.png"},
        {"type": "text", "text": "\nTable Recognition:"}
      ]
    },
    {
      "role": "assistant",
      "content": "<fcel>Name<fcel>Amount<nl><fcel>A<fcel>10<nl><fcel>B<fcel>20<nl>"
    }
  ],
  "postprocess_target": {
    "content_format": "html",
    "content": "<table><tbody><tr><td>Name</td><td>Amount</td></tr><tr><td>A</td><td>10</td></tr><tr><td>B</td><td>20</td></tr></tbody></table>"
  }
}
```

---

### 4.4 Image and chart blocks

Applies to:

```text
image, chart
```

Default input prompt:

```text
\nImage Analysis:
```

By default, the second-stage extraction code skips `image` and `chart` blocks unless `image_analysis=True`. ([GitHub][4]) The image/chart post-processing path can classify image-analysis output into image, chart, table, or equation-like results, and can convert pure table or formula detections into corresponding block types. ([GitHub][7])

For fine-tuning MinerU2.5’s standard document OCR behavior, image/chart records can be omitted initially. Add them only when the target application needs figure description, chart parsing, seal recognition, diagram extraction, or similar non-text visual understanding.

---

## 5. Recommended dataset directory

```text
mineru25_sft_dataset/
  train.jsonl
  val.jsonl
  test.jsonl
  images/
    pages/
      doc001_p000.png
      doc001_p001.png
    crops/
      doc001_p000_b000_title.png
      doc001_p000_b001_text.png
      doc001_p000_b002_table.png
      doc001_p000_b003_equation.png
  annotations/
    source_layout.json
    source_tables_html/
    source_formulas/
  README.md
```

Recommended JSONL fields:

```json
{
  "id": "string, globally unique",
  "task": "layout | text | table | equation | image_analysis | chart_analysis",
  "images": ["relative/path.png in source JSONL; [{\"bytes\": <binary>}] in portable VERL view Parquet"],
  "block_type": "optional block type for crop-level records",
  "bbox": "optional normalized bbox for crop-level records",
  "angle": "optional 0|90|180|270|null",
  "messages": "chat-format SFT messages",
  "target": "optional duplicate of assistant content for easier loading",
  "postprocess_target": "optional final ContentBlock-style expected result",
  "metadata": {
    "doc_id": "string",
    "page_index": 0,
    "source": "synthetic | human | mineru_assisted | public_dataset",
    "language": "en | zh | mixed | ...",
    "width": 2480,
    "height": 3508,
    "quality": "gold | silver | synthetic",
    "split": "train"
  }
}
```

The `messages` field should be the authoritative SFT input/output. The `target` field is redundant but convenient for validators and non-chat trainers.

---

## 6. Annotation-to-training conversion rules

### 6.1 Page layout annotations → layout records

Input annotation:

```json
{
  "page_image": "images/pages/doc001_p000.png",
  "width": 2480,
  "height": 3508,
  "blocks": [
    {
      "type": "title",
      "bbox_px": [178, 158, 2284, 372],
      "angle": 0
    },
    {
      "type": "table",
      "bbox_px": [253, 975, 2232, 2491],
      "angle": 0
    }
  ]
}
```

Output target:

```text
<|box_start|>72 45 921 106<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|><|box_start|>102 278 900 710<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>
```

### 6.2 Block annotations → crop recognition records

For each block:

1. Crop from the original page using the pixel bbox.
2. Rotate the crop to upright orientation for non-table blocks if your data pipeline follows `prepare_for_extract`.
3. Choose prompt by block type.
4. Store the raw recognition target.

Prompt mapping:

```python
PROMPTS = {
    "layout": "\nLayout Detection:",
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "image": "\nImage Analysis:",
    "chart": "\nImage Analysis:",
    "default": "\nText Recognition:"
}
```

### 6.3 Table annotations → OTSL targets

For standard MinerU2.5-style training, convert table ground truth to OTSL. Keep HTML as optional validation or postprocess target, but do not use HTML as the raw assistant target unless you intentionally modify the prompt and post-processing path.

### 6.4 Formula annotations → LaTeX targets

Store only the formula body:

```text
\frac{\partial \mathcal{L}}{\partial \theta}
```

Avoid:

```text
The formula is: \frac{\partial \mathcal{L}}{\partial \theta}
```

Avoid Markdown:

````text
```latex
\frac{\partial \mathcal{L}}{\partial \theta}
```
````

---

## 7. Validation checklist

Before training, run strict validation.

### 7.1 Layout validation

For every layout target:

* All records match the box grammar.
* All coordinates are integers in `[0, 1000]`.
* `x1 < x2`, `y1 < y2`.
* `block_type` is in the supported block type list.
* Rotation token is present and valid.
* No accidental spaces inside special tokens.
* Reading order is stable and deterministic.

### 7.2 Content validation

For crop-level records:

* Prompt matches block type.
* Text targets contain only the extracted content.
* Formula targets are valid LaTeX strings.
* Table targets use OTSL tokens and can be converted to HTML.
* No assistant target contains explanatory prose unless the task explicitly requires analysis.
* No Markdown fences around OTSL or LaTeX.
* Images are readable and paths are valid.

### 7.3 Round-trip validation

Recommended validators:

```python
# Pseudocode
for sample in jsonl:
    if sample["task"] == "layout":
        blocks = parse_layout_output(sample["messages"][-1]["content"])
        assert all_valid_content_blocks(blocks)

    if sample["task"] == "table":
        html = convert_otsl_to_html(sample["messages"][-1]["content"])
        assert html is not None
```

---

## 8. Optional extensions for MinerU2.5-derived research

Vanilla `mineru-vl-utils` expects standard OTSL for tables and does not expose a native diagram DSL training contract. For recursive tables or diagram extraction, extend both the prompt set and post-processing.

For recursive tables, one practical extension is enhanced OTSL with `<nest_start>` and `<nest_end>`, where the nested content must itself be a valid OTSL sequence. The uploaded paper describes this as a backward-compatible extension for nested table structures. 

Example:

```text
<fcel>Names<fcel>Amount<nl><fcel>Item 1<fcel><nest_start><fcel>Region<fcel>Amount<nl><fcel>CA<fcel>10<nl><fcel>NY<fcel>9<nl><nest_end><nl><fcel>Item 2<fcel>10<nl>
```

For diagrams, the same paper proposes D-Mermaid, extending Mermaid with tabular syntax and control tags for alignment, symmetry, and grouping.  Treat this as a new task family, for example:

```text
\nDiagram Recognition:
```

with a raw target such as:

```text
graph TD<nl>A[Input]<nl>B[Encoder]<nl>C[Decoder]<nl>A-->B<nl>B-->C
```

This requires custom parser/evaluator integration and should not be mixed into vanilla MinerU2.5 table/text/formula records without explicit prompt and post-process changes.

---

## 9. Minimal viable training set

A practical first fine-tuning dataset should include:

```text
layout records:
  full page image -> layout grammar

text records:
  text/title/caption crops -> plain text

table records:
  table crops -> OTSL

equation records:
  formula crops -> LaTeX
```

Recommended initial mixture:

| Task                 | Suggested share |
| -------------------- | --------------: |
| Layout detection     |          25–35% |
| Text recognition     |          30–40% |
| Table recognition    |          20–30% |
| Formula recognition  |           5–15% |
| Image/chart analysis |        Optional |

The most important alignment rule is: **train the model on the same raw strings that the inference parser expects**, not on the final pretty JSON/HTML output. For tables, that means OTSL raw output; for layout, that means the compact `<|box_start|>...<|ref_start|>...<|rotate_*|>` grammar; for formulas, that means LaTeX; for text, that means plain recognized content.

[1]: https://arxiv.org/abs/2509.22186 "[2509.22186] MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing"
[2]: https://github.com/opendatalab/mineru-vl-utils "GitHub - opendatalab/mineru-vl-utils: A Python package for interacting with the MinerU Vision-Language Model. · GitHub"
[3]: https://github.com/opendatalab/mineru-vl-utils/blob/main/mineru_vl_utils/structs.py "mineru-vl-utils/mineru_vl_utils/structs.py at main · opendatalab/mineru-vl-utils · GitHub"
[4]: https://github.com/opendatalab/mineru-vl-utils/blob/main/mineru_vl_utils/mineru_client.py "mineru-vl-utils/mineru_vl_utils/mineru_client.py at main · opendatalab/mineru-vl-utils · GitHub"
[5]: https://github.com/opendatalab/mineru-vl-utils/blob/main/mineru_vl_utils/vlm_client/base_client.py "mineru-vl-utils/mineru_vl_utils/vlm_client/base_client.py at main · opendatalab/mineru-vl-utils · GitHub"
[6]: https://github.com/opendatalab/mineru-vl-utils/blob/main/mineru_vl_utils/vlm_client/transformers_client.py "mineru-vl-utils/mineru_vl_utils/vlm_client/transformers_client.py at main · opendatalab/mineru-vl-utils · GitHub"
[7]: https://github.com/opendatalab/mineru-vl-utils/blob/main/mineru_vl_utils/post_process/__init__.py "mineru-vl-utils/mineru_vl_utils/post_process/__init__.py at main · opendatalab/mineru-vl-utils · GitHub"
[8]: https://github.com/opendatalab/mineru-vl-utils/blob/main/mineru_vl_utils/post_process/otsl2html.py "mineru-vl-utils/mineru_vl_utils/post_process/otsl2html.py at main · opendatalab/mineru-vl-utils · GitHub"
