# Processing Module and CLI Specification

Version: `v0.1`
Scope: Dataset processing implementation, source-to-canonical export, canonical-to-view export, SFT/RLVR view construction, reward adapter design, validation, and CLI.

---

## 1. Design Objective

This document extracts the code-related parts from the dataset format design and defines the Processing Module and CLI architecture.

The module must support:

1. Exporting heterogeneous source datasets into the canonical dataset format.
2. Building training-ready views for both **SFT** and **RL post-training**.
3. Supporting current RLVR based on **Normalized Levenshtein Distance**.
4. Supporting future task-specific rewards, such as:

   * TEDS for table recognition.
   * CDM for formula recognition.
   * IoU / mAP-style rewards for layout detection.
   * Graph similarity rewards for diagrams.
   * Rotation-aware text rewards for seals.
5. Supporting reward payload construction through adapters and configuration files.
6. Maintaining lineage from view records back to canonical records, image assets, and source annotations.

The design should avoid over-engineering. The recommended approach is a **registry + adapter + declarative configuration** pattern.

---

# 2. Key Design Principle

The Processing Module should separate four concerns:

```text
Data conversion      Source → Canonical
View materialization Canonical → View
Target serialization Canonical target → model-specific label string
Reward preparation   Canonical/view record → reward payload
```

Therefore:

> **A view record should not hard-code reward logic. It should reference a reward profile and carry the minimum payload required by that profile.**

This is necessary because SFT labels and initial RL labels may be the same under Levenshtein reward, but future RLVR rewards may require additional structured information beyond the plain label.

For example, a table reward based on TEDS may need canonical HTML or table-cell structure, while a formula reward based on CDM may need LaTeX plus rendering configuration. OmniDocBench evaluates different document components with different metrics: normalized edit distance for text, TEDS plus normalized edit distance for tables, and CDM plus normalized edit distance and BLEU for formulas. ([arXiv][1])

---

# 3. Lessons from RLVR-oriented OCR Training

The design should explicitly support reward-oriented training data, not only imitation data.

olmOCR2 is a useful reference because it trains an OCR-specialized VLM with RLVR using binary unit tests as verifiable rewards. Its synthetic pipeline generates documents with known HTML source and extracts test cases; GRPO is then applied using those test cases as binary-valued reward signals. ([arXiv][2])

The relevant takeaway is not that this project must copy binary unit tests immediately. The takeaway is:

> **RL views may require verifier-specific payloads, not just labels.**

For example, olmOCR-Bench test cases include text presence, text absence, natural reading order, table cell position checks, visual formula rendering checks, and baseline robustness checks. ([arXiv][2]) During training, each completion is scored by test cases, and the reward is the fraction of passing tests from `0.0` to `1.0`. ([arXiv][2])

This strongly motivates a processing architecture where `reward_profile` and `reward_payload` are first-class concepts.

---

# 4. Package Layout

Recommended package name:

Recommended internal layout:

```text
scr/data/
  cli.py

  config/
    loader.py
    resolver.py
    schema.py

  registry/
    source_registry.py
    task_registry.py
    serializer_registry.py
    reward_registry.py
    image_profile_registry.py

  sources/
    adapters/
      base.py
      docbank.py
      doclaynet.py
      mineru.py
      omnidocbench.py
      synthetic.py

  canonical/
    writer.py
    reader.py
    normalizer.py
    validator.py
    manifest.py
    schemas/

  assets/
    renderer.py
    cropper.py
    transformer.py
    manifest.py
    hash.py

  views/
    builder.py
    selector.py
    sampler.py
    splitter.py
    writer.py
    validator.py

  serializers/
    base.py
    layout_mineru.py
    table_otsl.py
    table_html.py
    formula_latex.py
    text_plain.py
    diagram_mermaid.py
    seal_text.py

  rewards/
    base.py
    levenshtein.py
    teds.py
    cdm.py
    layout_iou.py
    diagram_graph.py
    seal_circular.py
    unit_tests.py

  runtime/
    verl_export.py
    batch_reader.py
    collator_hints.py

  lineage/
    resolver.py
    trace.py

  stats/
    profiler.py
    report.py

  utils/
    io.py
    parquet.py
    json.py
    logging.py
    multiprocessing.py
```

---

# 5. Core Abstractions

## 5.1 Source Adapter

A `SourceAdapter` converts source-specific raw data into canonical entities, canonical records, and canonical assets.

```python
from abc import ABC, abstractmethod
from typing import Iterable

class SourceAdapter(ABC):
    name: str
    version: str

    @abstractmethod
    def scan_documents(self) -> Iterable[dict]:
        """Return source-level document descriptors."""

    @abstractmethod
    def export_documents(self) -> Iterable[dict]:
        """Export canonical document entities."""

    @abstractmethod
    def export_pages(self) -> Iterable[dict]:
        """Export canonical page entities and page image assets."""

    @abstractmethod
    def export_regions(self) -> Iterable[dict]:
        """Export canonical region entities and crop assets."""

    @abstractmethod
    def export_task_records(self, task: str) -> Iterable[dict]:
        """Export canonical task records for a specific task."""
```

Adapters should be thin and source-specific. They should not know anything about SFT, RLVR, VERL, or model prompts.

---

## 5.2 Canonical Writer

The canonical writer persists normalized entities, records, manifests, and asset references.

```python
class CanonicalWriter:
    def write_documents(self, records, source_name: str) -> None:
        ...

    def write_pages(self, records, source_name: str) -> None:
        ...

    def write_regions(self, records, source_name: str) -> None:
        ...

    def write_task_records(self, task: str, source_name: str, records) -> None:
        ...

    def write_manifest(self, manifest: dict) -> None:
        ...
```

Recommended behavior:

* Write partitioned Parquet files.
* Use deterministic file names or `part-xxxxx.parquet`.
* Never modify unrelated task/source partitions.
* Support overwrite mode per partition.
* Maintain source and task manifests.

---

## 5.3 Asset Manager

The asset manager handles rendered page images, crops, resized view images, checksums, and transform metadata.

```python
class AssetManager:
    def render_page(self, document, page_index: int, profile: dict) -> dict:
        """Render PDF page into canonical page image asset."""

    def crop_region(self, page_asset, bbox, profile: dict) -> dict:
        """Create canonical crop asset."""

    def transform_for_view(self, image_asset, profile: dict) -> dict:
        """Create optional view-specific image asset."""

    def write_asset_manifest(self, entries) -> None:
        ...
```

The asset manager must record:

* Parent asset ID.
* Transform specification hash.
* Output width and height.
* Output path.
* Checksum.
* Coordinate mapping.

This is critical for layout labels because image resizing changes coordinate systems.

---

## 5.4 View Builder

The view builder materializes SFT or RLVR training records.

```python
class ViewBuilder:
    def build(self, view_config: dict) -> None:
        records = self.load_canonical_partitions(view_config)
        records = self.select(records, view_config)
        records = self.split(records, view_config)
        records = self.sample(records, view_config)
        records = self.materialize(records, view_config)
        self.write(records, view_config)

    def materialize(self, records, view_config: dict):
        ...
```

The view builder should coordinate:

* Record selection.
* Sampling.
* Split assignment.
* Image preparation.
* Prompt rendering.
* Target serialization.
* Reward payload preparation.
* Parquet writing.

It should delegate task-specific logic to serializers and reward adapters.

---

# 6. Serializer Adapter Design

## 6.1 Purpose

A serializer converts a canonical target into a model-specific label string.

Examples:

| Task    | Canonical Target         | Serializer             | Output               |
| ------- | ------------------------ | ---------------------- | -------------------- |
| Layout  | bbox/label/rotation JSON | `mineru_layout_box_v1` | MinerU box string    |
| Table   | HTML/cells/OTSL          | `enhanced_otsl_v1`     | Enhanced OTSL string |
| Formula | LaTeX field              | `latex_plain_v1`       | LaTeX string         |
| Text    | Text field               | `plain_text_v1`        | Text string          |
| Diagram | Graph structure          | `d_mermaid_v1`         | D-Mermaid string     |
| Seal    | Segments                 | `seal_text_v1`         | Normalized seal text |

---

## 6.2 Interface

```python
from abc import ABC, abstractmethod

class TargetSerializer(ABC):
    name: str
    version: str
    task: str

    @abstractmethod
    def serialize(self, canonical_record: dict, context: dict) -> str:
        """Convert canonical target into model-specific label string."""

    def validate(self, canonical_record: dict) -> None:
        """Optional task-specific validation."""
```

The `context` should include:

```python
{
    "image_transform": {...},
    "coordinate_space": "view_image_pixel_xyxy",
    "model_family": "mineru2.5",
    "target_format": "mineru_box_string_v1"
}
```

For layout detection, the serializer must transform canonical coordinates into view image coordinates if the view image is resized or padded.

---

## 6.3 Example: MinerU Layout Serializer

Input canonical target:

```json
{
  "elements": [
    {
      "bbox": [100, 200, 500, 700],
      "label": "table",
      "rotation": 0
    }
  ],
  "coordinate_space": "canonical_page_pixel_xyxy"
}
```

If the view transform is:

```json
{
  "scale": 0.5,
  "pad_left": 0,
  "pad_top": 0
}
```

Output label:

```text
<|box_start|>50 100 250 350<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>
```

---

# 7. Reward Adapter Design

## 7.1 Purpose

A reward adapter computes a reward score from:

```text
model prediction + reward payload + optional runtime context
```

The reward adapter must not depend on the training framework. VERL should call it through a thin wrapper.

---

## 7.2 Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class RewardResult:
    score: float
    normalized_score: float
    passed: Optional[bool]
    details: dict
    error: Optional[str] = None

class RewardAdapter(ABC):
    name: str
    version: str
    task: str

    @abstractmethod
    def prepare_payload(self, canonical_record: dict, view_record: dict, config: dict) -> dict:
        """Prepare reward-specific ground truth and metadata."""

    @abstractmethod
    def score(self, prediction: str, payload: dict, context: dict) -> RewardResult:
        """Compute reward for one prediction."""

    def batch_score(self, predictions: list[str], payloads: list[dict], context: dict) -> list[RewardResult]:
        return [self.score(p, q, context) for p, q in zip(predictions, payloads)]

    def validate_payload(self, payload: dict) -> None:
        """Check reward payload integrity."""
```

---

## 7.3 Reward Payload Principle

The view record should contain either:

1. An inline `reward_payload` for small payloads.
2. A `reward_payload_path` for large payloads.
3. A `reward_payload_id` pointing to a sidecar Parquet/JSONL store.

Recommended default:

```text
Small payload: inline JSON string in Parquet
Large payload: sidecar JSONL/Parquet under views/<view>/reward_payloads/
```

---

## 7.4 Reward Result Contract

Every reward adapter should return a normalized score in `[0, 1]`.

```json
{
  "score": 0.873,
  "normalized_score": 0.873,
  "passed": null,
  "details": {
    "metric": "normalized_levenshtein",
    "edit_distance": 12,
    "target_length": 94
  },
  "error": null
}
```

For binary unit tests:

```json
{
  "score": 0.8,
  "normalized_score": 0.8,
  "passed": null,
  "details": {
    "num_tests": 10,
    "num_passed": 8,
    "failed_tests": ["table_cell_position_03", "formula_render_02"]
  },
  "error": null
}
```

---

# 8. Built-in Reward Adapters

## 8.1 Levenshtein Reward

Initial implementation.

```yaml
reward_profile:
  name: normalized_levenshtein_v1
  adapter: levenshtein
  tasks:
    - layout
    - text
    - table
    - formula
    - diagram
    - seal
  params:
    normalize_by: max_length
    case_sensitive: true
    strip_whitespace: false
    score_transform: similarity
```

Formula:

```text
reward = 1 - edit_distance(prediction, label) / max(len(prediction), len(label), 1)
```

Payload:

```json
{
  "label": "<ground_truth_string>",
  "normalization": {
    "case_sensitive": true,
    "strip_whitespace": false
  }
}
```

This reward is simple, stable, and easy to debug. It is appropriate for the first RLVR pipeline bring-up.

---

## 8.2 TEDS Reward for Tables

Future implementation.

```yaml
reward_profile:
  name: table_teds_v1
  adapter: teds
  tasks:
    - table
  params:
    prediction_format: html
    ground_truth_format: html
    normalize_html: true
    invalid_prediction_score: 0.0
```

Payload:

```json
{
  "ground_truth": {
    "html": "<table>...</table>",
    "cells": [...]
  },
  "options": {
    "normalize_html": true,
    "ignore_caption": false
  }
}
```

Use cases:

* HTML table prediction.
* OTSL prediction converted to HTML before scoring.
* Enhanced OTSL prediction converted to a table tree before scoring.

Best practice:

> TEDS reward should be implemented behind a table-normalization layer. Do not make the reward adapter directly depend on one model’s table output format.

---

## 8.3 CDM Reward for Formulas

Future implementation.

CDM is appropriate because formula recognition has many textually different but visually equivalent LaTeX representations. The CDM paper proposes an image-level metric that renders predicted and ground-truth LaTeX into formula images and performs spatially aware character-level matching. ([arXiv][3])

```yaml
reward_profile:
  name: formula_cdm_v1
  adapter: cdm
  tasks:
    - formula
  params:
    render_engine: python_katex
    invalid_latex_score: 0.0
    cache_rendered_formula: true
    timeout_seconds: 2
```

Payload:

```json
{
  "ground_truth": {
    "latex": "\\frac{a}{b}"
  },
  "rendering": {
    "engine": "python_katex",
    "font_size": 20,
    "dpi": 200
  }
}
```

Best practice:

* Cache rendered ground-truth formula images.
* Time-limit prediction rendering.
* Return `0.0` for invalid LaTeX unless a fallback string metric is explicitly configured.
* Record rendering errors in `RewardResult.details`.

---

## 8.4 Layout IoU Reward

Future implementation.

```yaml
reward_profile:
  name: layout_iou_f1_v1
  adapter: layout_iou
  tasks:
    - layout
  params:
    iou_threshold: 0.5
    class_sensitive: true
    score: f1
```

Payload:

```json
{
  "ground_truth": {
    "elements": [
      {
        "bbox": [100, 120, 500, 180],
        "label": "title",
        "rotation": 0
      }
    ],
    "coordinate_space": "view_image_pixel_xyxy"
  }
}
```

Important:

* The payload must use the same coordinate space as the model output.
* If the image was resized in the view, ground-truth boxes must already be transformed.

---

## 8.5 Diagram Graph Reward

Future implementation.

```yaml
reward_profile:
  name: diagram_graph_v1
  adapter: diagram_graph
  tasks:
    - diagram
  params:
    node_text_metric: normalized_levenshtein
    edge_metric: f1
    node_matcher: hungarian
```

Payload:

```json
{
  "ground_truth": {
    "nodes": [...],
    "edges": [...],
    "groups": [...],
    "blocks": [...]
  },
  "accepted_formats": [
    "d_mermaid",
    "mermaid",
    "json_graph"
  ]
}
```

Best practice:

* Convert model output into a normalized graph before scoring.
* Use robust node matching before evaluating edges.
* Keep parser errors distinguishable from semantic errors.

---

## 8.6 Seal Circular Text Reward

Future implementation.

```yaml
reward_profile:
  name: seal_circular_text_v1
  adapter: seal_circular
  tasks:
    - seal
  params:
    normalize_case: true
    remove_decorative_symbols: true
    circular_word_rotation: true
```

Payload:

```json
{
  "ground_truth": {
    "horizontal_segments": [["APPROVED"]],
    "circular_segments": [["GLOBAL", "FINANCE", "LIMITED"]]
  }
}
```

---

## 8.7 Unit-Test Reward

Future implementation inspired by olmOCR2.

```yaml
reward_profile:
  name: unit_tests_v1
  adapter: unit_tests
  tasks:
    - text
    - table
    - formula
    - reading_order
  params:
    aggregation: mean
    invalid_output_score: 0.0
```

Payload:

```json
{
  "tests": [
    {
      "id": "text_presence_001",
      "type": "text_presence",
      "phrase": "Total Revenue"
    },
    {
      "id": "text_absence_001",
      "type": "text_absence",
      "phrase": "Page 1"
    },
    {
      "id": "table_cell_position_001",
      "type": "table_cell_position",
      "row_text": "Revenue",
      "col_text": "2025",
      "value": "100"
    }
  ]
}
```

This reward type is useful for synthetic data where structured ground truth can be converted into verifier tests. It is also useful when multiple outputs are semantically acceptable but string edit distance is too brittle. olmOCR2 explicitly argues that floating document elements such as tables or figures may lack a single definitive ground-truth representation, and unit tests can treat different but equivalently correct outputs more fairly than edit distance. ([arXiv][2])

---

# 9. View Record Schema for SFT and RLVR

## 9.1 Common View Columns

All views should contain:

| Column                | Required | Description                   |
| --------------------- | -------: | ----------------------------- |
| `id`                  |      Yes | View record ID                |
| `stage`               |      Yes | `sft`, `rlvr`, or `eval`      |
| `task`                |      Yes | Task name                     |
| `image_path`          |      Yes | Lineage/debug image path      |
| `images`              | Optional | VERL-native nested image refs |
| `images_bytes`        | Optional | Embedded runtime image bytes  |
| `images_path`         | Optional | Source-reference filenames    |
| `prompt`              |      Yes | Model input prompt            |
| `label`               |      Yes | Ground-truth label string     |
| `source_name`         |      Yes | Source dataset                |
| `document_id`         |      Yes | Canonical document ID         |
| `page_id`             |      Yes | Canonical page ID             |
| `region_id`           | Optional | Canonical region ID           |
| `canonical_record_id` |      Yes | Link to canonical task record |
| `target_format`       |      Yes | Label serialization format    |
| `prompt_template_id`  |      Yes | Prompt template identifier    |
| `split`               |      Yes | `train`, `val`, `test`        |
| `metadata`            | Optional | JSON metadata                 |

---

## 9.2 SFT View Columns

For SFT, the minimum is:

```text
image_path
images_bytes or images_path
prompt
label
messages
```

Recommended additional columns:

| Column              | Description                     |
| ------------------- | ------------------------------- |
| `messages`          | VERL SFT user/assistant turns    |
| `loss_mask_policy`  | Optional token loss mask policy |
| `max_target_length` | Optional target length control  |
| `difficulty`        | Optional sampling metadata      |

Example:

```json
{
  "id": "view:sft:minneru25:000001",
  "stage": "sft",
  "task": "table",
  "image_path": "canonical/assets/regions/source=DocBank/000001.png",
  "images_bytes": ["<binary PNG or WebP bytes>"],
  "images_path": null,
  "prompt": "<image>\nTable Recognition:",
  "messages": [
    {"role": "user", "content": "<image>\nTable Recognition:"},
    {"role": "assistant", "content": "<fcel>Revenue<fcel>Amount<nl><fcel>2025<fcel>100"}
  ],
  "label": "<fcel>Revenue<fcel>Amount<nl><fcel>2025<fcel>100",
  "canonical_record_id": "table:DocBank:region_000002",
  "target_format": "enhanced_otsl_v1",
  "split": "train"
}
```

---

## 9.3 RLVR View Columns

For RLVR, add reward-related fields:

| Column                | Required | Description                             |
| --------------------- | -------: | --------------------------------------- |
| `reward_profile_id`   |      Yes | Reward profile name/version             |
| `reward_payload`      | Optional | Inline JSON payload                     |
| `reward_payload_id`   | Optional | ID of sidecar payload                   |
| `reward_payload_path` | Optional | Path to sidecar payload                 |
| `answer_key`          | Optional | Usually same as `label` for Levenshtein |
| `verifier_metadata`   | Optional | Reward-specific metadata                |

Example with Levenshtein reward:

```json
{
  "id": "view:rlvr:minneru25:000001",
  "stage": "rlvr",
  "task": "table",
  "image_path": "canonical/assets/regions/source=DocBank/000001.png",
  "images_bytes": ["<binary PNG or WebP bytes>"],
  "images_path": null,
  "prompt": "<image>\nTable Recognition:",
  "label": "<fcel>Revenue<fcel>Amount<nl><fcel>2025<fcel>100",
  "answer_key": "<fcel>Revenue<fcel>Amount<nl><fcel>2025<fcel>100",
  "reward_profile_id": "normalized_levenshtein_v1",
  "reward_payload": {
    "label": "<fcel>Revenue<fcel>Amount<nl><fcel>2025<fcel>100"
  },
  "canonical_record_id": "table:DocBank:region_000002",
  "split": "train"
}
```

Example with TEDS reward:

```json
{
  "id": "view:rlvr:table_teds:000001",
  "stage": "rlvr",
  "task": "table",
  "image_path": "canonical/assets/regions/source=DocBank/000001.png",
  "images_bytes": ["<binary PNG or WebP bytes>"],
  "images_path": null,
  "prompt": "<image>\nTable Recognition:",
  "label": "<table>...</table>",
  "reward_profile_id": "table_teds_v1",
  "reward_payload_id": "payload:table_teds:000001",
  "canonical_record_id": "table:DocBank:region_000002",
  "split": "train"
}
```

---

# 10. Configuration Design

## 10.1 Global Processing Config

Path:

```text
configs/processing.yaml
```

Example:

```yaml
paths:
  dataset_root: /data/docparse_dataset
  source_root: /data/docparse_dataset/sources
  canonical_root: /data/docparse_dataset/canonical
  view_root: /data/docparse_dataset/views

registries:
  source_adapters:
    docbank: docparse_dataset.sources.adapters.docbank.DocBankAdapter
    doclaynet: docparse_dataset.sources.adapters.doclaynet.DocLayNetAdapter
    mineru: docparse_dataset.sources.adapters.mineru.MinerUAdapter
    omnidocbench: docparse_dataset.sources.adapters.omnidocbench.OmniDocBenchAdapter

  serializers:
    mineru_layout_box_v1: docparse_dataset.serializers.layout_mineru.MinerULayoutSerializer
    enhanced_otsl_v1: docparse_dataset.serializers.table_otsl.EnhancedOTSLSerializer
    latex_plain_v1: docparse_dataset.serializers.formula_latex.LatexSerializer

  rewards:
    normalized_levenshtein_v1: docparse_dataset.rewards.levenshtein.NormalizedLevenshteinReward
    table_teds_v1: docparse_dataset.rewards.teds.TEDSReward
    formula_cdm_v1: docparse_dataset.rewards.cdm.CDMReward

execution:
  num_workers: 16
  log_level: INFO
  deterministic: true
```

---

## 10.2 View Config for SFT

```yaml
name: mineru25_sft_v1
stage: sft
model_family: mineru2.5

include:
  - task: layout
    sources: [DocLayNet, M6Doc]

  - task: table
    sources: [DocBank, PubTabNet]

  - task: formula
    sources: [DocBank, UniMER]

task_mixture:
  layout: 0.4
  table: 0.3
  formula: 0.2
  text: 0.1

prompt_templates:
  layout: templates/mineru25/layout.jinja
  table: templates/mineru25/table.jinja
  formula: templates/mineru25/formula.jinja

target_serialization:
  layout: mineru_layout_box_v1
  table: enhanced_otsl_v1
  formula: latex_plain_v1

image_policy:
  materialization:
    mode: embedded

split_policy:
  level: document
  train_ratio: 0.98
  val_ratio: 0.01
  test_ratio: 0.01
  seed: 42
```

---

## 10.3 View Config for Initial RLVR with Levenshtein

```yaml
name: mineru25_rlvr_levenshtein_v1
stage: rlvr
model_family: mineru2.5

include:
  - task: layout
    sources: [DocLayNet, M6Doc]

  - task: table
    sources: [DocBank, PubTabNet]

  - task: formula
    sources: [DocBank, UniMER]

prompt_templates:
  layout: templates/mineru25/layout.jinja
  table: templates/mineru25/table.jinja
  formula: templates/mineru25/formula.jinja

target_serialization:
  layout: mineru_layout_box_v1
  table: enhanced_otsl_v1
  formula: latex_plain_v1

reward_profile:
  default: normalized_levenshtein_v1

reward_payload:
  materialization: inline
  include_label: true

image_policy:
  materialization:
    mode: embedded

shard_policy:
  rows_per_shard: 128
```

In this initial mode, SFT and RLVR labels are essentially the same.

---

## 10.4 View Config for Future Task-Specific RLVR

```yaml
name: mineru25_rlvr_task_rewards_v1
stage: rlvr
model_family: mineru2.5

include:
  - task: table
    sources: [DocBank, PubTabNet]
    where:
      category:
        in: [standard_table, merged_cell_table, borderless_table]

  - task: formula
    sources: [UniMER, OmniDocBench]

  - task: layout
    sources: [DocLayNet, M6Doc]

prompt_templates:
  layout: templates/mineru25/layout.jinja
  table: templates/mineru25/table.jinja
  formula: templates/mineru25/formula.jinja

target_serialization:
  layout: mineru_layout_box_v1
  table: table_html_v1
  formula: latex_plain_v1

reward_profile:
  by_task:
    layout: layout_iou_f1_v1
    table: table_teds_v1
    formula: formula_cdm_v1

reward_payload:
  materialization: sidecar
  sidecar_format: parquet
  path: reward_payloads/
  include_canonical_target: true
  include_view_transform: true

image_policy:
  materialization:
    mode: embedded
```

---

# 11. Reward Payload Store

For non-trivial rewards, use sidecar files:

```text
views/<view_name>/
  train.parquet
  val.parquet
  test.parquet
  reward_payloads/
    train_payloads.parquet
    val_payloads.parquet
    test_payloads.parquet
```

Recommended payload columns:

| Column                   | Description         |
| ------------------------ | ------------------- |
| `reward_payload_id`      | Stable payload ID   |
| `view_record_id`         | Related view record |
| `reward_profile_id`      | Reward adapter ID   |
| `task`                   | Task                |
| `payload`                | JSON payload        |
| `payload_schema_version` | Schema version      |
| `created_by`             | Code version        |
| `config_hash`            | Reward config hash  |

This keeps `train.parquet` compact while allowing rich reward information.

---

# 12. VERL Integration

## 12.1 Data Contract

For VERL, the view should provide columns that can be mapped to the training data loader.

Recommended default:

| VERL Concept     | View Column                             |
| ---------------- | --------------------------------------- |
| Multimodal input | `images`, `images_bytes`, or `images_path` |
| Prompt           | `prompt`                                |
| Ground truth     | `label` or `answer_key`                 |
| Reward config    | `reward_profile_id`                     |
| Reward payload   | `reward_payload` or `reward_payload_id` |

`image_path` is retained for lineage and debugging. `embedded` mode writes `images_bytes: list<binary>`. `source_reference` mode writes dataset-root-relative paths into `images_path`: identity images reference canonical/source assets, while transformed images are saved under `views/<view>/assets/`. `nested_reference` mode writes the original VERL `images: [{"image": "..."}]` shape with the same dataset-root-relative path policy. VERL integration uses the repo-local dataset wrapper to resolve paths against `OCR_DATA_ROOT` before passing image inputs to VERL.

For large embedded-image SFT or RLVR views, write sharded split outputs with:

```yaml
shard_policy:
  rows_per_shard: 128
```

The builder writes `train/part-*.parquet`, `val/part-*.parquet`, and
`test/part-*.parquet` instead of a single split file. VERL SFT reads each file
with `pd.read_parquet(..., dtype_backend="pyarrow")`, so training launch configs
should pass the expanded shard file list rather than one large nested binary
Parquet.

---

## 12.2 Reward Function Wrapper

A thin VERL reward function should:

1. Read generated model output.
2. Read `reward_profile_id`.
3. Load inline or sidecar payload.
4. Dispatch to `RewardRegistry`.
5. Return scalar reward.

Pseudo-code:

```python
class VerlRewardWrapper:
    def __init__(self, reward_registry, payload_store):
        self.reward_registry = reward_registry
        self.payload_store = payload_store

    def __call__(self, data_item, model_output: str) -> float:
        profile_id = data_item["reward_profile_id"]
        adapter = self.reward_registry.get(profile_id)

        payload = data_item.get("reward_payload")
        if payload is None:
            payload = self.payload_store.get(data_item["reward_payload_id"])

        result = adapter.score(
            prediction=model_output,
            payload=payload,
            context={
                "task": data_item["task"],
                "view_record_id": data_item["id"],
            },
        )
        return result.normalized_score
```

The reward wrapper should not contain task-specific logic.

---

# 13. CLI Specification

## 13.1 Source Commands

### Validate source

```bash
docds validate-source DocBank
```

Options:

```bash
docds validate-source DocBank \
  --config configs/processing.yaml \
  --strict
```

Checks:

* Source directory exists.
* Required files exist.
* Raw files exist.
* Annotation files exist.
* Checksums are valid.
* Source metadata is valid.

---

### Export source to canonical

```bash
docds export-source DocBank \
  --source-config sources/DocBank/export.yaml
```

Options:

```bash
docds export-source DocBank \
  --tasks layout,table,formula \
  --overwrite-partitions \
  --num-workers 16 \
  --dry-run
```

Expected behavior:

* Export only requested tasks.
* Write entities under `canonical/entities/*/source=DocBank/`.
* Write task records under `canonical/records/<task>/source=DocBank/`.
* Write canonical image assets under `canonical/assets/`.
* Update manifests for the affected source/task partitions only.

---

## 13.2 Canonical Commands

### Validate canonical records

```bash
docds validate-canonical
```

Scoped validation:

```bash
docds validate-canonical \
  --task table \
  --source DocBank
```

Checks:

* Unique IDs.
* Valid entity references.
* Valid asset references.
* Valid schema.
* Valid coordinate spaces.
* Valid task targets.
* No broken lineage.

---

### Profile canonical data

```bash
docds profile-canonical \
  --task table \
  --source DocBank
```

Outputs:

```text
canonical/reports/table_DocBank_profile.json
canonical/reports/table_DocBank_profile.xlsx
```

The `.xlsx` report is optional and for manual inspection only.

---

### Build materialized index

```bash
docds build-index \
  --task table \
  --source DocBank \
  --by category
```

Creates lightweight pointer files, such as:

```text
canonical/indexes/table/source=DocBank/category=merged_cell_table.parquet
```

---

## 13.3 View Commands

### Build view

```bash
docds build-view views/mineru25_sft_v1/view.yaml
```

Options:

```bash
docds build-view views/mineru25_sft_v1/view.yaml \
  --split train,val,test \
  --num-workers 16 \
  --overwrite \
  --dry-run
```

Behavior:

* Load selected canonical partitions.
* Apply include/exclude rules.
* Apply secondary filters.
* Apply split policy.
* Apply task mixture sampling.
* Materialize images if configured.
* Render prompts.
* Serialize labels.
* Prepare reward payloads if `stage=rlvr`.
* Write Parquet files.

---

### Validate view

```bash
docds validate-view mineru25_sft_v1
```

Checks:

* Image paths exist.
* Prompts are non-empty.
* Labels are non-empty.
* Reward payloads exist for RLVR.
* Reward profiles are registered.
* Coordinate transforms are valid.
* Split leakage does not occur.
* Canonical lineage is resolvable.

---

### Inspect view

```bash
docds inspect-view mineru25_sft_v1 \
  --task table \
  --limit 10
```

Should display:

* Image path.
* Prompt.
* Label preview.
* Canonical record ID.
* Source name.
* Reward profile ID if applicable.

Optional:

```bash
docds inspect-view mineru25_sft_v1 \
  --open-image \
  --render-html-table
```

---

## 13.4 Reward Commands

### Validate reward profile

```bash
docds validate-reward-profile configs/rewards/table_teds_v1.yaml
```

Checks:

* Adapter exists.
* Required params exist.
* Payload schema is valid.

---

### Build reward payloads

Normally this happens inside `build-view`, but it can be run separately:

```bash
docds build-reward-payloads \
  --view mineru25_rlvr_task_rewards_v1 \
  --reward-profile table_teds_v1
```

---

### Smoke-test reward computation

```bash
docds reward-smoke-test \
  --view mineru25_rlvr_task_rewards_v1 \
  --limit 100
```

This should compute reward using the ground-truth label as prediction. For Levenshtein, this should usually return `1.0`.

---

### Score predictions

```bash
docds score-predictions \
  --view mineru25_rlvr_task_rewards_v1 \
  --predictions outputs/predictions.parquet \
  --output outputs/reward_scores.parquet
```

Expected prediction schema:

| Column       | Description         |
| ------------ | ------------------- |
| `id`         | View record ID      |
| `prediction` | Model output string |

Output score schema:

| Column              | Description          |
| ------------------- | -------------------- |
| `id`                | View record ID       |
| `reward_profile_id` | Reward profile       |
| `score`             | Raw score            |
| `normalized_score`  | Normalized reward    |
| `details`           | JSON diagnostics     |
| `error`             | Error message if any |

---

## 13.5 Lineage Commands

### Trace view record

```bash
docds trace --view-record-id view:mineru25_sft_v1:000001
```

Output should show:

```text
view record
  → canonical task record
    → region/page/document
      → canonical image asset
        → source raw file
          → source annotation
```

### Trace canonical record

```bash
docds trace --canonical-record-id table:DocBank:region_000002
```

---

# 14. Runtime Best Practices

## 14.1 Do not embed reward implementation in Parquet

Store:

```text
reward_profile_id
reward_payload / reward_payload_id
```

Do not store executable code or Python import paths inside the dataset records.

---

## 14.2 Keep SFT and RLVR views separate

Even if labels are currently identical, use separate view directories:

```text
views/mineru25_sft_v1/
views/mineru25_rlvr_levenshtein_v1/
```

Reason:

* Different sampling policy.
* Different prompt policy may be needed.
* Different image policy may be needed.
* RLVR needs reward payloads.
* RLVR may exclude records whose reward computation is expensive or unstable.

---

## 14.3 Keep labels and reward payloads related but distinct

For current Levenshtein RLVR:

```text
label == answer_key == reward_payload.label
```

For future task-specific RLVR:

```text
label may be model target string
reward_payload may be structured semantic ground truth
```

Example:

```text
label: enhanced OTSL string
reward_payload: normalized table tree or HTML
```

---

## 14.4 Use sidecar payloads for expensive rewards

Use sidecar payloads for:

* TEDS table trees.
* CDM rendering metadata.
* Unit-test lists.
* Diagram graph structures.
* Large normalized HTML.

This keeps training Parquet compact.

---

## 14.5 Add reward smoke tests before training

Before running GRPO/RLVR, always run:

```bash
docds reward-smoke-test --view <view_name> --limit 1000
```

Expected:

* Ground-truth-as-prediction should score near `1.0`.
* Empty prediction should score near `0.0`.
* Invalid prediction should not crash the reward function.

---

## 14.6 Version every adapter

Every adapter must expose:

```python
name: str
version: str
```

View records should store:

```text
target_format
reward_profile_id
schema_version
config_hash
code_version
```

This is important because reward definitions may change and invalidate old RLVR runs.

---

# 15. Recommended Implementation Phases

## Phase 1: Minimal SFT Pipeline

Implement:

* Source adapters.
* Canonical writer.
* Asset manager.
* View builder.
* Prompt rendering.
* Target serializers.
* SFT Parquet export.
* Basic validation.

Required serializers:

```text
mineru_layout_box_v1
enhanced_otsl_v1
latex_plain_v1
plain_text_v1
```

---

## Phase 2: Initial RLVR with Levenshtein

Implement:

* `RewardAdapter` interface.
* `normalized_levenshtein_v1`.
* RLVR view schema.
* Inline reward payloads.
* VERL reward wrapper.
* Reward smoke-test CLI.

This phase should not introduce TEDS/CDM yet.

---

## Phase 3: Task-Specific Reward Payloads

Implement:

* Sidecar reward payload store.
* Table HTML normalization.
* Formula LaTeX rendering cache.
* Reward profile validation.
* Prediction scoring CLI.

---

## Phase 4: Advanced Rewards

Implement:

* `table_teds_v1`.
* `formula_cdm_v1`.
* `layout_iou_f1_v1`.
* `diagram_graph_v1`.
* `seal_circular_text_v1`.
* Optional `unit_tests_v1`.

---

# 16. Final Recommendation

The final Processing Module should be organized around these stable extension points:

```text
SourceAdapter
TargetSerializer
RewardAdapter
AssetManager
ViewBuilder
Validator
LineageResolver
```

The most important architectural decision is:

> **SFT and RLVR should share the same canonical records and serializers, but RLVR must additionally attach a reward profile and reward payload.**

This allows the initial implementation to remain simple:

```text
RLVR reward = normalized Levenshtein(label, prediction)
```

while keeping the system ready for future task-specific verifiable rewards:

```text
table   → TEDS / unit tests
formula → CDM
layout  → IoU / mAP
diagram → graph matching
seal    → circular-aware matching
```

This design preserves the practicality of the current training workflow while avoiding a rewrite when the reward system becomes more sophisticated.

[1]: https://arxiv.org/html/2412.07626v1 "OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations"
[2]: https://arxiv.org/html/2510.19817v1 "olmOCR 2 Unit Test Rewards for Document OCR"
[3]: https://arxiv.org/abs/2409.03643 "[2409.03643] Image Over Text: Transforming Formula Recognition Evaluation with Character Detection Matching"
