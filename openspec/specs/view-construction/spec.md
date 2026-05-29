# View Construction Specification

## Purpose
The View layer materializes model-specific training samples from canonical records. The ViewBuilder coordinates record selection, sampling, split assignment, image preparation, prompt rendering, target serialization, reward payload preparation (for RLVR views), and Parquet writing. Views are the final bridge between canonical data and training frameworks such as VERL.

## Requirements

### Requirement: View Record Schema
Every view record SHALL contain the columns `id`, `task`, `image_path`, `prompt`, `label`, `source_name`, `document_id`, `page_id`, `region_id`, `canonical_record_id`, `target_format`, `prompt_template_id`, `split`, and `metadata`. The image data column populated (`images`, `images_bytes`, or `images_path`) SHALL correspond to the configured image materialization mode.

> Migration note: The source document describes three alternative image columns based on materialization mode but does not explicitly mandate "exactly one."

#### Scenario: SFT view record contains all required columns
- GIVEN a view built with stage `sft` and materialization mode `embedded`
- WHEN a view record is written to parquet
- THEN the record SHALL contain `id`, `task`, `image_path`, `images_bytes`, `prompt`, `label`, `source_name`, `document_id`, `page_id`, `region_id`, `canonical_record_id`, `target_format`, `prompt_template_id`, `split`, and `metadata`

#### Scenario: RLVR view record includes reward columns
- GIVEN a view built with stage `rlvr`
- WHEN a view record is written to parquet
- THEN the record SHALL contain all SFT columns plus `reward_profile_id`, and at least one of `reward_payload`, `reward_payload_id`, or `reward_payload_path`

#### Scenario: View record populates image column by mode
- GIVEN a view configuration with `materialization.mode` set to `embedded`
- WHEN a view record is materialized
- THEN `images_bytes` SHALL be populated and `images` and `images_path` SHALL be null

### Requirement: View Configuration File
The view configuration (`view.yaml`) SHALL define `name`, `stage`, `model_family`, `include` rules, `exclude` rules, `task_mixture` weights, `split_policy`, `prompt_templates`, `target_serialization`, and `image_policy`.

#### Scenario: Load valid view configuration
- GIVEN a `view.yaml` containing all required top-level keys
- WHEN the ViewBuilder loads the configuration
- THEN it SHALL parse `include`, `exclude`, `task_mixture`, `split_policy`, `prompt_templates`, `target_serialization`, and `image_policy` without error

#### Scenario: Reject configuration missing required keys
- GIVEN a YAML file that omits `split_policy`
- WHEN the ViewBuilder loads the configuration
- THEN it SHALL raise a configuration error indicating the missing key

#### Scenario: RLVR configuration requires reward_profile
- GIVEN a view configuration with `stage: rlvr`
- WHEN the configuration is validated
- THEN the system SHALL require `reward_profile` to be present

### Requirement: Primary Record Selection
The ViewBuilder SHALL select canonical records based on `task` and `source` as the primary selection axes. The `include` list defines which task+source combinations to accept; the `exclude` list removes matched combinations from the included set.

#### Scenario: Include specific task and sources
- GIVEN a view configuration with `include: [{task: table, sources: [DocBank, PubTabNet]}]`
- WHEN the ViewBuilder applies selection
- THEN only canonical records where `task == "table"` and `source_name in ["DocBank", "PubTabNet"]` SHALL be retained

#### Scenario: Exclude overrides include
- GIVEN a view configuration that includes `task: layout, sources: [DocBank, DocLayNet]` and excludes `task: layout, sources: [DocBank]`
- WHEN the ViewBuilder applies selection
- THEN layout records from DocLayNet SHALL be retained and layout records from DocBank SHALL be excluded

#### Scenario: Empty include list selects nothing
- GIVEN a view configuration with an empty `include` list
- WHEN the ViewBuilder applies selection
- THEN no canonical records SHALL be selected

### Requirement: Secondary Metadata Filtering
The ViewBuilder SHALL support optional `where` clauses on each include rule that filter records by metadata columns including `category`, `language`, and `quality_flags`.

> Migration note: The source document says filtering "can use" metadata columns. The SHALL applies to the filtering mechanism existing; individual filters are optional per include rule.

#### Scenario: Filter by category inclusion
- GIVEN an include rule with `where: {category: {in: [merged_cell_table, borderless_table]}}`
- WHEN the ViewBuilder applies secondary filtering
- THEN only records whose `category` is `merged_cell_table` or `borderless_table` SHALL be retained

#### Scenario: Filter by quality flags exclusion
- GIVEN an include rule with `where: {quality_flags: {not_contains: [needs_review]}}`
- WHEN the ViewBuilder applies secondary filtering
- THEN records whose `quality_flags` list contains `needs_review` SHALL be excluded

#### Scenario: Filter by language
- GIVEN an include rule with `where: {language: {in: [en, zh]}}`
- WHEN the ViewBuilder applies secondary filtering
- THEN only records whose `language` is `en` or `zh` SHALL be retained

### Requirement: Task Mixture Sampling
When `task_mixture` weights are specified, the ViewBuilder SHALL resample the selected records so that the proportion of records for each task approximates the configured weight.

#### Scenario: Apply mixture weights
- GIVEN selected records containing 10000 layout, 5000 table, and 500 formula records, and a `task_mixture` of `{layout: 0.4, table: 0.4, formula: 0.2}`
- WHEN the ViewBuilder applies mixture sampling
- THEN the output SHALL contain records in approximate proportion 4:4:2 across the three tasks

#### Scenario: Tasks not in mixture pass through unmodified
- GIVEN a `task_mixture` that specifies weights for `layout` and `table` but not `formula`
- WHEN the ViewBuilder applies mixture sampling
- THEN all `formula` records SHALL be retained without resampling

#### Scenario: Mixture sampling is deterministic with seed
- GIVEN a `task_mixture` configuration and a fixed `seed` value in the split policy
- WHEN the ViewBuilder samples twice with the same configuration
- THEN the sampled record sets SHALL be identical

### Requirement: Split Assignment
The ViewBuilder SHALL assign every record to exactly one of `train`, `val`, or `test` according to the `split_policy`. Splits SHALL be assigned at the configured level (`document`, `page`, or `record`) to prevent leakage.

#### Scenario: Split at document level
- GIVEN a `split_policy` with `level: document`, `train_ratio: 0.98`, `val_ratio: 0.01`, `test_ratio: 0.01`, and `seed: 42`
- WHEN the ViewBuilder assigns splits
- THEN all records belonging to the same `document_id` SHALL share the same split value, and the approximate proportions SHALL match the configured ratios

#### Scenario: Split at page level
- GIVEN a `split_policy` with `level: page`
- WHEN the ViewBuilder assigns splits
- THEN all records belonging to the same `page_id` SHALL share the same split value, independent of other pages in the same document

#### Scenario: Predefined split strategy
- GIVEN a `split_policy` with `strategy: predefined` and a `split_file` path
- WHEN the ViewBuilder assigns splits
- THEN splits SHALL be read from the specified file rather than computed

#### Scenario: No split leakage across train and val
- GIVEN records assigned to `train` and `val` splits at the document level
- WHEN split assignment is complete
- THEN no `document_id` SHALL appear in both `train` and `val` records

### Requirement: Prompt Rendering
The ViewBuilder SHALL render a `prompt` string for each record using a task-specific Jinja template. The rendered prompt SHALL be stored in the `prompt` column and the template identifier SHALL be stored in `prompt_template_id`.

#### Scenario: Render layout prompt from template
- GIVEN a canonical layout record and a template file `templates/mineru25/layout.jinja`
- WHEN the ViewBuilder renders the prompt
- THEN the output `prompt` SHALL be a non-empty string and `prompt_template_id` SHALL be `"mineru25_layout_v1"` or the configured identifier

#### Scenario: Template receives canonical record fields
- GIVEN a Jinja template that references `{{ task }}` and `{{ image_path }}`
- WHEN the template is rendered against a canonical record
- THEN the resulting prompt SHALL contain the task name and image path from that record

### Requirement: Target Serialization
The ViewBuilder SHALL convert canonical targets into model-specific label strings using registered serializer adapters identified by the `target_serialization` configuration. The serialized label SHALL be stored in the `label` column and the format identifier in `target_format`.

#### Scenario: Serialize layout target as MinerU box string
- GIVEN a canonical layout record with elements containing `bbox`, `label`, and `rotation`, and a target serialization of `mineru_layout_box_v1`
- WHEN the serializer processes the record
- THEN the output `label` SHALL be a MinerU-style box string such as `<|box_start|>50 100 250 350<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>`

#### Scenario: Serialize table target as enhanced OTSL
- GIVEN a canonical table record with HTML and cell data, and a target serialization of `enhanced_otsl_v1`
- WHEN the serializer processes the record
- THEN the output `label` SHALL be an OTSL string using `<fcel>`, `<nl>`, and related tokens

#### Scenario: Serializer transforms coordinates for resized images
- GIVEN a canonical layout record with bboxes in canonical page pixel coordinates and a view image resized by scale factor 0.5
- WHEN the serializer processes the record
- THEN the output label SHALL contain coordinates scaled by 0.5, matching the view image dimensions

### Requirement: Image Materialization Modes
The ViewBuilder SHALL support three image materialization modes: `embedded`, `source_reference`, and `nested_reference`. The mode is configured in `view.yaml` under `image_policy.materialization.mode`.

#### Scenario: Embedded mode stores binary bytes
- GIVEN a view configuration with `materialization.mode: embedded`
- WHEN the ViewBuilder materializes images
- THEN each record SHALL contain `images_bytes` as a list of binary blobs and `images_path` SHALL be null

#### Scenario: Source reference mode stores relative paths
- GIVEN a view configuration with `materialization.mode: source_reference`
- WHEN the ViewBuilder materializes images for records with no transform configured
- THEN each record SHALL contain `images_path` with dataset-root-relative paths to the canonical assets and `images_bytes` SHALL be null

#### Scenario: Source reference with transform writes view assets
- GIVEN a view configuration with `materialization.mode: source_reference` and a resize transform configured
- WHEN the ViewBuilder materializes images
- THEN transformed images SHALL be written under `views/<view_name>/assets/` and `images_path` SHALL reference those dataset-root-relative paths

#### Scenario: Nested reference mode stores VERL-native dicts
- GIVEN a view configuration with `materialization.mode: nested_reference`
- WHEN the ViewBuilder materializes images
- THEN each record SHALL contain `images` as a list of dicts in the shape `[{"image": "<dataset-root-relative-path>"}]`

### Requirement: Image Transform Configuration
The ViewBuilder SHALL apply optional per-task image transforms defined in `image_policy.transform`, including resize, padding, and format conversion. Transforms SHALL be recorded in the view asset manifest.

#### Scenario: Apply resize with aspect ratio preservation
- GIVEN a task transform with `resize: {max_side: 1024, keep_aspect_ratio: true}`
- WHEN the ViewBuilder transforms a 2000x1000 image
- THEN the output image SHALL have dimensions 1024x512

#### Scenario: Apply pad to multiple
- GIVEN a task transform with `pad: {enabled: true, pad_to_multiple: 28}`
- WHEN the ViewBuilder transforms a 500x300 image
- THEN the output dimensions SHALL each be a multiple of 28

#### Scenario: Transform recorded in asset manifest
- GIVEN a view image asset produced by a resize transform
- WHEN the asset manifest is written
- THEN the manifest entry for that asset SHALL include a `transform` field describing the operation, input size, output size, and coordinate mapping

### Requirement: Sharded Parquet Output
For large embedded-image views, the ViewBuilder SHALL write sharded split directories when `shard_policy` is configured. Each split directory SHALL contain numbered part files.

#### Scenario: Shard output by row count
- GIVEN a view configuration with `shard_policy: {rows_per_shard: 128}` and 300 training records
- WHEN the ViewBuilder writes the training split
- THEN the output SHALL be `train/part-00000.parquet` (128 rows), `train/part-00001.parquet` (128 rows), and `train/part-00002.parquet` (44 rows)

#### Scenario: Single file output when no shard policy
- GIVEN a view configuration without `shard_policy`
- WHEN the ViewBuilder writes the training split
- THEN the output SHALL be a single `train.parquet` file

#### Scenario: Shard applies to all splits
- GIVEN a view configuration with `shard_policy: {rows_per_shard: 100}`
- WHEN the ViewBuilder writes all splits
- THEN each of `train/`, `val/`, and `test/` SHALL be directories containing `part-*.parquet` files

### Requirement: SFT View Construction
The ViewBuilder SHALL produce SFT views with the standard `prompt` and `label` columns. SFT views SHOULD include `messages` as a list of user/assistant conversation turns for frameworks that require the chat format.

#### Scenario: SFT view contains messages column
- GIVEN a view configuration with `stage: sft`
- WHEN the ViewBuilder writes a record
- THEN the record SHOULD contain a `messages` column with a user turn containing the rendered prompt and an assistant turn containing the serialized label

> Migration note: The source document marks `messages` as "Recommended additional" for SFT views, not required.

### Requirement: RLVR View Construction
The ViewBuilder SHALL produce RLVR views that include reward-related columns. SFT and RLVR views SHOULD be written to separate view directories even when their labels are identical.

#### Scenario: RLVR view with inline Levenshtein payload
- GIVEN a view configuration with `stage: rlvr` and `reward_profile: {default: normalized_levenshtein_v1}` and `reward_payload: {materialization: inline}`
- WHEN the ViewBuilder writes a record
- THEN the record SHALL contain `reward_profile_id` set to `normalized_levenshtein_v1` and `reward_payload` containing an inline JSON payload with the label string

#### Scenario: RLVR view with sidecar payload
- GIVEN a view configuration with `stage: rlvr` and `reward_payload: {materialization: sidecar}`
- WHEN the ViewBuilder writes records
- THEN the record SHALL contain `reward_payload_id` and `reward_payload_path` pointing to a sidecar parquet file under `views/<view_name>/reward_payloads/`

#### Scenario: RLVR view with task-specific reward profiles
- GIVEN a view configuration with `reward_profile: {by_task: {layout: layout_iou_f1_v1, table: table_teds_v1}}`
- WHEN the ViewBuilder writes a layout record
- THEN the record SHALL contain `reward_profile_id` set to `layout_iou_f1_v1`

#### Scenario: SFT and RLVR views are separate directories
- GIVEN two view configurations, one with `stage: sft` named `mineru25_sft_v1` and one with `stage: rlvr` named `mineru25_rlvr_levenshtein_v1`
- WHEN both views are built
- THEN their outputs SHOULD reside in `views/mineru25_sft_v1/` and `views/mineru25_rlvr_levenshtein_v1/` respectively, with no shared parquet files

> Migration note: The source document recommends keeping SFT and RLVR views separate but uses "should," not "shall."

### Requirement: View Image Source Selection
The ViewBuilder SHALL select the appropriate canonical image source per task as specified in `image_policy.image_source`. For example, layout tasks use canonical page images while recognition tasks use canonical region crops.

#### Scenario: Layout task uses page image
- GIVEN `image_policy.image_source: {layout: canonical_page_image}`
- WHEN the ViewBuilder prepares images for a layout record
- THEN it SHALL resolve the `image_asset_id` to the canonical page render asset

#### Scenario: Table task uses region crop
- GIVEN `image_policy.image_source: {table: canonical_region_crop}`
- WHEN the ViewBuilder prepares images for a table record
- THEN it SHALL resolve the `image_asset_id` to the canonical region crop asset

### Requirement: View Output Directory Structure
The ViewBuilder SHALL write output under `views/<view_name>/` containing `view.yaml`, split parquet files or sharded directories, a `stats.json` summary, an optional `assets/` directory for transformed images, and a `manifests/asset_manifest.parquet` for view-level image assets.

#### Scenario: Non-sharded view directory layout
- GIVEN a successful non-sharded view build
- WHEN the build completes
- THEN the directory SHALL contain `view.yaml`, `train.parquet`, `val.parquet`, `test.parquet`, and `stats.json`

#### Scenario: Sharded view directory layout
- GIVEN a successful sharded view build
- WHEN the build completes
- THEN the directory SHALL contain `view.yaml`, `train/` with `part-*.parquet` files, `val/` with `part-*.parquet` files, `test/` with `part-*.parquet` files, and `stats.json`

#### Scenario: Stats file contains split counts
- GIVEN a successful view build producing 9800 train, 100 val, and 100 test records
- WHEN `stats.json` is written
- THEN it SHOULD contain per-split row counts

> Migration note: The source document mentions `stats.json` exists in the view directory but does not define its schema.
