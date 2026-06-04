# Validation Specification

## Purpose
Validation ensures data integrity at every stage of the document dataset pipeline: source ingestion, canonical storage, and view construction. Each validation layer checks constraints appropriate to its data model and raises descriptive errors when constraints are violated.

## Requirements

### Requirement: Source File Existence
Source validation SHALL verify that all declared raw files exist at their expected paths.

#### Scenario: Missing raw file detected
- GIVEN a source configuration that declares a raw file at `sources/publaynet/images/001.png`
- WHEN source validation runs
- THEN the system SHALL report an error if `001.png` does not exist

#### Scenario: All raw files present
- GIVEN a source configuration where every declared raw file exists
- WHEN source validation runs
- THEN the system SHALL pass the raw file existence check without errors

### Requirement: Source Annotation Existence
Source validation SHALL verify that all declared annotation files exist and are parseable.

#### Scenario: Missing annotation file
- GIVEN a source that references an annotation file `annotations/train.json`
- WHEN source validation runs
- THEN the system SHALL report an error if the annotation file is missing

#### Scenario: Malformed annotation file
- GIVEN an annotation file that does not conform to its declared format (e.g., invalid JSON)
- WHEN source validation runs
- THEN the system SHALL report an error indicating the format mismatch

### Requirement: Source Checksum Validation
Source validation SHALL verify that file checksums match their declared values.

#### Scenario: Checksum mismatch
- GIVEN a source file with a declared checksum of `abc123`
- WHEN source validation computes the actual checksum
- THEN the system SHALL report an error if the computed checksum differs

#### Scenario: Checksum match
- GIVEN a source file with a declared checksum
- WHEN source validation computes the actual checksum
- THEN the system SHALL pass without error when checksums match

### Requirement: Source Metadata Completeness
Source validation SHALL verify that required metadata fields are present in the source configuration.

#### Scenario: Missing required metadata field
- GIVEN a source configuration missing a required field such as `format` or `version`
- WHEN source validation runs
- THEN the system SHALL report an error listing the missing fields

### Requirement: Canonical Record ID Uniqueness
Canonical validation SHALL verify that every `record_id` across all canonical parquet files is globally unique.

#### Scenario: Duplicate record IDs detected
- GIVEN two canonical parquet files containing the same `record_id`
- WHEN canonical validation scans all files
- THEN the system SHALL raise a ValueError listing the duplicate IDs (up to 5 examples)

#### Scenario: All record IDs unique
- GIVEN canonical parquet files with no overlapping record IDs
- WHEN canonical validation runs
- THEN the system SHALL pass the uniqueness check without error

### Requirement: Canonical Required Columns
Canonical validation SHALL verify that every canonical parquet file contains all required columns.

#### Scenario: Missing required column
- GIVEN a canonical parquet file that lacks the `target` column
- WHEN canonical validation runs
- THEN the system SHALL raise a ValueError identifying the file and the missing column

#### Scenario: All required columns present
- GIVEN canonical parquet files containing `record_id`, `task`, `source_name`, `document_id`, `page_id`, `image_asset_id`, and `target`
- WHEN canonical validation runs
- THEN the system SHALL pass the column check without error

### Requirement: Canonical Record Filtering
Canonical validation SHALL support filtering by task and source to validate subsets of records.

#### Scenario: Filter by task
- GIVEN a canonical root with records for tasks `layout`, `table`, and `formula`
- WHEN `validate-canonical --task layout` is invoked
- THEN the system SHALL validate only records under the `layout` partition

#### Scenario: Filter by source
- GIVEN a canonical root with multiple sources
- WHEN `validate-canonical --source mineru_publaynet` is invoked
- THEN the system SHALL validate only files matching `source=mineru_publaynet`

### Requirement: View Required Columns
View validation SHALL verify that every view parquet file contains the required columns: `id`, `stage`, `task`, `image_path`, `prompt`, `label`, `canonical_record_id`, and `split`.

#### Scenario: Missing view column
- GIVEN a view parquet file that lacks the `prompt` column
- WHEN view validation runs
- THEN the system SHALL raise a ValueError identifying the file and the missing column

### Requirement: View Prompt and Label Non-Empty
View validation SHALL verify that every view record has a non-empty prompt and a non-empty label.

#### Scenario: Empty prompt
- GIVEN a view record with a `None` or empty string prompt
- WHEN view validation processes that row
- THEN the system SHALL raise a ValueError identifying the file, row index, and the empty prompt

#### Scenario: Empty label
- GIVEN a view record with a falsy label
- WHEN view validation processes that row
- THEN the system SHALL raise a ValueError identifying the file, row index, and the empty label

### Requirement: View RLVR Reward Fields
View validation SHALL verify that RLVR-stage records contain `reward_profile_id`, `data_source`, and `extra_info`.

#### Scenario: Missing reward profile for RLVR record
- GIVEN a view record with `stage=rlvr` and no `reward_profile_id`
- WHEN view validation processes that row
- THEN the system SHALL raise a ValueError indicating the missing reward_profile_id

#### Scenario: RLVR record with all reward fields
- GIVEN a view record with `stage=rlvr`, a valid `reward_profile_id`, `data_source`, and `extra_info`
- WHEN view validation processes that row
- THEN the system SHALL pass the RLVR-specific checks without error

### Requirement: View Split Leakage Detection
View validation SHALL detect when the same `document_id` appears in more than one split.

#### Scenario: Document in train and val
- GIVEN view records where `document_id=doc_42` appears in both the train and val splits
- WHEN view validation completes its cross-split scan
- THEN the system SHALL raise a ValueError listing leaked documents (up to 5 examples)

#### Scenario: No split leakage
- GIVEN view records where every `document_id` belongs to exactly one split
- WHEN view validation runs
- THEN the system SHALL pass the leakage check without error

### Requirement: View Image Consistency
View validation SHALL verify that image placeholder counts in prompts match the number of provided images.

#### Scenario: Placeholder count mismatch
- GIVEN a view record with 2 `<image>` placeholders in the prompt but 3 image entries
- WHEN view validation processes that row
- THEN the system SHALL raise a ValueError reporting the mismatch between placeholder count and image count

#### Scenario: Multiple image carriers
- GIVEN a view record that populates both `images_bytes` and `images_path`
- WHEN view validation processes that row
- THEN the system SHALL raise a ValueError indicating multiple image carriers are present

### Requirement: Reward Smoke Test Ground Truth Scoring
The reward smoke test SHALL score each RLVR record using its ground truth label as the prediction, expecting a normalized score near 1.0.

#### Scenario: Ground truth scores near 1.0
- GIVEN RLVR view records with Levenshtein reward profiles
- WHEN the reward smoke test scores each record using its label as prediction
- THEN the mean normalized_score SHALL be at least 0.95

#### Scenario: Empty prediction scores near 0.0
- GIVEN RLVR view records with non-empty labels
- WHEN the reward smoke test scores an empty string prediction
- THEN the normalized_score SHALL be at or near 0.0

### Requirement: Reward Smoke Test Error Resilience
The reward smoke test SHALL not crash when encountering invalid or malformed reward payloads.

#### Scenario: Invalid payload does not crash
- GIVEN a view record with a malformed `reward_payload` value
- WHEN the reward smoke test processes that record
- THEN the system SHALL report the error without terminating the entire smoke test
