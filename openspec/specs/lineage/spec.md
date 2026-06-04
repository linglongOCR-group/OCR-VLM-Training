# Lineage Specification

## Purpose
Lineage ensures that every view record can be traced backward through the full data provenance chain: from the training view record, through the canonical task record, to the canonical region/page/document, to the canonical image asset, and ultimately to the source raw file and annotation. This traceability is essential for auditing, debugging, and reproducing training data.

## Requirements

### Requirement: View-to-Canonical Record Traceability
Every view record SHALL contain a `canonical_record_id` field that references a valid canonical task record.

#### Scenario: Trace view record to canonical record
- GIVEN a view record with `canonical_record_id=rec:layout:mineru_publaynet:abc123`
- WHEN the lineage resolver traces the view record
- THEN the system SHALL locate the canonical record with matching `record_id` and return its full data

#### Scenario: Broken canonical reference
- GIVEN a view record whose `canonical_record_id` does not exist in any canonical parquet file
- WHEN the lineage resolver attempts to trace it
- THEN the system SHALL raise a KeyError indicating the canonical record was not found

### Requirement: Canonical Record Entity Chain
Every canonical task record SHALL reference valid `document_id`, `page_id`, and optionally `region_id` entities that form the canonical entity hierarchy.

#### Scenario: Record references valid document and page
- GIVEN a canonical record with `document_id=doc:source:001` and `page_id=page:source:001:0`
- WHEN the entity chain is resolved
- THEN the system SHALL confirm both the document and page exist in the canonical store

#### Scenario: Record with region reference
- GIVEN a canonical record with a non-null `region_id`
- WHEN the entity chain is resolved
- THEN the system SHALL confirm the region exists and its `page_id` matches the record's `page_id`

### Requirement: Canonical Asset Lineage
Every canonical task record SHALL reference an `image_asset_id` that resolves to a valid asset record in the canonical asset manifest.

#### Scenario: Asset resolution
- GIVEN a canonical record with `image_asset_id=asset:source:page:0`
- WHEN the asset manifest is consulted
- THEN the system SHALL return the asset record including `path`, `width`, `height`, `format`, and optional `parent_asset_id`

#### Scenario: Asset with parent transform
- GIVEN an asset record with a non-null `parent_asset_id` and `transform_spec_hash`
- WHEN the asset lineage is traced
- THEN the system SHALL confirm the parent asset exists and the transform hash is non-empty

### Requirement: Source Provenance Fields
Every canonical entity SHALL carry source provenance fields that trace back to the original source data.

#### Scenario: Document provenance
- GIVEN a canonical document record
- WHEN its provenance fields are inspected
- THEN it SHALL contain `source_name`, `source_document_id`, and `document_path` referencing the original raw file

#### Scenario: Region provenance
- GIVEN a canonical region record
- WHEN its provenance fields are inspected
- THEN it SHALL contain `source_name` and `source_annotation_id` referencing the original source annotation

#### Scenario: Asset provenance
- GIVEN a canonical asset record
- WHEN its provenance fields are inspected
- THEN it SHALL contain `source_name` and `path` referencing the raw source file, with an optional `checksum`

### Requirement: View Record Lineage Completeness
Every view record SHALL carry sufficient fields to reconstruct the full lineage chain without additional configuration.

#### Scenario: View record lineage fields
- GIVEN a view record
- WHEN its lineage-relevant fields are inspected
- THEN it SHALL contain `canonical_record_id`, `canonical_image_asset_id`, `document_id`, `page_id`, `source_name`, `prompt_template_id`, and `target_format`

#### Scenario: RLVR record additional lineage
- GIVEN an RLVR-stage view record
- WHEN its lineage fields are inspected
- THEN it SHALL additionally contain `reward_profile_id` linking to the reward configuration used during view construction

### Requirement: Lineage Resolver Indexing
The lineage resolver SHALL build an in-memory index of canonical record IDs to parquet file paths for efficient lookups.

#### Scenario: Index built lazily on first access
- GIVEN a lineage resolver with a canonical root containing records
- WHEN the first trace request is made
- THEN the system SHALL scan all record parquet files under `canonical_root/records/` and build an index mapping `record_id` to file path

#### Scenario: Index reused across lookups
- GIVEN a lineage resolver that has already built its index
- WHEN subsequent trace requests are made
- THEN the system SHALL reuse the cached index without re-scanning the filesystem

### Requirement: End-to-End Lineage Chain
The system SHALL support tracing a complete lineage chain from view record through every intermediate layer to the source raw file.

#### Scenario: Full chain trace
- GIVEN a view record with a known `canonical_record_id`
- WHEN `trace --view-record-id {id}` is invoked
- THEN the system SHALL return a JSON payload containing the view record, its canonical record, and all nested references through to the source

#### Scenario: Chain from canonical record only
- GIVEN a canonical record ID
- WHEN `trace --canonical-record-id {id}` is invoked
- THEN the system SHALL return the canonical record data including `document_id`, `page_id`, `region_id`, `image_asset_id`, and `source_name` sufficient to trace back to source files
