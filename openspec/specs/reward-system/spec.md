# Reward System Specification

## Purpose

The reward system provides a framework-agnostic interface for computing reward scores from model predictions, reward payloads, and optional runtime context. Reward adapters are registered in a RewardRegistry and dispatched by a thin VERL wrapper at training time. The system currently ships one built-in adapter (Normalized Levenshtein Distance) and is designed to accommodate future task-specific adapters.

## Requirements

### Requirement: RewardAdapter Abstract Interface

Every reward adapter SHALL implement the RewardAdapter abstract base class exposing `prepare_payload`, `score`, `batch_score`, and `validate_payload` methods. Each adapter SHALL declare `name`, `version`, and `task` class attributes.

#### Scenario: Adapter implements required methods

- **GIVEN** a concrete subclass of RewardAdapter
- **WHEN** the subclass is instantiated
- **THEN** it SHALL provide implementations of `prepare_payload(canonical_record, view_record, config)`, `score(prediction, payload, context)`, `batch_score(predictions, payloads, context)`, and `validate_payload(payload)`
- **AND** it SHALL expose `name: str`, `version: str`, and `task: str` class attributes

#### Scenario: Batch scoring defaults to element-wise delegation

- **GIVEN** a RewardAdapter subclass that does not override `batch_score`
- **WHEN** `batch_score` is called with a list of predictions and payloads
- **THEN** the default implementation SHALL delegate to `score` element-wise and return a list of RewardResult instances

#### Scenario: Payload validation enforces label key

- **GIVEN** the base RewardAdapter `validate_payload` method
- **WHEN** a payload dict without a `label` key is validated
- **THEN** the method SHALL raise a ValueError naming the missing field
- **AND** subclasses MAY tighten validation with additional constraints

### Requirement: RewardResult Data Contract

Every `score` and `batch_score` call SHALL return a RewardResult dataclass containing `score`, `normalized_score`, `passed`, `details`, and `error` fields. The `normalized_score` field SHALL always be in the range [0, 1].

#### Scenario: RewardResult contains all required fields

- **GIVEN** any reward adapter scoring call
- **WHEN** the adapter returns a RewardResult
- **THEN** the result SHALL contain `score` (float), `normalized_score` (float in [0, 1]), `passed` (bool or None), `details` (dict with per-metric diagnostics), and `error` (str or None)
- **AND** `normalized_score` SHALL be used by all callers as the canonical scalar reward

#### Scenario: Error result when scoring fails

- **GIVEN** an adapter that encounters an unrecoverable error during scoring
- **WHEN** the error is caught internally
- **THEN** the returned RewardResult SHALL have a non-null `error` field describing the failure
- **AND** `normalized_score` SHALL be 0.0

### Requirement: Normalized Levenshtein Distance Adapter

The system SHALL provide a built-in adapter named `normalized_levenshtein_v1` that computes reward as `1 - edit_distance(prediction, label) / max(len(prediction), len(label), 1)`. The adapter SHALL handle empty strings, be deterministic, and expose per-sample diagnostics.

#### Scenario: Score a text prediction with Levenshtein

- **GIVEN** a Levenshtein payload with `label = "hello world"`
- **WHEN** the adapter scores prediction `"helo world"`
- **THEN** `normalized_score` SHALL be approximately 0.909
- **AND** `details.edit_distance` SHALL be 1
- **AND** `details.denominator` SHALL be 11
- **AND** `error` SHALL be None

#### Scenario: Empty prediction against a non-empty label

- **GIVEN** a Levenshtein payload with `label = "invoice 123"`
- **WHEN** the adapter scores prediction `""`
- **THEN** `normalized_score` SHALL be 0.0
- **AND** `details.edit_distance` SHALL be 10

#### Scenario: Exact match returns 1.0

- **GIVEN** a Levenshtein payload with `label = "Revenue|Amount"`
- **WHEN** the adapter scores prediction `"Revenue|Amount"`
- **THEN** `normalized_score` SHALL be 1.0
- **AND** `details.edit_distance` SHALL be 0
- **AND** `details.denominator` SHALL equal the label length

#### Scenario: Empty prediction and empty label returns 1.0

- **GIVEN** a Levenshtein payload with `label = ""`
- **WHEN** the adapter scores prediction `""`
- **THEN** `normalized_score` SHALL be 1.0
- **AND** the denominator SHALL be clamped to 1 to avoid division by zero

#### Scenario: Case-insensitive normalization

- **GIVEN** a Levenshtein payload with `label = "Hello"` and normalization options `{"case_sensitive": false}`
- **WHEN** the adapter scores prediction `"hello"`
- **THEN** `normalized_score` SHALL be 1.0 because case differences are ignored

#### Scenario: Deterministic scoring

- **GIVEN** the same prediction string and the same payload
- **WHEN** the Levenshtein adapter scores the pair multiple times
- **THEN** every invocation SHALL return an identical RewardResult

### Requirement: RewardRegistry and Adapter Lookup

The system SHALL maintain a RewardRegistry that maps reward profile ID strings to RewardAdapter instances, providing default and configured registry factories with clear error messages on missing profiles.

#### Scenario: Default registry includes built-in adapter

- **GIVEN** a call to `default_reward_registry()`
- **WHEN** the registry is created
- **THEN** it SHALL register the built-in `normalized_levenshtein_v1` adapter

#### Scenario: Configured registry loads from processing config

- **GIVEN** a processing config with a `registries.rewards` section listing adapter entries
- **WHEN** `configured_reward_registry(processing_config)` is called
- **THEN** the registry SHALL load adapters from that config section
- **AND** it SHALL fall back to `default_reward_registry()` when no entries are configured

#### Scenario: Unregistered reward profile raises an error

- **GIVEN** a data item with `reward_profile_id = "table_teds_v1"` and only `normalized_levenshtein_v1` registered
- **WHEN** adapter lookup is performed
- **THEN** a clear KeyError or equivalent error SHALL be raised indicating the profile is not registered

### Requirement: Reward Payload Storage

View records SHALL carry reward payload information for RLVR views using one of three mutually exclusive mechanisms: inline, sidecar, or file reference. RLVR view records SHALL set `reward_profile_id` and carry at least one payload field.

#### Scenario: Inline payload stored in view record

- **GIVEN** an RLVR view record with a small payload such as a Levenshtein label string
- **WHEN** the view record is created
- **THEN** the record SHALL contain a `reward_payload` field with a JSON dict embedded directly in the Parquet row
- **AND** the record SHALL set `reward_profile_id`

#### Scenario: Sidecar payload for large structured data

- **GIVEN** an RLVR view record requiring a large or structured payload such as table HTML
- **WHEN** the view record is created
- **THEN** the record SHALL contain `reward_payload_id` referencing an external payload in a sidecar file under `views/<view>/reward_payloads/`
- **AND** sidecar payload loading is not yet implemented so adapters SHALL raise a clear error when encountering sidecar references

#### Scenario: View validation enforces payload constraint

- **GIVEN** an RLVR view record
- **WHEN** view validation runs
- **THEN** the record SHALL set `reward_profile_id`
- **AND** the record SHALL carry at least one of `reward_payload`, `reward_payload_id`, or `reward_payload_path`
- **AND** validation SHALL reject records missing all three payload fields

### Requirement: Framework Independence

RewardAdapter and RewardResult SHALL NOT import from or depend on any training framework. The rewards package SHALL remain importable without ML runtime dependencies. Framework integration SHALL be confined to a thin wrapper in a separate package.

#### Scenario: Rewards package imports without ML dependencies

- **GIVEN** a Python environment without VERL, transformers, or torch installed
- **WHEN** `tools.data_management.rewards` is imported
- **THEN** the import SHALL succeed without errors
- **AND** RewardAdapter and RewardResult SHALL be usable

#### Scenario: Framework integration isolated in wrapper

- **GIVEN** the VERL reward wrapper in `verl_plugins.rewards`
- **WHEN** the wrapper needs to call an adapter
- **THEN** it SHALL delegate to framework-independent adapters
- **AND** no training framework imports SHALL appear in the rewards package itself

### Requirement: VERL Reward Wrapper

The VerlRewardWrapper SHALL read the reward profile from data items, look up the adapter from the registry, load the payload, call the adapter, and return the normalized score. The wrapper SHALL NOT contain task-specific logic.

#### Scenario: Wrapper dispatches to the registry

- **GIVEN** a data item with `reward_profile_id = "normalized_levenshtein_v1"`, `reward_payload = {"label": "abc"}`, `task = "text"`, and `id = "view:rlvr:test:001"`
- **WHEN** `VerlRewardWrapper.__call__(data_item, "abc")` is invoked
- **THEN** the wrapper SHALL return 1.0 as the scalar reward
- **AND** the adapter layer SHALL NOT import any ML framework

#### Scenario: Wrapper rejects sidecar payloads until implemented

- **GIVEN** a data item with a sidecar `reward_payload_id` instead of an inline `reward_payload`
- **WHEN** the wrapper attempts to load the payload
- **THEN** the wrapper SHALL raise a clear error indicating sidecar payloads are not yet supported

#### Scenario: Compute score satisfies VERL contract

- **GIVEN** the `compute_score` function in `verl_plugins.rewards.aggregate`
- **WHEN** it is called with `data_source`, `solution_str`, `ground_truth`, and `extra_info` keyword arguments
- **THEN** it SHALL satisfy the VERL custom reward function contract and return a scalar reward

### Requirement: Reward Smoke Testing

The system SHALL provide `reward_smoke_test(view_root, limit)` that scores ground-truth labels against themselves and returns aggregate statistics. It SHALL be runnable from the CLI.

#### Scenario: Smoke test on an RLVR view

- **GIVEN** an RLVR view directory with 50 rows each carrying `reward_profile_id = "normalized_levenshtein_v1"` and `reward_payload.label` equal to the row's own label
- **WHEN** `reward_smoke_test(view_root, limit=50)` is called
- **THEN** the result SHALL contain `count = 50`, `min_score = 1.0`, `mean_score = 1.0`, `max_score = 1.0`

#### Scenario: Smoke test raises on empty view

- **GIVEN** an RLVR view directory with no RLVR rows
- **WHEN** `reward_smoke_test(view_root, limit=10)` is called
- **THEN** the function SHALL raise a clear error indicating no RLVR rows were found

#### Scenario: Smoke test runnable from CLI

- **GIVEN** the CLI entry point `docds reward-smoke-test`
- **WHEN** the command is invoked with `--view <name> --limit <n>`
- **THEN** the smoke test SHALL execute and print aggregate statistics

### Requirement: Prediction Scoring

The system SHALL provide `score_predictions(view_root, predictions_path, output_path)` that joins view rows with prediction rows, dispatches scoring, and writes output Parquet.

#### Scenario: Score predictions from a model output file

- **GIVEN** a predictions Parquet with rows having `id` and `prediction` columns and a view with matching rows carrying labels
- **WHEN** `score_predictions(view_root, predictions_path, output_path)` is called
- **THEN** the output Parquet SHALL contain columns `id`, `reward_profile_id`, `score`, `normalized_score`, `details`, and `error`

#### Scenario: Prediction input schema validation

- **GIVEN** a predictions file missing the `prediction` column
- **WHEN** `score_predictions` attempts to read the file
- **THEN** the function SHALL raise a clear error describing the required schema with `id` and `prediction` fields

### Requirement: Adapter Versioning and Reproducibility

Every reward adapter SHALL expose `name` and `version` as class attributes. View records SHALL store `reward_profile_id` encoding name and version. Changing the scoring formula SHALL increment the version suffix and register under a new profile ID.

#### Scenario: Profile ID encodes adapter name and version

- **GIVEN** a reward adapter with `name = "normalized_levenshtein"` and `version = "v1"`
- **WHEN** the adapter is registered
- **THEN** the `reward_profile_id` SHALL be `"normalized_levenshtein_v1"`

#### Scenario: Formula change requires new version

- **GIVEN** an existing adapter `normalized_levenshtein_v1` with a published scoring formula
- **WHEN** the scoring formula is modified
- **THEN** the version suffix SHALL be incremented to produce a new profile ID such as `normalized_levenshtein_v2`
- **AND** the old profile ID SHALL remain resolvable or produce a clear deprecation error

#### Scenario: View records enable reproducibility

- **GIVEN** an RLVR view record storing `reward_profile_id = "normalized_levenshtein_v1"`
- **WHEN** a training run is reproduced
- **THEN** the stored profile ID SHALL resolve to the same adapter version and scoring formula used during the original run
