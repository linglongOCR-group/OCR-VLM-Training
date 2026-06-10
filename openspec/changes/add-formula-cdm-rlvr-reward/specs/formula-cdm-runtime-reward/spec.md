## ADDED Requirements

### Requirement: Formula CDM Runtime Reward
The system SHALL support an opt-in runtime reward for GRPO Formula tasks that computes a CDM-style score from predicted LaTeX and reference LaTeX.

#### Scenario: Formula task uses CDM reward
- **GIVEN** a GRPO reward configuration that routes `formula` to `cdm_katex_v1`
- **WHEN** the VERL reward entrypoint scores a sample whose task type is `formula`
- **THEN** the system SHALL compute the sample reward using the CDM runtime reward
- **AND** the returned result SHALL include `reward_name = "cdm_latex_render"`, `reward_version`, `reward_profile_id`, `score`, and `reward_total`

#### Scenario: CDM score is normalized
- **GIVEN** the CDM runtime reward receives any service response
- **WHEN** it returns a reward result to VERL
- **THEN** `score` and `reward_total` SHALL be numeric values in the range `[0.0, 1.0]`

### Requirement: LaTeX-only Runtime Input Contract
The v1 CDM runtime reward SHALL use the model output string as predicted LaTeX and the VERL ground-truth string as reference LaTeX, without requiring rich OmniDocBench annotation metadata.

#### Scenario: Score from VERL reward inputs
- **GIVEN** VERL calls the reward entrypoint with `solution_str`, `ground_truth`, `data_source`, and `extra_info`
- **WHEN** the selected profile is `cdm_katex_v1`
- **THEN** the CDM client SHALL send `solution_str` as the prediction and `ground_truth` as the reference
- **AND** the score SHALL NOT require bbox, polygon, category, or official annotation payload fields

#### Scenario: Determine Formula task type
- **GIVEN** `extra_info.task_type` is `formula`
- **WHEN** the reward router selects a profile
- **THEN** it SHALL treat the sample as a Formula task
- **AND** if `extra_info.task_type` is absent, it SHALL use `data_source` as the fallback task indicator

### Requirement: Local CDM Service Contract
The CDM runtime reward SHALL call a local HTTP service that exposes health and scoring endpoints for online GRPO training.

#### Scenario: Health endpoint reports readiness
- **GIVEN** the CDM service is running locally
- **WHEN** a client calls `GET /health`
- **THEN** the response SHALL indicate whether the service is healthy
- **AND** it SHALL include service name, service version, renderer identifier, and browser readiness

#### Scenario: Score endpoint returns structured result
- **GIVEN** a client sends predicted and reference LaTeX to `POST /score`
- **WHEN** the service scores the pair
- **THEN** the response SHALL include a normalized score and diagnostic fields for render status, parse status, timeout status, and fallback usage

#### Scenario: Batch-shaped scoring is supported
- **GIVEN** a client sends multiple prediction/reference pairs to `POST /score_batch`
- **WHEN** the service scores the batch
- **THEN** the response SHALL return one structured result per input item
- **AND** each output item SHALL preserve the corresponding input identifier

### Requirement: CDM Algorithm Semantics
The CDM service SHALL implement CDM-style visual matching semantics based on LaTeX tokenization, rendered token geometry, assignment, geometric filtering, and F1 scoring.

#### Scenario: Identical formulas score high
- **GIVEN** a valid prediction LaTeX string that is visually equivalent to the reference LaTeX string
- **WHEN** the CDM service scores the pair
- **THEN** the normalized score SHALL be high and bounded by `1.0`

#### Scenario: Different formulas score lower
- **GIVEN** a valid prediction LaTeX string that differs visually or structurally from the reference LaTeX string
- **WHEN** the CDM service scores the pair
- **THEN** the normalized score SHALL be lower than the score for an equivalent pair under the same service version

#### Scenario: Token geometry contributes to matching
- **GIVEN** rendered reference and prediction formulas with identifiable token bounding boxes
- **WHEN** the CDM service computes a score
- **THEN** token identity, token position, and token order SHALL contribute to matching before F1 is computed

### Requirement: Per-sample CDM Failure Semantics
Per-sample CDM failures during training SHALL return zero reward with diagnostics instead of raising an exception to the training loop.

#### Scenario: Invalid predicted LaTeX
- **GIVEN** a Formula sample whose model prediction cannot be parsed or rendered
- **WHEN** the CDM runtime reward scores the sample
- **THEN** it SHALL return `score = 0.0` and `reward_total = 0.0`
- **AND** the result SHALL include diagnostic fields describing the parse or render failure

#### Scenario: Service call fails for one sample
- **GIVEN** the CDM service times out or returns a malformed response for one sample during training
- **WHEN** the CDM runtime reward handles the failure
- **THEN** it SHALL return `score = 0.0` and `reward_total = 0.0`
- **AND** it SHALL include compact error diagnostics without crashing the GRPO run
