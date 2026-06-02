## ADDED Requirements

### Requirement: KD-SFT Training Mode

The system SHALL provide a KD-SFT training mode for OCR/VLM SFT data that trains a student model with SFT loss, logits distillation loss, and hidden-state distillation loss from a frozen teacher model.

#### Scenario: Run KD-SFT with all v1 losses

- **GIVEN** valid OCR/VLM SFT train data, a compatible student model, a compatible teacher model, and KD-SFT enabled
- **WHEN** the KD-SFT trainer runs a training step
- **THEN** it SHALL compute SFT loss, logits KD loss, hidden KD loss, and the weighted total loss
- **AND** only the student model SHALL receive gradients
- **AND** the teacher model SHALL remain frozen

#### Scenario: Preserve existing SFT behavior

- **GIVEN** an engineer launches the existing SFT trainer and script
- **WHEN** no KD-SFT entrypoint is used
- **THEN** existing SFT behavior SHALL remain unchanged

### Requirement: FSDP Teacher Student Topology

KD-SFT SHALL run a co-located FSDP teacher and FSDP student on every distributed rank, with teacher outputs consumed locally on the same rank.

#### Scenario: Use one local microbatch for both models

- **GIVEN** a KD-SFT training rank receives a local microbatch
- **WHEN** teacher and student forwards run
- **THEN** both models SHALL consume the same input ids, attention mask, position ids, image tensors, image metadata, and loss mask
- **AND** the system SHALL NOT create a separate teacher dataloader

#### Scenario: Keep teacher tensors rank-local

- **WHEN** a teacher forward produces logits or hidden activations
- **THEN** those teacher tensors SHALL be consumed on the same rank for KD loss computation
- **AND** the system SHALL NOT all-gather teacher logits or teacher hidden activations across ranks

#### Scenario: Exclude teacher from optimizer state

- **WHEN** KD-SFT initializes the teacher model
- **THEN** the teacher SHALL be eval-only and forward-only
- **AND** the teacher SHALL NOT create optimizer state, scheduler state, or checkpoint payloads

### Requirement: Teacher Student Compatibility

KD-SFT SHALL require teacher and student models to share the same tokenizer, vocabulary size, hidden size, VLM input contract, and output-token semantics.

#### Scenario: Reject incompatible tokenizer or vocabulary

- **GIVEN** a KD-SFT config whose teacher and student tokenizers or vocabulary sizes differ
- **WHEN** startup precheck runs
- **THEN** the precheck SHALL fail before training starts

#### Scenario: Reject incompatible hidden size

- **GIVEN** a KD-SFT config whose teacher and student hidden sizes differ
- **WHEN** hidden KD is enabled and startup precheck runs
- **THEN** the precheck SHALL fail before training starts

### Requirement: Response Only KD Masking

KD-SFT SHALL apply SFT loss, logits KD loss, and hidden KD loss only to response/output tokens using the same shifted response-mask convention as the SFT trainer.

#### Scenario: Exclude prompt and padding tokens

- **GIVEN** a batch containing prompt tokens, response tokens, image placeholder tokens, and padding
- **WHEN** KD-SFT computes SFT, logits KD, and hidden KD losses
- **THEN** only response/output token positions SHALL contribute to those losses
- **AND** prompt, image placeholder, system/template, and padding positions SHALL be excluded

#### Scenario: Detect empty response supervision

- **GIVEN** a KD-SFT microbatch with no response-supervised tokens
- **WHEN** the trainer attempts to compute KD losses
- **THEN** the system SHALL report the empty response-token condition with rank and step context

### Requirement: Top-K Logits Distillation

KD-SFT SHALL support top-k teacher logits distillation as a first-class logits KD mode.

#### Scenario: Compute top-k teacher targets

- **GIVEN** logits KD is enabled with `mode: top_k`
- **WHEN** the teacher forward produces logits
- **THEN** KD-SFT SHALL retain teacher top-k token indices and teacher top-k log probabilities for response-supervised positions
- **AND** KD-SFT SHALL avoid retaining full teacher logits longer than needed for top-k target extraction

#### Scenario: Use renormalized top-k forward KL

- **GIVEN** `kd.logits.loss_type` is `renormalized_top_k_forward_kl`
- **WHEN** logits KD loss is computed
- **THEN** teacher and student probabilities over the teacher-selected top-k tokens SHALL be normalized over those top-k tokens before the forward KL is computed

#### Scenario: Use truncated forward KL

- **GIVEN** `kd.logits.loss_type` is `truncated_forward_kl`
- **WHEN** logits KD loss is computed
- **THEN** teacher top-k probabilities SHALL be used as a sparse teacher target
- **AND** student probabilities SHALL use the full student softmax denominator

### Requirement: Selected Layer Hidden Distillation

KD-SFT SHALL support selected-layer hidden-state distillation using an explicit layer map and selected-layer hooks.

#### Scenario: Require explicit hidden layer map

- **GIVEN** hidden KD is enabled
- **WHEN** startup precheck runs
- **THEN** `kd.hidden.layer_map` SHALL be present
- **AND** each mapping entry SHALL identify a valid student hidden index and a valid teacher hidden index

#### Scenario: Capture only mapped hidden layers

- **GIVEN** a valid hidden layer map
- **WHEN** teacher and student forwards run
- **THEN** KD-SFT SHALL capture only hidden activations for mapped teacher and student layers
- **AND** unmapped layer activations SHALL NOT be retained for hidden KD

#### Scenario: Support hidden loss types

- **GIVEN** hidden KD is enabled
- **WHEN** `kd.hidden.loss_type` is configured as `normalized_mse` or `cosine`
- **THEN** KD-SFT SHALL compute the corresponding hidden alignment loss over response-supervised token positions

#### Scenario: Weight mapped layers equally

- **GIVEN** a hidden layer map containing multiple mapping entries
- **WHEN** hidden KD loss is computed
- **THEN** each mapped layer pair SHALL contribute equally to the hidden KD raw loss

### Requirement: KD Loss Weight Schedules

KD-SFT SHALL compute independent SFT, logits KD, and hidden KD weights from global optimizer step.

#### Scenario: Evaluate schedules deterministically

- **GIVEN** all ranks have the same restored global optimizer step and KD schedule config
- **WHEN** loss weights are evaluated
- **THEN** every rank SHALL compute the same `lambda_sft`, `lambda_logit`, and `lambda_hidden` without additional distributed communication

#### Scenario: Support required schedule types

- **WHEN** KD-SFT validates schedule config
- **THEN** it SHALL support `constant`, `piecewise_linear`, `linear_warmup_constant`, and `linear_warmup_hold_decay`

#### Scenario: Use one weight set per optimizer step

- **GIVEN** a global optimizer step is split into multiple microbatches
- **WHEN** KD-SFT computes microbatch losses for that optimizer step
- **THEN** each microbatch SHALL use the same scheduled KD loss weights for that optimizer step

### Requirement: KD Metrics

KD-SFT SHALL log raw losses, weighted losses, active loss weights, and KD temperature for training observability.

#### Scenario: Log decomposed KD losses

- **WHEN** a KD-SFT training step logs metrics
- **THEN** metrics SHALL include `train/loss_sft_raw`, `train/loss_logit_raw`, `train/loss_hidden_raw`, `train/loss_sft_weighted`, `train/loss_logit_weighted`, and `train/loss_hidden_weighted`
- **AND** metrics SHALL include `train/lambda_sft`, `train/lambda_logit`, `train/lambda_hidden`, and `train/kd_temperature`

### Requirement: KD Precheck

KD-SFT SHALL run a mandatory config precheck before training starts and SHALL expose an optional dry-run batch precheck.

#### Scenario: Run mandatory config precheck

- **GIVEN** KD-SFT training is requested
- **WHEN** the trainer starts
- **THEN** it SHALL validate teacher path, student path, tokenizer compatibility, vocabulary size, hidden size, VLM processor compatibility, top-k config, hidden layer map, schedules, and teacher frozen state before training starts

#### Scenario: Run optional dry-run batch precheck

- **GIVEN** KD-SFT is configured with dry-run batch precheck
- **WHEN** precheck runs
- **THEN** it SHALL execute one real batch through teacher forward, student forward, top-k logits KD target extraction, hidden hook capture, and total loss composition without starting full training

### Requirement: KD-SFT Hydra Configuration

KD-SFT SHALL be configurable through Hydra under the existing SFT config family.

#### Scenario: Load KD-SFT config

- **WHEN** an engineer uses the KD-SFT Hydra config
- **THEN** the config SHALL expose teacher path, logits KD settings, hidden KD settings, loss schedules, precheck settings, model settings, data settings, checkpoint settings, and trainer settings
- **AND** command-line Hydra overrides SHALL be able to override KD-SFT config values
