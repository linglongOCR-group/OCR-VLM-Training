## ADDED Requirements

### Requirement: Multi-Node KD-SFT Training

The system SHALL launch multi-node KD-SFT training jobs from configuration without code edits, using the OCR/VLM SFT dataset path and distributed coordination appropriate to VERL FSDP training.

#### Scenario: Launching a KD-SFT training job

- **GIVEN** a valid KD-SFT YAML config and a KD-SFT shell launcher script
- **WHEN** an engineer sets student model path, teacher model path, train files, and required cluster environment variables and invokes the KD-SFT launch script
- **THEN** KD-SFT training SHALL initialize across all configured nodes
- **AND** the train parquet dataset SHALL be loaded through the OCR SFT dataset adapter
- **AND** each distributed rank SHALL initialize a trainable student and frozen forward-only teacher
- **AND** training SHALL proceed until the configured total epochs or total training steps are reached

#### Scenario: Running periodic validation during KD-SFT

- **WHEN** KD-SFT `test_freq` is set to a positive integer N
- **THEN** validation SHALL run every N steps using the student model
- **AND** validation loss SHALL be logged to all configured logger backends
- **AND** `val_before_train` SHALL trigger an initial validation pass before the first training step when set to true

### Requirement: KD-SFT Training Metrics

The system SHALL log KD-SFT scalar metrics that distinguish raw losses, weighted losses, and active loss weights.

#### Scenario: Logging KD-SFT loss components

- **WHEN** a KD-SFT training step completes
- **THEN** the logged metrics SHALL include raw SFT loss, raw logits KD loss, raw hidden KD loss, weighted SFT loss, weighted logits KD loss, weighted hidden KD loss, active SFT weight, active logits KD weight, active hidden KD weight, and KD temperature

#### Scenario: Logging KD-SFT with existing tracking backends

- **WHEN** KD-SFT is configured with console or W&B logging
- **THEN** KD-SFT metrics SHALL be emitted through the same tracking mechanism as existing SFT metrics

### Requirement: KD-SFT Configuration

The training configuration system SHALL support KD-SFT behavior through YAML config files and Hydra overrides.

#### Scenario: Configuring a KD-SFT run

- **WHEN** an engineer prepares a KD-SFT run
- **THEN** teacher model path, student model path, KD loss settings, hidden layer map, loss schedules, precheck mode, data paths, cluster topology, checkpoint settings, and logging settings SHALL be configurable without source code edits

#### Scenario: Overriding KD-SFT config

- **WHEN** an engineer passes Hydra overrides to the KD-SFT launch script
- **THEN** those overrides SHALL apply to the effective KD-SFT config
- **AND** the launch script SHALL preserve an argument tail for arbitrary Hydra overrides
