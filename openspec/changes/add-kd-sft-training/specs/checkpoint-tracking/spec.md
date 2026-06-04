## ADDED Requirements

### Requirement: KD-SFT Checkpoint Metadata

KD-SFT checkpoints SHALL record metadata that identifies the training mode, student model source, teacher model source, and KD configuration needed to interpret the student checkpoint.

#### Scenario: Saving KD-SFT checkpoint metadata

- **GIVEN** a KD-SFT training run saves a student checkpoint
- **WHEN** checkpoint metadata is written or registered
- **THEN** the metadata SHALL include `training_mode: kd_sft`
- **AND** the metadata SHALL include the student model identifier, teacher model identifier, global step, config hash, git commit, and checkpoint filesystem URI
- **AND** the checkpoint payload SHALL NOT include teacher optimizer state or teacher scheduler state

#### Scenario: Resuming KD-SFT from checkpoint

- **GIVEN** a KD-SFT run resumes from a saved checkpoint
- **WHEN** the checkpoint is loaded
- **THEN** the student model, optimizer state, trainer state, and global step SHALL be restored from the checkpoint
- **AND** KD loss schedules SHALL use the restored global step for subsequent weight evaluation
- **AND** the teacher model SHALL be loaded from the configured teacher source rather than from student checkpoint payloads

#### Scenario: Registering KD-SFT reference artifact

- **GIVEN** a KD-SFT checkpoint saved locally
- **WHEN** checkpoint metadata is registered to W&B as a reference artifact
- **THEN** the artifact SHALL use a local `file://` URI for checkpoint payloads
- **AND** the artifact metadata SHALL distinguish student checkpoint path from teacher model source
