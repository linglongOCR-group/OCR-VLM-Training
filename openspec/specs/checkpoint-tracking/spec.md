# Checkpoint Tracking Specification

## Purpose

Define the current behavior for checkpoint save, storage, resume, and metadata registration during SFT and GRPO training. This spec covers the deployed baseline; the proposed WandB artifact lineage system is tracked as a separate change in `openspec/changes/checkpoint-lineage-tracking/`.

## Requirements

### Requirement: Periodic Checkpoint Saving

The system SHALL save checkpoints on a configured interval during training.

#### Scenario: Checkpoint save triggered by step count

- GIVEN a training run with checkpoint save interval configured at every 100 steps
- WHEN the training loop reaches global step 100
- THEN the system SHALL save a checkpoint containing model state, optimizer state, and trainer state

### Requirement: Local Filesystem Storage

Checkpoint files SHALL be stored on local disk or mounted shared filesystem.

#### Scenario: Checkpoint directory structure

- GIVEN a training run with `CKPTS_DIR=/mnt/ckpts/ocr-vlm/grpo-smoke`
- WHEN a checkpoint is saved at global step 10
- THEN the system SHALL write checkpoint files to `/mnt/ckpts/ocr-vlm/grpo-smoke/global_step_10/`

### Requirement: No Cloud Upload of Checkpoint Payloads

The system SHALL NOT upload checkpoint model weight files to WandB cloud storage.

#### Scenario: Reference artifact registration

- GIVEN a checkpoint saved locally
- WHEN checkpoint metadata is registered to WandB
- THEN the system SHALL register only metadata as a reference artifact using `file://...` URIs
- AND checkpoint payload files SHALL NOT be uploaded

### Requirement: Checkpoint Metadata Fields

Each checkpoint artifact SHALL record at minimum: checkpoint logical name, filesystem URI, global step, epoch, training mode, model path, save timestamp, optimizer state inclusion flag, trainer state inclusion flag, resume compatibility version, config hash, and git commit.

#### Scenario: Metadata registration

- GIVEN a checkpoint saved at global step 10 in GRPO mode
- WHEN metadata is registered to WandB
- THEN the artifact SHALL contain `training_mode: grpo`, `global_step: 10`, `model_id` matching the base model, and a `config_hash` derived from the training configuration

### Requirement: Resume From Latest Checkpoint

The system SHALL support automatic resume from the latest valid checkpoint.

#### Scenario: Automatic resume after interruption

- GIVEN a training run that saved checkpoints at steps 10 and 20
- AND the run was interrupted after step 25
- WHEN the run is restarted with resume enabled
- THEN the system SHALL load the checkpoint from step 20 and continue training from step 21

### Requirement: Resume From Explicit Path

The system SHALL support manual resume from an explicit checkpoint path.

#### Scenario: Manual resume selection

- GIVEN a checkpoint at `/mnt/ckpts/ocr-vlm/grpo-smoke/global_step_10`
- WHEN the run is started with `RESUME_FROM_PATH` set to that path
- THEN the system SHALL load that specific checkpoint regardless of whether newer checkpoints exist

### Requirement: Resume Disabled Mode

The system SHALL support starting a fresh run without resume.

#### Scenario: Fresh start

- GIVEN an existing checkpoint directory from a previous run
- WHEN a new run is started with resume disabled
- THEN the system SHALL initialize training from the base model without loading any checkpoint

### Requirement: Checkpoint Validity Markers

Each saved checkpoint SHALL include metadata to verify usability for resume, including a save completion marker, trainer state presence indicator, config compatibility marker, and save timestamp.

#### Scenario: Incomplete checkpoint detection

- GIVEN a checkpoint directory where the save was interrupted
- WHEN the resume logic attempts to load the checkpoint
- THEN the system SHALL detect the missing completion marker and skip that checkpoint in favor of the next most recent valid checkpoint

### Requirement: Checkpoint Metadata CLI

The system SHALL provide a CLI command to register checkpoint metadata to WandB as a reference artifact.

#### Scenario: Manual metadata registration

- GIVEN a checkpoint directory `/mnt/ckpts/ocr-vlm/grpo-smoke/global_step_10`
- WHEN the user runs `python -m verl_plugins.callbacks.save_and_eval register-checkpoint --checkpoint-dir ... --global-step 10 --training-mode grpo`
- THEN the system SHALL create a WandB reference artifact with metadata and a `file://` URI pointing to the local path
