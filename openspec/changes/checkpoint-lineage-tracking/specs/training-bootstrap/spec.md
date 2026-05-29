# Delta for Training Bootstrap

## ADDED Requirements

### Requirement: Artifact-Based Model Path Resolution

The training launch layer SHALL accept either a filesystem path or a WandB artifact ID as the model path input.

#### Scenario: Launch with WandB artifact ID

- GIVEN a WandB artifact ID in the form `name:alias` or `name:version`
- WHEN the training launcher resolves the model path
- THEN the system SHALL fetch the artifact metadata, read `checkpoint_dir_rel`, join with `CKPT_ROOT`, and use the resulting absolute path as the model path

#### Scenario: Launch with filesystem path (backward compatible)

- GIVEN a raw filesystem path as the model path
- WHEN the training launcher resolves the model path
- THEN the system SHALL return the path unchanged and attempt to match it to an existing artifact for lineage tracking

### Requirement: Checkpoint Lineage Registration

The training code SHALL register each saved checkpoint as a WandB artifact with lineage metadata.

#### Scenario: Checkpoint save during SFT

- GIVEN a training run has started and the tracker is initialized
- WHEN a checkpoint is saved at global step N
- THEN the system SHALL create a `model-checkpoint` artifact with metadata including `checkpoint_dir_rel`, `global_step`, `training_mode`, `source_artifact`, and aliases `latest` and `step-N`

### Requirement: Source Model Registration

The training code SHALL register the input model as a consumed WandB artifact at run start.

#### Scenario: Seed model auto-registration

- GIVEN a model path that does not match any existing WandB artifact
- WHEN the tracker registers the source model
- THEN the system SHALL auto-register it as a seed model artifact with `is_seed=True`, `global_step=0`, and alias `seed`

### Requirement: Graceful WandB Degradation

The training code SHALL continue operating normally when WandB is unavailable.

#### Scenario: WandB offline during checkpoint save

- GIVEN WandB is unreachable or in offline mode
- WHEN a checkpoint save occurs
- THEN the system SHALL log a warning, skip artifact registration, and proceed with normal checkpoint file saving

### Requirement: Dataset Lineage

The training code SHALL register consumed datasets as WandB reference artifacts.

#### Scenario: Dataset registration

- GIVEN a training run with train and validation parquet files
- WHEN the tracker registers datasets
- THEN the system SHALL create `training-dataset` reference artifacts and call `run.use_artifact()` for each

## MODIFIED Requirements

### Requirement: Checkpoint Artifact Metadata

The checkpoint metadata SHALL include relative paths instead of absolute paths.

#### Scenario: Multi-node path portability

- GIVEN a checkpoint saved with `CKPT_ROOT=/mnt/nfs/checkpoints`
- WHEN the same checkpoint is resolved on a node with `CKPT_ROOT=/data/checkpoints`
- THEN the system SHALL correctly locate the checkpoint using `CKPT_ROOT` + `checkpoint_dir_rel`

## REMOVED Requirements

_None._
