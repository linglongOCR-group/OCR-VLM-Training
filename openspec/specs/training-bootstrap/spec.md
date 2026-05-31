# Training Bootstrap Specification

## Purpose

This specification defines the multi-node SFT and GRPO training system for OCR VLM fine-tuning. It covers training execution, W&B experiment tracking, checkpoint lifecycle, resume behavior, and configuration. The system uses VERL as the training framework, Ray for multi-node orchestration, W&B for tracking, and local or mounted shared filesystem for checkpoint storage.
## Requirements
### Requirement: Multi-Node SFT Training

The system SHALL launch multi-node SFT training jobs from configuration without code edits, using distributed coordination appropriate to the installed training framework.

#### Scenario: Launching an SFT training job

- **GIVEN** a valid YAML config and a shell launcher script
- **WHEN** an engineer sets MODEL_PATH, TRAIN_FILE, and optional VAL_FILE environment variables and invokes the SFT launch script
- **THEN** training initializes across all configured nodes
- **AND** the train parquet dataset is loaded through a dataset adapter that exposes prompt, target, and image fields
- **AND** the val parquet dataset is loaded if VAL_FILE is provided
- **AND** training proceeds until the configured total_epochs or total_training_steps are reached

#### Scenario: Running periodic validation during SFT

- **WHEN** test_freq is set to a positive integer N
- **THEN** validation runs every N steps
- **AND** validation loss is logged to all configured logger backends
- **AND** val_before_train triggers an initial validation pass before the first training step when set to true

### Requirement: Multi-Node GRPO Training

The system SHALL launch multi-node GRPO training jobs from configuration without code edits, using Ray for multi-node orchestration and vLLM for rollout generation.

#### Scenario: Launching a GRPO training job

- **GIVEN** a valid YAML config and the GRPO launch script
- **WHEN** an engineer sets MODEL_PATH and TRAIN_FILE environment variables and invokes the GRPO launch script
- **THEN** training initializes with actor, rollout, reference, and reward components
- **AND** the train parquet dataset is loaded through a dataset adapter that exposes prompt, image, and ground truth fields
- **AND** rollouts are generated via vLLM with the configured tensor_model_parallel_size
- **AND** training proceeds until total_epochs or total_training_steps are reached

#### Scenario: Producing rollouts and computing rewards

- **WHEN** the GRPO training loop generates model outputs for a batch
- **THEN** the reward function computes a scalar reward per sample by comparing predicted text to ground truth
- **AND** reward mean and standard deviation are recorded for the batch
- **AND** response length statistics are recorded for the batch

#### Scenario: Logging GRPO policy diagnostics

- **WHEN** use_kl_loss is enabled in the actor configuration
- **THEN** KL divergence metrics between the current policy and reference policy are logged
- **AND** entropy or other policy diagnostics are logged if available from the VERL trainer

### Requirement: Shared Reward Function

The system SHALL provide a single Normalized Levenshtein Distance reward implementation used across all GRPO tasks.

#### Scenario: Computing per-sample reward

- **WHEN** the reward function receives a prediction string, a ground_truth string, an optional task_type, and optional metadata
- **THEN** it returns a dict containing reward_total, reward_name, and reward_version
- **AND** reward_total is a normalized edit distance score in a stable documented range
- **AND** the function handles empty predictions and empty targets safely without error

#### Scenario: Reward function determinism

- **WHEN** the same prediction and ground_truth inputs are provided
- **THEN** the reward function returns identical reward_total values across invocations

#### Scenario: Reward function interface extensibility

- **GIVEN** a reward function with signature reward(prediction, ground_truth, task_type, metadata) -> dict
- **WHEN** future task-specific rewards are added
- **THEN** the GRPO training loop does not require modification to dispatch to new reward implementations
- **AND** the reward version is recorded in every reward output for traceability

### Requirement: W&B Run Metadata

Every training run SHALL record run-level metadata to W&B at initialization.

#### Scenario: Recording run metadata on initialization

- **WHEN** a training run starts with wandb in the logger list
- **THEN** W&B records project name, experiment name, and a unique run ID
- **AND** the config snapshot is logged as the run configuration
- **AND** the following metadata is captured: model identifier, dataset identifiers, training mode (sft or grpo), node count, device count per node, and launch timestamp

#### Scenario: Recording code version

- **WHEN** a training run initializes
- **THEN** the git commit hash of the working tree is captured and logged to W&B
- **AND** the run can be traced back to a specific code revision

### Requirement: W&B Training Metrics

The system SHALL log scalar training metrics to W&B at each logged step.

#### Scenario: Logging shared metrics during training

- **WHEN** a training step completes
- **THEN** global step, learning rate, and train loss are logged to W&B
- **AND** checkpoint save step is logged when a checkpoint is written

#### Scenario: Logging validation metrics

- **WHEN** a validation pass completes
- **THEN** validation loss is logged to W&B
- **AND** throughput-related metrics are logged if available

#### Scenario: Logging GRPO-specific metrics

- **WHEN** a GRPO training step completes
- **THEN** reward mean, reward standard deviation, response length statistics, and rollout count are logged
- **AND** KL-related metrics and policy diagnostics are logged if enabled by the actor configuration

### Requirement: W&B Rollout and Validation Logging

The system SHALL log selected rollout and validation samples to W&B for inspection.

#### Scenario: Logging validation generation samples

- **WHEN** log_val_generations is set to a positive integer N
- **THEN** N sample generations are logged per validation cycle
- **AND** each logged sample includes input prompt, target or reference text, and model output

#### Scenario: Logging GRPO rollout samples

- **WHEN** a GRPO training step produces rollouts
- **THEN** selected rollout samples are logged including input prompt, target, model output, reward value, task type, and sample ID
- **AND** samples are stored as W&B tables or equivalent structured logging

### Requirement: Checkpoint Save Policy

The system SHALL save checkpoints on a configured interval, containing all state required for training continuation.

#### Scenario: Saving a checkpoint on interval

- **WHEN** the global step is a multiple of save_freq
- **THEN** a checkpoint is saved containing model state, optimizer state, and trainer extra state
- **AND** the checkpoint is written to the configured local_root directory

#### Scenario: Configuring checkpoint contents

- **WHEN** save_contents is configured
- **THEN** only the specified components are included in the checkpoint
- **AND** the default contents are model, optimizer, and extra

### Requirement: Checkpoint Storage

Checkpoint files SHALL be stored on local disk or a mounted shared filesystem accessible to all training nodes.

#### Scenario: Storing checkpoints on shared filesystem

- **WHEN** CKPTS_DIR is set to a path on a mounted shared filesystem
- **THEN** all nodes write and read checkpoints from that shared path
- **AND** checkpoint files are never uploaded to W&B cloud storage

#### Scenario: Registering checkpoint metadata to W&B

- **WHEN** register_wandb_reference is true and a checkpoint is saved
- **THEN** checkpoint metadata is registered to W&B as a reference artifact with a file:// URI pointing to the local storage location
- **AND** the artifact records checkpoint name, global step, epoch, training mode, model path, save timestamp, config hash, and git commit

### Requirement: Checkpoint Resume

The system SHALL support resuming training from a saved checkpoint after interruption.

#### Scenario: Resuming from the latest checkpoint automatically

- **GIVEN** resume_mode is set to auto
- **WHEN** a training run starts and a valid checkpoint exists in the checkpoint directory
- **THEN** training resumes from the latest valid checkpoint
- **AND** training continues from the saved global step rather than restarting

#### Scenario: Resuming from an explicit checkpoint path

- **GIVEN** resume_mode is set to a specific path
- **WHEN** resume_from_path is set to a valid checkpoint directory
- **THEN** training resumes from that explicit checkpoint
- **AND** all model, optimizer, and trainer state are restored from the specified checkpoint

#### Scenario: Restarting without resume

- **GIVEN** resume_mode is set to disable or resume_from_path is null and no checkpoint is found
- **WHEN** a training run starts
- **THEN** training begins from scratch using the initial model weights
- **AND** no previous checkpoint state is loaded

### Requirement: Checkpoint Validity

Each saved checkpoint SHALL include or be accompanied by metadata sufficient to verify it is usable for resume. The metadata SHOULD include a save completion marker, trainer state presence indicator, config compatibility marker, and save timestamp.

#### Scenario: Verifying checkpoint validity before resume

- **WHEN** the system evaluates a checkpoint directory for resume
- **THEN** it checks for a save completion marker indicating the checkpoint was fully written
- **AND** it verifies trainer state presence
- **AND** it verifies config compatibility using resume_compatibility_version
- **AND** a checkpoint that fails any validity check is skipped during auto-resume

#### Scenario: Handling incomplete checkpoints

- **WHEN** a checkpoint directory is found without a completion marker
- **THEN** the checkpoint is treated as invalid and not used for resume
- **AND** the system proceeds to the next most recent checkpoint or starts fresh

### Requirement: Configuration System

All training behavior SHALL be controlled by configuration without requiring code changes per run.

#### Scenario: Configuring a training run

- **WHEN** an engineer prepares a training run
- **THEN** all operational parameters are set via YAML config files or environment variable overrides
- **AND** no source code edits are required to change model, data, hyperparameters, logging, or checkpoint settings

#### Scenario: Supporting config families

- **WHEN** the config system is initialized
- **THEN** the following config groups are available: trainer (cluster/runtime), model, data, sft trainer settings, grpo trainer settings, reward, wandb, checkpoint, and logging/validation
- **AND** each config family has a corresponding base configuration file

#### Scenario: Overriding config via environment variables

- **WHEN** an environment variable is set that corresponds to a config field
- **THEN** the environment variable value takes precedence over the YAML default
- **AND** supported overrides include MODEL_PATH, TRAIN_FILE, VAL_FILE, CKPTS_DIR, WANDB_PROJECT, EXPERIMENT_NAME, NNODES, NPUS_PER_NODE, RESUME_MODE, SAVE_FREQ, and TEST_FREQ

### Requirement: Config Snapshot and Reproducibility

The system SHALL save the effective configuration with every run to enable full reproducibility.

#### Scenario: Reproducing a training run

- **WHEN** a training run completes or is inspected after the fact
- **THEN** the run can be reproduced from the code revision, config snapshot, dataset version identifiers, model identifier, and checkpoint reference metadata
- **AND** the config snapshot is logged to W&B as part of the run metadata

### Requirement: Deterministic Experiment Naming

The system SHALL support deterministic experiment naming for run identification.

#### Scenario: Setting experiment name from environment

- **WHEN** EXPERIMENT_NAME is set as an environment variable
- **THEN** the W&B experiment name and checkpoint directory reflect that value
- **AND** runs with the same experiment name are grouped together in W&B

### Requirement: Dataset Loading for Training

The training system SHALL load parquet-based datasets using adapter classes that provide a stable interface to the trainer.

#### Scenario: Loading SFT training data

- **WHEN** TRAIN_FILE is set to one or more parquet file paths
- **THEN** the dataset adapter loads records from the specified files
- **AND** each record exposes prompt, target, and image fields expected by the model
- **AND** OCR_DATA_ROOT is prepended to relative image paths when set

#### Scenario: Loading GRPO training data

- **WHEN** TRAIN_FILE is set to one or more parquet file paths for GRPO
- **THEN** the dataset adapter loads records with prompt, image, and ground truth fields
- **AND** data_source is used as the reward dispatch key
- **AND** the dataset adapter handles multimodal samples including image-based inputs

#### Scenario: Handling optional validation data

- **WHEN** VAL_FILE is null or not set
- **THEN** validation is skipped or disabled depending on the training mode
- **AND** val_before_train and test_freq are adjusted accordingly

### Requirement: Training Mode Detection

The system SHALL clearly distinguish between SFT and GRPO training modes via configuration and entry point.

#### Scenario: Selecting SFT mode

- **WHEN** the SFT launch script is invoked
- **THEN** training uses the SFT entry point
- **AND** the training_mode metadata value is recorded as sft in W&B

#### Scenario: Selecting GRPO mode

- **WHEN** the GRPO launch script is invoked
- **THEN** training uses the GRPO entry point
- **AND** the training_mode metadata value is recorded as grpo in W&B

### Requirement: Shared Infrastructure Across Modes

SFT and GRPO training modes SHALL share common infrastructure for dataset format conventions, W&B integration, checkpoint path conventions, and experiment naming.

#### Scenario: Reusing W&B integration

- **WHEN** trainer.logger includes wandb
- **THEN** both SFT and GRPO runs initialize W&B with the same project, entity, and mode settings from the wandb config group
- **AND** both modes use the same checkpoint metadata registration mechanism

#### Scenario: Reusing checkpoint directory conventions

- **WHEN** a checkpoint is saved in either SFT or GRPO mode
- **THEN** the checkpoint directory follows the same structure under CKPTS_DIR
- **AND** the same resume logic applies regardless of training mode

### Requirement: Preserve Literal Assistant Media Tokens

The SFT runtime dataset wrapper SHALL preserve literal media-token text in non-user messages without allowing it to be interpreted as a multimodal placeholder.

#### Scenario: Assistant text contains `<video>`

- GIVEN an SFT row whose user message contains an image placeholder
- AND whose assistant text contains the literal string `<video>`
- WHEN the runtime dataset builds VERL multi-turn messages
- THEN the user image placeholder SHALL still become an image segment
- AND the assistant message SHALL remain a text segment containing `<video>`

#### Scenario: Assistant text contains `<image>`

- GIVEN an SFT row whose assistant text contains the literal string `<image>`
- WHEN the runtime dataset builds VERL multi-turn messages
- THEN the assistant text SHALL remain text and SHALL NOT create an additional image segment

