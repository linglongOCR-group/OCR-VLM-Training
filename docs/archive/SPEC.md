# Multi-Node SFT + GRPO Training Bootstrap Spec

## 1. Document Status

**Status:** Draft v0.1
**Audience:** Training / infra / data / platform engineers
**Language:** English
**Purpose:** Define the minimum viable system required to bring up and validate a multi-node SFT + GRPO training workflow for an OCR VLM on Ascend-based infrastructure, using VERL as the training framework and Weights & Biases (W&B) for experiment tracking.

---

## 2. Objective

The immediate objective is to **bring up and run through** the end-to-end workflow for:

1. **Multi-node SFT training**
2. **Multi-node GRPO training**
3. **W&B logging and metadata tracking**
4. **Checkpoint save / resume**
5. **Reference-artifact-based checkpoint registration in W&B**

This phase is explicitly a **bootstrap / enablement phase**, not a performance-optimization or reward-design phase.

The main deliverable is a stable, reproducible engineering baseline that proves:

* training code can run on multiple nodes,
* configuration files are sufficient to launch experiments,
* W&B captures the required metrics and metadata,
* checkpoints can be resumed after interruption,
* checkpoint metadata can be versioned in W&B without uploading model weights to the cloud,
* all task rewards in GRPO can be computed via a shared Normalized Levenshtein Distance implementation.

---

## 3. Scope

### 3.1 In Scope

* Multi-node SFT launch and successful completion on a small-to-medium validation workload
* Multi-node GRPO launch and successful completion on a small-to-medium validation workload
* Training-side code required to support:

  * dataset loading,
  * prompt / target formatting,
  * reward invocation,
  * logging hooks,
  * checkpoint hooks,
  * resume logic,
  * W&B integration
* Configuration files required to support:

  * cluster / runtime settings,
  * data settings,
  * model settings,
  * SFT settings,
  * GRPO settings,
  * checkpoint settings,
  * W&B settings
* W&B integration for:

  * run metadata,
  * scalar metrics,
  * rollout samples / validation generations,
  * checkpoint metadata as reference artifacts
* Local-only checkpoint storage for model weights and optimizer state
* Resume from latest valid checkpoint
* A single reward implementation used across all GRPO tasks:

  * Normalized Levenshtein Distance

### 3.2 Out of Scope

* Reward shaping beyond Normalized Levenshtein Distance
* Task-specific structural rewards for table / formula / diagram / seal tasks
* Throughput, MFU, or NPU utilization optimization
* Backend migration optimization (e.g. FSDP to Megatron / MindSpeed tuning)
* Production-grade dataset versioning platform
* Full benchmark suite design
* Model architecture redesign
* Large-scale hyperparameter search
* Inference serving

---

## 4. Background and Assumptions

This project targets an OCR VLM training workflow that will eventually support structured document understanding tasks. For the current phase, the focus is not model quality maximization, but system readiness.

The system will use:

* **VERL** as the primary training framework
* **Ray** for multi-node orchestration
* **W&B** for logging and experiment tracking
* **Local filesystem / NFS / mounted storage** for checkpoint files
* **W&B reference artifacts** for checkpoint metadata registration

This spec assumes that:

* the cluster environment is already provisioned and reachable,
* multi-node Ray can be started externally or from wrapper scripts,
* model and dataset access are already available,
* container/runtime setup is addressed separately.

---

## 5. Success Criteria

The bootstrap phase is considered successful when all of the following are true:

1. A multi-node SFT training job can be launched from config and run to completion.
2. A multi-node GRPO training job can be launched from config and run to completion.
3. Both jobs write training logs and metadata to W&B.
4. W&B contains:

   * run-level metadata,
   * core scalar metrics,
   * selected rollout / validation samples,
   * checkpoint metadata artifacts.
5. Checkpoint files remain in local or mounted storage only.
6. W&B artifact records contain references to checkpoint locations instead of uploading weights.
7. Training can resume from the latest valid checkpoint after interruption.
8. All GRPO tasks call the same Normalized Levenshtein Distance reward implementation.
9. The system can be launched by configuration without code edits per run.

---

## 6. System Overview

### 6.1 Execution Modes

The system shall support two training modes:

* **SFT mode**
* **GRPO mode**

Both modes shall share as much infrastructure as possible:

* dataset format conventions,
* runtime launch wrappers,
* W&B integration,
* checkpoint path conventions,
* experiment naming,
* multi-node orchestration pattern.

### 6.2 High-Level Components

The minimum system consists of:

1. **Training launch layer**

   * shell or Python wrappers to launch multi-node jobs
2. **Training configuration layer**

   * YAML config files and environment variable conventions
3. **Dataset adapter layer**

   * code that maps parquet records into model-ready inputs
4. **Reward layer**

   * a shared Normalized Levenshtein Distance reward implementation
5. **Logging layer**

   * W&B run initialization and metric logging
6. **Checkpoint layer**

   * local save, local resume, metadata registration to W&B

---

## 7. Functional Requirements

## 7.1 Training Code

The codebase shall provide the minimum functionality needed to support both SFT and GRPO.

### 7.1.1 Shared Requirements

The training code shall:

* load parquet-based datasets,
* support one file or multiple parquet files,
* support image-based multimodal samples,
* map dataset rows into prompt / target / image inputs,
* support deterministic experiment naming,
* log core metrics during training,
* save checkpoints on a configured interval,
* resume from the latest valid checkpoint,
* expose all run-time behavior through config.

### 7.1.2 SFT Requirements

The SFT path shall support:

* multi-node launch,
* train/validation dataset loading,
* loss logging,
* periodic validation,
* periodic checkpoint save,
* resume after interruption.

### 7.1.3 GRPO Requirements

The GRPO path shall support:

* multi-node launch,
* reward calculation from model outputs and ground truth,
* rollout logging,
* reward logging,
* KL / policy metrics logging if enabled by the selected VERL configuration,
* periodic validation,
* periodic checkpoint save,
* resume after interruption.

---

## 7.2 Reward Requirements

### 7.2.1 Temporary Reward Policy

For the current phase, **all GRPO tasks shall use Normalized Levenshtein Distance as the only reward signal**.

This is a temporary engineering simplification.

### 7.2.2 Reward Definition

The reward implementation shall:

* compare predicted text to target text,
* normalize edit distance by target length or agreed normalization rule,
* return a scalar reward in a stable and documented range,
* handle empty predictions and empty targets safely,
* be deterministic,
* expose per-sample reward values for logging and debugging.

### 7.2.3 Reward Interface

A single reward interface shall be defined so that future task-specific rewards can replace or extend the temporary reward without changing the GRPO training loop.

Suggested interface:

```python
reward(prediction, ground_truth, task_type, metadata) -> dict
```

Where the returned dict includes at least:

* `reward_total`
* `reward_name`
* `reward_version`
* optional diagnostic fields such as normalized distance and raw distance

---

## 7.3 W&B Requirements

W&B shall be integrated as the primary experiment tracking system for this phase.

### 7.3.1 Run Metadata

Each run shall record at minimum:

* project name
* experiment name
* run ID
* git commit or code version
* config snapshot
* model identifier
* dataset identifiers / dataset version strings
* training mode (`sft` or `grpo`)
* node count
* device count per node
* launch timestamp

### 7.3.2 Training Metrics

The system shall log the following metrics where applicable:

#### Shared metrics

* global step
* epoch
* learning rate
* train loss
* validation loss
* throughput-related metrics if available
* checkpoint save step

#### GRPO-specific metrics

* reward mean
* reward std
* response length statistics
* rollout count / sampling count
* KL-related metrics if enabled
* entropy or policy diagnostics if available

### 7.3.3 Rollout / Validation Logging

The system shall log selected examples for inspection, including:

* input prompt
* target / reference
* model output
* reward value
* task type
* sample ID

This may be stored as W&B tables or equivalent structured logging.

### 7.3.4 Checkpoint Metadata Tracking

Checkpoint metadata shall be registered to W&B as **reference artifacts**.

The artifact shall track metadata only; checkpoint files themselves shall remain in local or mounted storage.

Each checkpoint artifact shall record at minimum:

* checkpoint logical name
* checkpoint filesystem URI (`file://...`)
* global step
* epoch
* training mode
* model path / model ID
* save timestamp
* whether optimizer state is included
* whether extra trainer state is included
* resume compatibility version
* config hash
* git commit
* optional validation metrics available at save time

### 7.3.5 Offline vs Online Policy

The preferred mode for this phase is:

* W&B enabled for run tracking and metadata,
* checkpoint files stored locally,
* checkpoint metadata logged via reference artifacts.

Pure offline W&B mode is not the default target for this phase unless required by environment constraints.

---

## 7.4 Checkpoint and Resume Requirements

### 7.4.1 Save Policy

Checkpoints shall be saved on a configured interval.

A checkpoint shall include the minimum state needed for training continuation:

* model state
* optimizer state where applicable
* trainer / extra state required for resume

### 7.4.2 Storage Policy

Checkpoint files shall be stored on:

* local disk, or
* mounted shared filesystem, or
* NFS-equivalent storage accessible to the resume flow.

Checkpoint files shall **not** be uploaded to W&B cloud storage.

### 7.4.3 Resume Policy

The system shall support:

* resume from latest checkpoint automatically,
* resume from explicit checkpoint path manually,
* restart with resume disabled.

The selected policy shall be configurable.

### 7.4.4 Checkpoint Validity

Each saved checkpoint should include or be accompanied by minimal metadata to verify it is usable for resume, such as:

* save completion marker,
* trainer state presence,
* config compatibility marker,
* save timestamp.

---

## 7.5 Configuration Requirements

All operational behavior shall be controlled by configuration.

### 7.5.1 Required Config Families

The system shall provide the following config groups:

1. **cluster/runtime**
2. **model**
3. **data**
4. **sft trainer**
5. **grpo trainer**
6. **reward**
7. **wandb**
8. **checkpoint**
9. **logging / validation**

### 7.5.2 Config Principles

Configs shall:

* be human-readable,
* support multi-node settings,
* support environment-specific overrides,
* minimize duplicated values across SFT and GRPO,
* be suitable for source control,
* be saved with every run.

### 7.5.3 Required Runtime Parameters

The config system shall support at minimum:

* number of nodes
* devices per node
* train parquet paths (single or list)
* validation parquet paths (single or list)
* prompt key / image key / target key mapping
* batch size settings
* save frequency
* validation frequency
* validation-before-train toggle
* resume mode
* resume path
* local checkpoint root
* W&B project / experiment naming
* number of logged validation generations
* reward version

---

## 8. Dataset and Input Requirements

### 8.1 Dataset Format

For this phase, the system shall accept parquet-based dataset inputs.

The exact dataset schema may evolve, but the training adapter must expose a stable interface to the model/trainer.

### 8.2 Minimum Dataset Fields

For multimodal OCR-VLM training, the dataset view used by training shall provide at minimum:

* `sample_id`
* `prompt`
* `target`
* `images` or image reference field expected by the model stack
* optional `task_type`
* optional metadata fields used by reward or logging

### 8.3 Multiple Dataset Sources

The system shall support one or more parquet files as input for a run.

The data pipeline for this phase may treat the selected training view as already materialized and ready for loading.

Complex source lineage management is out of scope for this bootstrap phase, but the run metadata should still record dataset version identifiers.

---

## 9. Non-Functional Requirements

### 9.1 Reproducibility

A run must be reproducible from:

* code revision,
* config snapshot,
* dataset version identifiers,
* model identifier,
* checkpoint reference metadata.

### 9.2 Operability

The workflow must be operable by engineers without code changes per run.

Launching a new experiment should require only:

* selecting config,
* setting environment variables,
* starting the job.

### 9.3 Debuggability

The system must make it easy to answer:

* which config was used,
* which dataset version was used,
* which checkpoint was resumed from,
* what reward implementation was active,
* why a run failed or degraded.

### 9.4 Simplicity Over Completeness

If there is a conflict between a simpler implementation and a more general one, the simpler implementation should be preferred for this phase, provided it does not block later extension.

---

## 10. Deliverables

The bootstrap phase shall produce the following deliverables.

### 10.1 Code Deliverables

* shared dataset loading / formatting code
* SFT training entrypoint or wrapper
* GRPO training entrypoint or wrapper
* Normalized Levenshtein Distance reward module
* W&B logging integration module
* W&B reference artifact registration module for checkpoints
* checkpoint save / resume helper code

### 10.2 Config Deliverables

* base runtime config
* base data config
* base W&B config
* base checkpoint config
* SFT config
* GRPO config
* environment override examples for multi-node execution

### 10.3 Documentation Deliverables

* launch instructions
* required environment variables
* checkpoint directory conventions
* W&B naming conventions
* resume procedures
* known limitations

---

## 11. Acceptance Tests

The following tests define the minimum acceptance bar.

### 11.1 SFT Acceptance Test

* Launch a multi-node SFT job from config only.
* Confirm training starts and reaches at least one checkpoint save.
* Confirm validation runs at least once.
* Confirm metrics appear in W&B.
* Interrupt and resume the run.
* Confirm resumed training continues from checkpoint rather than restarting.

### 11.2 GRPO Acceptance Test

* Launch a multi-node GRPO job from config only.
* Confirm rollouts occur.
* Confirm reward is computed for all tasks using the shared Normalized Levenshtein implementation.
* Confirm reward metrics and rollout diagnostics appear in W&B.
* Confirm at least one checkpoint is saved.
* Interrupt and resume the run.
* Confirm resumed training continues from checkpoint rather than restarting.

### 11.3 Artifact Acceptance Test

* Save a checkpoint locally.
* Register checkpoint metadata to W&B as a reference artifact.
* Confirm W&B artifact version exists.
* Confirm the artifact points to local / mounted storage and does not upload the checkpoint payload.

---

## 12. Risks and Constraints

### 12.1 Known Risks

* multi-node instability unrelated to trainer logic,
* resume failures caused by incomplete checkpoint state,
* inconsistent filesystem visibility across nodes,
* missing or unstable rollout diagnostics,
* schema drift between dataset views and reward interface,
* W&B artifact registration failures caused by path formatting or permissions.

### 12.2 Temporary Constraints

* reward is intentionally under-specified and low-fidelity,
* benchmark quality is not the goal of this phase,
* config structure may evolve after the first successful end-to-end runs,
* checkpoint artifact policy is metadata-only by design.

---

## 13. Post-Bootstrap Follow-Ups

The following work items are intentionally deferred to later phases:

1. task-specific reward design for tables, formulas, diagrams, seals, and layout
2. stronger validation and benchmark suites
3. training performance tuning
4. storage lifecycle and retention policy automation
5. dataset lineage tooling and artifactized dataset registry
6. backend-specific optimization for large-scale training

---

## 14. Recommended Initial Implementation Choices

To minimize integration risk, the first implementation should follow these choices:

* use one stable dataset view per run,
* use one shared reward implementation for all GRPO tasks,
* keep checkpoint files on a shared or mounted filesystem,
* register checkpoint metadata to W&B via filesystem reference artifacts,
* enable periodic validation and checkpointing from the beginning,
* save config snapshots with every run,
* make resume support mandatory before broader experimentation.

---

## 15. Open Decisions

The following decisions remain open and should be finalized during implementation:

1. exact parquet schema for SFT and GRPO views
2. exact checkpoint directory structure
3. whether SFT and GRPO use separate or shared W&B projects
4. artifact naming and alias conventions
5. how many rollout / validation samples to log per validation cycle
6. whether resume validation should run automatically after restore
7. whether checkpoint registration happens synchronously or asynchronously after save

---

## 16. Summary

This spec defines a **bootstrap-phase engineering baseline** for multi-node SFT + GRPO training. The emphasis is on **system readiness, reproducibility, and observability**, not final model quality.

The required output of this phase is a working system that:

* launches multi-node SFT and GRPO,
* logs runs and metrics to W&B,
* tracks checkpoint metadata as reference artifacts,
* keeps checkpoint payloads in local storage,
* resumes correctly after interruption,
* uses a single shared Normalized Levenshtein Distance reward for all tasks.

Once this baseline is proven stable, the project can move to richer rewards, deeper evaluation, and performance optimization.
