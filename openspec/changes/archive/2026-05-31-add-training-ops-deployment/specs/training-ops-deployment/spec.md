## ADDED Requirements

### Requirement: YAML Training Ops Configuration
The system SHALL load training-ops settings from YAML files that separate reusable cluster inventory from one run profile per file.

#### Scenario: Load run config with referenced inventory
- **GIVEN** a run YAML file referencing an inventory YAML file
- **WHEN** the training-ops CLI loads the run config
- **THEN** the system SHALL resolve the inventory path, load selected nodes, and produce a validated effective configuration

#### Scenario: Reject missing required run fields
- **GIVEN** a run YAML file missing required fields for the selected training mode
- **WHEN** the training-ops CLI validates the run config
- **THEN** the system SHALL fail before remote execution and report the missing fields

#### Scenario: Snapshot effective configuration
- **GIVEN** a valid run YAML file and inventory YAML file
- **WHEN** a package, deploy, or launch command runs
- **THEN** the system SHALL write immutable snapshots of the run config, inventory config, and effective environment to the run ops state directory

### Requirement: Current Working Tree Packaging
The system SHALL package the current working tree into an artifact that includes uncommitted tracked and untracked local changes while excluding configured generated artifacts.

#### Scenario: Package dirty working tree
- **GIVEN** a working tree with uncommitted tracked changes and untracked source files
- **WHEN** the package command runs
- **THEN** the system SHALL include those files in the package unless they match an exclude rule

#### Scenario: Exclude large generated artifacts
- **GIVEN** generated directories such as checkpoints, datasets, caches, logs, W&B runs, outputs, and run state
- **WHEN** the package command runs
- **THEN** the system SHALL exclude those directories by default

#### Scenario: Write package manifest
- **GIVEN** a successful package command
- **WHEN** the package artifact is written
- **THEN** the system SHALL write a manifest containing branch, commit SHA, dirty status, release ID, timestamp, source path, artifact path, exclude rules, target nodes, and artifact SHA256 checksum

### Requirement: Shared Filesystem Deployment
The system SHALL support deployment to a shared directory visible from all selected hosts and containers at a unified absolute path.

#### Scenario: Create immutable shared release
- **GIVEN** a package artifact and a shared deployment config
- **WHEN** shared deployment runs
- **THEN** the system SHALL extract the artifact into a versioned release directory without modifying existing release contents

#### Scenario: Atomically update current symlink
- **GIVEN** a verified shared release directory
- **WHEN** shared deployment completes
- **THEN** the system SHALL atomically update the configured current symlink to point at the release directory

#### Scenario: Verify shared path on hosts and containers
- **GIVEN** selected inventory nodes and a shared release path
- **WHEN** shared deployment verifies the release
- **THEN** the system SHALL confirm the release path is visible on every host and inside every configured `verl-vlm-grpo` container

#### Scenario: Reuse matching existing shared release
- **GIVEN** a release directory that already exists with the same artifact checksum
- **WHEN** shared deployment runs for the same release
- **THEN** the system SHALL reuse the release if verification succeeds

#### Scenario: Reject conflicting shared release
- **GIVEN** a release directory that already exists with a different artifact checksum
- **WHEN** shared deployment runs for that release ID
- **THEN** the system SHALL fail without overwriting the existing release

### Requirement: Temporary Container Deployment
The system SHALL support deployment to `/tmp` on selected nodes or directly inside each selected container, with SSH tar streaming through host SSH and `docker exec` as the default transfer mechanism.

#### Scenario: Stream artifact into container
- **GIVEN** a package artifact and temporary deployment config targeting container storage
- **WHEN** temporary deployment runs
- **THEN** the system SHALL stream the artifact through SSH into `docker exec` and extract it inside the configured container path

#### Scenario: Verify container temporary deployment
- **GIVEN** a container temporary release path
- **WHEN** temporary deployment verification runs
- **THEN** the system SHALL verify available space, extraction success, checksum marker, and container-side path visibility on every selected node

#### Scenario: Support host temporary deployment when configured
- **GIVEN** a temporary deployment config targeting host `/tmp`
- **WHEN** temporary deployment runs
- **THEN** the system SHALL extract the release on each host and verify whether the host path is visible inside the configured container before launch

#### Scenario: Avoid docker cp by default
- **GIVEN** a temporary deployment config without explicit fallback settings
- **WHEN** temporary deployment transfers artifacts
- **THEN** the system SHALL NOT use `docker cp`

### Requirement: Remote Command Execution
The system SHALL execute remote host commands over SSH and container commands through host SSH plus `docker exec` into an existing container.

#### Scenario: Execute host command
- **GIVEN** a selected node
- **WHEN** a host-level training-ops command runs
- **THEN** the system SHALL execute the command over SSH to the node host

#### Scenario: Execute container command
- **GIVEN** a selected node with an existing `verl-vlm-grpo` container
- **WHEN** a container-level training-ops command runs
- **THEN** the system SHALL execute the command using host SSH and `docker exec` into that container

#### Scenario: Reject missing container
- **GIVEN** a selected node where the configured container is not running
- **WHEN** preflight or deployment verification runs
- **THEN** the system SHALL fail that node check and report the container error

#### Scenario: Record command metadata
- **GIVEN** any remote command
- **WHEN** the command finishes
- **THEN** the system SHALL record command label, node, host, rank, container, sanitized command, timestamps, exit code, and stdout/stderr log paths

### Requirement: Preflight Checks
The system SHALL provide read-only preflight checks before deployment or launch.

#### Scenario: Preflight verifies access and runtime
- **GIVEN** selected inventory nodes
- **WHEN** preflight runs
- **THEN** the system SHALL verify SSH access, container presence, NPU visibility, Python/runtime imports, configured model paths, configured data paths, checkpoint directory availability, and free space

#### Scenario: Preflight warns on active processes
- **GIVEN** selected nodes with existing Ray or training processes
- **WHEN** preflight runs
- **THEN** the system SHALL report warnings without stopping processes

### Requirement: Ray Operations
The system SHALL provide Ray start and status operations that run inside the selected containers and use the repository Ray helper scripts.

#### Scenario: Start Ray head
- **GIVEN** a selected head node and deployed project root
- **WHEN** the Ray head command runs
- **THEN** the system SHALL execute `scripts/cluster/start_ray_head.sh` inside the head container with explicit Ray environment variables

#### Scenario: Start Ray workers
- **GIVEN** selected worker nodes and a configured Ray head address
- **WHEN** the Ray worker command runs
- **THEN** the system SHALL execute `scripts/cluster/start_ray_worker.sh` inside each worker container with `RAY_HEAD_ADDRESS` set

#### Scenario: Check Ray status
- **GIVEN** selected nodes
- **WHEN** status runs
- **THEN** the system SHALL collect Ray status or process state from the configured containers and record the result in status metadata

### Requirement: Training Launch Integration
The system SHALL launch training by invoking the repository training scripts with explicit environment variables and recorded launch metadata.

#### Scenario: Launch GRPO
- **GIVEN** a GRPO run config, deployed project root, and Ray configuration
- **WHEN** `launch grpo` runs
- **THEN** the system SHALL execute `scripts/train/run_grpo_fsdp.sh` on the head node container with explicit `PROJECT_ROOT`, `PYTHONPATH`, `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE` or validation-disabling settings, `CKPTS_DIR`, `OCR_DATA_ROOT`, `NNODES`, `NPUS_PER_NODE`, and `RAY_ADDRESS`

#### Scenario: Launch SFT
- **GIVEN** an SFT run config and deployed project root
- **WHEN** `launch sft` runs
- **THEN** the system SHALL execute `scripts/train/run_multinode_sft_new.sh` on every selected node container with explicit `PROJECT_ROOT`, `PYTHONPATH`, `MODEL_PATH`, `TRAIN_FILES` or `TRAIN_FILE`, `VAL_FILES` or `VAL_FILE`, `CKPTS_DIR`, `OCR_DATA_ROOT`, `NNODES`, `NPUS_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `TRAIN_IFACE`

#### Scenario: Record launch metadata
- **GIVEN** a launch command
- **WHEN** the command is prepared
- **THEN** the system SHALL record the effective environment, extra script arguments, target nodes, release ID, deployed project root, and command log paths before execution

### Requirement: Ops State and Audit Metadata
The system SHALL store deployment and training operation metadata in a separate ops state directory organized by run ID.

#### Scenario: Create run ops state
- **GIVEN** a run ID
- **WHEN** any training-ops command runs
- **THEN** the system SHALL create or reuse `runs/training-ops/<run_id>/` or the configured ops state root for that run

#### Scenario: Write deployment manifest
- **GIVEN** a successful deployment
- **WHEN** deployment verification completes
- **THEN** the system SHALL write a deployment manifest containing release ID, timestamp, source path, target path, target nodes, artifact checksum, and per-node verification results

#### Scenario: Write status metadata
- **GIVEN** a status command
- **WHEN** status collection completes
- **THEN** the system SHALL write a status JSON document with latest known release, Ray, process, log, and warning information

### Requirement: Logs, Stop, and Cleanup
The system SHALL provide logs, stop, and cleanup operations that are explicit and safe by default.

#### Scenario: Collect logs
- **GIVEN** a run ops state directory with command records
- **WHEN** logs are requested
- **THEN** the system SHALL show or collect logs for the selected run, node, and command labels without requiring manual host inspection

#### Scenario: Stop requires explicit target
- **GIVEN** a stop command without an explicit target
- **WHEN** the command is parsed
- **THEN** the system SHALL fail and require a target such as Ray, GRPO, SFT, or run ID

#### Scenario: Cleanup defaults to dry run
- **GIVEN** a cleanup command without confirmation
- **WHEN** cleanup evaluates old packages, releases, logs, or temporary deployments
- **THEN** the system SHALL report planned deletions without deleting files

#### Scenario: Cleanup preserves active release by default
- **GIVEN** a shared deployment with a current symlink
- **WHEN** cleanup runs
- **THEN** the system SHALL NOT delete the release targeted by the current symlink unless explicitly forced

### Requirement: Training Ops Agent Skill
The system SHALL include a repo-local agent skill that guides agents through safe training-ops workflows.

#### Scenario: Skill covers operational sequence
- **GIVEN** an agent is asked to deploy or launch OCR-VLM training on Atlas nodes
- **WHEN** the agent uses the training-ops skill
- **THEN** the skill SHALL guide the agent through inventory review, preflight, package, deploy, verify, Ray startup, SFT or GRPO launch, logs, status, stop, and cleanup

#### Scenario: Skill requires evidence before launch
- **GIVEN** an agent is preparing to launch training
- **WHEN** deployment or preflight checks have not passed
- **THEN** the skill SHALL instruct the agent to run or inspect those checks before launch
