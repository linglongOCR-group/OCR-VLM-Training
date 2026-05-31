## MODIFIED Requirements

### Requirement: YAML Training Ops Configuration
The system SHALL load training-ops settings from YAML files that separate reusable cluster inventory from one run profile per file. Inventory nodes SHALL describe stable cluster resources and SHALL NOT be required to define launch-time training rank. Run node order SHALL define the active run topology.

#### Scenario: Load run config with referenced inventory
- **GIVEN** a run YAML file referencing an inventory YAML file
- **WHEN** the training-ops CLI loads the run config
- **THEN** the system SHALL resolve the inventory path, load selected nodes in run-specified order, derive contiguous run ranks for selected nodes, and produce a validated effective configuration

#### Scenario: Derive run ranks for selected subset
- **GIVEN** an inventory with four nodes
- **AND** a run YAML file selecting only the third and fourth inventory nodes in that order
- **WHEN** the training-ops CLI loads the run config
- **THEN** the selected nodes SHALL have derived run ranks `0` and `1`
- **AND** the derived run ranks SHALL NOT depend on any inventory-level position or rank value

#### Scenario: Resolve default head from selected run order
- **GIVEN** a run YAML file with multiple selected nodes
- **WHEN** the training-ops CLI loads the run config
- **THEN** the first selected node SHALL be the default head node

#### Scenario: Resolve explicit head from selected nodes
- **GIVEN** a run YAML file with `head_node` set to a selected node name
- **WHEN** the training-ops CLI loads the run config
- **THEN** the configured node SHALL be used as the head node

#### Scenario: Reject invalid explicit head
- **GIVEN** a run YAML file with `head_node` set to a node that is not selected by the run
- **WHEN** the training-ops CLI validates the run config
- **THEN** the system SHALL fail before remote execution and report the invalid head node

#### Scenario: Reject missing required run fields
- **GIVEN** a run YAML file missing required fields for the selected training mode
- **WHEN** the training-ops CLI validates the run config
- **THEN** the system SHALL fail before remote execution and report the missing fields

#### Scenario: Snapshot effective configuration
- **GIVEN** a valid run YAML file and inventory YAML file
- **WHEN** a package, deploy, or launch command runs
- **THEN** the system SHALL write immutable snapshots of the run config, inventory config, and effective environment to the run ops state directory

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
- **GIVEN** any remote command for a selected node
- **WHEN** the command finishes
- **THEN** the system SHALL record command label, node, host, derived run rank, container, sanitized command, timestamps, exit code, and stdout/stderr log paths

### Requirement: Ray Operations
The system SHALL provide Ray start and status operations that run inside the selected containers and use the repository Ray helper scripts.

#### Scenario: Start Ray head
- **GIVEN** a selected head node and deployed project root
- **WHEN** the Ray head command runs
- **THEN** the system SHALL execute `scripts/cluster/start_ray_head.sh` inside the head container with explicit Ray environment variables
- **AND** the command SHALL NOT require `NODE_RANK`

#### Scenario: Start Ray workers
- **GIVEN** selected worker nodes and a configured Ray head address
- **WHEN** the Ray worker command runs
- **THEN** the system SHALL execute `scripts/cluster/start_ray_worker.sh` inside each worker container with `RAY_HEAD_ADDRESS` set
- **AND** the command SHALL NOT require `NODE_RANK`

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
- **AND** the launch environment SHALL NOT require or render `NODE_RANK`

#### Scenario: Launch SFT
- **GIVEN** an SFT run config and deployed project root
- **WHEN** `launch sft` runs
- **THEN** the system SHALL execute `scripts/train/run_multinode_sft_new.sh` on every selected node container with explicit `PROJECT_ROOT`, `PYTHONPATH`, `MODEL_PATH`, `TRAIN_FILES` or `TRAIN_FILE`, `VAL_FILES` or `VAL_FILE`, `CKPTS_DIR`, `OCR_DATA_ROOT`, `NNODES`, `NPUS_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `TRAIN_IFACE`
- **AND** each `NODE_RANK` value SHALL equal the selected node's derived run rank

#### Scenario: Launch SFT on a selected subset
- **GIVEN** an inventory with four nodes
- **AND** an SFT run selecting the third and fourth inventory nodes in that order
- **WHEN** `launch sft` prepares commands
- **THEN** the command for the third inventory node SHALL render `NODE_RANK=0`
- **AND** the command for the fourth inventory node SHALL render `NODE_RANK=1`
- **AND** both commands SHALL render `NNODES=2`

#### Scenario: Record launch metadata
- **GIVEN** a launch command
- **WHEN** the command is prepared
- **THEN** the system SHALL record the effective environment, extra script arguments, target nodes, release ID, deployed project root, derived run ranks, and command log paths before execution
