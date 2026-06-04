## ADDED Requirements

### Requirement: KD-SFT Training Launch Integration

The training-ops system SHALL support launching KD-SFT by invoking the repository KD-SFT multi-node script with explicit environment variables and recorded launch metadata.

#### Scenario: Launch KD-SFT

- **GIVEN** a KD-SFT run config and deployed project root
- **WHEN** `launch kd-sft` runs
- **THEN** the system SHALL execute `scripts/train/run_multinode_kd_sft.sh` on every selected node container with explicit `PROJECT_ROOT`, `PYTHONPATH`, student model path, teacher model path, train files, validation files, checkpoint directory, OCR data root, node count, devices per node, node rank, master address, master port, and train interface
- **AND** each `NODE_RANK` value SHALL equal the selected node's derived run rank

#### Scenario: Launch KD-SFT on a selected subset

- **GIVEN** an inventory with four nodes
- **AND** a KD-SFT run selecting the third and fourth inventory nodes in that order
- **WHEN** `launch kd-sft` prepares commands
- **THEN** the command for the third selected node SHALL render `NODE_RANK=0`
- **AND** the command for the fourth selected node SHALL render `NODE_RANK=1`
- **AND** both commands SHALL render `NNODES=2`

#### Scenario: Record KD-SFT launch metadata

- **GIVEN** a KD-SFT launch command
- **WHEN** the command is prepared
- **THEN** the system SHALL record the effective environment, extra Hydra arguments, target nodes, release ID, deployed project root, derived run ranks, student model path, teacher model path, and command log paths before execution
