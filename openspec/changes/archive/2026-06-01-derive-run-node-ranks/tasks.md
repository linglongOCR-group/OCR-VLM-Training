## 1. Tests First

- [x] 1.1 Add config-loading tests for a four-node inventory where `run.nodes` selects the third and fourth nodes and derives run ranks `0` and `1`
- [x] 1.2 Add validation tests for duplicate selected nodes, unknown selected nodes, and `head_node` that is not part of the selected run nodes
- [x] 1.3 Add SFT launch tests proving selected subset commands render `NODE_RANK=0`, `NODE_RANK=1`, and `NNODES=2`
- [x] 1.4 Add GRPO and Ray command tests proving `NODE_RANK` is not rendered or required
- [x] 1.5 Add command metadata tests proving derived run rank is recorded with stable node identity

## 2. Configuration Model

- [x] 2.1 Stop requiring inventory node `rank` during inventory parsing
- [x] 2.2 Preserve deterministic inventory order from YAML list order or an explicitly named non-training ordering field
- [x] 2.3 Preserve ordered `run.nodes` selection instead of sorting selected nodes by inventory rank
- [x] 2.4 Introduce selected-node run-rank handling that pairs each selected inventory node with a contiguous derived run rank
- [x] 2.5 Implement default head-node resolution from the first selected run node
- [x] 2.6 Implement and validate optional `run.head_node` if included in the run config schema

## 3. Command Rendering

- [x] 3.1 Update effective environment rendering so SFT receives per-node `NODE_RANK=<derived run rank>`
- [x] 3.2 Update GRPO launch rendering so `NODE_RANK` is not included for Ray-based GRPO
- [x] 3.3 Update Ray head and worker command builders to use selected head-node identity and head address without node-rank assumptions
- [x] 3.4 Update SFT background log naming to use derived run rank or another collision-free selected-node identifier
- [x] 3.5 Update launch metadata to include derived run ranks for selected nodes

## 4. Audit State And CLI Output

- [x] 4.1 Update command records to store derived run rank with node name, host, and container
- [x] 4.2 Decide whether to keep a deprecated `rank` field as a compatibility alias or migrate fully to `run_rank`
- [x] 4.3 Update `inventory` CLI output to show stable inventory node data plus derived run rank for selected nodes
- [x] 4.4 Ensure snapshots preserve both original inventory YAML and effective selected-node topology

## 5. Examples And Guidance

- [x] 5.1 Update example inventory YAML files to remove launch-time `rank` requirements
- [x] 5.2 Update example run YAML files to rely on ordered `run.nodes` and optional `head_node` where useful
- [x] 5.3 Update `skills/training-ops/SKILL.md` if it refers to inventory rank or head/rank resolution

## 6. Verification

- [x] 6.1 Run focused training-ops tests
- [x] 6.2 Run OpenSpec validation for `derive-run-node-ranks`
- [x] 6.3 Inspect generated SFT and GRPO commands for representative full-cluster and subset run configs
