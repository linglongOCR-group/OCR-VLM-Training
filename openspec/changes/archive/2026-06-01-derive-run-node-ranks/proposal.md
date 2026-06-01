## Why

Training-ops currently treats node rank as reusable inventory data, but launch rank is only valid within a specific selected run topology. This causes incorrect `NODE_RANK` values when a run selects a subset of a larger cluster, such as using nodes 2 and 3 as a two-node job that must rank them 0 and 1.

## What Changes

- Derive run-level node ranks from the ordered `run.nodes` selection instead of reading training rank from inventory.
- Treat the first selected run node as the default head node for Ray and torchrun coordination, with room for an explicit `run.head_node` override if needed.
- Preserve inventory as stable cluster topology: node name, host, host IP, train interface, SSH/container defaults, shared paths, and optional non-training ordering metadata.
- Render `NODE_RANK` only for torchrun-based SFT launches, using contiguous run ranks from `0..NNODES-1`.
- Do not render or require `NODE_RANK` for Ray-based GRPO launches; Ray uses cluster membership and head address instead.
- Record both stable node identity and derived run rank in command metadata so audit logs remain clear.
- **BREAKING**: inventory node `rank` must no longer be required as the source of launch-time `NODE_RANK`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `training-ops-deployment`: Change the configuration and launch contract so training ranks are derived from run node order, not stored in reusable cluster inventory.

## Scope

In scope: training-ops YAML semantics, config validation, selected-node ordering, head-node resolution, effective environment rendering, SFT/GRPO launch command builders, command metadata, example configs, and tests.

Out of scope: changing trainer internals, replacing Ray startup scripts, changing torchrun scripts, introducing scheduler placement policies, or implementing hard cluster locks.

## Impact

Affected areas include `tools/training_ops/`, `configs/ops/`, `openspec/specs/training-ops-deployment/spec.md`, and `tests/test_training_ops.py`. Existing run YAMLs that select all inventory nodes in rank order should keep equivalent launch behavior after migration. Inventory YAMLs that currently require `rank` will need migration to remove or rename that field.

## Risks

- Existing inventories may depend on `rank` for display order; migration should preserve deterministic ordering through inventory order or an explicitly named non-training field.
- Changing head-node selection semantics could affect Ray startup if existing configs relied on inventory rank sorting instead of `run.nodes` order.
- Audit records must stay interpretable after separating node identity from run rank.

## Open Questions

- Should `run.head_node` be implemented immediately, or should v1 rely only on the first selected node?
- If inventory ordering is still useful, should the field be named `order`, `cluster_index`, or left as YAML list order only?
