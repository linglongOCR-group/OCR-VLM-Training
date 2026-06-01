## Context

Training-ops separates reusable cluster inventory from per-run YAML profiles. The current implementation requires each inventory node to define `rank` and reuses that value for inventory sorting, selected node ordering, command metadata, SFT background log naming, and SFT `NODE_RANK`.

That conflates stable cluster topology with run topology. A physical cluster may contain four nodes, but a particular run may select only the latter two nodes. For that run, torchrun requires those two selected nodes to be ranked `0` and `1`, not their position in the larger inventory.

GRPO and SFT also differ operationally in this repository. GRPO is launched once on the Ray head after Ray head/workers are started. SFT is launched on every selected node through `torchrun`, which requires a contiguous per-run `NODE_RANK`.

## Goals / Non-Goals

**Goals:**

- Make ordered `run.nodes` the source of run topology.
- Derive contiguous run ranks from selected node order.
- Use derived run rank for SFT `NODE_RANK`, SFT background logs, and command metadata.
- Avoid requiring or rendering `NODE_RANK` for Ray-based GRPO launches.
- Keep inventory focused on stable node identity and connection/runtime properties.
- Preserve deterministic head-node resolution, defaulting to the first selected node.

**Non-Goals:**

- Do not change VERL trainer internals or the shell training scripts.
- Do not introduce automatic scheduling or resource placement.
- Do not change how Ray itself assigns worker/task ranks internally.
- Do not add cluster locking or process orchestration beyond current training-ops scope.

## Decisions

### Decision: `run.nodes` order defines run topology

The run config's selected node list will preserve user-specified order. The first selected node is rank `0`, the second rank `1`, and so on.

Rationale:

- The run profile is where the operator declares which subset is active.
- YAML list order is simple, readable, and avoids manually maintained rank fields.
- The behavior matches torchrun's requirement for contiguous node ranks within the launched job.

Alternative considered: keep inventory `rank` and add per-run rank overrides. This was rejected because it preserves two sources of truth and makes common subset launches unnecessarily error-prone.

### Decision: Head node defaults to first selected node

The default head node for Ray and SFT master address resolution will be the first selected run node. An optional `run.head_node` may be added if explicit head selection is needed.

Rationale:

- Ordered selection already conveys operator intent.
- The same default works for Ray head startup and torchrun master address.
- An explicit override can support advanced cases without requiring rank fields.

Alternative considered: keep selecting the head by lowest inventory rank. This fails for subset runs where the intended head is determined by the run profile.

### Decision: Derive run-rank data without mutating inventory nodes

The implementation should introduce an effective selected-node representation or helper that pairs an inventory node with its derived `run_rank`.

Rationale:

- Inventory nodes remain reusable physical resources.
- Command builders can access both stable identity and run-specific rank.
- Metadata can record both `node` and `run_rank` without implying that rank is a physical property.

Alternative considered: overwrite `Node.rank` after selection. This hides the distinction and risks leaking run-specific state into inventory snapshots.

### Decision: SFT gets `NODE_RANK`; GRPO does not

For SFT, training-ops will render `NODE_RANK=<run_rank>` for every selected node because the SFT script invokes `torchrun --node_rank`. For GRPO, training-ops will not require or render `NODE_RANK`; GRPO uses Ray cluster membership plus `RAY_ADDRESS`.

Rationale:

- This matches the repository's launch scripts.
- It avoids inventing a fake rank concept for Ray-based GRPO.
- It makes effective environments easier to inspect.

Alternative considered: render `NODE_RANK` for all modes for consistency. This was rejected because consistency here would imply a false contract for Ray.

## Risks / Trade-offs

- Existing inventories that depend on `rank` for ordering -> Preserve YAML list order or migrate to a clearly named non-training field such as `order` only if needed.
- Existing tests may assume inventory rank sorting -> Replace with tests for ordered run-node selection and derived run ranks.
- Audit consumers may read `rank` from command metadata -> Migrate metadata to `run_rank` while preserving node name and host; optionally keep `rank` as a compatibility alias only if necessary.
- `run.head_node` override can conflict with `run.nodes` -> Validate that any explicit head node is included in selected nodes.

## Migration Plan

1. Add failing tests for a four-node inventory where a run selects nodes 2 and 3 and expects SFT `NODE_RANK=0` and `NODE_RANK=1`.
2. Stop requiring `rank` in inventory parsing; preserve node order from YAML.
3. Add derived selected-node run rank handling.
4. Update head-node resolution to use first selected node or validated `run.head_node`.
5. Update SFT, Ray, status, logs, and command metadata paths to use derived run rank where applicable.
6. Update example inventories and run configs.
7. Update the repo-local training-ops skill if it describes inventory rank.

Rollback is straightforward because this is a config-model change in training-ops. Reverting the change restores inventory-rank behavior, but any migrated inventories without `rank` would need the old field restored.

## Open Questions

- Should command metadata keep a deprecated `rank` field alongside `run_rank` for compatibility, or switch fully to `run_rank`?
- Should inventory support an optional `order` field, or is YAML list order sufficient for all known workflows?
