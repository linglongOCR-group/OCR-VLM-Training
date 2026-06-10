## Why

Formula RLVR currently uses string-distance reward, which poorly reflects visual equivalence and layout-sensitive LaTeX quality. To align Formula reinforcement learning with OmniDocBench evaluation, GRPO needs an opt-in online CDM reward that can score Formula rollouts during complete training runs.

## What Changes

- Add an opt-in mixed runtime reward profile for GRPO:
  - Formula tasks route to a CDM LaTeX-render reward.
  - Non-formula tasks keep the existing normalized Levenshtein reward.
- Add a runtime reward routing/profile contract that separates reward selection from `reward_version` provenance metadata.
- Add a Python CDM reward client used by VERL runtime reward code.
- Add a local Node.js LTS + Playwright/Chromium CDM service required for v1 online training.
- Add GRPO launch/preflight behavior that fails before training if CDM is configured but the local service is not healthy.
- Preserve backward compatibility for existing Levenshtein-only GRPO configs.
- Use OmniDocBench CDM only as an algorithm reference: token/bbox matching and F1 scoring are relevant; its module/process layout is not a design constraint.

## Scope

In scope for v1 is complete GRPO training with Formula CDM reward, including runtime dispatch, service health checks, scoring calls, failure diagnostics, and tests. Out of scope are Table TEDS implementation, rich OmniDocBench annotation payloads, official XeLaTeX parity evaluation, VERL batch reward-manager migration, Bun support, and automatic multi-node service orchestration.

## Capabilities

### New Capabilities
- `formula-cdm-runtime-reward`: Online Formula CDM reward support for GRPO, including runtime routing, CDM service contract, failure semantics, and launch readiness checks.

### Modified Capabilities
- `reward-system`: Add runtime reward routing/profile requirements and clarify that `reward_version` is provenance metadata, not the selector.
- `training-ops-deployment`: Add CDM service preflight expectations for CDM-enabled GRPO launches.

## Impact

- Affected Python runtime reward code: `verl_plugins/rewards/aggregate.py` and new runtime reward registry/client modules.
- Affected training configs/scripts: GRPO reward kwargs, CDM-enabled run config, and preflight checks.
- New service code: local Node.js LTS + Playwright/Chromium CDM service.
- New dependencies: Node.js package metadata and Playwright/Chromium runtime requirements for CDM-enabled runs.
- Tests: runtime reward routing, CDM client contract, preflight behavior, backward compatibility, and optional real-service integration tests.

## Risks

- CDM rendering latency may bottleneck GRPO if service caching/page reuse is insufficient.
- Renderer deployment differs across local, CI, and NPU training containers.
- Online KaTeX/Chromium CDM is an approximation of official OmniDocBench CDM; correlation should be measured separately.
- Per-sample invalid model LaTeX must not crash training, while infrastructure failures must still be caught by preflight.
- Existing offline reward adapters and online VERL reward code are separate stacks; runtime registry changes must avoid increasing divergence.

## Open Questions

- Which exact CDM service port and default timeout should be used in checked-in configs?
- Should v1 include `/score_batch` as a public endpoint immediately, or only batch-shaped internals plus `/score`?
- Where should Node service lifecycle documentation live: training script comments, service README, or training-ops docs?
