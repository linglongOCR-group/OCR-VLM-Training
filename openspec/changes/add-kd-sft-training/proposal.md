## Why

The current OCR/VLM training stack supports SFT and GRPO, but the pruned MinerU-derived student needs teacher-student recovery training rather than ordinary SFT alone. KD-SFT adds teacher logits and hidden-state supervision while preserving the existing VERL/Ascend training workflow, giving engineers a reproducible way to evaluate whether the pruned student can recover teacher behavior and task quality.

## Scope

In scope:

- Add KD-SFT as a new training capability for the existing OCR/VLM SFT data path.
- Use co-located FSDP teacher and FSDP student.
- Support response-only SFT, top-k logits KD, and selected-layer hidden KD in v1.
- Add global-step loss schedules, config precheck, Hydra config, and a multi-node launch wrapper.
- Add unit and local/small smoke coverage for the KD-SFT math and trainer path.

Out of scope:

- RL/on-policy KD, teacher-generated pseudo labels, offline hidden caches, multi-teacher KD, tokenizer/vocab remapping, projection heads, and attention-map KD.
- Automated real MinerU/Ascend smoke as a required acceptance gate.

## What Changes

**KD-SFT Training Mode**
- From: SFT trains a single student model using masked next-token loss.
- To: KD-SFT trains a student with SFT, top-k logits KD, and hidden KD from a frozen teacher.
- Reason: Ordinary recovery fine-tuning has not restored the desired pruned-model quality.
- Impact: Adds a new training path without changing existing SFT behavior.

**Teacher Deployment**
- From: SFT has no teacher model.
- To: KD-SFT creates a forward-only FSDP teacher on every rank and consumes rank-local teacher outputs inside the same microbatch flow as the student.
- Reason: Teacher logits and hidden states are too large to move through driver or cross-rank payloads.
- Impact: New memory and distributed-order constraints for KD-SFT runs.

**Configuration And Launch**
- From: SFT uses Hydra configs plus `run_multinode_sft_new.sh`.
- To: KD-SFT uses `configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml` and `scripts/train/run_multinode_kd_sft.sh`, with Hydra overrides preserved.
- Reason: KD behavior should be config-first while retaining the existing Ascend launch ergonomics.
- Impact: New config/script surfaces for operators.

## Capabilities

### New Capabilities

- `kd-sft-training`: Defines KD-SFT training behavior, teacher/student compatibility, KD losses, schedules, hidden hooks, masking, metrics, and validation.

### Modified Capabilities

- `training-bootstrap`: Adds KD-SFT as a supported training mode and defines training metrics/configuration behavior.
- `training-ops-deployment`: Adds launch integration for the KD-SFT multi-node script.
- `checkpoint-tracking`: Records KD-SFT checkpoint metadata and resume behavior as a supported training mode.

## Impact

Affected code and config:

- `verl_plugins/trainers/` for the KD-SFT trainer, worker/engine integration, losses, schedules, hooks, and precheck.
- `configs/train/verl/sft/` for the KD-SFT Hydra config.
- `scripts/train/` for the multi-node KD-SFT launch wrapper.
- `tests/` for KD loss, schedule, hook, precheck, masking, and smoke coverage.
- Existing SFT and GRPO entrypoints should remain compatible.

## Risks

- FSDP teacher plus FSDP student can exceed HBM on long OCR batches.
- Hidden hooks can reduce gradient checkpointing memory savings.
- Incorrect response-mask shifting can silently train the wrong token positions.
- Teacher/student FSDP collectives can deadlock if ranks take divergent branches.

## Open Questions

- Whether full-vocabulary KL should be added as a debug-only mode in v1 or left for a later change.
- Which tiny model fixture is best for the local FSDP smoke test.
