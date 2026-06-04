# KD-SFT Brainstorm Capture

## Background

The project currently provides OCR/VLM SFT and GRPO training integrations on top of VERL for Ascend/NPU infrastructure. The immediate new requirement is teacher-student knowledge distillation for a layer-pruned MinerU-derived VLM student. The original model is the teacher; the student is a decoder-layer-pruned model intended to recover generation quality and efficiency through SFT plus distillation.

The referenced discussion log is `docs/Conversation-Teacher-Student-Knowledge-Distillation.md`. The repository context is the current `ocr-vlm-tuning` workspace, not an external repository. Relevant current surfaces include:

- `verl_plugins/trainers/sft_trainer.py`
- `scripts/train/run_multinode_sft_new.sh`
- `configs/train/verl/sft/qwen2_5_vl_fsdp.yaml`
- `tools/data_management/runtime/verl_multimodal_dataset.py`
- `/verl/verl/trainer/sft_trainer.py`
- `/verl/verl/workers/engine/fsdp/transformer_impl.py`
- `/verl/recipe/gkd/megatron/` as a reference for sparse top-k KD ideas, but not as a direct implementation target.

## Scope

Build KD-SFT v1 for the existing OCR/VLM SFT stack. The implementation must support FSDP teacher plus FSDP student, top-k logits distillation, selected-layer hidden-state distillation, response-only masking, global-step loss schedules, mandatory compatibility precheck, Hydra config under `configs/train/verl/sft/`, and a multi-node launch wrapper modeled after `run_multinode_sft_new.sh`.

Out of scope for v1:

- RL/on-policy KD.
- Teacher-generated pseudo-label generation.
- Offline hidden-state cache.
- Multi-teacher distillation.
- Tokenizer/vocab remapping.
- Projection heads for hidden-size mismatch.
- Attention-map distillation.
- Real MinerU/Ascend smoke as an automated acceptance gate. The user will run that after implementation.

## Decisions

### Teacher Deployment

The preferred v1 topology is co-located FSDP teacher plus FSDP student.

Every rank constructs both models. The teacher is frozen, eval-only, forward-only, and has no optimizer, scheduler, or checkpoint state. The student remains trainable and checkpointed through the existing SFT lifecycle. Teacher and student consume the same local microbatch from the existing distributed SFT dataloader. Teacher logits and hidden states stay rank-local; no teacher activations are gathered across ranks.

All ranks must execute the same distributed order:

1. Teacher forward.
2. Student forward.
3. Student backward.
4. Optimizer step.

### V1 Losses

V1 must include all three losses:

- SFT loss.
- Logits KD loss.
- Hidden-state KD loss.

The total loss is:

```text
loss_total =
  lambda_sft * loss_sft
  + lambda_logit * loss_logit
  + lambda_hidden * loss_hidden
```

Both logits KD and hidden KD are important for v1 and should not be deferred.

### Hidden-State Alignment

Hidden alignment must be explicit and config-driven. The implementation must not assume a fixed layer-pruning method, because future students may use different cutting strategies.

The layer map uses HuggingFace hidden-state semantics:

- `hidden_states[0]` is the embedding output.
- `hidden_states[i + 1]` is the output after decoder layer `i`.

When hidden KD is enabled, `kd.hidden.layer_map` is required. Each mapped layer contributes equally. No per-layer hidden weights are included in v1. Hidden loss type is configurable:

- `normalized_mse` as default.
- `cosine` as supported ablation.

Hidden KD applies only to response/output tokens.

### Hidden Capture

V1 should use selected-layer hooks, not `output_hidden_states=True`, as the default implementation path. Hooks should capture only layers referenced by `kd.hidden.layer_map`.

Teacher hook captures are detached immediately. Student hook captures retain gradients. Captures must be reset per microbatch and validated so stale activations cannot leak across steps.

Student gradient checkpointing remains allowed with hidden KD hooks enabled. Startup logging should warn that selected hidden-hook layers retain activations and reduce checkpointing's memory savings.

### Logits KD

Top-k teacher logits KD is a first-class v1 feature and should be the scalable default.

V1 supports both:

- `renormalized_top_k_forward_kl` as default.
- `truncated_forward_kl` as an ablation option.

Full-vocabulary KL may be retained as debug or future ablation if feasible, but v1 should not depend on full-vocab teacher targets as the scalable path.

Logits KD applies only to response/output tokens.

### Masking

All three losses use response-only supervision:

- SFT CE: response tokens only.
- Logits KD: response tokens only.
- Hidden KD: response tokens only.

Prompt tokens, image placeholder tokens, padding tokens, and system/template tokens are excluded. The implementation must respect VERL's shifted `loss_mask` convention.

### Loss Schedules

Loss-weight schedules are first-class v1 requirements. Schedule weights are computed from global optimizer step, not local dataloader step or microbatch index.

Constraints:

- Deterministic across ranks.
- Based on global optimizer step.
- Restored correctly after checkpoint resume by using the restored global step.
- Independent per loss term.
- Compatible with gradient accumulation.
- No extra distributed communication.

Required schedule types:

- `constant`
- `piecewise_linear`

Recommended schedule types:

- `linear_warmup_constant`
- `linear_warmup_hold_decay`

Default schedule intent:

- SFT remains constant throughout training.
- Logits KD warms up then remains constant.
- Hidden KD warms up, holds, then decays.

Metrics must log raw losses, weighted losses, lambdas, and temperature:

- `train/loss_sft_raw`
- `train/loss_logit_raw`
- `train/loss_hidden_raw`
- `train/loss_sft_weighted`
- `train/loss_logit_weighted`
- `train/loss_hidden_weighted`
- `train/lambda_sft`
- `train/lambda_logit`
- `train/lambda_hidden`
- `train/kd_temperature`

### Compatibility Contract

V1 hard-requires teacher and student to have the same:

- Tokenizer.
- Vocab size.
- Hidden size.
- VLM input contract.
- Output-token semantics.

No projection heads, vocab mapping, or cross-tokenizer KD are included in v1.

### Precheck

Mandatory startup precheck mode is `config`. Optional stronger mode is `dry_run_batch`.

The required config precheck validates:

- Teacher and student paths exist and load.
- Tokenizer vocab sizes match.
- Hidden sizes match.
- Processor/VLM contract is compatible.
- Top-k is valid and no larger than vocab size.
- Hidden layer map exists and is valid when hidden KD is enabled.
- Hidden hook targets can be resolved.
- Schedules are valid.
- Teacher is frozen and has no optimizer/scheduler/checkpoint participation.

`dry_run_batch` should be first-class and easy to enable, but not mandatory by default.

### Config And Script Surface

KD-SFT should use Hydra like VERL. The script is an operational wrapper, not the source of truth.

Accepted surfaces:

- Hydra config under `configs/train/verl/sft/`, specifically `configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml`.
- Entrypoint `verl_plugins.trainers.kd_sft_trainer`.
- Launch wrapper `scripts/train/run_multinode_kd_sft.sh`, modeled after `scripts/train/run_multinode_sft_new.sh`.

The script should set Ascend/runtime/env defaults and pass Hydra overrides while preserving `"$@"` for additional user overrides.

## Recommended Architecture

Add a repo-local KD-SFT trainer/worker path under `verl_plugins`, reusing the existing OCR SFT dataset, collator, checkpoint handler, logging pattern, resume behavior, and launch conventions.

Primary components:

1. `KDSFTTrainer`
   - Mirrors `OcrSFTTrainer`.
   - Adds KD config extraction and precheck.
   - Uses a KD-capable training worker.

2. `KDTrainingWorker`
   - Owns teacher and student model engines on each rank.
   - Runs both forwards and composes losses inside rank-local microbatch execution.

3. `KDFSDPEngine`
   - Extends or wraps VERL's FSDP LM-head path enough to expose raw logits and selected hidden hook captures.
   - Avoids modifying `/verl/` directly unless there is no viable plugin-local path.

4. `kd_losses`
   - Implements top-k logits KD and hidden KD as unit-testable pure functions.

5. `kd_schedules`
   - Implements stateless schedule evaluation by global optimizer step.

6. `hidden_hooks`
   - Registers, validates, captures, and clears selected-layer hooks.

7. `kd_precheck`
   - Validates compatibility and optional dry-run batch behavior.

## Data Flow

1. Dataloader and collator remain the same as OCR SFT.
2. Each rank receives one local TensorDict batch from the existing distributed sampler.
3. KD worker converts that batch into microbatches like VERL FSDP SFT.
4. Teacher FSDP engine runs first in eval/no-grad mode using the same model inputs.
5. Teacher logits are reduced immediately to response-token top-k indices/logprobs.
6. Teacher selected hidden activations are captured by configured hooks and detached.
7. Student FSDP engine runs on the same microbatch.
8. Student logits are used for SFT and top-k KD.
9. Student selected hidden activations are captured by hooks with gradients retained.
10. Response-only masks are applied to all three losses.
11. Scheduled lambdas are evaluated from global optimizer step and applied.
12. Backward runs only through the student.
13. Metrics include raw losses, weighted losses, lambdas, temperature, top-k mode, and hidden token/layer counts.

## Error Handling

The implementation should fail early and loudly on incompatible model/config conditions. Runtime guards should detect stale hidden captures, missing captures, zero response-token batches, invalid top-k tensors, NaN/Inf losses, and rank-divergent enabled/disabled branches.

## Acceptance Target

Implementation acceptance is unit and local/small smoke coverage:

- Config validation.
- KD schedule math.
- Top-k logits losses.
- Hidden losses.
- Hidden hooks.
- Response-only shifted-mask behavior.
- Tiny/mock trainer smoke for FSDP teacher plus FSDP student with top-k logits KD, hidden hooks, and gradient checkpointing allowed.

The user will run the real NPU/MinerU smoke after implementation.
