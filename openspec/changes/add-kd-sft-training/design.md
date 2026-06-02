## Context

The repository currently layers OCR/VLM training behavior over VERL through `verl_plugins`, train scripts, and Hydra configs. SFT uses `verl_plugins.trainers.sft_trainer`, the OCR multimodal dataset adapter, and VERL's FSDP trainer/engine. The immediate requirement is recovery distillation for a pruned MinerU-derived VLM student using the original model as teacher.

The key implementation constraint is that VERL's current SFT loss receives processed label `log_probs`, not raw logits or hidden states. Full KD-SFT therefore cannot be implemented by only replacing the SFT loss function. Teacher forward, student forward, top-k teacher target extraction, selected hidden-hook capture, and loss composition need to happen inside a rank-local training worker/engine flow.

## Goals / Non-Goals

**Goals:**

- Add a KD-SFT training mode for OCR/VLM SFT data.
- Run both teacher and student under FSDP on each rank.
- Compute response-only SFT, top-k logits KD, and selected-layer hidden KD in v1.
- Keep teacher outputs rank-local.
- Make KD behavior Hydra-configurable and launchable through a multi-node Ascend wrapper.
- Provide deterministic global-step loss schedules and mandatory compatibility precheck.
- Preserve existing SFT and GRPO behavior.

**Non-Goals:**

- RL/on-policy KD, OPD, or rollout-time distillation.
- Teacher-generated pseudo-label data.
- Offline hidden-state caching.
- Tokenizer/vocab remapping or hidden projection heads.
- Attention-map distillation.
- Mandatory full MinerU/Ascend smoke in automated tests.

## Decisions

### D1: Add A Repo-Local KD-SFT Trainer Path

- **Choice:** Add `verl_plugins.trainers.kd_sft_trainer` with a KD-capable worker/engine path.
- **Reason:** KD-SFT needs raw logits, selected hidden activations, teacher forward, and student backward in one rank-local microbatch flow.
- **Alternative considered:** Only replace VERL's SFT `loss_fn`. Rejected because current SFT loss receives log-probs after raw logits have been reduced.
- **Alternative considered:** Patch `/verl/` directly. Rejected unless unavoidable because this repo already isolates OCR-specific behavior in `verl_plugins`.

### D2: Use Co-Located FSDP Teacher And FSDP Student

- **Choice:** Every rank owns a frozen forward-only FSDP teacher and a trainable FSDP student.
- **Reason:** This reduces persistent teacher memory compared with full replication and avoids moving teacher tensors through driver-side or cross-rank payloads.
- **Alternative considered:** Replicated teacher per rank. Rejected as the preferred path because HBM pressure is a primary concern.
- **Alternative considered:** Separate teacher service. Rejected for v1 because hidden states would be expensive to transmit and the engineering scope is larger.

### D3: Make Top-K Logits KD First-Class

- **Choice:** Implement top-k logits KD as the scalable default with `renormalized_top_k_forward_kl` and `truncated_forward_kl`.
- **Reason:** Full-vocabulary teacher targets are expensive at OCR sequence lengths, while top-k keeps teacher supervision compact and ablation-friendly.
- **Alternative considered:** Full-vocabulary KL only. Rejected because it is likely too memory-heavy for the v1 operating point.

### D4: Use Explicit Hidden Layer Maps And Selected-Layer Hooks

- **Choice:** Require `kd.hidden.layer_map` and register hooks only for mapped teacher/student decoder layers.
- **Reason:** Future pruning strategies may not be alternating layer drops, and `output_hidden_states=True` can retain too many activations.
- **Alternative considered:** Hardcode alternating layer-pair mapping. Rejected because it would not support other cutting methods.
- **Alternative considered:** Use `output_hidden_states=True`. Rejected as the default because it creates unnecessary HBM pressure.

### D5: Apply KD Only To Response Tokens

- **Choice:** SFT, logits KD, and hidden KD all use response-only supervision.
- **Reason:** This matches the current SFT training contract and avoids optimizing prompt, image placeholder, system/template, or padding positions.
- **Alternative considered:** Hidden KD over prompt plus response. Rejected for v1 because it adds memory pressure and broader semantics.

### D6: Use Global Optimizer Step Loss Schedules

- **Choice:** Compute independent loss weights from global optimizer step.
- **Reason:** This is deterministic across ranks, compatible with gradient accumulation, and resumes correctly when global step is restored.
- **Alternative considered:** Per-microbatch or dataloader-step schedules. Rejected because they can drift under accumulation and resume.

### D7: Keep KD-SFT Hydra-First

- **Choice:** Add `configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml` and a wrapper `scripts/train/run_multinode_kd_sft.sh`.
- **Reason:** VERL and this repo already use Hydra overrides with scripts for operational defaults. KD behavior should remain expressible in config files and CLI overrides.
- **Alternative considered:** Script-only env configuration. Rejected because it would make KD experiments harder to reproduce.

## Risks / Trade-offs

- [Risk] FSDP teacher and student may still exceed HBM with long VLM sequences. Mitigation: top-k logits KD, selected-layer hidden hooks, response-only masking, small microbatch defaults, and optional dry-run batch.
- [Risk] Hidden hooks can partially defeat gradient checkpointing. Mitigation: allow checkpointing but log a startup warning and retain only mapped layer activations.
- [Risk] Mask shifting can be wrong for nested/no-padding tensors. Mitigation: add focused masking tests against VERL SFT shifted `loss_mask` semantics.
- [Risk] Rank-divergent branches can deadlock FSDP collectives. Mitigation: validate enabled KD branches at startup and require all ranks to run the same teacher/student forward order.
- [Trade-off] A KD-specific worker/engine is more code than a loss-only patch. Accepted because it keeps raw tensors rank-local and makes the real KD objective implementable.
- [Trade-off] Config precheck is mandatory but dry-run batch is optional. Accepted because config precheck is cheap enough for every run, while dry-run batch requires full distributed initialization.

## Migration Plan

1. Add KD-SFT modules under `verl_plugins/trainers/` without modifying existing SFT entrypoints.
2. Add KD-SFT Hydra config under `configs/train/verl/sft/`.
3. Add `scripts/train/run_multinode_kd_sft.sh` modeled after `run_multinode_sft_new.sh`.
4. Add tests for schedules, losses, hooks, precheck, masking, and a tiny/local smoke path.
5. Keep existing `run_multinode_sft_new.sh`, SFT config, and GRPO paths unchanged.
6. After local verification, the user runs the real NPU/MinerU smoke manually.

Rollback strategy:

- Existing SFT/GRPO paths are unaffected and can be used if KD-SFT fails.
- KD-SFT can be disabled by not invoking the new entrypoint/script/config.
- Because teacher checkpoints are not saved, rollback does not require deleting teacher state from saved student checkpoints.

## Open Questions

- Whether to include full-vocabulary KL as a debug-only option in the initial implementation.
- Which tiny model fixture should be used for local FSDP smoke tests.
