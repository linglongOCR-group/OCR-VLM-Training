## 1. KD Utility Modules

- [x] 1.1 Add KD schedule evaluator with `constant`, `piecewise_linear`, `linear_warmup_constant`, and `linear_warmup_hold_decay` support.
- [x] 1.2 Add top-k logits KD loss functions for `renormalized_top_k_forward_kl` and `truncated_forward_kl`.
- [x] 1.3 Add hidden KD loss functions for `normalized_mse` and `cosine`.
- [x] 1.4 Add response-mask alignment helpers that preserve VERL SFT shifted `loss_mask` semantics.
- [x] 1.5 Add selected-layer hidden hook utilities that register mapped layers, capture activations, detach teacher captures, and clear state per microbatch.

## 2. KD Configuration And Precheck

- [x] 2.1 Add KD config parsing and validation helpers for teacher path, logits KD, hidden KD, schedules, and precheck settings.
- [x] 2.2 Add teacher/student compatibility precheck for tokenizer, vocab size, hidden size, VLM processor contract, layer-map bounds, top-k bounds, and frozen teacher state.
- [x] 2.3 Add optional dry-run batch precheck entrypoint that can exercise teacher forward, student forward, top-k target extraction, hidden hooks, and total loss composition.
- [x] 2.4 Add KD-SFT Hydra config under `configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml`.

## 3. KD-SFT Trainer And Worker

- [x] 3.1 Add `verl_plugins.trainers.kd_sft_trainer` entrypoint that mirrors current OCR SFT dataset, dataloader, validation, logging, resume, and checkpoint flow.
- [x] 3.2 Add KD-capable training worker that owns trainable student and forward-only teacher engines on each rank.
- [x] 3.3 Add or wrap an FSDP LM-head engine path that exposes raw logits and selected hidden-hook captures before log-prob reduction.
- [x] 3.4 Implement rank-local KD microbatch execution order: teacher forward, top-k target extraction, student forward, loss composition, student backward.
- [x] 3.5 Ensure teacher has no optimizer, scheduler, checkpoint payload, or gradient participation.
- [x] 3.6 Log raw KD losses, weighted KD losses, active lambdas, KD temperature, top-k mode, and hidden capture diagnostics.
- [x] 3.7 Preserve existing SFT trainer and script behavior when KD-SFT entrypoint is not used.

## 4. Launch, Ops, And Checkpoint Integration

- [x] 4.1 Add `scripts/train/run_multinode_kd_sft.sh` modeled after `run_multinode_sft_new.sh` with Hydra override tail support.
- [x] 4.2 Add KD-SFT launch support to training-ops configuration and command rendering if training-ops mode dispatch is implemented for SFT/GRPO.
- [x] 4.3 Extend checkpoint metadata registration or callback paths to record `training_mode: kd_sft`, student source, teacher source, config hash, git commit, and local checkpoint URI.
- [x] 4.4 Ensure KD loss schedules use restored global optimizer step after checkpoint resume.

## 5. Tests And Verification

- [x] 5.1 Add unit tests for KD schedules including resume/global-step behavior.
- [x] 5.2 Add unit tests for top-k logits KD losses, temperature handling, and response masks.
- [x] 5.3 Add unit tests for hidden KD losses and equal mapped-layer weighting.
- [x] 5.4 Add unit tests for selected-layer hooks, capture reset, mapped-only capture, and teacher detach behavior.
- [x] 5.5 Add unit tests for KD config precheck failures and valid config success.
- [x] 5.6 Add masking tests proving KD losses use response-only shifted `loss_mask` semantics.
- [x] 5.7 Add a one-batch KD-SFT dry-run/local smoke test covering teacher forward, student forward, top-k KD, hidden hooks, and gradient checkpointing-allowed configuration. Real NPU FSDP smoke remains manual per v1 acceptance.
- [x] 5.8 Run the focused KD-SFT test suite and record the verification command/output.
