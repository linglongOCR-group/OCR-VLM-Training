# KD-SFT Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add KD-SFT training for the OCR/VLM SFT stack with FSDP teacher, FSDP student, top-k logits KD, selected-layer hidden KD hooks, global-step schedules, config precheck, Hydra config, and multi-node launch support.

**Architecture:** Add repo-local KD-SFT modules under `verl_plugins/trainers/` and keep existing SFT/GRPO paths unchanged. Reuse the OCR SFT dataset/collator/trainer loop style, but introduce a KD-capable worker/engine path because raw logits and selected hidden activations are needed before VERL reduces logits to label log-probs. Keep teacher tensors rank-local and save only student training state.

**Tech Stack:** Python, PyTorch, torch.distributed FSDP, TensorDict, Hydra/OmegaConf, VERL trainer/worker APIs, pytest, Ascend torch_npu runtime through existing launch scripts.

---

## File Structure

- Create `verl_plugins/trainers/kd_schedules.py`: stateless global-step lambda schedules.
- Create `verl_plugins/trainers/kd_losses.py`: top-k logits KD, hidden KD, response-mask helpers.
- Create `verl_plugins/trainers/kd_hidden_hooks.py`: selected-layer hook registration and capture lifecycle.
- Create `verl_plugins/trainers/kd_precheck.py`: config/model compatibility checks and optional dry-run helper.
- Create `verl_plugins/trainers/kd_worker.py`: KD training worker owning student and teacher engines.
- Create `verl_plugins/trainers/kd_engine.py`: KD-capable FSDP LM-head engine wrapper/subclass exposing raw logits and hook captures.
- Create `verl_plugins/trainers/kd_sft_trainer.py`: Hydra entrypoint and trainer loop modeled on `verl_plugins/trainers/sft_trainer.py`.
- Create `configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml`: KD-SFT Hydra config.
- Create `scripts/train/run_multinode_kd_sft.sh`: Ascend multi-node launcher modeled on `scripts/train/run_multinode_sft_new.sh`.
- Modify `tools/training_ops/config.py`, `tools/training_ops/launch.py`, and related tests only if training-ops currently dispatches explicit SFT/GRPO modes.
- Modify `verl_plugins/callbacks/save_and_eval.py` only if checkpoint metadata registration needs explicit `kd_sft` fields.
- Add tests:
  - `tests/test_kd_schedules.py`
  - `tests/test_kd_losses.py`
  - `tests/test_kd_hidden_hooks.py`
  - `tests/test_kd_precheck.py`
  - `tests/test_kd_sft_masking.py`
  - `tests/test_kd_sft_trainer.py`

---

### Task 1: KD Utility Modules

**Files:**
- Create: `verl_plugins/trainers/kd_schedules.py`
- Create: `verl_plugins/trainers/kd_losses.py`
- Create: `verl_plugins/trainers/kd_hidden_hooks.py`
- Test: `tests/test_kd_schedules.py`
- Test: `tests/test_kd_losses.py`
- Test: `tests/test_kd_hidden_hooks.py`

- [ ] **Step 1: Add failing tests for constant and warmup/decay schedules**

Create `tests/test_kd_schedules.py` with these initial tests:

```python
import pytest

from verl_plugins.trainers.kd_schedules import evaluate_schedule


def test_constant_schedule_returns_value_for_any_global_step():
    cfg = {"type": "constant", "value": 1.0}

    assert evaluate_schedule(cfg, global_step=0) == pytest.approx(1.0)
    assert evaluate_schedule(cfg, global_step=25) == pytest.approx(1.0)


def test_linear_warmup_hold_decay_schedule_uses_global_step():
    cfg = {
        "type": "linear_warmup_hold_decay",
        "start": 0.0,
        "peak": 0.2,
        "warmup_steps": 10,
        "hold_steps": 20,
        "decay_steps": 10,
        "end": 0.0,
    }

    assert evaluate_schedule(cfg, global_step=0) == pytest.approx(0.0)
    assert evaluate_schedule(cfg, global_step=5) == pytest.approx(0.1)
    assert evaluate_schedule(cfg, global_step=10) == pytest.approx(0.2)
    assert evaluate_schedule(cfg, global_step=30) == pytest.approx(0.2)
    assert evaluate_schedule(cfg, global_step=35) == pytest.approx(0.1)
    assert evaluate_schedule(cfg, global_step=40) == pytest.approx(0.0)
```

- [ ] **Step 2: Run schedule tests and confirm import failure**

Run: `pytest tests/test_kd_schedules.py -q`

Expected: FAIL because `verl_plugins.trainers.kd_schedules` does not exist.

- [ ] **Step 3: Implement schedule evaluator**

Create `verl_plugins/trainers/kd_schedules.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evaluate_schedule(config: Mapping[str, Any], *, global_step: int) -> float:
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    schedule_type = str(config.get("type", "")).strip()
    if schedule_type == "constant":
        return float(config["value"])
    if schedule_type == "piecewise_linear":
        return _piecewise_linear(config, global_step=global_step)
    if schedule_type == "linear_warmup_constant":
        return _linear_warmup_constant(config, global_step=global_step)
    if schedule_type == "linear_warmup_hold_decay":
        return _linear_warmup_hold_decay(config, global_step=global_step)
    raise ValueError(f"Unsupported KD schedule type: {schedule_type!r}")


def validate_schedule(config: Mapping[str, Any]) -> None:
    evaluate_schedule(config, global_step=0)


def _linear_interpolate(start: float, end: float, *, index: int, steps: int) -> float:
    if steps <= 0:
        return end
    ratio = min(max(index / steps, 0.0), 1.0)
    return start + (end - start) * ratio


def _linear_warmup_constant(config: Mapping[str, Any], *, global_step: int) -> float:
    start = float(config.get("start", 0.0))
    peak = float(config["peak"])
    warmup_steps = int(config.get("warmup_steps", 0))
    if global_step < warmup_steps:
        return _linear_interpolate(start, peak, index=global_step, steps=warmup_steps)
    return peak


def _linear_warmup_hold_decay(config: Mapping[str, Any], *, global_step: int) -> float:
    start = float(config.get("start", 0.0))
    peak = float(config["peak"])
    end = float(config.get("end", 0.0))
    warmup_steps = int(config.get("warmup_steps", 0))
    hold_steps = int(config.get("hold_steps", 0))
    decay_steps = int(config.get("decay_steps", 0))
    if global_step < warmup_steps:
        return _linear_interpolate(start, peak, index=global_step, steps=warmup_steps)
    decay_start = warmup_steps + hold_steps
    if global_step < decay_start:
        return peak
    return _linear_interpolate(peak, end, index=global_step - decay_start, steps=decay_steps)


def _piecewise_linear(config: Mapping[str, Any], *, global_step: int) -> float:
    points = [(int(point["step"]), float(point["value"])) for point in config["points"]]
    if not points:
        raise ValueError("piecewise_linear schedule requires at least one point")
    points.sort()
    if global_step <= points[0][0]:
        return points[0][1]
    for (left_step, left_value), (right_step, right_value) in zip(points, points[1:], strict=False):
        if global_step <= right_step:
            return _linear_interpolate(left_value, right_value, index=global_step - left_step, steps=right_step - left_step)
    return points[-1][1]
```

- [ ] **Step 4: Run schedule tests**

Run: `pytest tests/test_kd_schedules.py -q`

Expected: PASS.

- [ ] **Step 5: Add failing logits KD tests**

Create `tests/test_kd_losses.py` with:

```python
import torch
import torch.nn.functional as F

from verl_plugins.trainers.kd_losses import (
    hidden_cosine_loss,
    hidden_normalized_mse,
    renormalized_top_k_forward_kl,
    truncated_top_k_forward_kl,
)


def test_renormalized_top_k_forward_kl_matches_manual_value():
    student_logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]])
    teacher_log_probs = torch.log(torch.tensor([[[0.75, 0.25]]]))
    teacher_indices = torch.tensor([[[0, 1]]])
    mask = torch.tensor([[True]])

    loss = renormalized_top_k_forward_kl(student_logits, teacher_log_probs, teacher_indices, mask)

    student_top = F.softmax(torch.tensor([2.0, 1.0]), dim=-1)
    teacher_top = torch.tensor([0.75, 0.25])
    expected = torch.sum(teacher_top * (torch.log(teacher_top) - torch.log(student_top)))
    assert loss == torch.approx(expected.item())


def test_truncated_top_k_forward_kl_uses_full_student_denominator():
    student_logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]])
    teacher_log_probs = torch.log(torch.tensor([[[0.75, 0.25]]]))
    teacher_indices = torch.tensor([[[0, 1]]])
    mask = torch.tensor([[True]])

    loss = truncated_top_k_forward_kl(student_logits, teacher_log_probs, teacher_indices, mask)

    student_full = F.log_softmax(student_logits[0, 0], dim=-1)
    expected = torch.sum(torch.tensor([0.75, 0.25]) * (teacher_log_probs[0, 0] - student_full[:2]))
    assert loss == torch.approx(expected.item())
```

- [ ] **Step 6: Implement logits and hidden losses**

In `verl_plugins/trainers/kd_losses.py`, implement:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _gather_student_top_k(student_logits: torch.Tensor, teacher_indices: torch.Tensor) -> torch.Tensor:
    return student_logits.gather(dim=-1, index=teacher_indices.to(student_logits.device))


def renormalized_top_k_forward_kl(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    student_top_logits = _gather_student_top_k(student_logits, teacher_topk_indices)
    student_log_probs = F.log_softmax(student_top_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_topk_log_probs.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return _masked_mean(per_token, response_mask)


def truncated_top_k_forward_kl(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    student_full_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    student_selected = student_full_log_probs.gather(dim=-1, index=teacher_topk_indices.to(student_logits.device))
    teacher_probs = teacher_topk_log_probs.float().exp()
    per_token = (teacher_probs * (teacher_topk_log_probs.float() - student_selected)).sum(dim=-1)
    return _masked_mean(per_token, response_mask)


def hidden_normalized_mse(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    student_norm = F.normalize(student_hidden.float(), dim=-1)
    teacher_norm = F.normalize(teacher_hidden.float(), dim=-1)
    per_token = (student_norm - teacher_norm).pow(2).mean(dim=-1)
    return _masked_mean(per_token, response_mask)


def hidden_cosine_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    per_token = 1.0 - F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1)
    return _masked_mean(per_token, response_mask)
```

- [ ] **Step 7: Add hidden-loss and hook tests**

Extend `tests/test_kd_losses.py` with hidden-loss tests. Create `tests/test_kd_hidden_hooks.py` with a small module containing `torch.nn.ModuleList` layers and verify mapped layers are captured, captures clear, and teacher captures detach.

- [ ] **Step 8: Implement hidden hook utility**

Create `verl_plugins/trainers/kd_hidden_hooks.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HiddenLayerMapEntry:
    student_hidden_index: int
    teacher_hidden_index: int


@dataclass
class HiddenCaptureStore:
    captures: dict[int, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.captures.clear()

    def record(self, hidden_index: int, value: Any, *, detach: bool) -> None:
        if detach and hasattr(value, "detach"):
            value = value.detach()
        self.captures[hidden_index] = value
```

Add hook registration functions that resolve decoder layer modules by hidden index and install `register_forward_hook` callbacks. Use HF semantics: hidden index `i + 1` maps to decoder layer index `i`.

- [ ] **Step 9: Run utility tests**

Run: `pytest tests/test_kd_schedules.py tests/test_kd_losses.py tests/test_kd_hidden_hooks.py -q`

Expected: PASS.

- [ ] **Step 10: Commit utility modules**

```bash
git add verl_plugins/trainers/kd_schedules.py verl_plugins/trainers/kd_losses.py verl_plugins/trainers/kd_hidden_hooks.py tests/test_kd_schedules.py tests/test_kd_losses.py tests/test_kd_hidden_hooks.py
git commit -m "feat(kd-sft): add kd schedule loss and hook utilities"
```

### Task 2: KD Configuration And Precheck

**Files:**
- Create: `verl_plugins/trainers/kd_precheck.py`
- Create: `configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml`
- Test: `tests/test_kd_precheck.py`

- [ ] **Step 1: Add failing precheck tests**

Create `tests/test_kd_precheck.py` with simple fake config/model objects and tests for invalid top-k, missing hidden layer map, vocab mismatch, and hidden-size mismatch.

- [ ] **Step 2: Implement config validation helpers**

Create `verl_plugins/trainers/kd_precheck.py` with functions:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verl_plugins.trainers.kd_schedules import validate_schedule


@dataclass(frozen=True)
class ModelContract:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    processor_type: str | None = None


def validate_top_k(*, top_k: int, vocab_size: int) -> None:
    if top_k <= 0:
        raise ValueError("kd.logits.top_k must be positive")
    if top_k > vocab_size:
        raise ValueError("kd.logits.top_k must not exceed vocab size")


def validate_layer_map(layer_map: list[Mapping[str, Any]], *, student_layers: int, teacher_layers: int) -> None:
    if not layer_map:
        raise ValueError("kd.hidden.layer_map is required when hidden KD is enabled")
    for entry in layer_map:
        s_idx = int(entry["student_hidden_index"])
        t_idx = int(entry["teacher_hidden_index"])
        if s_idx < 1 or s_idx > student_layers:
            raise ValueError(f"Invalid student hidden index: {s_idx}")
        if t_idx < 1 or t_idx > teacher_layers:
            raise ValueError(f"Invalid teacher hidden index: {t_idx}")


def validate_contract(student: ModelContract, teacher: ModelContract) -> None:
    if student.vocab_size != teacher.vocab_size:
        raise ValueError("Teacher and student vocab sizes must match")
    if student.hidden_size != teacher.hidden_size:
        raise ValueError("Teacher and student hidden sizes must match")
    if student.processor_type != teacher.processor_type:
        raise ValueError("Teacher and student processor types must match")


def validate_kd_config(config: Mapping[str, Any], *, student: ModelContract, teacher: ModelContract) -> None:
    validate_contract(student, teacher)
    kd = config["kd"]
    validate_top_k(top_k=int(kd["logits"]["top_k"]), vocab_size=student.vocab_size)
    if kd.get("hidden", {}).get("enabled", False):
        validate_layer_map(kd["hidden"].get("layer_map", []), student_layers=student.num_hidden_layers, teacher_layers=teacher.num_hidden_layers)
    for loss_cfg in kd["losses"].values():
        validate_schedule(loss_cfg["schedule"])
```

- [ ] **Step 3: Add KD-SFT Hydra config**

Create `configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml` by starting from `configs/train/verl/sft/qwen2_5_vl_fsdp.yaml` and adding a `kd:` block:

```yaml
entrypoint: verl_plugins.trainers.kd_sft_trainer
backend: fsdp
model:
  path: ${oc.env:MODEL_PATH}
  use_remove_padding: true
  freeze_vision_tower: ${oc.env:FREEZE_VISION_TOWER,False}
  trust_remote_code: true
  enable_gradient_checkpointing: true
kd:
  enabled: true
  teacher:
    path: ${oc.env:TEACHER_MODEL_PATH}
    strategy: fsdp2
  precheck:
    required: true
    mode: ${oc.env:KD_PRECHECK_MODE,config}
    allow_dry_run_batch: true
  logits:
    enabled: true
    mode: top_k
    top_k: ${oc.env:KD_TOP_K,64}
    temperature: ${oc.env:KD_TEMPERATURE,2.0}
    loss_type: ${oc.env:KD_LOGIT_LOSS_TYPE,renormalized_top_k_forward_kl}
  hidden:
    enabled: true
    loss_type: ${oc.env:KD_HIDDEN_LOSS_TYPE,normalized_mse}
    token_mask: response
    layer_map: ${oc.decode:${oc.env:KD_HIDDEN_LAYER_MAP,[]}}
  losses:
    sft:
      schedule:
        type: constant
        value: 1.0
    logits:
      schedule:
        type: linear_warmup_constant
        start: 0.0
        peak: ${oc.env:KD_LOGIT_WEIGHT,0.3}
        warmup_steps: ${oc.env:KD_LOGIT_WARMUP_STEPS,200}
    hidden:
      schedule:
        type: linear_warmup_hold_decay
        start: 0.0
        peak: ${oc.env:KD_HIDDEN_WEIGHT,0.1}
        warmup_steps: ${oc.env:KD_HIDDEN_WARMUP_STEPS,200}
        hold_steps: ${oc.env:KD_HIDDEN_HOLD_STEPS,2000}
        decay_steps: ${oc.env:KD_HIDDEN_DECAY_STEPS,1000}
        end: 0.0
```

- [ ] **Step 4: Run precheck tests**

Run: `pytest tests/test_kd_precheck.py -q`

Expected: PASS.

- [ ] **Step 5: Commit config and precheck**

```bash
git add verl_plugins/trainers/kd_precheck.py configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml tests/test_kd_precheck.py
git commit -m "feat(kd-sft): add kd config and precheck validation"
```

### Task 3: KD-SFT Trainer And Worker

**Files:**
- Create: `verl_plugins/trainers/kd_engine.py`
- Create: `verl_plugins/trainers/kd_worker.py`
- Create: `verl_plugins/trainers/kd_sft_trainer.py`
- Test: `tests/test_kd_sft_masking.py`
- Test: `tests/test_kd_sft_trainer.py`

- [ ] **Step 1: Add masking tests before trainer implementation**

Create `tests/test_kd_sft_masking.py` with a batch-shaped mask and assertions that shifted response masks exclude prompt and padding positions. Use tensors only; do not require FSDP in this test.

- [ ] **Step 2: Add trainer import and existing SFT preservation test**

Create `tests/test_kd_sft_trainer.py` with:

```python
import importlib


def test_kd_sft_trainer_module_imports():
    module = importlib.import_module("verl_plugins.trainers.kd_sft_trainer")
    assert hasattr(module, "run_kd_sft")
```

Run: `pytest tests/test_kd_sft_trainer.py -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Create KD trainer entrypoint skeleton**

Create `verl_plugins/trainers/kd_sft_trainer.py` modeled on `verl_plugins/trainers/sft_trainer.py`. Define `KDSFTTrainer`, `run_kd_sft(config)`, and Hydra `main(config)`. Reuse SFT dataset building and validation structure first, then wire KD worker in later steps.

- [ ] **Step 4: Create KD worker skeleton**

Create `verl_plugins/trainers/kd_worker.py` with `KDTrainingWorker` that accepts student and teacher worker configs. Provide methods `reset()`, `train_batch(data)`, `infer_batch(data)`, `save_checkpoint(...)`, and `load_checkpoint(...)`.

- [ ] **Step 5: Create KD engine wrapper skeleton**

Create `verl_plugins/trainers/kd_engine.py` with a wrapper/subclass around VERL's FSDP LM-head engine. Add an output structure containing:

```python
{
    "raw_logits": raw_output.logits,
    "log_probs": log_probs,
    "hidden_captures": capture_store.captures,
}
```

Keep this local to the KD worker path.

- [ ] **Step 6: Implement teacher/student microbatch order**

In `KDTrainingWorker.train_batch`, enforce:

1. Teacher engine eval forward with no grad.
2. Teacher top-k extraction on response positions.
3. Student train forward.
4. SFT loss, logits KD loss, hidden KD loss.
5. Student backward only.
6. Student optimizer and scheduler step.

- [ ] **Step 7: Implement metrics**

Return metrics keys:

```python
train/loss_sft_raw
train/loss_logit_raw
train/loss_hidden_raw
train/loss_sft_weighted
train/loss_logit_weighted
train/loss_hidden_weighted
train/lambda_sft
train/lambda_logit
train/lambda_hidden
train/kd_temperature
```

- [ ] **Step 8: Add local tiny smoke test**

Add a smoke test that uses tiny fake modules or a tiny HF-compatible model fixture to verify that KD worker setup can execute one loss-composition path. If FSDP cannot run in the local test environment, isolate the rank-local KD loss composition into a testable method and leave distributed FSDP smoke to the user's NPU run.

- [ ] **Step 9: Run trainer tests**

Run: `pytest tests/test_kd_sft_masking.py tests/test_kd_sft_trainer.py -q`

Expected: PASS.

- [ ] **Step 10: Commit trainer and worker**

```bash
git add verl_plugins/trainers/kd_engine.py verl_plugins/trainers/kd_worker.py verl_plugins/trainers/kd_sft_trainer.py tests/test_kd_sft_masking.py tests/test_kd_sft_trainer.py
git commit -m "feat(kd-sft): add kd trainer and worker path"
```

### Task 4: Launch, Ops, And Checkpoint Integration

**Files:**
- Create: `scripts/train/run_multinode_kd_sft.sh`
- Modify if needed: `tools/training_ops/config.py`
- Modify if needed: `tools/training_ops/launch.py`
- Modify if needed: `verl_plugins/callbacks/save_and_eval.py`
- Test: `tests/test_train_scripts.py`
- Test: `tests/test_training_ops.py`
- Test: `tests/test_checkpoint_artifacts.py`

- [ ] **Step 1: Add launcher script test**

Extend `tests/test_train_scripts.py` to assert `scripts/train/run_multinode_kd_sft.sh` exists and includes `-m verl_plugins.trainers.kd_sft_trainer`, `TEACHER_MODEL_PATH`, and `"$@"`.

- [ ] **Step 2: Add KD launch script**

Create `scripts/train/run_multinode_kd_sft.sh` from `scripts/train/run_multinode_sft_new.sh`. Change the module to `verl_plugins.trainers.kd_sft_trainer`, require `TEACHER_MODEL_PATH`, and pass Hydra overrides for the `kd.*` config block while preserving `"$@"`.

- [ ] **Step 3: Add training-ops mode only if mode dispatch is explicit**

Inspect `tools/training_ops/config.py` and `tools/training_ops/launch.py`. If modes are enumerated, add `kd-sft` or `kd_sft` consistently and render `scripts/train/run_multinode_kd_sft.sh` with student and teacher model paths. If launch mode is free-form script based, add no new dispatch code and document that no training-ops code change is needed.

- [ ] **Step 4: Extend checkpoint metadata**

If `verl_plugins/callbacks/save_and_eval.py` or related checkpoint artifact code validates training mode values, add `kd_sft`. Ensure metadata can include `student_model_id` and `teacher_model_id`.

- [ ] **Step 5: Run launch and checkpoint tests**

Run: `pytest tests/test_train_scripts.py tests/test_training_ops.py tests/test_checkpoint_artifacts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit launch and metadata integration**

```bash
git add scripts/train/run_multinode_kd_sft.sh tools/training_ops/config.py tools/training_ops/launch.py verl_plugins/callbacks/save_and_eval.py tests/test_train_scripts.py tests/test_training_ops.py tests/test_checkpoint_artifacts.py
git commit -m "feat(kd-sft): add launch and checkpoint integration"
```

If some listed files were not modified after inspection, omit them from `git add`.

### Task 5: Final Verification

**Files:**
- Modify: `openspec/changes/add-kd-sft-training/tasks.md`
- No production file creation expected in this task.

- [ ] **Step 1: Run focused KD-SFT tests**

Run:

```bash
pytest \
  tests/test_kd_schedules.py \
  tests/test_kd_losses.py \
  tests/test_kd_hidden_hooks.py \
  tests/test_kd_precheck.py \
  tests/test_kd_sft_masking.py \
  tests/test_kd_sft_trainer.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent regression tests**

Run:

```bash
pytest \
  tests/test_sft_freeze_vision.py \
  tests/test_sft_position_ids_patch.py \
  tests/test_sft_validation_flags.py \
  tests/test_train_scripts.py \
  tests/test_checkpoint_artifacts.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run OpenSpec status**

Run: `openspec status --change add-kd-sft-training`

Expected: OpenSpec reports required planning artifacts complete.

- [ ] **Step 4: Update task checklist**

Mark completed implementation tasks in `openspec/changes/add-kd-sft-training/tasks.md` as done only after corresponding verification passes.

- [ ] **Step 5: Commit final verification updates**

```bash
git add openspec/changes/add-kd-sft-training/tasks.md
git commit -m "chore(kd-sft): record implementation verification"
```
