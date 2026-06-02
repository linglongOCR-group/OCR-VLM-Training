from types import SimpleNamespace

import pytest
import torch

from verl_plugins.trainers.kd_worker import (
    KDStepConfig,
    compose_kd_sft_loss,
    extract_response_mask,
)


def _step_config(**overrides):
    values = {
        "global_step": 5,
        "top_k": 2,
        "temperature": 2.0,
        "logits_loss_type": "renormalized_top_k_forward_kl",
        "hidden_loss_type": "cosine",
        "layer_map": [{"student_hidden_index": 1, "teacher_hidden_index": 1}],
        "schedules": {
            "sft": {"type": "constant", "value": 1.0},
            "logits": {"type": "constant", "value": 0.5},
            "hidden": {"type": "constant", "value": 0.25},
        },
        "logits_enabled": True,
        "hidden_enabled": True,
    }
    values.update(overrides)
    return KDStepConfig(**values)


def test_extract_response_mask_preserves_shifted_loss_mask_semantics():
    data = {"loss_mask": torch.tensor([[False, True, True, False]])}

    mask = extract_response_mask(data, target_seq_len=3)

    assert torch.equal(mask, torch.tensor([[True, True, False]]))


def test_compose_kd_sft_loss_logs_raw_weighted_losses_lambdas_and_temperature():
    student_logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 4.0, 2.0]]], requires_grad=True)
    student_output = {
        "logits": student_logits,
        "log_probs": torch.tensor([[-0.5, -0.25]], requires_grad=True),
    }
    teacher_output = {"logits": teacher_logits}
    student_hidden = {1: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)}
    teacher_hidden = {1: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)}
    data = {"loss_mask": torch.tensor([[False, True, True]])}

    result = compose_kd_sft_loss(
        student_output=student_output,
        teacher_output=teacher_output,
        student_hidden_captures=student_hidden,
        teacher_hidden_captures=teacher_hidden,
        data=data,
        config=_step_config(),
    )
    result.loss.backward()

    assert result.loss.item() == pytest.approx(
        (
            result.metrics["train/loss_sft_weighted"]
            + result.metrics["train/loss_logit_weighted"]
            + result.metrics["train/loss_hidden_weighted"]
        )
    )
    assert result.metrics["train/lambda_sft"] == pytest.approx(1.0)
    assert result.metrics["train/lambda_logit"] == pytest.approx(0.5)
    assert result.metrics["train/lambda_hidden"] == pytest.approx(0.25)
    assert result.metrics["train/kd_temperature"] == pytest.approx(2.0)
    assert result.metrics["train/kd_logits_top_k"] == 2
    assert result.metrics["train/kd_hidden_layers"] == 1
    assert student_logits.grad is not None
    assert teacher_logits.grad is None
    assert teacher_hidden[1].grad is None


def test_compose_kd_sft_loss_reports_empty_response_tokens_with_context():
    data = {"loss_mask": torch.tensor([[False, False, False]])}
    output = {
        "logits": torch.zeros(1, 2, 4),
        "log_probs": torch.zeros(1, 2),
    }

    with pytest.raises(ValueError, match="rank=3.*global_step=9"):
        compose_kd_sft_loss(
            student_output=output,
            teacher_output={"logits": torch.zeros(1, 2, 4)},
            student_hidden_captures={},
            teacher_hidden_captures={},
            data=data,
            config=_step_config(global_step=9, hidden_enabled=False),
            context=SimpleNamespace(rank=3),
        )
