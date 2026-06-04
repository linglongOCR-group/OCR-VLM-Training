import pytest
import torch
import torch.nn.functional as F

from verl_plugins.trainers.kd_losses import (
    align_response_mask_for_shifted_logits,
    hidden_cosine_loss,
    hidden_kd_loss,
    hidden_normalized_mse,
    renormalized_top_k_forward_kl,
    truncated_forward_kl,
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
    assert loss.item() == pytest.approx(expected.item(), abs=1e-7)


def test_truncated_forward_kl_uses_full_student_denominator():
    student_logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]])
    teacher_log_probs = torch.log(torch.tensor([[[0.75, 0.25]]]))
    teacher_indices = torch.tensor([[[0, 1]]])
    mask = torch.tensor([[True]])

    loss = truncated_forward_kl(student_logits, teacher_log_probs, teacher_indices, mask)

    student_full = F.log_softmax(student_logits[0, 0], dim=-1)
    expected = torch.sum(torch.tensor([0.75, 0.25]) * (teacher_log_probs[0, 0] - student_full[:2]))
    assert loss.item() == pytest.approx(expected.item())


def test_top_k_losses_average_only_response_tokens():
    student_logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
            ]
        ]
    )
    teacher_log_probs = torch.log(torch.tensor([[[0.8, 0.2], [0.1, 0.9]]]))
    teacher_indices = torch.tensor([[[0, 1], [0, 1]]])
    mask = torch.tensor([[False, True]])

    loss = truncated_forward_kl(student_logits, teacher_log_probs, teacher_indices, mask)

    student_full = F.log_softmax(student_logits[0, 1], dim=-1)
    expected = torch.sum(torch.tensor([0.1, 0.9]) * (teacher_log_probs[0, 1] - student_full[:2]))
    assert loss.item() == pytest.approx(expected.item())


def test_top_k_losses_treat_teacher_logits_as_detached_targets():
    student_logits = torch.tensor([[[2.0, 1.0, 0.0]]], requires_grad=True)
    teacher_log_probs = torch.log(torch.tensor([[[0.75, 0.25]]])).requires_grad_()
    teacher_indices = torch.tensor([[[0, 1]]])
    mask = torch.tensor([[True]])

    loss = renormalized_top_k_forward_kl(student_logits, teacher_log_probs, teacher_indices, mask)
    loss.backward()

    assert student_logits.grad is not None
    assert teacher_log_probs.grad is None


def test_mask_alignment_rejects_batch_size_mismatch_after_shift():
    values = torch.ones(2, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="response_mask shape"):
        hidden_normalized_mse(values.unsqueeze(-1), values.unsqueeze(-1), mask)


def test_align_response_mask_preserves_verl_sft_shifted_loss_mask_semantics():
    loss_mask = torch.tensor([[False, True, True, False, True]])

    aligned = align_response_mask_for_shifted_logits(loss_mask, target_seq_len=4)

    assert torch.equal(aligned, torch.tensor([[True, True, False, True]]))


def test_align_response_mask_accepts_already_shifted_mask():
    loss_mask = torch.tensor([[True, False, True]])

    aligned = align_response_mask_for_shifted_logits(loss_mask, target_seq_len=3)

    assert torch.equal(aligned, loss_mask)


def test_hidden_normalized_mse_uses_response_mask():
    student = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    teacher = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[False, True]])

    loss = hidden_normalized_mse(student, teacher, mask)

    expected = torch.tensor(1.0)
    assert loss.item() == pytest.approx(expected.item())


def test_hidden_cosine_loss_uses_response_mask():
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    teacher = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[False, True]])

    loss = hidden_cosine_loss(student, teacher, mask)

    assert loss.item() == pytest.approx(1.0)


def test_hidden_kd_loss_weights_mapped_layers_equally():
    student_captures = {
        2: torch.tensor([[[1.0, 0.0]]]),
        4: torch.tensor([[[0.0, 1.0]]]),
    }
    teacher_captures = {
        3: torch.tensor([[[1.0, 0.0]]]),
        5: torch.tensor([[[1.0, 0.0]]]),
    }
    layer_map = [
        {"student_hidden_index": 2, "teacher_hidden_index": 3},
        {"student_hidden_index": 4, "teacher_hidden_index": 5},
    ]

    loss = hidden_kd_loss(
        student_captures,
        teacher_captures,
        layer_map,
        torch.tensor([[True]]),
        loss_type="cosine",
    )

    assert loss.item() == pytest.approx(0.5)


def test_hidden_losses_treat_teacher_hidden_as_detached_targets():
    student = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    mask = torch.tensor([[True]])

    loss = hidden_cosine_loss(student, teacher, mask)
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
