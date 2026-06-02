from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def align_response_mask_for_shifted_logits(loss_mask: torch.Tensor, *, target_seq_len: int) -> torch.Tensor:
    """Align VERL SFT loss_mask to next-token logits/hidden positions."""
    if loss_mask.ndim != 2:
        raise ValueError("loss_mask must have shape [batch, seq]")
    if target_seq_len <= 0:
        raise ValueError("target_seq_len must be positive")
    if loss_mask.shape[1] == target_seq_len + 1:
        return loss_mask[:, 1:].bool()
    if loss_mask.shape[1] == target_seq_len:
        return loss_mask.bool()
    raise ValueError(
        "loss_mask sequence length must equal target_seq_len or target_seq_len + 1 "
        f"(got {loss_mask.shape[1]} for target_seq_len={target_seq_len})"
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        mask = align_response_mask_for_shifted_logits(mask, target_seq_len=values.shape[1])
    if values.shape != mask.shape:
        raise ValueError(f"response_mask shape {tuple(mask.shape)} must match values shape {tuple(values.shape)}")
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
    teacher_log_probs = F.log_softmax(teacher_topk_log_probs.detach().float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return _masked_mean(per_token, response_mask)


def truncated_forward_kl(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    student_full_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    student_selected = student_full_log_probs.gather(
        dim=-1,
        index=teacher_topk_indices.to(student_logits.device),
    )
    teacher_log_probs = teacher_topk_log_probs.detach().float()
    teacher_probs = teacher_log_probs.exp()
    per_token = (teacher_probs * (teacher_log_probs - student_selected)).sum(dim=-1)
    return _masked_mean(per_token, response_mask)


def truncated_top_k_forward_kl(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    return truncated_forward_kl(
        student_logits,
        teacher_topk_log_probs,
        teacher_topk_indices,
        response_mask,
    )


def hidden_normalized_mse(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    student_norm = F.normalize(student_hidden.float(), dim=-1)
    teacher_norm = F.normalize(teacher_hidden.detach().float(), dim=-1)
    per_token = (student_norm - teacher_norm).pow(2).mean(dim=-1)
    return _masked_mean(per_token, response_mask)


def hidden_cosine_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    per_token = 1.0 - F.cosine_similarity(student_hidden.float(), teacher_hidden.detach().float(), dim=-1)
    return _masked_mean(per_token, response_mask)


def hidden_kd_loss(
    student_captures: Mapping[int, torch.Tensor],
    teacher_captures: Mapping[int, torch.Tensor],
    layer_map: Sequence[Mapping[str, Any] | Any],
    response_mask: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    losses = []
    for entry in layer_map:
        student_index = _entry_value(entry, "student_hidden_index")
        teacher_index = _entry_value(entry, "teacher_hidden_index")
        if loss_type == "normalized_mse":
            loss = hidden_normalized_mse(
                student_captures[student_index],
                teacher_captures[teacher_index],
                response_mask,
            )
        elif loss_type == "cosine":
            loss = hidden_cosine_loss(
                student_captures[student_index],
                teacher_captures[teacher_index],
                response_mask,
            )
        else:
            raise ValueError(f"Unsupported hidden KD loss type: {loss_type!r}")
        losses.append(loss)

    if not losses:
        raise ValueError("hidden KD layer_map must contain at least one mapping")
    return torch.stack(losses).mean()


def _entry_value(entry: Mapping[str, Any] | Any, key: str) -> int:
    if isinstance(entry, Mapping):
        return int(entry[key])
    return int(getattr(entry, key))
