from __future__ import annotations

from typing import Callable

import torch
from tensordict import TensorDict


def install_qwen_vl_position_ids_chunk_patch() -> bool:
    """Patch VERL jagged TensorDict helpers to preserve Qwen-VL mRoPE position IDs."""
    from verl.utils import tensordict_utils as tu

    patched = False

    original_chunk = tu.chunk_tensordict
    if not getattr(original_chunk, "_ocr_qwen_vl_position_ids_patch", False):

        def patched_chunk_tensordict(td: TensorDict, chunks: int) -> list[TensorDict]:
            if _has_qwen_vl_position_ids(td):
                return _chunk_tensordict_with_qwen_vl_position_ids(td, chunks, original_chunk)
            return original_chunk(td, chunks)

        patched_chunk_tensordict._ocr_qwen_vl_position_ids_patch = True
        patched_chunk_tensordict._ocr_original_chunk_tensordict = original_chunk
        tu.chunk_tensordict = patched_chunk_tensordict
        patched = True

    original_index_select = tu.index_select_tensor_dict
    if not getattr(original_index_select, "_ocr_qwen_vl_position_ids_patch", False):

        def patched_index_select_tensor_dict(td: TensorDict, indices: torch.Tensor | list[int]) -> TensorDict:
            if _has_qwen_vl_position_ids(td):
                return _index_select_tensordict_with_qwen_vl_position_ids(td, indices, original_index_select)
            return original_index_select(td, indices)

        patched_index_select_tensor_dict._ocr_qwen_vl_position_ids_patch = True
        patched_index_select_tensor_dict._ocr_original_index_select_tensor_dict = original_index_select
        tu.index_select_tensor_dict = patched_index_select_tensor_dict
        patched = True

    return patched


def _has_qwen_vl_position_ids(td: TensorDict) -> bool:
    if "position_ids" not in td.keys() or "input_ids" not in td.keys():
        return False

    position_ids = td["position_ids"]
    input_ids = td["input_ids"]
    if not (
        isinstance(position_ids, torch.Tensor)
        and position_ids.is_nested
        and position_ids.dim() == 3
        and isinstance(input_ids, torch.Tensor)
        and input_ids.is_nested
    ):
        return False

    values = position_ids.values()
    return values.dim() == 2 and values.shape[0] == 4


def _chunk_tensordict_with_qwen_vl_position_ids(
    td: TensorDict,
    chunks: int,
    original_chunk_tensordict: Callable[[TensorDict, int], list[TensorDict]],
) -> list[TensorDict]:
    assert isinstance(td, TensorDict) and len(td) % chunks == 0, (
        f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"
    )

    position_ids = td["position_ids"]
    sequence_lengths = _sequence_lengths(td)
    chunk_size = len(td) // chunks

    padded_position_ids = _to_padded_qwen_vl_position_ids(position_ids, sequence_lengths)
    td_without_position_ids = td.exclude("position_ids", inplace=False)
    chunked = original_chunk_tensordict(td_without_position_ids, chunks)

    for chunk_index, chunk_td in enumerate(chunked):
        start = chunk_index * chunk_size
        end = start + chunk_size
        chunk_td["position_ids"] = _from_padded_qwen_vl_position_ids(
            padded_position_ids,
            sequence_lengths,
            row_indices=range(start, end),
        )

    return chunked


def _index_select_tensordict_with_qwen_vl_position_ids(
    td: TensorDict,
    indices: torch.Tensor | list[int],
    original_index_select_tensor_dict: Callable[[TensorDict, torch.Tensor | list[int]], TensorDict],
) -> TensorDict:
    index_list = _indices_to_list(indices)
    position_ids = td["position_ids"]
    sequence_lengths = _sequence_lengths(td)
    padded_position_ids = _to_padded_qwen_vl_position_ids(position_ids, sequence_lengths)

    selected = original_index_select_tensor_dict(td.exclude("position_ids", inplace=False), indices)
    selected["position_ids"] = _from_padded_qwen_vl_position_ids(
        padded_position_ids,
        sequence_lengths,
        row_indices=index_list,
    )
    return selected


def _sequence_lengths(td: TensorDict) -> list[int]:
    return [int(length) for length in td["input_ids"].offsets().diff().tolist()]


def _indices_to_list(indices: torch.Tensor | list[int]) -> list[int]:
    if isinstance(indices, torch.Tensor):
        return [int(index) for index in indices.detach().cpu().tolist()]
    return [int(index) for index in indices]


def _to_padded_qwen_vl_position_ids(position_ids: torch.Tensor, sequence_lengths: list[int]) -> torch.Tensor:
    max_sequence_length = max(sequence_lengths) if sequence_lengths else 0
    if len(sequence_lengths) == 1 and tuple(position_ids.values().shape) == (4, sequence_lengths[0]):
        return position_ids.values().unsqueeze(0)

    return torch.nested.to_padded_tensor(
        position_ids,
        padding=0,
        output_size=(len(sequence_lengths), 4, max_sequence_length),
    )


def _from_padded_qwen_vl_position_ids(
    padded_position_ids: torch.Tensor,
    sequence_lengths: list[int],
    *,
    row_indices,
) -> torch.Tensor:
    selected_values = []
    selected_offsets = [0]
    total_length = 0

    for row_index in row_indices:
        sequence_length = sequence_lengths[row_index]
        selected_values.append(padded_position_ids[row_index, :, :sequence_length])
        total_length += sequence_length
        selected_offsets.append(total_length)

    if selected_values:
        values = torch.cat(selected_values, dim=1)
    else:
        values = padded_position_ids.new_empty((4, 0))

    offsets = torch.tensor(selected_offsets, device=values.device, dtype=torch.long)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets, jagged_dim=2)
