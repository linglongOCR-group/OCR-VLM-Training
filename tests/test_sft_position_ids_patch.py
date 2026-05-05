import torch

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import SFTTensorCollator

from verl_plugins.trainers.sft_position_ids import install_qwen_vl_position_ids_chunk_patch


def test_chunk_tensordict_preserves_qwen_vl_position_id_sequence_axis():
    install_qwen_vl_position_ids_chunk_patch()
    sample = {
        "input_ids": torch.arange(11),
        "loss_mask": torch.ones(11),
        "position_ids": torch.arange(44).view(4, 11),
    }
    collated = SFTTensorCollator("no_padding")([sample])
    data = tu.get_tensordict(
        tensor_dict=collated,
        non_tensor_dict={
            "micro_batch_size_per_gpu": 1,
        },
    )

    chunks = tu.chunk_tensordict(data, chunks=1)

    assert len(chunks) == 1
    position_ids = chunks[0]["position_ids"]
    assert tuple(position_ids.values().shape) == (4, 11)
    padded = torch.nested.to_padded_tensor(position_ids, 0, output_size=(1, 4, 11))
    assert torch.equal(padded[0], sample["position_ids"])


def test_chunk_tensordict_preserves_multiple_qwen_vl_position_id_lengths():
    install_qwen_vl_position_ids_chunk_patch()
    samples = [
        {
            "input_ids": torch.arange(5),
            "loss_mask": torch.ones(5),
            "position_ids": torch.arange(20).view(4, 5),
        },
        {
            "input_ids": torch.arange(7),
            "loss_mask": torch.ones(7),
            "position_ids": torch.arange(28).view(4, 7),
        },
    ]
    collated = SFTTensorCollator("no_padding")(samples)
    data = tu.get_tensordict(
        tensor_dict=collated,
        non_tensor_dict={
            "micro_batch_size_per_gpu": 1,
        },
    )

    chunks = tu.chunk_tensordict(data, chunks=2)

    assert [tuple(chunk["position_ids"].values().shape) for chunk in chunks] == [(4, 5), (4, 7)]
    for chunk, sample in zip(chunks, samples, strict=True):
        seq_len = sample["input_ids"].numel()
        padded = torch.nested.to_padded_tensor(chunk["position_ids"], 0, output_size=(1, 4, seq_len))
        assert torch.equal(padded[0], sample["position_ids"])


def test_index_select_tensor_dict_preserves_qwen_vl_position_ids():
    install_qwen_vl_position_ids_chunk_patch()
    samples = [
        {
            "input_ids": torch.arange(5),
            "loss_mask": torch.ones(5),
            "position_ids": torch.arange(20).view(4, 5),
        },
        {
            "input_ids": torch.arange(7),
            "loss_mask": torch.ones(7),
            "position_ids": torch.arange(28).view(4, 7),
        },
        {
            "input_ids": torch.arange(3),
            "loss_mask": torch.ones(3),
            "position_ids": torch.arange(12).view(4, 3),
        },
    ]
    collated = SFTTensorCollator("no_padding")(samples)
    data = tu.get_tensordict(
        tensor_dict=collated,
        non_tensor_dict={
            "micro_batch_size_per_gpu": 1,
        },
    )

    selected = tu.index_select_tensor_dict(data, [1, 0])

    assert tuple(selected["position_ids"].values().shape) == (4, 12)
    padded = torch.nested.to_padded_tensor(selected["position_ids"], 0, output_size=(2, 4, 7))
    assert torch.equal(padded[0, :, :7], samples[1]["position_ids"])
    assert torch.equal(padded[1, :, :5], samples[0]["position_ids"])


def test_index_select_tensor_dict_handles_singleton_qwen_vl_position_ids_after_verl_fixup():
    install_qwen_vl_position_ids_chunk_patch()
    sample = {
        "input_ids": torch.arange(11),
        "loss_mask": torch.ones(11),
        "position_ids": torch.arange(44).view(4, 11),
    }
    collated = SFTTensorCollator("no_padding")([sample])
    data = tu.get_tensordict(
        tensor_dict=collated,
        non_tensor_dict={
            "micro_batch_size_per_gpu": 1,
        },
    )
    tu.maybe_fix_3d_position_ids(data)

    selected = tu.index_select_tensor_dict(data, [0])

    assert tuple(selected["position_ids"].values().shape) == (4, 11)
    padded = torch.nested.to_padded_tensor(selected["position_ids"], 0, output_size=(1, 4, 11))
    assert torch.equal(padded[0], sample["position_ids"])
