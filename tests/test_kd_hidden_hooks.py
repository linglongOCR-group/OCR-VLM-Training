import pytest
import torch

from verl_plugins.trainers.kd_hidden_hooks import (
    HiddenCaptureStore,
    HiddenLayerMapEntry,
    register_mapped_hidden_hooks,
    resolve_decoder_layers,
)


class TinyDecoder(torch.nn.Module):
    def __init__(self, *, layout: str = "model.layers"):
        super().__init__()
        layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.Linear(2, 2, bias=False),
            ]
        )
        for index, layer in enumerate(layers, start=1):
            layer.weight.data.copy_(torch.eye(2) * index)

        if layout == "model.layers":
            self.model = torch.nn.Module()
            self.model.layers = layers
        elif layout == "model.model.language_model.layers":
            self.model = torch.nn.Module()
            self.model.model = torch.nn.Module()
            self.model.model.language_model = torch.nn.Module()
            self.model.model.language_model.layers = layers
        else:
            raise ValueError(layout)

    def forward(self, inputs):
        x = inputs
        for layer in resolve_decoder_layers(self):
            x = layer(x)
        return x


def test_resolve_decoder_layers_supports_common_hf_layouts():
    assert len(resolve_decoder_layers(TinyDecoder(layout="model.layers"))) == 3
    assert len(resolve_decoder_layers(TinyDecoder(layout="model.model.language_model.layers"))) == 3


def test_resolve_decoder_layers_unwraps_fsdp_style_wrappers():
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    wrapped = torch.nn.Module()
    wrapped.model = torch.nn.Module()
    wrapped.model.layers = layers
    module = torch.nn.Module()
    module._fsdp_wrapped_module = wrapped

    assert resolve_decoder_layers(module) is layers


def test_register_mapped_hooks_captures_only_mapped_layers_and_can_clear():
    model = TinyDecoder()
    store = HiddenCaptureStore()
    handles = register_mapped_hidden_hooks(
        model,
        [HiddenLayerMapEntry(student_hidden_index=1, teacher_hidden_index=2)],
        role="teacher",
        store=store,
        detach=True,
    )

    try:
        model(torch.tensor([[1.0, 1.0]], requires_grad=True))

        assert set(store.captures) == {2}
        assert store.captures[2].requires_grad is False

        store.clear()
        assert store.captures == {}
    finally:
        for handle in handles:
            handle.remove()


def test_student_hooks_keep_grad_and_hidden_index_zero_is_not_hookable():
    model = TinyDecoder()
    store = HiddenCaptureStore()

    with pytest.raises(ValueError, match="hidden index 0"):
        register_mapped_hidden_hooks(
            model,
            [HiddenLayerMapEntry(student_hidden_index=0, teacher_hidden_index=1)],
            role="student",
            store=store,
        )

    handles = register_mapped_hidden_hooks(
        model,
        [HiddenLayerMapEntry(student_hidden_index=2, teacher_hidden_index=3)],
        role="student",
        store=store,
        detach=False,
    )
    try:
        model(torch.tensor([[1.0, 1.0]], requires_grad=True))

        assert set(store.captures) == {2}
        assert store.captures[2].requires_grad is True
    finally:
        for handle in handles:
            handle.remove()
