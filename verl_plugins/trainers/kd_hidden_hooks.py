from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch


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
        if isinstance(value, (tuple, list)):
            value = value[0]
        if detach and hasattr(value, "detach"):
            value = value.detach()
        self.captures[hidden_index] = value


def resolve_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList | list[torch.nn.Module]:
    for root in _candidate_roots(model):
        for path in (
            "layers",
            "model.layers",
            "model.model.layers",
            "language_model.layers",
            "model.language_model.layers",
            "model.model.language_model.layers",
        ):
            layers = _resolve_attr_path(root, path)
            if layers is not None:
                return layers
    raise ValueError("Could not resolve decoder layers from common HF module layouts")


def register_mapped_hidden_hooks(
    model: torch.nn.Module,
    layer_map: Iterable[HiddenLayerMapEntry],
    *,
    role: Literal["student", "teacher"],
    store: HiddenCaptureStore,
    detach: bool | None = None,
) -> list[Any]:
    if detach is None:
        detach = role == "teacher"

    hidden_indices = {
        _entry_value(entry, "student_hidden_index") if role == "student" else _entry_value(entry, "teacher_hidden_index")
        for entry in layer_map
    }
    return register_hidden_hooks(model, hidden_indices, store=store, detach=detach)


def register_hidden_hooks(
    model: torch.nn.Module,
    hidden_indices: Iterable[int],
    *,
    store: HiddenCaptureStore,
    detach: bool,
) -> list[Any]:
    layers = resolve_decoder_layers(model)
    handles = []
    for hidden_index in sorted(set(int(index) for index in hidden_indices)):
        if hidden_index == 0:
            raise ValueError("hidden index 0 is the embedding output and is not hookable through decoder-layer hooks")
        layer_index = hidden_index - 1
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"hidden index {hidden_index} is out of range for {len(layers)} decoder layers")
        handles.append(
            layers[layer_index].register_forward_hook(
                _make_capture_hook(hidden_index, store=store, detach=detach)
            )
        )
    return handles


def _entry_value(entry: Mapping[str, Any] | Any, key: str) -> int:
    if isinstance(entry, Mapping):
        return int(entry[key])
    return int(getattr(entry, key))


def _make_capture_hook(hidden_index: int, *, store: HiddenCaptureStore, detach: bool):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        store.record(hidden_index, output, detach=detach)

    return hook


def _resolve_attr_path(obj: Any, path: str) -> Any | None:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    if isinstance(current, (torch.nn.ModuleList, list, tuple)):
        return current
    return None


def _candidate_roots(model: Any) -> list[Any]:
    roots = [model]
    for attr in ("module", "_fsdp_wrapped_module"):
        wrapped = getattr(model, attr, None)
        if wrapped is not None and wrapped not in roots:
            roots.append(wrapped)
    return roots
