from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any

from omegaconf import DictConfig, open_dict

from verl_plugins.trainers._utils import _as_bool


@dataclass(frozen=True)
class FreezeVisionTowerResult:
    frozen: bool
    reason: str


def extract_freeze_vision_tower(config: DictConfig | dict[str, Any]) -> bool:
    """Consume repo-local SFT freeze config before VERL builds HFModelConfig."""
    model_config = config.get("model", {}) if hasattr(config, "get") else {}
    if "freeze_vision_tower" not in model_config:
        return False

    with open_dict(model_config):
        raw_value = model_config.pop("freeze_vision_tower")
    return _as_bool(raw_value)


def find_vision_tower(module: Any) -> Any | None:
    nested_model = getattr(module, "model", None)
    if nested_model is not None and hasattr(nested_model, "visual"):
        return nested_model.visual
    if hasattr(module, "visual"):
        return module.visual
    return None


def freeze_vision_tower_if_needed(module: Any, *, enabled: bool) -> FreezeVisionTowerResult:
    if not enabled:
        return FreezeVisionTowerResult(frozen=False, reason="disabled")

    vision_tower = find_vision_tower(module)
    if vision_tower is None:
        return FreezeVisionTowerResult(frozen=False, reason="not_found")

    vision_tower.requires_grad_(False)
    return FreezeVisionTowerResult(frozen=True, reason="frozen")


def enable_fsdp_mixed_requires_grad_if_needed(
    engine: Any, *, enabled: bool, result: FreezeVisionTowerResult
) -> bool:
    if not enabled or not result.frozen:
        return False

    engine_config = getattr(engine, "engine_config", None)
    if engine_config is None or getattr(engine_config, "strategy", None) != "fsdp":
        return False
    if getattr(engine_config, "use_orig_params", False):
        return False

    try:
        engine_config.use_orig_params = True
    except FrozenInstanceError:
        object.__setattr__(engine_config, "use_orig_params", True)
    return True


def install_freeze_vision_tower_hook(engine: Any, *, enabled: bool) -> bool:
    if not enabled or getattr(engine, "_ocr_freeze_vision_tower_hook", False):
        return False

    original_build_fsdp_module = engine._build_fsdp_module

    def build_fsdp_module_with_frozen_vision(module):
        result = freeze_vision_tower_if_needed(module, enabled=True)
        use_orig_params_changed = enable_fsdp_mixed_requires_grad_if_needed(engine, enabled=True, result=result)
        if getattr(engine, "rank", 0) == 0:
            if result.frozen:
                print("[sft model] Vision tower frozen before FSDP wrapping.")
                if use_orig_params_changed:
                    print("[sft model] Set engine.use_orig_params=True for frozen FSDP parameters.")
            else:
                print("[sft model] freeze_vision_tower=True, but no vision tower was found.")
        return original_build_fsdp_module(module)

    build_fsdp_module_with_frozen_vision._ocr_original_build_fsdp_module = original_build_fsdp_module
    engine._build_fsdp_module = build_fsdp_module_with_frozen_vision
    engine._ocr_freeze_vision_tower_hook = True
    return True
