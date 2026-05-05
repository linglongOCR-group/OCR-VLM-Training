from dataclasses import dataclass
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl_plugins.trainers.sft_freeze import (
    enable_fsdp_mixed_requires_grad_if_needed,
    extract_freeze_vision_tower,
    find_vision_tower,
    freeze_vision_tower_if_needed,
    FreezeVisionTowerResult,
    install_freeze_vision_tower_hook,
)


def test_extract_freeze_vision_tower_consumes_model_config_key():
    config = OmegaConf.create({"model": {"path": "model", "freeze_vision_tower": True}})

    enabled = extract_freeze_vision_tower(config)

    assert enabled is True
    assert "freeze_vision_tower" not in config.model


def test_find_vision_tower_prefers_nested_visual_module():
    nested_visual = torch.nn.Linear(2, 2)
    top_visual = torch.nn.Linear(2, 2)
    module = SimpleNamespace(model=SimpleNamespace(visual=nested_visual), visual=top_visual)

    assert find_vision_tower(module) is nested_visual


def test_find_vision_tower_falls_back_to_top_level_visual_module():
    visual = torch.nn.Linear(2, 2)
    module = SimpleNamespace(visual=visual)

    assert find_vision_tower(module) is visual


def test_freeze_vision_tower_if_needed_freezes_only_visual_tower():
    visual = torch.nn.Linear(2, 2)
    language = torch.nn.Linear(2, 2)
    module = SimpleNamespace(model=SimpleNamespace(visual=visual), language=language)

    result = freeze_vision_tower_if_needed(module, enabled=True)

    assert result.frozen is True
    assert all(not param.requires_grad for param in visual.parameters())
    assert all(param.requires_grad for param in language.parameters())


def test_freeze_vision_tower_if_needed_is_noop_when_disabled():
    visual = torch.nn.Linear(2, 2)
    module = SimpleNamespace(visual=visual)

    result = freeze_vision_tower_if_needed(module, enabled=False)

    assert result.frozen is False
    assert result.reason == "disabled"
    assert all(param.requires_grad for param in visual.parameters())


def test_enable_fsdp_mixed_requires_grad_forces_use_orig_params_for_frozen_vision_tower():
    engine = SimpleNamespace(engine_config=SimpleNamespace(strategy="fsdp", use_orig_params=False))
    result = FreezeVisionTowerResult(frozen=True, reason="frozen")

    changed = enable_fsdp_mixed_requires_grad_if_needed(engine, enabled=True, result=result)

    assert changed is True
    assert engine.engine_config.use_orig_params is True


def test_enable_fsdp_mixed_requires_grad_updates_frozen_engine_config():
    @dataclass(frozen=True)
    class FrozenEngineConfig:
        strategy: str = "fsdp"
        use_orig_params: bool = False

    engine = SimpleNamespace(engine_config=FrozenEngineConfig())
    result = FreezeVisionTowerResult(frozen=True, reason="frozen")

    changed = enable_fsdp_mixed_requires_grad_if_needed(engine, enabled=True, result=result)

    assert changed is True
    assert engine.engine_config.use_orig_params is True


def test_install_freeze_vision_tower_hook_is_instance_scoped():
    class DummyEngine:
        def __init__(self):
            self.rank = 1
            self.engine_config = SimpleNamespace(strategy="fsdp2")
            self.calls = []

        def _build_fsdp_module(self, module):
            self.calls.append(module)
            return "wrapped"

    engine = DummyEngine()
    untouched = DummyEngine()
    visual = torch.nn.Linear(2, 2)
    module = SimpleNamespace(model=SimpleNamespace(visual=visual))

    installed = install_freeze_vision_tower_hook(engine, enabled=True)
    result = engine._build_fsdp_module(module)

    assert installed is True
    assert result == "wrapped"
    assert engine.calls == [module]
    assert all(not param.requires_grad for param in visual.parameters())
    assert not getattr(untouched, "_ocr_freeze_vision_tower_hook", False)
    assert untouched._build_fsdp_module.__func__ is DummyEngine._build_fsdp_module


def test_install_freeze_vision_tower_hook_disabled_is_noop():
    engine = SimpleNamespace(_build_fsdp_module=lambda module: module)

    installed = install_freeze_vision_tower_hook(engine, enabled=False)

    assert installed is False
    assert not getattr(engine, "_ocr_freeze_vision_tower_hook", False)
