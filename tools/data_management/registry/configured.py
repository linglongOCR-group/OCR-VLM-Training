from __future__ import annotations

from typing import Any

from tools.data_management.config.resolver import ProcessingConfig, import_from_dotted_path
from tools.data_management.registry.base import Registry
from tools.data_management.registry.reward_registry import default_reward_registry
from tools.data_management.registry.serializer_registry import default_serializer_registry


def configured_source_registry(processing: ProcessingConfig) -> Registry[type[Any]]:
    registry: Registry[type[Any]] = Registry()
    for key, import_path in (processing.registries.get("source_adapters") or {}).items():
        registry.register(key, import_from_dotted_path(import_path))
    return registry


def configured_serializer_registry(processing: ProcessingConfig) -> Registry[Any]:
    entries = processing.registries.get("serializers") or {}
    if not entries:
        return default_serializer_registry()
    registry: Registry[Any] = Registry()
    for key, import_path in entries.items():
        registry.register(key, import_from_dotted_path(import_path)())
    return registry


def configured_reward_registry(processing: ProcessingConfig) -> Registry[Any]:
    entries = processing.registries.get("rewards") or {}
    if not entries:
        return default_reward_registry()
    registry: Registry[Any] = Registry()
    for key, import_path in entries.items():
        registry.register(key, import_from_dotted_path(import_path)())
    return registry
