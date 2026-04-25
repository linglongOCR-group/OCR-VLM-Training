from __future__ import annotations

from typing import Callable

from docparse_dataset.sources.adapters.base import SourceAdapter


class SourceRegistry:
    _adapters: dict[str, Callable[[], SourceAdapter]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], SourceAdapter]) -> None:
        cls._adapters[name] = factory

    @classmethod
    def get(cls, name: str) -> SourceAdapter:
        if name not in cls._adapters:
            raise KeyError(f"source adapter not registered: {name!r}")
        return cls._adapters[name]()

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._adapters)
