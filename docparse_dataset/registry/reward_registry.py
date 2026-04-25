from __future__ import annotations

from typing import Callable

from docparse_dataset.rewards.base import RewardAdapter


class RewardRegistry:
    _adapters: dict[str, Callable[[], RewardAdapter]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], RewardAdapter]) -> None:
        cls._adapters[name] = factory

    @classmethod
    def get(cls, name: str) -> RewardAdapter:
        if name not in cls._adapters:
            raise KeyError(f"reward adapter not registered: {name!r}")
        return cls._adapters[name]()

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._adapters)
