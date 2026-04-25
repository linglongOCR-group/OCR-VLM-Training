from __future__ import annotations

from typing import Callable

from docparse_dataset.serializers.base import TargetSerializer


class SerializerRegistry:
    _serializers: dict[str, Callable[[], TargetSerializer]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], TargetSerializer]) -> None:
        cls._serializers[name] = factory

    @classmethod
    def get(cls, name: str) -> TargetSerializer:
        if name not in cls._serializers:
            raise KeyError(f"serializer not registered: {name!r}")
        return cls._serializers[name]()

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._serializers)
