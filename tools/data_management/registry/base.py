from __future__ import annotations

from typing import Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        if key in self._items:
            raise ValueError(f"duplicate registry key: {key}")
        self._items[key] = value

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(f"unregistered adapter/profile: {key}") from exc

    def keys(self) -> list[str]:
        return sorted(self._items)
