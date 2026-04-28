from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TargetSerializer(ABC):
    name: str
    version: str
    task: str

    @abstractmethod
    def serialize(self, canonical_record: dict[str, Any], context: dict[str, Any]) -> str:
        """Convert a canonical target into a model-specific label string."""

    def validate(self, canonical_record: dict[str, Any]) -> None:
        if canonical_record.get("task") != self.task:
            raise ValueError(f"{self.name} only supports task={self.task}")
