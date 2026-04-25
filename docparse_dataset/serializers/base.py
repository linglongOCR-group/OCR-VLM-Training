from __future__ import annotations

from abc import ABC, abstractmethod


class TargetSerializer(ABC):
    """Converts a canonical target into a model-specific label string."""

    name: str
    version: str
    task: str

    @abstractmethod
    def serialize(self, canonical_record: dict, context: dict | None = None) -> str:
        """Convert canonical target into model-specific label string."""

    def validate(self, canonical_record: dict) -> None:
        """Optional task-specific validation."""
