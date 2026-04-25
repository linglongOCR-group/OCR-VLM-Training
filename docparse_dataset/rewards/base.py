from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardResult:
    score: float
    normalized_score: float
    passed: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class RewardAdapter(ABC):
    """Computes reward from model prediction + reward payload + optional context."""

    name: str
    version: str
    task: str

    @abstractmethod
    def prepare_payload(self, canonical_record: dict, view_record: dict, config: dict) -> dict:
        """Prepare reward-specific ground truth and metadata."""

    @abstractmethod
    def score(self, prediction: str, payload: dict, context: dict | None = None) -> RewardResult:
        """Compute reward for one prediction."""

    def batch_score(self, predictions: list[str], payloads: list[dict], context: dict | None = None) -> list[RewardResult]:
        return [self.score(p, q, context) for p, q in zip(predictions, payloads)]

    def validate_payload(self, payload: dict) -> None:
        """Check reward payload integrity."""
