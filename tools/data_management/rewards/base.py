from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tools.data_management.schemas import RewardResult


class RewardAdapter(ABC):
    name: str
    version: str
    task: str = "*"

    @abstractmethod
    def prepare_payload(self, canonical_record: dict[str, Any], view_record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Prepare reward-specific ground truth and metadata."""

    @abstractmethod
    def score(self, prediction: str, payload: dict[str, Any], context: dict[str, Any]) -> RewardResult:
        """Compute reward for one prediction."""

    def batch_score(
        self,
        predictions: list[str],
        payloads: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[RewardResult]:
        return [self.score(prediction, payload, context) for prediction, payload in zip(predictions, payloads, strict=True)]

    def validate_payload(self, payload: dict[str, Any]) -> None:
        if "label" not in payload:
            raise ValueError(f"{self.name} payload requires label")
