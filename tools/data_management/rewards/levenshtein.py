from __future__ import annotations

from typing import Any

from tools.data_management.rewards.base import RewardAdapter
from tools.data_management.schemas import RewardResult


class NormalizedLevenshteinReward(RewardAdapter):
    name = "normalized_levenshtein_v1"
    version = "1.0.0"
    task = "*"

    def prepare_payload(self, canonical_record: dict[str, Any], view_record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {
            "label": view_record["label"],
            "normalization": {
                "case_sensitive": bool(config.get("case_sensitive", True)),
                "strip_whitespace": bool(config.get("strip_whitespace", False)),
            },
        }

    def score(self, prediction: str, payload: dict[str, Any], context: dict[str, Any]) -> RewardResult:
        self.validate_payload(payload)
        prediction = _normalize(prediction or "", payload.get("normalization") or {})
        label = _normalize(str(payload.get("label") or ""), payload.get("normalization") or {})
        distance = levenshtein_distance(prediction, label)
        denominator = max(len(prediction), len(label), 1)
        normalized_score = max(0.0, 1.0 - distance / denominator)
        return RewardResult(
            score=normalized_score,
            normalized_score=normalized_score,
            passed=None,
            details={
                "metric": "normalized_levenshtein",
                "edit_distance": distance,
                "denominator": denominator,
                "target_length": len(label),
            },
            error=None,
        )


def _normalize(value: str, options: dict[str, Any]) -> str:
    if options.get("strip_whitespace"):
        value = value.strip()
    if not options.get("case_sensitive", True):
        value = value.lower()
    return value


def levenshtein_distance(prediction: str, ground_truth: str) -> int:
    if prediction == ground_truth:
        return 0
    if not prediction:
        return len(ground_truth)
    if not ground_truth:
        return len(prediction)
    previous = list(range(len(ground_truth) + 1))
    for i, pred_char in enumerate(prediction, start=1):
        current = [i]
        for j, truth_char in enumerate(ground_truth, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (pred_char != truth_char),
                )
            )
        previous = current
    return previous[-1]
