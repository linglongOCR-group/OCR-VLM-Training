from __future__ import annotations

from typing import Any

from docparse_dataset.rewards.base import RewardAdapter, RewardResult


class NormalizedLevenshteinReward(RewardAdapter):
    """Normalized Levenshtein distance reward. Phase 2 initial implementation."""

    name = "normalized_levenshtein_v1"
    version = "levenshtein_v1"
    task = "*"

    def prepare_payload(self, canonical_record: dict, view_record: dict, config: dict | None = None) -> dict:
        return {
            "label": view_record.get("label", ""),
            "normalization": {
                "case_sensitive": True,
                "strip_whitespace": False,
            },
        }

    def score(self, prediction: str, payload: dict, context: dict | None = None) -> RewardResult:
        prediction = prediction or ""
        label = payload.get("label", "")
        normalize = payload.get("normalization", {})
        if normalize.get("strip_whitespace"):
            prediction = prediction.strip()
            label = label.strip()
        if not normalize.get("case_sensitive", True):
            prediction = prediction.lower()
            label = label.lower()

        raw_distance = _levenshtein_distance(prediction, label)
        if not label:
            normalized_distance = 0.0 if not prediction else 1.0
        else:
            normalized_distance = min(raw_distance / len(label), 1.0)

        score = max(0.0, 1.0 - normalized_distance)
        return RewardResult(
            score=score,
            normalized_score=score,
            details={
                "metric": "normalized_levenshtein",
                "edit_distance": raw_distance,
                "target_length": len(label),
                "prediction_length": len(prediction),
            },
        )

    def validate_payload(self, payload: dict) -> None:
        if "label" not in payload:
            raise ValueError("reward payload must contain 'label'")


def _levenshtein_distance(prediction: str, ground_truth: str) -> int:
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
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (pred_char != truth_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


# Legacy backward-compatible functions
REWARD_NAME = "normalized_levenshtein"
DEFAULT_REWARD_VERSION = "levenshtein_v1"


def normalized_levenshtein_reward(
    prediction: str,
    ground_truth: str,
    *,
    task_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    reward_version: str = DEFAULT_REWARD_VERSION,
) -> dict[str, Any]:
    adapter = NormalizedLevenshteinReward()
    result = adapter.score(prediction, {"label": ground_truth or ""})
    output: dict[str, Any] = {
        "reward_total": result.normalized_score,
        "reward_name": REWARD_NAME,
        "reward_version": reward_version,
        "raw_distance": result.details["edit_distance"],
        "normalized_distance": 1.0 - result.normalized_score,
    }
    if task_type is not None:
        output["task_type"] = task_type
    if metadata:
        output["metadata"] = metadata
    return output
