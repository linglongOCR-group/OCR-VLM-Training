from __future__ import annotations

from typing import Any


REWARD_NAME = "normalized_levenshtein"
DEFAULT_REWARD_VERSION = "levenshtein_v1"


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
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (pred_char != truth_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def normalized_levenshtein_reward(
    prediction: str,
    ground_truth: str,
    *,
    task_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    reward_version: str = DEFAULT_REWARD_VERSION,
) -> dict[str, Any]:
    prediction = prediction or ""
    ground_truth = ground_truth or ""
    raw_distance = levenshtein_distance(prediction, ground_truth)

    if not ground_truth:
        normalized_distance = 0.0 if not prediction else 1.0
    else:
        normalized_distance = min(raw_distance / len(ground_truth), 1.0)

    reward_total = max(0.0, 1.0 - normalized_distance)
    result: dict[str, Any] = {
        "reward_total": reward_total,
        "reward_name": REWARD_NAME,
        "reward_version": reward_version,
        "raw_distance": raw_distance,
        "normalized_distance": normalized_distance,
    }
    if task_type is not None:
        result["task_type"] = task_type
    if metadata:
        result["metadata"] = metadata
    return result
