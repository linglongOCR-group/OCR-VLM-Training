from __future__ import annotations

from typing import Any

from src.rewards.common import DEFAULT_REWARD_VERSION, normalized_levenshtein_reward


def reward(
    prediction: str,
    ground_truth: str,
    task_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    reward_version: str = DEFAULT_REWARD_VERSION,
) -> dict[str, Any]:
    return normalized_levenshtein_reward(
        prediction=prediction,
        ground_truth=ground_truth,
        task_type=task_type,
        metadata=metadata,
        reward_version=reward_version,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    reward_version: str = DEFAULT_REWARD_VERSION,
    **_: Any,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    result = reward(
        prediction=solution_str,
        ground_truth=ground_truth,
        task_type=extra_info.get("task_type"),
        metadata=extra_info,
        reward_version=reward_version,
    )
    result["score"] = result["reward_total"]
    if "sample_id" in extra_info:
        result["sample_id"] = extra_info["sample_id"]
    if "task_type" in extra_info:
        result["task_type"] = extra_info["task_type"]
    return result
