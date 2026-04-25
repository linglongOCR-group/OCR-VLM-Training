from __future__ import annotations

from typing import Any

from docparse_dataset.registry.reward_registry import RewardRegistry
from docparse_dataset.rewards.levenshtein import NormalizedLevenshteinReward


class VerlRewardWrapper:
    """Thin VERL-compatible reward function wrapper.

    Reads reward_profile_id, loads the adapter from registry,
    dispatches scoring, and returns a scalar reward.
    """

    def __init__(self, reward_registry: RewardRegistry | None = None):
        self._registry = reward_registry or RewardRegistry

    def __call__(self, data_item: dict, model_output: str) -> float:
        profile_id = data_item.get("reward_profile_id", "normalized_levenshtein_v1")
        adapter = self._registry.get(profile_id)

        payload = data_item.get("reward_payload")
        if payload is None:
            # Could load from sidecar store here in the future
            payload = {"label": data_item.get("reward_model", {}).get("ground_truth", "")}

        result = adapter.score(
            prediction=model_output,
            payload=payload,
            context={
                "task": data_item.get("extra_info", {}).get("task_type", ""),
                "sample_id": data_item.get("extra_info", {}).get("sample_id", ""),
            },
        )
        return result.normalized_score


# Legacy backward-compatible compute_score
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    reward_version: str = "levenshtein_v1",
    **_: Any,
) -> dict[str, Any]:
    from docparse_dataset.rewards.levenshtein import normalized_levenshtein_reward

    extra_info = extra_info or {}
    result = normalized_levenshtein_reward(
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
