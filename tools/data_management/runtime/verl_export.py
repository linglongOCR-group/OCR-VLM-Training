from __future__ import annotations

from typing import Any

from tools.data_management.registry.reward_registry import default_reward_registry


class VerlRewardWrapper:
    def __init__(self) -> None:
        self.reward_registry = default_reward_registry()

    def __call__(self, data_item: dict[str, Any], model_output: str) -> float:
        profile_id = data_item["reward_profile_id"]
        adapter = self.reward_registry.get(profile_id)
        payload = data_item.get("reward_payload")
        if payload is None:
            raise ValueError("sidecar reward payloads are not implemented in Phase 2")
        result = adapter.score(
            prediction=model_output,
            payload=payload,
            context={"task": data_item.get("task"), "view_record_id": data_item.get("id")},
        )
        return result.normalized_score
