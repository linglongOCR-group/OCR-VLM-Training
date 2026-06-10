from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from verl_plugins.rewards.cdm_client import CdmLatexRenderClient
from verl_plugins.rewards.common import DEFAULT_REWARD_VERSION, normalized_levenshtein_reward

RewardFn = Callable[..., dict[str, Any]]


def cdm_latex_render_reward(
    prediction: str,
    ground_truth: str,
    *,
    task_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    reward_version: str = "cdm_katex_v1",
    client: Any | None = None,
    service_url: str | None = None,
    timeout_ms: int | float = 1000,
    fail_score: float = 0.0,
    expected_version: str | None = None,
    opener: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if client is None and service_url:
        client = CdmLatexRenderClient(
            service_url=service_url,
            timeout_ms=timeout_ms,
            fail_score=fail_score,
            expected_version=expected_version,
            opener=opener,
        )
    if client is None:
        return {
            "reward_total": 0.0,
            "reward_name": "cdm_latex_render",
            "reward_version": reward_version,
            "diagnostics": {"error_type": "configuration_error", "message": "cdm_latex_render reward client is not configured"},
        }
    return client(
        prediction=prediction,
        ground_truth=ground_truth,
        task_type=task_type,
        metadata=metadata,
        reward_version=reward_version,
        **kwargs,
    )


REWARD_TYPE_REGISTRY: dict[str, RewardFn] = {
    "normalized_levenshtein": normalized_levenshtein_reward,
    "cdm_latex_render": cdm_latex_render_reward,
}


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


def _infer_task_type(*, extra_info: Mapping[str, Any], data_source: str) -> str | None:
    task_type = extra_info.get("task_type")
    if task_type:
        return str(task_type)
    if not data_source:
        return None
    return data_source.rsplit(":", 1)[-1] or None


def _select_reward_profile_id(
    *,
    task_type: str | None,
    routing: Mapping[str, Any] | None,
    reward_profile: str | None,
) -> str | None:
    if routing:
        by_task = routing.get("by_task") or {}
        if task_type and task_type in by_task:
            return by_task[task_type]
        return routing.get("default")
    if reward_profile:
        return reward_profile
    return None


def _resolve_reward_profile(
    *,
    profile_id: str,
    rewards: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not rewards or profile_id not in rewards:
        raise ValueError(f"missing reward profile '{profile_id}'")
    profile = rewards[profile_id]
    if not isinstance(profile, Mapping):
        raise ValueError(f"reward profile '{profile_id}' must be a mapping")
    return profile


def _compute_profile_score(
    *,
    profile_id: str,
    profile: Mapping[str, Any],
    solution_str: str,
    ground_truth: str,
    task_type: str | None,
    metadata: dict[str, Any],
    fallback_reward_version: str,
) -> dict[str, Any]:
    reward_type = profile.get("type")
    if not reward_type:
        raise ValueError(f"reward profile '{profile_id}' is missing required field 'type'")
    reward_fn = REWARD_TYPE_REGISTRY.get(str(reward_type))
    if reward_fn is None:
        raise ValueError(f"unsupported reward type '{reward_type}' in reward profile '{profile_id}'")

    profile_kwargs = dict(profile)
    profile_kwargs.pop("type", None)
    profile_reward_version = profile_kwargs.pop("version", None)
    if profile_reward_version is None:
        profile_reward_version = profile_kwargs.pop("reward_version", fallback_reward_version)
    else:
        profile_kwargs.pop("reward_version", None)

    result = reward_fn(
        prediction=solution_str,
        ground_truth=ground_truth,
        task_type=task_type,
        metadata=metadata,
        reward_version=profile_reward_version,
        **profile_kwargs,
    )
    result.setdefault("reward_name", str(reward_type))
    result.setdefault("reward_version", profile_reward_version)
    result["reward_profile_id"] = profile_id
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    reward_version: str = DEFAULT_REWARD_VERSION,
    **kwargs: Any,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    task_type = _infer_task_type(extra_info=extra_info, data_source=data_source)
    routing = kwargs.get("routing")
    rewards = kwargs.get("rewards")
    reward_profile = kwargs.get("reward_profile")
    profile_id = _select_reward_profile_id(
        task_type=task_type,
        routing=routing,
        reward_profile=reward_profile,
    )

    if profile_id is None:
        result = reward(
            prediction=solution_str,
            ground_truth=ground_truth,
            task_type=task_type,
            metadata=extra_info,
            reward_version=reward_version,
        )
    else:
        profile = _resolve_reward_profile(profile_id=profile_id, rewards=rewards)
        result = _compute_profile_score(
            profile_id=profile_id,
            profile=profile,
            solution_str=solution_str,
            ground_truth=ground_truth,
            task_type=task_type,
            metadata=extra_info,
            fallback_reward_version=reward_version,
        )

    result["score"] = result["reward_total"]
    if "sample_id" in extra_info:
        result["sample_id"] = extra_info["sample_id"]
    if task_type is not None:
        result["task_type"] = task_type
    return result
