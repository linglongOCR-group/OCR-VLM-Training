from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping
from typing import Any

from verl_plugins.rewards.cdm_client import CdmLatexRenderClient, CdmPreflightError

DEFAULT_CDM_REWARD_URL = "http://127.0.0.1:8765"
DEFAULT_CDM_REWARD_TIMEOUT_MS = 1000
DEFAULT_CDM_REWARD_FAIL_SCORE = 0.0
DEFAULT_CDM_REWARD_EXPECTED_VERSION = "cdm_katex_v1"


def run_preflight(
    *,
    routing: Mapping[str, Any] | None,
    rewards: Mapping[str, Any] | None,
    preflight_required: str | bool | None = None,
    client_cls: type[CdmLatexRenderClient] = CdmLatexRenderClient,
) -> str:
    if preflight_required is None:
        preflight_required = os.environ.get("CDM_REWARD_PREFLIGHT_REQUIRED", "True")
    if not _truthy(preflight_required):
        return "disabled"
    if not _has_cdm_route(routing=routing, rewards=rewards):
        return "skipped"

    service_url = os.environ.get("CDM_REWARD_URL", DEFAULT_CDM_REWARD_URL)
    timeout_ms = _parse_int_env("CDM_REWARD_TIMEOUT_MS", DEFAULT_CDM_REWARD_TIMEOUT_MS)
    fail_score = _parse_float_env("CDM_REWARD_FAIL_SCORE", DEFAULT_CDM_REWARD_FAIL_SCORE)
    expected_version = os.environ.get("CDM_REWARD_EXPECTED_VERSION", DEFAULT_CDM_REWARD_EXPECTED_VERSION)

    client = client_cls(
        service_url=service_url,
        timeout_ms=timeout_ms,
        fail_score=fail_score,
        expected_version=expected_version,
    )
    try:
        client.preflight()
    except CdmPreflightError as exc:
        raise CdmPreflightError(f"CDM preflight failed for {service_url}: {exc}") from exc
    return "checked"


def _has_cdm_route(*, routing: Mapping[str, Any] | None, rewards: Mapping[str, Any] | None) -> bool:
    if not routing or not rewards:
        return False
    profile_ids = []
    default_profile = routing.get("default")
    if default_profile:
        profile_ids.append(default_profile)
    by_task = routing.get("by_task") or {}
    if isinstance(by_task, Mapping):
        profile_ids.extend(by_task.values())

    for profile_id in profile_ids:
        profile = rewards.get(str(profile_id))
        if isinstance(profile, Mapping) and profile.get("type") == "cdm_latex_render":
            return True
    return False


def _truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = float(value)
    except ValueError as exc:
        raise CdmPreflightError(f"{name} must be an integer, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise CdmPreflightError(f"{name} must be finite, got {value!r}")
    if not parsed.is_integer():
        raise CdmPreflightError(f"{name} must be an integer, got {value!r}")
    return int(parsed)


def _parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name, str(default))
    try:
        parsed = float(value)
    except ValueError as exc:
        raise CdmPreflightError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise CdmPreflightError(f"{name} must be finite, got {value!r}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GRPO CDM reward launch preflight.")
    parser.add_argument("--routing-default", default=None)
    parser.add_argument("--formula-profile", default=None)
    parser.add_argument("--cdm-profile", default=None)
    parser.add_argument("--cdm-preflight-required", default=os.environ.get("CDM_REWARD_PREFLIGHT_REQUIRED", "True"))
    args = parser.parse_args(argv)

    routing: dict[str, Any] = {}
    if args.routing_default:
        routing["default"] = args.routing_default
    if args.formula_profile:
        routing["by_task"] = {"formula": args.formula_profile}

    rewards: dict[str, Any] = {}
    if args.cdm_profile:
        rewards[args.cdm_profile] = {"type": "cdm_latex_render"}
    for profile_id in [args.routing_default, args.formula_profile]:
        if profile_id and profile_id not in rewards:
            rewards[profile_id] = {"type": "normalized_levenshtein"}

    try:
        status = run_preflight(routing=routing, rewards=rewards, preflight_required=args.cdm_preflight_required)
    except CdmPreflightError as exc:
        print(str(exc), flush=True)
        return 1
    print(f"GRPO CDM preflight {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
