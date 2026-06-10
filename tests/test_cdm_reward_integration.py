from __future__ import annotations

import os

import pytest

from verl_plugins.rewards.aggregate import compute_score
from verl_plugins.rewards.cdm_client import CdmLatexRenderClient


pytestmark = pytest.mark.skipif(
    os.environ.get("CDM_REWARD_INTEGRATION") != "1" or not os.environ.get("CDM_REWARD_URL"),
    reason="set CDM_REWARD_INTEGRATION=1 and CDM_REWARD_URL to run real CDM reward integration tests",
)


def test_real_cdm_reward_service_handles_formula_and_invalid_latex_runtime_paths():
    service_url = os.environ["CDM_REWARD_URL"]
    expected_version = os.environ.get("CDM_REWARD_EXPECTED_VERSION", "cdm_katex_v1")
    timeout_ms = int(os.environ.get("CDM_REWARD_TIMEOUT_MS", "5000"))

    CdmLatexRenderClient(
        service_url=service_url,
        timeout_ms=timeout_ms,
        expected_version=expected_version,
    ).preflight()

    routing = {"default": "normalized_levenshtein_v1", "by_task": {"formula": "cdm_katex_v1"}}
    rewards = {
        "normalized_levenshtein_v1": {"type": "normalized_levenshtein", "version": "levenshtein_v1"},
        "cdm_katex_v1": {
            "type": "cdm_latex_render",
            "version": expected_version,
            "service_url": service_url,
            "timeout_ms": timeout_ms,
            "fail_score": 0.0,
            "expected_version": expected_version,
        },
    }

    valid_result = compute_score(
        data_source="ocr_vlm:region:formula",
        solution_str=r"\frac{1}{2}",
        ground_truth=r"\frac{1}{2}",
        extra_info={"sample_id": "integration-valid", "task_type": "formula"},
        routing=routing,
        rewards=rewards,
    )
    invalid_result = compute_score(
        data_source="ocr_vlm:region:formula",
        solution_str=r"\frac{1}{",
        ground_truth=r"\frac{1}{2}",
        extra_info={"sample_id": "integration-invalid", "task_type": "formula"},
        routing=routing,
        rewards=rewards,
    )
    text_result = compute_score(
        data_source="ocr_vlm:region:text_recognition",
        solution_str="helo",
        ground_truth="hello",
        extra_info={"sample_id": "integration-text", "task_type": "text_recognition"},
        routing=routing,
        rewards=rewards,
    )

    assert valid_result["reward_name"] == "cdm_latex_render"
    assert valid_result["reward_profile_id"] == "cdm_katex_v1"
    assert valid_result["score"] > 0.0
    assert invalid_result["reward_name"] == "cdm_latex_render"
    assert invalid_result["score"] == 0.0
    assert invalid_result["diagnostics"]
    assert text_result["reward_name"] == "normalized_levenshtein"
    assert text_result["reward_profile_id"] == "normalized_levenshtein_v1"
    assert text_result["score"] == 0.8
