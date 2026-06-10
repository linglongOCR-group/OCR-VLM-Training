import pytest

import verl_plugins.rewards.aggregate as aggregate
from verl_plugins.rewards.aggregate import compute_score
from verl_plugins.rewards.common import normalized_levenshtein_reward


def test_normalized_levenshtein_reward_exact_match():
    result = normalized_levenshtein_reward("invoice 123", "invoice 123")

    assert result["reward_total"] == 1.0
    assert result["normalized_distance"] == 0.0
    assert result["raw_distance"] == 0


def test_normalized_levenshtein_reward_empty_target_policy():
    empty_match = normalized_levenshtein_reward("", "")
    empty_mismatch = normalized_levenshtein_reward("text", "")

    assert empty_match["reward_total"] == 1.0
    assert empty_mismatch["reward_total"] == 0.0
    assert empty_mismatch["normalized_distance"] == 1.0


def test_compute_score_matches_verl_custom_reward_contract():
    result = compute_score(
        data_source="ocr_vlm:region:text_recognition",
        solution_str="helo",
        ground_truth="hello",
        extra_info={"sample_id": "s1", "task_type": "text_recognition"},
        reward_version="test_v1",
    )

    assert result["score"] == 0.8
    assert result["reward_total"] == 0.8
    assert result["reward_name"] == "normalized_levenshtein"
    assert result["reward_version"] == "test_v1"
    assert result["sample_id"] == "s1"
    assert result["task_type"] == "text_recognition"
    assert "data_source" not in result


def test_legacy_reward_version_does_not_select_a_different_implementation():
    result = compute_score(
        data_source="ocr_vlm:region:formula",
        solution_str="x+1",
        ground_truth="x+2",
        extra_info={"task_type": "formula"},
        reward_version="cdm_katex_v1",
    )

    assert result["score"] == result["reward_total"]
    assert result["reward_name"] == "normalized_levenshtein"
    assert result["reward_version"] == "cdm_katex_v1"


def test_compute_score_routes_formula_task_to_configured_cdm_reward(monkeypatch):
    def fake_cdm_reward(prediction, ground_truth, *, task_type=None, metadata=None, reward_version=None, **kwargs):
        assert prediction == r"\\frac{1}{2}"
        assert ground_truth == r"\\frac{1}{3}"
        assert task_type == "formula"
        assert metadata is not None
        assert metadata["sample_id"] == "f1"
        assert kwargs == {"endpoint": "fake"}
        return {
            "reward_total": 0.75,
            "reward_name": "cdm_latex_render",
            "reward_version": reward_version,
        }

    monkeypatch.setitem(aggregate.REWARD_TYPE_REGISTRY, "cdm_latex_render", fake_cdm_reward)

    result = compute_score(
        data_source="ocr_vlm:region:formula",
        solution_str=r"\\frac{1}{2}",
        ground_truth=r"\\frac{1}{3}",
        extra_info={"sample_id": "f1", "task_type": "formula"},
        routing={"default": "normalized_levenshtein_v1", "by_task": {"formula": "cdm_katex_v1"}},
        rewards={
            "normalized_levenshtein_v1": {"type": "normalized_levenshtein"},
            "cdm_katex_v1": {"type": "cdm_latex_render", "version": "cdm_katex_v1", "endpoint": "fake"},
        },
    )

    assert result["score"] == 0.75
    assert result["reward_total"] == 0.75
    assert result["reward_name"] == "cdm_latex_render"
    assert result["reward_version"] == "cdm_katex_v1"
    assert result["reward_profile_id"] == "cdm_katex_v1"
    assert result["sample_id"] == "f1"
    assert result["task_type"] == "formula"


def test_compute_score_treats_reward_profile_as_label_when_routing_is_present(monkeypatch):
    def fake_cdm_reward(prediction, ground_truth, *, task_type=None, metadata=None, reward_version=None, **kwargs):
        assert prediction == r"\\frac{1}{2}"
        assert ground_truth == r"\\frac{1}{3}"
        assert task_type == "formula"
        assert metadata is not None
        assert metadata["sample_id"] == "f1"
        assert kwargs == {"endpoint": "fake"}
        return {
            "reward_total": 0.75,
            "reward_name": "cdm_latex_render",
            "reward_version": reward_version,
        }

    monkeypatch.setitem(aggregate.REWARD_TYPE_REGISTRY, "cdm_latex_render", fake_cdm_reward)

    result = compute_score(
        data_source="ocr_vlm:region:formula",
        solution_str=r"\\frac{1}{2}",
        ground_truth=r"\\frac{1}{3}",
        extra_info={"sample_id": "f1", "task_type": "formula"},
        reward_profile="mixed_doc_rlvr_v1",
        routing={"default": "normalized_levenshtein_v1", "by_task": {"formula": "cdm_katex_v1"}},
        rewards={
            "normalized_levenshtein_v1": {"type": "normalized_levenshtein"},
            "cdm_katex_v1": {"type": "cdm_latex_render", "version": "cdm_katex_v1", "endpoint": "fake"},
        },
    )

    assert result["score"] == 0.75
    assert result["reward_total"] == 0.75
    assert result["reward_name"] == "cdm_latex_render"
    assert result["reward_version"] == "cdm_katex_v1"
    assert result["reward_profile_id"] == "cdm_katex_v1"
    assert result["sample_id"] == "f1"
    assert result["task_type"] == "formula"


def test_compute_score_infers_formula_task_from_data_source_suffix(monkeypatch):
    def fake_cdm_reward(prediction, ground_truth, *, task_type=None, metadata=None, reward_version=None, **kwargs):
        assert task_type == "formula"
        return {
            "reward_total": 0.9,
            "reward_name": "cdm_latex_render",
            "reward_version": reward_version,
        }

    monkeypatch.setitem(aggregate.REWARD_TYPE_REGISTRY, "cdm_latex_render", fake_cdm_reward)

    result = compute_score(
        data_source="ocr_vlm:region:formula",
        solution_str=r"\\int x dx",
        ground_truth=r"\\int x dx",
        extra_info={"sample_id": "f2"},
        routing={"default": "normalized_levenshtein_v1", "by_task": {"formula": "cdm_katex_v1"}},
        rewards={
            "normalized_levenshtein_v1": {"type": "normalized_levenshtein", "version": "levenshtein_v1"},
            "cdm_katex_v1": {"type": "cdm_latex_render", "version": "cdm_katex_v2"},
        },
    )

    assert result["score"] == 0.9
    assert result["reward_profile_id"] == "cdm_katex_v1"
    assert result["reward_version"] == "cdm_katex_v2"
    assert result["task_type"] == "formula"


def test_compute_score_uses_profile_version_for_reward_provenance():
    result = compute_score(
        data_source="formula",
        solution_str="x+1",
        ground_truth="x+1",
        extra_info={},
        routing={"default": "normalized_levenshtein_v1", "by_task": {"formula": "formula_levenshtein"}},
        rewards={
            "normalized_levenshtein_v1": {"type": "normalized_levenshtein", "version": "default_profile_v1"},
            "formula_levenshtein": {"type": "normalized_levenshtein", "version": "profile_version_v2"},
        },
    )

    assert result["reward_profile_id"] == "formula_levenshtein"
    assert result["reward_version"] == "profile_version_v2"
    assert result["task_type"] == "formula"


def test_compute_score_uses_default_profile_for_non_formula_and_unknown_tasks():
    routing = {"default": "normalized_levenshtein_v1", "by_task": {"formula": "cdm_katex_v1"}}
    rewards = {
        "normalized_levenshtein_v1": {"type": "normalized_levenshtein", "reward_version": "levenshtein_v1"},
        "cdm_katex_v1": {"type": "cdm_latex_render", "reward_version": "cdm_katex_v1"},
    }

    text_result = compute_score(
        data_source="ocr_vlm:region:text_recognition",
        solution_str="helo",
        ground_truth="hello",
        extra_info={"task_type": "text_recognition"},
        routing=routing,
        rewards=rewards,
    )
    unknown_result = compute_score(
        data_source="ocr_vlm:region:other",
        solution_str="abc",
        ground_truth="abc",
        extra_info={"task_type": "table"},
        routing=routing,
        rewards=rewards,
    )

    assert text_result["reward_name"] == "normalized_levenshtein"
    assert text_result["reward_profile_id"] == "normalized_levenshtein_v1"
    assert text_result["score"] == text_result["reward_total"] == 0.8
    assert unknown_result["reward_name"] == "normalized_levenshtein"
    assert unknown_result["reward_profile_id"] == "normalized_levenshtein_v1"
    assert unknown_result["score"] == unknown_result["reward_total"] == 1.0


def test_compute_score_raises_clear_error_for_missing_profile():
    with pytest.raises(ValueError, match="missing reward profile 'missing_profile'"):
        compute_score(
            data_source="ocr_vlm:region:formula",
            solution_str="x",
            ground_truth="x",
            extra_info={"task_type": "formula"},
            routing={"default": "normalized_levenshtein_v1", "by_task": {"formula": "missing_profile"}},
            rewards={"normalized_levenshtein_v1": {"type": "normalized_levenshtein"}},
        )


def test_compute_score_raises_clear_error_for_unsupported_reward_type():
    with pytest.raises(ValueError, match="unsupported reward type 'exact_match'"):
        compute_score(
            data_source="ocr_vlm:region:text_recognition",
            solution_str="x",
            ground_truth="x",
            extra_info={"task_type": "text_recognition"},
            routing={"default": "exact_match_v1"},
            rewards={"exact_match_v1": {"type": "exact_match"}},
        )
