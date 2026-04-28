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
