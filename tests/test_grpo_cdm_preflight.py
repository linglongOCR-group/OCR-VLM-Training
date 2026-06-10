from __future__ import annotations

import pytest

from verl_plugins.rewards.cdm_client import CdmPreflightError
from tools.training_ops.grpo_cdm_preflight import run_preflight


class FakeCdmClient:
    calls = []
    should_fail = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).calls.append(kwargs)

    def preflight(self):
        if type(self).should_fail:
            raise CdmPreflightError("CDM service browser not ready")


def setup_function():
    FakeCdmClient.calls = []
    FakeCdmClient.should_fail = False


def test_cdm_enabled_preflight_uses_env_defaults_and_runs_probe(monkeypatch):
    monkeypatch.setenv("CDM_REWARD_URL", "http://cdm.example:8765")
    monkeypatch.setenv("CDM_REWARD_TIMEOUT_MS", "2500")
    monkeypatch.setenv("CDM_REWARD_FAIL_SCORE", "0.25")
    monkeypatch.setenv("CDM_REWARD_EXPECTED_VERSION", "cdm_katex_v2")

    status = run_preflight(
        routing={"default": "lev", "by_task": {"formula": "cdm"}},
        rewards={"lev": {"type": "normalized_levenshtein"}, "cdm": {"type": "cdm_latex_render"}},
        client_cls=FakeCdmClient,
    )

    assert status == "checked"
    assert FakeCdmClient.calls == [
        {
            "service_url": "http://cdm.example:8765",
            "timeout_ms": 2500,
            "fail_score": 0.25,
            "expected_version": "cdm_katex_v2",
        }
    ]


def test_cdm_enabled_preflight_failure_identifies_url_and_readiness_condition(monkeypatch):
    monkeypatch.setenv("CDM_REWARD_URL", "http://cdm.example:8765")
    FakeCdmClient.should_fail = True

    with pytest.raises(CdmPreflightError, match="http://cdm.example:8765.*browser not ready"):
        run_preflight(
            routing={"default": "cdm"},
            rewards={"cdm": {"type": "cdm_latex_render"}},
            client_cls=FakeCdmClient,
        )


def test_levenshtein_only_preflight_skips_cdm_service(monkeypatch):
    monkeypatch.setenv("CDM_REWARD_URL", "http://unreachable-cdm.example:8765")

    status = run_preflight(
        routing={"default": "lev"},
        rewards={"lev": {"type": "normalized_levenshtein"}},
        client_cls=FakeCdmClient,
    )

    assert status == "skipped"
    assert FakeCdmClient.calls == []


def test_preflight_disabled_skips_cdm_service_even_when_routed(monkeypatch):
    monkeypatch.setenv("CDM_REWARD_PREFLIGHT_REQUIRED", "False")

    status = run_preflight(
        routing={"default": "cdm"},
        rewards={"cdm": {"type": "cdm_latex_render"}},
        client_cls=FakeCdmClient,
    )

    assert status == "disabled"
    assert FakeCdmClient.calls == []


def test_preflight_required_argument_overrides_env_default(monkeypatch):
    monkeypatch.delenv("CDM_REWARD_PREFLIGHT_REQUIRED", raising=False)

    status = run_preflight(
        routing={"default": "cdm"},
        rewards={"cdm": {"type": "cdm_latex_render"}},
        preflight_required="False",
        client_cls=FakeCdmClient,
    )

    assert status == "disabled"
    assert FakeCdmClient.calls == []


@pytest.mark.parametrize(
    ("env_name", "env_value", "message"),
    [
        ("CDM_REWARD_TIMEOUT_MS", "slow", "CDM_REWARD_TIMEOUT_MS"),
        ("CDM_REWARD_FAIL_SCORE", "zero", "CDM_REWARD_FAIL_SCORE"),
        ("CDM_REWARD_FAIL_SCORE", "nan", "CDM_REWARD_FAIL_SCORE.*finite"),
    ],
)
def test_malformed_numeric_env_raises_preflight_error(monkeypatch, env_name, env_value, message):
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(CdmPreflightError, match=message):
        run_preflight(
            routing={"default": "cdm"},
            rewards={"cdm": {"type": "cdm_latex_render"}},
            client_cls=FakeCdmClient,
        )

    assert FakeCdmClient.calls == []
