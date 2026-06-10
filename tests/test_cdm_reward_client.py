from __future__ import annotations

import json
import socket
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from verl_plugins.rewards.cdm_client import CdmLatexRenderClient, CdmPreflightError
from verl_plugins.rewards.aggregate import compute_score


class FakeHttpResponse:
    def __init__(self, payload, *, status=200):
        self.status = status
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.timeouts = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def request_json(request):
    return json.loads(request.data.decode("utf-8"))


def test_health_success_returns_compact_service_metadata():
    opener = FakeOpener([
        FakeHttpResponse({
            "service": "cdm-latex-render",
            "version": "cdm_katex_v1",
            "renderer": "katex-chromium",
            "browser_ready": True,
        })
    ])
    client = CdmLatexRenderClient(
        service_url="http://cdm.test/",
        timeout_ms=2500,
        expected_version="cdm_katex_v1",
        opener=opener,
    )

    health = client.check_health()

    assert health == {
        "service": "cdm-latex-render",
        "version": "cdm_katex_v1",
        "renderer": "katex-chromium",
        "browser_ready": True,
    }
    assert opener.requests[0].full_url == "http://cdm.test/health"
    assert opener.timeouts == [2.5]


def test_health_wrong_version_raises_preflight_error():
    opener = FakeOpener([
        FakeHttpResponse({"version": "other", "browser_ready": True})
    ])
    client = CdmLatexRenderClient(
        service_url="http://cdm.test",
        expected_version="cdm_katex_v1",
        opener=opener,
    )

    with pytest.raises(CdmPreflightError, match="expected cdm_katex_v1"):
        client.check_health()


def test_health_browser_not_ready_raises_preflight_error():
    opener = FakeOpener([
        FakeHttpResponse({"version": "cdm_katex_v1", "browser_ready": False})
    ])
    client = CdmLatexRenderClient(service_url="http://cdm.test", opener=opener)

    with pytest.raises(CdmPreflightError, match="browser not ready"):
        client.check_health()


@pytest.mark.parametrize("flag", ["ok", "healthy"])
def test_health_explicit_unhealthy_flag_raises_preflight_error(flag):
    opener = FakeOpener([
        FakeHttpResponse({"version": "cdm_katex_v1", "browser_ready": True, flag: False})
    ])
    client = CdmLatexRenderClient(service_url="http://cdm.test", opener=opener)

    with pytest.raises(CdmPreflightError, match=flag):
        client.check_health()


@pytest.mark.parametrize(
    ("health_response", "message"),
    [
        (URLError("connection refused"), "connection refused"),
        (HTTPError("http://cdm.test/health", 503, "unavailable", hdrs=Message(), fp=None), "unavailable"),
        (FakeHttpResponse(b"not json"), "health check failed"),
        (FakeHttpResponse(["not", "an", "object"]), "health response must be a JSON object"),
    ],
)
def test_health_readiness_failures_raise_preflight_error(health_response, message):
    opener = FakeOpener([health_response])
    client = CdmLatexRenderClient(service_url="http://cdm.test", opener=opener)

    with pytest.raises(CdmPreflightError, match=message):
        client.check_health()


def test_preflight_runs_health_and_tiny_scoring_probe():
    opener = FakeOpener([
        FakeHttpResponse({"version": "cdm_katex_v1", "browser_ready": True}),
        FakeHttpResponse({"score": 1.0, "diagnostics": {"render_status": "ok"}}),
    ])
    client = CdmLatexRenderClient(service_url="http://cdm.test", opener=opener)

    client.preflight()

    assert [request.full_url for request in opener.requests] == [
        "http://cdm.test/health",
        "http://cdm.test/score",
    ]
    assert request_json(opener.requests[1]) == {"prediction": "x", "reference": "x"}


@pytest.mark.parametrize(
    ("score_response", "message"),
    [
        (URLError(socket.timeout("timed out")), "timed out"),
        (FakeHttpResponse({"diagnostics": {"render_status": "ok"}}), "missing score"),
        (FakeHttpResponse({"score": 1.5}), "out of range"),
    ],
)
def test_preflight_rejects_unscorable_probe_even_with_positive_fail_score(score_response, message):
    opener = FakeOpener([
        FakeHttpResponse({"version": "cdm_katex_v1", "browser_ready": True}),
        score_response,
    ])
    client = CdmLatexRenderClient(service_url="http://cdm.test", fail_score=0.5, opener=opener)

    with pytest.raises(CdmPreflightError, match=message):
        client.preflight()


def test_score_success_returns_runtime_reward_result_and_posts_latex_pair():
    opener = FakeOpener([
        FakeHttpResponse({"score": 0.875, "diagnostics": {"render_status": "ok"}})
    ])
    client = CdmLatexRenderClient(
        service_url="http://cdm.test/api",
        timeout_ms=1000,
        fail_score=0.0,
        opener=opener,
    )

    result = client.score(
        prediction=r"\\frac{1}{2}",
        ground_truth=r"\\frac{1}{3}",
        reward_version="profile-v1",
    )

    assert result == {
        "reward_total": 0.875,
        "reward_name": "cdm_latex_render",
        "reward_version": "profile-v1",
        "diagnostics": {"render_status": "ok"},
    }
    assert opener.requests[0].full_url == "http://cdm.test/api/score"
    assert request_json(opener.requests[0]) == {
        "prediction": r"\\frac{1}{2}",
        "reference": r"\\frac{1}{3}",
    }
    assert opener.timeouts == [1.0]


def test_structured_render_error_returns_zero_reward_with_compact_diagnostics():
    opener = FakeOpener([
        FakeHttpResponse({
            "score": 0.0,
            "diagnostics": {
                "render_status": "error",
                "parse_status": "failed",
                "message": "KaTeX parse error: Expected '}', got EOF with a very long traceback that should not be copied wholesale",
            },
        })
    ])
    client = CdmLatexRenderClient(service_url="http://cdm.test", opener=opener)

    result = client.score(prediction=r"\\frac{1}{", ground_truth="x", reward_version="cdm_katex_v1")

    assert result["reward_total"] == 0.0
    assert result["diagnostics"]["render_status"] == "error"
    assert len(result["diagnostics"]["message"]) <= 120


def test_http_error_returns_zero_reward_even_with_positive_fail_score():
    opener = FakeOpener([
        HTTPError("http://cdm.test/score", 503, "unavailable", hdrs=Message(), fp=None)
    ])
    client = CdmLatexRenderClient(service_url="http://cdm.test", fail_score=0.5, opener=opener)

    result = client.score(prediction="x", ground_truth="x", reward_version="cdm_katex_v1")

    assert result["reward_total"] == 0.0
    assert result["diagnostics"] == {"error_type": "http_error", "status": 503, "message": "unavailable"}


def test_timeout_returns_fail_score_diagnostic_result():
    opener = FakeOpener([URLError(socket.timeout("timed out"))])
    client = CdmLatexRenderClient(service_url="http://cdm.test", fail_score=0.0, opener=opener)

    result = client.score(prediction="x", ground_truth="x", reward_version="cdm_katex_v1")

    assert result["reward_total"] == 0.0
    assert result["diagnostics"]["error_type"] == "timeout"


def test_malformed_json_returns_fail_score_diagnostic_result():
    opener = FakeOpener([FakeHttpResponse(b"not json")])
    client = CdmLatexRenderClient(service_url="http://cdm.test", fail_score=0.0, opener=opener)

    result = client.score(prediction="x", ground_truth="x", reward_version="cdm_katex_v1")

    assert result["reward_total"] == 0.0
    assert result["diagnostics"]["error_type"] == "malformed_json"


def test_missing_score_returns_fail_score_diagnostic_result():
    opener = FakeOpener([FakeHttpResponse({"diagnostics": {"render_status": "ok"}})])
    client = CdmLatexRenderClient(service_url="http://cdm.test", fail_score=0.0, opener=opener)

    result = client.score(prediction="x", ground_truth="x", reward_version="cdm_katex_v1")

    assert result["reward_total"] == 0.0
    assert result["diagnostics"]["error_type"] == "malformed_score"
    assert "missing score" in result["diagnostics"]["message"]


def test_out_of_range_score_returns_fail_score_diagnostic_result():
    opener = FakeOpener([FakeHttpResponse({"score": 1.5})])
    client = CdmLatexRenderClient(service_url="http://cdm.test", fail_score=0.0, opener=opener)

    result = client.score(prediction="x", ground_truth="x", reward_version="cdm_katex_v1")

    assert result["reward_total"] == 0.0
    assert result["diagnostics"]["error_type"] == "malformed_score"
    assert "out of range" in result["diagnostics"]["message"]


def test_compute_score_routes_cdm_profile_to_http_client():
    opener = FakeOpener([FakeHttpResponse({"score": 0.5, "diagnostics": {"render_status": "ok"}})])

    result = compute_score(
        data_source="ocr_vlm:region:formula",
        solution_str="x+1",
        ground_truth="x+2",
        extra_info={"task_type": "formula"},
        routing={"default": "cdm"},
        rewards={
            "cdm": {
                "type": "cdm_latex_render",
                "service_url": "http://cdm.test",
                "timeout_ms": 1500,
                "fail_score": 0.0,
                "version": "cdm_katex_v1",
                "expected_version": "cdm_katex_v1",
                "opener": opener,
            }
        },
    )

    assert result["score"] == result["reward_total"] == 0.5
    assert result["reward_name"] == "cdm_latex_render"
    assert result["reward_version"] == "cdm_katex_v1"
    assert result["reward_profile_id"] == "cdm"
    assert request_json(opener.requests[0]) == {"prediction": "x+1", "reference": "x+2"}
