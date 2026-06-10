from __future__ import annotations

import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, build_opener


REWARD_NAME = "cdm_latex_render"
DEFAULT_CDM_REWARD_VERSION = "cdm_katex_v1"


class CdmPreflightError(RuntimeError):
    """Raised when the CDM service is not ready for launch-time use."""


class CdmLatexRenderClient:
    def __init__(
        self,
        *,
        service_url: str,
        timeout_ms: int | float = 1000,
        fail_score: float = 0.0,
        expected_version: str | None = None,
        opener: Any | None = None,
    ) -> None:
        if not service_url:
            raise ValueError("service_url is required")
        self.service_url = service_url.rstrip("/") + "/"
        self.timeout_seconds = float(timeout_ms) / 1000.0
        self.fail_score = _coerce_score(fail_score, field_name="fail_score")
        self.expected_version = expected_version
        self.opener = opener if opener is not None else build_opener()

    def check_health(self) -> dict[str, Any]:
        try:
            payload = self._request_json("GET", "health")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            raise CdmPreflightError(f"CDM health check failed: {_preflight_error_message(exc)}") from exc
        if not isinstance(payload, dict):
            raise CdmPreflightError("health response must be a JSON object")

        version = payload.get("version")
        if self.expected_version and version != self.expected_version:
            raise CdmPreflightError(f"CDM service version mismatch: expected {self.expected_version}, got {version}")
        for flag in ("ok", "healthy"):
            if payload.get(flag) is False:
                raise CdmPreflightError(f"CDM service reported {flag}=false")
        if payload.get("browser_ready") is not True:
            raise CdmPreflightError("CDM service browser not ready")

        return {
            key: payload[key]
            for key in ("service", "version", "renderer", "browser_ready")
            if key in payload
        }

    def preflight(self) -> None:
        self.check_health()
        try:
            payload = self._request_json("POST", "score", {"prediction": "x", "reference": "x"})
            reward_total, _diagnostics = self._normalize_score_payload(payload)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            raise CdmPreflightError(f"CDM scoring probe failed: {_preflight_error_message(exc)}") from exc
        if reward_total <= 0.0:
            raise CdmPreflightError("CDM scoring probe failed: score must be positive for x/x")

    def score(self, *, prediction: str, ground_truth: str, reward_version: str = DEFAULT_CDM_REWARD_VERSION) -> dict[str, Any]:
        try:
            payload = self._request_json("POST", "score", {"prediction": prediction or "", "reference": ground_truth or ""})
            reward_total, diagnostics = self._normalize_score_payload(payload)
            return _reward_result(reward_total=reward_total, reward_version=reward_version, diagnostics=diagnostics)
        except HTTPError as exc:
            return self._failure_result(
                reward_version=reward_version,
                diagnostics={"error_type": "http_error", "status": exc.code, "message": _truncate(exc.reason)},
            )
        except URLError as exc:
            return self._failure_result(reward_version=reward_version, diagnostics=_url_error_diagnostics(exc))
        except TimeoutError as exc:
            return self._failure_result(
                reward_version=reward_version,
                diagnostics={"error_type": "timeout", "message": _truncate(str(exc) or "timed out")},
            )
        except json.JSONDecodeError as exc:
            return self._failure_result(
                reward_version=reward_version,
                diagnostics={"error_type": "malformed_json", "message": _truncate(str(exc))},
            )
        except ValueError as exc:
            return self._failure_result(
                reward_version=reward_version,
                diagnostics={"error_type": "malformed_score", "message": _truncate(str(exc))},
            )

    def __call__(
        self,
        *,
        prediction: str,
        ground_truth: str,
        reward_version: str = DEFAULT_CDM_REWARD_VERSION,
        **_: Any,
    ) -> dict[str, Any]:
        return self.score(prediction=prediction, ground_truth=ground_truth, reward_version=reward_version)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(urljoin(self.service_url, path), data=data, headers=headers, method=method)
        with self.opener.open(request, timeout=self.timeout_seconds) as response:
            body = response.read()
        return json.loads(body.decode("utf-8"))

    def _normalize_score_payload(self, payload: Any) -> tuple[float, dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError("score response must be a JSON object")
        if "score" not in payload:
            raise ValueError("missing score")
        score = _coerce_score(payload["score"], field_name="score")
        diagnostics = _compact_diagnostics(payload.get("diagnostics") or {})
        return score, diagnostics

    def _failure_result(self, *, reward_version: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
        return _reward_result(
            reward_total=0.0,
            reward_version=reward_version,
            diagnostics=_compact_diagnostics(diagnostics),
        )


def _reward_result(*, reward_total: float, reward_version: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "reward_total": reward_total,
        "reward_name": REWARD_NAME,
        "reward_version": reward_version,
        "diagnostics": diagnostics,
    }


def _coerce_score(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{field_name} out of range [0, 1]")
    return score


def _compact_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"message": _truncate(str(value))}
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            compact[str(key)] = _truncate(item) if isinstance(item, str) else item
        elif isinstance(item, list):
            compact[str(key)] = [_truncate(str(entry)) for entry in item[:5]]
        else:
            compact[str(key)] = _truncate(str(item))
    return compact


def _compact_message(diagnostics: dict[str, Any]) -> str:
    message = diagnostics.get("message") or diagnostics.get("error_type") or diagnostics
    return _truncate(str(message))


def _preflight_error_message(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return _truncate(str(exc.reason))
    if isinstance(exc, URLError):
        diagnostics = _url_error_diagnostics(exc)
        return _compact_message(diagnostics)
    return _truncate(str(exc))


def _truncate(value: str, limit: int = 120) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _url_error_diagnostics(exc: URLError) -> dict[str, Any]:
    reason = exc.reason
    if isinstance(reason, socket.timeout):
        return {"error_type": "timeout", "message": _truncate(str(reason) or "timed out")}
    return {"error_type": "connection_error", "message": _truncate(str(reason))}
