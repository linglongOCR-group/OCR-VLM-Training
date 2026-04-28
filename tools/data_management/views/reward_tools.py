from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tools.data_management.registry.reward_registry import default_reward_registry
from tools.data_management.utils.io import write_parquet


def reward_smoke_test(view_root: str | Path, *, limit: int = 100) -> dict[str, Any]:
    rows = _read_view_rows(view_root, limit=limit)
    registry = default_reward_registry()
    scores = []
    for row in rows:
        if row.get("stage") != "rlvr":
            continue
        reward = registry.get(row["reward_profile_id"])
        payload = _payload(row)
        result = reward.score(str(row.get("answer_key") or row["label"]), payload, {"task": row["task"], "view_record_id": row["id"]})
        scores.append(result.normalized_score)
    if not scores:
        raise ValueError("no rlvr rows available for reward smoke test")
    return {"count": len(scores), "min_score": min(scores), "mean_score": sum(scores) / len(scores), "max_score": max(scores)}


def score_predictions(view_root: str | Path, predictions_path: str | Path, output_path: str | Path) -> int:
    view_rows = {row["id"]: row for row in _read_view_rows(view_root, limit=None)}
    predictions = pd.read_parquet(predictions_path).to_dict(orient="records")
    registry = default_reward_registry()
    output = []
    for prediction in predictions:
        view_id = prediction["id"]
        row = view_rows[view_id]
        reward = registry.get(row["reward_profile_id"])
        result = reward.score(str(prediction.get("prediction") or ""), _payload(row), {"task": row["task"], "view_record_id": view_id})
        data = result.to_dict()
        data["id"] = view_id
        data["reward_profile_id"] = row["reward_profile_id"]
        output.append(data)
    return write_parquet(output_path, output)


def _read_view_rows(view_root: str | Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(view_root).glob("*.parquet")):
        frame = pd.read_parquet(path)
        rows.extend(frame.to_dict(orient="records"))
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("reward_payload")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise ValueError(f"view row {row.get('id')} has no inline reward payload")
