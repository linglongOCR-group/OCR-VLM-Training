from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evaluate_schedule(config: Mapping[str, Any], *, global_step: int) -> float:
    if global_step < 0:
        raise ValueError("global_step must be non-negative")

    schedule_type = str(config.get("type", "")).strip()
    if schedule_type == "constant":
        return float(config["value"])
    if schedule_type == "piecewise_linear":
        return _piecewise_linear(config, global_step=global_step)
    if schedule_type == "linear_warmup_constant":
        return _linear_warmup_constant(config, global_step=global_step)
    if schedule_type == "linear_warmup_hold_decay":
        return _linear_warmup_hold_decay(config, global_step=global_step)
    raise ValueError(f"Unsupported KD schedule type: {schedule_type!r}")


def validate_schedule(config: Mapping[str, Any]) -> None:
    evaluate_schedule(config, global_step=0)


def _linear_interpolate(start: float, end: float, *, index: int, steps: int) -> float:
    if steps <= 0:
        return end
    ratio = min(max(index / steps, 0.0), 1.0)
    return start + (end - start) * ratio


def _linear_warmup_constant(config: Mapping[str, Any], *, global_step: int) -> float:
    start = float(config.get("start", 0.0))
    peak = float(config["peak"])
    warmup_steps = int(config.get("warmup_steps", 0))
    if global_step < warmup_steps:
        return _linear_interpolate(start, peak, index=global_step, steps=warmup_steps)
    return peak


def _linear_warmup_hold_decay(config: Mapping[str, Any], *, global_step: int) -> float:
    start = float(config.get("start", 0.0))
    peak = float(config["peak"])
    end = float(config.get("end", 0.0))
    warmup_steps = int(config.get("warmup_steps", 0))
    hold_steps = int(config.get("hold_steps", 0))
    decay_steps = int(config.get("decay_steps", 0))

    if global_step < warmup_steps:
        return _linear_interpolate(start, peak, index=global_step, steps=warmup_steps)

    decay_start = warmup_steps + hold_steps
    if global_step <= decay_start:
        return peak
    return _linear_interpolate(
        peak,
        end,
        index=global_step - decay_start,
        steps=decay_steps,
    )


def _piecewise_linear(config: Mapping[str, Any], *, global_step: int) -> float:
    points = [(int(point["step"]), float(point["value"])) for point in config["points"]]
    if not points:
        raise ValueError("piecewise_linear schedule requires at least one point")

    points.sort()
    if global_step <= points[0][0]:
        return points[0][1]

    for (left_step, left_value), (right_step, right_value) in zip(points, points[1:], strict=False):
        if global_step <= right_step:
            return _linear_interpolate(
                left_value,
                right_value,
                index=global_step - left_step,
                steps=right_step - left_step,
            )

    return points[-1][1]
