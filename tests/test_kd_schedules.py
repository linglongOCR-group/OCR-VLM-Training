import pytest

from verl_plugins.trainers.kd_schedules import evaluate_schedule, validate_schedule


def test_constant_schedule_returns_value_for_any_global_step():
    cfg = {"type": "constant", "value": 1.0}

    assert evaluate_schedule(cfg, global_step=0) == pytest.approx(1.0)
    assert evaluate_schedule(cfg, global_step=25) == pytest.approx(1.0)


def test_piecewise_linear_schedule_interpolates_and_clamps():
    cfg = {
        "type": "piecewise_linear",
        "points": [
            {"step": 10, "value": 0.5},
            {"step": 0, "value": 0.0},
            {"step": 20, "value": 1.0},
        ],
    }

    assert evaluate_schedule(cfg, global_step=0) == pytest.approx(0.0)
    assert evaluate_schedule(cfg, global_step=5) == pytest.approx(0.25)
    assert evaluate_schedule(cfg, global_step=10) == pytest.approx(0.5)
    assert evaluate_schedule(cfg, global_step=15) == pytest.approx(0.75)
    assert evaluate_schedule(cfg, global_step=30) == pytest.approx(1.0)


def test_linear_warmup_constant_schedule_reaches_peak():
    cfg = {
        "type": "linear_warmup_constant",
        "start": 0.0,
        "peak": 0.3,
        "warmup_steps": 6,
    }

    assert evaluate_schedule(cfg, global_step=0) == pytest.approx(0.0)
    assert evaluate_schedule(cfg, global_step=3) == pytest.approx(0.15)
    assert evaluate_schedule(cfg, global_step=6) == pytest.approx(0.3)
    assert evaluate_schedule(cfg, global_step=20) == pytest.approx(0.3)


def test_linear_warmup_hold_decay_schedule_uses_global_step():
    cfg = {
        "type": "linear_warmup_hold_decay",
        "start": 0.0,
        "peak": 0.2,
        "warmup_steps": 10,
        "hold_steps": 20,
        "decay_steps": 10,
        "end": 0.0,
    }

    assert evaluate_schedule(cfg, global_step=0) == pytest.approx(0.0)
    assert evaluate_schedule(cfg, global_step=5) == pytest.approx(0.1)
    assert evaluate_schedule(cfg, global_step=10) == pytest.approx(0.2)
    assert evaluate_schedule(cfg, global_step=30) == pytest.approx(0.2)
    assert evaluate_schedule(cfg, global_step=35) == pytest.approx(0.1)
    assert evaluate_schedule(cfg, global_step=40) == pytest.approx(0.0)


def test_schedule_rejects_negative_global_step_and_unknown_type():
    with pytest.raises(ValueError, match="global_step"):
        evaluate_schedule({"type": "constant", "value": 1.0}, global_step=-1)

    with pytest.raises(ValueError, match="Unsupported KD schedule type"):
        validate_schedule({"type": "cosine"})
