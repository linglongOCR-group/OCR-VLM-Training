from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_plugins.trainers import sft_trainer
from verl_plugins.trainers.sft_trainer import OcrSFTTrainer


class FakeEngine:
    def __init__(self):
        self.dp_rank = 0

    def is_mp_src_rank_with_outputs(self):
        return True

    def get_data_parallel_rank(self):
        return self.dp_rank

    def get_data_parallel_group(self):
        return None


class FakeTrainingClient:
    def __init__(self, losses):
        self.losses = list(losses)
        self.infer_batches = []

    def infer_batch(self, data):
        self.infer_batches.append(data)
        loss = self.losses.pop(0)
        return {"metrics": {"loss": torch.tensor(loss)}}


class FakeTracking:
    def __init__(self):
        self.logged = []

    def log(self, data, step):
        self.logged.append((data, step))


def _fake_trainer(*, validate_only=False, val_before_train=False, val_batches=2):
    trainer = OcrSFTTrainer.__new__(OcrSFTTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "validate_only": validate_only,
                "val_before_train": val_before_train,
                "project_name": "test",
                "experiment_name": "test",
                "logger": ["console"],
            },
            "model": {
                "use_remove_padding": True,
            },
            "data": {
                "use_dynamic_bsz": False,
                "max_token_len_per_gpu": 1024,
                "micro_batch_size_per_gpu": 1,
                "pad_mode": "no_padding",
            },
        }
    )
    trainer.engine = FakeEngine()
    trainer.training_client = FakeTrainingClient([1.0, 3.0])
    trainer.device_name = "cpu"
    trainer.global_batch_size = 2
    trainer.resume_global_step = 0
    trainer.model_config = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    trainer.val_dataloader = [{"batch": index} for index in range(val_batches)]
    return trainer


def test_run_validation_once_averages_loss_and_logs(monkeypatch):
    trainer = _fake_trainer()
    tracking = FakeTracking()
    monkeypatch.setattr(sft_trainer.tu, "get_tensordict", lambda tensor_dict, non_tensor_dict: tensor_dict)

    metric = trainer._run_validation_once(
        trainer._build_validation_meta_info(),
        tracking=tracking,
        step=7,
    )

    assert metric == {"val/loss": 2.0}
    assert tracking.logged == [({"val/loss": 2.0}, 7)]
    assert trainer.training_client.infer_batches == [{"batch": 0}, {"batch": 1}]


def test_validate_only_runs_validation_without_training_loop(monkeypatch):
    trainer = _fake_trainer(validate_only=True)
    tracking = FakeTracking()
    calls = []

    monkeypatch.setattr(trainer, "_build_tracking", lambda: tracking)
    monkeypatch.setattr(
        trainer,
        "_run_validation_once",
        lambda meta_info, tracking, step: calls.append((meta_info, tracking, step)) or {"val/loss": 1.0},
    )
    monkeypatch.setattr(
        trainer,
        "_fit_with_optional_pre_validation",
        lambda run_pre_validation: pytest.fail("validate_only should not enter training"),
    )

    trainer.fit()

    assert calls == [(trainer._build_validation_meta_info(), tracking, 0)]


def test_val_before_train_enters_training_loop_with_pre_validation_enabled(monkeypatch):
    trainer = _fake_trainer(val_before_train=True)
    calls = []

    monkeypatch.setattr(trainer, "_fit_with_optional_pre_validation", lambda run_pre_validation: calls.append(run_pre_validation))

    trainer.fit()

    assert calls == [True]


def test_validation_flags_require_validation_dataloader():
    trainer = _fake_trainer(validate_only=True, val_batches=0)
    trainer.val_dataloader = None

    with pytest.raises(ValueError, match="requires data.val_files"):
        trainer.fit()
