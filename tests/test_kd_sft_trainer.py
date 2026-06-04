from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl_plugins.trainers.kd_sft_trainer import OcrKDSFTTrainer


class _FakeTrainingClient:
    def __init__(self):
        self.reset_calls = 0
        self.dry_run_batches = []
        self.teacher_engine = SimpleNamespace(
            module=SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=False)])
        )

    def reset(self):
        self.reset_calls += 1

    def dry_run_batch(self, data):
        self.dry_run_batches.append(data)


def _trainer(precheck_mode: str = "config"):
    trainer = OcrKDSFTTrainer.__new__(OcrKDSFTTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "total_training_steps": None,
                "total_epochs": 1,
                "save_freq": 10,
                "test_freq": 10,
            },
            "model": {"use_remove_padding": False},
            "data": {
                "use_dynamic_bsz": False,
                "max_token_len_per_gpu": 128,
                "micro_batch_size_per_gpu": 1,
                "pad_mode": "no_padding",
            },
        }
    )
    trainer.kd_config = SimpleNamespace(precheck={"mode": precheck_mode})
    trainer.train_dataloader = [{"input_ids": torch.tensor([[1, 2]]), "loss_mask": torch.tensor([[0, 1]])}]
    trainer.optimizer_config = SimpleNamespace(total_training_steps=None)
    trainer.model_config = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    trainer.global_batch_size = 1
    trainer.training_client = _FakeTrainingClient()
    trainer._get_batch_seqlens = lambda data: [2]
    return trainer


def test_kd_sft_init_engine_runs_dry_run_batch_precheck_when_selected():
    trainer = _trainer(precheck_mode="dry_run_batch")

    trainer._init_engine()

    assert trainer.training_client.reset_calls == 1
    assert len(trainer.training_client.dry_run_batches) == 1


def test_kd_sft_init_engine_does_not_run_dry_run_batch_for_config_precheck():
    trainer = _trainer(precheck_mode="config")

    trainer._init_engine()

    assert trainer.training_client.reset_calls == 1
    assert trainer.training_client.dry_run_batches == []


def test_kd_sft_dry_run_batch_precheck_uses_first_training_batch_with_sft_meta():
    trainer = _trainer(precheck_mode="dry_run_batch")

    trainer._run_dry_run_batch_precheck()

    data = trainer.training_client.dry_run_batches[0]
    assert torch.equal(data["input_ids"], torch.tensor([[1, 2]]))
    assert data["use_remove_padding"] is False
    assert data["global_batch_size"] == 1
    assert data["pad_token_id"] == 0
    assert data["global_token_num"] == [2]
