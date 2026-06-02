from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.logger import log_with_rank
from verl.workers.engine_workers import TrainingWorkerConfig

from verl_plugins.trainers.kd_precheck import ModelContract, parse_kd_config, validate_kd_config, validate_teacher_frozen
from verl_plugins.trainers.kd_worker import KDTrainingWorker, clone_teacher_engine_config
from verl_plugins.trainers.sft_freeze import extract_freeze_vision_tower, install_freeze_vision_tower_hook
from verl_plugins.trainers.sft_position_ids import install_qwen_vl_position_ids_chunk_patch
from verl_plugins.trainers.sft_trainer import OcrSFTTrainer

logger = logging.getLogger(__file__)


class OcrKDSFTTrainer(OcrSFTTrainer):
    def _build_config(self):
        super()._build_config()
        parsed = parse_kd_config(OmegaConf.to_container(self.config, resolve=True))
        teacher_model_config = OmegaConf.create(OmegaConf.to_container(self.config.model, resolve=True))
        teacher_model_config.path = parsed.teacher_path
        teacher_model_config.hf_config_path = parsed.teacher_path
        teacher_model_config.tokenizer_path = parsed.teacher_path
        self.teacher_model_config = omega_conf_to_dataclass(teacher_model_config)
        self.kd_config = validate_kd_config(
            OmegaConf.to_container(self.config, resolve=True),
            student=_model_contract_from_config(self.model_config),
            teacher=_model_contract_from_config(self.teacher_model_config),
            student_tokenizer=self.model_config.tokenizer,
            teacher_tokenizer=self.teacher_model_config.tokenizer,
            student_processor=self.model_config.processor,
            teacher_processor=self.teacher_model_config.processor,
        )

    def _build_engine(self):
        import verl_plugins.trainers.kd_engine  # noqa: F401

        student_worker_config = TrainingWorkerConfig(
            model_type="kd_language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )
        teacher_engine_config = clone_teacher_engine_config(self.engine_config)
        self.training_client = KDTrainingWorker(
            config=student_worker_config,
            teacher_model_config=self.teacher_model_config,
            teacher_engine_config=teacher_engine_config,
            kd_config=self.kd_config,
        )
        self.engine = self.training_client.engine
        install_freeze_vision_tower_hook(self.engine, enabled=self._freeze_vision_tower)

    def _init_engine(self):
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch
        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.training_client.reset()
        validate_teacher_frozen(self.training_client.teacher_engine.module)
        log_with_rank(
            "KD-SFT config precheck passed and teacher is frozen.",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )


def _model_contract_from_config(model_config) -> ModelContract:
    hf_config = model_config.hf_config
    tokenizer = model_config.tokenizer
    processor = model_config.processor
    processor_inputs = tuple(getattr(processor, "model_input_names", ()) or ())
    tokenizer_name = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "tokenizer_path", None)
    return ModelContract(
        vocab_size=int(getattr(hf_config, "vocab_size")),
        hidden_size=int(getattr(hf_config, "hidden_size")),
        num_hidden_layers=int(getattr(hf_config, "num_hidden_layers")),
        model_path=str(model_config.path),
        tokenizer_path=str(model_config.tokenizer_path),
        tokenizer_name=str(tokenizer_name) if tokenizer_name is not None else None,
        processor_type=type(processor).__name__ if processor is not None else None,
        input_contract=processor_inputs,
        output_token_semantics=str(getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or ""),
    )


def run_kd_sft(config) -> None:
    freeze_vision_tower = extract_freeze_vision_tower(config)
    install_qwen_vl_position_ids_chunk_patch()

    initialize_global_process_group()
    trainer = OcrKDSFTTrainer(config=config, freeze_vision_tower=freeze_vision_tower)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(
    config_path=str(Path(__import__("verl").__file__).parent / "trainer" / "config"),
    config_name="sft_trainer_engine",
    version_base=None,
)
def main(config) -> None:
    auto_set_device(config)
    run_kd_sft(config)


if __name__ == "__main__":
    main()
