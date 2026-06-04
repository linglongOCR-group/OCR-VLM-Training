from __future__ import annotations

import hydra
import logging
import time
from pathlib import Path

import torch
import torch.distributed
from omegaconf import OmegaConf
from tensordict.tensorclass import NonTensorData
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.tracking import Tracking

from verl_plugins.trainers._utils import _as_bool
from verl_plugins.trainers.sft_freeze import extract_freeze_vision_tower, install_freeze_vision_tower_hook
from verl_plugins.trainers.sft_position_ids import install_qwen_vl_position_ids_chunk_patch


logger = logging.getLogger(__file__)


class OcrSFTTrainer(SFTTrainer):
    def __init__(self, config, *, freeze_vision_tower: bool) -> None:
        self._freeze_vision_tower = freeze_vision_tower
        super().__init__(config=config)

    def _build_engine(self):
        super()._build_engine()
        install_freeze_vision_tower_hook(self.engine, enabled=self._freeze_vision_tower)

    def fit(self):
        if self._trainer_bool("validate_only"):
            self._ensure_validation_available()
            tracking = self._build_tracking() if self._is_logging_rank() else None
            metric = self._run_validation_once(
                self._build_validation_meta_info(),
                tracking=tracking,
                step=self.resume_global_step,
            )
            if self._is_logging_rank():
                print(f"Validate-only metrics: {metric}")
            return

        self._fit_with_optional_pre_validation(run_pre_validation=self._trainer_bool("val_before_train"))

    def _fit_with_optional_pre_validation(self, *, run_pre_validation: bool) -> None:
        if run_pre_validation:
            self._ensure_validation_available()

        is_logging = self._is_logging_rank()

        if is_logging:
            tracking = self._build_tracking()
        else:
            tracking = None

        global_step = self.resume_global_step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        start_epoch = global_step // self.steps_per_epoch
        meta_info = self._build_validation_meta_info()

        if run_pre_validation:
            last_valid_metric = self._run_validation_once(meta_info, tracking=tracking, step=global_step)

        train_time = 0.0
        _train_start = time.monotonic()
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"rank {self.rank}: At start of epoch {epoch}", logger=logger)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data)
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                if global_step == self.start_profile_step:
                    self.training_client.start_profile()
                output = self.training_client.train_batch(data=data)

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = _normalize_training_metrics(tu.get(output, "metrics"))

                    for key in ["loss", "grad_norm", "lr", "mfu"]:
                        if key in metrics.keys():
                            value = metrics.pop(key)
                            metrics[f"train/{key}"] = value

                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    if self.engine.get_data_parallel_rank() == 0 and tracking is not None:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    last_valid_metric = self._run_validation_once(meta_info, tracking=tracking, step=global_step)

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    aggressive_empty_cache(force_sync=True)
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        train_time = time.monotonic() - _train_start
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return

    def _run_validation_once(self, meta_info: dict, *, tracking: Tracking | None, step: int) -> dict[str, float] | None:
        self._ensure_validation_available()
        val_losses = []
        for val_data in self.val_dataloader:
            val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
            output = self.training_client.infer_batch(val_data)

            if self.engine.is_mp_src_rank_with_outputs():
                metrics = _metrics_from_output(output)
                val_losses.append(_as_loss_tensor(metrics["loss"], self.device_name))

        metric = None
        if self.engine.is_mp_src_rank_with_outputs():
            if not val_losses:
                raise ValueError("SFT validation requires data.val_files with at least one complete validation batch")
            val_loss = torch.stack(val_losses).mean()
            dp_group = self.engine.get_data_parallel_group()
            if dp_group is not None:
                torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
            metric = {"val/loss": val_loss.detach().item()}

        if self._is_logging_rank() and tracking is not None and metric is not None:
            tracking.log(data=metric, step=step)
        _distributed_barrier_if_initialized()
        return metric

    def _build_validation_meta_info(self) -> dict:
        return {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

    def _build_tracking(self) -> Tracking:
        return Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def _ensure_validation_available(self) -> None:
        if self.val_dataloader is None:
            raise ValueError("SFT validation requires data.val_files and a non-empty validation dataloader")

    def _is_logging_rank(self) -> bool:
        return self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

    def _trainer_bool(self, name: str) -> bool:
        return _as_bool(self.config.trainer.get(name, False))


def _as_loss_tensor(value, device_name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device_name)
    return torch.tensor(value, device=device_name)


def _metrics_from_output(output) -> dict:
    if isinstance(output, dict):
        return output["metrics"]
    return tu.get(output, "metrics")


_KD_LOSS_METRIC_KEYS = {
    "train/loss_sft_raw",
    "train/loss_logit_raw",
    "train/loss_hidden_raw",
    "train/loss_sft_weighted",
    "train/loss_logit_weighted",
    "train/loss_hidden_weighted",
}

_KD_CONSTANT_METRIC_KEYS = {
    "train/lambda_sft",
    "train/lambda_logit",
    "train/lambda_hidden",
    "train/kd_temperature",
    "train/kd_logits_top_k",
    "train/kd_logits_loss_type",
    "train/kd_hidden_layers",
}


def _normalize_training_metrics(metrics: dict) -> dict:
    return {key: _normalize_training_metric_value(key, value) for key, value in metrics.items()}


def _normalize_training_metric_value(key: str, value):
    if not isinstance(value, (list, tuple)):
        return _scalar_metric_value(value)

    if key in _KD_CONSTANT_METRIC_KEYS:
        flattened = _flatten_metric_values(value)
        return _scalar_metric_value(flattened[0]) if flattened else None

    if key in _KD_LOSS_METRIC_KEYS:
        return _reduce_microbatch_loss_metric(value)

    flattened = [_scalar_metric_value(item) for item in _flatten_metric_values(value)]
    numeric_values = [item for item in flattened if _is_numeric_metric_value(item)]
    if len(numeric_values) == len(flattened) and numeric_values:
        return sum(float(item) for item in numeric_values) / len(numeric_values)
    if flattened and all(item == flattened[0] for item in flattened):
        return flattened[0]
    return value


def _reduce_microbatch_loss_metric(value) -> float:
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, (list, tuple)) for item in value):
        rank_sums = [_sum_numeric_metric_values(rank_values) for rank_values in value]
        return sum(rank_sums) / len(rank_sums)
    return _sum_numeric_metric_values(value)


def _sum_numeric_metric_values(value) -> float:
    flattened = [_scalar_metric_value(item) for item in _flatten_metric_values(value)]
    return sum(float(item) for item in flattened if _is_numeric_metric_value(item))


def _flatten_metric_values(value) -> list:
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_metric_values(item))
        return flattened
    return [value]


def _scalar_metric_value(value):
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() == 1:
            return detached.item()
        return detached.float().mean().item()
    return value


def _is_numeric_metric_value(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _distributed_barrier_if_initialized() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def run_sft(config) -> None:
    freeze_vision_tower = extract_freeze_vision_tower(config)
    install_qwen_vl_position_ids_chunk_patch()

    initialize_global_process_group()
    trainer = OcrSFTTrainer(config=config, freeze_vision_tower=freeze_vision_tower)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path=str(Path(__import__("verl").__file__).parent / "trainer" / "config"), config_name="sft_trainer_engine", version_base=None)
def main(config) -> None:
    auto_set_device(config)
    run_sft(config)


if __name__ == "__main__":
    main()
