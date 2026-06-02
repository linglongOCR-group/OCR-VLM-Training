from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from codetiming import Timer
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine import EngineRegistry
from verl.workers.engine_workers import TrainingWorker
from verl.workers.config import TrainingWorkerConfig

from verl_plugins.trainers.kd_hidden_hooks import (
    HiddenCaptureStore,
    HiddenLayerMapEntry,
    register_mapped_hidden_hooks,
)
from verl_plugins.trainers.kd_losses import (
    align_response_mask_for_shifted_logits,
    hidden_kd_loss,
    renormalized_top_k_forward_kl,
    truncated_forward_kl,
)
from verl_plugins.trainers.kd_precheck import extract_top_k_teacher_targets
from verl_plugins.trainers.kd_schedules import evaluate_schedule

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class KDStepConfig:
    global_step: int
    top_k: int
    temperature: float
    logits_loss_type: str
    hidden_loss_type: str
    layer_map: Sequence[Mapping[str, Any] | HiddenLayerMapEntry]
    schedules: Mapping[str, Mapping[str, Any]]
    logits_enabled: bool = True
    hidden_enabled: bool = True


@dataclass(frozen=True)
class KDLossResult:
    loss: torch.Tensor
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class KDStepContext:
    rank: int


def extract_response_mask(data: Mapping[str, Any], *, target_seq_len: int) -> torch.Tensor:
    if "loss_mask" in data:
        loss_mask = data["loss_mask"]
        if getattr(loss_mask, "is_nested", False):
            return torch.roll(loss_mask.values(), shifts=-1, dims=0).bool()
        return align_response_mask_for_shifted_logits(loss_mask, target_seq_len=target_seq_len)
    if "response_mask" in data:
        return data["response_mask"].bool()
    raise ValueError("KD-SFT requires loss_mask or response_mask in each microbatch")


def compose_kd_sft_loss(
    *,
    student_output: Mapping[str, torch.Tensor],
    teacher_output: Mapping[str, torch.Tensor],
    student_hidden_captures: Mapping[int, torch.Tensor],
    teacher_hidden_captures: Mapping[int, torch.Tensor],
    data: Mapping[str, Any],
    config: KDStepConfig,
    context: KDStepContext | Any | None = None,
) -> KDLossResult:
    student_log_probs = _as_dense_or_flat(student_output["log_probs"])
    response_mask = extract_response_mask(data, target_seq_len=student_log_probs.shape[-1]).to(student_log_probs.device)
    if response_mask.sum().item() == 0:
        rank = getattr(context, "rank", "unknown")
        raise ValueError(f"KD-SFT microbatch has no response-supervised tokens: rank={rank} global_step={config.global_step}")

    lambda_sft = evaluate_schedule(config.schedules["sft"], global_step=config.global_step)
    lambda_logit = (
        evaluate_schedule(config.schedules["logits"], global_step=config.global_step)
        if config.logits_enabled
        else 0.0
    )
    lambda_hidden = (
        evaluate_schedule(config.schedules["hidden"], global_step=config.global_step)
        if config.hidden_enabled
        else 0.0
    )

    loss_sft_raw = _masked_negative_log_prob(student_log_probs, response_mask)

    zero = loss_sft_raw.new_zeros(())
    loss_logit_raw = zero
    if config.logits_enabled and lambda_logit != 0.0:
        student_logits = _as_dense_or_flat(student_output["logits"]) / float(config.temperature)
        teacher_logits = _as_dense_or_flat(teacher_output["logits"]).detach()
        teacher_targets = extract_top_k_teacher_targets(
            teacher_logits,
            top_k=int(config.top_k),
            temperature=float(config.temperature),
            loss_type=config.logits_loss_type,
        )
        if config.logits_loss_type == "renormalized_top_k_forward_kl":
            loss_logit_raw = renormalized_top_k_forward_kl(
                student_logits,
                teacher_targets["log_probs"],
                teacher_targets["indices"],
                response_mask,
            )
        elif config.logits_loss_type == "truncated_forward_kl":
            loss_logit_raw = truncated_forward_kl(
                student_logits,
                teacher_targets["log_probs"],
                teacher_targets["indices"],
                response_mask,
            )
        else:
            raise ValueError(f"Unsupported logits KD loss type: {config.logits_loss_type!r}")

    loss_hidden_raw = zero
    if config.hidden_enabled and lambda_hidden != 0.0:
        student_hidden = {key: _as_dense_or_flat(value) for key, value in student_hidden_captures.items()}
        teacher_hidden = {key: _as_dense_or_flat(value).detach() for key, value in teacher_hidden_captures.items()}
        loss_hidden_raw = hidden_kd_loss(
            student_hidden,
            teacher_hidden,
            config.layer_map,
            response_mask,
            loss_type=config.hidden_loss_type,
        )

    loss_sft_weighted = loss_sft_raw * lambda_sft
    loss_logit_weighted = loss_logit_raw * lambda_logit
    loss_hidden_weighted = loss_hidden_raw * lambda_hidden
    total_loss = loss_sft_weighted + loss_logit_weighted + loss_hidden_weighted

    metrics = {
        "train/loss_sft_raw": _metric(loss_sft_raw),
        "train/loss_logit_raw": _metric(loss_logit_raw),
        "train/loss_hidden_raw": _metric(loss_hidden_raw),
        "train/loss_sft_weighted": _metric(loss_sft_weighted),
        "train/loss_logit_weighted": _metric(loss_logit_weighted),
        "train/loss_hidden_weighted": _metric(loss_hidden_weighted),
        "train/lambda_sft": float(lambda_sft),
        "train/lambda_logit": float(lambda_logit),
        "train/lambda_hidden": float(lambda_hidden),
        "train/kd_temperature": float(config.temperature),
        "train/kd_logits_top_k": int(config.top_k),
        "train/kd_logits_loss_type": config.logits_loss_type,
        "train/kd_hidden_layers": int(len(config.layer_map) if config.hidden_enabled else 0),
    }
    return KDLossResult(loss=total_loss, metrics=metrics)


class KDTrainingWorker(TrainingWorker):
    def __init__(self, config: TrainingWorkerConfig, *, teacher_model_config, teacher_engine_config, kd_config):
        import verl_plugins.trainers.kd_engine  # noqa: F401

        super().__init__(config=config)
        self.kd_config = kd_config
        self.teacher_model_config = teacher_model_config
        self.teacher_engine_config = teacher_engine_config
        self.teacher_engine_config.forward_only = True
        self.teacher_engine_config.optimizer_offload = False
        self.teacher_engine_config.use_remove_padding = self.teacher_model_config.use_remove_padding
        self.teacher_engine_config.use_fused_kernels = self.teacher_model_config.use_fused_kernels
        self.teacher_engine = EngineRegistry.new(
            model_type="kd_language_model",
            backend=self.teacher_engine_config.strategy,
            model_config=self.teacher_model_config,
            engine_config=self.teacher_engine_config,
            optimizer_config=None,
            checkpoint_config=self.checkpoint_config,
        )
        self.student_hidden_store = HiddenCaptureStore()
        self.teacher_hidden_store = HiddenCaptureStore()
        self._student_hook_handles = []
        self._teacher_hook_handles = []

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_batch(self, data: TensorDict) -> TensorDict:
        assert not self.engine_config.forward_only, "Can't run KD-SFT train_batch with a forward-only student engine."
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)
        self._assign_default_batch_meta(data)

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            self.teacher_engine.eval_mode(disable_auto_offload=True),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self._kd_train_batch(data)
        delta_time = timer.last

        lr = self.engine.lr_scheduler_step() if tu.get(data, key="update_lr_scheduler", default=False) else None
        if self.engine.is_mp_src_rank_with_outputs():
            output.pop("model_output", None)
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        return super().infer_batch(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        self.engine.initialize()
        self.teacher_engine.initialize()
        for parameter in self.teacher_engine.module.parameters():
            parameter.requires_grad_(False)
        self._install_hidden_hooks()

    def _kd_train_batch(self, data: TensorDict) -> dict[str, Any]:
        from verl.workers.engine.base import maybe_fix_3d_position_ids
        from verl.workers.engine.utils import prepare_micro_batches, postprocess_batch_func

        maybe_fix_3d_position_ids(data)
        self.engine.optimizer_zero_grad()
        tu.assign_non_tensor(data, sp_size=self.engine.ulysses_sequence_parallel_size)
        batch_num_tokens = data["loss_mask"].sum().to(self.device_name)
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.engine.get_data_parallel_group())
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.engine.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.engine.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )
        output_lst = []
        for micro_batch in micro_batches:
            self.student_hidden_store.clear()
            self.teacher_hidden_store.clear()
            with torch.no_grad():
                _, teacher_meta = self.teacher_engine.forward_step(
                    micro_batch,
                    loss_function=None,
                    forward_only=True,
                )
            teacher_output = teacher_meta["model_output"]

            def kd_loss(model_output, data, dp_group=None):
                result = compose_kd_sft_loss(
                    student_output=model_output,
                    teacher_output=teacher_output,
                    student_hidden_captures=self.student_hidden_store.captures,
                    teacher_hidden_captures=self.teacher_hidden_store.captures,
                    data=data,
                    config=self._step_config(data),
                    context=KDStepContext(rank=self.rank),
                )
                return result.loss, result.metrics

            loss, student_meta = self.engine.forward_step(micro_batch, loss_function=kd_loss, forward_only=False)
            loss.backward()
            output_lst.append(student_meta)

        outputs = postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
        grad_norm = self.engine.optimizer_step()
        if self.engine.is_mp_src_rank_with_outputs():
            outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

    def _step_config(self, data: TensorDict) -> KDStepConfig:
        global_step = int(tu.get_non_tensor_data(data=data, key="global_step", default=0))
        logits = self.kd_config.logits
        hidden = self.kd_config.hidden
        return KDStepConfig(
            global_step=global_step,
            top_k=int(logits.get("top_k", 0)),
            temperature=float(logits.get("temperature", 1.0)),
            logits_loss_type=str(logits.get("loss_type", "renormalized_top_k_forward_kl")),
            hidden_loss_type=str(hidden.get("loss_type", "normalized_mse")),
            layer_map=hidden.get("layer_map", []),
            schedules=self.kd_config.schedules,
            logits_enabled=bool(logits.get("enabled", True)),
            hidden_enabled=bool(hidden.get("enabled", True)),
        )

    def _install_hidden_hooks(self) -> None:
        for handle in [*self._student_hook_handles, *self._teacher_hook_handles]:
            handle.remove()
        self._student_hook_handles = []
        self._teacher_hook_handles = []

        hidden = self.kd_config.hidden
        if not bool(hidden.get("enabled", True)):
            return
        layer_map = [HiddenLayerMapEntry(**dict(entry)) for entry in hidden.get("layer_map", [])]
        self._student_hook_handles = register_mapped_hidden_hooks(
            self.engine.module,
            layer_map,
            role="student",
            store=self.student_hidden_store,
        )
        self._teacher_hook_handles = register_mapped_hidden_hooks(
            self.teacher_engine.module,
            layer_map,
            role="teacher",
            store=self.teacher_hidden_store,
        )

    def _assign_default_batch_meta(self, data: TensorDict) -> None:
        default_keys = dict(
            use_remove_padding=self.model_config.use_remove_padding,
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )
        for key, value in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: value})


def clone_teacher_model_config(student_model_config, teacher_path: str):
    teacher_model_config = copy.deepcopy(student_model_config)
    teacher_model_config.path = teacher_path
    teacher_model_config.hf_config_path = teacher_path
    teacher_model_config.tokenizer_path = teacher_path
    teacher_model_config.local_path = teacher_path
    teacher_model_config.local_hf_config_path = teacher_path
    teacher_model_config.local_tokenizer_path = teacher_path
    return teacher_model_config


def clone_teacher_engine_config(student_engine_config):
    teacher_engine_config = copy.deepcopy(student_engine_config)
    teacher_engine_config.forward_only = True
    teacher_engine_config.optimizer_offload = False
    return teacher_engine_config


def _masked_negative_log_prob(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if log_probs.shape != mask.shape:
        mask = align_response_mask_for_shifted_logits(mask, target_seq_len=log_probs.shape[-1])
    mask = mask.to(log_probs.device, dtype=log_probs.dtype)
    return -(log_probs * mask).sum() / mask.sum().clamp_min(1.0)


def _as_dense_or_flat(tensor: torch.Tensor) -> torch.Tensor:
    if getattr(tensor, "is_nested", False):
        return tensor.values()
    return tensor


def _metric(value: torch.Tensor) -> float:
    return float(value.detach().float().item())
