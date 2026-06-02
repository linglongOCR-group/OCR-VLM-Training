from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from verl_plugins.trainers.kd_schedules import validate_schedule


@dataclass(frozen=True)
class ModelContract:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    model_path: str | None = None
    tokenizer_path: str | None = None
    tokenizer_name: str | None = None
    processor_type: str | None = None
    input_contract: Sequence[str] = field(default_factory=tuple)
    output_token_semantics: str | None = None


@dataclass(frozen=True)
class KDSFTPrecheckConfig:
    teacher_path: str
    student_path: str
    logits: Mapping[str, Any]
    hidden: Mapping[str, Any]
    schedules: Mapping[str, Mapping[str, Any]]
    precheck: Mapping[str, Any]


@dataclass(frozen=True)
class DryRunPrecheckResult:
    teacher_output: Any
    student_output: Any
    top_k_targets: Any
    total_loss: Any


def parse_kd_config(config: Mapping[str, Any]) -> KDSFTPrecheckConfig:
    kd = _as_mapping(config.get("kd"), "kd")
    teacher = _as_mapping(kd.get("teacher"), "kd.teacher")
    teacher_path = _required_str(teacher.get("path"), "kd.teacher.path")
    student_path = _required_str(_as_mapping(config.get("model", {}), "model").get("path"), "model.path")

    logits = _as_mapping(kd.get("logits", {}), "kd.logits")
    hidden = _as_mapping(kd.get("hidden", {}), "kd.hidden")
    precheck = _as_mapping(kd.get("precheck", {}), "kd.precheck")
    schedules = _extract_schedules(kd)

    _validate_precheck_settings(precheck)
    _validate_logits_settings(logits)
    _validate_hidden_settings(hidden)
    _validate_schedule_set(schedules, logits_enabled=_is_enabled(logits), hidden_enabled=_is_enabled(hidden))

    return KDSFTPrecheckConfig(
        teacher_path=teacher_path,
        student_path=student_path,
        logits=logits,
        hidden=hidden,
        schedules=schedules,
        precheck=precheck,
    )


def validate_top_k(*, top_k: int, vocab_size: int) -> None:
    if top_k <= 0:
        raise ValueError("kd.logits.top_k must be positive")
    if top_k > vocab_size:
        raise ValueError("kd.logits.top_k must not exceed vocab size")


def validate_layer_map(
    layer_map: Sequence[Mapping[str, Any] | Any],
    *,
    student_layers: int,
    teacher_layers: int,
) -> None:
    if not layer_map:
        raise ValueError("kd.hidden.layer_map is required when hidden KD is enabled")
    for entry in layer_map:
        student_index = _entry_int(entry, "student_hidden_index")
        teacher_index = _entry_int(entry, "teacher_hidden_index")
        if student_index < 1 or student_index > student_layers:
            raise ValueError(f"Invalid student hidden index: {student_index}")
        if teacher_index < 1 or teacher_index > teacher_layers:
            raise ValueError(f"Invalid teacher hidden index: {teacher_index}")


def validate_contract(
    student: ModelContract,
    teacher: ModelContract,
    *,
    hidden_kd_enabled: bool,
    student_tokenizer: Any | None = None,
    teacher_tokenizer: Any | None = None,
    student_processor: Any | None = None,
    teacher_processor: Any | None = None,
) -> None:
    _validate_tokenizer_contract(
        student,
        teacher,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )
    if student.vocab_size != teacher.vocab_size:
        raise ValueError("Teacher and student vocab sizes must match")
    if student.output_token_semantics and teacher.output_token_semantics and student.output_token_semantics != teacher.output_token_semantics:
        raise ValueError("Teacher and student output-token semantics must match")
    if hidden_kd_enabled and student.hidden_size != teacher.hidden_size:
        raise ValueError("Teacher and student hidden sizes must match")
    _validate_processor_contract(
        student,
        teacher,
        student_processor=student_processor,
        teacher_processor=teacher_processor,
    )


def validate_teacher_frozen(teacher_model: Any) -> None:
    if teacher_model is None:
        return
    parameters = teacher_model.parameters() if hasattr(teacher_model, "parameters") else ()
    if any(bool(getattr(parameter, "requires_grad", False)) for parameter in parameters):
        raise ValueError("KD teacher model must be frozen before training starts")


def validate_kd_config(
    config: Mapping[str, Any],
    *,
    student: ModelContract,
    teacher: ModelContract,
    student_tokenizer: Any | None = None,
    teacher_tokenizer: Any | None = None,
    student_processor: Any | None = None,
    teacher_processor: Any | None = None,
    teacher_model: Any | None = None,
) -> KDSFTPrecheckConfig:
    parsed = parse_kd_config(config)
    hidden_enabled = _is_enabled(parsed.hidden)

    validate_contract(
        student,
        teacher,
        hidden_kd_enabled=hidden_enabled,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_processor=student_processor,
        teacher_processor=teacher_processor,
    )
    if _is_enabled(parsed.logits):
        validate_top_k(top_k=int(parsed.logits["top_k"]), vocab_size=student.vocab_size)
    if hidden_enabled:
        validate_layer_map(
            parsed.hidden.get("layer_map", []),
            student_layers=student.num_hidden_layers,
            teacher_layers=teacher.num_hidden_layers,
        )
    validate_teacher_frozen(teacher_model)
    return parsed


def extract_top_k_teacher_targets(
    teacher_logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float = 1.0,
    loss_type: str = "renormalized_top_k_forward_kl",
) -> dict[str, torch.Tensor]:
    validate_top_k(top_k=top_k, vocab_size=int(teacher_logits.shape[-1]))
    if temperature <= 0:
        raise ValueError("kd.logits.temperature must be positive")
    scaled = teacher_logits.detach().float() / float(temperature)
    top_values, top_indices = torch.topk(scaled, k=top_k, dim=-1)
    if loss_type == "renormalized_top_k_forward_kl":
        top_log_probs = F.log_softmax(top_values, dim=-1)
    elif loss_type == "truncated_forward_kl":
        top_log_probs = F.log_softmax(scaled, dim=-1).gather(dim=-1, index=top_indices)
    else:
        raise ValueError(f"Unsupported logits KD loss type: {loss_type!r}")
    return {
        "indices": top_indices,
        "log_probs": top_log_probs,
    }


def dry_run_batch_precheck(
    batch: Mapping[str, Any],
    *,
    teacher_forward: Callable[[Mapping[str, Any]], Any],
    student_forward: Callable[[Mapping[str, Any]], Any],
    extract_top_k_targets: Callable[[Any], Any] | None = None,
    hidden_hook_runner: Callable[[Callable[[], DryRunPrecheckResult]], DryRunPrecheckResult] | None = None,
    compose_total_loss: Callable[..., Any],
    top_k: int | None = None,
    temperature: float = 1.0,
    logits_loss_type: str = "renormalized_top_k_forward_kl",
) -> DryRunPrecheckResult:
    def run_contract() -> DryRunPrecheckResult:
        teacher_output = teacher_forward(batch)
        if extract_top_k_targets is None:
            if top_k is None:
                raise ValueError("dry-run batch precheck requires configured top_k when no extractor is injected")
            if not hasattr(teacher_output, "logits"):
                raise ValueError("teacher forward output must expose logits for top-k target extraction")
            top_k_targets = extract_top_k_teacher_targets(
                teacher_output.logits,
                top_k=top_k,
                temperature=temperature,
                loss_type=logits_loss_type,
            )
        else:
            top_k_targets = extract_top_k_targets(teacher_output)
        student_output = student_forward(batch)
        total_loss = compose_total_loss(
            batch=batch,
            teacher_output=teacher_output,
            student_output=student_output,
            top_k_targets=top_k_targets,
        )
        return DryRunPrecheckResult(
            teacher_output=teacher_output,
            student_output=student_output,
            top_k_targets=top_k_targets,
            total_loss=total_loss,
        )

    if hidden_hook_runner is not None:
        return hidden_hook_runner(run_contract)
    return run_contract()


def _extract_schedules(kd: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if "schedules" in kd:
        return _as_mapping(kd["schedules"], "kd.schedules")
    if "losses" in kd:
        return {
            name: _as_mapping(loss_cfg, f"kd.losses.{name}")["schedule"]
            for name, loss_cfg in _as_mapping(kd["losses"], "kd.losses").items()
        }
    raise ValueError("kd.schedules is required")


def _validate_schedule_set(
    schedules: Mapping[str, Mapping[str, Any]],
    *,
    logits_enabled: bool,
    hidden_enabled: bool,
) -> None:
    required = ["sft"]
    if logits_enabled:
        required.append("logits")
    if hidden_enabled:
        required.append("hidden")
    for name in required:
        if name not in schedules:
            raise ValueError(f"kd.schedules.{name} is required")
        validate_schedule(_as_mapping(schedules[name], f"kd.schedules.{name}"))


def _validate_precheck_settings(precheck: Mapping[str, Any]) -> None:
    mode = precheck.get("mode", "config")
    if mode not in {"config", "dry_run_batch"}:
        raise ValueError("kd.precheck.mode must be 'config' or 'dry_run_batch'")


def _validate_logits_settings(logits: Mapping[str, Any]) -> None:
    if not _is_enabled(logits):
        return
    if logits.get("mode", "top_k") != "top_k":
        raise ValueError("kd.logits.mode must be 'top_k'")
    if int(logits.get("top_k", 0)) <= 0:
        raise ValueError("kd.logits.top_k must be positive")
    if float(logits.get("temperature", 1.0)) <= 0:
        raise ValueError("kd.logits.temperature must be positive")
    loss_type = logits.get("loss_type", "renormalized_top_k_forward_kl")
    if loss_type not in {"renormalized_top_k_forward_kl", "truncated_forward_kl"}:
        raise ValueError(f"Unsupported logits KD loss type: {loss_type!r}")


def _validate_hidden_settings(hidden: Mapping[str, Any]) -> None:
    if not _is_enabled(hidden):
        return
    loss_type = hidden.get("loss_type", "normalized_mse")
    if loss_type not in {"normalized_mse", "cosine"}:
        raise ValueError(f"Unsupported hidden KD loss type: {loss_type!r}")


def _validate_tokenizer_contract(
    student: ModelContract,
    teacher: ModelContract,
    *,
    student_tokenizer: Any | None,
    teacher_tokenizer: Any | None,
) -> None:
    student_identity = _tokenizer_identity(student_tokenizer, fallback=student)
    teacher_identity = _tokenizer_identity(teacher_tokenizer, fallback=teacher)
    if student_identity and teacher_identity and student_identity != teacher_identity:
        raise ValueError("Teacher and student tokenizers must match")


def _validate_processor_contract(
    student: ModelContract,
    teacher: ModelContract,
    *,
    student_processor: Any | None,
    teacher_processor: Any | None,
) -> None:
    student_type, student_inputs = _processor_contract(student_processor, fallback=student)
    teacher_type, teacher_inputs = _processor_contract(teacher_processor, fallback=teacher)
    if student_type and teacher_type and student_type != teacher_type:
        raise ValueError("Teacher and student VLM processor types must match")
    if student_inputs and teacher_inputs and student_inputs != teacher_inputs:
        raise ValueError("Teacher and student VLM processor input contract fields must match")


def _tokenizer_identity(tokenizer: Any | None, *, fallback: ModelContract) -> str | None:
    if tokenizer is not None:
        for attr in ("name_or_path", "tokenizer_path", "vocab_file"):
            value = getattr(tokenizer, attr, None)
            if value:
                return str(value)
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, Mapping) and init_kwargs.get("name_or_path"):
            return str(init_kwargs["name_or_path"])
    if fallback.tokenizer_path:
        return fallback.tokenizer_path
    if fallback.tokenizer_name:
        return fallback.tokenizer_name
    return None


def _processor_contract(processor: Any | None, *, fallback: ModelContract) -> tuple[str | None, tuple[str, ...]]:
    processor_type = fallback.processor_type
    input_contract = tuple(fallback.input_contract)
    if processor is not None:
        processor_type = getattr(processor, "processor_type", None) or type(processor).__name__
        names = getattr(processor, "model_input_names", None) or getattr(processor, "input_fields", None)
        if names:
            input_contract = tuple(str(name) for name in names)
    return (str(processor_type) if processor_type else None, input_contract)


def _is_enabled(config: Mapping[str, Any]) -> bool:
    return bool(config.get("enabled", True))


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{name} must be a mapping")


def _required_str(value: Any, name: str) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} is required")
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _entry_int(entry: Mapping[str, Any] | Any, key: str) -> int:
    if isinstance(entry, Mapping):
        return int(entry[key])
    return int(getattr(entry, key))
