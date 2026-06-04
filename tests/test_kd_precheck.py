from types import SimpleNamespace

import pytest
import torch

from verl_plugins.trainers.kd_precheck import (
    ModelContract,
    dry_run_batch_precheck,
    extract_top_k_teacher_targets,
    parse_kd_config,
    validate_kd_config,
)


def _valid_config(**kd_overrides):
    kd = {
        "teacher": {"path": "/models/teacher"},
        "precheck": {"required": True, "mode": "config"},
        "logits": {
            "enabled": True,
            "mode": "top_k",
            "top_k": 2,
            "temperature": 2.0,
            "loss_type": "renormalized_top_k_forward_kl",
        },
        "hidden": {
            "enabled": True,
            "loss_type": "normalized_mse",
            "layer_map": [{"student_hidden_index": 1, "teacher_hidden_index": 2}],
        },
        "schedules": {
            "sft": {"type": "constant", "value": 1.0},
            "logits": {
                "type": "linear_warmup_constant",
                "start": 0.0,
                "peak": 0.3,
                "warmup_steps": 10,
            },
            "hidden": {
                "type": "linear_warmup_hold_decay",
                "start": 0.0,
                "peak": 0.1,
                "warmup_steps": 10,
                "hold_steps": 20,
                "decay_steps": 10,
                "end": 0.0,
            },
        },
    }
    kd.update(kd_overrides)
    return {"model": {"path": "/models/student"}, "kd": kd}


def _contract(**overrides):
    values = {
        "vocab_size": 8,
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "tokenizer_path": "/models/shared-tokenizer",
        "tokenizer_name": "shared-tokenizer",
        "processor_type": "Qwen2VLProcessor",
        "input_contract": ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"),
        "output_token_semantics": "qwen2_5_vl_chat_template",
    }
    values.update(overrides)
    return ModelContract(**values)


def _frozen_teacher():
    return SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=False)])


def test_parse_kd_config_extracts_student_teacher_paths_and_precheck_settings():
    parsed = parse_kd_config(_valid_config())

    assert parsed.teacher_path == "/models/teacher"
    assert parsed.student_path == "/models/student"
    assert parsed.precheck["mode"] == "config"
    assert parsed.logits["top_k"] == 2
    assert parsed.hidden["layer_map"] == [{"student_hidden_index": 1, "teacher_hidden_index": 2}]


def test_parse_kd_config_requires_student_path():
    config = _valid_config()
    config["model"].pop("path")

    with pytest.raises(ValueError, match="model.path is required"):
        parse_kd_config(config)


def test_validate_kd_config_rejects_top_k_outside_student_vocab():
    config = _valid_config(logits={"enabled": True, "mode": "top_k", "top_k": 9})

    with pytest.raises(ValueError, match="kd.logits.top_k must not exceed vocab size"):
        validate_kd_config(config, student=_contract(), teacher=_contract(), teacher_model=_frozen_teacher())


def test_validate_kd_config_rejects_missing_hidden_layer_map_when_hidden_kd_enabled():
    config = _valid_config(hidden={"enabled": True, "loss_type": "cosine", "layer_map": []})

    with pytest.raises(ValueError, match="kd.hidden.layer_map is required"):
        validate_kd_config(config, student=_contract(), teacher=_contract(), teacher_model=_frozen_teacher())


def test_validate_kd_config_rejects_output_token_semantics_mismatch():
    with pytest.raises(ValueError, match="output-token semantics"):
        validate_kd_config(
            _valid_config(),
            student=_contract(output_token_semantics="student-chat-template"),
            teacher=_contract(output_token_semantics="teacher-chat-template"),
            teacher_model=_frozen_teacher(),
        )


def test_validate_kd_config_rejects_tokenizer_vocab_and_processor_mismatch():
    with pytest.raises(ValueError, match="tokenizers must match"):
        validate_kd_config(
            _valid_config(),
            student=_contract(tokenizer_path="/models/student-tokenizer"),
            teacher=_contract(tokenizer_path="/models/teacher-tokenizer"),
            teacher_model=_frozen_teacher(),
        )

    with pytest.raises(ValueError, match="vocab sizes must match"):
        validate_kd_config(
            _valid_config(),
            student=_contract(vocab_size=8),
            teacher=_contract(vocab_size=9),
            teacher_model=_frozen_teacher(),
        )

    with pytest.raises(ValueError, match="VLM processor input contract"):
        validate_kd_config(
            _valid_config(),
            student=_contract(input_contract=("input_ids", "pixel_values")),
            teacher=_contract(input_contract=("input_ids", "attention_mask", "pixel_values")),
            teacher_model=_frozen_teacher(),
        )


class _FakeProcessor:
    model_input_names = ("input_ids", "pixel_values")

    def __init__(self, name_or_path):
        self.name_or_path = name_or_path


def test_processor_contract_allows_same_class_with_different_model_paths():
    validate_kd_config(
        _valid_config(hidden={"enabled": False}),
        student=_contract(processor_type=None, input_contract=()),
        teacher=_contract(processor_type=None, input_contract=()),
        student_processor=_FakeProcessor("/models/student"),
        teacher_processor=_FakeProcessor("/models/teacher"),
        teacher_model=_frozen_teacher(),
    )


def test_tokenizer_identity_allows_same_path_from_tokenizer_object_or_contract():
    validate_kd_config(
        _valid_config(),
        student=_contract(tokenizer_path=None, tokenizer_name=None),
        teacher=_contract(tokenizer_path="/models/shared-tokenizer", tokenizer_name=None),
        student_tokenizer=SimpleNamespace(name_or_path="/models/shared-tokenizer"),
        teacher_model=_frozen_teacher(),
    )


class _FakeTokenizer:
    def __init__(self, name_or_path, vocab):
        self.name_or_path = name_or_path
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)


def test_tokenizer_contract_allows_different_paths_when_tokenizer_vocab_matches():
    validate_kd_config(
        _valid_config(),
        student=_contract(tokenizer_path="/models/student-trim", tokenizer_name="/models/student-trim"),
        teacher=_contract(tokenizer_path="/models/origin", tokenizer_name="/models/origin"),
        student_tokenizer=_FakeTokenizer("/models/student-trim", {"a": 1, "b": 2}),
        teacher_tokenizer=_FakeTokenizer("/models/origin", {"a": 1, "b": 2}),
        teacher_model=_frozen_teacher(),
    )

    with pytest.raises(ValueError, match="tokenizers must match"):
        validate_kd_config(
            _valid_config(),
            student=_contract(tokenizer_path="/models/student-trim", tokenizer_name="/models/student-trim"),
            teacher=_contract(tokenizer_path="/models/origin", tokenizer_name="/models/origin"),
            student_tokenizer=_FakeTokenizer("/models/student-trim", {"a": 1, "b": 2}),
            teacher_tokenizer=_FakeTokenizer("/models/origin", {"a": 1, "c": 2}),
            teacher_model=_frozen_teacher(),
        )


def test_hidden_size_is_required_to_match_only_when_hidden_kd_enabled():
    validate_kd_config(
        _valid_config(hidden={"enabled": False}),
        student=_contract(hidden_size=16),
        teacher=_contract(hidden_size=32),
        teacher_model=_frozen_teacher(),
    )

    with pytest.raises(ValueError, match="hidden sizes must match"):
        validate_kd_config(
            _valid_config(),
            student=_contract(hidden_size=16),
            teacher=_contract(hidden_size=32),
            teacher_model=_frozen_teacher(),
        )


def test_validate_kd_config_rejects_layer_map_bounds_and_unfrozen_teacher():
    with pytest.raises(ValueError, match="Invalid teacher hidden index: 5"):
        validate_kd_config(
            _valid_config(
                hidden={
                    "enabled": True,
                    "loss_type": "normalized_mse",
                    "layer_map": [{"student_hidden_index": 1, "teacher_hidden_index": 5}],
                }
            ),
            student=_contract(),
            teacher=_contract(),
            teacher_model=_frozen_teacher(),
        )

    trainable_teacher = SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=True)])
    with pytest.raises(ValueError, match="teacher model must be frozen"):
        validate_kd_config(
            _valid_config(),
            student=_contract(),
            teacher=_contract(),
            teacher_model=trainable_teacher,
        )


def test_extract_top_k_teacher_targets_uses_full_teacher_denominator_for_truncated_kl():
    logits = torch.tensor([[[4.0, 2.0, 1.0]]])

    targets = extract_top_k_teacher_targets(
        logits,
        top_k=2,
        temperature=2.0,
        loss_type="truncated_forward_kl",
    )

    expected = torch.log_softmax(logits / 2.0, dim=-1).gather(dim=-1, index=targets["indices"])
    assert torch.equal(targets["indices"], torch.tensor([[[0, 1]]]))
    assert torch.allclose(targets["log_probs"], expected)


def test_extract_top_k_teacher_targets_renormalizes_for_renormalized_kl():
    logits = torch.tensor([[[4.0, 2.0, 1.0]]])

    targets = extract_top_k_teacher_targets(
        logits,
        top_k=2,
        temperature=2.0,
        loss_type="renormalized_top_k_forward_kl",
    )

    assert torch.exp(targets["log_probs"]).sum(dim=-1).item() == pytest.approx(1.0)


def test_dry_run_batch_precheck_requires_configured_top_k_for_fallback_extractor():
    batch = {"input_ids": torch.tensor([[1, 2]])}

    with pytest.raises(ValueError, match="requires configured top_k"):
        dry_run_batch_precheck(
            batch,
            teacher_forward=lambda _: SimpleNamespace(logits=torch.zeros(1, 1, 4)),
            student_forward=lambda _: SimpleNamespace(logits=torch.zeros(1, 1, 4)),
            compose_total_loss=lambda **_: torch.tensor(0.0),
        )


def test_dry_run_batch_precheck_fallback_uses_configured_top_k_temperature_and_loss_type():
    batch = {"input_ids": torch.tensor([[1, 2]])}

    result = dry_run_batch_precheck(
        batch,
        teacher_forward=lambda _: SimpleNamespace(logits=torch.tensor([[[4.0, 2.0, 1.0]]])),
        student_forward=lambda _: SimpleNamespace(logits=torch.zeros(1, 1, 3)),
        compose_total_loss=lambda **_: torch.tensor(0.0),
        top_k=2,
        temperature=2.0,
        logits_loss_type="truncated_forward_kl",
    )

    expected = torch.log_softmax(torch.tensor([[[4.0, 2.0, 1.0]]]) / 2.0, dim=-1).gather(
        dim=-1,
        index=result.top_k_targets["indices"],
    )
    assert result.top_k_targets["indices"].shape == (1, 1, 2)
    assert torch.allclose(result.top_k_targets["log_probs"], expected)


def test_dry_run_batch_precheck_exposes_v1_forward_topk_hidden_and_loss_contract():
    events = []
    batch = {"input_ids": torch.tensor([[1, 2]])}

    def teacher_forward(received_batch):
        events.append(("teacher_forward", received_batch))
        return SimpleNamespace(logits=torch.tensor([[[1.0, 3.0, 2.0]]]))

    def student_forward(received_batch):
        events.append(("student_forward", received_batch))
        return SimpleNamespace(logits=torch.tensor([[[2.0, 1.0, 0.0]]]))

    def extract_top_k_targets(teacher_output):
        events.append(("top_k", teacher_output.logits.shape))
        return {"indices": torch.tensor([[[1, 2]]]), "log_probs": torch.log(torch.tensor([[[0.7, 0.3]]]))}

    def hidden_hook_runner(run_forwards):
        events.append(("hidden_hooks", None))
        return run_forwards()

    def compose_total_loss(**kwargs):
        events.append(("total_loss", sorted(kwargs)))
        return torch.tensor(1.25)

    result = dry_run_batch_precheck(
        batch,
        teacher_forward=teacher_forward,
        student_forward=student_forward,
        extract_top_k_targets=extract_top_k_targets,
        hidden_hook_runner=hidden_hook_runner,
        compose_total_loss=compose_total_loss,
    )

    assert result.total_loss.item() == pytest.approx(1.25)
    assert [event[0] for event in events] == [
        "hidden_hooks",
        "teacher_forward",
        "top_k",
        "student_forward",
        "total_loss",
    ]
    assert events[1][1] is batch
    assert result.top_k_targets["indices"].shape == (1, 1, 2)
