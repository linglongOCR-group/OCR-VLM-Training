from pathlib import Path

import yaml


def test_multinode_sft_script_uses_ocr_multimodal_dataset_cls():
    script = Path("scripts/train/run_multinode_sft.sh").read_text()

    required_snippets = [
        'PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}',
        'export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"',
        "FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-False}",
        "-m verl_plugins.trainers.sft_trainer",
        'TRAIN_FILES="${TRAIN_FILES:-${TRAIN_FILE:-}}"',
        'VAL_FILES="${VAL_FILES:-${VAL_FILE:-null}}"',
        'USE_OCR_DATASET="${USE_OCR_DATASET:-True}"',
        'data.val_files="${VAL_FILES}"',
        'OCR_DATASET_ARGS=(+data.image_key=runtime_images',
        '+data.data_root="${OCR_DATA_ROOT}"',
        'data.custom_cls.name=OcrMultiTurnSFTDataset)',
        'OCR_DATASET_ARGS=(data.image_key=images)',
        '"${OCR_DATASET_ARGS[@]}"',
        '+model.freeze_vision_tower="${FREEZE_VISION_TOWER}"',
        'data.num_workers="${DATA_NUM_WORKERS}"',
        'data.train_max_samples="${TRAIN_MAX_SAMPLES}"',
        'data.val_max_samples="${VAL_MAX_SAMPLES}"',
        'data.ignore_input_ids_mismatch="${IGNORE_INPUT_IDS_MISMATCH}"',
        'trainer.project_name="${WANDB_PROJECT}"',
        'trainer.logger="${TRAINER_LOGGER}"',
        'trainer.default_local_dir="${CKPTS_DIR}"',
        'trainer.resume_from_path="${RESUME_FROM_PATH}"',
        'trainer.validate_only="${VALIDATE_ONLY:-False}"',
        'trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"',
    ]

    for snippet in required_snippets:
        assert snippet in script

    assert "  data.image_key=runtime_images" not in script


def test_single_node_sft_script_uses_repo_local_sft_entrypoint():
    script = Path("scripts/train/run_sft.sh").read_text()

    assert "FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-False}" in script
    assert "-m verl_plugins.trainers.sft_trainer" in script
    assert 'USE_OCR_DATASET="${USE_OCR_DATASET:-True}"' in script
    assert 'OCR_DATASET_ARGS=(+data.image_key=runtime_images' in script
    assert '+data.data_root="${OCR_DATA_ROOT}"' in script
    assert 'OCR_DATASET_ARGS=(data.image_key=images)' in script
    assert '"${OCR_DATASET_ARGS[@]}"' in script
    assert '+model.freeze_vision_tower="${FREEZE_VISION_TOWER}"' in script
    assert 'trainer.validate_only="${VALIDATE_ONLY:-False}"' in script
    assert 'trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"' in script


def test_sft_config_uses_repo_local_sft_entrypoint():
    config = Path("configs/train/verl/sft/qwen2_5_vl_fsdp.yaml").read_text()

    assert "entrypoint: verl_plugins.trainers.sft_trainer" in config
    assert "freeze_vision_tower: ${oc.env:FREEZE_VISION_TOWER,False}" in config
    assert "validate_only: ${oc.env:VALIDATE_ONLY,False}" in config
    assert "val_before_train: ${oc.env:VAL_BEFORE_TRAIN,False}" in config


def test_multinode_sft_script_does_not_keep_old_default_data_paths():
    script = Path("scripts/train/run_multinode_sft.sh").read_text()

    assert "/data/mineru25_sft/train.parquet" not in script
    assert "/data/mineru25_sft/val.parquet" not in script
    assert "run_mineru25_sft_dual_node.sh" not in script


def test_multinode_kd_sft_script_uses_repo_local_kd_entrypoint():
    script = Path("scripts/train/run_multinode_kd_sft.sh").read_text()

    required_snippets = [
        'TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:?TEACHER_MODEL_PATH is required}"',
        '-m verl_plugins.trainers.kd_sft_trainer',
        'importlib.import_module("verl_plugins.trainers.kd_sft_trainer")',
        '+kd.teacher.path="${TEACHER_MODEL_PATH}"',
        '+kd.logits.top_k="${KD_TOP_K}"',
        '+kd.logits.temperature="${KD_TEMPERATURE}"',
        '+kd.logits.loss_type="${KD_LOGIT_LOSS_TYPE}"',
        '+kd.hidden.layer_map="${KD_HIDDEN_LAYER_MAP}"',
        '+kd.schedules.logits.type=linear_warmup_constant',
        '+kd.schedules.hidden.type=linear_warmup_hold_decay',
        '"$@"',
    ]

    for snippet in required_snippets:
        assert snippet in script
    assert 'KD_HIDDEN_LAYER_MAP="${KD_HIDDEN_LAYER_MAP:-' not in script
    assert "KD_HIDDEN_LAYER_MAP='[{student_hidden_index:1,teacher_hidden_index:1}]'" in script


def test_kd_sft_config_uses_repo_local_kd_entrypoint_and_default_layer_map():
    config = Path("configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml").read_text()

    assert "entrypoint: verl_plugins.trainers.kd_sft_trainer" in config
    assert "teacher:" in config
    assert "path: ${oc.env:TEACHER_MODEL_PATH}" in config
    assert "loss_type: ${oc.env:KD_LOGIT_LOSS_TYPE,renormalized_top_k_forward_kl}" in config
    assert "student_hidden_index" in config
    assert "teacher_hidden_index" in config


def test_grpo_script_runs_cdm_preflight_only_for_cdm_reward_profiles():
    script = Path("scripts/train/run_grpo_fsdp.sh").read_text()

    assert 'REWARD_PROFILE="${REWARD_PROFILE:-levenshtein_only_v1}"' in script
    assert 'CDM_REWARD_PROFILE="${CDM_REWARD_PROFILE:-cdm_katex_v1}"' in script
    assert 'if [ "${REWARD_PROFILE}" = "formula_cdm_v1" ]; then' in script
    assert "python -m tools.training_ops.grpo_cdm_preflight" in script
    assert '--routing-default="${DEFAULT_REWARD_PROFILE}"' in script
    assert '--formula-profile="${CDM_REWARD_PROFILE}"' in script
    assert '--cdm-profile="${CDM_REWARD_PROFILE}"' in script
    assert '--cdm-preflight-required="${CDM_REWARD_PREFLIGHT_REQUIRED:-True}"' in script
    assert '+reward.custom_reward_function.reward_kwargs.routing.default="${DEFAULT_REWARD_PROFILE}"' in script
    assert '+reward.custom_reward_function.reward_kwargs.routing.by_task.formula="${CDM_REWARD_PROFILE}"' in script
    assert '+reward.custom_reward_function.reward_kwargs.rewards."${CDM_REWARD_PROFILE}".type=cdm_latex_render' in script
    assert '+reward.custom_reward_function.reward_kwargs.rewards."${CDM_REWARD_PROFILE}".preflight_required="${CDM_REWARD_PREFLIGHT_REQUIRED:-True}"' in script


def test_cdm_grpo_reward_config_uses_explicit_routing_profiles():
    config = yaml.safe_load(Path("configs/train/verl/rl/grpo_formula_cdm_reward.yaml").read_text())
    reward_kwargs = config["reward"]["custom_reward_function"]["reward_kwargs"]

    assert reward_kwargs["reward_profile"] == "formula_cdm_v1"
    assert reward_kwargs["routing"] == {
        "default": "normalized_levenshtein_v1",
        "by_task": {"formula": "cdm_katex_v1"},
    }
    assert reward_kwargs["rewards"]["normalized_levenshtein_v1"] == {
        "type": "normalized_levenshtein",
        "version": "levenshtein_v1",
    }
    assert reward_kwargs["rewards"]["cdm_katex_v1"] == {
        "type": "cdm_latex_render",
        "version": "cdm_katex_v1",
        "service_url": "${oc.env:CDM_REWARD_URL,http://127.0.0.1:8765}",
        "timeout_ms": "${oc.env:CDM_REWARD_TIMEOUT_MS,1000}",
        "fail_score": "${oc.env:CDM_REWARD_FAIL_SCORE,0.0}",
        "expected_version": "${oc.env:CDM_REWARD_EXPECTED_VERSION,cdm_katex_v1}",
        "preflight_required": "${oc.env:CDM_REWARD_PREFLIGHT_REQUIRED,True}",
    }
