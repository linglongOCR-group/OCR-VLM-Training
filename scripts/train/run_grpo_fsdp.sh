#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
TRAIN_FILE=${TRAIN_FILE:?TRAIN_FILE is required}
USE_VALIDATION=${USE_VALIDATION:-True}
if [ "${USE_VALIDATION}" = "True" ]; then
  VAL_FILE=${VAL_FILE:?VAL_FILE is required when USE_VALIDATION=True}
else
  VAL_FILE=${VAL_FILE:-"${TRAIN_FILE}"}
  VAL_BEFORE_TRAIN=False
  TEST_FREQ=-1
  LOG_VAL_GENERATIONS=0
fi
CKPTS_DIR=${CKPTS_DIR:-"${PROJECT_ROOT}/checkpoints/grpo-fsdp-vllm"}
WANDB_PROJECT=${WANDB_PROJECT:-ocr-vlm-training}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo-fsdp-vllm}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_CUDAGRAPH_MODE=${ROLLOUT_CUDAGRAPH_MODE:-NONE}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-False}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
ACTOR_ENTROPY_CHECKPOINTING=${ACTOR_ENTROPY_CHECKPOINTING:-False}
ACTOR_ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ACTOR_ENTROPY_FROM_LOGITS_WITH_CHUNKING:-False}
REF_ENTROPY_CHECKPOINTING=${REF_ENTROPY_CHECKPOINTING:-False}
REF_ENTROPY_FROM_LOGITS_WITH_CHUNKING=${REF_ENTROPY_FROM_LOGITS_WITH_CHUNKING:-False}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
RESUME_MODE=${RESUME_MODE:-auto}
VALIDATE_GRPO_VIEW=${VALIDATE_GRPO_VIEW:-True}
VALIDATE_GRPO_MAX_ROWS_PER_FILE=${VALIDATE_GRPO_MAX_ROWS_PER_FILE:-1000}
OCR_DATA_ROOT=${OCR_DATA_ROOT:-}
DISABLE_FLASHCOMM_FOR_TP1=${DISABLE_FLASHCOMM_FOR_TP1:-True}
ENABLE_ASCEND_PERF_ENV=${ENABLE_ASCEND_PERF_ENV:-False}
ENABLE_JEMALLOC=${ENABLE_JEMALLOC:-False}

if [ "${ENABLE_ASCEND_PERF_ENV}" = "True" ]; then
  export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
  export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
  export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-1}
  export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
  export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
  export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
  export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
  export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=${VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE:-1}
  export VLLM_ASCEND_ENABLE_PREFETCH_MLP=${VLLM_ASCEND_ENABLE_PREFETCH_MLP:-1}
fi

if [ "${ENABLE_JEMALLOC}" = "True" ] && [ -z "${LD_PRELOAD:-}" ]; then
  if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
  elif [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
  elif [ -f /usr/local/lib/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
  fi
fi

export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export WANDB_MODE=${WANDB_MODE:-offline}
if [ "${DISABLE_FLASHCOMM_FOR_TP1}" = "True" ] && [ "${ROLLOUT_TP_SIZE}" -le 1 ]; then
  export VLLM_ASCEND_ENABLE_FLASHCOMM=0
  export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
fi

if [ "${VALIDATE_GRPO_VIEW}" = "True" ]; then
  VALIDATE_IMAGE_ARGS=()
  if [ "${USE_VALIDATION}" = "True" ]; then
    python -m tools.data_management.validate_grpo_view \
      --max-rows-per-file="${VALIDATE_GRPO_MAX_ROWS_PER_FILE}" \
      "${VALIDATE_IMAGE_ARGS[@]}" \
      "${TRAIN_FILE}" \
      "${VAL_FILE}"
  else
    python -m tools.data_management.validate_grpo_view \
      --max-rows-per-file="${VALIDATE_GRPO_MAX_ROWS_PER_FILE}" \
      "${VALIDATE_IMAGE_ARGS[@]}" \
      "${TRAIN_FILE}"
  fi
fi

RAY_ARGS=()
if [ -n "${RAY_ADDRESS:-}" ]; then
  RAY_ARGS+=(+ray_kwargs.ray_init.address="${RAY_ADDRESS}")
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.image_key=runtime_images \
  +data.data_root="${OCR_DATA_ROOT}" \
  data.custom_cls.path="${PROJECT_ROOT}/tools/data_management/runtime/verl_multimodal_dataset.py" \
  data.custom_cls.name=OcrRLHFDataset \
  data.reward_fn_key=data_source \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${ACTOR_USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.01}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.entropy_checkpointing="${ACTOR_ENTROPY_CHECKPOINTING}" \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking="${ACTOR_ENTROPY_FROM_LOGITS_WITH_CHUNKING}" \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-False}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="${ROLLOUT_CUDAGRAPH_MODE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.entropy_checkpointing="${REF_ENTROPY_CHECKPOINTING}" \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking="${REF_ENTROPY_FROM_LOGITS_WITH_CHUNKING}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path="${PROJECT_ROOT}/verl_plugins/rewards/aggregate.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.reward_version=levenshtein_v1 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.device=npu \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS:-8}" \
  "${RAY_ARGS[@]}" \
  "$@"
