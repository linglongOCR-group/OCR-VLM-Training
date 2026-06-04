# Single-node SFT ablation notes

Use this reference when launching multiple independent one-node/8-NPU SFT ablations on Atlas A2 nodes.

## W&B-preserving relaunch pattern

When the baseline/reference script uses W&B, keep W&B enabled for ablations. If online W&B auth was previously broken but the user says it is now configured on the nodes:

1. Stop the old run processes for the exact experiment/master port.
2. Confirm `npu-smi info` shows no running NPU processes on the target node.
3. Relaunch with online W&B:

```bash
unset WANDB_MODE
export TRAINER_LOGGER='["console", "wandb"]'
export WANDB_PROJECT="$PROJECT_NAME"
export MASTER_ADDR=<node-host-plane-ip>
export MASTER_PORT=<unique-port>
export NNODES=1
export NODE_RANK=0
export NPUS_PER_NODE=8
export NPU_PER_NODE=8
export TOTAL_NPU=8
export TRAIN_IFACE=<inventory-train-iface>
export PROJECT_ROOT=/tmp/ocr-vlm-training/releases/<release-id>
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
# export MODEL_PATH, TRAIN_FILES, VAL_FILES, CKPTS_DIR, PROJECT_NAME, EXPERIMENT_NAME, GLOBAL_BATCH_SIZE, etc.
cd "$PROJECT_ROOT"
nohup bash scripts/train/run_multinode_sft_new.sh engine.param_offload=false engine.optimizer_offload=false \
  > training-sft-rank0.log 2>&1 &
echo $! > training-sft-rank0.pid
```

Do not use `VAR=value cd "$PROJECT_ROOT" && ...`; those assignments apply only to `cd` and will not be exported to the training script. A common symptom is:

```text
scripts/train/run_multinode_sft_new.sh: line 48: MASTER_ADDR: Please set MASTER_ADDR to node-0 host-plane IP
```

## Verification checklist

For each run, verify all of the following before reporting success:

- Process args show the intended data view, checkpoint path, output dir, experiment name, and `trainer.logger=["console", "wandb"]`.
- Log shows `MASTER_ADDR`, `TRAIN_FILES`, `VAL_FILES`, `GLOBAL_BATCH_SIZE`, `torch_npu`, and expected NPU count.
- Model loads, e.g. `Qwen2VLForConditionalGeneration contains ... parameters`.
- Dataset initializes and `Epoch 1/2` appears.
- At least one training `step:<n>` appears, unless the run resumed past a long prefetch point and the user only asked for startup.
- W&B is online: log contains credentials loaded/current login, `Syncing run ...`, project URL, and run URL.

Example W&B success lines:

```text
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /root/.netrc.
wandb: Currently logged in as: ...
wandb: Syncing run <experiment_name>
wandb: View project at https://wandb.ai/<entity>/<project>
wandb: View run at https://wandb.ai/<entity>/<project>/runs/<run_id>
```

## Deleted or stale logs after replacing a live release

If a release directory is replaced while its process is still running, the process may continue writing to deleted-but-open file descriptors. Do not relaunch solely because the new `training-sft-rank0.log` is missing/stale. Inspect `/proc/<launcher_pid>/fd/1` or `/proc/<rank0_pid>/fd/1` inside the container and confirm no matching `torchrun`/trainer process is alive before launching a replacement.
