# OCR VLM Training Bootstrap

This repository is a thin integration layer around VERL for OCR VLM SFT and
GRPO experiments on Ascend/NPU infrastructure.

The project deliberately avoids reimplementing training. VERL owns distributed
training, rollout, checkpoint save/load, FSDP, vLLM/vLLM-Ascend, Megatron, and
MindSpeed integration. This repo provides:

- spec-aligned source, canonical, and view dataset processing,
- canonical-to-SFT/RLVR parquet view construction,
- an initial Normalized Levenshtein reward adapter for RLVR bring-up,
- W&B reference-artifact helpers for local checkpoints,
- launch wrappers and config examples.

## Data Module

The data module follows the dataset specs under `docs/`:

```text
Source -> Canonical -> View
```

Canonical data stores model-independent documents, pages, regions, task records,
image asset manifests, and lineage. Views materialize model-specific SFT or RLVR
training parquet with prompts, serialized labels, split assignments, and reward
payloads.

Set `OCR_DATA_ROOT` once per environment. The data CLI derives
`sources/`, `canonical/`, and `views/` from that root unless an explicit path is
provided:

```bash
export OCR_DATA_ROOT=/home/byhou/datasets/ocr-training
```

MinerU-annotated datasets can be converted into canonical partitions with the
repo-local data CLI:

```bash
scripts/data/docds export-source mineru \
  --source-config configs/data/sources/docbank_mineru.yaml
```

For a scoped export:

```bash
scripts/data/docds export-source mineru \
  --source-config configs/data/sources/docbank_mineru.yaml \
  --tasks layout,table,formula,text
```

By default, malformed annotation samples stop the export. Use `--skip-errors`
only when you want those samples listed in the manifest and omitted from the
canonical shards. `allow_unreadable_images` only controls missing or
permission-denied source images.

The profile writes partitioned parquet under:

```text
/home/byhou/datasets/ocr-training/canonical/entities/documents/source=DocBank_500K/
/home/byhou/datasets/ocr-training/canonical/entities/pages/source=DocBank_500K/
/home/byhou/datasets/ocr-training/canonical/entities/regions/source=DocBank_500K/
/home/byhou/datasets/ocr-training/canonical/records/<task>/source=DocBank_500K/
/home/byhou/datasets/ocr-training/canonical/assets/manifests/source=DocBank_500K/
/home/byhou/datasets/ocr-training/canonical/manifests/sources/DocBank_500K.json
```

Validate canonical output:

```bash
scripts/data/docds validate-canonical --source DocBank_500K
```

Build and validate a training view from `views/<view_name>/view.yaml`:

```bash
scripts/data/docds build-view /home/byhou/datasets/ocr-training/views/mineru25_rlvr/view.yaml
scripts/data/docds validate-view /home/byhou/datasets/ocr-training/views/mineru25_rlvr
scripts/data/docds reward-smoke-test --view /home/byhou/datasets/ocr-training/views/mineru25_rlvr --limit 1000
```

## GRPO With Ray

Start Ray on the head node:

```bash
bash scripts/cluster/start_ray_head.sh
```

Start Ray on worker nodes:

```bash
RAY_HEAD_ADDRESS=head-node:6379 bash scripts/cluster/start_ray_worker.sh
```

Launch the default FSDP + vLLM GRPO path:

```bash
MODEL_PATH=/mnt/models/Qwen2.5-VL-7B-Instruct \
TRAIN_FILE=/mnt/data/views/train_grpo.parquet \
VAL_FILE=/mnt/data/views/val_grpo.parquet \
CKPTS_DIR=/mnt/ckpts/ocr-vlm/grpo-smoke \
NNODES=2 \
NPUS_PER_NODE=8 \
RAY_ADDRESS=auto \
bash scripts/train/run_grpo_fsdp.sh
```

## SFT

VERL's installed SFT entrypoint is `verl.trainer.sft_trainer`. The included SFT
wrapper uses that path directly with `torchrun`, because that is the stable
entrypoint in the current installed VERL package.

```bash
MODEL_PATH=/mnt/models/Qwen2.5-VL-7B-Instruct \
TRAIN_FILE=/mnt/data/views/train_sft.parquet \
VAL_FILE=/mnt/data/views/val_sft.parquet \
CKPTS_DIR=/mnt/ckpts/ocr-vlm/sft-smoke \
bash scripts/train/run_sft.sh
```

## Checkpoint Metadata

Checkpoint files remain local or on mounted storage. Register metadata in W&B
as a reference artifact:

```bash
python -m verl_plugins.callbacks.save_and_eval register-checkpoint \
  --checkpoint-dir /mnt/ckpts/ocr-vlm/grpo-smoke/global_step_10 \
  --global-step 10 \
  --training-mode grpo \
  --model-id /mnt/models/Qwen2.5-VL-7B-Instruct \
  --config-hash "$(sha256sum configs/train/verl/rl/grpo_fsdp_vllm_910b.yaml | awk '{print $1}')" \
  --wandb-project ocr-vlm-training
```

The artifact uses `file://...` references and does not upload checkpoint
payloads.
