# OCR VLM Training Bootstrap

This repository is a thin integration layer around VERL for OCR VLM SFT and
GRPO experiments on Ascend/NPU infrastructure.

The project deliberately avoids reimplementing training. VERL owns distributed
training, rollout, checkpoint save/load, FSDP, vLLM/vLLM-Ascend, Megatron, and
MindSpeed integration. This repo provides:

- multi-granularity OCR data source schemas,
- exporters from source records to VERL-compatible parquet views,
- a temporary shared Normalized Levenshtein reward for all GRPO tasks,
- W&B reference-artifact helpers for local checkpoints,
- launch wrappers and config examples.

## Data Module

Source data is stored at its natural granularity:

- `PageRecord`: full page data for page markdown, layout, reading order, and page-level evaluation.
- `RegionRecord`: crop or patch data for text, table, formula, diagram, or seal recognition.
- `DetectionRecord`: object detection or classification data for pages, synthetic canvases, or isolated regions.

The source records are not forced into one physical page schema. Training uses
derived views:

- `sft`: emits `messages` and `images` for VERL SFT.
- `grpo`: emits `prompt`, `images`, `reward_model`, `data_source`, and `extra_info` for VERL GRPO.
- `layout`: emits image plus object instances for layout/detection training or later evaluation tooling.

MinerU-annotated datasets can be converted into canonical records directly. The
converter is generic; DocBank is just the first profile:

```bash
python -m src.data.mineru_export --profile configs/data_profiles/docbank_mineru.yaml
```

For a quick smoke export:

```bash
python -m src.data.mineru_export \
  --profile configs/data_profiles/docbank_mineru.yaml \
  --max-samples 10
```

By default, malformed annotation samples stop the export. Use `--skip-errors`
only when you want those samples listed in the manifest and omitted from the
canonical shards. `allow_unreadable_images` only controls missing or
permission-denied source images.

The profile writes sharded parquet under:

```text
/home/byhou/datasets/ocr-training/canonical/page_records/DocBank_500K/
/home/byhou/datasets/ocr-training/canonical/detection_records/DocBank_500K/
/home/byhou/datasets/ocr-training/canonical/region_records/DocBank_500K/
/home/byhou/datasets/ocr-training/canonical/_manifests/
```

Example:

```bash
python -m src.data.export_views \
  --input data/source/regions.jsonl \
  --view grpo \
  --output data/views/region_grpo.parquet
```

For larger canonical datasets, export sharded training views from canonical
directories instead of loading everything into one process:

```bash
python -m src.data.export_views \
  --input /home/byhou/datasets/ocr-training/canonical/page_records/DocBank_500K \
          /home/byhou/datasets/ocr-training/canonical/region_records/DocBank_500K \
  --view sft \
  --output-dir /home/byhou/datasets/ocr-training/views/DocBank_500K/sft \
  --shard-size 10000
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
python -m src.callbacks.save_and_eval register-checkpoint \
  --checkpoint-dir /mnt/ckpts/ocr-vlm/grpo-smoke/global_step_10 \
  --global-step 10 \
  --training-mode grpo \
  --model-id /mnt/models/Qwen2.5-VL-7B-Instruct \
  --config-hash "$(sha256sum configs/rl/grpo_fsdp_vllm_910b.yaml | awk '{print $1}')" \
  --wandb-project ocr-vlm-training
```

The artifact uses `file://...` references and does not upload checkpoint
payloads.
