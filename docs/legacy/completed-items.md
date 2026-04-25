# Completed Items

Date: 2026-04-21

## Data Model And Conversion

- Implemented multi-granularity canonical records:
  - `PageRecord`
  - `RegionRecord`
  - `DetectionRecord`
- Added canonical validation and parquet round-trip support for records loaded from pandas/PyArrow.
- Implemented a generic MinerU annotation exporter, not DocBank-specific.
- Added a DocBank MinerU profile at `configs/data_profiles/docbank_mineru.yaml`.
- Exported DocBank MinerU annotations into canonical parquet records under:
  - `/home/byhou/datasets/ocr-training/canonical/page_records/DocBank_500K`
  - `/home/byhou/datasets/ocr-training/canonical/detection_records/DocBank_500K`
  - `/home/byhou/datasets/ocr-training/canonical/region_records/DocBank_500K`
- Canonical export counts:
  - Page records: 374,833
  - Detection records: 374,833
  - Region records: 5,705,857
  - Errors: 0

## Training Views

- Extended the view exporter to support sharded parquet output from canonical record directories.
- Exported DocBank training views under `/home/byhou/datasets/ocr-training/views/DocBank_500K`.
- Exported views:
  - `sft`: 6,080,690 rows, 609 shards
  - `grpo`: 6,080,690 rows, 609 shards
  - `layout`: 374,833 rows, 38 shards
- View inputs:
  - `sft`: page records + region records
  - `grpo`: page records + region records
  - `layout`: detection records
- Representative parquet readback succeeded for all three views.

## VERL Training Scaffold

- Added VERL-oriented training entry scripts for SFT and GRPO.
- Added initial configuration files for:
  - SFT FSDP
  - GRPO FSDP + vLLM
  - GRPO Megatron/vLLM
  - GRPO MindSpeed path
- Added runtime helpers for environment checks, Ray setup, and NPU tuning.
- Added reward modules and aggregation for OCR/document tasks.
- Added callback scaffolding for checkpoint artifact metadata and evaluation hooks.

## Verification

Fresh verification was run after the final code changes:

```bash
python -m pytest tests -q
```

Result:

```text
19 passed in 0.77s
```

Additional checks:

```bash
python -m compileall -q src
python -c "import glob, yaml; [yaml.safe_load(open(p)) for p in glob.glob('configs/**/*.yaml', recursive=True)]; print('yaml ok')"
```

Results:

```text
compileall exit code: 0
yaml ok
```

## VERL Qwen2.5-VL-3B GRPO NPU Environment Check

Date: 2026-04-21

Goal: run the upstream VERL Qwen2.5-VL-3B GRPO NPU example path to verify the installed Ray, VERL, vLLM-Ascend, torch, and torch_npu environment.

Relevant upstream script:

- `/verl/examples/grpo_trainer/run_qwen2_5_vl_3b_npu.sh`
- `/verl/tests/special_npu/nightly_ci_ascend/run_grpo_qwen25-vl-3b-instruct_fsdp_npu.sh`

Local assets staged for the check:

- Model: `/tmp/Qwen2.5-VL-3B-Instruct`
- One-row smoke data: `/tmp/verl-qwen25vl-3b-geo3k-smoke`
- Eight-row smoke data: `/tmp/verl-qwen25vl-3b-geo3k-smoke8`

Environment versions observed:

- `verl`: `0.8.0.dev0`
- `ray`: `2.55.0`
- `torch`: `2.8.0`
- `torch_npu`: `2.8.0`
- `vllm`: `0.13.0+empty`
- `transformers`: `4.57.6`

Execution summary:

- Downloaded/staged `Qwen/Qwen2.5-VL-3B-Instruct`.
- Created Geo3K-shaped multimodal parquet smoke data with matching `<image>` prompt placeholders.
- Verified Ray must advertise an explicit `NPU` resource for VERL placement groups.
- One-NPU Ray run reached dataset loading and worker placement, but failed vLLM-Ascend initialization due to one-card memory pressure when actor/ref/vLLM were colocated.
- Restarted Ray as a full single-node resource pool with `8 GPU` and `8 NPU`.
- Full-node run reached:
  - Ascend device detection and `trainer.device=npu`.
  - Ray connection and 8 worker scheduling.
  - Train/validation parquet load and prompt filtering.
  - Qwen2.5-VL checkpoint shard loading across workers.
  - FSDP initialization across 8 ranks.
  - vLLM-Ascend server startup.
  - Rollout generation and transition into old log-prob computation.

Final result:

The environment is partially functional for VERL/Qwen2.5-VL/vLLM-Ascend on NPU, but the GRPO smoke run did not complete. The final full-node failure occurs during actor old log-prob computation:

```text
RuntimeError: The size of tensor a (167) must match the size of tensor b (4)
at non-singleton dimension 2
...
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
apply_multimodal_rotary_pos_emb
```

Interpretation:

- This reproduces the same mRoPE/log-prob shape mismatch seen in earlier OCR GRPO smoke runs.
- The failure occurs after vLLM rollout startup and generation, inside VERL actor/FSDP log-prob calculation.
- The issue is therefore not specific to the OCR canonical dataset or OCR reward wrapper.
- The likely compatibility boundary is the installed combination of VERL, Transformers `4.57.6`, Qwen2.5-VL, and torch_npu during multimodal rotary position embedding in actor log-prob computation.

Cleanup:

- Ray was stopped after the run.
- No active `verl.trainer.main_ppo`, vLLM, EngineCore, or Ray training processes remained after cleanup.
