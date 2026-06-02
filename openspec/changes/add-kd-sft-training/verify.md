# Verification Report

**Change**: `add-kd-sft-training`
**Verified at**: `2026-06-02 18:45 UTC`
**Verifier**: `Codex`

---

## 1. Structural Validation (`openspec validate --all --json`)

- [x] All items returned `"valid": true`

**Result**:

```text
openspec validate --all --json
summary: 14 items, 14 passed, 0 failed
changes: add-kd-sft-training valid; checkpoint-lineage-tracking valid
specs: 12 valid, 0 failed
```

| Item | Type | Issues |
|---|---|---|
| - | - | - |

---

## 2. Task Completion (`tasks.md`)

- [x] All `- [ ]` checkboxes have been converted to `- [x]`

**Unchecked task scan**:

```text
grep -n '^- [ ]' openspec/changes/add-kd-sft-training/tasks.md
(no output)
```

| Task | Unfinished reason | Blocks archive |
|---|---|---|
| - | - | - |

---

## 3. Delta Spec Sync State

| Capability | Sync state | Notes |
|---|---|---|
| `kd-sft-training` | N/A | New capability; no existing `openspec/specs/kd-sft-training/spec.md` yet. |
| `checkpoint-tracking` | needs sync | Delta spec differs from existing spec and should be applied during archive. |
| `training-bootstrap` | needs sync | Delta spec differs from existing spec and should be applied during archive. |
| `training-ops-deployment` | needs sync | Delta spec differs from existing spec and should be applied during archive. |

---

## 4. Design / Specs Coherence Spot Check

| Sample | design.md decision | specs correspondence | Drift |
|---|---|---|---|
| Co-located FSDP teacher/student | D2 requires rank-local teacher and student. | `FSDP Teacher Student Topology` scenarios cover same inputs, rank-local tensors, and no teacher optimizer/checkpoint state. | None |
| Top-k logits KD | D3 selects top-k KD with renormalized and truncated KL. | `Top-K Logits Distillation` scenarios cover top-k extraction, renormalized top-k KL, and truncated KL. | None |
| Explicit hidden layer map/hooks | D4 selects explicit maps and selected-layer hooks. | `Selected Layer Hidden Distillation` scenarios cover required map, mapped-only capture, supported losses, equal weighting. | None |
| Global-step schedules | D6 selects optimizer-step schedules. | `KD Loss Weight Schedules` scenarios cover deterministic schedule evaluation and microbatch stability. | None |
| Hydra-first launch | D7 selects KD-SFT config plus multinode wrapper. | `KD-SFT Hydra Configuration` and training-ops delta cover config/script launch surface. | None |

**Drift warnings**: none.

---

## 5. Implementation Signal

- [x] Worktree was clean before writing this `verify.md` artifact.
- [x] Implementation is committed locally.
- [ ] Pushed status not verified; no push was requested in this apply step.

**Commit range**: `origin/data-module..HEAD`

```text
745aa8b Add KD-SFT training mode
```

**Schema precheck note**: the schema command using `origin/main` / `origin/master` returned `0` because this repository branch is based on `origin/data-module` and no `origin/main` or `origin/master` merge-base exists in this worktree. The applicable branch-local evidence command was:

```text
git log --oneline origin/data-module..HEAD | wc -l
1
```

---

## 6. Front-Door Routing Leak Detector

Command:

```bash
ls docs/superpowers/specs/*.md 2>/dev/null
```

- [x] No files were reported.

| File | Captured in change | Recommended action |
|---|---|---|
| - | - | - |

---

## 7. Deferred Manual Dogfood vs Automated Test Equivalence

`plan.md` contains no `[~]` deferred rows.

The real NPU smoke remains intentionally manual per v1 acceptance and user direction. Local automated coverage includes:

| Manual / smoke concern | Equivalent automated test | Coverage assessment | Real gap? |
|---|---|---|---|
| KD loss composition before full training | `tests/test_kd_worker.py` | SFT/logits/hidden loss composition, response-only shifted masks, global-step lambdas, teacher target detach, empty-response reporting. | No for local composition; yes for hardware-specific FSDP/NPU execution. |
| KD config/precheck surface | `tests/test_kd_precheck.py` | Teacher/student paths, tokenizer/vocab/hidden/processor/output-token compatibility, top-k target semantics, dry-run callable contract. | No for config validation. |
| Launch/ops/checkpoint surface | `tests/test_train_scripts.py`, `tests/test_training_ops.py`, `tests/test_checkpoint_artifacts.py` | KD script, training-ops `kd_sft` dispatch, checkpoint metadata teacher/student provenance. | No for rendering/metadata. |

Hardware-specific FSDP teacher + FSDP student behavior on NPU is not automatically tested here; follow-up is the user-planned real NPU smoke.

---

## Verification Commands

```text
bash -n scripts/train/run_multinode_kd_sft.sh
python -m py_compile tools/training_ops/config.py tools/training_ops/launch.py tools/training_ops/cli.py verl_plugins/callbacks/save_and_eval.py verl_plugins/trainers/kd_sft_trainer.py verl_plugins/trainers/kd_engine.py verl_plugins/trainers/kd_worker.py
TEACHER_MODEL_PATH=/tmp/teacher MODEL_PATH=/tmp/student TRAIN_FILE=/tmp/train.parquet VAL_FILE=null PROJECT_ROOT=/tmp/ocr-vlm-tuning-add-kd-sft-training-impl OCR_DATA_ROOT=/tmp python - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml')
resolved = OmegaConf.to_container(cfg, resolve=True)
print(resolved['kd']['hidden']['layer_map'])
PY
pytest tests/test_kd_precheck.py tests/test_kd_worker.py tests/test_kd_losses.py tests/test_kd_schedules.py tests/test_kd_hidden_hooks.py tests/test_train_scripts.py tests/test_training_ops.py tests/test_checkpoint_artifacts.py tests/test_sft_freeze_vision.py tests/test_sft_position_ids_patch.py tests/test_sft_validation_flags.py -q
```

Latest pytest result:

```text
93 passed, 5 warnings in 19.92s
```

---

## Overall Decision

- [ ] PASS
- [x] PASS WITH WARNINGS - local tests and OpenSpec validation pass; real NPU FSDP smoke is intentionally deferred to the user; implementation commit is local and not pushed.
- [ ] FAIL

**Next step**: run retrospective, then archive/apply delta specs when ready. The user can run the real NPU KD-SFT smoke from `scripts/train/run_multinode_kd_sft.sh`.
