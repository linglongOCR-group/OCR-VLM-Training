# Retrospective: add-kd-sft-training

> Written: 2026-06-02 (after verify passed)
> Commit range: `origin/data-module..HEAD`
> Worktree: `/tmp/ocr-vlm-tuning-add-kd-sft-training-impl`

---

## 0. Evidence

- **Commit range**: `origin/data-module..HEAD` (2 commits at write-time)
- **Diff size**: +4184 / -16 lines across 32 files
- **Tasks done**: 28/28 (`grep -cE '^\s*- \[x\]' tasks.md` -> 28)
- **Active hours**: one implementation session on 2026-06-02
- **Subagent dispatches**: 4 material dispatches (Task 1 worker, Task 1 spec reviewer, Task 1 code reviewer, Task 2 worker) plus Task 2 reviewers
- **New external dependencies**: none
- **Bugs encountered post-merge**: none; not merged yet
- **OpenSpec validate state at archive**: not archived yet; verify-time `openspec validate --all --json` passed 14/14 items
- **Test coverage signal**: final focused pytest `93 passed, 5 warnings in 19.92s`; shell syntax `bash -n scripts/train/run_multinode_kd_sft.sh`; py_compile for modified integration files

Commit chain:

```text
745aa8b Add KD-SFT training mode
0583982 Add KD-SFT verification report
```

---

## 1. Wins

- The requirements converged into a concrete OpenSpec change before implementation. Evidence: `openspec/changes/add-kd-sft-training/{brainstorm.md,proposal.md,design.md,plan.md}` plus 28/28 task completion.
- KD utility behavior was isolated and reviewed before trainer integration. Evidence: `tests/test_kd_schedules.py`, `tests/test_kd_losses.py`, `tests/test_kd_hidden_hooks.py`; Task 1 reviewer found teacher-detach and mask-broadcast defects that were fixed before downstream use.
- The implementation preserves the existing SFT path by using new KD-specific modules and entrypoints. Evidence: `verl_plugins/trainers/kd_sft_trainer.py`, `verl_plugins/trainers/kd_worker.py`, `scripts/train/run_multinode_kd_sft.sh`; existing SFT regression tests are in the final 93-test command.
- Ops integration was covered by existing test surfaces rather than left as script-only behavior. Evidence: `tests/test_train_scripts.py`, `tests/test_training_ops.py`, `tests/test_checkpoint_artifacts.py`.

## 2. Misses

- 🟡 [painful] The worktree was first created under `/root/.config/superpowers/worktrees`, outside the writable roots for `apply_patch`. Evidence: manual/scripted edits were needed until the worktree was moved to `/tmp/ocr-vlm-tuning-add-kd-sft-training-impl`.
- 🟡 [painful] The schema verify precheck assumes `origin/main` or `origin/master`, but this repo branch is based on `origin/data-module`. Evidence: `verify.md` records the hard-coded precheck returning `0` while `git log --oneline origin/data-module..HEAD | wc -l` returned `1` before the verify commit.
- 🟡 [painful] The first Hydra layer-map default used `oc.decode` in a way OmegaConf could not parse. Evidence: config smoke failed until `configs/train/verl/sft/kd_qwen2_5_vl_fsdp.yaml` switched to a normal YAML default list.
- 📌 [nit] Importing VERL tests is slow/noisy because it loads NPU and Megatron helpers even for unit-only tests. Evidence: final pytest warning block from `torch_npu`, `mindspeed`, and Ray state imports.

## 3. Plan deviations

| Plan task | What changed | Why |
|-----------|--------------|-----|
| 5.7 tiny/local trainer smoke | Recorded as one-batch KD dry-run/local smoke rather than automated real FSDP NPU smoke | User had explicitly chosen one-batch dry-run/local verification and planned to run real NPU smoke manually. |
| Verify commit precheck | Used `origin/data-module..HEAD` instead of `origin/main/master..HEAD` | This repository's active base branch is `data-module`; no `origin/main` or `origin/master` merge-base exists. |
| Hydra config layer map | Defaulted to a normal YAML list with env-overridable indices instead of `oc.decode` of a whole env list | OmegaConf parsing rejected the nested `oc.decode` default; whole-list flexibility remains available through Hydra CLI/script overrides. |

## 4. Skill / workflow compliance

| Skill                                            | Used |
|--------------------------------------------------|------|
| superpowers:brainstorming                        | ✓ |
| superpowers:writing-plans                        | ✓ |
| superpowers:using-git-worktrees                  | ✓ |
| superpowers:subagent-driven-development          | ✓ |
| (transitive) superpowers:test-driven-development | ✓ |
| (transitive) superpowers:requesting-code-review  | ✓ |
| superpowers:finishing-a-development-branch       | ✓ |

### Deliberately Skipped Skills

None. All apply-phase skills were used directly or through the OpenSpec artifact workflow and subagent review loop.

## 5. Surprises

- The code-quality review of Task 1 exposed that teacher-detach should be defensive inside KD loss helpers, not only assumed from `torch.no_grad()` in the trainer.
- The top-k extractor needed loss-type-aware semantics: `renormalized_top_k_forward_kl` should normalize over the top-k slice, while `truncated_forward_kl` needs teacher log-probs from the full vocabulary denominator.
- `OmegaConf.load` accepts unresolved required env variables until access/resolve time, so the config smoke had to provide `TEACHER_MODEL_PATH`, `MODEL_PATH`, `TRAIN_FILE`, and `PROJECT_ROOT` before checking full resolution.

## 6. Promote candidates → long-term learning

- [ ] 🟡 **Worktrees must be created under writable roots when apply_patch is expected** → **Promote to project CLAUDE.md**
  > **Why**: Creating the initial worktree under `/root/.config/superpowers/worktrees` made normal patch tooling fail in this sandbox.
  > **How to apply**: When sandbox writable roots are explicit, choose a worktree path under one of them, such as `/tmp`, before starting implementation.

- [ ] 🟡 **OpenSpec verify should support non-main base branches** → **Promote to schema**
  > **Why**: The schema precheck hard-coded `origin/main` / `origin/master`, while this repo's current base is `origin/data-module`.
  > **How to apply**: In repositories with feature branches based on non-main integration branches, verify should derive the upstream base from branch tracking metadata or accept an override.

- [ ] 📌 **Hydra list defaults should prefer YAML-native structures over nested resolver strings** → **Promote to memory**
  > **Why**: The `oc.decode` default for `kd.hidden.layer_map` failed OmegaConf grammar parsing.
  > **How to apply**: For structured defaults in repo configs, use YAML lists/dicts and reserve resolver strings for scalar values or externally validated CLI overrides.
