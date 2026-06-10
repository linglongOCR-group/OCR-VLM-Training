## Context

Current GRPO reward execution enters `verl_plugins.rewards.aggregate.compute_score(data_source, solution_str, ground_truth, extra_info, reward_version, **kwargs)`. That path currently computes normalized Levenshtein for every task and uses `reward_version` only as returned provenance metadata. The data-management reward adapter registry is more structured, but the active GRPO launch path does not use it directly.

Formula tasks need a reward closer to OmniDocBench CDM than string edit distance. The OmniDocBench CDM reference algorithm tokenizes and normalizes LaTeX, renders token-colored formulas, extracts per-token bounding boxes, matches reference and prediction tokens with Hungarian assignment over token, position, and order costs, filters outliers with RANSAC, and reports F1 as the scalar score. The reference implementation is useful for algorithm semantics, but its file-based batch pipeline and module layout are not suitable for online GRPO reward serving.

The accepted v1 scope is online-first: a CDM-enabled GRPO run must be trainable end-to-end, not merely smoke-tested offline.

## Goals / Non-Goals

**Goals:**

- Support complete GRPO training with Formula samples scored by CDM.
- Preserve existing Levenshtein-only GRPO behavior when no routing config is provided.
- Introduce explicit runtime reward routing/profile config instead of overloading `reward_version`.
- Add a Python CDM client and a real local Node.js LTS + Playwright/Chromium CDM service.
- Fail fast before training when CDM is configured but the local service is unavailable or incompatible.
- Return zero reward plus diagnostics for per-sample CDM render/score failures.
- Keep the first runtime input contract LaTeX-only: prediction from `solution_str`, reference from `ground_truth`, task from `extra_info.task_type` or `data_source`.

**Non-Goals:**

- Implement Table TEDS in v1.
- Pass rich OmniDocBench annotation metadata into online rewards in v1.
- Reproduce the official XeLaTeX CDM evaluation pipeline online.
- Switch VERL to a batch reward manager in v1.
- Support Bun as the v1 browser-rendering runtime.
- Automatically orchestrate CDM services across cluster nodes.

## Decisions

### Decision: Use runtime reward routing profiles

Runtime reward kwargs will support a mixed profile shape:

```yaml
reward_kwargs:
  reward_profile: mixed_doc_rlvr_v1
  routing:
    default: normalized_levenshtein_v1
    by_task:
      formula: cdm_katex_v1
  rewards:
    normalized_levenshtein_v1:
      type: normalized_levenshtein
      version: levenshtein_v1
    cdm_katex_v1:
      type: cdm_latex_render
      version: cdm_katex_v1
      service_url: ${oc.env:CDM_REWARD_URL,http://127.0.0.1:8765}
      timeout_ms: 3000
      fail_score: 0.0
      preflight_required: true
```

The routing key selects a reward profile ID. Each profile references a reward type and version. The returned `reward_version` remains metric/provenance metadata and is not itself the selector.

Alternatives considered:

- Reuse `reward_version` for dispatch: rejected because current code treats it as metadata, and overloading it would make future Table TEDS routing unclear.
- Hard-code `formula` dispatch in `aggregate.py`: rejected because it would not scale to future `table: teds_v1` routing.

### Decision: Keep `compute_score` as the VERL entrypoint

The public VERL custom reward function remains `compute_score(...)`. Internally it determines task type, selects a profile, invokes the matching runtime reward implementation, and returns a dict containing `score`, `reward_total`, `reward_name`, `reward_version`, `reward_profile_id`, task metadata, and reward-specific diagnostics.

This minimizes migration risk for existing GRPO scripts while enabling a more structured reward layer.

### Decision: Put CDM service behind a Python HTTP client

The Python runtime reward implementation will call a local CDM HTTP service instead of embedding browser rendering in Python reward code. The client owns timeout handling, response validation, score normalization, and failure-to-zero conversion.

Alternatives considered:

- Python-managed Node subprocess: rejected for v1 because browser lifecycle, logs, restarts, and multi-node behavior are more reliable when the service is explicit.
- Pure Python parser-only approximation: rejected because v1 should align Formula reward with rendered CDM behavior.

### Decision: Implement a real local Node.js LTS + Playwright/Chromium service in v1

The service exposes `GET /health`, `POST /score`, and a batch-shaped `POST /score_batch` endpoint or internal path. It owns browser/page lifecycle, KaTeX rendering, token/bbox extraction, CDM matching, caching, service version reporting, and structured error responses.

Node.js LTS is selected because Playwright’s official support path is Node.js, and the browser/layout path is expected to dominate runtime cost. Bun remains a future benchmark candidate for parser-only or pure string-processing components.

### Decision: Preserve CDM algorithm semantics, not reference module design

The implementation should preserve the algorithmic shape from OmniDocBench CDM:

1. Normalize/tokenize LaTeX.
2. Render identifiable per-token visual units.
3. Extract token bounding boxes from rendered output.
4. Match reference and prediction tokens with token, position, and order costs.
5. Filter geometric outliers.
6. Use F1 over matched tokens as the scalar reward.

The online service may implement rendering with KaTeX/Chromium rather than the reference file-based XeLaTeX/ImageMagick path, as long as responses expose enough diagnostics to evaluate correlation later.

### Decision: Separate preflight failure from per-sample failure

If CDM is configured and health/scoring preflight fails, launch fails before training. If an individual sampled prediction fails to parse/render/score during training, the reward returns `0.0` with compact diagnostics and does not crash the GRPO run.

## Risks / Trade-offs

- CDM service latency bottlenecks training → Use persistent Chromium pages, caching, bounded timeouts, and batch-shaped service internals; defer VERL batch manager migration until the scalar path works.
- Renderer unavailable in NPU containers → Add explicit preflight and document service startup requirements for CDM-enabled runs.
- Online KaTeX/Chromium CDM diverges from official CDM → Treat v1 as online reward approximation and keep official/evaluation parity as a later calibration task.
- Service crashes mid-training → Per-sample calls return zero plus connection diagnostics; operators can monitor service logs and restart externally.
- Reward stacks diverge further → Place shared runtime reward semantics in small modules and keep result metadata consistent with existing reward adapter concepts.
- Diagnostics become too large for reward logs → Store compact fields in reward dict and keep verbose service traces in service logs.

## Migration Plan

1. Add runtime registry/routing code with Levenshtein default behavior and tests.
2. Add CDM client with mocked HTTP tests.
3. Add local CDM service with health and scoring endpoints.
4. Add CDM-enabled GRPO config/script knobs and preflight.
5. Add optional real-service integration tests gated by environment variables.
6. Run existing reward/data/training tests to verify legacy behavior is unchanged.

Rollback strategy: use existing Levenshtein-only configs or remove the routing block from `reward_kwargs`. Because CDM is opt-in, legacy GRPO launches should continue to use normalized Levenshtein without starting the service.

## Open Questions

- The default CDM port should be finalized before implementation; `8765` is the proposed default.
- The v1 public API should include `/score_batch` if implementation effort is modest; otherwise the internals should remain batch-shaped and `/score_batch` can be completed immediately after scalar training works.
- Documentation location should be finalized during implementation; the service likely needs its own README plus GRPO launch notes.
