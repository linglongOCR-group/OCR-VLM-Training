# CDM LaTeX Render Service

Local Node.js service for the `cdm_latex_render` runtime reward profile. It exposes:

- `GET /health` with service metadata, version, renderer identifier, browser readiness, and cache stats.
- `POST /score` with `{ "prediction": "...", "reference": "..." }` and a normalized CDM-style score.
- `POST /score_batch` with `{ "items": [{ "id": "...", "prediction": "...", "reference": "..." }] }`; results preserve each input `id`.

## Runtime

The package targets Node.js LTS and lists KaTeX plus Playwright dependencies. The renderer loads them lazily, renders normalized LaTeX through KaTeX, opens Chromium through Playwright, and extracts token bounding boxes from `[data-token-index]` DOM spans with `getBoundingClientRect()` under the request timeout. `/health` performs and caches a lightweight Chromium launch readiness probe, so a fresh service reports `browser_ready: true` when KaTeX, Playwright, and a launchable browser are available before any scoring request. If KaTeX, Playwright, or a launchable browser are unavailable in the current container, `/health` reports `browser_ready: false` and scoring uses a deterministic token pseudo-layout fallback. This keeps unit tests and Python reward-client development runnable without network installs while still using the real KaTeX/Chromium path when runtime dependencies are ready.

```bash
cd services/cdm_latex_render
npm test
npm start
```

The default port is `8765`; override with `PORT` or `CDM_LATEX_RENDER_PORT`.

## Local service startup

From the repository root, install the service dependencies once and start the service on each host that will score Formula rewards:

```bash
cd services/cdm_latex_render
npm install
npx playwright install chromium  # optional when the container already has a launchable Chromium
PORT=8765 npm start
```

Confirm readiness before training:

```bash
curl -fsS http://127.0.0.1:8765/health
curl -fsS -X POST http://127.0.0.1:8765/score \
  -H 'Content-Type: application/json' \
  -d '{"prediction":"\\\\frac{1}{2}","reference":"\\\\frac{1}{2}"}'
```

`/health` should report `browser_ready: true` for the real Chromium scoring path. If the service is run without Playwright/Chromium, it remains useful for unit tests and development fallback behavior, but launch preflight for CDM-enabled GRPO treats `browser_ready: false` as not ready.

## CDM-enabled GRPO launch

Run one CDM service per training node and point that node's GRPO process at its local service. The launch script only enables CDM routing when `REWARD_PROFILE=formula_cdm_v1`; otherwise it keeps the Levenshtein-only path and skips CDM preflight.

On every training node, start the service and export the same reward env vars before invoking `scripts/train/run_grpo_fsdp.sh`:

```bash
export CDM_REWARD_URL=http://127.0.0.1:8765
export CDM_REWARD_TIMEOUT_MS=1000
export CDM_REWARD_FAIL_SCORE=0.0
export CDM_REWARD_EXPECTED_VERSION=cdm_katex_v1
export CDM_REWARD_PREFLIGHT_REQUIRED=True
export REWARD_PROFILE=formula_cdm_v1
export DEFAULT_REWARD_PROFILE=normalized_levenshtein_v1
export CDM_REWARD_PROFILE=cdm_katex_v1

MODEL_PATH=/mnt/models/MinerU2.5 \
TRAIN_FILE=/mnt/data/views/grpo/train.parquet \
VAL_FILE=/mnt/data/views/grpo/val.parquet \
NNODES=<cluster_node_count> \
NPUS_PER_NODE=8 \
RAY_ADDRESS=<head-node-ip>:6379 \
bash scripts/train/run_grpo_fsdp.sh
```

For `tools/training_ops` run files, include the same env block under `training.env` for each node or release. Example values:

```yaml
training:
  env:
    REWARD_PROFILE: formula_cdm_v1
    DEFAULT_REWARD_PROFILE: normalized_levenshtein_v1
    CDM_REWARD_PROFILE: cdm_katex_v1
    CDM_REWARD_URL: http://127.0.0.1:8765
    CDM_REWARD_TIMEOUT_MS: 1000
    CDM_REWARD_FAIL_SCORE: 0.0
    CDM_REWARD_EXPECTED_VERSION: cdm_katex_v1
    CDM_REWARD_PREFLIGHT_REQUIRED: "True"
```

To run the optional real-service Python integration test after the service is up:

```bash
CDM_REWARD_INTEGRATION=1 CDM_REWARD_URL=http://127.0.0.1:8765 \
  python -m pytest tests/test_cdm_reward_integration.py -q
```

## v1 cache boundary

The service maintains a small in-memory render cache keyed by the raw formula string. It caches normalized tokens and extracted renderer boxes, including deterministic pseudo-layout boxes when fallback is used, for repeated formulas within one Node process. The cache is intentionally local-only: it is not shared across worker processes, has no disk persistence, and is cleared on service restart. This boundary is sufficient for repeated online GRPO samples in v1 and can be replaced by a browser-page/render-result cache if longer-lived browser/page pooling is added.
