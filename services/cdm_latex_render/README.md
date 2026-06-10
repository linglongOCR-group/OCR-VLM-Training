# CDM LaTeX Render Service

Local Node.js service for the `cdm_latex_render` runtime reward profile. It exposes:

- `GET /health` with service metadata, version, renderer identifier, browser readiness, and cache stats.
- `POST /score` with `{ "prediction": "...", "reference": "..." }` and a normalized CDM-style score.
- `POST /score_batch` with `{ "items": [{ "id": "...", "prediction": "...", "reference": "..." }] }`; results preserve each input `id`.

## Runtime

The package targets Node.js LTS and lists KaTeX plus Playwright dependencies. The renderer loads them lazily, renders normalized LaTeX through KaTeX, opens Chromium through Playwright, and extracts token bounding boxes from `[data-token-index]` DOM spans with `getBoundingClientRect()` under the request timeout. If KaTeX, Playwright, or a launchable browser are unavailable in the current container, `/health` reports `browser_ready: false` and scoring uses a deterministic token pseudo-layout fallback. This keeps unit tests and Python reward-client development runnable without network installs while still using the real KaTeX/Chromium path when runtime dependencies are ready.

```bash
cd services/cdm_latex_render
npm test
npm start
```

The default port is `8765`; override with `PORT` or `CDM_LATEX_RENDER_PORT`.

## v1 cache boundary

The service maintains a small in-memory render cache keyed by the raw formula string. It caches normalized tokens and extracted renderer boxes, including deterministic pseudo-layout boxes when fallback is used, for repeated formulas within one Node process. The cache is intentionally local-only: it is not shared across worker processes, has no disk persistence, and is cleared on service restart. This boundary is sufficient for repeated online GRPO samples in v1 and can be replaced by a browser-page/render-result cache if longer-lived browser/page pooling is added.
