## 1. Runtime Reward Routing

- [x] 1.1 Add runtime reward profile parsing for `routing.default`, `routing.by_task`, and `rewards` blocks while preserving legacy `reward_version` behavior
- [x] 1.2 Add runtime reward type registry for `normalized_levenshtein` and `cdm_latex_render`
- [x] 1.3 Refactor `verl_plugins.rewards.aggregate.compute_score` to select reward profiles by task and return consistent `score`, `reward_total`, `reward_name`, `reward_version`, and `reward_profile_id` fields
- [x] 1.4 Add clear configuration errors for missing reward profiles and unsupported reward types
- [x] 1.5 Add unit tests proving legacy Levenshtein-only configs still work unchanged
- [x] 1.6 Add unit tests for Formula-to-CDM routing, default routing, unknown task fallback, and metadata fields

## 2. Python CDM Client

- [x] 2.1 Implement a Python CDM HTTP client with configurable service URL, timeout, fail score, and expected version
- [x] 2.2 Implement service health/preflight checks for `/health` and a tiny scoring probe
- [x] 2.3 Implement `/score` request handling for prediction/reference LaTeX pairs
- [x] 2.4 Validate and normalize service responses, including score range checks and compact diagnostics
- [x] 2.5 Convert connection failures, timeouts, malformed JSON, and malformed score responses into zero-reward diagnostic results during sample scoring
- [x] 2.6 Add mocked client tests for health success, wrong version, browser-not-ready, score success, structured render error, HTTP error, timeout, malformed JSON, missing score, and out-of-range score

## 3. CDM Service

- [ ] 3.1 Create the local Node.js service package structure for the CDM LaTeX render service
- [ ] 3.2 Add Node.js package metadata and dependencies for the HTTP server, Playwright/Chromium, KaTeX, and test tooling
- [ ] 3.3 Implement `GET /health` with service name, service version, renderer identifier, and browser readiness
- [ ] 3.4 Implement LaTeX normalization/tokenization suitable for the v1 CDM service, informed by the OmniDocBench CDM algorithm but not coupled to its module layout
- [ ] 3.5 Implement token-identifiable KaTeX/Chromium rendering and bounding-box extraction with bounded timeouts
- [ ] 3.6 Implement CDM matching using token identity, token position, token order, geometric outlier filtering, and F1 scoring
- [ ] 3.7 Implement `POST /score` returning normalized score and diagnostics for render status, parse status, timeout status, and fallback usage
- [ ] 3.8 Implement `POST /score_batch` or a public-compatible batch path that preserves input item identifiers
- [ ] 3.9 Add service-side caching for repeated formulas or clearly document the v1 cache boundary if only minimal caching is implemented
- [ ] 3.10 Add Node service tests for health, identical formula scoring, different formula scoring, invalid LaTeX failure response, and batch item mapping

## 4. GRPO Configuration and Launch Preflight

- [ ] 4.1 Add an opt-in CDM-enabled GRPO reward config or script configuration using explicit runtime routing profiles
- [ ] 4.2 Add environment/config support for `CDM_REWARD_URL`, timeout, fail score, expected version, and preflight-required behavior
- [ ] 4.3 Add GRPO launch preflight that detects CDM reward routes and checks CDM service health before training
- [ ] 4.4 Add GRPO launch preflight scoring probe for a small known Formula pair
- [ ] 4.5 Ensure Levenshtein-only GRPO launches do not require the CDM service
- [ ] 4.6 Add tests for CDM-enabled preflight success, CDM preflight failure, and Levenshtein-only launch compatibility

## 5. End-to-End Validation

- [ ] 5.1 Add optional real-service integration tests gated by an environment variable so normal CI does not require Playwright/Chromium
- [ ] 5.2 Verify a CDM-enabled reward call returns `reward_name = cdm_latex_render` for Formula samples and normalized Levenshtein for non-Formula samples
- [ ] 5.3 Verify invalid model LaTeX returns `0.0` with diagnostics instead of raising during runtime scoring
- [ ] 5.4 Run existing reward, data-pipeline, and GRPO configuration tests to confirm backward compatibility
- [ ] 5.5 Document how to start the CDM service locally and how to run a CDM-enabled GRPO launch on each training node
