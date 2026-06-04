# Delta for Checkpoint Tracking

## ADDED Requirements

### Requirement: Lineage DAG via WandB Artifacts

The system SHALL build a checkpoint derivation DAG using WandB's native artifact consumption and production graph.

#### Scenario: SFT followed by GRPO

- GIVEN a seed model registered as `ocr-vlm/base:seed`
- WHEN an SFT run consumes the seed and produces `ocr-vlm/sft-v1:step-500`
- AND a GRPO run consumes `ocr-vlm/sft-v1:latest` and produces `ocr-vlm/grpo-v2:step-200`
- THEN `trace_lineage("ocr-vlm/grpo-v2:step-200")` SHALL return the chain from `grpo-v2:step-200` through the GRPO run to `sft-v1:latest` through the SFT run to `base:seed`

### Requirement: Lineage Tracing

The system SHALL support tracing the derivation chain from any checkpoint back to the root seed model.

#### Scenario: Trace to root

- GIVEN a trained checkpoint artifact
- WHEN `trace_lineage(artifact_id)` is called
- THEN the system SHALL return a list of `(artifact_id, training_mode, run_id)` tuples from the given checkpoint to the root seed model
