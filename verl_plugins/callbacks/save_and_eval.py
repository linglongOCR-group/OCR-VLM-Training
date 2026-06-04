from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CheckpointArtifactMetadata:
    checkpoint_name: str
    checkpoint_dir: Path
    global_step: int
    training_mode: str
    model_id: str
    config_hash: str
    git_commit: str
    student_source: str | None = None
    teacher_source: str | None = None
    local_checkpoint_uri: str | None = None
    epoch: int | None = None
    optimizer_state_included: bool = True
    trainer_state_included: bool = True
    resume_compatibility_version: str = "verl-bootstrap-v1"
    validation_metrics: dict[str, Any] | None = None

    def to_wandb_metadata(self) -> dict[str, Any]:
        checkpoint_dir = self.checkpoint_dir.resolve()
        metadata = asdict(self)
        metadata["checkpoint_dir"] = str(checkpoint_dir)
        metadata["checkpoint_uri"] = f"file://{checkpoint_dir}"
        metadata["local_checkpoint_uri"] = metadata["local_checkpoint_uri"] or metadata["checkpoint_uri"]
        metadata["save_timestamp"] = datetime.now(UTC).isoformat()
        return metadata


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit(default: str = "unknown") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return default
    return result.stdout.strip() or default


def register_checkpoint_reference(
    *,
    run: Any,
    metadata: CheckpointArtifactMetadata,
    wandb_module: Any,
    aliases: list[str] | None = None,
) -> Any:
    checkpoint_dir = metadata.checkpoint_dir
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")

    artifact = wandb_module.Artifact(
        name=metadata.checkpoint_name,
        type="model-checkpoint",
        metadata=metadata.to_wandb_metadata(),
    )
    artifact.add_reference(f"file://{checkpoint_dir.resolve()}")
    run.log_artifact(artifact, aliases=aliases or ["latest", f"step-{metadata.global_step}"])
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Register local checkpoint metadata as a W&B reference artifact.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register-checkpoint")
    register.add_argument("--checkpoint-dir", required=True)
    register.add_argument("--checkpoint-name")
    register.add_argument("--global-step", type=int, required=True)
    register.add_argument("--training-mode", choices=["sft", "grpo", "kd_sft"], required=True)
    register.add_argument("--model-id", required=True)
    register.add_argument("--config-hash", required=True)
    register.add_argument("--student-source", default=None)
    register.add_argument("--teacher-source", default=None)
    register.add_argument("--git-commit", default=None)
    register.add_argument("--wandb-project", required=True)
    register.add_argument("--wandb-entity", default=None)
    register.add_argument("--wandb-run-id", default=None)
    args = parser.parse_args()

    if args.command == "register-checkpoint":
        import wandb

        checkpoint_dir = Path(args.checkpoint_dir)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            job_type="checkpoint-registration",
        )
        metadata = CheckpointArtifactMetadata(
            checkpoint_name=args.checkpoint_name or checkpoint_dir.name,
            checkpoint_dir=checkpoint_dir,
            global_step=args.global_step,
            training_mode=args.training_mode,
            model_id=args.model_id,
            config_hash=args.config_hash,
            student_source=args.student_source,
            teacher_source=args.teacher_source,
            git_commit=args.git_commit or git_commit(),
        )
        register_checkpoint_reference(run=run, metadata=metadata, wandb_module=wandb)
        run.finish()


if __name__ == "__main__":
    main()
