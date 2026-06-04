from pathlib import Path

from verl_plugins.callbacks.save_and_eval import CheckpointArtifactMetadata, register_checkpoint_reference


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.references = []

    def add_reference(self, uri):
        self.references.append(uri)


class FakeRun:
    def __init__(self):
        self.logged = []

    def log_artifact(self, artifact, aliases=None):
        self.logged.append((artifact, aliases))


class FakeWandb:
    def __init__(self):
        self.artifacts = []

    def Artifact(self, name, type, metadata):
        artifact = FakeArtifact(name, type, metadata)
        self.artifacts.append(artifact)
        return artifact


def test_checkpoint_registration_uses_file_reference(tmp_path):
    ckpt_dir = tmp_path / "global_step_10"
    ckpt_dir.mkdir()
    marker = ckpt_dir / "complete.marker"
    marker.write_text("ok")
    run = FakeRun()
    wandb_mod = FakeWandb()
    metadata = CheckpointArtifactMetadata(
        checkpoint_name="global_step_10",
        checkpoint_dir=ckpt_dir,
        global_step=10,
        training_mode="grpo",
        model_id="Qwen/Qwen2.5-VL",
        config_hash="abc123",
        git_commit="unknown",
    )

    artifact = register_checkpoint_reference(
        run=run,
        metadata=metadata,
        wandb_module=wandb_mod,
        aliases=["latest", "step-10"],
    )

    assert artifact.references == [f"file://{ckpt_dir.resolve()}"]
    assert artifact.metadata["checkpoint_uri"] == f"file://{ckpt_dir.resolve()}"
    assert artifact.metadata["optimizer_state_included"] is True
    assert run.logged == [(artifact, ["latest", "step-10"])]


def test_checkpoint_registration_rejects_missing_directory(tmp_path):
    metadata = CheckpointArtifactMetadata(
        checkpoint_name="missing",
        checkpoint_dir=Path(tmp_path / "missing"),
        global_step=1,
        training_mode="sft",
        model_id="model",
        config_hash="hash",
        git_commit="unknown",
    )

    try:
        register_checkpoint_reference(run=FakeRun(), metadata=metadata, wandb_module=FakeWandb())
    except FileNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing checkpoint directory should fail")


def test_kd_sft_checkpoint_metadata_records_teacher_student_sources(tmp_path):
    ckpt_dir = tmp_path / "global_step_20"
    ckpt_dir.mkdir()
    metadata = CheckpointArtifactMetadata(
        checkpoint_name="global_step_20",
        checkpoint_dir=ckpt_dir,
        global_step=20,
        training_mode="kd_sft",
        model_id="student-model",
        config_hash="hash",
        git_commit="commit",
        student_source="/mnt/models/student",
        teacher_source="/mnt/models/teacher",
    )

    payload = metadata.to_wandb_metadata()

    assert payload["training_mode"] == "kd_sft"
    assert payload["student_source"] == "/mnt/models/student"
    assert payload["teacher_source"] == "/mnt/models/teacher"
    assert payload["local_checkpoint_uri"] == f"file://{ckpt_dir.resolve()}"
