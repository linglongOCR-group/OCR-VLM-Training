from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from tools.training_ops.cli import main as trainops_main
from tools.training_ops.cleanup import plan_cleanup
from tools.training_ops.config import load_run_context
from tools.training_ops.deploy import deploy_shared, deploy_tmp
from tools.training_ops.errors import CommandExecutionError
from tools.training_ops.executor import Executor, FakeExecutor
from tools.training_ops.launch import build_grpo_launches, build_sft_launches, launch_metadata
from tools.training_ops.package import create_package
from tools.training_ops.preflight import run_preflight
from tools.training_ops.ray import build_ray_head_command, build_ray_worker_commands
from tools.training_ops.state import OpsState
from tools.training_ops.status import list_logs


def _write_inventory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "cluster": {
                    "name": "atlas-a2-test",
                    "ssh_user": "root",
                    "default_container": "verl-vlm-grpo",
                    "shared_paths": {
                        "code_root": "/mnt/shared/ocr-vlm-training",
                        "ops_state_root": "/mnt/shared/ocr-vlm-training-runs",
                    },
                    "nodes": [
                        {
                            "name": "node0",
                            "host": "atlas-a2-00",
                            "host_ip": "10.0.0.10",
                            "train_iface": "bond0",
                        },
                        {
                            "name": "node1",
                            "host": "atlas-a2-01",
                            "host_ip": "10.0.0.11",
                            "train_iface": "bond0",
                            "container": "custom-verl",
                        },
                    ],
                }
            },
            sort_keys=False,
        )
    )


def _write_run(path: Path, inventory: Path, *, mode: str = "grpo", deploy_method: str = "shared") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "id": f"{mode}-smoke",
                    "mode": mode,
                    "inventory": str(inventory),
                    "nodes": ["node0", "node1"],
                    "deployment": {
                        "method": deploy_method,
                        "release_root": "/mnt/shared/ocr-vlm-training/releases",
                        "current_link": "/mnt/shared/ocr-vlm-training/current",
                        "container_release_root": "/tmp/ocr-vlm-training/releases",
                    },
                    "paths": {
                        "model_path": "/mnt/models/MinerU2.5",
                        "train_file": "/mnt/data/train.parquet",
                        "val_file": "/mnt/data/val.parquet",
                        "ckpts_dir": "/mnt/ckpts/run",
                        "ocr_data_root": "/mnt/data",
                    },
                    "training": {
                        "env": {
                            "NPUS_PER_NODE": 8,
                            "WANDB_MODE": "offline",
                            "EXPERIMENT_NAME": f"{mode}-smoke",
                        },
                        "args": ["trainer.logger=[\"console\"]"],
                    },
                }
            },
            sort_keys=False,
        )
    )


def _write_four_node_inventory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "cluster": {
                    "name": "atlas-a2-four-node-test",
                    "ssh_user": "root",
                    "default_container": "verl-vlm-grpo",
                    "shared_paths": {
                        "code_root": "/mnt/shared/ocr-vlm-training",
                        "ops_state_root": "/mnt/shared/ocr-vlm-training-runs",
                    },
                    "nodes": [
                        {"name": f"node{index}", "host": f"atlas-a2-0{index}", "host_ip": f"10.0.0.1{index}", "train_iface": "bond0"}
                        for index in range(4)
                    ],
                }
            },
            sort_keys=False,
        )
    )


def _write_subset_run(
    path: Path,
    inventory: Path,
    *,
    mode: str = "sft",
    nodes: list[str] | None = None,
    head_node: str | None = None,
) -> None:
    _write_run(path, inventory, mode=mode)
    payload = yaml.safe_load(path.read_text())
    payload["run"]["nodes"] = nodes or ["node2", "node3"]
    if head_node is not None:
        payload["run"]["head_node"] = head_node
    run = payload["run"]
    run["paths"]["model_path"] = "/mnt/models/MinerU2.5"
    run["paths"]["train_file"] = "/mnt/data/train.parquet"
    run["paths"]["val_file"] = "/mnt/data/val.parquet"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_load_run_context_resolves_inventory_nodes_and_effective_env(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)

    context = load_run_context(run, repo_root=tmp_path)

    assert context.run_id == "grpo-smoke"
    assert [node.name for node in context.selected_nodes] == ["node0", "node1"]
    assert context.head_node.name == "node0"
    assert context.selected_nodes[1].container == "custom-verl"
    assert context.effective_env()["MODEL_PATH"] == "/mnt/models/MinerU2.5"
    assert context.effective_env()["NNODES"] == "2"
    assert context.effective_env()["RAY_ADDRESS"] == "auto"


def test_load_run_context_derives_run_ranks_from_selected_node_order(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_four_node_inventory(inventory)
    _write_subset_run(run, inventory, nodes=["node2", "node3"])

    context = load_run_context(run, repo_root=tmp_path)

    assert [node.name for node in context.selected_nodes] == ["node2", "node3"]
    assert [node.run_rank for node in context.selected_nodes] == [0, 1]
    assert context.head_node.name == "node2"


def test_load_run_context_accepts_selected_head_node_override(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_four_node_inventory(inventory)
    _write_subset_run(run, inventory, nodes=["node2", "node3"], head_node="node3")

    context = load_run_context(run, repo_root=tmp_path)

    assert context.head_node.name == "node3"
    assert context.head_node.run_rank == 1


@pytest.mark.parametrize(
    ("nodes", "head_node", "match"),
    [
        (["node2", "node2"], None, "duplicate"),
        (["node2", "missing"], None, "unknown"),
        (["node2", "node3"], "node1", "head_node"),
    ],
)
def test_load_run_context_validates_selected_nodes_and_head_node(
    tmp_path: Path,
    nodes: list[str],
    head_node: str | None,
    match: str,
) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_four_node_inventory(inventory)
    _write_subset_run(run, inventory, nodes=nodes, head_node=head_node)

    with pytest.raises(ValueError, match=match):
        load_run_context(run, repo_root=tmp_path)


def test_load_run_context_rejects_missing_required_fields(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)
    payload = yaml.safe_load(run.read_text())
    del payload["run"]["paths"]["model_path"]
    run.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="MODEL_PATH"):
        load_run_context(run, repo_root=tmp_path)


def test_ops_state_writes_snapshots_and_json_metadata(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)
    context = load_run_context(run, repo_root=tmp_path)

    state = OpsState.create(context, root=tmp_path / "ops")
    state.snapshot_configs()
    manifest = state.write_json("status.json", {"ok": True})

    assert (state.root / "run.yaml").is_file()
    assert (state.root / "inventory.yaml").is_file()
    assert (state.root / "effective-env.json").is_file()
    assert json.loads(manifest.read_text()) == {"ok": True}


def test_create_package_includes_dirty_files_and_excludes_generated_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    source.mkdir()
    (source / "tools").mkdir()
    (source / "tools" / "feature.py").write_text("print('local change')\n")
    (source / "notes.txt").write_text("untracked local note\n")
    (source / "checkpoints").mkdir()
    (source / "checkpoints" / "model.bin").write_text("large")
    (source / "wandb").mkdir()
    (source / "wandb" / "run.log").write_text("log")

    result = create_package(
        source_root=source,
        output_dir=tmp_path / "ops",
        release_id="rel-test",
        target_nodes=["node0"],
    )

    with tarfile.open(result.artifact_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "tools/feature.py" in names
    assert "notes.txt" in names
    assert "checkpoints/model.bin" not in names
    assert "wandb/run.log" not in names
    assert result.manifest["artifact_sha256"]


def test_create_package_rebuilds_same_release_after_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    source.mkdir()
    (source / "file.txt").write_text("payload\n")
    first = create_package(source_root=source, output_dir=tmp_path / "ops", release_id="rel-test")
    (source / "file.txt").write_text("updated\n")

    second = create_package(source_root=source, output_dir=tmp_path / "ops", release_id="rel-test")

    assert second.artifact_path == first.artifact_path
    assert second.manifest["artifact_sha256"] != first.manifest["artifact_sha256"]
    with tarfile.open(second.artifact_path, "r:gz") as archive:
        extracted = archive.extractfile("file.txt")
        assert extracted is not None
        assert extracted.read().decode() == "updated\n"


def test_create_package_excludes_artifacts_by_default(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    source.mkdir()
    (source / "source.py").write_text("ok\n")
    (source / "artifacts").mkdir()
    (source / "artifacts" / "generated.txt").write_text("generated\n")

    result = create_package(source_root=source, output_dir=tmp_path / "ops", release_id="rel-test")

    with tarfile.open(result.artifact_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "source.py" in names
    assert "artifacts/generated.txt" not in names


def test_fake_executor_records_host_and_container_commands(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)
    context = load_run_context(run, repo_root=tmp_path)
    state = OpsState.create(context, root=tmp_path / "ops")
    executor = FakeExecutor(state)

    executor.host(context.selected_nodes[0], "check", "true")
    executor.container(context.selected_nodes[0], "container-check", "python -V")

    assert executor.commands[0].argv[:3] == ["ssh", "root@atlas-a2-00", "true"]
    assert executor.commands[1].argv[:2] == ["ssh", "root@atlas-a2-00"]
    assert len(executor.commands[1].argv) == 3
    assert "docker exec verl-vlm-grpo bash -lc" in executor.commands[1].argv[2]
    assert (state.root / "commands.jsonl").is_file()


def test_executor_raises_on_failed_required_command(tmp_path: Path, monkeypatch) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)
    context = load_run_context(run, repo_root=tmp_path)
    state = OpsState.create(context, root=tmp_path / "ops")
    executor = Executor(state)

    class Completed:
        returncode = 23
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr("tools.training_ops.executor.subprocess.run", lambda *args, **kwargs: Completed())

    with pytest.raises(CommandExecutionError, match="required"):
        executor.host(context.head_node, "required", "false")

    result = executor.host(context.head_node, "warning", "false", allow_failure=True)
    assert result.record.exit_code == 23


def test_deploy_shared_builds_release_visibility_and_symlink_commands(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)
    context = load_run_context(run, repo_root=tmp_path)
    state = OpsState.create(context, root=tmp_path / "ops")
    package = create_package(source_root=tmp_path, output_dir=state.packages_dir, release_id="rel-test")
    executor = FakeExecutor(state)

    manifest = deploy_shared(context, package, state, executor)

    joined = "\n".join(record.command for record in executor.commands)
    assert "tar -xzf" in joined
    assert "mv -Tf" in joined
    assert "artifact_sha256" in joined
    assert "test -d /mnt/shared/ocr-vlm-training/releases/rel-test" in joined
    assert "docker exec" in joined
    assert manifest["target_path"] == "/mnt/shared/ocr-vlm-training/releases/rel-test"


def test_deploy_tmp_defaults_to_container_tar_streaming_and_not_docker_cp(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory, deploy_method="tmp")
    context = load_run_context(run, repo_root=tmp_path)
    state = OpsState.create(context, root=tmp_path / "ops")
    package = create_package(source_root=tmp_path, output_dir=state.packages_dir, release_id="rel-test")
    executor = FakeExecutor(state)

    manifest = deploy_tmp(context, package, state, executor)

    joined = "\n".join(record.command for record in executor.commands)
    assert "docker exec -i" in joined
    assert "tar -xzf -" in joined
    assert ".training-ops-release.json" in joined
    assert "docker cp" not in joined
    assert manifest["target_path"] == "/tmp/ocr-vlm-training/releases/rel-test"


def test_ray_and_launch_command_builders_render_expected_scripts(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)
    context = load_run_context(run, repo_root=tmp_path)

    assert "cd $PROJECT_ROOT && bash scripts/cluster/start_ray_head.sh" in build_ray_head_command(context).command
    workers = build_ray_worker_commands(context)
    assert len(workers) == 1
    assert "RAY_HEAD_ADDRESS=10.0.0.10:6379" in workers[0].command

    grpo = build_grpo_launches(context, project_root="/mnt/shared/ocr-vlm-training/current")
    assert len(grpo) == 1
    assert "cd $PROJECT_ROOT && bash scripts/train/run_grpo_fsdp.sh" in grpo[0].command
    assert "PROJECT_ROOT=/mnt/shared/ocr-vlm-training/current" in grpo[0].command
    assert "RAY_ADDRESS=auto" in grpo[0].command

    _write_run(run, inventory, mode="sft")
    sft_context = load_run_context(run, repo_root=tmp_path)
    sft = build_sft_launches(sft_context, project_root="/mnt/shared/ocr-vlm-training/current")
    assert len(sft) == 2
    assert "cd $PROJECT_ROOT && bash scripts/train/run_multinode_sft_new.sh" in sft[0].command
    assert "NODE_RANK=1" in sft[1].command
    assert "MASTER_ADDR=10.0.0.10" in sft[1].command


def test_subset_sft_launch_uses_contiguous_derived_run_ranks(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_four_node_inventory(inventory)
    _write_subset_run(run, inventory, mode="sft", nodes=["node2", "node3"])
    context = load_run_context(run, repo_root=tmp_path)

    sft = build_sft_launches(context, project_root="/mnt/shared/ocr-vlm-training/current")

    assert [command.node.name for command in sft] == ["node2", "node3"]
    assert "NODE_RANK=0" in sft[0].command
    assert "NODE_RANK=1" in sft[1].command
    assert "NNODES=2" in sft[0].command
    assert "NNODES=2" in sft[1].command
    assert "MASTER_ADDR=10.0.0.12" in sft[1].command


def test_grpo_and_ray_commands_do_not_render_node_rank(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_four_node_inventory(inventory)
    _write_subset_run(run, inventory, mode="grpo", nodes=["node2", "node3"])
    payload = yaml.safe_load(run.read_text())
    payload["run"]["training"]["env"]["NODE_RANK"] = 99
    run.write_text(yaml.safe_dump(payload, sort_keys=False))
    context = load_run_context(run, repo_root=tmp_path)

    commands = [
        build_ray_head_command(context).command,
        *[command.command for command in build_ray_worker_commands(context)],
        *[command.command for command in build_grpo_launches(context, project_root="/mnt/shared/ocr-vlm-training/current")],
    ]

    assert all("NODE_RANK" not in command for command in commands)
    assert "NODE_RANK" not in context.effective_env()
    assert "RAY_HEAD_ADDRESS=10.0.0.12:6379" in commands[1]
    assert launch_metadata(context, project_root="/mnt/shared/ocr-vlm-training/current")["run_ranks"] == {"node2": 0, "node3": 1}


def test_inventory_cli_outputs_selected_nodes_with_run_ranks(tmp_path: Path, capsys) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_four_node_inventory(inventory)
    _write_subset_run(run, inventory, mode="sft", nodes=["node2", "node3"])

    trainops_main(["--run", str(run), "--repo-root", str(tmp_path), "inventory"])

    output = json.loads(capsys.readouterr().out)
    assert output["nodes"] == [
        {
            "name": "node2",
            "host": "atlas-a2-02",
            "host_ip": "10.0.0.12",
            "train_iface": "bond0",
            "ssh_user": "root",
            "container": "verl-vlm-grpo",
            "run_rank": 0,
        },
        {
            "name": "node3",
            "host": "atlas-a2-03",
            "host_ip": "10.0.0.13",
            "train_iface": "bond0",
            "ssh_user": "root",
            "container": "verl-vlm-grpo",
            "run_rank": 1,
        },
    ]


def test_command_records_store_derived_run_rank_with_node_identity(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_four_node_inventory(inventory)
    _write_subset_run(run, inventory, mode="sft", nodes=["node2", "node3"])
    context = load_run_context(run, repo_root=tmp_path)
    state = OpsState.create(context, root=tmp_path / "ops")
    executor = FakeExecutor(state)

    executor.host(context.selected_nodes[1], "check", "true")

    record = executor.commands[0]
    assert record.node == "node3"
    assert record.host == "atlas-a2-03"
    assert record.run_rank == 1
    saved = json.loads((state.root / "commands.jsonl").read_text())
    assert saved["node"] == "node3"
    assert saved["run_rank"] == 1


def test_preflight_required_path_checks_are_strict_and_quoted(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)
    payload = yaml.safe_load(run.read_text())
    payload["run"]["paths"]["model_path"] = "/mnt/models/model with space"
    run.write_text(yaml.safe_dump(payload))
    context = load_run_context(run, repo_root=tmp_path)
    state = OpsState.create(context, root=tmp_path / "ops")
    executor = FakeExecutor(state)

    run_preflight(context, executor)

    commands = "\n".join(record.command for record in executor.commands)
    labels = "\n".join(record.label for record in executor.commands)
    runtime = next(record.command for record in executor.commands if record.label == "preflight-runtime")
    assert "npu-smi info >/dev/null 2>&1 || true" not in runtime
    assert "npu-smi info >/dev/null 2>&1" in runtime
    assert "/mnt/models/model with space" in commands
    assert "preflight-path" in labels
    assert "preflight-active-processes" in labels


def test_stop_run_is_scoped_and_does_not_allow_remote_failure(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    state_root = tmp_path / "state"
    _write_inventory(inventory)
    _write_run(run, inventory)

    trainops_main(["--run", str(run), "--repo-root", str(tmp_path), "--state-root", str(state_root), "--dry-run", "stop", "--target", "run"])

    records = (state_root / "grpo-smoke" / "commands.jsonl").read_text()
    assert "TRAINOPS_RUN_ID=grpo-smoke" in records
    assert "pkill -f grpo-smoke || true" not in records


def test_background_launch_metadata_registers_training_log_paths(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    state_root = tmp_path / "state"
    _write_inventory(inventory)
    _write_run(run, inventory)

    trainops_main(
        [
            "--run",
            str(run),
            "--repo-root",
            str(tmp_path),
            "--state-root",
            str(state_root),
            "--dry-run",
            "launch",
            "grpo",
            "--background",
            "--project-root",
            "/workspace/release",
        ]
    )

    launch = json.loads((state_root / "grpo-smoke" / "launch-metadata.json").read_text())
    assert launch["background_log_paths"] == ["/workspace/release/training-grpo.log"]
    assert "/workspace/release/training-grpo.log" in [str(path) for path in list_logs(state_root / "grpo-smoke")]


def test_cleanup_plan_preserves_current_release_by_default(tmp_path: Path) -> None:
    release_root = tmp_path / "releases"
    old_release = release_root / "old"
    current_release = release_root / "current-target"
    old_release.mkdir(parents=True)
    current_release.mkdir()
    current_link = tmp_path / "current"
    current_link.symlink_to(current_release)

    plan = plan_cleanup(release_root=release_root, current_link=current_link, dry_run=True)

    assert old_release in plan.candidates
    assert current_release not in plan.candidates
    assert plan.dry_run is True


def test_cli_dry_run_status_records_commands_without_live_ssh(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    state_root = tmp_path / "state"
    _write_inventory(inventory)
    _write_run(run, inventory)

    trainops_main(["--run", str(run), "--repo-root", str(tmp_path), "--state-root", str(state_root), "--dry-run", "status"])

    status = json.loads((state_root / "grpo-smoke" / "status.json").read_text())
    assert status["run_id"] == "grpo-smoke"
    commands = (state_root / "grpo-smoke" / "commands.jsonl").read_text()
    assert "docker exec" in commands


def test_cli_stop_requires_explicit_target(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.yaml"
    run = tmp_path / "run.yaml"
    _write_inventory(inventory)
    _write_run(run, inventory)

    with pytest.raises(SystemExit):
        trainops_main(["--run", str(run), "stop"])
