from io import StringIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.data_management.deploy import (
    build_view_deploy_manifest,
    main as deploy_main,
    rsync_command_for_target,
    tar_commands_for_target,
)
from tools.data_management.progress import ProgressReporter


def test_build_view_deploy_manifest_includes_view_files_and_nested_assets(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    view_root = dataset_root / "views" / "sample_view"
    train_dir = view_root / "train"
    train_dir.mkdir(parents=True)
    (view_root / "stats.json").write_text("{}")
    (view_root / "view.yaml").write_text("name: sample_view\n")

    table = pa.Table.from_pylist(
        [
            {
                "id": "row-1",
                "images": [{"image": "canonical/assets/files/source=A/img-1.jpg"}],
                "images_path": None,
            },
            {
                "id": "row-2",
                "images": [{"image": "canonical/assets/files/source=A/img-1.jpg"}],
                "images_path": None,
            },
            {
                "id": "row-3",
                "images": [{"image": "sources/B/images/img-2.jpg"}],
                "images_path": None,
            },
        ]
    )
    pq.write_table(table, train_dir / "part-00000.parquet")

    manifest = tmp_path / "manifest.txt"
    report = build_view_deploy_manifest(dataset_root, view_root, manifest, batch_size=2)

    lines = manifest.read_text().splitlines()
    assert lines == sorted(
        [
            "canonical/assets/files/source=A/img-1.jpg",
            "sources/B/images/img-2.jpg",
            "views/sample_view/stats.json",
            "views/sample_view/train/part-00000.parquet",
            "views/sample_view/view.yaml",
        ]
    )
    assert report.view_files == 3
    assert report.image_references == 3
    assert report.unique_files == 5


def test_build_view_deploy_manifest_reports_progress_and_supports_workers(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    view_root = dataset_root / "views" / "sample_view"
    train_dir = view_root / "train"
    train_dir.mkdir(parents=True)
    for index in range(2):
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "id": f"row-{index}",
                        "images": [{"image": f"canonical/assets/files/source=A/img-{index}.jpg"}],
                    }
                ]
            ),
            train_dir / f"part-{index:05d}.parquet",
        )

    stream = StringIO()
    progress = ProgressReporter(enabled=True, log_every=1, stream=stream, force_tty=False)
    report = build_view_deploy_manifest(
        dataset_root,
        "sample_view",
        tmp_path / "manifest.txt",
        num_workers=2,
        progress=progress,
    )

    output = stream.getvalue()
    assert "deploy-manifest phase=start" in output
    assert "deploy-manifest" in output
    assert "phase=scan-images" in output
    assert "phase=done" in output
    assert report.image_references == 2


def test_build_view_deploy_manifest_excludes_manifest_scratch_files_inside_view(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    view_root = dataset_root / "views" / "sample_view"
    train_dir = view_root / "train"
    train_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": "row-1", "images": [{"image": "canonical/assets/files/source=A/img-1.jpg"}]}]
        ),
        train_dir / "part-00000.parquet",
    )

    manifest = view_root / "manifest.txt"
    build_view_deploy_manifest(dataset_root, "sample_view", manifest)

    lines = manifest.read_text().splitlines()
    assert "views/sample_view/manifest.txt" not in lines
    assert all(".sqlite3" not in line for line in lines)


def test_rsync_command_for_target_preserves_dataset_relative_layout(tmp_path: Path) -> None:
    command = rsync_command_for_target(
        source_root=Path("/data/root"),
        manifest_path=tmp_path / "manifest.txt",
        target="worker-0",
        remote_root="/local/DataMgmt",
        ssh_options=["-p", "2222"],
        rsync_options=["--delete"],
        dry_run=True,
    )

    assert command[:4] == ["rsync", "-a", "--info=progress2", "--partial"]
    assert "--dry-run" in command
    assert "--files-from" in command
    assert "-e" in command
    assert "ssh -p 2222" in command
    assert "/data/root/" in command
    assert "worker-0:/local/DataMgmt/" in command


def test_tar_commands_for_target_preserve_dataset_relative_layout(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    tar_command, ssh_command = tar_commands_for_target(
        source_root=Path("/data/root"),
        manifest_path=manifest,
        target="worker-0",
        remote_root="/local/DataMgmt",
        ssh_options=["-p", "2222"],
        skip_mkdir=False,
    )

    assert tar_command == ["tar", "-C", "/data/root", "-cf", "-", "-T", str(manifest)]
    assert ssh_command[:4] == ["ssh", "-p", "2222", "worker-0"]
    assert "mkdir -p /local/DataMgmt" in ssh_command[-1]
    assert "tar -C /local/DataMgmt -xf -" in ssh_command[-1]


def test_deploy_cli_dry_run_prints_commands_without_running_subprocess(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dataset_root = tmp_path / "dataset"
    view_root = dataset_root / "views" / "sample_view"
    train_dir = view_root / "train"
    train_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": "row-1", "images": [{"image": "canonical/assets/files/source=A/img-1.jpg"}]}]
        ),
        train_dir / "part-00000.parquet",
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not execute subprocesses")

    monkeypatch.setattr("tools.data_management.deploy.subprocess.run", fail_run)

    deploy_main(
        [
            "--source-root",
            str(dataset_root),
            "--view",
            "sample_view",
            "--remote-root",
            "/local/DataMgmt",
            "--node",
            "worker-0",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert "ssh worker-0 'mkdir -p /local/DataMgmt'" in captured.err
    assert "rsync" in captured.err
    assert "worker-0:/local/DataMgmt/" in captured.err
    assert "export OCR_DATA_ROOT=/local/DataMgmt" in captured.out
    assert "REMOTE_VIEW_ROOT=/local/DataMgmt/views/sample_view" in captured.out


def test_deploy_cli_tar_dry_run_prints_pipeline_without_running_subprocess(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dataset_root = tmp_path / "dataset"
    view_root = dataset_root / "views" / "sample_view"
    train_dir = view_root / "train"
    train_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": "row-1", "images": [{"image": "canonical/assets/files/source=A/img-1.jpg"}]}]
        ),
        train_dir / "part-00000.parquet",
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not execute subprocesses")

    monkeypatch.setattr("tools.data_management.deploy.subprocess.run", fail_run)
    monkeypatch.setattr("tools.data_management.deploy.subprocess.Popen", fail_run)

    deploy_main(
        [
            "--source-root",
            str(dataset_root),
            "--view",
            "sample_view",
            "--remote-root",
            "/local/DataMgmt",
            "--node",
            "worker-0",
            "--transfer-mode",
            "tar",
            "--ssh-option=-p",
            "--ssh-option=2222",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert "tar -C" in captured.err
    assert "| ssh -p 2222 worker-0" in captured.err
    assert "tar -C /local/DataMgmt -xf -" in captured.err
    assert "REMOTE_VIEW_ROOT=/local/DataMgmt/views/sample_view" in captured.out


def test_deploy_cli_rejects_rsync_options_in_tar_mode(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        deploy_main(
            [
                "--source-root",
                str(tmp_path),
                "--view",
                "sample_view",
                "--remote-root",
                "/local/DataMgmt",
                "--node",
                "worker-0",
                "--transfer-mode",
                "tar",
                "--rsync-option=--delete",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "--rsync-option is only valid with --transfer-mode rsync" in captured.err
