from __future__ import annotations

import argparse
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq

from tools.data_management.paths import DATA_ROOT_ENV, validate_relative_path
from tools.data_management.progress import ProgressReporter


@dataclass(frozen=True)
class DeployManifestReport:
    source_root: Path
    view_root: Path
    manifest_path: Path
    view_files: int
    image_references: int
    unique_files: int


def build_view_deploy_manifest(
    source_root: str | Path,
    view: str | Path,
    manifest_path: str | Path,
    *,
    batch_size: int = 8192,
    check_assets: bool = False,
    num_workers: int = 1,
    progress: ProgressReporter | None = None,
) -> DeployManifestReport:
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    source_root = Path(source_root).expanduser().resolve()
    view_root = resolve_view_root(source_root, view)
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    progress = progress or ProgressReporter.disabled()

    with tempfile.TemporaryDirectory(prefix="docds-deploy-manifest-") as db_tmp:
        with _ManifestStore(Path(db_tmp) / "manifest.sqlite3") as store:
            progress.log(
                "deploy-manifest",
                phase="start",
                source_root=source_root,
                view_root=view_root,
                manifest=manifest_path,
                num_workers=num_workers,
            )
            view_files = 0
            for file_path in sorted(path for path in view_root.rglob("*") if path.is_file()):
                if file_path.resolve() == manifest_path:
                    continue
                store.add(_dataset_relative(file_path, source_root))
                view_files += 1
            progress.log("deploy-manifest", phase="view-files", count=view_files)

            image_references = 0
            missing_assets: list[str] = []
            parquet_files = list(_iter_view_parquet_files(view_root))
            progress.log("deploy-manifest", phase="scan-images", parquet_files=len(parquet_files))
            for scanned, result in enumerate(
                _iter_parquet_reference_results(parquet_files, batch_size=batch_size, num_workers=num_workers),
                start=1,
            ):
                for image_reference in result.references:
                    store.add(image_reference)
                    if check_assets and not (source_root / image_reference).is_file():
                        missing_assets.append(image_reference)
                        if len(missing_assets) >= 10:
                            break
                image_references += result.image_references
                progress.update(
                    "deploy-manifest",
                    scanned,
                    total=len(parquet_files),
                    phase="scan-images",
                    file=result.path,
                    image_references=image_references,
                )
                if missing_assets:
                    break
            if missing_assets:
                examples = ", ".join(missing_assets[:5])
                raise FileNotFoundError(f"missing image assets under {source_root}: {examples}")

            progress.log("deploy-manifest", phase="write-manifest")
            unique_files = store.write(manifest_path)
            progress.finish(
                "deploy-manifest",
                total=unique_files,
                view_files=view_files,
                image_references=image_references,
                manifest=manifest_path,
            )

    return DeployManifestReport(
        source_root=source_root,
        view_root=view_root,
        manifest_path=manifest_path,
        view_files=view_files,
        image_references=image_references,
        unique_files=unique_files,
    )


def resolve_view_root(source_root: str | Path, view: str | Path) -> Path:
    source_root = Path(source_root).expanduser().resolve()
    view_path = Path(view).expanduser()
    if view_path.is_absolute():
        resolved = view_path.resolve()
    elif view_path.parts and view_path.parts[0] == "views":
        resolved = (source_root / view_path).resolve()
    else:
        resolved = (source_root / "views" / view_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"view does not exist: {resolved}")
    _dataset_relative(resolved, source_root)
    return resolved


def rsync_command_for_target(
    *,
    source_root: str | Path,
    manifest_path: str | Path,
    target: str,
    remote_root: str,
    ssh_options: Sequence[str] = (),
    rsync_options: Sequence[str] = (),
    dry_run: bool = False,
) -> list[str]:
    command = [
        "rsync",
        "-a",
        "--info=progress2",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--relative",
        "--files-from",
        str(Path(manifest_path).expanduser()),
    ]
    if dry_run:
        command.append("--dry-run")
    command.extend(rsync_options)
    if ssh_options:
        command.extend(["-e", shlex.join(["ssh", *ssh_options])])
    command.extend([_as_dir_arg(source_root), f"{target}:{_as_remote_dir_arg(remote_root)}"])
    return command


def tar_commands_for_target(
    *,
    source_root: str | Path,
    manifest_path: str | Path,
    target: str,
    remote_root: str,
    ssh_options: Sequence[str] = (),
    skip_mkdir: bool = False,
) -> tuple[list[str], list[str]]:
    tar_command = [
        "tar",
        "-C",
        str(Path(source_root).expanduser()),
        "-cf",
        "-",
        "-T",
        str(Path(manifest_path).expanduser()),
    ]
    remote_root_arg = shlex.quote(remote_root)
    remote_command = f"tar -C {remote_root_arg} -xf -"
    if not skip_mkdir:
        remote_command = f"mkdir -p {remote_root_arg} && {remote_command}"
    ssh_command = ["ssh", *ssh_options, target, remote_command]
    return tar_command, ssh_command


def ssh_mkdir_command(*, target: str, remote_root: str, ssh_options: Sequence[str] = ()) -> list[str]:
    return ["ssh", *ssh_options, target, f"mkdir -p {shlex.quote(remote_root)}"]


def deploy_view_to_targets(
    *,
    source_root: str | Path,
    view: str | Path,
    targets: Sequence[str],
    remote_root: str,
    manifest_path: str | Path,
    batch_size: int = 8192,
    check_assets: bool = False,
    ssh_options: Sequence[str] = (),
    rsync_options: Sequence[str] = (),
    transfer_mode: str = "rsync",
    dry_run: bool = False,
    skip_mkdir: bool = False,
    num_workers: int = 1,
    progress: ProgressReporter | None = None,
) -> DeployManifestReport:
    if transfer_mode not in {"rsync", "tar"}:
        raise ValueError(f"unsupported transfer mode: {transfer_mode}")
    report = build_view_deploy_manifest(
        source_root,
        view,
        manifest_path,
        batch_size=batch_size,
        check_assets=check_assets,
        num_workers=num_workers,
        progress=progress,
    )
    for target in targets:
        if transfer_mode == "rsync":
            if not skip_mkdir:
                _run(ssh_mkdir_command(target=target, remote_root=remote_root, ssh_options=ssh_options), dry_run=dry_run)
            _run(
                rsync_command_for_target(
                    source_root=report.source_root,
                    manifest_path=report.manifest_path,
                    target=target,
                    remote_root=remote_root,
                    ssh_options=ssh_options,
                    rsync_options=rsync_options,
                    dry_run=dry_run,
                ),
                dry_run=dry_run,
            )
        else:
            _run_tar_pipeline(
                *tar_commands_for_target(
                    source_root=report.source_root,
                    manifest_path=report.manifest_path,
                    target=target,
                    remote_root=remote_root,
                    ssh_options=ssh_options,
                    skip_mkdir=skip_mkdir,
                ),
                dry_run=dry_run,
            )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="deploy_view_to_nodes",
        description="Deploy a document dataset view and referenced assets to training nodes.",
    )
    parser.add_argument("--source-root", default=os.environ.get(DATA_ROOT_ENV), help=f"local {DATA_ROOT_ENV}")
    parser.add_argument("--view", required=True, help="view name, views/<name>, or absolute view root")
    parser.add_argument("--remote-root", required=True, help=f"remote {DATA_ROOT_ENV} to create/populate")
    parser.add_argument("--node", action="append", default=[], help="remote SSH target; repeatable")
    parser.add_argument("--nodes", help="comma-separated remote SSH targets")
    parser.add_argument("--nodes-file", help="file containing one remote SSH target per line")
    parser.add_argument("--manifest", help="path for the generated dataset-relative transfer manifest")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=1, help="parallel parquet scanners for manifest analysis")
    parser.add_argument("--check-assets", action="store_true", help="stat every referenced asset before transfer")
    parser.add_argument("--ssh-option", action="append", default=[], help="extra ssh option token; repeatable")
    parser.add_argument("--rsync-option", action="append", default=[], help="extra rsync option token; repeatable")
    parser.add_argument(
        "--transfer-mode",
        choices=("rsync", "tar"),
        default="rsync",
        help="transfer backend; tar streams immediately but is best for initial seeding",
    )
    parser.add_argument("--dry-run", action="store_true", help="print and dry-run transfer commands")
    parser.add_argument("--skip-mkdir", action="store_true", help="do not create remote root with ssh first")
    parser.add_argument("--progress", dest="progress", action="store_true", default=None)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.source_root:
        raise SystemExit(f"--source-root or {DATA_ROOT_ENV} is required")
    targets = _parse_targets(args.node, args.nodes, args.nodes_file)
    if not targets:
        raise SystemExit("at least one --node, --nodes entry, or --nodes-file entry is required")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be positive")
    if args.transfer_mode == "tar" and args.rsync_option:
        parser.error("--rsync-option is only valid with --transfer-mode rsync")

    manifest_path: Path
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="docds-deploy-")
        manifest_path = Path(temp_dir.name) / f"{Path(args.view).name}.files"
    try:
        report = deploy_view_to_targets(
            source_root=args.source_root,
            view=args.view,
            targets=targets,
            remote_root=args.remote_root,
            manifest_path=manifest_path,
            batch_size=args.batch_size,
            check_assets=args.check_assets,
            ssh_options=args.ssh_option,
            rsync_options=args.rsync_option,
            transfer_mode=args.transfer_mode,
            dry_run=args.dry_run,
            skip_mkdir=args.skip_mkdir,
            num_workers=args.num_workers,
            progress=_make_progress(args, root=Path(args.source_root)),
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    remote_view = _dataset_relative(report.view_root, report.source_root)
    print(f"manifest={report.manifest_path}")
    print(f"view_files={report.view_files}")
    print(f"image_references={report.image_references}")
    print(f"unique_files={report.unique_files}")
    print("remote training overrides:")
    print(f"  export {DATA_ROOT_ENV}={shlex.quote(args.remote_root)}")
    print(f"  REMOTE_VIEW_ROOT={shlex.quote(_as_remote_dir_arg(args.remote_root) + remote_view)}")
    print("  pass that view's train/*.parquet files to TRAIN_FILES as before")


@dataclass(frozen=True)
class _ParquetReferenceResult:
    path: str
    references: tuple[str, ...]
    image_references: int


def _iter_parquet_reference_results(
    parquet_files: Sequence[Path],
    *,
    batch_size: int,
    num_workers: int,
) -> Iterable[_ParquetReferenceResult]:
    if num_workers == 1:
        for parquet_path in parquet_files:
            yield _collect_parquet_image_references(str(parquet_path), batch_size)
        return
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(
            _collect_parquet_image_references,
            [str(path) for path in parquet_files],
            [batch_size] * len(parquet_files),
        )


def _collect_parquet_image_references(path: str, batch_size: int) -> _ParquetReferenceResult:
    references: set[str] = set()
    image_references = 0
    parquet = pq.ParquetFile(path)
    column_names = set(parquet.schema_arrow.names)
    columns = [column for column in ("images", "images_path") if column in column_names]
    if columns:
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            for row in batch.to_pylist():
                for image_reference in _row_image_references(row):
                    references.add(image_reference)
                    image_references += 1
    return _ParquetReferenceResult(
        path=path,
        references=tuple(sorted(references)),
        image_references=image_references,
    )


def _iter_view_parquet_files(view_root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for split in ("train", "val", "test"):
        split_dir = view_root / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.parquet")):
            seen.add(path)
            yield path
    for path in sorted(view_root.rglob("*.parquet")):
        if path not in seen:
            yield path


def _row_image_references(row: dict[str, Any]) -> Iterable[str]:
    for image_reference in _as_list(row.get("images_path")):
        if not isinstance(image_reference, str):
            raise ValueError(f"images_path entries must be strings, got {image_reference!r}")
        validate_relative_path(image_reference)
        yield image_reference
    for image in _as_list(row.get("images")):
        if not isinstance(image, dict) or not image.get("image"):
            raise ValueError(f"images entries must be dictionaries with image paths, got {image!r}")
        image_reference = image["image"]
        if not isinstance(image_reference, str):
            raise ValueError(f"images entries must contain string image paths, got {image!r}")
        validate_relative_path(image_reference)
        yield image_reference


def _parse_targets(nodes: Sequence[str], comma_nodes: str | None, nodes_file: str | None) -> list[str]:
    targets: list[str] = []
    targets.extend(nodes)
    if comma_nodes:
        targets.extend(node.strip() for node in comma_nodes.split(",") if node.strip())
    if nodes_file:
        targets.extend(
            line.strip()
            for line in Path(nodes_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return list(dict.fromkeys(targets))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
        if value is None:
            return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _dataset_relative(path: str | Path, source_root: Path) -> str:
    relative = Path(path).resolve().relative_to(source_root.resolve()).as_posix()
    validate_relative_path(relative)
    return relative


def _as_dir_arg(path: str | Path) -> str:
    text = str(Path(path).expanduser())
    return text.rstrip("/") + "/"


def _as_remote_dir_arg(path: str) -> str:
    return path.rstrip("/") + "/"


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print(shlex.join(command), file=sys.stderr)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _run_tar_pipeline(tar_command: Sequence[str], ssh_command: Sequence[str], *, dry_run: bool) -> None:
    print(f"{shlex.join(tar_command)} | {shlex.join(ssh_command)}", file=sys.stderr)
    if dry_run:
        return

    tar_process = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
    assert tar_process.stdout is not None
    try:
        ssh_process = subprocess.Popen(ssh_command, stdin=tar_process.stdout)
    finally:
        tar_process.stdout.close()

    ssh_return = ssh_process.wait()
    tar_return = tar_process.wait()
    if tar_return:
        raise subprocess.CalledProcessError(tar_return, tar_command)
    if ssh_return:
        raise subprocess.CalledProcessError(ssh_return, ssh_command)


def _make_progress(args: argparse.Namespace, *, root: Path | None) -> ProgressReporter:
    if args.quiet:
        return ProgressReporter.disabled()
    enabled = bool(args.progress) if args.progress is not None else sys.stderr.isatty()
    return ProgressReporter(enabled=enabled, log_every=args.log_every, root=root)


class _ManifestStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> "_ManifestStore":
        try:
            self.db_path.unlink()
        except FileNotFoundError:
            pass
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("create table manifest(path text primary key)")
        return self

    def __exit__(self, *exc: object) -> None:
        if self.conn is not None:
            self.conn.close()
        try:
            self.db_path.unlink()
        except FileNotFoundError:
            pass

    def add(self, path: str) -> None:
        validate_relative_path(path)
        assert self.conn is not None
        self.conn.execute("insert or ignore into manifest(path) values (?)", (path,))

    def write(self, manifest_path: Path) -> int:
        assert self.conn is not None
        cursor = self.conn.execute("select path from manifest order by path")
        count = 0
        with manifest_path.open("w") as handle:
            for (path,) in cursor:
                handle.write(path)
                handle.write("\n")
                count += 1
        return count


if __name__ == "__main__":
    main()
