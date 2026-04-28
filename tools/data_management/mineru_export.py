from __future__ import annotations

from pathlib import Path

from tools.data_management.sources.adapters.mineru import MinerUExportOptions, MinerUSourceAdapter


def load_options_from_profile(path: str | Path) -> MinerUExportOptions:
    return MinerUSourceAdapter.from_profile(path).options


def export_mineru_dataset(options: MinerUExportOptions):
    if options.output_root is None:
        raise ValueError("MinerUExportOptions.output_root is required")
    return MinerUSourceAdapter(options).export(options.output_root)


def main() -> None:
    from tools.data_management.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
