from __future__ import annotations

from pathlib import Path

from tools.data_management.views import ViewBuilder


def build_view(view_config: str | Path):
    builder = ViewBuilder.from_config_path(view_config)
    return builder.build(view_config)


def main() -> None:
    from tools.data_management.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
