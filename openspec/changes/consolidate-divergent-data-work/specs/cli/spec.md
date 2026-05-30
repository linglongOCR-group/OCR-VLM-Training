# Delta for CLI

## ADDED Requirements

### Requirement: View Deployment CLI

The repository SHALL provide a standalone CLI for staging a view and its referenced assets to one or more training nodes.

#### Scenario: Build transfer manifest

- GIVEN a dataset root containing a view under `views/<name>`
- WHEN `scripts/data/deploy_view_to_nodes --source-root <root> --view <name> --remote-root <remote> --node <target>` runs
- THEN the CLI SHALL build a dataset-relative manifest containing view files and all assets referenced by `images` or `images_path` parquet columns

#### Scenario: Rsync transfer

- GIVEN the default transfer mode
- WHEN the deployment CLI transfers a manifest to a target node
- THEN it SHALL create the remote root unless skipped
- AND it SHALL invoke `rsync` with `--files-from`, `--relative`, partial-transfer support, and the generated manifest

#### Scenario: Tar transfer

- GIVEN `--transfer-mode tar`
- WHEN the deployment CLI transfers a manifest to a target node
- THEN it SHALL stream `tar -cf - -T <manifest>` from the source root through `ssh`
- AND it SHALL extract into the remote root

#### Scenario: Dry run

- GIVEN `--dry-run`
- WHEN deployment is requested
- THEN the CLI SHALL build the manifest and print the transfer commands
- AND it SHALL NOT execute ssh, rsync, or tar subprocesses that mutate remote state

#### Scenario: Operator training overrides

- GIVEN manifest construction completes
- WHEN the CLI exits successfully
- THEN it SHALL print the remote `OCR_DATA_ROOT` value and staged view root path needed for training launchers

### Requirement: Deployment Observability and Controls

The deployment CLI SHALL expose controls for large-view manifest construction.

#### Scenario: Parallel manifest scan

- GIVEN a view with many parquet shards
- WHEN `--num-workers N` is provided with `N > 1`
- THEN parquet image-reference scanning SHALL use parallel workers

#### Scenario: Progress output

- GIVEN progress is enabled
- WHEN manifest construction scans view files and parquet shards
- THEN the CLI SHALL report phases and scan progress instead of remaining silent

#### Scenario: Asset existence check

- GIVEN `--check-assets`
- WHEN referenced image assets are missing under the source root
- THEN the CLI SHALL fail before transfer and report examples of missing assets
