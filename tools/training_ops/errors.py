from __future__ import annotations


class TrainingOpsError(RuntimeError):
    """Base exception for training-ops failures."""


class ConfigError(TrainingOpsError, ValueError):
    """Raised when inventory or run configuration is invalid."""


class DeploymentError(TrainingOpsError):
    """Raised when deployment cannot be completed safely."""


class CommandExecutionError(TrainingOpsError):
    """Raised when a required remote command exits with a non-zero status."""
