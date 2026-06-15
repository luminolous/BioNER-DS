"""Custom exceptions used across the BioNER pipeline."""

from __future__ import annotations


class ConfigValidationError(ValueError):
    """Raised when a YAML experiment config fails validation."""


class DataValidationError(ValueError):
    """Raised when a dataset file violates the expected schema or label space."""


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be located or loaded."""
