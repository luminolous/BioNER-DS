"""Logging configuration for training and inference runs."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional


def _silence_known_warnings() -> None:
    """Filter advisory warnings that clutter run output without being actionable.

    Each suppressed warning is intentionally narrow so unexpected variants of
    the same library still surface. Revisit when the upstream APIs change.
    """
    # HuggingFace Hub spams a one-time symlink advisory on Windows; harmless.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # `transformers.Trainer(tokenizer=...)` is deprecated in favor of
    # `processing_class=`. We will migrate when HF 5.0 is the stable line; until
    # then the warning fires every Trainer construction.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*tokenizer.*deprecated.*",
    )

    # Xet storage backend advisory from huggingface_hub; falls back to HTTP.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*Xet Storage.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*hf_xet.*",
    )

    # Lingering symlink advisory for older huggingface_hub on Windows.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*cache-system uses symlinks.*",
    )


def configure_logging(
    output_dir: str | Path,
    level: str = "INFO",
    run_name: Optional[str] = None,
) -> Path:
    """Configure root logging to write to both a file and the console.

    Args:
        output_dir: Directory where the log file will be written.
        level: Logging level name (e.g., ``"INFO"`` or ``"DEBUG"``).
        run_name: Optional run identifier inserted into the log file name.

    Returns:
        Path to the log file that was created.
    """
    _silence_known_warnings()

    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    log_file = log_dir / f"run_{stamp}{suffix}.log"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    return log_file
