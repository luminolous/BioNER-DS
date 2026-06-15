"""Logging configuration for training and inference runs."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


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
