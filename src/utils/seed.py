"""Deterministic seeding helpers for reproducibility across runs."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, PyTorch (CPU + CUDA) and hash randomization.

    Args:
        seed: Non-negative integer seed value applied to every RNG.

    Notes:
        True bit-exact reproducibility on GPU is not guaranteed due to
        non-deterministic CUDA kernels (e.g. atomic adds). Small numeric drift
        across runs with the same seed is expected and documented in spec 06.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch must be installed at runtime
        logger.warning("PyTorch not available; skipping torch seeding.")


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process deterministically.

    Args:
        worker_id: Worker index supplied by ``torch.utils.data.DataLoader``.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def log_environment_info(extra: Optional[dict] = None) -> dict:
    """Collect and log a snapshot of the runtime environment.

    Args:
        extra: Optional extra fields to merge into the snapshot.

    Returns:
        Dictionary describing the active Python / library / GPU environment.
    """
    import platform
    import sys

    info: dict = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch_version"] = "unavailable"

    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "unavailable"

    if extra:
        info.update(extra)

    logger.info("Environment snapshot: %s", info)
    return info
