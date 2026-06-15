"""Top-level training entry point.

Dispatches to the appropriate trainer based on ``config.training.strategy``.
Sequential and joint trainers ship in later phases; this module wires them up
behind the same CLI surface so the bash scripts and the notebook stay stable.

Usage:

    python train.py --config configs/config_3_pubmedbert.yaml --seed 42

Optional CLI overrides: ``--output_dir``, ``--epochs``, ``--batch_size``,
``--learning_rate``, ``--smoke_test``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict

from src.config import ExperimentConfig, load_config
from src.utils.logging import configure_logging
from src.utils.seed import log_environment_info

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BioNER experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file (e.g. configs/config_3_pubmedbert.yaml).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed in YAML.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output base directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-device batch size.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run a tiny iteration over a sample of the data to validate the pipeline.",
    )
    return parser.parse_args()


def _select_trainer(strategy: str) -> Callable[..., Dict[str, Dict[str, Any]]]:
    """Map a strategy name to its trainer entry point."""
    if strategy == "single":
        from src.training.trainer_single import train_single

        return train_single
    if strategy == "sequential":
        from src.training.trainer_sequential import train_sequential  # pragma: no cover (Phase 4)

        return train_sequential
    if strategy in {"joint_uniform", "joint_noise_aware"}:
        from src.training.trainer_joint import train_joint  # pragma: no cover (Phase 4)

        return train_joint
    raise ValueError(f"Unknown training strategy {strategy!r}.")


def main() -> int:
    args = _parse_args()
    overrides = {
        "seed": args.seed,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    config: ExperimentConfig = load_config(args.config, overrides=overrides)
    seed = args.seed if args.seed is not None else config.seed
    log_dir = Path(config.output.base_dir) / "logs" / config.experiment.name
    log_file = configure_logging(log_dir, level="INFO", run_name=f"seed_{seed}")

    logger.info("Resolved config: %s", args.config)
    logger.info("Seed: %d", seed)
    logger.info("Strategy: %s", config.training.strategy)
    log_environment_info(extra={"config": args.config, "log_file": str(log_file)})

    trainer_fn = _select_trainer(config.training.strategy)
    trainer_kwargs: Dict[str, Any] = {"config": config, "seed": seed}
    if args.smoke_test:
        trainer_kwargs["smoke_test"] = True

    results = trainer_fn(**trainer_kwargs)
    logger.info("Training complete. Test-set summary: %s", _summarize(results))
    return 0


def _summarize(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Distil per-test-set results to overall F1 for the closing log line."""
    summary: Dict[str, Dict[str, float]] = {}
    for name, payload in results.items():
        overall = payload.get("overall", {})
        summary[name] = {
            "f1": float(overall.get("f1", 0.0)),
            "precision": float(overall.get("precision", 0.0)),
            "recall": float(overall.get("recall", 0.0)),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
