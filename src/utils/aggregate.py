"""Multi-seed aggregation: per-config mean ± std reporting.

Reads every ``eval_<test_set>.json`` written under
``outputs/results/<config_name>/seed_<seed>/`` and produces:

* ``aggregated_results.json`` — machine-readable mean/std/values per metric.
* ``aggregated_results.md`` — paper-ready Markdown table (one section per
  test set, overall + per-entity).

Can be invoked as a module:

    python -m src.utils.aggregate --config configs/config_3_pubmedbert.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import load_config

logger = logging.getLogger(__name__)


def _compute_mean_std(values: List[float]) -> Dict[str, float]:
    """Return mean / std / raw list for a numeric column."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "values": [float(v) for v in arr.tolist()],
    }


def _aggregate_overall(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate the ``overall`` block across seeds."""
    out: Dict[str, Dict[str, float]] = {}
    for metric in ("precision", "recall", "f1"):
        values = [float(run["overall"].get(metric, 0.0)) for run in runs]
        out[metric] = _compute_mean_std(values)
    return out


def _aggregate_per_entity(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate the ``per_entity`` block across seeds."""
    entity_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    supports: Dict[str, List[int]] = defaultdict(list)

    for run in runs:
        for entity, metrics in run.get("per_entity", {}).items():
            for key in ("precision", "recall", "f1"):
                entity_metrics[entity][key].append(float(metrics.get(key, 0.0)))
            supports[entity].append(int(metrics.get("support", 0)))

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for entity, metrics in entity_metrics.items():
        aggregated[entity] = {key: _compute_mean_std(values) for key, values in metrics.items()}
        aggregated[entity]["support"] = {
            "mean": float(np.mean(supports[entity])) if supports[entity] else 0.0,
            "values": supports[entity],
        }
    return aggregated


def aggregate_seeds(
    config_name: str,
    base_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    """Aggregate all evaluation JSONs for one experiment.

    Args:
        config_name: ``experiment.name`` from the YAML (e.g. ``"config_3_pubmedbert"``).
        base_dir: Outputs root (defaults to ``"outputs"``).

    Returns:
        Dict keyed by test-set name with the structured aggregated payload.

    Raises:
        FileNotFoundError: If no seed directories with eval JSONs exist.
    """
    results_dir = Path(base_dir) / "results" / config_name
    if not results_dir.exists():
        raise FileNotFoundError(
            f"No results directory for config {config_name!r} at {results_dir}."
        )

    seed_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories under {results_dir}.")

    per_test_set: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seed_dir in seed_dirs:
        for eval_file in sorted(seed_dir.glob("eval_*.json")):
            with eval_file.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            per_test_set[payload["test_set"]].append(payload)

    if not per_test_set:
        raise FileNotFoundError(
            f"No eval_*.json files found under {results_dir} (checked {len(seed_dirs)} seeds)."
        )

    aggregated: Dict[str, Any] = {"config_name": config_name, "test_sets": {}}
    for test_set, runs in per_test_set.items():
        aggregated["test_sets"][test_set] = {
            "num_seeds": len(runs),
            "seeds": [int(r["seed"]) for r in runs],
            "overall": _aggregate_overall(runs),
            "per_entity": _aggregate_per_entity(runs),
        }

    json_path = results_dir / "aggregated_results.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(aggregated, fp, indent=2)
    logger.info("Wrote aggregated JSON to %s", json_path)

    md_path = results_dir / "aggregated_results.md"
    md_path.write_text(_format_markdown_table(aggregated), encoding="utf-8")
    logger.info("Wrote aggregated Markdown to %s", md_path)

    return aggregated


def _format_metric(stat: Dict[str, float], digits: int = 3) -> str:
    """Format a mean ± std cell for the Markdown table."""
    mean = stat.get("mean", 0.0)
    std = stat.get("std", 0.0)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _format_markdown_table(aggregated: Dict[str, Any]) -> str:
    """Render the aggregated payload into a paper-friendly Markdown document."""
    lines: List[str] = []
    lines.append(f"# Config: {aggregated['config_name']}")
    lines.append("")

    for test_set, payload in aggregated["test_sets"].items():
        lines.append(f"## Test Set: {test_set}")
        lines.append("")
        lines.append(f"Aggregated over {payload['num_seeds']} seeds: {payload['seeds']}")
        lines.append("")
        lines.append("| Metric | Mean ± Std |")
        lines.append("|---|---|")
        for metric in ("precision", "recall", "f1"):
            stat = payload["overall"][metric]
            lines.append(f"| Overall {metric.capitalize()} | {_format_metric(stat, 4)} |")
        lines.append("")

        if payload["per_entity"]:
            lines.append("### Per-Entity")
            lines.append("")
            lines.append("| Entity | Precision | Recall | F1 | Support |")
            lines.append("|---|---|---|---|---|")
            for entity, metrics in payload["per_entity"].items():
                support = metrics.get("support", {})
                support_mean = support.get("mean", 0.0) if isinstance(support, dict) else 0.0
                lines.append(
                    "| {entity} | {p} | {r} | {f1} | {sup:.0f} |".format(
                        entity=entity,
                        p=_format_metric(metrics["precision"]),
                        r=_format_metric(metrics["recall"]),
                        f1=_format_metric(metrics["f1"]),
                        sup=support_mean,
                    )
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _resolve_config_name(arg: Optional[str], config_name: Optional[str]) -> str:
    """Return ``config_name`` directly, or derive it from a YAML path."""
    if config_name:
        return config_name
    if arg:
        return load_config(arg).experiment.name
    raise ValueError("Either --config or --config_name must be provided.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed evaluation results.")
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML config used during training.")
    parser.add_argument("--config_name", type=str, default=None, help="Experiment name (skip YAML lookup).")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Outputs root directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config_name = _resolve_config_name(args.config, args.config_name)
    aggregate_seeds(config_name, base_dir=args.base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
