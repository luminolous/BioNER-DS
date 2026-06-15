"""Shared helpers used by every training strategy.

Centralises:

* ``TrainingArguments`` construction from an :class:`ExperimentConfig`.
* Output directory layout enforcement (checkpoints / logs / results).
* Snapshotting the resolved YAML config next to the run results.
* Final test-set evaluation + JSON serialization.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.config import ExperimentConfig, dump_config_snapshot
from src.data.loaders import build_test_datasets
from src.evaluation.metrics import (
    compute_metrics_factory,
    evaluate_predictions,
    save_evaluation_json,
)

logger = logging.getLogger(__name__)


def build_run_dirs(config: ExperimentConfig, seed: int) -> Dict[str, Path]:
    """Materialize the output directory layout described in spec 01.

    Returns a dict with keys ``checkpoint``, ``results``, and ``logs`` whose
    values are :class:`pathlib.Path` objects guaranteed to exist.
    """
    base = Path(config.output.base_dir)
    name = config.experiment.name
    dirs = {
        "checkpoint": base / "checkpoints" / name / f"seed_{seed}",
        "results": base / "results" / name / f"seed_{seed}",
        "logs": base / "logs" / name,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_training_arguments(
    config: ExperimentConfig,
    seed: int,
    output_dir: str | Path,
    *,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    run_name: Optional[str] = None,
) -> TrainingArguments:
    """Materialize a :class:`TrainingArguments` from the config.

    Args:
        config: Resolved experiment configuration.
        seed: Run seed (also used for ``data_seed``).
        output_dir: Trainer-managed checkpoint directory.
        epochs: Optional override (used by sequential phases).
        learning_rate: Optional override (used by sequential phases).
        run_name: Optional human-readable run name for logging.
    """
    final_epochs = epochs if epochs is not None else config.training.epochs
    final_lr = learning_rate if learning_rate is not None else config.training.learning_rate
    save_strategy = "epoch"
    eval_strategy = "epoch"
    if config.output.eval_steps and config.output.eval_steps > 0:
        eval_strategy = "steps"
        save_strategy = "steps"

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=final_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=max(1, config.training.batch_size * 2),
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=final_lr,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        fp16=config.training.fp16,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=config.output.eval_steps or None,
        save_steps=config.output.eval_steps or 500,
        logging_steps=config.output.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=config.output.save_total_limit,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=config.runtime.num_workers,
        dataloader_pin_memory=config.runtime.pin_memory,
        report_to=["none"],
        run_name=run_name or f"{config.experiment.name}_seed{seed}",
    )
    return args


def snapshot_config(config: ExperimentConfig, results_dir: Path) -> Path:
    """Write a YAML snapshot of the resolved config under ``results_dir``."""
    return dump_config_snapshot(config, results_dir / "config_snapshot.yaml")


def save_best_model(trainer: Trainer, target_dir: Path, tokenizer: PreTrainedTokenizerBase) -> Path:
    """Persist the trainer's best model and tokenizer into ``target_dir``.

    Args:
        trainer: Trained :class:`transformers.Trainer` (with the best model
            already loaded thanks to ``load_best_model_at_end=True``).
        target_dir: Destination directory for the final ``best_model``.
        tokenizer: Tokenizer to save alongside the weights.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(target_dir))
    tokenizer.save_pretrained(str(target_dir))
    return target_dir.resolve()


def cleanup_intermediate_checkpoints(checkpoint_dir: Path, keep: Iterable[str]) -> None:
    """Remove ``checkpoint-XXXX`` directories not in ``keep`` to save disk space."""
    keep_set = set(keep)
    for entry in checkpoint_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("checkpoint-") and entry.name not in keep_set:
            shutil.rmtree(entry, ignore_errors=True)


def evaluate_on_test_sets(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    seed: int,
    results_dir: Path,
    *,
    max_examples: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate the trainer on every test set selected in the config.

    Args:
        trainer: Trained trainer to invoke ``predict`` on.
        tokenizer: Tokenizer for dataset construction.
        config: Experiment configuration.
        seed: Run seed (recorded in the JSON output).
        results_dir: Per-seed results directory.
        max_examples: Optional cap (used by smoke tests).
    """
    test_datasets = build_test_datasets(config, tokenizer, max_examples=max_examples)
    results: Dict[str, Dict[str, Any]] = {}

    for name, dataset in test_datasets.items():
        logger.info("Evaluating on test set '%s' (%d examples)", name, len(dataset))
        prediction_output = trainer.predict(dataset, metric_key_prefix=name)
        report = evaluate_predictions(
            np.asarray(prediction_output.predictions),
            np.asarray(prediction_output.label_ids),
            id2label=config.id2label,
        )
        payload = {
            **report,
            "trainer_metrics": {
                k: float(v) for k, v in prediction_output.metrics.items() if isinstance(v, (int, float))
            },
        }
        save_evaluation_json(
            payload,
            results_dir / f"eval_{name}.json",
            config_name=config.experiment.name,
            seed=seed,
            test_set=name,
        )
        _log_evaluation_summary(name, payload)
        results[name] = payload

    return results


def _log_evaluation_summary(test_set: str, payload: Dict[str, Any]) -> None:
    """Log a compact summary block for one test set."""
    overall = payload["overall"]
    logger.info(
        "Test '%s' overall: precision=%.4f, recall=%.4f, f1=%.4f",
        test_set,
        overall["precision"],
        overall["recall"],
        overall["f1"],
    )
    for entity, metrics in payload["per_entity"].items():
        logger.info(
            "Test '%s' entity %s: f1=%.4f, p=%.4f, r=%.4f, support=%d",
            test_set,
            entity,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["support"],
        )


MetricsFn = Callable[[Any], Dict[str, float]]


def make_compute_metrics(config: ExperimentConfig) -> MetricsFn:
    """Build the ``compute_metrics`` callable bound to the config's label space."""
    return compute_metrics_factory(config.id2label)


def standard_data_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorForTokenClassification:
    """Return a vanilla token-classification collator."""
    return DataCollatorForTokenClassification(tokenizer=tokenizer)
