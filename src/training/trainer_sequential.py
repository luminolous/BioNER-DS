"""Sequential training strategy for Config 4 (BC5CDR -> PubMed silver).

Two-phase fine-tune that re-uses the standard HuggingFace ``Trainer`` per
phase. Phase 1 trains the 9-output head on BC5CDR (where Virus/Gene labels are
all ``O``). Phase 2 loads the phase-1 best checkpoint and continues training on
the silver corpus so Virus/Gene heads receive positive examples. The final
test-set evaluation runs on the phase-2 best model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import Trainer

from src.config import ExperimentConfig, PhaseConfig
from src.data.loaders import build_phase_dataset
from src.data.dataset import NERDataset
from src.models.ner_model import build_model, build_tokenizer
from src.training.trainer_base import (
    build_run_dirs,
    build_training_arguments,
    cleanup_intermediate_checkpoints,
    evaluate_on_test_sets,
    make_compute_metrics,
    save_best_model,
    snapshot_config,
    standard_data_collator,
)
from src.utils.exceptions import ConfigValidationError
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def _build_validation(
    config: ExperimentConfig,
    tokenizer,
    validation_source_name: str,
    *,
    max_examples: Optional[int],
) -> NERDataset:
    """Build a validation dataset using the phase-specific validation source."""
    if validation_source_name == config.data.validation_source.name:
        src = config.data.validation_source
    else:
        candidates = {s.name: s for s in config.data.train_sources}
        if validation_source_name not in candidates:
            raise ConfigValidationError(
                f"Validation source {validation_source_name!r} is not defined in "
                f"data.validation_source or data.train_sources."
            )
        src = candidates[validation_source_name]
    return NERDataset(
        path=src.path,
        tokenizer=tokenizer,
        label_field=src.label_field,
        label2id=config.label2id,
        max_length=config.model.max_length,
        source_tag=src.source_tag,
        sample_weight=1.0,
        max_examples=max_examples,
    )


def _run_phase(
    config: ExperimentConfig,
    seed: int,
    phase_label: str,
    phase: PhaseConfig,
    tokenizer,
    model,
    phase_dir: Path,
    *,
    smoke_test: bool,
    max_train_examples: Optional[int],
    max_eval_examples: Optional[int],
) -> Trainer:
    """Train a single phase of the sequential pipeline."""
    train_ds = build_phase_dataset(
        config,
        tokenizer,
        phase.train_source,
        max_examples=max_train_examples,
    )
    val_ds = _build_validation(
        config,
        tokenizer,
        phase.validation_source,
        max_examples=max_eval_examples,
    )

    epochs = 1 if smoke_test else phase.epochs
    training_args = build_training_arguments(
        config,
        seed=seed,
        output_dir=phase_dir,
        epochs=epochs,
        learning_rate=phase.learning_rate,
        run_name=f"{config.experiment.name}_seed{seed}_{phase_label}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=standard_data_collator(tokenizer),
        compute_metrics=make_compute_metrics(config),
    )

    logger.info(
        "Sequential %s: train=%d, val=%d, epochs=%d, lr=%g",
        phase_label,
        len(train_ds),
        len(val_ds),
        epochs,
        phase.learning_rate,
    )
    trainer.train()
    save_best_model(trainer, phase_dir / "best_model", tokenizer)
    cleanup_intermediate_checkpoints(phase_dir, keep={"best_model"})

    final_metrics = trainer.evaluate(val_ds, metric_key_prefix=f"{phase_label}_val")
    logger.info(
        "Sequential %s validation: f1=%.4f, precision=%.4f, recall=%.4f",
        phase_label,
        float(final_metrics.get(f"{phase_label}_val_f1", 0.0)),
        float(final_metrics.get(f"{phase_label}_val_precision", 0.0)),
        float(final_metrics.get(f"{phase_label}_val_recall", 0.0)),
    )
    trainer._final_validation_metrics = final_metrics  # type: ignore[attr-defined]
    return trainer


def train_sequential(
    config: ExperimentConfig,
    seed: int,
    *,
    smoke_test: bool = False,
    max_train_examples: Optional[int] = None,
    max_eval_examples: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run the two-phase sequential trainer for Config 4."""
    if config.training.strategy != "sequential":
        raise ValueError(
            f"train_sequential expects strategy='sequential', got "
            f"{config.training.strategy!r}."
        )
    if config.training.phase1 is None or config.training.phase2 is None:
        raise ConfigValidationError(
            "Sequential strategy requires both phase1 and phase2 configs."
        )

    set_all_seeds(seed)
    if smoke_test:
        max_train_examples = max_train_examples or 16
        max_eval_examples = max_eval_examples or 16

    dirs = build_run_dirs(config, seed)
    snapshot_config(config, dirs["results"])
    phase1_dir = dirs["checkpoint"] / "phase1"
    phase2_dir = dirs["checkpoint"] / "phase2"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    phase2_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(config)
    model = build_model(config)

    logger.info("=== Sequential Phase 1: BC5CDR training ===")
    trainer_p1 = _run_phase(
        config,
        seed,
        "phase1",
        config.training.phase1,
        tokenizer,
        model,
        phase1_dir,
        smoke_test=smoke_test,
        max_train_examples=max_train_examples,
        max_eval_examples=max_eval_examples,
    )
    phase1_metrics = getattr(trainer_p1, "_final_validation_metrics", {})

    logger.info("=== Sequential Phase 2: Silver corpus training ===")
    model_phase2 = build_model(config, pretrained_path=str(phase1_dir / "best_model"))
    trainer_p2 = _run_phase(
        config,
        seed,
        "phase2",
        config.training.phase2,
        tokenizer,
        model_phase2,
        phase2_dir,
        smoke_test=smoke_test,
        max_train_examples=max_train_examples,
        max_eval_examples=max_eval_examples,
    )
    phase2_metrics = getattr(trainer_p2, "_final_validation_metrics", {})

    _log_forgetting_summary(phase1_metrics, phase2_metrics)

    return evaluate_on_test_sets(
        trainer_p2,
        tokenizer,
        config,
        seed=seed,
        results_dir=dirs["results"],
        max_examples=max_eval_examples,
    )


def _log_forgetting_summary(
    phase1_metrics: Dict[str, float],
    phase2_metrics: Dict[str, float],
) -> None:
    """Log catastrophic forgetting deltas for the two anchor entities."""
    for entity in ("Chemical", "Disease"):
        before = float(phase1_metrics.get(f"phase1_val_{entity}_f1", 0.0))
        after = float(phase2_metrics.get(f"phase2_val_{entity}_f1", 0.0))
        if before == 0.0 and after == 0.0:
            continue
        delta = after - before
        logger.info(
            "Forgetting score for %s: phase1_f1=%.4f, phase2_f1=%.4f, delta=%+.4f",
            entity,
            before,
            after,
            delta,
        )
    for entity in ("Virus", "Gene"):
        after = float(phase2_metrics.get(f"phase2_val_{entity}_f1", 0.0))
        if after:
            logger.info("New-entity F1 after phase 2 (%s): %.4f", entity, after)
