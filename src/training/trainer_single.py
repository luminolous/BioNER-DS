"""Single-source training strategy for Config 1, 2, and 3.

Trains a HuggingFace token-classification model on one labelled dataset,
selects the best checkpoint by validation F1, and evaluates on the configured
test sets. Returns a dict of per-test-set evaluation payloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import Trainer

from src.config import ExperimentConfig
from src.data.loaders import build_single_train_dataset, build_validation_dataset
from src.models.ner_model import build_model_and_tokenizer
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
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def train_single(
    config: ExperimentConfig,
    seed: int,
    *,
    smoke_test: bool = False,
    max_train_examples: Optional[int] = None,
    max_eval_examples: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run a single-source training + evaluation cycle.

    Args:
        config: Resolved experiment configuration (strategy must be ``"single"``).
        seed: Random seed for reproducibility and ``TrainingArguments``.
        smoke_test: If True, force tiny epoch/batch caps to validate plumbing.
        max_train_examples: Optional cap on training set size.
        max_eval_examples: Optional cap on validation/test set sizes.

    Returns:
        Mapping of ``test_set_name -> evaluation payload``.
    """
    if config.training.strategy != "single":
        raise ValueError(
            f"train_single expects strategy='single', got {config.training.strategy!r}."
        )

    set_all_seeds(seed)

    if smoke_test:
        max_train_examples = max_train_examples or 16
        max_eval_examples = max_eval_examples or 16

    dirs = build_run_dirs(config, seed)
    snapshot_config(config, dirs["results"])

    model, tokenizer = build_model_and_tokenizer(config)

    train_dataset = build_single_train_dataset(
        config, tokenizer, max_examples=max_train_examples
    )
    val_dataset = build_validation_dataset(
        config, tokenizer, max_examples=max_eval_examples
    )

    training_args = build_training_arguments(
        config,
        seed=seed,
        output_dir=dirs["checkpoint"],
        epochs=1 if smoke_test else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=standard_data_collator(tokenizer),
        compute_metrics=make_compute_metrics(config),
    )

    logger.info(
        "Starting training for %s (seed=%d, smoke_test=%s, train=%d, val=%d)",
        config.experiment.name,
        seed,
        smoke_test,
        len(train_dataset),
        len(val_dataset),
    )
    trainer.train()
    save_best_model(trainer, dirs["checkpoint"] / "best_model", tokenizer)
    cleanup_intermediate_checkpoints(dirs["checkpoint"], keep={"best_model"})

    return evaluate_on_test_sets(
        trainer,
        tokenizer,
        config,
        seed=seed,
        results_dir=dirs["results"],
        max_examples=max_eval_examples,
    )
