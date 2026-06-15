"""Joint training strategies for Config 5 (uniform) and Config 6 (noise-aware).

Both strategies concatenate BC5CDR and PubMed silver into one training stream.
Config 5 uses uniform sample weights (1.0 for both sources). Config 6 reduces
the silver weight (default 0.3) so the loss penalises noisy supervision less.

Sample weights flow through :class:`DataCollatorForWeightedNER` and are applied
to per-example mean cross entropy in :class:`WeightedTrainer.compute_loss`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import Trainer

from src.config import ExperimentConfig
from src.data.collator import DataCollatorForWeightedNER
from src.data.loaders import build_joint_train_dataset, build_validation_dataset
from src.models.ner_model import build_model_and_tokenizer
from src.training.trainer_base import (
    build_run_dirs,
    build_training_arguments,
    cleanup_intermediate_checkpoints,
    evaluate_on_test_sets,
    make_compute_metrics,
    save_best_model,
    snapshot_config,
)
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-example sample weights to the loss.

    Per-token cross entropy is averaged within each example (ignoring -100
    positions), the per-example loss is multiplied by the example's weight,
    and the batch mean is returned. Evaluation skips weighting because the
    validation/test sets always use ``sample_weight=1.0``.
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        sample_weights = inputs.pop("sample_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        num_labels = int(logits.size(-1))
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        per_token = per_token.view(labels.shape)

        valid = (labels != -100).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        per_example = (per_token * valid).sum(dim=1) / denom

        if sample_weights is None:
            loss = per_example.mean()
        else:
            weights = sample_weights.to(per_example.device, dtype=per_example.dtype)
            loss = (per_example * weights).mean()

        return (loss, outputs) if return_outputs else loss


def train_joint(
    config: ExperimentConfig,
    seed: int,
    *,
    smoke_test: bool = False,
    max_train_examples: Optional[int] = None,
    max_eval_examples: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run a joint (uniform or noise-aware) training cycle."""
    if config.training.strategy not in {"joint_uniform", "joint_noise_aware"}:
        raise ValueError(
            f"train_joint expects strategy in {{joint_uniform, joint_noise_aware}}, "
            f"got {config.training.strategy!r}."
        )

    set_all_seeds(seed)
    if smoke_test:
        max_train_examples = max_train_examples or 16
        max_eval_examples = max_eval_examples or 16

    dirs = build_run_dirs(config, seed)
    snapshot_config(config, dirs["results"])

    model, tokenizer = build_model_and_tokenizer(config)
    train_dataset = build_joint_train_dataset(
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

    collator = DataCollatorForWeightedNER(tokenizer=tokenizer)

    weights_summary = config.training.source_weights or {}
    logger.info(
        "Joint training (%s): combined=%d, val=%d, source_weights=%s",
        config.training.strategy,
        len(train_dataset),
        len(val_dataset),
        weights_summary,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(config),
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
