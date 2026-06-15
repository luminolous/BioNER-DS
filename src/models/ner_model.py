"""Backbone model factory for token classification.

The factory injects ``num_labels``, ``id2label``, and ``label2id`` from the
config into HuggingFace's ``AutoModelForTokenClassification`` so the checkpoint
``config.json`` carries the correct label mapping for downstream inference.
"""

from __future__ import annotations

import logging
from typing import Tuple

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def build_tokenizer(config: ExperimentConfig) -> PreTrainedTokenizerBase:
    """Instantiate the fast tokenizer for the configured backbone.

    Args:
        config: Resolved experiment configuration.

    Returns:
        A HuggingFace fast tokenizer.
    """
    logger.info("Loading tokenizer for backbone=%s", config.model.backbone)
    return AutoTokenizer.from_pretrained(config.model.backbone, use_fast=True)


def build_model(
    config: ExperimentConfig,
    pretrained_path: str | None = None,
) -> PreTrainedModel:
    """Instantiate a token-classification model with the active label mapping.

    Args:
        config: Resolved experiment configuration.
        pretrained_path: Optional checkpoint directory to load weights from
            instead of the HF Hub backbone (used by sequential phase 2).

    Returns:
        ``AutoModelForTokenClassification`` ready for training.
    """
    source = pretrained_path or config.model.backbone
    logger.info(
        "Building token-classification model from %s with num_labels=%d",
        source,
        config.model.num_labels,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        source,
        num_labels=config.model.num_labels,
        id2label={int(k): v for k, v in config.id2label.items()},
        label2id=dict(config.label2id),
        ignore_mismatched_sizes=False,
    )
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
    return model


def build_model_and_tokenizer(
    config: ExperimentConfig,
    pretrained_path: str | None = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Convenience wrapper that returns both the model and its tokenizer."""
    tokenizer = build_tokenizer(config)
    model = build_model(config, pretrained_path=pretrained_path)
    return model, tokenizer
