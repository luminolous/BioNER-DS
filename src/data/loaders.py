"""High-level dataset assembly per training strategy.

These helpers wrap :class:`src.data.dataset.NERDataset` with the source/weight
combinations dictated by the experiment config. They are intentionally thin so
that trainers can compose datasets without re-implementing branching logic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from torch.utils.data import ConcatDataset, Dataset

from src.config import DataSource, ExperimentConfig
from src.data.dataset import NERDataset
from src.utils.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)


def _resolve_weight(config: ExperimentConfig, source_tag: str) -> float:
    """Return the per-source weight encoded in the config (defaults to 1.0)."""
    weights = config.training.source_weights
    if not weights:
        return 1.0
    if source_tag not in weights:
        raise ConfigValidationError(
            f"source_weights does not contain entry for tag {source_tag!r}."
        )
    return float(weights[source_tag])


def build_single_train_dataset(
    config: ExperimentConfig,
    tokenizer,
    *,
    max_examples: Optional[int] = None,
) -> NERDataset:
    """Construct the training dataset for the ``single`` strategy."""
    if len(config.data.train_sources) != 1:
        raise ConfigValidationError(
            "build_single_train_dataset expects exactly one train source."
        )
    src = config.data.train_sources[0]
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


def build_joint_train_dataset(
    config: ExperimentConfig,
    tokenizer,
    *,
    max_examples: Optional[int] = None,
) -> Dataset:
    """Concatenate all train sources with their per-source sample weights."""
    if len(config.data.train_sources) < 2:
        raise ConfigValidationError(
            "Joint strategies require at least two train sources."
        )
    parts: List[NERDataset] = []
    for src in config.data.train_sources:
        weight = _resolve_weight(config, src.source_tag)
        parts.append(
            NERDataset(
                path=src.path,
                tokenizer=tokenizer,
                label_field=src.label_field,
                label2id=config.label2id,
                max_length=config.model.max_length,
                source_tag=src.source_tag,
                sample_weight=weight,
                max_examples=max_examples,
            )
        )
    logger.info(
        "Concatenated %d datasets with sizes %s for joint training.",
        len(parts),
        [len(p) for p in parts],
    )
    return ConcatDataset(parts)


def build_phase_dataset(
    config: ExperimentConfig,
    tokenizer,
    source_name: str,
    *,
    max_examples: Optional[int] = None,
) -> NERDataset:
    """Build a dataset for one phase of the sequential trainer (Config 4)."""
    src = _find_source(config.data.train_sources, source_name)
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


def build_validation_dataset(
    config: ExperimentConfig,
    tokenizer,
    *,
    max_examples: Optional[int] = None,
) -> NERDataset:
    """Construct the validation dataset declared in the config."""
    src = config.data.validation_source
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


def build_test_datasets(
    config: ExperimentConfig,
    tokenizer,
    *,
    max_examples: Optional[int] = None,
) -> Dict[str, NERDataset]:
    """Build all test datasets selected by ``evaluation.test_sets_to_evaluate``."""
    selected = set(config.evaluation.test_sets_to_evaluate)
    if not selected:
        selected = {s.name for s in config.data.test_sources}
    result: Dict[str, NERDataset] = {}
    for src in config.data.test_sources:
        if src.name not in selected:
            continue
        result[src.name] = NERDataset(
            path=src.path,
            tokenizer=tokenizer,
            label_field=src.label_field,
            label2id=config.label2id,
            max_length=config.model.max_length,
            source_tag=src.source_tag,
            sample_weight=1.0,
            max_examples=max_examples,
        )
    return result


def _find_source(sources: List[DataSource], name: str) -> DataSource:
    for src in sources:
        if src.name == name:
            return src
    raise ConfigValidationError(
        f"No data source named {name!r} (available: {[s.name for s in sources]})."
    )
