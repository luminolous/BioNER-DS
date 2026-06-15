"""Smoke test for Phase 1: load config, build small datasets, verify mapping.

Run from the project root:

    python scripts/smoke_phase1.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from src.config import LABEL2ID_5TAG, LABEL2ID_9TAG, load_config  # noqa: E402
from src.data.dataset import NERDataset  # noqa: E402
from src.data.loaders import (  # noqa: E402
    build_single_train_dataset,
    build_test_datasets,
    build_validation_dataset,
)
from src.utils.logging import configure_logging  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

logger = logging.getLogger("smoke_phase1")


def main() -> int:
    configure_logging(ROOT / "outputs" / "logs" / "smoke_phase1", level="INFO")
    set_all_seeds(42)

    logger.info("Label spaces: 5tag=%d, 9tag=%d", len(LABEL2ID_5TAG), len(LABEL2ID_9TAG))
    assert LABEL2ID_5TAG["O"] == 0
    assert LABEL2ID_9TAG["B-Virus"] == 5
    assert LABEL2ID_9TAG["B-Gene"] == 7

    config_path = ROOT / "configs" / "config_3_pubmedbert.yaml"
    config = load_config(config_path)
    logger.info(
        "Loaded config: id=%d, name=%s, strategy=%s, label_space=%s",
        config.experiment.id,
        config.experiment.name,
        config.training.strategy,
        config.data.label_space,
    )
    assert config.model.num_labels == 5
    assert "O" in config.label2id

    # Use a small tokenizer that does not require downloading PubMedBERT just
    # for the smoke test. We point at bert-base-uncased's tokenizer if the
    # network is unavailable; otherwise we use the configured backbone.
    backbone = config.model.backbone
    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        logger.info("Loaded tokenizer for backbone=%s", backbone)
    except Exception as exc:  # pragma: no cover - network conditioned
        logger.warning(
            "Could not download %s tokenizer (%s); falling back to bert-base-uncased.",
            backbone,
            exc,
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    train_ds = build_single_train_dataset(config, tokenizer, max_examples=32)
    val_ds = build_validation_dataset(config, tokenizer, max_examples=32)
    test_dss = build_test_datasets(config, tokenizer, max_examples=32)

    logger.info("Train sample size: %d", len(train_ds))
    logger.info("Validation sample size: %d", len(val_ds))
    for name, ds in test_dss.items():
        logger.info("Test set %s: %d examples loaded.", name, len(ds))

    first = train_ds[0]
    logger.info(
        "First example: input_ids len=%d, labels len=%d, sample_weight=%.2f",
        len(first["input_ids"]),
        len(first["labels"]),
        float(first["sample_weight"].item()),
    )
    assert len(first["input_ids"]) == len(first["labels"])

    distinct_labels = set(first["labels"])
    distinct_labels.discard(-100)
    assert all(0 <= lbl < config.model.num_labels for lbl in distinct_labels), distinct_labels

    logger.info("Phase 1 smoke test PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
