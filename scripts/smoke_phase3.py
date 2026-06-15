"""Phase 3 smoke test: every config YAML loads and validates.

For configs whose trainers already exist (1, 2, 3), also instantiate the
datasets with a small sample to confirm the YAML paths and label fields work
end-to-end. Sequential / joint trainers ship in Phase 4 -- this script only
runs the config + dataset wiring for them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.loaders import (  # noqa: E402
    build_joint_train_dataset,
    build_phase_dataset,
    build_single_train_dataset,
    build_test_datasets,
    build_validation_dataset,
)
from src.utils.logging import configure_logging  # noqa: E402

logger = logging.getLogger("smoke_phase3")


CONFIGS = [
    ("configs/config_1_bert_base.yaml", "single", "bert-base-uncased"),
    ("configs/config_2_biobert.yaml", "single", "dmis-lab/biobert-base-cased-v1.2"),
    ("configs/config_3_pubmedbert.yaml", "single", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
    ("configs/config_4_sequential.yaml", "sequential", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
    ("configs/config_5_joint_uniform.yaml", "joint_uniform", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
    ("configs/config_6_joint_noise_aware.yaml", "joint_noise_aware", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
]


def _load_tokenizer(name: str):
    try:
        return AutoTokenizer.from_pretrained(name, use_fast=True)
    except Exception as exc:  # pragma: no cover - network conditioned
        logger.warning("Falling back to bert-base-uncased tokenizer (%s).", exc)
        return AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)


def main() -> int:
    configure_logging(ROOT / "outputs" / "logs" / "smoke_phase3", level="INFO")

    for cfg_rel, expected_strategy, backbone in CONFIGS:
        cfg_path = ROOT / cfg_rel
        logger.info("=== %s ===", cfg_rel)
        config = load_config(cfg_path)
        assert config.training.strategy == expected_strategy, config.training.strategy

        tokenizer = _load_tokenizer(backbone)

        if expected_strategy == "single":
            train = build_single_train_dataset(config, tokenizer, max_examples=8)
            assert len(train) > 0
            logger.info("  single train -> %d examples", len(train))
        elif expected_strategy == "sequential":
            p1 = build_phase_dataset(
                config,
                tokenizer,
                config.training.phase1.train_source,
                max_examples=8,
            )
            p2 = build_phase_dataset(
                config,
                tokenizer,
                config.training.phase2.train_source,
                max_examples=8,
            )
            assert len(p1) > 0 and len(p2) > 0
            logger.info(
                "  sequential phase1=%d, phase2=%d",
                len(p1),
                len(p2),
            )
        else:
            joint = build_joint_train_dataset(config, tokenizer, max_examples=8)
            assert len(joint) > 0
            logger.info("  joint combined -> %d examples", len(joint))

        val = build_validation_dataset(config, tokenizer, max_examples=8)
        tests = build_test_datasets(config, tokenizer, max_examples=8)
        logger.info(
            "  validation=%d, test_sets=%s",
            len(val),
            {k: len(v) for k, v in tests.items()},
        )

    logger.info("Phase 3 smoke test PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
