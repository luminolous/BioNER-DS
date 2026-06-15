"""Standalone evaluation entry point.

Re-evaluates a trained checkpoint against a single test set declared in the
experiment config, without re-running training. Writes a JSON file in the
schema described in ``specs/05_evaluation_spec.md``.

Usage:

    python evaluate.py \\
        --checkpoint outputs/checkpoints/config_4_sequential/seed_42/phase2/best_model \\
        --config configs/config_4_sequential.yaml \\
        --test_set test_pubmed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from src.config import ExperimentConfig, load_config
from src.data.dataset import NERDataset
from src.evaluation.metrics import evaluate_predictions, save_evaluation_json
from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate a trained NER checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint directory.")
    parser.add_argument("--config", type=str, required=True, help="Original YAML config used during training.")
    parser.add_argument("--test_set", type=str, required=True, help="Name of the test source declared in the config.")
    parser.add_argument("--output", type=str, default=None, help="Optional override output JSON path.")
    parser.add_argument("--seed", type=int, default=0, help="Seed value recorded in the output JSON.")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'.")
    return parser.parse_args()


def _select_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_test_source(config: ExperimentConfig, name: str):
    for src in config.data.test_sources:
        if src.name == name:
            return src
    raise ValueError(
        f"Test source {name!r} not in config (available: {[s.name for s in config.data.test_sources]})."
    )


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    test_src = _find_test_source(config, args.test_set)

    default_output = (
        Path(config.output.base_dir)
        / "results"
        / config.experiment.name
        / f"seed_{args.seed}"
        / f"eval_{args.test_set}.json"
    )
    output_path = Path(args.output) if args.output else default_output
    configure_logging(output_path.parent / "logs", run_name=f"evaluate_{args.test_set}")

    device = _select_device(args.device)
    logger.info("Loading checkpoint from %s (device=%s)", args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint).to(device).eval()

    dataset = NERDataset(
        path=test_src.path,
        tokenizer=tokenizer,
        label_field=test_src.label_field,
        label2id=config.label2id,
        max_length=config.model.max_length,
        source_tag=test_src.source_tag,
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("sample_weights", None)
            labels = batch["labels"]
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    max_len = max(len(seq) for seq in all_predictions)
    pred_array = np.full((len(all_predictions), max_len), 0, dtype=np.int64)
    label_array = np.full((len(all_labels), max_len), -100, dtype=np.int64)
    for i, (p, l) in enumerate(zip(all_predictions, all_labels)):
        pred_array[i, : len(p)] = p
        label_array[i, : len(l)] = l

    report: Dict[str, Any] = evaluate_predictions(pred_array, label_array, id2label=config.id2label)
    save_evaluation_json(
        report,
        output_path,
        config_name=config.experiment.name,
        seed=args.seed,
        test_set=args.test_set,
    )
    logger.info(
        "Evaluation done. Overall: precision=%.4f, recall=%.4f, f1=%.4f",
        report["overall"]["precision"],
        report["overall"]["recall"],
        report["overall"]["f1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
