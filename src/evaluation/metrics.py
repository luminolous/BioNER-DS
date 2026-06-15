"""seqeval-backed metrics for NER training and final evaluation.

``compute_metrics`` is plugged into the HuggingFace ``Trainer`` so that
validation prints precision / recall / F1 (overall + per-entity) at each
evaluation step. ``evaluate_predictions`` is the offline variant used by
``evaluate.py`` and the trainer wrappers when serializing JSON results.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

import json
import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2

logger = logging.getLogger(__name__)


def _decode_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: Mapping[int, str],
) -> tuple[List[List[str]], List[List[str]]]:
    """Convert (B, T) prediction / label arrays into seqeval-friendly lists.

    Positions where the gold label is ``-100`` are dropped from both sequences
    so that special tokens and subword continuations do not contribute.
    """
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    true_labels: List[List[str]] = []
    pred_labels: List[List[str]] = []
    for pred_seq, label_seq in zip(predictions, labels):
        cur_true: List[str] = []
        cur_pred: List[str] = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            cur_true.append(id2label[int(label_id)])
            cur_pred.append(id2label[int(pred_id)])
        true_labels.append(cur_true)
        pred_labels.append(cur_pred)
    return true_labels, pred_labels


def compute_metrics_factory(id2label: Mapping[int, str]):
    """Return a ``compute_metrics`` callable bound to a given label mapping.

    The callable accepts a HuggingFace ``EvalPrediction`` (or a 2-tuple) and
    returns a flat metrics dict suitable for ``Trainer``'s logging.
    """

    def _compute(eval_prediction) -> Dict[str, float]:
        if hasattr(eval_prediction, "predictions"):
            predictions = eval_prediction.predictions
            labels = eval_prediction.label_ids
        else:
            predictions, labels = eval_prediction

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        true_labels, pred_labels = _decode_predictions(predictions, labels, id2label)

        results: Dict[str, float] = {
            "precision": float(
                precision_score(true_labels, pred_labels, mode="strict", scheme=IOB2, zero_division=0)
            ),
            "recall": float(
                recall_score(true_labels, pred_labels, mode="strict", scheme=IOB2, zero_division=0)
            ),
            "f1": float(
                f1_score(true_labels, pred_labels, mode="strict", scheme=IOB2, zero_division=0)
            ),
        }

        report = classification_report(
            true_labels,
            pred_labels,
            mode="strict",
            scheme=IOB2,
            output_dict=True,
            zero_division=0,
        )
        for entity_type, scores in report.items():
            if entity_type in {"micro avg", "macro avg", "weighted avg"}:
                continue
            if not isinstance(scores, dict):
                continue
            results[f"{entity_type}_precision"] = float(scores.get("precision", 0.0))
            results[f"{entity_type}_recall"] = float(scores.get("recall", 0.0))
            results[f"{entity_type}_f1"] = float(scores.get("f1-score", 0.0))
            results[f"{entity_type}_support"] = float(scores.get("support", 0))
        return results

    return _compute


def evaluate_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: Mapping[int, str],
) -> Dict[str, Any]:
    """Compute a structured per-entity report from raw predictions.

    Returns:
        Dict with ``overall`` precision/recall/F1 plus ``per_entity`` breakdown
        suitable for the JSON format described in ``specs/05_evaluation_spec.md``.
    """
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    true_labels, pred_labels = _decode_predictions(predictions, labels, id2label)

    overall = {
        "precision": float(
            precision_score(true_labels, pred_labels, mode="strict", scheme=IOB2, zero_division=0)
        ),
        "recall": float(
            recall_score(true_labels, pred_labels, mode="strict", scheme=IOB2, zero_division=0)
        ),
        "f1": float(
            f1_score(true_labels, pred_labels, mode="strict", scheme=IOB2, zero_division=0)
        ),
    }

    report = classification_report(
        true_labels,
        pred_labels,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )
    per_entity: Dict[str, Dict[str, float]] = {}
    for entity_type, scores in report.items():
        if entity_type in {"micro avg", "macro avg", "weighted avg"}:
            continue
        if not isinstance(scores, dict):
            continue
        if scores.get("support", 0) == 0:
            continue
        per_entity[entity_type] = {
            "precision": round(float(scores.get("precision", 0.0)), 4),
            "recall": round(float(scores.get("recall", 0.0)), 4),
            "f1": round(float(scores.get("f1-score", 0.0)), 4),
            "support": int(scores.get("support", 0)),
        }

    return {
        "overall": {k: round(v, 4) for k, v in overall.items()},
        "per_entity": per_entity,
        "num_examples": len(true_labels),
        "num_gold_entities": sum(p["support"] for p in per_entity.values()),
    }


def save_evaluation_json(
    payload: Dict[str, Any],
    target_path: str | Path,
    *,
    config_name: str,
    seed: int,
    test_set: str,
) -> Path:
    """Persist an evaluation payload using the schema from spec 05 section 3."""
    record = {
        "config_name": config_name,
        "seed": seed,
        "test_set": test_set,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2, ensure_ascii=False)
    logger.info("Saved evaluation results to %s", target)
    return target.resolve()
