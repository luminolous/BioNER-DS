"""NER dataset implementation with BIO-to-subword alignment.

The dataset:

* Loads a JSONL file into memory.
* Validates the schema and label space per ``specs/02_data_spec.md``.
* Applies BIO consistency rules (no leading ``I-``, no mismatched transitions).
* Tokenizes with a HuggingFace tokenizer and aligns labels so that only the
  first subword of each word carries the label; subsequent subwords get
  ``-100`` (ignored by ``CrossEntropyLoss``).
* Optionally attaches a per-example ``sample_weight`` used by the noise-aware
  joint trainer (Config 6).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Fraction of entity-start fixes above which we promote the warning to ERROR.
BIO_FIX_WARN_THRESHOLD = 0.005


def _ensure_bio_consistency(
    labels: List[str],
    label_space: Dict[str, int],
    stats: Counter,
) -> List[str]:
    """Apply spec-mandated BIO repair rules in place and tally corrections.

    Args:
        labels: List of BIO tag strings for a single sentence.
        label_space: Mapping of valid label strings to IDs.
        stats: Counter shared across the dataset to accumulate repair counts.

    Returns:
        The (possibly modified) list of labels.

    Raises:
        DataValidationError: If a label is not present in ``label_space``.
    """
    repaired: List[str] = []
    prev_entity: Optional[str] = None
    for tag in labels:
        if tag not in label_space:
            raise DataValidationError(
                f"Label {tag!r} is not present in the active label space "
                f"({sorted(label_space)}). Check that label_space matches the dataset file."
            )
        if tag == "O":
            repaired.append("O")
            prev_entity = None
            continue

        prefix, entity_type = tag.split("-", 1)
        if prefix == "I":
            if prev_entity is None or prev_entity != entity_type:
                repaired_tag = f"B-{entity_type}"
                stats["bio_repair_i_to_b"] += 1
                repaired.append(repaired_tag)
                prev_entity = entity_type
                continue
        repaired.append(tag)
        prev_entity = entity_type

    return repaired


def tokenize_and_align_labels(
    tokens: List[str],
    labels: List[str],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 512,
) -> Dict[str, Any]:
    """Tokenize pre-split tokens and align BIO labels to subword positions.

    Args:
        tokens: Pre-tokenized word list for a sentence.
        labels: BIO labels with ``len(labels) == len(tokens)``.
        tokenizer: HuggingFace fast tokenizer.
        label2id: Mapping of label strings to integer IDs.
        max_length: Maximum tokenized sequence length.

    Returns:
        Dict containing ``input_ids``, ``attention_mask``, and aligned
        ``labels`` (with ``-100`` at special-token and continuation positions).
    """
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    word_ids = tokenized.word_ids()
    aligned: List[int] = []
    previous_word_id: Optional[int] = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != previous_word_id:
            aligned.append(label2id[labels[word_id]])
        else:
            aligned.append(-100)
        previous_word_id = word_id
    tokenized["labels"] = aligned
    return tokenized


class NERDataset(Dataset):
    """Token-classification dataset backed by a JSONL file.

    Args:
        path: Path to the JSONL file.
        tokenizer: HuggingFace fast tokenizer for the chosen backbone.
        label_field: JSON field containing the BIO labels to use (e.g.
            ``"decoded_tags"`` or ``"decoded_tags_pseudo_final"``).
        label2id: Mapping of label strings to integer IDs.
        max_length: Maximum tokenized sequence length.
        source_tag: Logical tag (``"gold"`` or ``"silver"``) attached to every
            example; used by joint trainers.
        sample_weight: Per-example weight emitted alongside the sample for the
            weighted-loss trainer. Defaults to 1.0.
        max_examples: Optional cap on the number of examples to load (handy for
            smoke tests).
    """

    REQUIRED_FIELDS = ("tokens",)

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        label_field: str,
        label2id: Dict[str, int],
        max_length: int = 512,
        source_tag: str = "gold",
        sample_weight: float = 1.0,
        max_examples: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {self.path}. Verify the YAML config 'path' field."
            )

        self.tokenizer = tokenizer
        self.label_field = label_field
        self.label2id = dict(label2id)
        self.max_length = int(max_length)
        self.source_tag = source_tag
        self.sample_weight = float(sample_weight)

        self.raw_examples: List[Dict[str, Any]] = []
        self._encoded: List[Dict[str, Any]] = []
        self._label_counter: Counter = Counter()
        self._bio_stats: Counter = Counter()

        self._load(max_examples=max_examples)
        self._log_statistics()

    def _load(self, max_examples: Optional[int]) -> None:
        """Read the JSONL file, apply BIO repair, and tokenize each sentence."""
        skipped_empty = 0
        with self.path.open("r", encoding="utf-8") as fp:
            for line_idx, line in enumerate(fp):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataValidationError(
                        f"Invalid JSON at {self.path}:{line_idx + 1}: {exc.msg}"
                    ) from exc

                for required in self.REQUIRED_FIELDS:
                    if required not in example:
                        raise DataValidationError(
                            f"Missing required field {required!r} in "
                            f"{self.path}:{line_idx + 1}."
                        )
                if self.label_field not in example:
                    raise DataValidationError(
                        f"Missing label field {self.label_field!r} in "
                        f"{self.path}:{line_idx + 1}. Available keys: {list(example.keys())}."
                    )

                tokens: List[str] = example["tokens"]
                labels: List[str] = example[self.label_field]

                if len(tokens) == 0:
                    skipped_empty += 1
                    continue
                if len(tokens) != len(labels):
                    raise DataValidationError(
                        f"Token/label length mismatch in {self.path}:{line_idx + 1} "
                        f"({len(tokens)} tokens vs {len(labels)} labels)."
                    )

                labels = _ensure_bio_consistency(labels, self.label2id, self._bio_stats)
                self._label_counter.update(labels)

                encoded = tokenize_and_align_labels(
                    tokens=tokens,
                    labels=labels,
                    tokenizer=self.tokenizer,
                    label2id=self.label2id,
                    max_length=self.max_length,
                )

                self.raw_examples.append({"tokens": tokens, "labels": labels})
                self._encoded.append(encoded)

                if max_examples is not None and len(self._encoded) >= max_examples:
                    break

        if skipped_empty:
            logger.warning(
                "Skipped %d empty sentences in %s.", skipped_empty, self.path.name
            )

        total_tokens = sum(self._label_counter.values())
        repair_share = (
            self._bio_stats["bio_repair_i_to_b"] / total_tokens if total_tokens else 0.0
        )
        if repair_share > BIO_FIX_WARN_THRESHOLD:
            logger.warning(
                "BIO repair affected %.2f%% of tokens in %s (threshold %.2f%%). "
                "Inspect upstream tagger output.",
                repair_share * 100,
                self.path.name,
                BIO_FIX_WARN_THRESHOLD * 100,
            )

    def _log_statistics(self) -> None:
        """Emit a one-line summary about the loaded examples."""
        total_tokens = sum(self._label_counter.values())
        logger.info(
            "Loaded %d sentences (%d tokens) from %s with label_field=%r, source=%r, "
            "sample_weight=%.3f.",
            len(self._encoded),
            total_tokens,
            self.path.name,
            self.label_field,
            self.source_tag,
            self.sample_weight,
        )
        top_labels = ", ".join(
            f"{lbl}={cnt}" for lbl, cnt in self._label_counter.most_common(5)
        )
        logger.info("Top-5 labels in %s: %s", self.path.name, top_labels)

    def label_distribution(self) -> Dict[str, int]:
        """Return the per-label token count after BIO repair."""
        return dict(self._label_counter)

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        encoded = self._encoded[idx]
        item: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
            "sample_weight": torch.tensor(self.sample_weight, dtype=torch.float32),
        }
        return item
