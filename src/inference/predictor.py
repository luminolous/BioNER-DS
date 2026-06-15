"""NER inference wrapper used by ``predict.py`` and the downstream KG team.

``NERPredictor`` loads a fine-tuned token-classification checkpoint together
with its tokenizer, runs batched inference over pre-tokenized sentences, and
returns one entity record per detected span in the schema specified in
``specs/06_implementation_notes.md`` section 15.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def _select_device(device: Optional[str]) -> torch.device:
    """Pick a torch device, defaulting to CUDA when available."""
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NERPredictor:
    """Convenience wrapper around a fine-tuned token-classification model."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}. Pass the "
                f"folder produced by training (contains config.json + tokenizer files)."
            )
        self.device = _select_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        logger.info("Loading tokenizer from %s", self.checkpoint_dir)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            str(self.checkpoint_dir), use_fast=True
        )
        logger.info("Loading model from %s", self.checkpoint_dir)
        self.model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            str(self.checkpoint_dir)
        ).to(self.device)
        self.model.eval()

        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        logger.info("Loaded label space (%d labels): %s", len(self.id2label), self.id2label)

    @torch.no_grad()
    def predict(
        self,
        sentences: Sequence[Sequence[str]],
    ) -> List[List[Dict[str, Any]]]:
        """Run inference over a batch of pre-tokenized sentences.

        Args:
            sentences: Iterable where each element is a list of tokens (words).

        Returns:
            A list with one entry per input sentence; each entry is a list of
            entity dicts with keys ``entity``, ``type``, ``start_token``,
            ``end_token``, and ``score``.
        """
        all_results: List[List[Dict[str, Any]]] = []
        sentences_list = [list(s) for s in sentences]
        for start in range(0, len(sentences_list), self.batch_size):
            batch_tokens = sentences_list[start : start + self.batch_size]
            results = self._predict_batch(batch_tokens)
            all_results.extend(results)
        return all_results

    def _predict_batch(
        self,
        batch_tokens: List[List[str]],
    ) -> List[List[Dict[str, Any]]]:
        """Run a single forward pass and decode spans for each example."""
        encoded = self.tokenizer(
            batch_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)

        decoded: List[List[Dict[str, Any]]] = []
        for example_idx, tokens in enumerate(batch_tokens):
            word_ids = self._word_ids(batch_tokens, example_idx, encoded)
            word_labels, word_scores = self._aggregate_to_words(
                word_ids=word_ids,
                pred_ids=pred_ids[example_idx].tolist(),
                probs=probs[example_idx].tolist(),
                num_words=len(tokens),
            )
            decoded.append(self._extract_spans(tokens, word_labels, word_scores))
        return decoded

    def _word_ids(
        self,
        batch_tokens: List[List[str]],
        example_idx: int,
        encoded: Dict[str, torch.Tensor],
    ) -> List[Optional[int]]:
        """Recover ``word_ids`` for one example by re-tokenizing it.

        ``BatchEncoding.word_ids(i)`` is only available on the encoding object
        returned by the tokenizer call; we keep a reference and look up by
        index. We re-call ``word_ids`` defensively because ``encoded`` is moved
        to the device above and the helper lives on the CPU object.
        """
        tokens = batch_tokens[example_idx]
        single = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return single.word_ids()

    def _aggregate_to_words(
        self,
        word_ids: List[Optional[int]],
        pred_ids: List[int],
        probs: List[List[float]],
        num_words: int,
    ) -> tuple[List[str], List[float]]:
        """Reduce subword predictions to one label + score per original word."""
        word_labels: List[str] = ["O"] * num_words
        word_scores: List[float] = [0.0] * num_words
        for token_pos, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id >= num_words:
                continue
            if word_labels[word_id] != "O":
                continue  # first subword wins, matching training-time alignment
            pred_id = pred_ids[token_pos]
            word_labels[word_id] = self.id2label[int(pred_id)]
            word_scores[word_id] = float(probs[token_pos][int(pred_id)])
        return word_labels, word_scores

    @staticmethod
    def _extract_spans(
        tokens: List[str],
        labels: List[str],
        scores: List[float],
    ) -> List[Dict[str, Any]]:
        """Convert BIO labels into entity span records."""
        spans: List[Dict[str, Any]] = []
        current_type: Optional[str] = None
        current_start: int = 0
        current_scores: List[float] = []

        def _flush(end_exclusive: int) -> None:
            if current_type is None:
                return
            entity_text = " ".join(tokens[current_start:end_exclusive])
            mean_score = sum(current_scores) / max(1, len(current_scores))
            spans.append(
                {
                    "entity": entity_text,
                    "type": current_type,
                    "start_token": current_start,
                    "end_token": end_exclusive - 1,
                    "score": round(float(mean_score), 4),
                }
            )

        for idx, (label, score) in enumerate(zip(labels, scores)):
            if label == "O":
                _flush(idx)
                current_type = None
                current_scores = []
                continue
            prefix, entity_type = label.split("-", 1)
            if prefix == "B" or current_type != entity_type:
                _flush(idx)
                current_type = entity_type
                current_start = idx
                current_scores = [score]
            else:
                current_scores.append(score)

        _flush(len(labels))
        return spans

    def predict_records(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        token_field: str = "tokens",
        id_field: str = "sentence_id",
    ) -> List[Dict[str, Any]]:
        """Run inference on iterable JSONL records, preserving the input id.

        Args:
            records: Iterable of dicts; each must contain ``token_field``.
            token_field: Key carrying the pre-split token list.
            id_field: Optional key copied into the output (``None`` if absent).
        """
        records_list = list(records)
        token_lists = [rec[token_field] for rec in records_list]
        predictions = self.predict(token_lists)
        output: List[Dict[str, Any]] = []
        for rec, preds in zip(records_list, predictions):
            output.append(
                {
                    "sentence_id": rec.get(id_field),
                    "tokens": rec[token_field],
                    "predictions": preds,
                }
            )
        return output
