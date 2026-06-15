"""Data collator variants for NER training.

``DataCollatorForWeightedNER`` extends the HuggingFace token classification
collator with per-example ``sample_weights``, which the noise-aware joint
trainer (Config 6) uses to down-weight the silver corpus.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import DataCollatorForTokenClassification


class DataCollatorForWeightedNER(DataCollatorForTokenClassification):
    """Collator that batches a per-example ``sample_weights`` tensor.

    Falls back to weight=1.0 if a feature is missing the field, so the same
    collator can serve uniform (Config 5) and noise-aware (Config 6) trainers.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_weights: List[torch.Tensor] = []
        for feat in features:
            value = feat.pop("sample_weight", 1.0)
            if isinstance(value, torch.Tensor):
                sample_weights.append(value.to(torch.float32).reshape(()))
            else:
                sample_weights.append(torch.tensor(float(value), dtype=torch.float32))
        batch = super().__call__(features)
        batch["sample_weights"] = torch.stack(sample_weights)
        return batch
