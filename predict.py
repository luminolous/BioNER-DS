"""Standalone inference entry point for downstream KG construction.

Reads a JSONL or plain text file, runs the configured checkpoint, and writes
predictions in the format expected by the KG team
(``specs/06_implementation_notes.md`` section 15).

Examples:

    python predict.py \\
        --checkpoint outputs/checkpoints/config_4_sequential/seed_42/phase2/best_model \\
        --input dataset/pubmed/pubmed_test.jsonl \\
        --output outputs/predictions/config4_pubmed.jsonl

    python predict.py --checkpoint <ckpt> --input free_text.txt --output preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.inference.predictor import NERPredictor
from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NER inference for KG ingestion.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint directory.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL (with 'tokens') or plain text file.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'.")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length.")
    parser.add_argument("--token_field", type=str, default="tokens", help="JSON field holding the token list.")
    parser.add_argument("--id_field", type=str, default="sentence_id", help="Optional JSON id field carried into outputs.")
    return parser.parse_args()


def _load_records(path: Path, token_field: str) -> List[Dict[str, Any]]:
    """Load JSONL records, or split plain-text lines into whitespace tokens."""
    if path.suffix.lower() in {".jsonl", ".json"}:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc.msg}") from exc
                if token_field not in record:
                    raise ValueError(
                        f"Record at {path}:{line_no} is missing required field {token_field!r}."
                    )
                records.append(record)
        return records

    with path.open("r", encoding="utf-8") as fp:
        return [
            {"sentence_id": str(idx), token_field: line.strip().split()}
            for idx, line in enumerate(fp)
            if line.strip()
        ]


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    configure_logging(output_path.parent / "inference_logs", run_name="predict")

    predictor = NERPredictor(
        checkpoint_dir=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    records = _load_records(Path(args.input), token_field=args.token_field)
    logger.info("Loaded %d input records from %s", len(records), args.input)

    predictions = predictor.predict_records(
        records,
        token_field=args.token_field,
        id_field=args.id_field,
    )
    _write_jsonl(predictions, output_path)
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
