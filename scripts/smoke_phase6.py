"""Phase 6 smoke test: synthesize 3-seed JSONs and run the aggregator."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.aggregate import aggregate_seeds


def main() -> int:
    base = ROOT / "outputs" / "results" / "config_3_pubmedbert"
    src = base / "seed_42"
    if not src.is_dir():
        print(f"Seed 42 directory missing at {src}. Run a training pass first.")
        return 1

    for sd, perturb in ((1337, 0.01), (2024, -0.005)):
        dst = base / f"seed_{sd}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        fp = dst / "eval_test_bc5cdr.json"
        data = json.loads(fp.read_text(encoding="utf-8"))
        data["seed"] = sd
        data["overall"]["f1"] = round(data["overall"]["f1"] + perturb, 4)
        data["overall"]["precision"] = round(data["overall"]["precision"] + perturb * 1.2, 4)
        data["overall"]["recall"] = round(data["overall"]["recall"] + perturb * 0.5, 4)
        fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Synthesized seed_{sd} -> {fp}")

    aggregated = aggregate_seeds("config_3_pubmedbert", base_dir=str(ROOT / "outputs"))
    print("\nAggregated test sets:", list(aggregated["test_sets"]))
    payload = aggregated["test_sets"]["test_bc5cdr"]
    f1 = payload["overall"]["f1"]
    print(f"  overall F1 mean={f1['mean']:.4f}, std={f1['std']:.4f}, values={f1['values']}")
    print(f"  per-entity keys: {list(payload['per_entity'])}")
    print("\nAggregated markdown:")
    print((base / "aggregated_results.md").read_text(encoding="utf-8"))
    print("Phase 6 smoke test PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
