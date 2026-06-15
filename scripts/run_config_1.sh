#!/usr/bin/env bash
# Run experiment for Config 1 (BERT-base + BC5CDR).
# Usage: bash scripts/run_config_1.sh [seeds]
# Example: bash scripts/run_config_1.sh "42,1337,2024"

set -euo pipefail

CONFIG="configs/config_1_bert_base.yaml"
SEEDS="${1:-42,1337,2024}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

for seed in "${SEED_ARRAY[@]}"; do
    echo "=== Running Config 1 (BERT-base) with seed=${seed} ==="
    python train.py --config "$CONFIG" --seed "$seed"
done

echo "=== Aggregating results across seeds for ${CONFIG} ==="
python -m src.utils.aggregate --config "$CONFIG"
