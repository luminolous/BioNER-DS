#!/usr/bin/env bash
# Run experiment for Config 2 (BioBERT + BC5CDR).
# Usage: bash scripts/run_config_2.sh [seeds]
# Example: bash scripts/run_config_2.sh "42,1337,2024"

set -euo pipefail

# Always run from the project root regardless of caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/config_2_biobert.yaml"
SEEDS="${1:-42,1337,2024}"

IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

for seed in "${SEED_ARRAY[@]}"; do
    echo "=== Running Config 2 (BioBERT) with seed=${seed} ==="
    python train.py --config "$CONFIG" --seed "$seed"
done

echo "=== Aggregating results across seeds for ${CONFIG} ==="
python -m src.utils.aggregate --config "$CONFIG"
