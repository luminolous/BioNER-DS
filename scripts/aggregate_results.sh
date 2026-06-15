#!/usr/bin/env bash
# Aggregate multi-seed evaluation JSON files for every config.
# Usage: bash scripts/aggregate_results.sh

set -euo pipefail

for cfg in configs/config_*.yaml; do
    echo "=== Aggregating ${cfg} ==="
    python -m src.utils.aggregate --config "$cfg"
done
