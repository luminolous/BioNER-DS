#!/usr/bin/env bash
# Aggregate multi-seed evaluation JSON files for every config.
# Usage: bash scripts/aggregate_results.sh

set -euo pipefail

# Always run from the project root regardless of caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

for cfg in configs/config_*.yaml; do
    # Skip the base.yaml -- it is an inheritance target, not a standalone config.
    if [[ "$cfg" == *"/base.yaml" ]]; then
        continue
    fi
    echo "=== Aggregating ${cfg} ==="
    python -m src.utils.aggregate --config "$cfg"
done
