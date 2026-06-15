#!/usr/bin/env bash
# Sequentially execute every config in 1..6 with the provided seeds.
# Usage: bash scripts/run_all.sh [seeds]
# Example: bash scripts/run_all.sh "42,1337,2024"

set -euo pipefail

SEEDS="${1:-42,1337,2024}"

for cfg in 1 2 3 4 5 6; do
    echo ">>> Running scripts/run_config_${cfg}.sh"
    bash "scripts/run_config_${cfg}.sh" "$SEEDS"
done
