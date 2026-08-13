#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 run_scheduler.py \
  --output-dir outputs/scheduler_run \
  --num-items 400 \
  --storage-budget 120 \
  --energy-budget 160

