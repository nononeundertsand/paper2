#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 run_synthetic.py \
  --output-dir outputs/synthetic_run \
  --device auto \
  --base-epochs 8 \
  --memory-epochs 10 \
  --batch-size 128

