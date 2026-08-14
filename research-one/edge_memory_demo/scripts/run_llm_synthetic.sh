#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_llm_synthetic.sh MODEL_NAME_OR_LOCAL_PATH"
  echo "Example: scripts/run_llm_synthetic.sh Qwen/Qwen2.5-0.5B-Instruct"
  exit 1
fi

python3 run_llm_synthetic.py \
  --model-name-or-path "$1" \
  --output-dir outputs/llm_synthetic \
  --device auto \
  --fp16 \
  --feature-batch-size 16 \
  --batch-size 64 \
  --base-epochs 8 \
  --memory-epochs 10 \
  --thresholds 0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90

