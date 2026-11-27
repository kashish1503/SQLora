#!/usr/bin/env bash
# Rank ablation: same data/steps, only r changes (alpha = 2r keeps the alpha/r scale fixed).
set -euo pipefail
CONFIG=${1:-configs/qwen05b_sql.yaml}

for r in 1 4 16 64; do
  out="outputs/sweep/r${r}"
  python scripts/train.py --config "$CONFIG" \
    --override lora.r=$r lora.alpha=$((2 * r)) train.output_dir=$out
  python scripts/eval_sql.py --config "$CONFIG" --adapter "$out" --limit 500
done

python scripts/summarize.py outputs/sweep
