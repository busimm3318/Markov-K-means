#!/usr/bin/env bash
# Application studies (image quantization and VQ codebooks). Logs go to experiments/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/logs
SUB=$(seq -s, 1 4 100)
case "${1:-all}" in
  color|all)
    python -m experiments.color_quant --download --images 1-100 --k 64 >> experiments/logs/color_quant.log 2>&1
    python -m experiments.color_quant --images "$SUB" --k 16 256 >> experiments/logs/color_quant.log 2>&1 ;;&
  vq|all)
    python -m experiments.vq_codebook >> experiments/logs/vq_codebook.log 2>&1 ;;
esac
