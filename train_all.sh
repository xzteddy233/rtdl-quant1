#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

FORCE="${1:-}"
if [[ -n "$FORCE" && "$FORCE" != "--force" ]]; then
  echo "Usage: ./train_all.sh [--force]" >&2
  exit 2
fi

MODELS=(mlp resnet ftt)
OUTPUTS=(
  "rtdl_quant/outputs/alpha158_mlp"
  "rtdl_quant/outputs/alpha158_resnet"
  "rtdl_quant/outputs/alpha158_ft_transformer"
)

for index in "${!MODELS[@]}"; do
  model="${MODELS[$index]}"
  output="${OUTPUTS[$index]}"
  metrics="$output/metrics.csv"

  if [[ -f "$metrics" && "$FORCE" != "--force" ]]; then
    echo "[$model] Existing result found; skipping."
    continue
  fi

  echo "[$model] Training started."
  ./train_model.sh "$model"
  echo "[$model] Training completed."
done

.venv/bin/python -m rtdl_quant.scripts.compare_models

echo "All model results are ready."
echo "Comparison: rtdl_quant/outputs/model_comparison.csv"
