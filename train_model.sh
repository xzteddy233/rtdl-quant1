#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
case "$MODEL" in
  mlp)
    CONFIG="rtdl_quant/configs/mlp.yaml"
    ;;
  resnet)
    CONFIG="rtdl_quant/configs/resnet.yaml"
    ;;
  ftt|ft_transformer)
    CONFIG="rtdl_quant/configs/ft_transformer.yaml"
    ;;
  *)
    echo "Usage: ./train_model.sh {mlp|resnet|ftt}" >&2
    exit 2
    ;;
esac

exec "$(dirname "$0")/run.sh" --config "$CONFIG"
