#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi

if ! .venv/bin/python -c "import torch, pyarrow, yaml" >/dev/null 2>&1; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/rtdl-quant-matplotlib"
mkdir -p "$MPLCONFIGDIR"

exec .venv/bin/python main.py "$@"
