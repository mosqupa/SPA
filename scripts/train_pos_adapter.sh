#!/bin/bash

# Launch entry point for PositionAdapter training (no-gate version).
#
# Runs in the FOREGROUND: progress prints live to your terminal, and a copy
# is tee'd to outputs/train_adapter.log. Ctrl+C stops the training; the
# terminal stays busy for the whole run (~2.2h for 30k samples).
#
# All knobs are env vars with defaults matching the first ablation run.
#
# Usage:
#   bash scripts/train_pos_adapter.sh                     # 30k samples (~2.2h)
#   MAX_SAMPLES=120624 bash scripts/train_pos_adapter.sh  # full official train (~9h)
#   MAX_SAMPLES=120624 SAVE_PATH=outputs/pos_adapter_full.pt bash scripts/train_pos_adapter.sh

set -e
set -o pipefail
export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="${CONDA_PYTHON:-/opt/conda/envs/vlm/bin/python}"

MAX_SAMPLES="${MAX_SAMPLES:-30000}"            # 120624 = full official train split
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-3}"
ADAPTER_TYPE="${ADAPTER_TYPE:-raw}"        # fourier (default) | raw — ablation variant
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/outputs/pos_adapter_${ADAPTER_TYPE}.pt}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/outputs/train_adapter_${ADAPTER_TYPE}.log}"

cd "$PROJECT_ROOT"

echo "training (foreground): max-samples=$MAX_SAMPLES lr=$LR adapter-type=$ADAPTER_TYPE save-path=$SAVE_PATH"
echo "log copy: $LOG_FILE"
echo ""

"$CONDA_PYTHON" "$PROJECT_ROOT/scripts/train_pos_adapter.py" \
    --max-samples "$MAX_SAMPLES" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --adapter-type "$ADAPTER_TYPE" \
    --save-path "$SAVE_PATH" 2>&1 | tee "$LOG_FILE"
