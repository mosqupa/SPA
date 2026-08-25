#!/bin/bash

set -e
export HF_HUB_OFFLINE=1

cd /opt/data/private/VLM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================ CONFIG ============================
# Edit the defaults below, then one-click run:
#     bash scripts/eval_refcoco.sh
# (env vars set on the command line still override them, e.g.
#     KEEP_RATIO=0.25 bash scripts/eval_refcoco.sh)
: "${SPLIT:=refcoco_val_questions}"
: "${MODEL_PATH:=$PROJECT_ROOT/models/llava-v1.5-7b}"
: "${MODEL_NAME:=llava-v1.5-7b}"
: "${DATA_DIR:=$PROJECT_ROOT/data/refcoco}"
: "${CONDA_PYTHON:=/opt/conda/envs/vlm/bin/python}"
: "${PRUNING_METHOD:=random}"
: "${KEEP_RATIO:=0.5}"
: "${USE_2D_PE:=false}"          # true -> fixed 2D sincos PE (scaled by PE_SCALE)
: "${PE_SCALE:=}"  # space-separated list of PE scales to sweep; empty -> single run at 1.0
: "${SHUFFLE_PE:=false}"
: "${NOISE:=false}"
: "${POS_ADAPTER:=true}"       # true -> learnable PositionAdapter (replaces 2D PE)
: "${ADAPTER_PATH:=outputs/pos_adapter.pt}"           # e.g. outputs/pos_adapter.pt
: "${ADAPTER_TYPE:=fourier}"   # fourier (default) | raw — must match the checkpoint
: "${SHUFFLE_COORDS:=false}"    # shuffle the coords fed to the adapter
# ================================================================

if [ -z "$PE_SCALE" ]; then
    PE_SCALE=1.0
fi

PE_FLAG=""
if [ "$USE_2D_PE" = "true" ] || [ "$USE_2D_PE" = "1" ]; then
    PE_FLAG="--use-2d-pe"
fi
SHUFFLE_FLAG=""
if [ "$SHUFFLE_PE" = "true" ] || [ "$SHUFFLE_PE" = "1" ]; then
    SHUFFLE_FLAG="--shuffle-pe"
fi
NOISE_FLAG=""
if [ "$NOISE" = "true" ] || [ "$NOISE" = "1" ]; then
    NOISE_FLAG="--use-noise"
fi
ADAPTER_FLAG=""
if [ "$POS_ADAPTER" = "true" ] || [ "$POS_ADAPTER" = "1" ]; then
    PE_FLAG=""  # adapter replaces 2D PE; do not pass --use-2d-pe
    ADAPTER_FLAG="--use-pos-adapter"
    if [ -n "$ADAPTER_PATH" ]; then
        ADAPTER_FLAG="$ADAPTER_FLAG --adapter-path $ADAPTER_PATH"
    fi
    if [ -n "${ADAPTER_TYPE:-}" ]; then
        ADAPTER_FLAG="$ADAPTER_FLAG --adapter-type $ADAPTER_TYPE"
    fi
    if [ "$SHUFFLE_COORDS" = "true" ] || [ "$SHUFFLE_COORDS" = "1" ]; then
        ADAPTER_FLAG="$ADAPTER_FLAG --shuffle-coords"
    fi
fi

PE_DESC="none"
if [ -n "$ADAPTER_FLAG" ]; then
    PE_DESC="pos-adapter ($ADAPTER_TYPE, ${ADAPTER_PATH:-random init})"
    if [ "$SHUFFLE_COORDS" = "true" ] || [ "$SHUFFLE_COORDS" = "1" ]; then
        PE_DESC="$PE_DESC + shuffle_coords"
    fi
elif [ -n "$PE_FLAG" ]; then
    PE_DESC="2dpe (scale=$PE_SCALE)"
    [ -n "$SHUFFLE_FLAG" ] && PE_DESC="$PE_DESC + shuffle"
    [ -n "$NOISE_FLAG" ] && PE_DESC="$PE_DESC + noise"
fi

echo "========================================="
echo "  RefCOCO — $SPLIT"
echo "  Model: $MODEL_PATH"
echo "  Config: $PRUNING_METHOD pruning @ keep=$KEEP_RATIO | PE: $PE_DESC"
echo "  Scales: $PE_SCALE"
echo "========================================="

# Convert data for evaluation
"$CONDA_PYTHON" "$PROJECT_ROOT/scripts/refcoco_converter.py" \
    --splits-dir "$DATA_DIR" \
    --output-dir "$DATA_DIR/converted" \
    --splits "${SPLIT%.json}"

# Inference + Evaluation — one run per PE scale in the sweep
for SCALE in $PE_SCALE; do
    echo ""
    echo ">>> pe_scale = $SCALE"
    "$CONDA_PYTHON" "$PROJECT_ROOT/scripts/inference_refcoco.py" \
        --split "$SPLIT" \
        --model-path "$MODEL_PATH" \
        --model-name "$MODEL_NAME" \
        --data-dir "$DATA_DIR" \
        --pruning-method "$PRUNING_METHOD" \
        --keep-ratio "$KEEP_RATIO" \
        $PE_FLAG \
        --pe-scale "$SCALE" \
        $SHUFFLE_FLAG \
        $NOISE_FLAG \
        $ADAPTER_FLAG
done
