#!/bin/bash
# Official LLaVA v1.5 POPE evaluation pipeline.
#
# Usage:
#   bash scripts/official_eval_pope.sh
#
# Data setup (one-time):
#   python scripts/setup_official_gqa_pope.py

set -e
export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLAVA_ROOT="$PROJECT_ROOT/LLaVA"

cd "$LLAVA_ROOT"

# --- Config ---
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/llava-v1.5-7b}"
MODEL_NAME="${MODEL_NAME:-llava-v1.5-7b}"
LOAD_4BIT="${LOAD_4BIT:-1}"

FOURBIT_FLAG=""
if [ "$LOAD_4BIT" = "1" ]; then
    FOURBIT_FLAG="--load-4bit"
fi

echo "========================================="
echo "  POPE Evaluation)"
echo "  Model:  $MODEL_PATH"
echo "========================================="

# Step 1 — Inference
echo "[1/2] Running inference..."
python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    $FOURBIT_FLAG
echo "[1/2] Inference done."

# Step 2 — Evaluate with POPE scorer
echo "[2/2] Scoring..."
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$MODEL_NAME.jsonl
