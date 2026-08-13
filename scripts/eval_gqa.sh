#!/bin/bash
# Official LLaVA v1.5 GQA evaluation pipeline.
#
# Usage:
#   bash scripts/official_eval_gqa.sh                    # single GPU
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/official_eval_gqa.sh  # multi-GPU
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
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

# Build optional flags
FOURBIT_FLAG=""
if [ "$LOAD_4BIT" = "1" ]; then
    FOURBIT_FLAG="--load-4bit"
fi

# --- Multi-GPU ---
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "========================================="
echo "  GQA Evaluation"
echo "  Model:  $MODEL_PATH"
echo "  GPUs:   $gpu_list ($CHUNKS chunks)"
echo "========================================="

# Step 1 — Inference
echo "[1/3] Running inference..."
for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --question-file "./playground/data/eval/gqa/$SPLIT.jsonl" \
        --image-folder "./playground/data/eval/gqa/data/images" \
        --answers-file "./playground/data/eval/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        $FOURBIT_FLAG &
done
wait
echo "[1/3] Inference done."

# Step 2 — Merge chunks
echo "[2/3] Merging chunks..."
ANSWER_DIR="./playground/data/eval/gqa/answers/$SPLIT/$MODEL_NAME"
output_file="$ANSWER_DIR/merge.jsonl"
> "$output_file"
for IDX in $(seq 0 $((CHUNKS - 1))); do
    cat "$ANSWER_DIR/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
echo "[2/3] Merged → $output_file ($(wc -l < "$output_file") lines)"

# Step 3 — Convert + Evaluate
echo "[3/3] Converting & scoring..."
python scripts/convert_gqa_for_eval.py \
    --src "$output_file" \
    --dst "$GQADIR/testdev_balanced_predictions.json"

cd "$GQADIR"
python eval/eval.py --tier testdev_balanced
