#!/bin/bash
# TextVQA val sweep: keep_ratio x {no adapter, adapter}, one job per GPU.
#
# Usage:
#   bash scripts/eval_textvqa.sh                            # uses the CONFIG block below
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval_textvqa.sh   # restrict GPUs
#   KEEP_RATIOS="1.0 0.5 0.25" bash scripts/eval_textvqa.sh
#
# Each combo runs scripts/textvqa_inference.py with an explicit env. Results
# land in data/textvqa/answers/<model>/random_<kr>[_adapter]/; per-job logs in
# LOG_DIR; a summary matrix is printed at the end.

# NOTE: not `set -e` — one failed combo must not abort the rest of the sweep.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1

# ============================ CONFIG ============================
: "${KEEP_RATIOS:=0.5 0.25}"         # space-separated keep ratios
: "${USE_ADAPTER:=no adapter}" # space-separated adapter variants (no / adapter)
: "${ADAPTER_PATH:=outputs/pos_adapter.pt}"
: "${MODEL_NAME:=llava-v1.5-7b}"
: "${PRUNING_METHOD:=random}"
: "${CONDA_PYTHON:=/opt/conda/envs/vlm/bin/python}"
: "${SKIP_EXISTING:=0}"              # skip combos whose metrics.txt exists
: "${LOG_DIR:=$PROJECT_ROOT/outputs/textvqa_logs}"
# ================================================================

mkdir -p "$LOG_DIR"
read -ra KEEP_RATIOS <<< "$KEEP_RATIOS"
read -ra ADAPTERS <<< "$USE_ADAPTER"

# GPU list: explicit CUDA_VISIBLE_DEVICES wins; otherwise auto-detect all GPUs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPULIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    [ ${#GPULIST[@]} -gt 0 ] || GPULIST=(0)   # fallback: no nvidia-smi → assume GPU 0
fi
N_GPUS=${#GPULIST[@]}

# Build task list: one entry per (keep_ratio, adapter) combo
TASKS=()
for kr in "${KEEP_RATIOS[@]}"; do
    for a in "${ADAPTERS[@]}"; do
        TASKS+=("$kr $a")
    done
done
N_TASKS=${#TASKS[@]}

echo "=============================================="
echo "  TextVQA val sweep"
echo "  keep_ratio: ${KEEP_RATIOS[*]}"
echo "  variants:   ${ADAPTERS[*]} ($ADAPTER_PATH)"
echo "  combos:     $N_TASKS"
echo "  GPUs:       ${GPULIST[*]}"
echo "=============================================="

run_task() { # $1 = keep_ratio, $2 = variant (no|adapter)
    local kr=$1 v=$2
    local tag="${PRUNING_METHOD}_${kr}"
    [ "$v" = "adapter" ] && tag="${tag}_adapter"
    local out_dir="$PROJECT_ROOT/data/textvqa/answers/$MODEL_NAME/$tag"
    local log_file="$LOG_DIR/keep${kr}_${v}.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_dir/metrics.txt" ]; then
        echo "[skip] $tag"
        return 0
    fi

    local ADAPTER_ARGS=""
    if [ "$v" = "adapter" ]; then
        ADAPTER_ARGS="--use-pos-adapter --adapter-path $ADAPTER_PATH"
    fi
    "$CONDA_PYTHON" "$PROJECT_ROOT/scripts/inference_textvqa.py" \
        --model-name "$MODEL_NAME" \
        --keep-ratio "$kr" \
        --pruning-method "$PRUNING_METHOD" \
        $ADAPTER_ARGS > "$log_file" 2>&1 &
    local py_pid=$!
    echo "$py_pid" >> "$PY_PID_FILE"
    if ! wait "$py_pid"; then
        echo "[FAIL] $tag  (log: $log_file)"
        return 1
    fi
    echo "[ ok ] $tag"
}

FAILED_FILE="$LOG_DIR/.failed"
: > "$FAILED_FILE"

# Track python PIDs so Ctrl+C can kill the whole tree (workers alone would
# leave orphaned python processes running the current combo).
PY_PID_FILE="$LOG_DIR/.pids"
: > "$PY_PID_FILE"
WORKER_PIDS=()

_CLEANUP_DONE=0
cleanup() {
    [ "$_CLEANUP_DONE" = "1" ] && exit 130
    _CLEANUP_DONE=1
    echo "Interrupted — killing workers..."
    [ ${#WORKER_PIDS[@]} -gt 0 ] && kill "${WORKER_PIDS[@]}" 2>/dev/null
    [ -s "$PY_PID_FILE" ] && xargs -r kill < "$PY_PID_FILE" 2>/dev/null
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

worker() { # $1 = GPU slot index
    local slot=$1 gpu=${GPULIST[$slot]}
    local i
    for ((i = slot; i < N_TASKS; i += N_GPUS)); do
        read -r kr v <<< "${TASKS[$i]}"
        echo "[gpu $gpu] keep=$kr variant=$v"
        if ! CUDA_VISIBLE_DEVICES="$gpu" run_task "$kr" "$v"; then
            echo "${PRUNING_METHOD}_${kr}_${v}" >> "$FAILED_FILE"
        fi
    done
}

for slot in $(seq 0 $((N_GPUS - 1))); do
    ( worker "$slot" || true ) &
    WORKER_PIDS+=($!)
done
wait

if [ -s "$FAILED_FILE" ]; then
    echo "Failed combos: $(tr '\n' ' ' < "$FAILED_FILE")"
fi

# --- Summary matrix ---
KEEP_RATIOS_STR="${KEEP_RATIOS[*]}" USE_ADAPTER_STR="${ADAPTERS[*]}" \
MODEL_NAME="$MODEL_NAME" PRUNING_METHOD="$PRUNING_METHOD" \
LOG_DIR="$LOG_DIR" PROJECT_ROOT="$PROJECT_ROOT" python3 - <<'PYEOF'
import os
from pathlib import Path

root = Path(os.environ["PROJECT_ROOT"])
model = os.environ["MODEL_NAME"]
method = os.environ["PRUNING_METHOD"]
keep_ratios = os.environ["KEEP_RATIOS_STR"].split()
variants = os.environ["USE_ADAPTER_STR"].split()

def load_score(kr, tag):
    metrics = root / "data/textvqa/answers" / model / f"{method}_{kr}{tag}" / "metrics.txt"
    if not metrics.is_file():
        return None
    for line in metrics.read_text().splitlines():
        if line.strip().startswith("Accuracy:"):
            return float(line.split(":")[1].strip().rstrip("%"))
    return None

print(f"\n{'='*60}\nSummary [TextVQA val] — accuracy (%)\n{'='*60}")
header = "keep_ratio\\variant" + "".join(f"{v:>14}" for v in variants)
print(header)
rows = []
for kr in keep_ratios:
    cells = []
    for v in variants:
        tag = "" if v == "no" else "_adapter"
        s = load_score(kr, tag)
        cells.append("" if s is None else f"{s:14.2f}")
        rows.append([kr, v, s])
    print(f"{float(kr):>12} " + "".join(cells))

csv_path = Path(os.environ["LOG_DIR"]) / "summary_textvqa.csv"
with open(csv_path, "w") as f:
    f.write("keep_ratio,variant,acc\n")
    for kr, v, s in rows:
        f.write(f"{kr},{v},{s if s is not None else ''}\n")
print(f"\nCSV saved: {csv_path}")
missing = [r for r in rows if r[2] is None]
if missing:
    print(f"WARNING: {len(missing)} combo(s) missing — rerun the sweep to fill gaps (SKIP_EXISTING skips done ones)")
PYEOF

echo "Done. Logs: $LOG_DIR"
