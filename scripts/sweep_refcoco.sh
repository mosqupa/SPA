#!/bin/bash
# RefCOCO val sweep over keep_ratio × pe_scale, distributed across GPUs.
#
# Usage:
#   bash scripts/sweep_refcoco.sh                       # auto-detect all GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/sweep_refcoco.sh  # restrict to specific GPUs
#
# Multi-node (same command on every node):
#   N_NODES=2 NODE_RANK=0 bash scripts/sweep_refcoco.sh
#   N_NODES=2 NODE_RANK=1 bash scripts/sweep_refcoco.sh
#
# Overridable env:
#   KEEP_RATIOS      space-separated ratios      (default "0.75 0.5 0.25 0.125")
#   PE_SCALES        space-separated pe scales   (default "0.01 0.03 0.05 0.1 0.2 0.5")
#   SPLIT            default refcoco_val_questions
#   MODEL_PATH       default models/llava-v1.5-7b
#   MODEL_NAME       default llava-v1.5-7b
#   PRUNING_METHOD   default random
#   SKIP_EXISTING=1  skip combos whose metrics.txt already exists (default off)

# NOTE: not `set -e` — one failed combo must not abort the rest of the sweep.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="/opt/conda/envs/vlm/bin/python"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1

SPLIT="${SPLIT:-refcoco_val_questions}"
MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
MODEL_NAME="${MODEL_NAME:-llava-v1.5-7b}"
PRUNING_METHOD="${PRUNING_METHOD:-random}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
N_NODES="${N_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

read -ra KEEP_RATIOS <<< "${KEEP_RATIOS:-0.75 0.5 0.25 0.125}"
read -ra PE_SCALES <<< "${PE_SCALES:-0.5}"

# GPU list: explicit CUDA_VISIBLE_DEVICES wins; otherwise auto-detect all GPUs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPULIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    [ ${#GPULIST[@]} -gt 0 ] || GPULIST=(0)   # fallback: no nvidia-smi → assume GPU 0
fi
N_GPUS=${#GPULIST[@]}

# Build task list: one entry per (keep_ratio, pe_scale) combo
TASKS=()
for kr in "${KEEP_RATIOS[@]}"; do
    for ps in "${PE_SCALES[@]}"; do
        TASKS+=("$kr $ps")
    done
done
N_TASKS=${#TASKS[@]}

# Task i → node (i % N_NODES); within a node → GPU slot ((i / N_NODES) % N_GPUS)
is_mine() { [ $(( $1 % N_NODES )) -eq "$NODE_RANK" ]; }
gpu_slot() { echo $(( ($1 / N_NODES) % N_GPUS )); }

LOG_DIR="$PROJECT_ROOT/data/refcoco/sweep_logs"
mkdir -p "$LOG_DIR"

# Track python PIDs so Ctrl+C can kill the whole tree.
# (Background jobs in a non-interactive shell ignore SIGINT, so a plain ^C
# only kills this script and leaves workers + python running on the GPUs.)
PY_PID_FILE="$LOG_DIR/.pids"
: > "$PY_PID_FILE"
WORKER_PIDS=()

_CLEANUP_DONE=0
cleanup() {
    # Re-entry guard: the group kill below TERMs ourselves too, which would
    # re-trigger this trap forever without the guard.
    [ "$_CLEANUP_DONE" = "1" ] && exit 130
    _CLEANUP_DONE=1
    echo "Interrupted — killing workers..."
    # 1. Kill worker subshells first so they can't spawn new python tasks.
    [ ${#WORKER_PIDS[@]} -gt 0 ] && kill "${WORKER_PIDS[@]}" 2>/dev/null
    # 2. Kill recorded python processes.
    [ -s "$PY_PID_FILE" ] && xargs -r kill < "$PY_PID_FILE" 2>/dev/null
    # 3. Belt & suspenders: kill the whole process group (only if we lead one).
    if [ "$$" = "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then
        kill -TERM -- -"$$" 2>/dev/null
    fi
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

echo "=============================================="
echo "  RefCOCO sweep — $SPLIT"
echo "  keep_ratio: ${KEEP_RATIOS[*]}"
echo "  pe_scale:   ${PE_SCALES[*]}"
echo "  combos:     $N_TASKS  (this node: $(( (N_TASKS + N_NODES - 1 - NODE_RANK) / N_NODES )))"
echo "  GPUs:       ${GPULIST[*]} (node $NODE_RANK/$N_NODES)"
echo "=============================================="

run_task() { # $1 = keep_ratio, $2 = pe_scale
    local kr=$1 ps=$2
    local tag="${PRUNING_METHOD}_${kr}_2dpe_${ps}"
    local out_dir="$PROJECT_ROOT/data/refcoco/answers/$SPLIT/$MODEL_NAME/$tag"
    local log_file="$LOG_DIR/$tag.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_dir/metrics.txt" ]; then
        echo "[skip] $tag"
        return 0
    fi

    "$CONDA_PYTHON" scripts/refcoco_inference.py \
        --split "$SPLIT" \
        --model-path "$MODEL_PATH" \
        --model-name "$MODEL_NAME" \
        --data-dir data/refcoco \
        --pruning-method "$PRUNING_METHOD" \
        --keep-ratio "$kr" \
        --use-2d-pe \
        --pe-scale "$ps" > "$log_file" 2>&1 &
    local py_pid=$!
    echo "$py_pid" >> "$PY_PID_FILE"
    if ! wait "$py_pid"; then
        echo "[FAIL] $tag  (log: $log_file)"
        return 1
    fi
    echo "[ ok ] $tag"
}

worker() { # $1 = GPU slot index on this node
    local slot=$1 gpu=${GPULIST[$slot]}
    local i
    for ((i = 0; i < N_TASKS; i++)); do
        is_mine "$i" || continue
        [ "$(gpu_slot "$i")" -eq "$slot" ] || continue
        read -r kr ps <<< "${TASKS[$i]}"
        local tag="${PRUNING_METHOD}_${kr}_2dpe_${ps}"
        echo "[gpu $gpu] $tag"
        if ! CUDA_VISIBLE_DEVICES="$gpu" run_task "$kr" "$ps"; then
            echo "$tag" >> "$FAILED_FILE"
        fi
    done
}

FAILED_FILE="$LOG_DIR/.failed_${NODE_RANK}"
: > "$FAILED_FILE"

for slot in $(seq 0 $((N_GPUS - 1))); do
    ( worker "$slot" || true ) &
    WORKER_PIDS+=($!)
done
wait

if [ -s "$FAILED_FILE" ]; then
    echo "Failed combos on node $NODE_RANK: $(tr '\n' ' ' < "$FAILED_FILE")"
fi

# --- Aggregate: per-combo scores → summary matrix (run only on node 0) ---
if [ "$NODE_RANK" = "0" ]; then
    KEEP_RATIOS_STR="${KEEP_RATIOS[*]}" PE_SCALES_STR="${PE_SCALES[*]}" \
    SPLIT="$SPLIT" MODEL_NAME="$MODEL_NAME" PRUNING_METHOD="$PRUNING_METHOD" \
    PROJECT_ROOT="$PROJECT_ROOT" "$CONDA_PYTHON" - <<'PYEOF'
import os
from pathlib import Path

root = Path(os.environ["PROJECT_ROOT"])
split = os.environ["SPLIT"]
model = os.environ["MODEL_NAME"]
method = os.environ["PRUNING_METHOD"]
keep_ratios = os.environ["KEEP_RATIOS_STR"].split()
pe_scales = os.environ["PE_SCALES_STR"].split()

def load_score(kr, ps):
    metrics = root / "data/refcoco/answers" / split / model / f"{method}_{kr}_2dpe_{ps}" / "metrics.txt"
    if not metrics.is_file():
        return None
    for line in metrics.read_text().splitlines():
        if line.strip().startswith("Acc@IoU=0.5:"):
            return float(line.split("(")[1].rstrip("%)"))
    return None

print(f"\n{'='*60}\nSummary — Acc@IoU=0.5 (%)\n{'='*60}")
header = "keep_ratio\\pe_scale " + "".join(f"{float(p):>10}" for p in pe_scales)
print(header)
rows = []
for kr in keep_ratios:
    cells = []
    for ps in pe_scales:
        s = load_score(kr, ps)
        cells.append("" if s is None else f"{s:10.2f}")
        rows.append([kr, ps, s])
    print(f"{float(kr):>16} " + "".join(cells))

csv_path = root / "data/refcoco/sweep_logs" / f"summary_{method}.csv"
with open(csv_path, "w") as f:
    f.write("keep_ratio,pe_scale,acc@0.5\n")
    for kr, ps, s in rows:
        f.write(f"{kr},{ps},{s if s is not None else ''}\n")
print(f"\nCSV saved: {csv_path}")
missing = [r for r in rows if r[2] is None]
if missing:
    print(f"WARNING: {len(missing)} combo(s) missing — rerun with SKIP_EXISTING=1 to fill gaps")
PYEOF
else
    echo "Aggregation skipped (node $NODE_RANK/$N_NODES — run it on node 0)"
fi

echo "Done. Logs: $LOG_DIR"
