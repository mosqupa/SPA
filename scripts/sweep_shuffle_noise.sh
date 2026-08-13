#!/bin/bash
# RefCOCO val ablation: keep_ratio × pe_scale × (shuffle-PE | PE-noise), multi-GPU.
#
# Design (all with --use-2d-pe):
#   keep_ratio e.g. 0.5, 0.25
#   pe_scale   e.g. 0.5, 0.75
#   mode shuffle:  --shuffle-pe (PE positions shuffled, no noise)
#   mode noise:    --use-noise  (PE positions kept, noise added)
# => |keep_ratios| × |pe_scales| × |modes| experiments, round-robin across GPUs.
#
# Usage:
#   bash scripts/sweep_shuffle_noise.sh                       # auto-detect all GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/sweep_shuffle_noise.sh
#   KEEP_RATIOS="0.25" PE_SCALES="0.75" bash scripts/sweep_shuffle_noise.sh
#
# Overridable env:
#   KEEP_RATIOS      space-separated ratios   (default "0.5 0.25")
#   PE_SCALES        space-separated scales   (default "0.5")
#   MODES            space-separated modes    (default "shuffle noise")
#   SPLIT            default refcoco_val_questions
#   MODEL_PATH       default models/llava-v1.5-7b
#   MODEL_NAME       default llava-v1.5-7b
#   PRUNING_METHOD   default random
#   CONDA_PYTHON     default /opt/conda/envs/vlm/bin/python
#   SKIP_EXISTING=1  skip combos whose metrics.txt already exists (default on)

# NOTE: not `set -e` — one failed combo must not abort the rest of the sweep.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="${CONDA_PYTHON:-/opt/conda/envs/vlm/bin/python}"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1

SPLIT="${SPLIT:-refcoco_val_questions}"
MODEL_PATH="${MODEL_PATH:-models/llava-v1.5-7b}"
MODEL_NAME="${MODEL_NAME:-llava-v1.5-7b}"
PRUNING_METHOD="${PRUNING_METHOD:-random}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

read -ra KEEP_RATIOS <<< "${KEEP_RATIOS:-0.25}"
read -ra PE_SCALES <<< "${PE_SCALES:-0.75}"
read -ra MODES <<< "${MODES:-shuffle noise}"

# GPU list: explicit CUDA_VISIBLE_DEVICES wins; otherwise auto-detect all GPUs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPULIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    [ ${#GPULIST[@]} -gt 0 ] || GPULIST=(0)   # fallback: no nvidia-smi → assume GPU 0
fi
N_GPUS=${#GPULIST[@]}

# Build task list: one entry per (keep_ratio, pe_scale, mode) combo
TASKS=()
for kr in "${KEEP_RATIOS[@]}"; do
    for scale in "${PE_SCALES[@]}"; do
        for mode in "${MODES[@]}"; do
            TASKS+=("$kr $scale $mode")
        done
    done
done
N_TASKS=${#TASKS[@]}

LOG_DIR="$PROJECT_ROOT/data/refcoco/sweep_logs"
mkdir -p "$LOG_DIR"

# Track python PIDs so Ctrl+C can kill the whole tree.
PY_PID_FILE="$LOG_DIR/.pids_shuffle_noise"
: > "$PY_PID_FILE"
WORKER_PIDS=()

_CLEANUP_DONE=0
cleanup() {
    [ "$_CLEANUP_DONE" = "1" ] && exit 130
    _CLEANUP_DONE=1
    echo "Interrupted — killing workers..."
    [ ${#WORKER_PIDS[@]} -gt 0 ] && kill "${WORKER_PIDS[@]}" 2>/dev/null
    [ -s "$PY_PID_FILE" ] && xargs -r kill < "$PY_PID_FILE" 2>/dev/null
    if [ "$$" = "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then
        kill -TERM -- -"$$" 2>/dev/null
    fi
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

echo "=============================================="
echo "  RefCOCO shuffle/noise ablation — $SPLIT"
echo "  keep_ratio: ${KEEP_RATIOS[*]}  pe_scale: ${PE_SCALES[*]}"
echo "  modes:      ${MODES[*]}"
echo "  combos:     $N_TASKS"
echo "  GPUs:       ${GPULIST[*]}"
echo "=============================================="

tag_for() { # $1 = keep_ratio, $2 = pe_scale, $3 = mode
    local kr=$1 scale=$2 mode=$3
    case "$mode" in
        shuffle) echo "${PRUNING_METHOD}_${kr}_2dpe_${scale}_shuffle" ;;
        noise)   echo "${PRUNING_METHOD}_${kr}_2dpe_${scale}_noise" ;;
    esac
}

run_task() { # $1 = keep_ratio, $2 = pe_scale, $3 = mode
    local kr=$1 scale=$2 mode=$3
    local tag; tag=$(tag_for "$kr" "$scale" "$mode")
    local out_dir="$PROJECT_ROOT/data/refcoco/answers/$SPLIT/$MODEL_NAME/$tag"
    local log_file="$LOG_DIR/$tag.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_dir/metrics.txt" ]; then
        echo "[skip] $tag"
        return 0
    fi

    local mode_flag
    case "$mode" in
        shuffle) mode_flag="--shuffle-pe" ;;
        noise)   mode_flag="--use-noise" ;;
    esac

    "$CONDA_PYTHON" scripts/refcoco_inference.py \
        --split "$SPLIT" \
        --model-path "$MODEL_PATH" \
        --model-name "$MODEL_NAME" \
        --data-dir data/refcoco \
        --pruning-method "$PRUNING_METHOD" \
        --keep-ratio "$kr" \
        --use-2d-pe \
        --pe-scale "$scale" \
        $mode_flag > "$log_file" 2>&1 &
    local py_pid=$!
    echo "$py_pid" >> "$PY_PID_FILE"
    if ! wait "$py_pid"; then
        echo "[FAIL] $tag  (log: $log_file)"
        return 1
    fi
    echo "[ ok ] $tag"
}

worker() { # $1 = GPU slot index
    local slot=$1 gpu=${GPULIST[$slot]}
    local i
    for ((i = 0; i < N_TASKS; i++)); do
        [ $((i % N_GPUS)) -eq "$slot" ] || continue
        read -r kr scale mode <<< "${TASKS[$i]}"
        local tag; tag=$(tag_for "$kr" "$scale" "$mode")
        echo "[gpu $gpu] $tag"
        if ! CUDA_VISIBLE_DEVICES="$gpu" run_task "$kr" "$scale" "$mode"; then
            echo "$tag" >> "$FAILED_FILE"
        fi
    done
}

FAILED_FILE="$LOG_DIR/.failed_shuffle_noise"
: > "$FAILED_FILE"

for slot in $(seq 0 $((N_GPUS - 1))); do
    ( worker "$slot" || true ) &
    WORKER_PIDS+=($!)
done
wait

if [ -s "$FAILED_FILE" ]; then
    echo "Failed combos: $(tr '\n' ' ' < "$FAILED_FILE")"
fi

# --- Aggregate: per-combo scores → summary per pe_scale ---
KEEP_RATIOS_STR="${KEEP_RATIOS[*]}" PE_SCALES_STR="${PE_SCALES[*]}" MODES_STR="${MODES[*]}" \
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
modes = os.environ["MODES_STR"].split()

def tag_for(kr, scale, mode):
    return f"{method}_{kr}_2dpe_{scale}_{mode}"

def load_score(kr, scale, mode):
    metrics = root / "data/refcoco/answers" / split / model / f"{tag_for(kr, scale, mode)}" / "metrics.txt"
    if not metrics.is_file():
        return None
    for line in metrics.read_text().splitlines():
        if line.strip().startswith("Acc@IoU=0.5:"):
            return float(line.split("(")[1].rstrip("%)"))
    return None

rows = []
for scale in pe_scales:
    print(f"\n{'='*60}\nSummary — Acc@IoU=0.5 (%)  (pe_scale={scale})\n{'='*60}")
    print(f"keep_ratio\\mode" + "".join(f"{m:>14}" for m in modes))
    for kr in keep_ratios:
        cells = []
        for mode in modes:
            s = load_score(kr, scale, mode)
            cells.append("" if s is None else f"{s:14.2f}")
            rows.append([kr, scale, mode, s])
        print(f"{float(kr):>12} " + "".join(cells))

csv_path = root / "data/refcoco/sweep_logs" / "summary_shuffle_noise.csv"
with open(csv_path, "w") as f:
    f.write("keep_ratio,pe_scale,mode,acc@0.5\n")
    for kr, scale, mode, s in rows:
        f.write(f"{kr},{scale},{mode},{s if s is not None else ''}\n")
print(f"\nCSV saved: {csv_path}")
missing = [r for r in rows if r[3] is None]
if missing:
    print(f"WARNING: {len(missing)} combo(s) missing — rerun with SKIP_EXISTING=1 to fill gaps")
PYEOF

echo "Done. Logs: $LOG_DIR"
