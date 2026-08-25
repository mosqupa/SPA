#!/bin/bash
# RefCOCO sweep over keep_ratio × PE variant, one eval per GPU in parallel.
#
# Usage:
#   bash scripts/sweep_refcoco.sh                        # uses the CONFIG block below
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/sweep_refcoco.sh   # restrict GPUs
#
# Multi-node (same command on every node):
#   N_NODES=2 NODE_RANK=0 bash scripts/sweep_refcoco.sh
#   N_NODES=2 NODE_RANK=1 bash scripts/sweep_refcoco.sh
#
# Each job runs scripts/eval_refcoco.sh with an explicit env (independent of
# that script's CONFIG block), so every (keep_ratio × variant) combo is a
# fully self-contained experiment. Results land in the usual answer dirs;
# per-job logs in LOG_DIR.

# NOTE: not `set -e` — one failed combo must not abort the rest of the sweep.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1

# ============================ CONFIG ============================
: "${SPLIT:=refcoco_plus_val_questions refcocog_val_questions}"  # default splits
: "${SPLITS:=$SPLIT}"                     # space-separated splits to sweep
: "${MODEL_NAME:=llava-v1.5-7b}"
: "${PRUNING_METHOD:=random}"
: "${KEEP_RATIOS:=0.25 0.125}"        # space-separated keep ratios
# PE variants (space-separated):
#   no2dpe           no positional encoding
#   2dpe / 2dpe_0.5  fixed 2D sincos PE (scale from token suffix or PE_SCALE)
#   shuffle          fixed 2D PE + shuffled positions
#   noise            fixed 2D PE replaced by unit-norm noise
#   adapter          trained PositionAdapter (scale from PE_SCALE)
#   adapter_pe0.5    trained PositionAdapter, explicit delta scale (single value)
#   adapter_shuffle  trained PositionAdapter + shuffled coords
: "${VARIANTS:=no2dpe adapter}"
: "${ADAPTER_PATH:=outputs/pos_adapter.pt}"
: "${ADAPTER_TYPE:=fourier}"   # fourier | raw — MUST match ADAPTER_PATH
: "${PE_SCALE:=}"
: "${CONDA_PYTHON:=/opt/conda/envs/vlm/bin/python}"
: "${SKIP_EXISTING:=1}"               # skip combos whose metrics.txt exists
: "${LOG_DIR:=$PROJECT_ROOT/outputs/sweep_logs}"
: "${N_NODES:=1}"
: "${NODE_RANK:=0}"
# ================================================================

mkdir -p "$LOG_DIR"
read -ra SPLITS <<< "$SPLITS"
read -ra KEEP_RATIOS <<< "$KEEP_RATIOS"
read -ra VARIANTS <<< "$VARIANTS"

# Empty PE_SCALE means a single scale of 1.0 — scale-sweep variants (adapter /
# 2dpe / shuffle / noise) must still expand to ONE task instead of zero.
[ -z "$PE_SCALE" ] && PE_SCALE=1.0

# GPU list: explicit CUDA_VISIBLE_DEVICES wins; otherwise auto-detect all GPUs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPULIST < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    [ ${#GPULIST[@]} -gt 0 ] || GPULIST=(0)   # fallback: no nvidia-smi → assume GPU 0
fi
N_GPUS=${#GPULIST[@]}

# Split a "variant@scale" task token into (variant, scale); for plain variants
# the scale falls back to the global PE_SCALE.
split_variant() { # $1=token -> sets V_TOKEN, V_SCALE
    local t=$1
    if [[ "$t" == *@* ]]; then V_TOKEN="${t%@*}"; V_SCALE="${t#*@}"; else V_TOKEN="$t"; V_SCALE=""; fi
}

# variant -> explicit env for eval_refcoco.sh (overrides its CONFIG block so a
# sweep job is independent of whatever defaults were last edited there)
variant_env() {
    split_variant "$1"
    local v=$V_TOKEN sc=${V_SCALE:-$PE_SCALE}
    case "$v" in
        no2dpe)
            echo "USE_2D_PE=false POS_ADAPTER=false SHUFFLE_PE=false NOISE=false SHUFFLE_COORDS=false";;
        2dpe)
            echo "USE_2D_PE=true PE_SCALE=$sc POS_ADAPTER=false SHUFFLE_PE=false NOISE=false SHUFFLE_COORDS=false";;
        2dpe_*)
            echo "USE_2D_PE=true PE_SCALE=${v#2dpe_} POS_ADAPTER=false SHUFFLE_PE=false NOISE=false SHUFFLE_COORDS=false";;
        shuffle)
            echo "USE_2D_PE=true PE_SCALE=$sc SHUFFLE_PE=true POS_ADAPTER=false NOISE=false SHUFFLE_COORDS=false";;
        noise)
            echo "USE_2D_PE=true PE_SCALE=$sc NOISE=true POS_ADAPTER=false SHUFFLE_PE=false SHUFFLE_COORDS=false";;
        adapter)
            echo "POS_ADAPTER=true ADAPTER_PATH=$ADAPTER_PATH ADAPTER_TYPE=$ADAPTER_TYPE PE_SCALE=$sc USE_2D_PE=false SHUFFLE_PE=false NOISE=false SHUFFLE_COORDS=false";;
        adapter_pe*)   # explicit per-scale token, e.g. adapter_pe0.25 (not expanded)
            echo "POS_ADAPTER=true ADAPTER_PATH=$ADAPTER_PATH ADAPTER_TYPE=$ADAPTER_TYPE PE_SCALE=${v#adapter_pe} USE_2D_PE=false SHUFFLE_PE=false NOISE=false SHUFFLE_COORDS=false";;
        adapter_shuffle)
            echo "POS_ADAPTER=true ADAPTER_PATH=$ADAPTER_PATH ADAPTER_TYPE=$ADAPTER_TYPE PE_SCALE=$sc SHUFFLE_COORDS=true USE_2D_PE=false SHUFFLE_PE=false NOISE=false";;
        *)
            echo "ERROR: unknown variant '$v' (see CONFIG block)" >&2; exit 1;;
    esac
}

# adapter tag base: inference_refcoco.py names raw-type results adapter_raw_pe*
adapter_base() { if [ "$ADAPTER_TYPE" = "raw" ]; then echo "adapter_raw"; else echo "adapter"; fi; }

# variant -> answer-dir tag (must match inference_refcoco.py's pe_tag)
variant_tag() {
    split_variant "$1"
    local v=$V_TOKEN sc=${V_SCALE:-$PE_SCALE}
    case "$v" in
        # no-PE results are written to random_<kr>/ with no suffix in
        # inference_refcoco.py (pe_tag="") — empty tag keeps the paths aligned.
        no2dpe)          echo "";;
        2dpe)            echo "2dpe_$sc";;
        2dpe_*)          echo "2dpe_${v#2dpe_}";;
        shuffle)         echo "2dpe_${sc}_shuffle";;
        noise)           echo "2dpe_${sc}_noise";;
        adapter)         echo "$(adapter_base)_pe$sc";;
        adapter_pe*)     echo "$(adapter_base)_pe${v#adapter_pe}";;
        adapter_shuffle) echo "$(adapter_base)_pe${sc}_shuffle";;
    esac
}

# Build task list: one entry per (split, keep_ratio, variant) combo.
# Variants that take a scale (adapter / 2dpe / shuffle / noise) expand across
# every value of PE_SCALE — a multi-value PE_SCALE is a sweep dimension, each
# scale becoming its own task (variant token "adapter@2.0").
SCALE_SWEEP_VARIANTS="adapter adapter_shuffle 2dpe shuffle noise"
TASKS=()
for sp in "${SPLITS[@]}"; do
    for kr in "${KEEP_RATIOS[@]}"; do
        for v in "${VARIANTS[@]}"; do
            if [[ " $SCALE_SWEEP_VARIANTS " == *" $v "* ]]; then
                for sc in $PE_SCALE; do
                    TASKS+=("$sp $kr $v@$sc")
                done
            else
                TASKS+=("$sp $kr $v")
            fi
        done
    done
done
N_TASKS=${#TASKS[@]}

# Task i → node (i % N_NODES); within a node → GPU slot ((i / N_NODES) % N_GPUS)
is_mine() { [ $(( $1 % N_NODES )) -eq "$NODE_RANK" ]; }
gpu_slot() { echo $(( ($1 / N_NODES) % N_GPUS )); }

# Track python PIDs so Ctrl+C can kill the whole tree.
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
    if [ "$$" = "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then
        kill -TERM -- -"$$" 2>/dev/null
    fi
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

echo "=============================================="
echo "  RefCOCO sweep"
echo "  splits:     ${SPLITS[*]}"
echo "  keep_ratio: ${KEEP_RATIOS[*]}"
echo "  variants:   ${VARIANTS[*]}"
echo "  combos:     $N_TASKS  (this node: $(( (N_TASKS + N_NODES - 1 - NODE_RANK) / N_NODES )))"
echo "  GPUs:       ${GPULIST[*]} (node $NODE_RANK/$N_NODES)"
echo "=============================================="

run_task() { # $1 = split, $2 = keep_ratio, $3 = variant (possibly variant@scale)
    local sp=$1 kr=$2 v=$3
    local vt
    vt="$(variant_tag "$v")"
    local tag="${PRUNING_METHOD}_${kr}"
    [ -n "$vt" ] && tag="${tag}_${vt}"   # no trailing underscore for no-PE
    local out_dir="$PROJECT_ROOT/data/refcoco/answers/$sp/$MODEL_NAME/$tag"
    local log_file="$LOG_DIR/${sp}_${tag}.log"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_dir/metrics.txt" ]; then
        echo "[skip] $tag"
        return 0
    fi

    local job_env
    job_env="$(variant_env "$v")"
    eval "env $job_env CONDA_PYTHON='$CONDA_PYTHON' KEEP_RATIO=$kr SPLIT=$sp \
        bash '$SCRIPT_DIR/eval_refcoco.sh'" > "$log_file" 2>&1 &
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
        read -r sp kr v <<< "${TASKS[$i]}"
        echo "[gpu $gpu] split=$sp variant=$v keep=$kr"
        if ! CUDA_VISIBLE_DEVICES="$gpu" run_task "$sp" "$kr" "$v"; then
            echo "$sp ${PRUNING_METHOD}_${kr}_$(variant_tag "$v")" >> "$FAILED_FILE"
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
    # variant -> tag table for the python summary
    VARIANT_TAGS=""
    for v in "${VARIANTS[@]}"; do
        VARIANT_TAGS+="$v:$(variant_tag "$v")"$'\n'
    done
    KEEP_RATIOS_STR="${KEEP_RATIOS[*]}" VARIANT_TAGS="$VARIANT_TAGS" \
    SPLITS_STR="${SPLITS[*]}" MODEL_NAME="$MODEL_NAME" PRUNING_METHOD="$PRUNING_METHOD" \
    LOG_DIR="$LOG_DIR" PROJECT_ROOT="$PROJECT_ROOT" python3 - <<'PYEOF'
import os
from pathlib import Path

root = Path(os.environ["PROJECT_ROOT"])
splits = os.environ["SPLITS_STR"].split()
model = os.environ["MODEL_NAME"]
method = os.environ["PRUNING_METHOD"]
keep_ratios = os.environ["KEEP_RATIOS_STR"].split()
variants = [line.split(":") for line in os.environ["VARIANT_TAGS"].splitlines() if line]

for split in splits:
    def load_score(kr, tag):
        # empty tag (no-PE) -> "random_0.5" with no trailing underscore
        dirname = f"{method}_{kr}" + (f"_{tag}" if tag else "")
        metrics = root / "data/refcoco/answers" / split / model / dirname / "metrics.txt"
        if not metrics.is_file():
            return None
        for line in metrics.read_text().splitlines():
            if line.strip().startswith("Acc@IoU=0.5:"):
                return float(line.split("(")[1].rstrip("%)"))
        return None

    print(f"\n{'='*70}\nSummary [{split}] — Acc@IoU=0.5 (%)\n{'='*70}")
    # column headers use the variant NAME (tags can be empty for no-PE)
    header = "keep_ratio\\variant " + "".join(f"{n:>18}" for n, _ in variants)
    print(header)
    rows = []
    for kr in keep_ratios:
        cells = []
        for name, tag in variants:
            s = load_score(kr, tag)
            cells.append("" if s is None else f"{s:18.2f}")
            rows.append([kr, name, s])
        print(f"{float(kr):>14} " + "".join(cells))

    csv_path = Path(os.environ["LOG_DIR"]) / f"summary_{method}_{split}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("keep_ratio,variant,acc@0.5\n")
        for kr, name, s in rows:
            f.write(f"{kr},{name},{s if s is not None else ''}\n")
    print(f"\nCSV saved: {csv_path}")
    missing = [r for r in rows if r[2] is None]
    if missing:
        print(f"WARNING: {len(missing)} combo(s) missing — rerun the sweep to fill gaps (SKIP_EXISTING skips done ones)")
PYEOF
else
    echo "Aggregation skipped (node $NODE_RANK/$N_NODES — run it on node 0)"
fi

echo "Done. Logs: $LOG_DIR"
