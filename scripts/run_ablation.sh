#!/bin/bash
# PPD Ablation Runner — Multi-seed, multi-task experiments
# =============================================================
# Usage:
#   ./scripts/run_ablation.sh <task_name>          # single task, seeds 0-7
#   SEED_START=0 SEED_END=0 ./scripts/run_ablation.sh AntMorphology-Exact-v0
#   ALPHA=0.0 ./scripts/run_ablation.sh Superconductor-RandomForest-v0
#   RUN_EVAL=true ./scripts/run_ablation.sh ...
#
# Config via env vars:
#   ALPHA=0.1             distill loss weight (0=offline-only)
#   UW_BETA=1.0           uncertainty weighting (0=off)
#   GLOBAL_REFINE_RATIO=2 refinement samples per anchor
#   DISTILL_MIN_PRED_Q=0.0 quality gate threshold (0.8 for discrete)
#   SKIP_OOD=false        skip OOD filter
#   GPU=0                 GPU device
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
PIPELINE="$ROOT_DIR/run.py"
EVAL_SCRIPT="$ROOT_DIR/eval/eval_design_bench.py"

TASK_NAME="${1:?Usage: $0 <task_name>}"

# ── Config ─────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/$TASK_NAME}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-7}"
GPU="${GPU:-0}"

# Distillation
ALPHA="${ALPHA:-0.15}"
TFM_SYNTH_RATIO="${TFM_SYNTH_RATIO:-0.005}"
UW_BETA="${UW_BETA:-1.0}"
GLOBAL_REFINE_RATIO="${GLOBAL_REFINE_RATIO:-2}"
GLOBAL_SAMPLING="${GLOBAL_SAMPLING:-sobol}"
DISTILL_MIN_PRED_Q="${DISTILL_MIN_PRED_Q:-0.0}"  # 0.8 for discrete tasks (TFBind)
SKIP_OOD="${SKIP_OOD:-false}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-4096}"

# MLP training
EPOCHS="${EPOCHS:-100}"
LIST_LOSS="${LIST_LOSS:-listnet}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"

# Search
SEARCH_LR="${SEARCH_LR:-1e-3}"
SEARCH_STEPS="${SEARCH_STEPS:-200}"

# Eval
RUN_EVAL="${RUN_EVAL:-true}"
PIPELINE_ENV="${PIPELINE_ENV:-}"
DESIGN_BENCH_ENV="${DESIGN_BENCH_ENV:-design_bench}"
DRY_RUN="${DRY_RUN:-false}"

# Per-task overrides (large datasets)
case "$TASK_NAME" in
    TFBind10*)
        MAX_SAMPLES=10000
        TEACHER_CTX_MAX=100000
        ;;
    *)
        MAX_SAMPLES=-1
        TEACHER_CTX_MAX=0
        ;;
esac

DATA_X="$DATA_DIR/${TASK_NAME}_X.npy"
DATA_Y="$DATA_DIR/${TASK_NAME}_y.npy"

if [[ ! -f "$DATA_X" ]]; then
    echo "ERROR: Data not found: $DATA_X"
    echo "Place design_bench data under $DATA_DIR/"
    exit 1
fi

# ── GPU validation ─────────────────────────────────────────
if [[ -n "$GPU" ]] && command -v nvidia-smi &>/dev/null; then
    _max_gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sort -n | tail -1 || echo "-1")
    if [[ "$GPU" -gt "$_max_gpu" ]]; then
        echo "ERROR: GPU $GPU does not exist (available: 0..$_max_gpu)"
        exit 1
    fi
fi

echo "============================================================"
echo "PPD — $TASK_NAME"
echo "Alpha=$ALPHA  UW_BETA=$UW_BETA  refine=$GLOBAL_REFINE_RATIO  sampling=$GLOBAL_SAMPLING"
echo "min_pred_q=$DISTILL_MIN_PRED_Q  skip_ood=$SKIP_OOD"
echo "Seeds: $SEED_START..$SEED_END"
echo "Pipeline env: $PIPELINE_ENV  |  Eval env: $DESIGN_BENCH_ENV"
echo "Results: $RESULTS_DIR"
echo "============================================================"

# Verify pipeline env exists
if ! conda env list 2>/dev/null | grep -q "^$PIPELINE_ENV "; then
    echo "ERROR: conda env '$PIPELINE_ENV' not found."
    echo "Create it: conda create -n $PIPELINE_ENV python=3.10 -y && conda activate $PIPELINE_ENV && pip install tabpfn scikit-learn scipy torch"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
    SAVE_PATH="$RESULTS_DIR/${TASK_NAME}_seed${SEED}.npy"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY_RUN] seed=$SEED -> $SAVE_PATH"
        continue
    fi

    echo ""
    echo "── seed $SEED ──"

    conda run -n "$PIPELINE_ENV" python -u "$PIPELINE" \
        --task_name "$TASK_NAME" \
        --data_X "$DATA_X" \
        --data_y "$DATA_Y" \
        --save_path "$SAVE_PATH" \
        --seed "$SEED" \
        --gpu "$GPU" \
        --maximize_flag true \
        --alpha "$ALPHA" \
        --teacher_synth_ratio "$TFM_SYNTH_RATIO" \
        --teacher_predict_batch_size "$PREDICT_BATCH_SIZE" \
        --max_samples "$MAX_SAMPLES" \
        --teacher_context_max "$TEACHER_CTX_MAX" \
        --global_sampling_method "$GLOBAL_SAMPLING" \
        --global_refine_ratio "$GLOBAL_REFINE_RATIO" \
        --uncertainty_weight_beta "$UW_BETA" \
        --distill_min_pred_quantile "$DISTILL_MIN_PRED_Q" \
        --skip_ood_filter "$SKIP_OOD" \
        --epochs "$EPOCHS" \
        --list_loss "$LIST_LOSS" \
        --learning_rate "$LEARNING_RATE" \
        --search_lr "$SEARCH_LR" \
        --search_steps "$SEARCH_STEPS" \
        2>&1 | tail -40

    echo "  seed $SEED done."
done

# ── Eval ──
if [[ "$RUN_EVAL" == "true" && "$DRY_RUN" != "true" ]]; then
    echo ""
    echo "=== Running oracle eval ==="

    # MuJoCo license path for Ant/DKitty tasks
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/lxt/.mujoco/mujoco200/bin"

    EVAL_PATHS=()
    for SEED in $(seq "$SEED_START" "$SEED_END"); do
        EVAL_PATHS+=("$RESULTS_DIR/${TASK_NAME}_seed${SEED}.npy")
    done

    if conda env list 2>/dev/null | grep -q "^$DESIGN_BENCH_ENV "; then
        conda run -n "$DESIGN_BENCH_ENV" python "$EVAL_SCRIPT" \
            --task "$TASK_NAME" \
            --designs_paths "${EVAL_PATHS[@]}" \
            --results_dir "$RESULTS_DIR" \
            --method "PPD"
    else
        echo "[Warn] Eval env '$DESIGN_BENCH_ENV' not found. Skipping eval."
        echo "Raw candidates saved. Eval manually later."
    fi
fi

echo ""
echo "Done. Results in: $RESULTS_DIR"
