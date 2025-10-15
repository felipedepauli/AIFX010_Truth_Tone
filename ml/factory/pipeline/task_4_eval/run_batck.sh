#!/usr/bin/env bash
set -euo pipefail

# Batch inference runner for predict_batch.py

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../.. && pwd)"
PYTHON_BIN="${PYTHON:-python}"

CSV="${1:-${CSV:-$ROOT/ml/factory/pipeline/task_0_data/data_processed/dev_clips.csv}}"
MODEL="${2:-${MODEL:-$ROOT/ml/factory/pipeline/task_3_train/ml/factory/experiments/run0/clipclf.joblib}}"
SUFFIX="${SUFFIX:-.feat.npz}"
OUTDIR="${OUTDIR:-$ROOT/ml/factory/experiments/run0}"
TOPK="${TOPK:-25}"

mkdir -p "$OUTDIR"

echo "[INFO] CSV:      $CSV"
echo "[INFO] MODEL:    $MODEL"
echo "[INFO] SUFFIX:   $SUFFIX"
echo "[INFO] OUTDIR:   $OUTDIR"

# (Opcional) Auto-extrair sidecars se faltarem: AUTO_EXTRACT=1
if [[ "${AUTO_EXTRACT:-}" == "1" ]]; then
  echo "[AUTO] extracting missing sidecars using MFCC+Ambient..."
  "$PYTHON_BIN" "$ROOT/ml/factory/pipeline/task_1_preprocess/extract_features.py" \
    --csv "$CSV" \
    --suffix "$SUFFIX" \
    --use-mfcc --use-ambient \
    --sr 16000
fi

EXTRA_ARGS=()
if [[ -n "${THRESH:-}" ]]; then
  EXTRA_ARGS+=(--pos-class fake --threshold "$THRESH")
  echo "[INFO] Using threshold for 'fake': $THRESH"
fi

"$PYTHON_BIN" "$ROOT/ml/factory/pipeline/task_4_eval/predict_batch.py" \
  --csv "$CSV" \
  --model "$MODEL" \
  --suffix "$SUFFIX" \
  --out-json "$OUTDIR/dev_metrics.json" \
  --out-preds "$OUTDIR/dev_predictions.csv" \
  --dump-topk "$TOPK" \
  "${EXTRA_ARGS[@]}"

echo "[OK] Wrote:"
echo "  - $OUTDIR/dev_metrics.json"
echo "  - $OUTDIR/dev_predictions.csv"



try:
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
except Exception:
    pass
import gc; gc.collect()
