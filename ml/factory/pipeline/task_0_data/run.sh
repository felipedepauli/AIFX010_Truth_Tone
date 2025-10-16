#!/usr/bin/env bash
set -e
################################################################################
# Build unified manifests for audio deepfake datasets (ADD, PartialSpoof, HAD)
# Felipe de Pauli — 2025
#
# Example:
#   bash run.sh --add --partialspoof --had
#   bash run.sh --add-root data/raw/ADD/ADD_train_dev --combine
################################################################################

# ---------- CONFIG DEFAULTS ----------
ROOT_DIR="ml/data/raw"
OUT_DIR="ml/data/processed"
PYTHON_BIN="python3"
SCRIPT="ml/factory/pipeline/task_0_data/build_manifests.py"

# ---------- ARGUMENT PARSER ----------
USE_ADD=false
USE_PS=false
USE_HAD=false
DO_COMBINE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --add) USE_ADD=true ;;
    --partialspoof|--ps) USE_PS=true ;;
    --had) USE_HAD=true ;;
    --combine) DO_COMBINE=true ;;
    --add-root) ADD_ROOT="$2"; shift ;;
    --partialspoof-root|--ps-root) PS_ROOT="$2"; shift ;;
    --had-root) HAD_ROOT="$2"; shift ;;
    --out|--combine-out) OUT_DIR="$2"; shift ;;
    --python) PYTHON_BIN="$2"; shift ;;
    *) echo "[WARN] Unknown option: $1" ;;
  esac
  shift
done

# ---------- DETECT ROOTS ----------
[[ -z "$ADD_ROOT" && "$USE_ADD" == true ]] && ADD_ROOT="$ROOT_DIR/ADD/ADD_train_dev"
[[ -z "$PS_ROOT"  && "$USE_PS"  == true ]] && PS_ROOT="$ROOT_DIR/PartialSpoof"
[[ -z "$HAD_ROOT" && "$USE_HAD" == true ]] && HAD_ROOT="$ROOT_DIR/HAD"

# ---------- SUMMARY ----------
echo "============================================================"
echo "[BUILD MANIFESTS]"
[[ "$USE_ADD" == true ]] && echo "  • ADD ............... $ADD_ROOT"
[[ "$USE_PS"  == true ]] && echo "  • PartialSpoof ...... $PS_ROOT"
[[ "$USE_HAD" == true ]] && echo "  • HAD ............... $HAD_ROOT"
[[ "$DO_COMBINE" == true ]] && echo "  • Combine output .... $OUT_DIR"
echo "============================================================"

# ---------- EXECUTION ----------
CMD=("$PYTHON_BIN" "$SCRIPT")

[[ "$USE_ADD" == true ]] && CMD+=("--add-root" "$ADD_ROOT")
[[ "$USE_PS"  == true ]] && CMD+=("--partialspoof-root" "$PS_ROOT")
[[ "$USE_HAD" == true ]] && CMD+=("--had-root" "$HAD_ROOT")
[[ "$DO_COMBINE" == true ]] && CMD+=("--combine-out" "$OUT_DIR")

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"

echo "============================================================"
echo "[DONE] Manifests built successfully."
echo "Check:"
[[ "$USE_ADD" == true ]] && echo "  → $ADD_ROOT/_manifests/"
[[ "$USE_PS"  == true ]] && echo "  → $PS_ROOT/_manifests/"
[[ "$USE_HAD" == true ]] && echo "  → $HAD_ROOT/_manifests/"
[[ "$DO_COMBINE" == true ]] && echo "  → $OUT_DIR/"
echo "============================================================"
