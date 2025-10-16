#!/usr/bin/env bash
set -e
################################################################################
# AIFX010 - Hoax Hertz
# Automated training script for clip-level classifiers (SVM + LogReg)
# with Optuna hyperparameter optimization.
# Author: Felipe de Pauli
################################################################################

CSV_PATH="ml/factory/pipeline/task_0_data/data_processed/train_clips.csv"
EXP_DIR="ml/factory/experiments/run_auto"
mkdir -p "${EXP_DIR}"

# echo "======================================================================"
# echo "[1/2] Training SVM model with Optuna..."
# echo "======================================================================"

# python ml/factory/pipeline/task_3_train/train_clip_classifier.py \
#   --csv "${CSV_PATH}" \
#   --out-model "${EXP_DIR}/clipclf_svm_optuna.joblib" \
#   --out-report "${EXP_DIR}/clipclf_svm_optuna_report.json" \
#   --clf svm \
#   --use-optuna \
#   --optuna-ntrials 30

# echo "---------------------------------------------------------------------"
# echo "[SVM] Training complete."
# echo "---------------------------------------------------------------------"

echo "======================================================================"
echo "[2/2] Training Logistic Regression model with Optuna..."
echo "======================================================================"

python ml/factory/pipeline/task_3_train/train_clip_classifier.py \
  --csv "${CSV_PATH}" \
  --out-model "${EXP_DIR}/clipclf_logreg_optuna.joblib" \
  --out-report "${EXP_DIR}/clipclf_logreg_optuna_report.json" \
  --clf logreg \
  --use-optuna \
  --optuna-ntrials 30

echo "---------------------------------------------------------------------"
echo "[LogReg] Training complete."
echo "---------------------------------------------------------------------"

echo "======================================================================"
echo "[3/3] Generating summary report..."
echo "======================================================================"

python - <<'EOF'
import json, os, pathlib
exp_dir = pathlib.Path("ml/factory/experiments/run_auto")
svm_report = exp_dir / "clipclf_svm_optuna_report.json"
logreg_report = exp_dir / "clipclf_logreg_optuna_report.json"

def load(p):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

svm = load(svm_report)
logreg = load(logreg_report)

summary = {
    "svm_accuracy": svm.get("accuracy"),
    "logreg_accuracy": logreg.get("accuracy"),
    "svm_classes": svm.get("classes"),
    "logreg_classes": logreg.get("classes"),
    "best_model": "svm" if svm.get("accuracy", 0) > logreg.get("accuracy", 0) else "logreg"
}

with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
EOF

echo "======================================================================"
echo "[DONE] All trainings completed successfully!"
echo "======================================================================"
