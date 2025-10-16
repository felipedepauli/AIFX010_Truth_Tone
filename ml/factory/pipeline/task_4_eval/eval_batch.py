#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference over a CSV of [path,label] (or a folder tree), using:
- a trained .joblib model (cuML or scikit-learn pipeline)
- precomputed sidecar feature files "<audio><suffix>" (e.g., .feat.npz)

Outputs:
- JSON metrics summary (accuracy, per-class report, confusion matrix, ROC-AUC if binary)
- CSV with per-file predictions and scores
- Optional top-K lists (highest "fake" and highest "real" probabilities)

Example:
  python ml/factory/pipeline/task_4_eval/predict_batch.py \
    --csv ml/factory/pipeline/task_0_data/data_processed/dev_clips.csv \
    --model ml/factory/experiments/run0/clipclf.joblib \
    --suffix .feat.npz \
    --out-json ml/factory/experiments/run0/dev_metrics.json \
    --out-preds ml/factory/experiments/run0/dev_predictions.csv \
    --dump-topk 25
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _load_pairs_from_csv(csv_path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row.get("path"); lab = row.get("label")
            if p and lab:
                rows.append((p, lab))
    return rows


def _scan_tree(root: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for label_dir in sorted(Path(root).glob("*")):
        if not label_dir.is_dir():
            continue
        lab = label_dir.name
        for audio in sorted(label_dir.rglob("*")):
            if audio.suffix.lower() in AUDIO_EXTS:
                pairs.append((str(audio), lab))
    return pairs


def _sidecar(audio_path: str, suffix: str) -> Path:
    p = Path(audio_path)
    return p.with_suffix(p.suffix + suffix)


def _load_sidecar(npz_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load vector + meta from a .feat.npz sidecar."""
    z = np.load(npz_path, allow_pickle=False)
    vec = z["vector"].astype(np.float32)
    meta: Dict[str, Any] = {}
    # meta is stored as JSON string (if present)
    try:
        m = z["meta"]
        if m.dtype.kind in {"U", "S", "O"}:
            meta = json.loads(m.item())
    except Exception:
        pass
    return vec, meta


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = e / np.sum(e)
    return s.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch inference from sidecar features")
    # data
    ap.add_argument("--csv", type=str, default=None, help="CSV with columns [path,label]")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Folder tree data_root/<label>/*.(wav|flac|mp3|m4a|ogg)")
    ap.add_argument("--suffix", type=str, default=".feat.npz",
                    help="Sidecar suffix (default: .feat.npz)")
    # model
    ap.add_argument("--model", required=True, type=str, help="Path to .joblib model")
    # outputs
    ap.add_argument("--out-json", required=True, type=str,
                    help="Where to save summary metrics JSON")
    ap.add_argument("--out-preds", type=str, default=None,
                    help="Optional CSV with per-file predictions")
    ap.add_argument("--dump-topk", type=int, default=0,
                    help="If >0, include top-K lists in JSON for highest fake/real probs")
    # optional decision thresholding for positive class (binary only)
    ap.add_argument("--pos-class", type=str, default="fake",
                    help="Positive class name for thresholding/AUROC (default: fake)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Optional probability threshold for pos-class (binary only)")
    args = ap.parse_args()

    # -------- data source --------
    if not args.csv and not args.data_root:
        raise SystemExit("Provide --csv or --data-root.")
    pairs = _load_pairs_from_csv(args.csv) if args.csv else _scan_tree(args.data_root)
    if not pairs:
        raise SystemExit("No files found.")

    # -------- load model --------
    bundle = joblib.load(args.model)
    pipe = bundle["model"]
    model_info: Dict[str, Any] = bundle.get("model_info", {})
    label_names: List[str] = (
        model_info.get("label_names") or model_info.get("classes") or ["fake", "real"]
    )
    vector_dim = int(model_info.get("vector_dim", 0))
    model_hash = str(model_info.get("cfg_hash", ""))

    # estimator classes_ exist and are 0..C-1 from training with LabelEncoder
    est = getattr(pipe, "named_steps", {}).get("clf", None)
    if est is not None and hasattr(est, "classes_"):
        # sanity: classes_ length should equal label_names length
        if len(getattr(est, "classes_")) != len(label_names):
            print("[WARN] estimator.classes_ size != label_names size; will rely on index order")

    # -------- iterate over files --------
    n = len(pairs)
    y_true_str: List[str] = []
    y_pred_str: List[str] = []
    y_pos_score: List[float] = []   # for ROC-AUC (score or prob of pos-class)
    per_file_rows: List[List[Any]] = []

    missing_sidecars = 0
    dropped_badvec = 0
    mismatched_dim = 0
    mismatched_hash = 0

    # for top-K reporting later
    pos_name = args.pos_class
    if pos_name not in label_names:
        print(f"[WARN] pos-class '{pos_name}' not in label_names {label_names}; using first label")
        pos_name = label_names[0]
    pos_idx = label_names.index(pos_name)

    for i, (apath, lab) in enumerate(pairs, 1):
        side = _sidecar(apath, args.suffix)
        if not side.exists():
            missing_sidecars += 1
            continue

        try:
            vec, meta = _load_sidecar(side)
            vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
        except Exception as e:
            print(f"[WARN] failed to load sidecar {side}: {e}")
            dropped_badvec += 1
            continue

        # basic checks: dimension + cfg hash
        if vector_dim and vec.size != vector_dim:
            mismatched_dim += 1
            continue
        sh = str(meta.get("cfg_hash", ""))
        if model_hash and sh and sh != model_hash:
            mismatched_hash += 1
            # still proceed; just log divergence

        X = vec.reshape(1, -1)

        # ---- predict ----
        probs: Optional[np.ndarray] = None
        pred_idx: Optional[int] = None

        # 1) try predict_proba (preferred)
        if hasattr(pipe, "predict_proba"):
            try:
                p = pipe.predict_proba(X)
                probs = np.asarray(p[0], dtype=np.float32)  # (C,)
            except Exception:
                probs = None

        # 2) fallback to decision_function -> softmax or sign
        if probs is None:
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X)
                scores = np.asarray(scores[0], dtype=np.float32)
                if scores.ndim == 0:
                    # scalar for binary; make two-class scores [s, -s]
                    scores = np.array([scores.item(), -scores.item()], dtype=np.float32)
                probs = _softmax(scores)
            else:
                # 3) last resort: discrete predict, make a one-hot
                pred_idx = int(pipe.predict(X)[0])
                probs = np.zeros((len(label_names),), dtype=np.float32)
                if 0 <= pred_idx < len(probs):
                    probs[pred_idx] = 1.0

        # pick predicted index
        if pred_idx is None:
            pred_idx = int(np.argmax(probs))

        pred_lab = label_names[pred_idx]
        y_true_str.append(lab)
        y_pred_str.append(pred_lab)

        # store positive-class score for ROC/AUROC
        pos_score = float(probs[min(pos_idx, len(probs)-1)])
        y_pos_score.append(pos_score)

        # optional thresholding (binary only)
        if args.threshold is not None and len(label_names) == 2:
            pred_lab_thr = pos_name if pos_score >= float(args.threshold) else (
                label_names[1 - pos_idx]
            )
            y_pred_str[-1] = pred_lab_thr  # override discrete prediction

        # record per-file row
        # Also store both fake and real probs if available
        row = [apath, lab, pred_lab]
        for name in label_names:
            j = label_names.index(name)
            row.append(float(probs[min(j, len(probs)-1)]))
        per_file_rows.append(row)

        if i % 2000 == 0:
            print(f"[INFO] processed {i}/{n} files...")

    # -------- aggregate metrics --------
    if not y_true_str:
        raise SystemExit("No usable examples (all missing/invalid sidecars).")

    acc = accuracy_score(y_true_str, y_pred_str)
    rep = classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=label_names).tolist()

    # ROC-AUC if binary and we have pos scores aligned with y_true
    roc_auc = None
    if len(label_names) == 2:
        y_true_bin = np.array([1 if t == pos_name else 0 for t in y_true_str], dtype=np.int32)
        try:
            roc_auc = float(roc_auc_score(y_true_bin, np.array(y_pos_score, dtype=np.float32)))
        except Exception:
            roc_auc = None

    # -------- write outputs --------
    # predictions CSV
    if args.out_preds:
        headers = ["path", "true_label", "pred_label"] + [f"prob_{n}" for n in label_names]
        Path(args.out_preds).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_preds, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(per_file_rows)
        print(f"[OK] wrote predictions CSV: {args.out_preds}")

    # top-K lists (by pos-class prob)
    topk = {}
    if args.dump_topk and len(per_file_rows) > 0:
        k = int(args.dump_topk)

        # índices das colunas de prob no per_file_rows
        # row = [path, true_label, pred_label, prob_label0, prob_label1, ...]
        pos_col = 3 + pos_idx

        # Top-K da classe positiva
        scored_pos = [(r[0], r[1], r[2], float(r[pos_col])) for r in per_file_rows]
        top_pos = sorted(scored_pos, key=lambda t: -t[3])[:k]

        # Se binário, também computa Top-K da classe "oposta"
        if len(label_names) == 2:
            neg_idx = 1 - pos_idx
            neg_col = 3 + neg_idx
            scored_neg = [(r[0], r[1], r[2], float(r[neg_col])) for r in per_file_rows]
            top_neg = sorted(scored_neg, key=lambda t: -t[3])[:k]
        else:
            top_neg = []

        topk = {
            f"top_{pos_name}": [
                {"path": p, "true": y, "pred": yhat, "prob": s} for (p, y, yhat, s) in top_pos
            ],
            f"top_{label_names[1 - pos_idx] if len(label_names) == 2 else 'other'}": [
                {"path": p, "true": y, "pred": yhat, "prob": s} for (p, y, yhat, s) in top_neg
            ],
        }


    summary = {
        "model": str(args.model),
        "label_names": label_names,
        "vector_dim": vector_dim,
        "cfg_hash": model_hash,
        "n_total_csv": n,
        "n_used": int(len(y_true_str)),
        "n_missing_sidecars": int(missing_sidecars),
        "n_mismatched_dim": int(mismatched_dim),
        "n_mismatched_hash": int(mismatched_hash),
        "n_bad_sidecar": int(dropped_badvec),
        "accuracy": float(acc),
        "report": rep,
        "confusion_matrix": {"labels": label_names, "matrix": cm},
        "roc_auc_pos_class": { "name": pos_name, "roc_auc": roc_auc },
        "threshold": args.threshold,
        "topk": topk,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved metrics JSON: {out_json}")

    # attempt to quiet cuML/CuPy teardown warnings
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import gc
        del pipe
        gc.collect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
