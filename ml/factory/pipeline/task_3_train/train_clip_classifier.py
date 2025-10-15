#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a clip-level classifier using precomputed sidecar feature files (*.feat.npz).

- Reads a CSV [path,label] or a folder tree (labels = folder names),
  looks for "<path>.feat.npz" (configurable via --suffix).
- Stacks vectors (dimension-checked via cfg_hash), scales and trains (SVM/LogReg).
- GPU-first (cuML); CPU sklearn fallback.

Example:
  python ml/factory/pipeline/task_3_train/train_clip_classifier.py \
    --csv ml/factory/pipeline/task_0_data/data_processed/train_clips.csv \
    --suffix .feat.npz \
    --out-model ml/factory/experiments/run0/clipclf.joblib \
    --out-report ml/factory/experiments/run0/clipclf_report.json \
    --clf svm
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import collections

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# ======== GPU first, CPU fallback ========
try:
    # RAPIDS/cuML (GPU)
    from cuml.preprocessing import StandardScaler  # type: ignore
    from cuml.svm import SVC  # type: ignore
    from cuml.linear_model import LogisticRegression  # type: ignore
    GPU_ML = True
except Exception:
    # scikit-learn (CPU)
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    GPU_ML = False
# =========================================

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _load_csv(csv_path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row.get("path")
            lab = row.get("label")
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
    # Produz "<audio>.<suffix>", por ex: "a.wav" -> "a.wav.feat.npz"
    p = Path(audio_path)
    return p.with_suffix(p.suffix + suffix)


def _load_vec(npz_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    z = np.load(npz_path, allow_pickle=False)
    vec = z["vector"].astype(np.float32)
    meta: Dict[str, Any] = {}
    if "meta" in z.files:
        try:
            raw = z["meta"]
            if raw.shape == () and raw.dtype.kind in {"U", "S", "O"}:
                meta = json.loads(raw.item())
        except Exception:
            meta = {}
    return vec, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a clip-level classifier from precomputed features")
    # data
    ap.add_argument("--csv", type=str, default=None, help="CSV with columns [path,label]")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Folder tree data_root/<label>/*.(wav|flac|mp3|m4a|ogg)")
    ap.add_argument("--suffix", type=str, default=".feat.npz",
                    help="Sidecar suffix produced by extract_features.py")
    # training
    ap.add_argument("--clf", type=str, default="logreg", choices=["logreg", "svm"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    # output
    ap.add_argument("--out-model", required=True, type=str)
    ap.add_argument("--out-report", type=str, default=None)
    args = ap.parse_args()

    # ----- data listing -----
    if not args.csv and not args.data_root:
        raise SystemExit("Provide --csv or --data-root.")
    pairs = _load_csv(args.csv) if args.csv else _scan_tree(args.data_root)
    if not pairs:
        raise SystemExit("No files.")

    # ----- load sidecars -----
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    cfg_hashes: List[str] = []
    missing = 0

    for i, (apath, lab) in enumerate(pairs, 1):
        sp = _sidecar(apath, args.suffix)
        if not sp.exists():
            missing += 1
            if i % 2000 == 0:
                print(f"[MISS] {missing} missing sidecars so far (e.g., {sp})")
            continue
        try:
            v, meta = _load_vec(sp)
            v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
            if v.ndim != 1 or v.size == 0:
                continue
            X_list.append(v)
            y_list.append(lab)
            cfg_hashes.append(str(meta.get("cfg_hash", "")))
        except Exception as e:
            print(f"[WARN] failed to load {sp}: {e}")

        if i % 2000 == 0:
            print(f"[INFO] loaded {i}/{len(pairs)} | usable={len(X_list)} | missing={missing}")

    if not X_list:
        raise SystemExit("No vectors loaded. Did you run extract_features.py?")

    # ----- keep only majority cfg_hash and consistent dimensionality -----
    mode_hash, _ = collections.Counter(cfg_hashes).most_common(1)[0]
    ref_dim = int(X_list[0].size)
    kept_X, kept_y = [], []
    dropped = 0
    for v, y, h in zip(X_list, y_list, cfg_hashes):
        if h == mode_hash and v.size == ref_dim:
            kept_X.append(v)
            kept_y.append(y)
        else:
            dropped += 1

    X = np.vstack([v.reshape(1, -1) for v in kept_X]).astype(np.float32, copy=False)
    labels = np.array(kept_y)
    print(f"[CLEAN] kept={len(kept_X)} | dropped={dropped} | missing_sidecars={missing}")

    # ----- encode labels to integers (required for cuML; works on sklearn too) -----
    le = LabelEncoder()
    y_all = le.fit_transform(labels)              # e.g., ['fake','real'] -> [0,1]
    class_names = le.classes_.tolist()            # e.g., ['fake','real']

    print(
        f"[INFO] X shape: {X.shape} | classes: {class_names} | "
        f"real/fake counts: {(labels=='real').sum()}/{(labels=='fake').sum()}"
    )

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_all, test_size=args.test_size, random_state=args.seed, stratify=y_all
    )

    # ----- classifier (balanced for imbalance) -----
    if args.clf == "svm":
        clf = SVC(
            C=2.0, kernel="rbf", probability=True, gamma="scale",
            class_weight="balanced", random_state=args.seed
        )
    else:
        clf = LogisticRegression(
            max_iter=2000, C=2.0, class_weight="balanced", random_state=args.seed
        )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])

    print(f"[BOOT] using {'GPU/cuML' if GPU_ML else 'CPU/sklearn'}")
    pipe.fit(Xtr, ytr)

    # ----- evaluation -----
    yhat_num = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat_num)
    yte_str = le.inverse_transform(yte)
    yhat_str = le.inverse_transform(yhat_num)
    print(f"[OK] Val accuracy: {acc:.3f}")
    rep = classification_report(
        yte_str, yhat_str, labels=class_names, target_names=class_names,
        output_dict=True, zero_division=0
    )

    # ----- save model -----
    model_info = {
        "label_names": class_names,            # ordem consistente com a saída
        "vector_dim": int(X.shape[1]),
        "gpu_ml": bool(GPU_ML),
        "cfg_hash": mode_hash,
    }
    obj = {
        "model": pipe,
        "model_info": model_info,
        "label_encoder_classes_": class_names,  # para inferência posterior
    }
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, args.out_model)
    print(f"[OK] Saved model: {args.out_model}")

    if args.out_report:
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "report": rep,
                    "n_samples": int(len(labels)),
                    "cfg_hash": mode_hash,
                    "classes": class_names,
                },
                f,
                indent=2,
            )
        print(f"[OK] Saved report: {args.out_report}")


if __name__ == "__main__":
    main()
