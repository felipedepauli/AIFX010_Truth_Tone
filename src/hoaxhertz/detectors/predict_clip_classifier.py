# ml/predict_clip_classifier.py
"""
Clip-level classifier inference (loads a trained scikit-learn model and predicts a label for an audio file).

What this does
--------------
1) Preprocess audio (load + resample + peak-normalize) using your project's preproc.
2) Extract clip-level features (MFCC / Ambient AES / ENF) using the same extractors as training.
3) Concatenate feature vectors in a fixed order and feed them to a saved model (.joblib).
4) Output a JSON with predicted label, class probabilities, and some diagnostics.

Assumptions
-----------
- The trained model was saved by your training script (e.g., ml/train_clip_classifier.py) as a
  scikit-learn Pipeline or estimator in .joblib format. Ideally the file also contains a
  small metadata dict with:
    model_info = {
      "feature_spec": {"mfcc": True, "ambient": True, "enf": True},
      "label_names": ["classA", "classB", ...],
      "mfcc_cfg": {...}, "ambient_cfg": {...}, "enf_cfg": {...}
    }
  If missing, you can set flags here (--use-mfcc, --use-ambient, --use-enf) and default params.

- The model Pipeline should include its own scaler/standardizer if needed (e.g., StandardScaler).

Usage
-----
python ml/predict_clip_classifier.py \
  --audio ml/research/sample_data/sample.wav \
  --model ml/models_registry/clipclf.joblib \
  --out-json artifacts/plots/clip_pred.json \
  --use-mfcc --use-ambient --use-enf

Optional:
  --dump-xnpy artifacts/plots/clip_vector.npy  # saves the concatenated feature vector

Notes
-----
- Keep MFCC/AES/ENF parameters consistent with training (or load from model metadata).
- If vector dimensionality doesn't match what the model expects, you'll get a shape error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# --- IO / model ---
try:
    import joblib
except Exception as e:  # pragma: no cover
    raise RuntimeError("joblib is required to load the classifier (.joblib).") from e

# --- your preproc ---
from hoaxhertz.preproc.base import PreprocConfig, LoadAndStandardize

# --- feature extractors ---
from hoaxhertz.features import PreprocInput
from hoaxhertz.features.mfcc import MFCCConfig, MFCCFeatureExtractor
from hoaxhertz.features.ambient import AmbientConfig, AmbientFeatureExtractor
from hoaxhertz.features.enf import ENFConfig, ENFFeatureExtractor


# ---------------------------------------------------------------------------
# Feature helpers (clip-level vectors)
# ---------------------------------------------------------------------------
def _mfcc_vector(y: np.ndarray, sr: int, cfg_dict: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a clip-level MFCC vector using MFCCFeatureExtractor.
    Returns (vector, meta).
    """
    cfg = MFCCConfig(
        n_mfcc=int(cfg_dict.get("n_mfcc", 20)),
        n_fft=cfg_dict.get("n_fft", None),
        hop_length=cfg_dict.get("hop_length", None),
        n_mels=int(cfg_dict.get("n_mels", 40)),
        fmin=float(cfg_dict.get("fmin", 50.0)),
        fmax=None if cfg_dict.get("fmax", None) is None else float(cfg_dict["fmax"]),
        dct_type=int(cfg_dict.get("dct_type", 2)),
        lifter=int(cfg_dict.get("lifter", 0)),
        htk=bool(cfg_dict.get("htk", False)),
        norm=str(cfg_dict.get("norm", "ortho")),
        use_deltas=bool(cfg_dict.get("use_deltas", True)),
        use_delta2=bool(cfg_dict.get("use_delta2", False)),
        cmvn=bool(cfg_dict.get("cmvn", False)),
        use_robust=bool(cfg_dict.get("use_robust", False)),
        return_framewise=False,
        vector_include_deltas=bool(cfg_dict.get("vector_include_deltas", True)),
        vector_include_delta2=bool(cfg_dict.get("vector_include_delta2", False)),
    )
    extr = MFCCFeatureExtractor()
    extr.configure(cfg)
    out = extr.extract(PreprocInput(y=y, sr=sr))
    vec = out.vector.astype(float)
    meta = {"dim": int(vec.size), "mfcc_meta": out.meta}
    return vec, meta


def _ambient_vector(y: np.ndarray, sr: int, cfg_dict: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a clip-level Ambient AES vector using AmbientFeatureExtractor.
    Returns (vector, meta).
    """
    cfg = AmbientConfig(
        n_mels=int(cfg_dict.get("n_mels", 40)),
        n_fft=int(cfg_dict.get("n_fft", 1024)),
        hop_length=int(cfg_dict.get("hop_length", 160)),
        fmin=int(cfg_dict.get("fmin", 50)),
        fmax=None if cfg_dict.get("fmax", None) is None else int(cfg_dict["fmax"]),
        use_robust=bool(cfg_dict.get("use_robust", True)),
        use_vad=bool(cfg_dict.get("use_vad", True)),
        vad_percentile=int(cfg_dict.get("vad_percentile", 30)),
        return_framewise=False,
        fast_signature=bool(cfg_dict.get("fast_signature", False)),
    )
    extr = AmbientFeatureExtractor()
    extr.configure(cfg)
    out = extr.extract(PreprocInput(y=y, sr=sr))
    vec = out.vector.astype(float)
    meta = {"dim": int(vec.size), "ambient_meta": out.meta}
    return vec, meta


def _enf_vector(y: np.ndarray, sr: int, cfg_dict: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a clip-level ENF vector using ENFFeatureExtractor.
    Returns (vector, meta).
    """
    cfg = ENFConfig(
        nominal=float(cfg_dict.get("nominal", 60.0)),
        bw=float(cfg_dict.get("bw", 1.5)),
        strategy=str(cfg_dict.get("strategy", "stft")),  # "stft" or "zc"
        order=int(cfg_dict.get("order", 6)),
        stft_nperseg=int(cfg_dict.get("stft_nperseg", 4096)),
        stft_noverlap=int(cfg_dict.get("stft_noverlap", 2048)),
        smooth_sec=float(cfg_dict.get("smooth_sec", 0.25)),
        smooth_method=str(cfg_dict.get("smooth_method", "savgol")),
        return_framewise=False,
    )
    extr = ENFFeatureExtractor()
    extr.configure(cfg)
    out = extr.extract(PreprocInput(y=y, sr=sr))
    vec = out.vector.astype(float)  # [mean_dev, std_dev, min, max, slope_hz_per_min]
    meta = {"dim": int(vec.size), "enf_meta": out.meta}
    return vec, meta


def _concat_features(parts: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Concatenate feature vectors in a deterministic order: MFCC | AMBIENT | ENF.
    Returns (X, dims) where dims has the slice sizes for debugging.
    """
    order = ["mfcc", "ambient", "enf"]
    xs = []
    dims: Dict[str, int] = {}
    for k in order:
        v = parts.get(k)
        if v is not None and v.size:
            xs.append(v)
            dims[k] = int(v.size)
        else:
            dims[k] = 0
    if not xs:
        return np.zeros((0,), dtype=float), dims
    X = np.concatenate(xs, axis=0).astype(float)
    return X, dims


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _load_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a joblib model. If it contains metadata under 'model_info', return it.
    """
    obj = joblib.load(model_path)
    meta = {}
    # common patterns: dict with 'model' and 'model_info', or a Pipeline with .classes_
    if isinstance(obj, dict):
        model = obj.get("model", obj)
        meta = obj.get("model_info", obj.get("meta", {})) or {}
    else:
        model = obj
        # try to find attached metadata
        meta = getattr(model, "model_info", {}) or getattr(model, "meta", {}) or {}
    return model, meta


def _predict_proba(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (proba, classes_) robustly.
    If the model has predict_proba, use it. Otherwise fall back to decision_function→softmax.
    """
    X2 = X.reshape(1, -1)
    # try predict_proba
    if hasattr(model, "predict_proba"):
        P = model.predict_proba(X2)[0]
        classes = getattr(model, "classes_", None)
        if classes is None:
            # Pipeline? try final_estimator_
            classes = getattr(getattr(model, "named_steps", {}).get("clf", model), "classes_", None)
        if classes is None:
            # still None → build indices
            classes = np.arange(len(P))
        return np.asarray(P, dtype=float), np.asarray(classes)
    # decision_function → softmax
    if hasattr(model, "decision_function"):
        S = np.atleast_1d(model.decision_function(X2)).ravel()
        # avoid overflow
        S = S - np.max(S)
        e = np.exp(S)
        P = e / np.sum(e)
        classes = getattr(model, "classes_", np.arange(len(P)))
        return np.asarray(P, dtype=float), np.asarray(classes)
    # last resort: predict label only
    yhat = model.predict(X2)[0]
    # make a 1-hot-ish vector
    classes = getattr(model, "classes_", np.asarray([yhat]))
    P = np.zeros((len(classes),), dtype=float)
    # if yhat is in classes, set prob=1 for that class
    try:
        idx = list(classes).index(yhat)  # type: ignore
        P[idx] = 1.0
    except Exception:
        pass
    return P, np.asarray(classes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a clip label with a trained classifier (.joblib).")
    ap.add_argument("--audio", required=True, type=str, help="Audio file path")
    ap.add_argument("--model", required=True, type=str, help="Path to .joblib model")
    ap.add_argument("--out-json", required=True, type=str, help="Output JSON with prediction")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate (preproc resample)")
    # feature toggles (used if model metadata not present)
    ap.add_argument("--use-mfcc", action="store_true", help="Include MFCC vector")
    ap.add_argument("--use-ambient", action="store_true", help="Include Ambient AES vector")
    ap.add_argument("--use-enf", action="store_true", help="Include ENF vector")
    # optional overrides for feature configs (keep minimal; extend if needed)
    ap.add_argument("--mfcc-n", type=int, default=20)
    ap.add_argument("--mfcc-mels", type=int, default=40)
    ap.add_argument("--ambient-mels", type=int, default=40)
    ap.add_argument("--enf-nominal", type=float, default=60.0)
    ap.add_argument("--dump-xnpy", type=str, default=None, help="Optional path to save the concatenated vector (.npy)")
    args = ap.parse_args()

    # 1) Load model (+metadata if available)
    model, model_meta = _load_model(args.model)

    # Resolve feature spec: prefer model metadata; else use CLI toggles
    feat_spec = model_meta.get("feature_spec", {})
    if not feat_spec:
        feat_spec = {
            "mfcc": bool(args.use_mfcc),
            "ambient": bool(args.use_ambient),
            "enf": bool(args.use_enf),
        }
        # if none selected, default to MFCC only (safe fallback)
        if not any(feat_spec.values()):
            feat_spec["mfcc"] = True

    # 2) Preprocess audio
    pp_cfg = PreprocConfig(target_sr=int(args.sr))
    y, sr = LoadAndStandardize(args.audio, pp_cfg)

    # 3) Build feature vectors as per spec (with params from metadata if present)
    parts: Dict[str, np.ndarray] = {}
    fmeta: Dict[str, Any] = {}

    if feat_spec.get("mfcc", False):
        mfcc_cfg = model_meta.get("mfcc_cfg", {
            "n_mfcc": args.mfcc_n, "n_mels": args.mfcc_mels, "cmvn": False,
            "use_deltas": True, "use_delta2": False, "use_robust": False,
            "vector_include_deltas": True, "vector_include_delta2": False,
        })
        v, m = _mfcc_vector(y, sr, mfcc_cfg)
        parts["mfcc"] = v; fmeta.update(m)

    if feat_spec.get("ambient", False):
        amb_cfg = model_meta.get("ambient_cfg", {
            "n_mels": args.ambient_mels, "use_robust": True, "use_vad": True, "vad_percentile": 30,
            "n_fft": 1024, "hop_length": 160, "fmin": 50, "fmax": None, "fast_signature": False
        })
        v, m = _ambient_vector(y, sr, amb_cfg)
        parts["ambient"] = v; fmeta.update(m)

    if feat_spec.get("enf", False):
        enf_cfg = model_meta.get("enf_cfg", {
            "nominal": args.enf_nominal, "strategy": "stft", "bw": 1.5,
            "stft_nperseg": 4096, "stft_noverlap": 2048, "smooth_sec": 0.25, "smooth_method": "savgol"
        })
        v, m = _enf_vector(y, sr, enf_cfg)
        parts["enf"] = v; fmeta.update(m)

    X, dims = _concat_features(parts)
    if X.size == 0:
        raise SystemExit("No features produced. Enable at least one feature (MFCC/Ambient/ENF).")

    # Optional: dump vector
    if args.dump_xnpy:
        Path(args.dump_xnpy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_xnpy, X)
        print(f"[OK] Saved feature vector: {args.dump_xnpy} (shape={X.shape})")

    # 4) Predict
    proba, classes = _predict_proba(model, X)
    # map indices → names
    label_names = model_meta.get("label_names", None)
    if label_names is None and hasattr(model, "classes_"):
        # scikit-learn stores classes_ on the estimator (strings or ints)
        label_names = list(getattr(model, "classes_"))  # type: ignore
    if label_names is None:
        # fallback: cast classes to string
        label_names = [str(c) for c in classes]

    # choose top-1
    if proba.size:
        top_idx = int(np.argmax(proba))
        top_label = str(label_names[top_idx]) if top_idx < len(label_names) else str(top_idx)
        top_score = float(np.max(proba))
    else:
        # if we only had predict() fallback
        yhat = model.predict(X.reshape(1, -1))[0]
        top_label = str(yhat); top_score = 1.0

    # pack per-class dict
    class_probs = {}
    for i, name in enumerate(label_names):
        p = float(proba[i]) if i < proba.size else (1.0 if name == top_label else 0.0)
        class_probs[str(name)] = p

    # 5) Save output JSON
    out = {
        "audio": str(args.audio),
        "sr": int(sr),
        "model": str(args.model),
        "features_used": {"mfcc": bool("mfcc" in parts), "ambient": bool("ambient" in parts), "enf": bool("enf" in parts)},
        "feature_dims": dims,
        "vector_dim": int(X.size),
        "pred_label": top_label,
        "pred_score": top_score,
        "class_probs": class_probs,
        "model_info": {
            "label_names": label_names,
            "feature_spec": feat_spec,
        },
        "feature_meta": fmeta,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved prediction to: {args.out_json}")
    print(f"[OK] Top-1: {top_label} (score={top_score:.3f})")


if __name__ == "__main__":
    main()
