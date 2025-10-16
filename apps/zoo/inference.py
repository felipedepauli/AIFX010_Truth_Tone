#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-audio inference using trained .joblib model and on-the-fly feature extraction.

- Loads model (cuML or scikit-learn pipeline)
- Runs hoaxhertz preprocessing (LoadAndStandardize)
- Extracts MFCC / Ambient / ENF features automatically to match model vector_dim
- Predicts "real" / "fake" with confidence

Usage:
  python inference.py \
      --model ml/factory/experiments/run0/clipclf.joblib \
      --audio Insanity.wav
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import joblib
import sys

# --- safe fallback for GPU-trained models ---
try:
    import cuml  # noqa
except ImportError:
    import sklearn.svm, sklearn.linear_model, sklearn.preprocessing
    sys.modules['cuml.svm'] = sklearn.svm
    sys.modules['cuml.linear_model'] = sklearn.linear_model
    sys.modules['cuml.preprocessing'] = sklearn.preprocessing

# --- hoaxhertz feature stack ---
from hoaxhertz.preproc import PreprocConfig, LoadAndStandardize
from hoaxhertz.features import PreprocInput
from hoaxhertz.features.mfcc import MFCCConfig, MFCCFeatureExtractor
from hoaxhertz.features.ambient import AmbientConfig, AmbientFeatureExtractor
from hoaxhertz.features.enf import ENFConfig, ENFFeatureExtractor


def _extract_vector(y: np.ndarray, sr: int, spec: Dict[str, bool]) -> np.ndarray:
    """Extracts feature vector from waveform based on spec."""
    parts = {}
    if spec.get("mfcc", False):
        e = MFCCFeatureExtractor()
        e.configure(MFCCConfig(n_mfcc=20, n_mels=40, use_deltas=True, use_delta2=False,
                               vector_include_deltas=True, vector_include_delta2=False))
        parts["mfcc"] = e.extract(PreprocInput(y=y, sr=sr)).vector.astype(np.float32)
    if spec.get("ambient", False):
        e = AmbientFeatureExtractor()
        e.configure(AmbientConfig(n_mels=40, n_fft=1024, hop_length=160, fmin=50,
                                  use_vad=True, use_robust=True, vad_percentile=30))
        parts["ambient"] = e.extract(PreprocInput(y=y, sr=sr)).vector.astype(np.float32)
    if spec.get("enf", False):
        e = ENFFeatureExtractor()
        e.configure(ENFConfig(nominal=60.0, bw=1.5, strategy="stft",
                              stft_nperseg=4096, stft_noverlap=2048,
                              smooth_sec=0.25, smooth_method="savgol"))
        parts["enf"] = e.extract(PreprocInput(y=y, sr=sr)).vector.astype(np.float32)
    # concat fixed
    vec = np.concatenate([v for v in parts.values() if v is not None], axis=0)
    return np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)


def main():
    ap = argparse.ArgumentParser(description="Inference from raw audio using trained model")
    ap.add_argument("--model", required=True, help="Path to .joblib trained model")
    ap.add_argument("--audio", required=True, help="Audio file (.wav, .m4a, etc.)")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    # --- Load model ---
    bundle = joblib.load(args.model)
    pipe = bundle["model"]
    info: Dict[str, Any] = bundle.get("model_info", {})
    labels = info.get("label_names", ["fake", "real"])
    vector_dim = int(info.get("vector_dim", 0))

    print(f"[BOOT] Loaded model ({labels}) | expected_dim={vector_dim}")

    # --- Preprocess & extract ---
    y, sr = LoadAndStandardize(args.audio, PreprocConfig(target_sr=args.sr))

    # try each combination until one matches model's vector_dim
    combos = [
        {"mfcc": True},
        {"mfcc": True, "enf": True},
        {"mfcc": True, "ambient": True},
        {"mfcc": True, "ambient": True, "enf": True}
    ]
    v = None
    for spec in combos:
        try:
            vec = _extract_vector(y, sr, spec)
            if abs(vec.size - vector_dim) <= 2:  # tolerance
                v = vec
                print(f"[AUTO] Using feature spec: {spec} ({vec.size} dims)")
                break
        except Exception:
            continue

    if v is None:
        raise SystemExit(f"Could not produce compatible vector (expected {vector_dim})")

    X = v.reshape(1, -1)
    y_prob = pipe.predict_proba(X)[0]
    idx = int(np.argmax(y_prob))
    pred = labels[idx]
    conf = float(y_prob[idx])

    print(f"[PRED] {Path(args.audio).name:40s} -> {pred.upper():5s} ({conf:.3f})")

if __name__ == "__main__":
    main()
