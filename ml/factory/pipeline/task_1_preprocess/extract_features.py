#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract clip-level features (MFCC / Ambient AES / ENF) and save a sidecar .feat.npz
next to each audio file.

- Input can be a CSV with columns [path,label] or a folder tree data_root/<label>/*.(wav|flac|mp3|m4a|ogg)
- Output: one NPZ per audio: "<audio_path>.feat.npz" in the same directory as the audio.
- Each NPZ contains:
    - vector: float32 [D]   (fixed-size, padded if any sub-vector fails)
    - meta:   JSON string with configs, version, dims and a cfg_hash for safety.

Why sidecar? If algo/config muda, você não reprocessa tudo — só os faltantes ou hash diferente.

Example:
  python ml/factory/pipeline/task_1_preprocess/extract_features.py \
    --csv ml/factory/pipeline/task_0_data/data_processed/train_clips.csv \
    --use-mfcc --use-ambient --sr 16000 --jobs 8
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import multiprocessing as mp

import numpy as np

# project preproc + features
from hoaxhertz.preproc import PreprocConfig, LoadAndStandardize
from hoaxhertz.features import PreprocInput
from hoaxhertz.features.mfcc import MFCCConfig, MFCCFeatureExtractor
from hoaxhertz.features.ambient import AmbientConfig, AmbientFeatureExtractor
from hoaxhertz.features.enf import ENFConfig, ENFFeatureExtractor

FEATURE_VERSION = "v1"   # bump se alterar o cálculo do vetor

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# ---------- data loaders ----------
def _load_csv(csv_path: str) -> List[Tuple[str, Optional[str]]]:
    rows: List[Tuple[str, Optional[str]]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row.get("path")
            lab = row.get("label") if "label" in (row or {}) else None
            if p:
                rows.append((p, lab))
    return rows

def _scan_tree(root: str) -> List[Tuple[str, Optional[str]]]:
    pairs: List[Tuple[str, Optional[str]]] = []
    for label_dir in sorted(Path(root).glob("*")):
        if not label_dir.is_dir():
            continue
        maybe_label = label_dir.name
        for audio in sorted(label_dir.rglob("*")):
            if audio.suffix.lower() in AUDIO_EXTS:
                pairs.append((str(audio), maybe_label))
    return pairs

# ---------- feature helpers ----------
def _cfg_hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]

def _expected_dims(feat_spec: Dict[str, bool],
                   mfcc_cfg: Dict[str, Any],
                   ambient_cfg: Dict[str, Any],
                   enf_cfg: Dict[str, Any]) -> Tuple[Dict[str, int], int]:
    dmfcc = 0
    if feat_spec.get("mfcc", False):
        base = 2 * int(mfcc_cfg.get("n_mfcc", 20))  # mean/std
        dmfcc = base
        if mfcc_cfg.get("vector_include_deltas", True) and mfcc_cfg.get("use_deltas", True):
            dmfcc += base
        if mfcc_cfg.get("vector_include_delta2", False) and mfcc_cfg.get("use_delta2", False):
            dmfcc += base
    damb = 0
    if feat_spec.get("ambient", False):
        damb = 2 * int(ambient_cfg.get("n_mels", 40)) + 2  # + SFM mean/std
    denf = 5 if feat_spec.get("enf", False) else 0
    ed = {"mfcc": dmfcc, "ambient": damb, "enf": denf}
    return ed, (dmfcc + damb + denf)

def _concat_fixed(parts: Dict[str, np.ndarray], expected: Dict[str, int]) -> np.ndarray:
    out: List[np.ndarray] = []
    for k in ["mfcc", "ambient", "enf"]:
        d = int(expected.get(k, 0))
        if d <= 0:
            continue
        v = parts.get(k)
        if v is None or v.size == 0:
            out.append(np.zeros(d, dtype=np.float32))
        else:
            v = v.astype(np.float32, copy=False).ravel()
            if v.size == d:
                out.append(v)
            else:
                w = np.zeros(d, dtype=np.float32)
                w[:min(d, v.size)] = v[:min(d, v.size)]
                out.append(w)
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)

def _mfcc_vector(y: np.ndarray, sr: int, cfg: Dict[str, Any]) -> np.ndarray:
    c = MFCCConfig(
        n_mfcc=int(cfg.get("n_mfcc", 20)),
        n_fft=cfg.get("n_fft", None),
        hop_length=cfg.get("hop_length", None),
        n_mels=int(cfg.get("n_mels", 40)),
        fmin=float(cfg.get("fmin", 50.0)),
        fmax=None if cfg.get("fmax") is None else float(cfg["fmax"]),
        dct_type=int(cfg.get("dct_type", 2)),
        lifter=int(cfg.get("lifter", 0)),
        htk=bool(cfg.get("htk", False)),
        norm=str(cfg.get("norm", "ortho")),
        use_deltas=bool(cfg.get("use_deltas", True)),
        use_delta2=bool(cfg.get("use_delta2", False)),
        cmvn=bool(cfg.get("cmvn", False)),
        use_robust=bool(cfg.get("use_robust", False)),
        return_framewise=False,
        vector_include_deltas=bool(cfg.get("vector_include_deltas", True)),
        vector_include_delta2=bool(cfg.get("vector_include_delta2", False)),
    )
    e = MFCCFeatureExtractor(); e.configure(c)
    return e.extract(PreprocInput(y=y, sr=sr)).vector.astype(np.float32, copy=False)

def _ambient_vector(y: np.ndarray, sr: int, cfg: Dict[str, Any]) -> np.ndarray:
    c = AmbientConfig(
        n_mels=int(cfg.get("n_mels", 40)),
        n_fft=int(cfg.get("n_fft", 1024)),
        hop_length=int(cfg.get("hop_length", 160)),
        fmin=int(cfg.get("fmin", 50)),
        fmax=None if cfg.get("fmax") is None else int(cfg["fmax"]),
        use_robust=bool(cfg.get("use_robust", True)),
        use_vad=bool(cfg.get("use_vad", True)),
        vad_percentile=int(cfg.get("vad_percentile", 30)),
        return_framewise=False,
        fast_signature=bool(cfg.get("fast_signature", False)),
    )
    e = AmbientFeatureExtractor(); e.configure(c)
    return e.extract(PreprocInput(y=y, sr=sr)).vector.astype(np.float32, copy=False)

def _enf_vector(y: np.ndarray, sr: int, cfg: Dict[str, Any]) -> np.ndarray:
    c = ENFConfig(
        nominal=float(cfg.get("nominal", 60.0)),
        bw=float(cfg.get("bw", 1.5)),
        strategy=str(cfg.get("strategy", "stft")),
        order=int(cfg.get("order", 6)),
        stft_nperseg=int(cfg.get("stft_nperseg", 4096)),
        stft_noverlap=int(cfg.get("stft_noverlap", 2048)),
        smooth_sec=float(cfg.get("smooth_sec", 0.25)),
        smooth_method=str(cfg.get("smooth_method", "savgol")),
        return_framewise=False,
    )
    e = ENFFeatureExtractor(); e.configure(c)
    return e.extract(PreprocInput(y=y, sr=sr)).vector.astype(np.float32, copy=False)

def _sidecar_path(audio_path: str, suffix: str) -> Path:
    # same directory, file.wav + ".feat.npz" (default)
    return Path(audio_path).with_suffix(Path(audio_path).suffix + suffix)

def _process_one(args_tuple) -> Tuple[str, bool, str]:
    (path, sr, feat_spec, mfcc_cfg, ambient_cfg, enf_cfg, suffix, overwrite) = args_tuple
    try:
        outp = _sidecar_path(path, suffix)
        if outp.exists() and not overwrite:
            return (path, True, "cached")

        y, sr0 = LoadAndStandardize(path, PreprocConfig(target_sr=int(sr)))

        parts: Dict[str, np.ndarray] = {}
        if feat_spec.get("mfcc", False):
            parts["mfcc"] = _mfcc_vector(y, sr0, mfcc_cfg)
        if feat_spec.get("ambient", False):
            parts["ambient"] = _ambient_vector(y, sr0, ambient_cfg)
        if feat_spec.get("enf", False):
            parts["enf"] = _enf_vector(y, sr0, enf_cfg)

        expected_dict, expected_total = _expected_dims(feat_spec, mfcc_cfg, ambient_cfg, enf_cfg)
        vec = _concat_fixed(parts, expected_dict)
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)

        meta = {
            "audio": path,
            "sr": int(sr0),
            "version": FEATURE_VERSION,
            "feature_spec": feat_spec,
            "mfcc_cfg": mfcc_cfg,
            "ambient_cfg": ambient_cfg,
            "enf_cfg": enf_cfg,
            "expected_dims": expected_dict,
            "vector_dim": int(vec.size),
        }
        meta["cfg_hash"] = _cfg_hash(
            {"v": FEATURE_VERSION, "sr": int(sr0), "feat_spec": feat_spec,
             "mfcc": mfcc_cfg, "ambient": ambient_cfg, "enf": enf_cfg}
        )

        np.savez_compressed(outp, vector=vec.astype(np.float32), meta=json.dumps(meta))
        return (path, True, "ok")
    except Exception as e:
        return (path, False, f"err: {e}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract features and save sidecar .feat.npz files")
    # data
    ap.add_argument("--csv", type=str, default=None, help="CSV with columns [path,label]")
    ap.add_argument("--data-root", type=str, default=None, help="Folder tree data_root/<label>/*.(wav|flac|mp3|m4a|ogg)")
    # preproc
    ap.add_argument("--sr", type=int, default=16000)
    # which features
    ap.add_argument("--use-mfcc", action="store_true")
    ap.add_argument("--use-ambient", action="store_true")
    ap.add_argument("--use-enf", action="store_true")
    # params (minimal)
    ap.add_argument("--mfcc-n", type=int, default=20)
    ap.add_argument("--mfcc-mels", type=int, default=40)
    ap.add_argument("--ambient-mels", type=int, default=40)
    ap.add_argument("--enf-nominal", type=float, default=60.0)
    # output control
    ap.add_argument("--suffix", type=str, default=".feat.npz", help='Sidecar suffix (default: ".feat.npz")')
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    # choose data source
    if not args.csv and not args.data_root:
        raise SystemExit("Provide --csv or --data-root.")
    pairs = _load_csv(args.csv) if args.csv else _scan_tree(args.data_root)
    if not pairs:
        raise SystemExit("No audio files found.")

    feat_spec = {"mfcc": args.use_mfcc, "ambient": args.use_ambient, "enf": args.use_enf}
    if not any(feat_spec.values()):
        feat_spec["mfcc"] = True  # safe default

    mfcc_cfg = {
        "n_mfcc": args.mfcc_n, "n_mels": args.mfcc_mels,
        "cmvn": False, "use_deltas": True, "use_delta2": False,
        "use_robust": False, "vector_include_deltas": True, "vector_include_delta2": False,
    }
    ambient_cfg = {
        "n_mels": args.ambient_mels, "use_robust": True, "use_vad": True, "vad_percentile": 30,
        "n_fft": 1024, "hop_length": 160, "fmin": 50, "fmax": None, "fast_signature": False
    }
    enf_cfg = {
        "nominal": args.enf_nominal, "strategy": "stft", "bw": 1.5,
        "stft_nperseg": 4096, "stft_noverlap": 2048, "smooth_sec": 0.25, "smooth_method": "savgol"
    }

    print(f"[BOOT] extracting with feat_spec={feat_spec} | sr={args.sr}")
    # parallel map
    tasks = [
        (path, args.sr, feat_spec, mfcc_cfg, ambient_cfg, enf_cfg, args.suffix, args.overwrite)
        for (path, _lab) in pairs
    ]
    ok = bad = 0
    with mp.get_context("spawn").Pool(processes=max(1, args.jobs)) as pool:
        for i, (path, success, msg) in enumerate(pool.imap_unordered(_process_one, tasks), 1):
            if success:
                ok += 1
            else:
                bad += 1
            if i % 200 == 0:
                print(f"[INFO] {i}/{len(tasks)} processed | ok={ok} bad={bad} | last={msg}")

    print(f"[DONE] total={len(tasks)} | ok={ok} | bad={bad}")

if __name__ == "__main__":
    main()
