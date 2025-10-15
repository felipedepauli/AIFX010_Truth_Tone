# # -*- coding: utf-8 -*-
# """
# End-to-end audio forensics pipeline (single-file)
# - Pre-processing
# - Feature extraction (MFCC / ENF / Ambient AES) with a shared cache
# - Detectors (ENF / Ambient / Locutor)
# - Event fusion
# - Pydantic models for typed, JSON-serializable results

# This file is self-contained and resilient:
# - If an optional dependency or local module is missing, it degrades gracefully.
# - Detectors can consume features from the shared cache to avoid recomputation.
# """

# from __future__ import annotations
# from typing import Any, Dict, List, Optional, Tuple
# from dataclasses import dataclass, field
# import json
# import logging
# import time

# # ---- Logging ----------------------------------------------------------------
# logger = logging.getLogger("hoaxhertz.pipeline")
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter(
#         "[%(asctime)s] %(levelname)s - %(name)s: %(message)s", "%H:%M:%S"
#     )
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
# logger.setLevel(logging.INFO)

# # ---- Pydantic models (v2) ---------------------------------------------------
# try:
#     from pydantic import BaseModel, Field
# except Exception:  # pragma: no cover
#     # Minimal fallback if pydantic is not installed
#     class BaseModel:  # type: ignore
#         def model_dump(self, **_: Any) -> Dict[str, Any]:
#             return self.__dict__

#         def model_dump_json(self, **_: Any) -> str:
#             return json.dumps(self.__dict__)

#     def Field(default: Any = None, **kwargs: Any) -> Any:  # type: ignore
#         return default


# class EvidenceModel(BaseModel):
#     """Detector-specific evidence fields (free-form)."""
#     data: Dict[str, Any] = Field(default_factory=dict)


# class EventModel(BaseModel):
#     """Standardized event schema for all detectors."""
#     t_start: float = 0.0
#     t_end: float = 0.0
#     label: str = "event"
#     score: float = 0.0
#     sources: List[str] = Field(default_factory=list)
#     evidence: Dict[str, Any] = Field(default_factory=dict)


# class PipelineSummary(BaseModel):
#     """Structured pipeline output including metadata and events."""
#     ok: bool = True
#     sr: int = 0
#     duration_sec: float = 0.0
#     steps_ms: Dict[str, float] = Field(default_factory=dict)
#     events: List[EventModel] = Field(default_factory=list)
#     features_present: Dict[str, bool] = Field(default_factory=dict)
#     message: str = ""


# # ---- Config -----------------------------------------------------------------
# try:
#     from hoaxhertz.config import settings  # pydantic-settings (recommended)
#     DEFAULT_SR = settings.default_sr
# except Exception:
#     DEFAULT_SR = 16000  # fallback


# # ---- Pre-processing ----------------------------------------------------------
# try:
#     from hoaxhertz.preproc import load_audio, normalize_peak
# except Exception:
#     # Very small fallback to keep the pipeline runnable in minimal environments.
#     import numpy as np
#     import soundfile as sf  # optional dependency; if missing, raise at runtime

#     def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple["np.ndarray", int]:
#         """Fallback loader using soundfile (mono conversion and optional resample)."""
#         x, sr = sf.read(path, always_2d=False)
#         if x.ndim > 1:
#             x = x.mean(axis=1)
#         x = x.astype("float32")
#         if target_sr and target_sr != sr:
#             # Simple resample via librosa/scipy if available, else raise
#             try:
#                 import librosa  # type: ignore
#                 x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
#                 sr = target_sr
#             except Exception as e:
#                 raise RuntimeError(f"Resampling required but librosa not available: {e}") from e
#         return x, sr

#     def normalize_peak(x: "np.ndarray", peak: float = 0.99) -> "np.ndarray":
#         m = float(max(1e-12, abs(x).max()))
#         return (x / m) * peak


# # ---- Features ---------------------------------------------------------------
# # We try to import the project's canonical implementations; if not present, we keep going.

# # MFCC
# try:
#     from hoaxhertz.features.mfcc import compute_mfcc  # expected to return dict with "mfcc", "times", ...
# except Exception:
#     compute_mfcc = None  # type: ignore

# # ENF
# try:
#     from hoaxhertz.features.enf import estimate_enf  # expected to return a trajectory object
# except Exception:
#     estimate_enf = None  # type: ignore

# # Ambient AES
# try:
#     from hoaxhertz.features.ambient import compute_aes
# except Exception:
#     compute_aes = None  # type: ignore


# @dataclass
# class FeatureCache:
#     """Shared feature cache to avoid recomputation across detectors."""
#     mfcc: Optional[Dict[str, Any]] = None
#     enf: Optional[Any] = None
#     aes: Optional[Dict[str, Any]] = None
#     flags: Dict[str, bool] = field(default_factory=dict)


# # ---- Detectors --------------------------------------------------------------
# # We will import detectors if available. Each detector should implement:
# #   predict(x: np.ndarray, sr: int, feats: FeatureCache) -> List[EventModel]

# DetectorFn = Any  # callable type alias for simplicity

# def _safe_import_detectors() -> Dict[str, DetectorFn]:
#     dets: Dict[str, DetectorFn] = {}

#     # ENF detector
#     try:
#         from hoaxhertz.detectors.enfdet import EnfDetector  # type: ignore

#         def _enf_predict(x, sr, feats: FeatureCache) -> List[EventModel]:
#             d = EnfDetector()
#             events = d.predict(x, sr, feats)  # should return list[dict] or list[EventModel]
#             return [_coerce_event(e, source="enf") for e in events]

#         dets["enf"] = _enf_predict
#     except Exception:
#         pass

#     # Ambient detector
#     try:
#         from hoaxhertz.detectors.ambientdet import AmbientDetector  # type: ignore

#         def _ambient_predict(x, sr, feats: FeatureCache) -> List[EventModel]:
#             d = AmbientDetector()
#             events = d.predict(x, sr, feats)
#             return [_coerce_event(e, source="ambient") for e in events]

#         dets["ambient"] = _ambient_predict
#     except Exception:
#         pass

#     # Locutor detector
#     try:
#         from hoaxhertz.detectors.locutor import LocutorDetector  # type: ignore

#         def _locutor_predict(x, sr, feats: FeatureCache) -> List[EventModel]:
#             d = LocutorDetector()
#             events = d.predict(x, sr, feats)
#             return [_coerce_event(e, source="locutor") for e in events]

#         dets["locutor"] = _locutor_predict
#     except Exception:
#         pass

#     return dets


# def _coerce_event(e: Any, source: str) -> EventModel:
#     """Normalize arbitrary detector outputs into EventModel."""
#     if isinstance(e, EventModel):
#         if source and source not in e.sources:
#             e.sources = list(set((e.sources or []) + [source]))
#         return e

#     # dictionary-like
#     t_start = float(e.get("t_start", 0.0))
#     t_end = float(e.get("t_end", 0.0))
#     label = str(e.get("label", "event"))
#     score = float(e.get("score", 0.0))
#     sources = list(set((e.get("sources") or []) + [source]))
#     evidence = dict(e.get("evidence", {}))
#     return EventModel(
#         t_start=t_start, t_end=t_end, label=label, score=score, sources=sources, evidence=evidence
#     )


# # ---- Fusion -----------------------------------------------------------------
# def _fuse_events(events_by_model: Dict[str, List[EventModel]]) -> List[EventModel]:
#     """Fuse events using the project's util if available; otherwise a simple fallback."""
#     # Try project's fusion first
#     try:
#         from hoaxhertz.utils.fusion import fuse_events  # type: ignore

#         packed: Dict[str, List[Dict[str, Any]]] = {}
#         for k, lst in events_by_model.items():
#             packed[k] = [e.model_dump() for e in lst]
#         fused = fuse_events(packed)  # expected to return list[dict]
#         return [_coerce_event(e, source="fused") for e in fused]
#     except Exception:
#         # Minimal fallback: flatten and return highest-score unique intervals
#         all_events = [e for _k, lst in events_by_model.items() for e in lst]
#         all_events.sort(key=lambda x: (-x.score, x.t_start))
#         # Optional: naive non-maximum suppression on time overlaps
#         fused: List[EventModel] = []
#         for ev in all_events:
#             if not fused:
#                 fused.append(ev)
#                 continue
#             last = fused[-1]
#             overlap = max(0.0, min(last.t_end, ev.t_end) - max(last.t_start, ev.t_start))
#             union = (last.t_end - last.t_start) + (ev.t_end - ev.t_start) - overlap
#             iou = overlap / union if union > 0 else 0.0
#             if iou < 0.5:
#                 fused.append(ev)
#             else:
#                 # keep the one with higher score; merge sources
#                 if ev.score > last.score:
#                     ev.sources = list(set(ev.sources + last.sources))
#                     fused[-1] = ev
#                 else:
#                     last.sources = list(set(last.sources + ev.sources))
#         return fused


# # ---- Pipeline ---------------------------------------------------------------
# @dataclass
# class PipelineConfig:
#     """Lightweight config for the pipeline."""
#     target_sr: int = DEFAULT_SR
#     normalize_peak: float = 0.98
#     compute_mfcc: bool = True
#     compute_enf: bool = True
#     compute_aes: bool = True
#     run_detectors: Optional[List[str]] = None  # e.g., ["enf","ambient","locutor"]
#     do_fusion: bool = True
#     print_json: bool = False

# import sys
# def run_pipeline(audio_path: str, cfg: Optional[PipelineConfig] = None) -> PipelineSummary:
#     """
#     Run the full pipeline over an audio file and return a structured summary.
#     - Loads and normalizes audio
#     - Extracts features into a shared cache
#     - Runs the available detectors
#     - Fuses events (if enabled)
#     """
#     import numpy as np

#     # --- Pipeline step 0: Initialization ---
#     t0 = time.time()                        # Record start time for overall pipeline
#     steps_ms: Dict[str, int] = {}         # Dictionary to store timing for each step (ms)
#     features_present: Dict[str, bool] = {}  # Track which features were successfully extracted

#     if cfg is None:
#         cfg = PipelineConfig()              # Use default config if none provided

#     # --- Pipeline step 1: Load and normalize audio ---
#     t = time.time()                                         # Start timing for load+normalize
#     x, sr = load_audio(audio_path, target_sr=cfg.target_sr) # Load audio and resample if needed
#     x = normalize_peak(x, peak=cfg.normalize_peak)          # Normalize audio peak amplitude
#     steps_ms["load+normalize"] = int((time.time() - t) * 1000.0) # Record elapsed time in ms

#     duration_sec = float(len(x) / max(sr, 1))  # Compute duration in seconds
    

#     # 2) Feature extraction (with graceful degrade)
#     feats = FeatureCache()

#     # MFCC
#     if cfg.compute_mfcc:
#         t = time.time()
#         try:
#             if compute_mfcc is not None:
#                 feats.mfcc = compute_mfcc(x, sr)  # expected dict with "mfcc" and "times"
#                 feats.flags["mfcc"] = True
#             else:
#                 feats.flags["mfcc"] = False
#         except Exception as e:
#             logger.warning(f"MFCC failed: {e}")
#             feats.flags["mfcc"] = False
#         steps_ms["features.mfcc"] = (time.time() - t) * 1000.0
#         features_present["mfcc"] = feats.flags.get("mfcc", False)

#     # ENF
#     if cfg.compute_enf:
#         t = time.time()
#         try:
#             if estimate_enf is not None:
#                 feats.enf = estimate_enf(x, sr)  # expected trajectory
#                 feats.flags["enf"] = True
#             else:
#                 feats.flags["enf"] = False
#         except Exception as e:
#             logger.warning(f"ENF estimation failed: {e}")
#             feats.flags["enf"] = False
#         steps_ms["features.enf"] = (time.time() - t) * 1000.0
#         features_present["enf"] = feats.flags.get("enf", False)

#     # AES
#     if cfg.compute_aes:
#         t = time.time()
#         try:
#             if compute_aes is not None:
#                 feats.aes = compute_aes(x, sr)
#                 feats.flags["aes"] = True
#             else:
#                 feats.flags["aes"] = False
#         except Exception as e:
#             logger.warning(f"Ambient AES failed: {e}")
#             feats.flags["aes"] = False
#         steps_ms["features.aes"] = (time.time() - t) * 1000.0
#         features_present["aes"] = feats.flags.get("aes", False)

#     for key, value in steps_ms.items():
#         print(f"{key}: {value}ms")
#     sys.exit(0)

#     # 3) Detectors
#     events_by_model: Dict[str, List[EventModel]] = {}
#     t = time.time()
#     available = _safe_import_detectors()

#     # Choose detectors to run
#     det_order = cfg.run_detectors if cfg.run_detectors else list(available.keys())

#     for name in det_order:
#         pred_fn = available.get(name)
#         if not pred_fn:
#             logger.info(f"Detector '{name}' not available. Skipping.")
#             continue
#         try:
#             evs = pred_fn(x, sr, feats)
#             events_by_model[name] = evs
#         except Exception as e:
#             logger.warning(f"Detector '{name}' failed: {e}")
#             events_by_model[name] = []
#     steps_ms["detectors"] = (time.time() - t) * 1000.0

#     # 4) Fusion
#     t = time.time()
#     fused: List[EventModel] = []
#     if cfg.do_fusion and events_by_model:
#         fused = _fuse_events(events_by_model)
#     else:
#         # If fusion disabled, flatten raw events
#         fused = [ev for _k, lst in events_by_model.items() for ev in lst]
#     steps_ms["fusion"] = (time.time() - t) * 1000.0

#     # 5) Pack summary
#     summary = PipelineSummary(
#         ok=True,
#         sr=sr,
#         duration_sec=duration_sec,
#         steps_ms=steps_ms,
#         features_present=features_present,
#         events=fused,
#         message="ok",
#     )

#     # Optional: print JSON to stdout (useful for CLI)
#     if cfg.print_json:
#         try:
#             print(summary.model_dump_json(indent=2))  # pydantic v2
#         except Exception:
#             # fallback
#             print(json.dumps(summary.__dict__, default=lambda o: o.__dict__, indent=2))

#     return summary


# # ---- Optional: minimal CLI entry point --------------------------------------
# def _main():
#     """Minimal CLI wrapper: python -m hoaxhertz.pipeline <audio_path> [--no-json]."""
#     import argparse
#     parser = argparse.ArgumentParser(description="Run the hoaxhertz pipeline on an audio file.")
#     parser.add_argument("audio_path", type=str, help="Path to the audio file")
#     parser.add_argument("--no-json", action="store_true", help="Do not print JSON to stdout")
#     parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate")
#     parser.add_argument("--no-fusion", action="store_true", help="Disable fusion step")
#     args = parser.parse_args()

#     cfg = PipelineConfig(
#         target_sr=args.sr,
#         do_fusion=(not args.no_fusion),
#         print_json=(not args.no_json),
#     )
#     run_pipeline(args.audio_path, cfg=cfg)


# if __name__ == "__main__":
#     _main()



# apps/zoo/detect_splicing.py
"""
Minimal splicing detector:
- Preprocess audio (load + peak normalize)
- Extract framewise features (Ambient log-mel and/or MFCC)
- Aggregate with sliding windows (mean+std per coefficient)
- Change-point detection with ruptures (PELT, 'rbf' cost)
- Save boundaries (seconds) as JSON (and optional plot)

Usage (examples)
---------------
python apps/zoo/detect_splicing.py \
  --audio ml/research/sample_data/example.wav \
  --use-ambient --use-mfcc \
  --win-sec 1.5 --hop-sec 0.5 \
  --ruptures-pen 10 --ruptures-min-size 5 \
  --out-json artifacts/plots/splicing_boundaries.json \
  --plot artifacts/plots/splicing_boundaries.png

Dependencies
------------
pip install ruptures librosa numpy soundfile matplotlib
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ruptures as rpt

# --- project preproc ---
from hoaxhertz.preproc import load_audio, normalize_peak

# --- project features ---
from hoaxhertz.features.ambient import AmbientConfig, AmbientFeatureExtractor
from hoaxhertz.features.mfcc import MFCCConfig, MFCCFeatureExtractor
from hoaxhertz.features import PreprocInput


# ------------------------------
# Feature extraction (framewise)
# ------------------------------
def extract_framewise_ambient(y: np.ndarray, sr: int, n_mels: int = 40) -> Dict[str, np.ndarray]:
    """
    Return framewise Ambient payload using AES extractor:
      - "feat": (T, D) matrix to be window-aggregated (we use log-mel bands)
      - "times": (T,) timestamps in seconds
    """
    cfg = AmbientConfig(
        n_mels=n_mels,
        return_framewise=True,
        use_robust=True,   # robust aggregation is for clip-level only; ok to keep True
        use_vad=True,      # focus on low-energy frames (ambient)
        vad_percentile=30,
    )
    extr = AmbientFeatureExtractor()
    extr.configure(cfg)
    out = extr.extract(PreprocInput(y=y, sr=sr))
    fw = out.framewise or {}
    logmel = fw.get("logmel", None)   # (T, M)
    times = fw.get("times", None)     # (T,)
    if logmel is None or times is None:
        return {"feat": np.zeros((0, 0), dtype=float), "times": np.zeros((0,), dtype=float)}
    return {"feat": logmel.astype(float), "times": times.astype(float)}


def extract_framewise_mfcc(
    y: np.ndarray, sr: int, n_mfcc: int = 20, n_mels: int = 40, cmvn: bool = False
) -> Dict[str, np.ndarray]:
    """
    Return framewise MFCC payload:
      - "feat": (T, D) MFCC matrix
      - "times": (T,) timestamps in seconds
    """
    cfg = MFCCConfig(
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        cmvn=cmvn,               # optional per-clip CMVN
        return_framewise=True,
        use_deltas=False,        # keep simplest path for splicing
        use_delta2=False,
    )
    extr = MFCCFeatureExtractor()
    extr.configure(cfg)
    out = extr.extract(PreprocInput(y=y, sr=sr))
    fw = out.framewise or {}
    mfcc = fw.get("mfcc", None)   # (T, D)
    times = fw.get("times", None) # (T,)
    if mfcc is None or times is None:
        return {"feat": np.zeros((0, 0), dtype=float), "times": np.zeros((0,), dtype=float)}
    return {"feat": mfcc.astype(float), "times": times.astype(float)}


# ------------------------------
# Sliding windows + aggregation
# ------------------------------
def sliding_windows(T: int, win: int, hop: int) -> List[Tuple[int, int]]:
    """Generate inclusive-exclusive [start, end) windows over T frames."""
    if T <= 0 or win <= 0 or hop <= 0:
        return []
    out = []
    i = 0
    while i + win <= T:
        out.append((i, i + win))
        i += hop
    if not out and T > 0:
        out.append((0, T))
    return out


def window_and_aggregate(
    feat: np.ndarray, times: np.ndarray, win_sec: float, hop_sec: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert framewise (T, D) into a windowed sequence (K, 2*D) using mean+std per window.
    The window timestamp is the end time of the window.
    """
    if feat.size == 0 or times.size == 0:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    # Estimate frame rate from time stamps
    dt = np.diff(times).mean() if times.size > 1 else 0.01
    fps = 1.0 / max(dt, 1e-6)

    win = max(1, int(round(win_sec * fps)))
    hop = max(1, int(round(hop_sec * fps)))

    idx = sliding_windows(T=feat.shape[0], win=win, hop=hop)
    if not idx:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    seq = []
    marks = []
    for a, b in idx:
        Fw = feat[a:b, :]  # (W, D)
        mu = Fw.mean(axis=0)
        sd = Fw.std(axis=0)
        seq.append(np.concatenate([mu, sd], axis=0))
        marks.append(times[b - 1])

    return np.vstack(seq).astype(float), np.asarray(marks, dtype=float)


# ------------------------------
# Change-point detection
# ------------------------------
def detect_changes(Wseq: np.ndarray, pen: float, min_size: int) -> List[int]:
    """
    Run PELT with 'rbf' cost on standardized window sequence (K, d).
    Returns a list of end-indexes (ruptures convention).
    """
    if Wseq.size == 0:
        return []
    # Standardize per dimension
    Z = (Wseq - Wseq.mean(axis=0, keepdims=True)) / (Wseq.std(axis=0, keepdims=True) + 1e-9)
    algo = rpt.Pelt(model="rbf", min_size=max(2, int(min_size)), jump=1).fit(Z)
    bkps = algo.predict(pen=float(pen))
    return bkps


def merge_close(times: List[float], min_gap_sec: float) -> List[float]:
    """Keep first boundary and drop following ones within min_gap_sec."""
    if not times:
        return []
    times = sorted(times)
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] >= min_gap_sec:
            out.append(t)
    return out


# ------------------------------
# CLI
# ------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal splicing detection (Ambient/MFCC + ruptures).")
    ap.add_argument("--audio", required=True, type=str, help="Audio file path")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate (resample)")
    ap.add_argument("--peak", type=float, default=0.98, help="Peak normalize target (0..1)")

    ap.add_argument("--use-ambient", action="store_true", help="Use Ambient (log-mel) framewise")
    ap.add_argument("--use-mfcc", action="store_true", help="Use MFCC framewise")
    ap.add_argument("--mfcc-n", type=int, default=20, help="Number of MFCC coefficients")
    ap.add_argument("--mfcc-mels", type=int, default=40, help="Number of mel bands for MFCC")
    ap.add_argument("--mfcc-cmvn", action="store_true", help="Apply per-clip CMVN on MFCCs")

    ap.add_argument("--win-sec", type=float, default=1.5, help="Sliding window size (seconds)")
    ap.add_argument("--hop-sec", type=float, default=0.5, help="Sliding window hop (seconds)")
    ap.add_argument("--ruptures-pen", type=float, default=10.0, help="PELT penalty (larger=fewer breaks)")
    ap.add_argument("--ruptures-min-size", type=int, default=5, help="Minimum segment length (windows)")
    ap.add_argument("--min-gap-sec", type=float, default=0.7, help="Merge boundaries closer than this")

    ap.add_argument("--out-json", required=True, type=str, help="Output JSON path with boundaries")
    ap.add_argument("--plot", type=str, default=None, help="Optional PNG path to save a simple plot")

    args = ap.parse_args()

    if not (args.use_ambient or args.use_mfcc):
        raise SystemExit("Enable at least one feature: --use-ambient and/or --use-mfcc")

    # 1) Preprocess
    y, sr = load_audio(args.audio, target_sr=args.sr)
    y = normalize_peak(y, peak=float(args.peak))

    # 2) Framewise features
    feats: List[np.ndarray] = []
    times_ref = None

    if args.use_ambient:
        fw = extract_framewise_ambient(y, sr)
        W, t = window_and_aggregate(fw["feat"], fw["times"], args.win_sec, args.hop_sec)
        if W.size:
            feats.append(W)
            times_ref = t if times_ref is None else times_ref

    if args.use_mfcc:
        fw = extract_framewise_mfcc(y, sr, n_mfcc=args.mfcc_n, n_mels=args.mfcc_mels, cmvn=args.mfcc_cmvn)
        W, t = window_and_aggregate(fw["feat"], fw["times"], args.win_sec, args.hop_sec)
        if W.size:
            feats.append(W)
            if times_ref is None:
                times_ref = t

    if not feats or times_ref is None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"boundaries_sec": [], "notes": "no features/windows"}, f, indent=2)
        print("[WARN] No features/windows extracted. Saved empty result.")
        return

    # 3) Concatenate modalities and detect changes
    Wseq = np.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]
    bkps = detect_changes(Wseq, pen=args.ruptures_pen, min_size=args.ruptures_min_size)

    # Map to seconds (use window end timestamps; ignore the last endpoint if it's the very end)
    boundaries = []
    for b in bkps:
        if 0 < b <= len(times_ref) - 1:
            boundaries.append(float(times_ref[b - 1]))

    boundaries = merge_close(boundaries, min_gap_sec=float(args.min_gap_sec))

    # 4) Save JSON
    payload = {
        "audio": str(args.audio),
        "sr": int(sr),
        "win_sec": float(args.win_sec),
        "hop_sec": float(args.hop_sec),
        "boundaries_sec": boundaries,
        "n_windows": int(len(times_ref)),
        "ruptures": {"pen": float(args.ruptures_pen), "min_size": int(args.ruptures_min_size)},
        "features": {"ambient": bool(args.use_ambient), "mfcc": bool(args.use_mfcc)},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Saved boundaries to: {args.out_json}")

    # 5) Optional quick plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            xs = np.arange(len(times_ref))
            plt.figure()
            plt.plot(xs, Wseq.mean(axis=1))  # crude 1D view of the window sequence
            for t in boundaries:
                k = int(np.argmin(np.abs(times_ref - t)))
                plt.axvline(k, linestyle="--")
            plt.title("Window sequence (mean over dims) + detected boundaries")
            plt.xlabel("window index")
            plt.ylabel("feature mean")
            Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.plot, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved plot to: {args.plot}")
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")


if __name__ == "__main__":
    main()
