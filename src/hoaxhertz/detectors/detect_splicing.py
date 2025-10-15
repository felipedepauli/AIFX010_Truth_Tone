# src/hoaxhertz/detectors/detect_splicing.py
"""
Splicing detection via sliding-window features + change-point detection (ruptures).

What’s new
----------
- --n-bkps K : force exactly K change-points via Binary Segmentation (Binseg).
- --robust-agg : robust window aggregation (median + IQR).
- --plot : save a quick PNG with the window sequence and detected boundaries.

Pipeline
--------
1) Preprocess (LoadAndStandardize from your preproc).
2) Framewise features (Ambient log-mel and/or MFCC).
3) Sliding windows -> per-window stats (mean+std OR median+IQR).
4) Change-point detection:
     - If --n-bkps is given, use Binseg(model="rbf").predict(n_bkps=K).
     - Else use PELT(model="rbf").predict(pen=...).
5) Save boundaries (seconds) to JSON, and an optional plot.

Dependencies
------------
- numpy, ruptures, matplotlib (optional for --plot)
- hoaxhertz.preproc.PreprocConfig, LoadAndStandardize
- hoaxhertz.features (Ambient/MFCC)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import ruptures as rpt

# --- your preproc ---
from hoaxhertz.preproc.base import PreprocConfig, LoadAndStandardize

# --- project feature extractors ---
from hoaxhertz.features.ambient import AmbientConfig, AmbientFeatureExtractor
from hoaxhertz.features.mfcc import MFCCConfig, MFCCFeatureExtractor
from hoaxhertz.features import PreprocInput


# ---------------------------------------------------------------------------
# Sliding windows helpers
# ---------------------------------------------------------------------------
def _sliding_windows(T: int, win: int, hop: int) -> List[Tuple[int, int]]:
    """Generate inclusive-exclusive [start, end) windows over T frames."""
    if T <= 0 or win <= 0 or hop <= 0:
        return []
    out, i = [], 0
    while i + win <= T:
        out.append((i, i + win))
        i += hop
    if not out and T > 0:
        out.append((0, T))
    return out


def _agg_window(F: np.ndarray, robust: bool) -> np.ndarray:
    """
    Aggregate a (W, D) matrix over time into (2*D,).

    robust=False -> concat(mean, std)
    robust=True  -> concat(median, IQR)
    """
    if F.ndim != 2 or F.size == 0:
        return np.zeros((0,), dtype=float)
    if robust:
        med = np.median(F, axis=0)
        p25 = np.percentile(F, 25, axis=0)
        p75 = np.percentile(F, 75, axis=0)
        iqr = p75 - p25
        return np.concatenate([med, iqr], axis=0).astype(float)
    else:
        mu = F.mean(axis=0)
        sd = F.std(axis=0)
        return np.concatenate([mu, sd], axis=0).astype(float)


# ---------------------------------------------------------------------------
# Framewise feature extraction
# ---------------------------------------------------------------------------
def _extract_framewise_ambient(y: np.ndarray, sr: int, n_mels: int = 40) -> Dict[str, np.ndarray]:
    """
    Return:
      - "feat": (T, D) log-mel frames (to be window-aggregated)
      - "times": (T,) frame timestamps (s)
    """
    cfg = AmbientConfig(
        n_mels=n_mels,
        return_framewise=True,
        use_robust=True,     # robust affects clip-level; OK to keep True
        use_vad=True,
        vad_percentile=30,
    )
    extr = AmbientFeatureExtractor()
    extr.configure(cfg)
    out = extr.extract(PreprocInput(y=y, sr=sr))
    fw = out.framewise or {}
    logmel = fw.get("logmel", None)  # (T, M)
    times = fw.get("times", None)    # (T,)
    if logmel is None or times is None:
        return {"feat": np.zeros((0, 0), dtype=float), "times": np.zeros((0,), dtype=float)}
    return {"feat": logmel.astype(float), "times": times.astype(float)}


def _extract_framewise_mfcc(
    y: np.ndarray, sr: int, n_mfcc: int = 20, n_mels: int = 40, cmvn: bool = False
) -> Dict[str, np.ndarray]:
    """
    Return:
      - "feat": (T, D) MFCC frames
      - "times": (T,) frame timestamps (s)
    """
    cfg = MFCCConfig(
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        cmvn=cmvn,
        return_framewise=True,
        use_deltas=False,
        use_delta2=False,
    )
    extr = MFCCFeatureExtractor()
    extr.configure(cfg)
    out = extr.extract(PreprocInput(y=y, sr=sr))
    fw = out.framewise or {}
    mfcc = fw.get("mfcc", None)    # (T, D)
    times = fw.get("times", None)  # (T,)
    if mfcc is None or times is None:
        return {"feat": np.zeros((0, 0), dtype=float), "times": np.zeros((0,), dtype=float)}
    return {"feat": mfcc.astype(float), "times": times.astype(float)}


# ---------------------------------------------------------------------------
# Windowing + change-point detection
# ---------------------------------------------------------------------------
def _window_sequence(
    feat: np.ndarray, times: np.ndarray, win_sec: float, hop_sec: float, robust: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert framewise (T, D) into windowed sequence (K, 2*D) with timestamps (window end).
    """
    if feat.size == 0 or times.size == 0:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    # Estimate frame rate from times
    dt = np.diff(times).mean() if times.size > 1 else 0.01
    fps = 1.0 / max(dt, 1e-6)

    win = max(1, int(round(win_sec * fps)))
    hop = max(1, int(round(hop_sec * fps)))

    idx = _sliding_windows(feat.shape[0], win, hop)
    if not idx:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    seq, tmarks = [], []
    for a, b in idx:
        Fw = feat[a:b, :]              # (W, D)
        seq.append(_agg_window(Fw, robust=robust))  # (2*D,)
        tmarks.append(times[b - 1])    # window end as timestamp

    return np.vstack(seq).astype(float), np.asarray(tmarks, dtype=float)


def _detect_changes(
    Wseq: np.ndarray,
    pen: float,
    min_size: int,
    n_bkps: Optional[int] = None,
) -> List[int]:
    """
    Change-point detection on standardized window sequence (K, d).

    - If n_bkps is provided (>0), use Binary Segmentation (Binseg) and predict(n_bkps=K).
      (Compatível com versões do ruptures em que PELT não aceita n_bkps.)
    - Caso contrário, use PELT com penalidade (predict(pen=...)).
    """
    if Wseq.size == 0:
        return []
    Z = (Wseq - Wseq.mean(axis=0, keepdims=True)) / (Wseq.std(axis=0, keepdims=True) + 1e-9)

    if n_bkps is not None and int(n_bkps) > 0:
        algo = rpt.Binseg(model="rbf", min_size=max(2, int(min_size)), jump=1).fit(Z)
        return algo.predict(n_bkps=int(n_bkps))
    else:
        algo = rpt.Pelt(model="rbf", min_size=max(2, int(min_size)), jump=1).fit(Z)
        return algo.predict(pen=float(pen))


def _merge_close(times: List[float], min_gap_sec: float) -> List[float]:
    """Keep the first boundary and drop following ones closer than min_gap_sec."""
    if not times:
        return []
    times = sorted(times)
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] >= min_gap_sec:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Splicing detection (Ambient/MFCC + ruptures).")
    ap.add_argument("--audio", required=True, type=str, help="Audio file path")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate (resample in preproc)")

    ap.add_argument("--use-ambient", action="store_true", help="Use Ambient framewise features")
    ap.add_argument("--use-mfcc", action="store_true", help="Use MFCC framewise features")
    ap.add_argument("--mfcc-n", type=int, default=20, help="Number of MFCC coefficients")
    ap.add_argument("--mfcc-mels", type=int, default=40, help="Number of mel bands for MFCC")
    ap.add_argument("--mfcc-cmvn", action="store_true", help="Apply per-clip CMVN on MFCCs")

    ap.add_argument("--win-sec", type=float, default=1.5, help="Sliding window size (seconds)")
    ap.add_argument("--hop-sec", type=float, default=0.5, help="Sliding window hop (seconds)")
    ap.add_argument("--robust-agg", action="store_true", help="Use robust window aggregation (median+IQR)")

    ap.add_argument("--ruptures-pen", type=float, default=10.0, help="PELT penalty (ignored if --n-bkps is set)")
    ap.add_argument("--ruptures-min-size", type=int, default=5, help="Minimum segment length (windows)")
    ap.add_argument("--n-bkps", type=int, default=None, help="Force exactly K change-points (uses Binseg)")
    ap.add_argument("--min-gap-sec", type=float, default=0.7, help="Merge breaks closer than this (seconds)")

    ap.add_argument("--out-json", required=True, type=str, help="Output JSON path with boundaries")
    ap.add_argument("--plot", type=str, default=None, help="Optional PNG path to save a simple plot")

    args = ap.parse_args()

    if not (args.use_ambient or args.use_mfcc):
        raise SystemExit("Enable at least one feature: --use-ambient and/or --use-mfcc")

    # 1) Preprocess (load + resample + peak-normalize)
    pp_cfg = PreprocConfig(target_sr=int(args.sr))
    y, sr = LoadAndStandardize(args.audio, pp_cfg)

    # 2) Framewise -> windowed
    feats: List[np.ndarray] = []
    t_ref: np.ndarray | None = None

    if args.use_ambient:
        fw = _extract_framewise_ambient(y, sr)
        if fw["feat"].size > 0:
            Fw, tw = _window_sequence(fw["feat"], fw["times"], args.win_sec, args.hop_sec, robust=args.robust_agg)
            feats.append(Fw)
            t_ref = tw if t_ref is None else t_ref

    if args.use_mfcc:
        fw = _extract_framewise_mfcc(
            y, sr, n_mfcc=int(args.mfcc_n), n_mels=int(args.mfcc_mels), cmvn=bool(args.mfcc_cmvn)
        )
        if fw["feat"].size > 0:
            Fw, tw = _window_sequence(fw["feat"], fw["times"], args.win_sec, args.hop_sec, robust=args.robust_agg)
            feats.append(Fw)
            if t_ref is None:
                t_ref = tw

    if not feats or t_ref is None or feats[0].size == 0:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"boundaries_sec": [], "notes": "no features/windows"}, f, indent=2)
        print("[WARN] No features/windows extracted. Saved empty result.")
        return

    # 3) Concatenate and detect changes
    Wseq = np.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]
    bkps = _detect_changes(
        Wseq,
        pen=float(args.ruptures_pen),
        min_size=int(args.ruptures_min_size),
        n_bkps=args.n_bkps,
    )

    # Map to seconds
    boundaries = []
    for b in bkps:
        if 0 < b <= len(t_ref) - 1:
            boundaries.append(float(t_ref[b - 1]))
    boundaries = _merge_close(boundaries, min_gap_sec=float(args.min_gap_sec))

    # 4) Save JSON
    result = {
        "audio": str(args.audio),
        "sr": int(sr),
        "win_sec": float(args.win_sec),
        "hop_sec": float(args.hop_sec),
        "robust_agg": bool(args.robust_agg),
        "boundaries_sec": boundaries,
        "n_windows": int(len(t_ref)),
        "ruptures": {
            "pen": float(args.ruptures_pen),
            "min_size": int(args.ruptures_min_size),
            "n_bkps": None if args.n_bkps is None else int(args.n_bkps),
            "algo": "binseg" if args.n_bkps else "pelt",
        },
        "features": {"ambient": bool(args.use_ambient), "mfcc": bool(args.use_mfcc)},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Saved boundaries to: {args.out_json}")

    # 5) Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            xs = np.arange(len(t_ref))
            seq_mean = Wseq.mean(axis=1) if Wseq.ndim == 2 else Wseq
            plt.figure()
            plt.plot(xs, seq_mean, label="window seq (mean over dims)")
            for t in boundaries:
                k = int(np.argmin(np.abs(t_ref - t)))
                plt.axvline(k, linestyle="--", alpha=0.8, label=None)
            title = "Window sequence + detected boundaries"
            if args.n_bkps: title += f" (Binseg, K={args.n_bkps})"
            plt.title(title)
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
