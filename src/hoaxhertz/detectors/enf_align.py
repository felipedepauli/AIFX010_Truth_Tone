# src/hoaxhertz/detectors/enf_align.py
"""
Align an audio's estimated ENF to an external ENF log via normalized cross-correlation.

What it does
------------
1) Preprocess audio (load + resample + peak-normalize) with your project's preproc.
2) Estimate ENF trajectory (times, fi) using ENFFeatureExtractor (zc or stft).
3) Load an external ENF log from CSV (time, fi).
4) Resample both ENF sequences onto a common time grid (e.g., every 1.0 s).
5) Optionally detrend and z-score the sequences.
6) Compute normalized cross-correlation over +/- max_lag_sec, find best lag.
7) Output alignment diagnostics to JSON and (optionally) a PNG plot.

Typical usage
-------------
python src/hoaxhertz/detectors/enf_align.py \
  --audio ml/research/sample_data/sample.wav \
  --log-csv ml/research/sample_data/enf_log.csv \
  --csv-time-col time --csv-fi-col fi --csv-delim , \
  --nominal 60 --strategy stft --bw 1.5 --smooth-sec 0.25 \
  --resample-sec 1.0 --max-lag-sec 120 \
  --detrend linear \
  --out-json artifacts/plots/enf_align.json \
  --plot artifacts/plots/enf_align.png

CSV format
----------
- Must contain a time column (seconds) and a frequency column (Hz).
- If your times are not in seconds-from-start, convert them offline, or use
  --time-scale and --time-offset to adapt (time' = time*scale + offset).

Notes
-----
- This is a baseline (magnitude-domain). For forensic-grade alignment:
  phase-unwrapping, harmonic fusion, Kalman smoothing, quality scoring, etc.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# --- your preproc ---
from hoaxhertz.preproc.base import PreprocConfig, LoadAndStandardize

# --- ENF extractor ---
from hoaxhertz.features.enf import ENFConfig, ENFFeatureExtractor
from hoaxhertz.features import PreprocInput


# ---------------------------------------------------------------------------
# Helpers: IO and basic transforms
# ---------------------------------------------------------------------------
def _load_enf_csv(
    path: str,
    time_col: str = "time",
    fi_col: str = "fi",
    delim: str = ",",
    has_header: bool = True,
    time_scale: float = 1.0,
    time_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an external ENF CSV: return (times_sec, fi_hz) as float arrays.
    - time' = time * time_scale + time_offset
    """
    ts: List[float] = []
    fs: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        if has_header:
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                try:
                    t = float(row[time_col]) * time_scale + time_offset
                    fi = float(row[fi_col])
                    ts.append(t)
                    fs.append(fi)
                except Exception:
                    continue
        else:
            # No header: assume [time, fi, ...] in the first two columns
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    t = float(row[0]) * time_scale + time_offset
                    fi = float(row[1])
                    ts.append(t)
                    fs.append(fi)
                except Exception:
                    continue
    t_arr = np.asarray(ts, dtype=float)
    f_arr = np.asarray(fs, dtype=float)
    # ensure strictly increasing times (monotonic), dropping ties
    keep = np.argsort(t_arr, kind="mergesort")
    t_arr = t_arr[keep]
    f_arr = f_arr[keep]
    uniq = np.where(np.diff(np.r_[[-np.inf], t_arr]) > 0)[0]  # drop duplicates
    return t_arr[uniq], f_arr[uniq]


def _detrend(x: np.ndarray, method: str = "none") -> np.ndarray:
    """
    Detrend a 1D series:
      - none   : return as-is
      - demean : subtract mean
      - linear : remove best-fit line (least squares)
    """
    if x.size == 0:
        return x
    if method == "demean":
        return x - np.mean(x)
    if method == "linear":
        t = np.arange(x.size, dtype=float)
        A = np.vstack([t, np.ones_like(t)]).T
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return x - (m * t + b)
    return x


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x)) if x.size else 1.0
    return (x - mu) / max(sd, 1e-9)


def _interp_to_grid(times: np.ndarray, fi: np.ndarray, t0: float, t1: float, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate an ENF sequence to a uniform time grid within [t0, t1].
    """
    if t1 <= t0 or step <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    grid = np.arange(t0, t1 + 1e-9, step, dtype=float)
    if times.size < 2:
        return grid, np.full_like(grid, np.nan, dtype=float)
    y = np.interp(grid, times, fi, left=np.nan, right=np.nan)
    return grid, y


# ---------------------------------------------------------------------------
# Cross-correlation (normalized), restricted lags
# ---------------------------------------------------------------------------
def _xcorr_norm_at_lags(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation r[lag] for lags in [-max_lag, ..., +max_lag].
    x and y must be the same length and already z-scored/detrended.
    r[k] = corr( x[t], y[t+k] ) using the overlapping part, normalized by N_overlap.
    Returns:
      lags  : integer lags (samples)
      rvals : correlation values in [-1, 1] (best effort normalization)
    """
    assert x.size == y.size and x.ndim == y.ndim == 1
    N = x.size
    max_lag = min(max_lag, N - 2) if N >= 3 else 0
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    rvals = np.empty_like(lags, dtype=float)

    for i, k in enumerate(lags):
        if k >= 0:
            xw = x[: N - k]
            yw = y[k:]
        else:
            xw = x[-k:]
            yw = y[: N + k]
        n = xw.size
        if n < 2:
            r = np.nan
        else:
            # xw,yw should already be z-scored; still subtract mean in window for safety
            xw = (xw - xw.mean()) / (xw.std() + 1e-9)
            yw = (yw - yw.mean()) / (yw.std() + 1e-9)
            r = float(np.dot(xw, yw) / max(n, 1))
        rvals[i] = r
    return lags, rvals


# ---------------------------------------------------------------------------
# Main alignment routine
# ---------------------------------------------------------------------------
def _estimate_audio_enf(
    y: np.ndarray,
    sr: int,
    nominal: float,
    strategy: str = "stft",
    bw: float = 1.5,
    smooth_sec: float = 0.25,
    smooth_method: str = "savgol",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Run ENFFeatureExtractor and return (times, fi, stats).
    """
    cfg = ENFConfig(
        nominal=float(nominal),
        bw=float(bw),
        strategy=str(strategy),
        smooth_sec=float(smooth_sec),
        smooth_method=str(smooth_method),
        return_framewise=True,
    )
    extr = ENFFeatureExtractor()
    extr.configure(cfg)
    out = extr.extract(PreprocInput(y=y, sr=sr))
    fw = out.framewise or {}
    times = fw.get("times", np.zeros((0,), dtype=float))
    fi = fw.get("fi", np.zeros((0,), dtype=float))

    # simple stats relative to nominal
    dev = fi - nominal
    stats = {
        "available": float(fi.size),
        "mean_dev": float(np.mean(dev)) if fi.size else float("nan"),
        "std_dev": float(np.std(dev)) if fi.size else float("nan"),
        "min": float(np.min(fi)) if fi.size else float("nan"),
        "max": float(np.max(fi)) if fi.size else float("nan"),
    }
    return times.astype(float), fi.astype(float), stats


def align_enf(
    audio_path: str,
    log_csv: str,
    out_json: str,
    plot_path: Optional[str] = None,
    # preproc
    target_sr: int = 16000,
    # ENF extraction
    nominal: float = 60.0,
    strategy: str = "stft",
    bw: float = 1.5,
    smooth_sec: float = 0.25,
    smooth_method: str = "savgol",
    # CSV parsing
    csv_time_col: str = "time",
    csv_fi_col: str = "fi",
    csv_delim: str = ",",
    csv_has_header: bool = True,
    time_scale: float = 1.0,
    time_offset: float = 0.0,
    # alignment
    resample_sec: float = 1.0,
    detrend_mode: str = "linear",  # none|demean|linear
    max_lag_sec: float = 120.0,
) -> Dict[str, object]:
    """
    Driver: run the whole alignment and return a result dict (also saved as JSON).
    """
    # 1) Audio -> y,sr
    pp_cfg = PreprocConfig(target_sr=int(target_sr))
    y, sr = LoadAndStandardize(audio_path, pp_cfg)

    # 2) Audio ENF
    ta, fia, stats_audio = _estimate_audio_enf(
        y, sr,
        nominal=nominal, strategy=strategy, bw=bw,
        smooth_sec=smooth_sec, smooth_method=smooth_method,
    )
    if fia.size < 4:
        payload = {
            "ok": False,
            "reason": "insufficient_audio_enf",
            "audio": audio_path,
            "log_csv": log_csv,
            "sr": int(sr),
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    # 3) External ENF
    te_raw, fie_raw = _load_enf_csv(
        log_csv, time_col=csv_time_col, fi_col=csv_fi_col,
        delim=csv_delim, has_header=csv_has_header,
        time_scale=time_scale, time_offset=time_offset,
    )
    if te_raw.size < 4:
        payload = {
            "ok": False,
            "reason": "insufficient_external_enf",
            "audio": audio_path,
            "log_csv": log_csv,
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    # 4) Find overlap and uniform grid
    # we use absolute seconds from audio start for ta; external is relative to its own start
    # choose [t0, t1] from intersection after rough trimming
    t0 = max(float(ta[0]), float(te_raw[0]))
    t1 = min(float(ta[-1]), float(te_raw[-1]))
    # If there is no natural overlap, we still align by correlation allowing lags later.
    # Build each on its own full range, then restrict by lags in cross-corr step.
    # Here, to keep it simple, create grids on the *common* range (conservative).
    # If too small, widen to each own range and allow lag compensation.
    if t1 - t0 < resample_sec * 10:
        # conservative fallback: use max of starts and min of ends after padding
        t0 = min(float(ta[0]), float(te_raw[0]))
        t1 = max(float(ta[-1]), float(te_raw[-1]))

    tg, fia_g = _interp_to_grid(ta, fia, t0, t1, resample_sec)
    _, fie_g = _interp_to_grid(te_raw, fie_raw, t0, t1, resample_sec)

    # Remove NaNs (missing samples after interpolation)
    mask = np.isfinite(fia_g) & np.isfinite(fie_g)
    tg = tg[mask]; fia_g = fia_g[mask]; fie_g = fie_g[mask]
    if tg.size < 16:
        payload = {
            "ok": False,
            "reason": "insufficient_overlap_after_interp",
            "n_grid": int(tg.size),
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    # 5) Detrend + z-score
    fia_d = _zscore(_detrend(fia_g, detrend_mode))
    fie_d = _zscore(_detrend(fie_g, detrend_mode))

    # 6) Correlation over +- max_lag
    Lmax = int(round(max_lag_sec / resample_sec))
    lags, rvals = _xcorr_norm_at_lags(fia_d, fie_d, Lmax)
    # best lag (samples) and as seconds
    k_best = int(np.nanargmax(rvals)) if rvals.size else 0
    lag_samp = int(lags[k_best]) if rvals.size else 0
    rho_max = float(rvals[k_best]) if rvals.size else float("nan")
    lag_sec = float(lag_samp * resample_sec)

    # 7) Pack result
    result: Dict[str, object] = {
        "ok": True,
        "audio": audio_path,
        "log_csv": log_csv,
        "sr": int(sr),
        "nominal": float(nominal),
        "strategy": strategy,
        "bw": float(bw),
        "smooth_sec": float(smooth_sec),
        "smooth_method": smooth_method,
        "resample_sec": float(resample_sec),
        "detrend": detrend_mode,
        "max_lag_sec": float(max_lag_sec),
        "lag_sec": lag_sec,          # shift to apply to external log so it aligns to audio
        "rho_max": rho_max,          # peak normalized correlation [-1,1]
        "n_overlap": int(tg.size),
        "audio_stats": stats_audio,
    }

    # 8) Save JSON
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Saved alignment JSON to: {out_json}")

    # 9) Optional plot
    if plot_path:
        try:
            import matplotlib.pyplot as plt

            # Build aligned view: shift external by best lag in samples
            # We visualize on index-domain to avoid re-interp drift
            x = np.arange(tg.size)
            if lag_samp >= 0:
                a_view = fia_d[: tg.size - lag_samp]
                e_view = fie_d[lag_samp:]
                xi = x[: tg.size - lag_samp]
            else:
                a_view = fia_d[-lag_samp:]
                e_view = fie_d[: tg.size + lag_samp]
                xi = x[: tg.size + lag_samp]

            # Correlation curve vs lag (in seconds)
            lsec = lags * resample_sec

            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
            ax[0].plot(xi, a_view, label="audio ENF (z-scored)")
            ax[0].plot(xi, e_view, label=f"external ENF shifted ({lag_sec:+.1f}s)")
            ax[0].set_title("ENF (z-scored) after lag alignment")
            ax[0].set_xlabel("grid index (≈ seconds)")
            ax[0].set_ylabel("z-score")
            ax[0].legend(loc="best")

            ax[1].plot(lsec, rvals)
            ax[1].axvline(lag_sec, linestyle="--", label=f"ρ_max={rho_max:.3f} at {lag_sec:+.1f}s")
            ax[1].set_title("Normalized cross-correlation vs lag")
            ax[1].set_xlabel("lag (seconds)")
            ax[1].set_ylabel("corr")
            ax[1].legend(loc="best")

            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved plot to: {plot_path}")
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Align audio ENF to an external ENF log via NCC.")
    ap.add_argument("--audio", required=True, type=str, help="Audio file path")
    ap.add_argument("--log-csv", required=True, type=str, help="External ENF CSV (time, fi)")

    # preproc
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate for preprocess")

    # ENF extraction
    ap.add_argument("--nominal", type=float, default=60.0, help="Nominal ENF (50 or 60 Hz)")
    ap.add_argument("--strategy", type=str, default="stft", choices=["stft", "zc"], help="ENF backend")
    ap.add_argument("--bw", type=float, default=1.5, help="Half-band for search/BP (Hz)")
    ap.add_argument("--smooth-sec", type=float, default=0.25, help="Smoothing seconds for the ENF curve")
    ap.add_argument("--smooth-method", type=str, default="savgol", choices=["savgol", "median"], help="Smoothing filter")

    # CSV parsing
    ap.add_argument("--csv-time-col", type=str, default="time", help="CSV column with time (seconds)")
    ap.add_argument("--csv-fi-col", type=str, default="fi", help="CSV column with frequency (Hz)")
    ap.add_argument("--csv-delim", type=str, default=",", help="CSV delimiter")
    ap.add_argument("--csv-has-header", action="store_true", help="CSV has header row")
    ap.add_argument("--time-scale", type=float, default=1.0, help="Multiply times by this factor")
    ap.add_argument("--time-offset", type=float, default=0.0, help="Add this offset to times (seconds)")

    # alignment
    ap.add_argument("--resample-sec", type=float, default=1.0, help="Uniform grid step (seconds)")
    ap.add_argument("--max-lag-sec", type=float, default=120.0, help="Max lag to search (seconds)")
    ap.add_argument("--detrend", type=str, default="linear", choices=["none", "demean", "linear"], help="Detrend mode before NCC")

    # outputs
    ap.add_argument("--out-json", required=True, type=str, help="Output JSON with alignment diagnostics")
    ap.add_argument("--plot", type=str, default=None, help="Optional PNG path to save a diagnostic plot")
    args = ap.parse_args()

    align_enf(
        audio_path=args.audio,
        log_csv=args.log_csv,
        out_json=args.out_json,
        plot_path=args.plot,
        target_sr=args.sr,
        nominal=args.nominal,
        strategy=args.strategy,
        bw=args.bw,
        smooth_sec=args.smooth_sec,
        smooth_method=args.smooth_method,
        csv_time_col=args.csv_time_col,
        csv_fi_col=args.csv_fi_col,
        csv_delim=args.csv_delim,
        csv_has_header=bool(args.csv_has_header),
        time_scale=args.time_scale,
        time_offset=args.time_offset,
        resample_sec=args.resample_sec,
        detrend_mode=args.detrend,
        max_lag_sec=args.max_lag_sec,
    )


if __name__ == "__main__":
    main()
