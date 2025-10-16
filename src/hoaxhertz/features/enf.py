# src/hoaxhertz/features/enf.py
"""
ENF (Electric Network Frequency) feature extractor.

What is ENF?
------------
ENF is the instantaneous frequency of the mains power grid (usually 50 Hz or 60 Hz).
It fluctuates slowly around the nominal value (e.g., 60 ± 0.02 Hz) depending on the
generation/consumption balance. That fluctuation can leak into audio recordings via:

- Electromagnetic coupling (cables, power supplies, grounding paths)
- Harmonic content from switched-mode power supplies / lighting (e.g., 100/120 Hz)
- Indirect acoustic pickup of hum-like components (fans, motors, fixtures)

Why estimate ENF from audio?
----------------------------
The ENF trace fi(t) can be used for audio authenticity checks, temporal correlation
with external ENF logs, and edit/splicing detection. In many cases, a clean ENF line
(or its harmonics) exists within the recording bandwidth, allowing time-resolved
estimation.

How this module works (baseline):
---------------------------------
- It provides two lightweight backends (Strategy Pattern):
  1) Time domain ("zc"): band-pass filter around nominal and estimate the period
     from zero-crossings. Very fast; works best with strong SNR near nominal.
     IMPORTANT: consecutive zero-crossings correspond to half a cycle, so we use
     0.5 / period to get Hz.
  2) Frequency domain ("stft"): compute an STFT and, per frame, pick the spectral
     peak closest to nominal within a narrow band. More robust to noise; absolute
     frequency resolution depends on window length (and improves with sub-bin
     interpolation or phase-based estimators, which are beyond this baseline).

- The estimated ENF trajectory fi(t) may be optionally smoothed (Savitzky–Golay or
  median filter) to reduce jitter/quantization.
- A short, ML-friendly summary vector is produced at clip-level:
    [mean_dev, std_dev, min, max, slope_hz_per_min]
  where mean_dev/std_dev are computed relative to the nominal.

Outputs
-------
- vector: fixed-length summary (float[5]) suitable for ML or indexing
- framewise: dict with "times" (s) and "fi" (Hz) for time-series analyses
- params/meta: reproducibility and diagnostics about estimation quality

Notes & Caveats
---------------
- This is a minimal, easy-to-extend baseline. For forensic-grade ENF consider:
  * harmonic fusion (e.g., tracking at 100/120 Hz, 150/180 Hz, etc.)
  * sub-bin peak interpolation or phase-based estimators
  * phase unwrapping across frames
  * per-frame quality scoring (e.g., peak-to-median ratio in the search band)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy.signal as sg
from pydantic import ConfigDict

from hoaxhertz.features import (
    FeatureConfigBase,
    FeatureExtractor,
    FeatureOutput,
    PreprocInput,
)

# =============================================================================
# Utility functions
# =============================================================================


def _safe_len(x: np.ndarray | None) -> int:
    """
    Return len(x) if x is a numpy array, otherwise 0.
    Useful for defensive checks on optional arrays.
    """
    return int(x.shape[0]) if isinstance(x, np.ndarray) else 0


def _savgol_or_median(
    x: np.ndarray, sr: float, smooth_sec: float, method: str = "savgol"
) -> np.ndarray:
    """
    Smooth a 1D series using either Savitzky–Golay or a median filter.

    Parameters
    ----------
    x : np.ndarray
        1D input series (e.g., ENF trajectory fi over time).
    sr : float
        Sampling rate of the input series x (NOT the audio sr).
        Example: if x has one value per 50 ms, sr ≈ 20 Hz.
    smooth_sec : float
        Desired window length in seconds (<=0 disables smoothing).
    method : {"savgol","median"}
        Smoothing method. Savitzky–Golay preserves trends well; median is
        robust to outliers.

    Returns
    -------
    np.ndarray
        Smoothed series (same shape as x). Returns x unchanged on errors.
    """
    if x is None or x.size == 0 or smooth_sec <= 0:
        return x

    # Convert desired seconds to a window length in samples of the series x.
    win = max(3, int(round(smooth_sec * sr)))
    if win % 2 == 0:
        win += 1  # most 1D smoothing kernels require odd window size

    if method == "median":
        # Median filter is excellent to remove isolated spikes/outliers
        return sg.medfilt(x, kernel_size=win)

    # Savitzky–Golay: requires window_length >= polyorder+2 and an odd window
    poly = min(3, max(1, win // 5))  # small polynomial order is usually enough
    if win <= poly:
        # Ensure valid window_length > poly and odd
        win = poly + 2 + (1 - (poly % 2))
    if win % 2 == 0:
        win += 1

    try:
        return sg.savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
    except Exception:
        # On numerical edge cases, just return the original series
        return x


def _summary_stats(fi: np.ndarray, nominal: float, times: np.ndarray | None) -> dict[str, float]:
    """
    Compute summary statistics for an ENF trajectory around a nominal frequency.

    The goal is to produce a short, ML-ready descriptor that captures central
    tendency, variability, range, and linear trend (slope).

    Parameters
    ----------
    fi : np.ndarray
        ENF estimates over time (Hz). Shape: (T,)
    nominal : float
        Nominal mains frequency (50.0 or 60.0).
    times : np.ndarray | None
        Timestamps in seconds for fi (same shape), used to estimate slope.
        If absent or invalid, slope is reported as NaN.

    Returns
    -------
    dict[str, float]
        {
          "available": number_of_points,
          "mean_dev": mean(fi - nominal),
          "std_dev":  std(fi - nominal),
          "min":      min(fi),
          "max":      max(fi),
          "slope_hz_per_min": linear_trend_in_Hz_per_min
        }
    """
    if fi is None or fi.size == 0:
        return {
            "available": 0.0,
            "mean_dev": np.nan,
            "std_dev": np.nan,
            "min": np.nan,
            "max": np.nan,
            "slope_hz_per_min": np.nan,
        }

    dev = fi - nominal
    mean_dev = float(np.mean(dev))
    std_dev = float(np.std(dev))
    fmin = float(np.min(fi))
    fmax = float(np.max(fi))

    slope = np.nan
    if isinstance(times, np.ndarray) and times.size == fi.size and times.size >= 2:
        # Fit fi(t) = m * t + b in least squares and report slope in Hz/min
        t = times - times[0]
        A = np.vstack([t, np.ones_like(t)]).T
        m, _ = np.linalg.lstsq(A, fi, rcond=None)[0]
        slope = float(m * 60.0)  # Hz/s -> Hz/min

    return {
        "available": float(fi.size),
        "mean_dev": mean_dev,
        "std_dev": std_dev,
        "min": fmin,
        "max": fmax,
        "slope_hz_per_min": slope,
    }


# =============================================================================
# Strategy interface and concrete strategies
# =============================================================================


class ENFStrategy(ABC):
    """
    Strategy interface for ENF estimation backends.

    Implementations must return (times, fi):
      - times: np.ndarray, shape (T,), timestamps in seconds
      - fi:    np.ndarray, shape (T,), ENF estimates in Hz (one per time stamp)
    """

    @abstractmethod
    def estimate(
        self,
        y: np.ndarray,
        sr: int,
        nominal: float,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate ENF trajectory from an audio waveform.

        Parameters
        ----------
        y : np.ndarray
            Audio samples (mono). Shape: (N,)
        sr : int
            Audio sampling rate in Hz.
        nominal : float
            Nominal mains frequency (50.0 or 60.0).
        **kwargs : Any
            Strategy-specific parameters (e.g., bw, order, STFT sizes).

        Returns
        -------
        (times, fi) : (np.ndarray, np.ndarray)
            times: seconds, shape (T,)
            fi:    Hz,      shape (T,)
        """
        ...


class ZeroCrossBandpassStrategy(ENFStrategy):
    """
    Time-domain baseline: band-pass around nominal ± bw, then estimate
    instantaneous frequency from zero-crossings.

    Rationale
    ---------
    - Band-pass filtering isolates the narrowband hum near nominal.
    - In a quasi-sinusoid, consecutive zero-crossings correspond to HALF a cycle.
      If Δt is the time between adjacent zero-crossings, the frequency is:
          f ≈ 0.5 / Δt
      (Using 1/Δt would overestimate ~2×.)

    Strengths / Limitations
    -----------------------
    + Extremely fast and simple
    - Sensitive to noise and spurious crossings
    - Requires sufficient SNR around nominal
    """

    def estimate(
        self,
        y: np.ndarray,
        sr: int,
        nominal: float,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        bw: float = float(kwargs.get("bw", 1.5))
        order: int = int(kwargs.get("order", 6))

        # Design a Butterworth band-pass in normalized frequencies [0, 1]
        low = max(0.0, (nominal - bw) / (sr / 2.0))
        high = min(0.999, (nominal + bw) / (sr / 2.0))
        if low >= high:
            return np.array([], dtype=float), np.array([], dtype=float)

        sos = sg.iirfilter(order, [low, high], btype="band", ftype="butter", output="sos")
        z = sg.sosfilt(sos, y)

        # Locate zero-crossings (sign changes). We need at least 2 to estimate one period.
        zc = np.where(np.diff(np.signbit(z)))[0]
        if zc.size < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        # Estimate periods between consecutive crossings (half-cycle)
        periods_half = np.diff(zc) / float(sr)  # seconds
        fi = 0.5 / np.maximum(periods_half, 1e-9)  # Hz, guard against divide-by-zero

        # Timestamp each fi at the midpoint between the two crossings
        t = (zc[:-1] + zc[1:]) / 2.0 / float(sr)

        return t.astype(float), fi.astype(float)


class STFTPeakTrackingStrategy(ENFStrategy):
    """
    Frequency-domain baseline: track, for each STFT frame, the magnitude peak
    closest to the nominal within a narrow band.

    Rationale
    ---------
    - More robust to broadband noise than zero-crossings.
    - Absolute frequency resolution is limited by the FFT bin spacing (sr/nperseg);
      smoothing can reduce quantization jitter, but for sub-Hz accuracy one should
      use sub-bin interpolation or phase-based estimators.

    Strengths / Limitations
    -----------------------
    + Tolerates noisier conditions
    - Coarse resolution without sub-bin interpolation
    - More computationally expensive than zero-crossings
    """

    def estimate(
        self,
        y: np.ndarray,
        sr: int,
        nominal: float,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        nperseg: int = int(kwargs.get("stft_nperseg", 4096))
        noverlap: int = int(kwargs.get("stft_noverlap", nperseg // 2))
        bw_hz: float = float(kwargs.get("bw", 1.5))  # half-bandwidth around nominal, in Hz

        freqs, times, Zxx = sg.stft(
            y, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary="zeros"
        )
        mag = np.abs(Zxx)  # shape: (F, T)

        # Restrict to the search band [nominal - bw, nominal + bw]
        fmin = max(0.0, nominal - bw_hz)
        fmax = nominal + bw_hz
        band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if band.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        # For each frame, pick the frequency bin with largest magnitude in the band
        submag = mag[band, :]  # shape: (Fb, T)
        idx = np.argmax(submag, axis=0)  # argmax over Fb for each of T frames
        fi = freqs[band[idx]]  # shape: (T,)

        return times.astype(float), fi.astype(float)


# =============================================================================
# Configuration and extractor
# =============================================================================


class ENFConfig(FeatureConfigBase):
    """
    Configuration model for ENFFeatureExtractor.

    Fields
    ------
    nominal : float
        Nominal mains frequency. Use 60.0 in 60 Hz regions, 50.0 in 50 Hz regions.
    bw : float
        Half-bandwidth (Hz) for both band-pass (zc) and search band (stft).
        Effective search range is [nominal - bw, nominal + bw].

    strategy : {"zc","stft"}
        Backend selection. "zc" = ZeroCrossBandpassStrategy, "stft" = STFTPeakTrackingStrategy.

    order : int
        IIR filter order for the band-pass in zero-crossing strategy.

    stft_nperseg : int
        STFT window length (samples). Larger windows increase frequency resolution
        (sr / nperseg) but also latency/cost.

    stft_noverlap : int
        STFT overlap (samples). Typical values are nperseg // 2 or higher.

    smooth_sec : float
        Smoothing window in seconds for the ENF trajectory (0 disables smoothing).

    smooth_method : {"savgol","median"}
        Smoothing method. Savitzky–Golay preserves slopes; median is robust to spikes.

    return_framewise : bool
        Whether to include the per-frame trajectory {"times","fi"} in the output.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "enf"

    # Nominal and search band
    nominal: float = 60.0
    bw: float = 1.5

    # Strategy selection
    strategy: str = "zc"  # "zc" (ZeroCrossBandpass) | "stft"
    order: int = 6  # IIR order for band-pass (zc)
    stft_nperseg: int = 4096  # STFT window (stft)
    stft_noverlap: int = 2048  # STFT overlap (stft)

    # Optional smoothing on the estimated trajectory
    smooth_sec: float = 0.25  # seconds (0 = disabled)
    smooth_method: str = "savgol"  # "savgol" | "median"

    # Output controls
    return_framewise: bool = True  # ENF is inherently temporal; keep True by default


class ENFFeatureExtractor(FeatureExtractor):
    """
    ENF feature extractor implementing the generic FeatureExtractor interface.

    Processing steps
    ----------------
    1) Choose an estimation strategy ("zc" or "stft").
    2) Estimate the ENF trajectory: times (s) and fi (Hz).
    3) Optionally smooth the fi series to reduce jitter.
    4) Summarize the trajectory into a compact vector:
         [mean_dev, std_dev, min, max, slope_hz_per_min]
    5) Return framewise data (times, fi) for temporal analyses if requested.

    Notes
    -----
    - The strategy registry makes it easy to add new backends
      (e.g., harmonic fusion, phase-based trackers).
    """

    def __init__(self) -> None:
        super().__init__()
        # Registry of available strategies (extensible)
        self._strategies: dict[str, ENFStrategy] = {
            "zc": ZeroCrossBandpassStrategy(),
            "stft": STFTPeakTrackingStrategy(),
        }

    def configure(self, cfg: FeatureConfigBase) -> None:
        """
        Validate and store the configuration model.
        """
        if not isinstance(cfg, ENFConfig):
            raise TypeError("ENFFeatureExtractor expects ENFConfig.")
        self._cfg = cfg

    def _pick_strategy(self, name: str) -> ENFStrategy:
        """
        Retrieve a strategy instance by name or raise a helpful error.
        """
        if name not in self._strategies:
            raise ValueError(
                f"Unknown ENF strategy '{name}'. Available: {list(self._strategies.keys())}"
            )
        return self._strategies[name]

    def extract(self, pre: PreprocInput) -> FeatureOutput:
        """
        Run the configured ENF pipeline on a preprocessed input.

        Parameters
        ----------
        pre : PreprocInput
            Must provide y (np.ndarray) and sr (int).

        Returns
        -------
        FeatureOutput
            vector : np.ndarray, shape (5,)
                [mean_dev, std_dev, min, max, slope_hz_per_min]
            framewise : Optional[dict]
                {"times": (T,), "fi": (T,)} if return_framewise=True, else None
            meta : dict
                {"available": T, "nominal": nominal, "strategy": name}
            params : dict
                Estimation parameters for reproducibility.
        """
        cfg = self.cfg  # type: ignore[assignment]
        if pre.y is None or pre.sr is None:
            raise ValueError("PreprocInput must provide y and sr.")

        strat = self._pick_strategy(cfg.strategy)

        # 1) Estimate ENF trajectory (times in seconds, fi in Hz)
        times, fi = strat.estimate(
            y=pre.y,
            sr=int(pre.sr),
            nominal=float(cfg.nominal),
            bw=float(cfg.bw),
            order=int(cfg.order),
            stft_nperseg=int(cfg.stft_nperseg),
            stft_noverlap=int(cfg.stft_noverlap),
        )

        # 2) Optional smoothing
        #    Build an equivalent "sampling rate" for the fi series from time deltas.
        if fi.size > 1 and cfg.smooth_sec > 0:
            dt = np.diff(times).mean() if times.size > 1 else 0.0
            curve_sr = 1.0 / max(dt, 1e-6)  # avoid division by zero
            fi = _savgol_or_median(fi, curve_sr, cfg.smooth_sec, method=cfg.smooth_method)

        # 3) Compute summary statistics (ML-ready vector)
        stats = _summary_stats(fi, cfg.nominal, times)
        vector = np.array(
            [
                stats["mean_dev"],
                stats["std_dev"],
                stats["min"],
                stats["max"],
                stats["slope_hz_per_min"],
            ],
            dtype=float,
        )

        # 4) Build optional framewise payload
        framewise: dict[str, np.ndarray] | None = None
        if cfg.return_framewise:
            framewise = {
                "times": times.astype(float),
                "fi": fi.astype(float),
            }

        # 5) Return structured output with meta and reproducibility params
        return FeatureOutput(
            vector=vector,
            framewise=framewise,
            meta={
                "available": stats["available"],
                "nominal": float(cfg.nominal),
                "strategy": cfg.strategy,
            },
            params={
                "bw": float(cfg.bw),
                "order": int(cfg.order),
                "stft_nperseg": int(cfg.stft_nperseg),
                "stft_noverlap": int(cfg.stft_noverlap),
                "smooth_sec": float(cfg.smooth_sec),
                "smooth_method": cfg.smooth_method,
                "sr": int(pre.sr),
            },
        )
