# src/hoaxhertz/features/mfcc.py
"""
MFCC (Mel-Frequency Cepstral Coefficients) feature extractor.

What are MFCCs?
---------------
MFCCs are a compact representation of the short-term spectral envelope of audio.
They are computed by:
  1) Framing & windowing the signal
  2) Magnitude spectrum (STFT)
  3) Mel filterbank integration (psychoacoustic frequency scale)
  4) Log compression
  5) DCT (Discrete Cosine Transform) across mel bands

Why use MFCCs?
--------------
- They summarize the spectral *shape* (formant structure / timbre) rather than
  raw spectral magnitudes.
- Widely useful for speech/music tasks and as robust low-dimensional features.

This extractor:
---------------
- Aligns STFT parameters to the preprocessing stage when available (n_fft/hop).
- Optionally applies per-clip CMVN (cepstral mean and variance normalization).
- Can compute Δ (delta) and ΔΔ (delta-delta) both framewise and, optionally,
  include their aggregated statistics in the clip-level vector.
- Produces:
    * framewise: time series (T, D) for MFCC (and optional deltas)
    * vector   : clip-level summary by aggregating over time
                 (either mean+std or median+IQR, per coefficient)

Shapes & Notation
-----------------
- T = number of frames
- D = number of MFCC coefficients (cfg.n_mfcc)
- Framewise outputs stored as (T, D) for consistency across modules
  (Librosa returns MFCC as (D, T); we transpose to (T, D)).

Notes
-----
- For speech at 16 kHz, typical choices are: n_fft=1024 (≈64 ms), hop=160 (10 ms),
  n_mels ∈ [40, 80], n_mfcc ∈ [13, 20]. Choose fmin/fmax to match your band of interest.
- CMVN usually helps reduce channel effects; consider enabling it for classification tasks.
"""

from __future__ import annotations

import librosa
import numpy as np
from pydantic import ConfigDict

from hoaxhertz.features import (
    FeatureConfigBase,
    FeatureExtractor,
    FeatureOutput,
    PreprocInput,
)

# =============================================================================
# Configuration model
# =============================================================================


class MFCCConfig(FeatureConfigBase):
    """
    Configuration for MFCCFeatureExtractor.

    Core MFCC parameters (mirrors librosa.feature.mfcc):
    ----------------------------------------------------
    n_mfcc   : number of cepstral coefficients D to compute.
    n_fft    : STFT window size (samples). If None, use pre.n_fft if provided,
               else fall back to librosa default (2048 here).
    hop_length : STFT hop (samples). If None, use pre.hop if provided,
                 else fall back to librosa-like default (512 here).
    n_mels   : number of mel bands.
    fmin/fmax: mel filterbank frequency range (Hz). fmax=None means Nyquist.
    dct_type : DCT variant; 2 is the standard for MFCCs.
    lifter   : cepstral liftering parameter (0=disabled).
    htk      : use HTK-style mel scale if True, else Slaney-style (librosa default).
    norm     : DCT normalization mode ("ortho" by default).

    Extras:
    -------
    use_deltas      : compute Δ (1st derivative) framewise.
    use_delta2      : compute ΔΔ (2nd derivative) framewise.
    cmvn            : apply per-clip CMVN to MFCC (mean 0, std 1) over time.
    use_robust      : if True, aggregate with (median, IQR); else (mean, std).
    return_framewise: include framewise tensors in the output FeatureOutput.
    vector_include_deltas  : include aggregated Δ stats in clip-level vector.
    vector_include_delta2  : include aggregated ΔΔ stats in clip-level vector.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "mfcc"

    # Core MFCC params
    n_mfcc: int = 20
    n_fft: int | None = None           # if None, use pre.n_fft or 2048
    hop_length: int | None = None      # if None, use pre.hop or 512
    n_mels: int = 40
    fmin: float = 50.0
    fmax: float | None = None
    dct_type: int = 2
    lifter: int = 0
    htk: bool = False
    norm: str = "ortho"

    # Extras
    use_deltas: bool = True
    use_delta2: bool = False
    cmvn: bool = False  # per-clip CMVN

    # Aggregation (for clip-level vector)
    use_robust: bool = False           # median+IQR if True; else mean+std
    return_framewise: bool = True      # keep framewise outputs

    # Include deltas in clip-level vector?
    vector_include_deltas: bool = True
    vector_include_delta2: bool = False


# =============================================================================
# Utilities
# =============================================================================


def _frame_times(T: int, sr: int, hop_length: int | None) -> np.ndarray:
    """
    Compute frame center times in seconds for T frames, given audio sr and hop.
    If sr/hop_length are not valid, return a simple arange [0..T-1] as a fallback.
    """
    if hop_length is None or sr <= 0:
        return np.arange(T, dtype=float)
    return librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)


def _cmvn_inplace(M: np.ndarray) -> None:
    """
    Apply per-clip CMVN (mean 0, std 1) per coefficient along time, in-place.

    Accepts either (D, T) or (T, D). The function will internally operate on (D, T)
    and then put the data back in the original layout.

    Rationale
    ---------
    CMVN reduces channel/microphone effects by normalizing each cepstral dimension
    over time. It generally helps downstream classifiers to focus on content rather
    than absolute scale.

    Notes
    -----
    - Adds a small epsilon to the std to avoid division-by-zero.
    - If M is empty or not 2D, does nothing.
    """
    if M.ndim != 2 or M.size == 0:
        return

    # Heuristic: if the first dimension is smaller, we assume (D, T); else (T, D).
    transposed = False
    if M.shape[0] < M.shape[1]:
        X = M  # (D, T)
    else:
        X = M.T  # (D, T)
        transposed = True

    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-9
    X -= mu
    X /= std

    if transposed:
        M[:] = X.T
    else:
        M[:] = X


def _aggregate_over_time(F: np.ndarray, robust: bool) -> np.ndarray:
    """
    Aggregate a 2D matrix over the time axis into a single vector.

    Inputs can be (T, D) or (D, T). The function ensures a (D, T) view and then
    aggregates each coefficient over time using:
      - robust=True  -> (median, IQR) per coefficient  -> vector length 2*D
      - robust=False -> (mean,   std) per coefficient  -> vector length 2*D

    Returns
    -------
    np.ndarray
        A (2*D,) vector. Returns an empty (0,) vector if F is not 2D or empty.
    """
    if F.ndim != 2 or F.size == 0:
        return np.zeros((0,), dtype=float)

    # Ensure X is (D, T). Typically T >> D.
    X = F.T if F.shape[0] > F.shape[1] else F

    if robust:
        med = np.median(X, axis=1)
        p25 = np.percentile(X, 25, axis=1)
        p75 = np.percentile(X, 75, axis=1)
        iqr = p75 - p25
        vec = np.concatenate([med, iqr], axis=0)
    else:
        mu = X.mean(axis=1)
        std = X.std(axis=1)
        vec = np.concatenate([mu, std], axis=0)

    return vec.astype(float)


# =============================================================================
# Extractor
# =============================================================================


class MFCCFeatureExtractor(FeatureExtractor):
    """
    MFCC extractor conforming to the generic FeatureExtractor interface.

    Processing steps
    ----------------
    1) Resolve STFT parameters (n_fft/hop) from config or preprocessing metadata.
    2) Compute MFCCs with librosa (shape (D, T)).
    3) Optionally apply per-clip CMVN to MFCCs (over time).
    4) Build framewise outputs in (T, D): MFCC (and optionally Δ, ΔΔ).
    5) Aggregate over time to produce a clip-level vector (mean+std OR median+IQR),
       optionally concatenating aggregated Δ and/or ΔΔ statistics.
    6) Return FeatureOutput with vector, optional framewise dict, meta, and params.

    Notes
    -----
    - Frame times are computed from (sr, hop_length) for easy alignment/plotting.
    - The clip-level vector length depends on:
        base: 2*D
        +Δ   : 2*D (if vector_include_deltas and use_deltas)
        +ΔΔ  : 2*D (if vector_include_delta2 and use_delta2)
    """

    def configure(self, cfg: FeatureConfigBase) -> None:
        """
        Validate and store the configuration model.
        """
        if not isinstance(cfg, MFCCConfig):
            raise TypeError("MFCCFeatureExtractor expects MFCCConfig.")
        self._cfg = cfg

    def extract(self, pre: PreprocInput) -> FeatureOutput:
        """
        Run the MFCC pipeline on a preprocessed input.

        Parameters
        ----------
        pre : PreprocInput
            Must provide y (np.ndarray) and sr (int). May optionally provide
            n_fft/hop to align with upstream preprocessing.

        Returns
        -------
        FeatureOutput
            vector    : np.ndarray, clip-level aggregated descriptor
            framewise : Optional[dict], with keys:
                        "times": (T,), "mfcc": (T, D),
                        optionally "delta": (T, D), "delta2": (T, D)
            meta      : dict with aggregation choices and CMVN flags
            params    : dict of MFCC/STFT parameters for reproducibility
        """
        cfg = self.cfg  # type: ignore[assignment]
        if pre.y is None or pre.sr is None:
            raise ValueError("PreprocInput must provide y and sr.")

        # 1) Resolve STFT params (prefer config, then preprocessing, else defaults)
        n_fft = (
            cfg.n_fft if cfg.n_fft is not None else (pre.n_fft if pre.n_fft is not None else 2048)
        )
        hop_length = (
            cfg.hop_length
            if cfg.hop_length is not None
            else (pre.hop if pre.hop is not None else 512)
        )

        # 2) Compute base MFCCs. Librosa returns (n_mfcc, T) == (D, T).
        MF = librosa.feature.mfcc(
            y=pre.y,
            sr=int(pre.sr),
            n_mfcc=int(cfg.n_mfcc),
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            n_mels=int(cfg.n_mels),
            fmin=float(cfg.fmin),
            fmax=None if cfg.fmax is None else float(cfg.fmax),
            dct_type=int(cfg.dct_type),
            lifter=int(cfg.lifter),
            htk=bool(cfg.htk),
            norm=cfg.norm,
        )

        # 3) Optional per-clip CMVN on MFCCs (operates along time dimension)
        if cfg.cmvn:
            _cmvn_inplace(MF)  # safe for (D, T)

        # 4) Build framewise payload in (T, D)
        T = int(MF.shape[1])
        mfcc_TD = MF.T.astype(float)
        framewise: dict[str, np.ndarray] = {"mfcc": mfcc_TD}

        d1 = d2 = None
        if cfg.use_deltas:
            # librosa.feature.delta expects (D, T); returns (D, T). We transpose to (T, D).
            d1 = librosa.feature.delta(MF, order=1).T.astype(float)
            framewise["delta"] = d1
        if cfg.use_delta2:
            d2 = librosa.feature.delta(MF, order=2).T.astype(float)
            framewise["delta2"] = d2

        # Frame center times in seconds (T,)
        times = _frame_times(T=T, sr=int(pre.sr), hop_length=int(hop_length))
        framewise["times"] = times

        # 5) Clip-level vector by aggregating over time
        vec_parts = []
        vec_parts.append(_aggregate_over_time(mfcc_TD, robust=cfg.use_robust))
        if cfg.vector_include_deltas and d1 is not None:
            vec_parts.append(_aggregate_over_time(d1, robust=cfg.use_robust))
        if cfg.vector_include_delta2 and d2 is not None:
            vec_parts.append(_aggregate_over_time(d2, robust=cfg.use_robust))

        vector = np.concatenate(vec_parts, axis=0) if vec_parts else np.zeros((0,), dtype=float)

        # 6) Return structured output
        return FeatureOutput(
            vector=vector,
            framewise=framewise if cfg.return_framewise else None,
            meta={
                "agg": "robust" if cfg.use_robust else "meanstd",
                "cmvn": bool(cfg.cmvn),
                "with_delta": bool(cfg.use_deltas),
                "with_delta2": bool(cfg.use_delta2),
                "vector_include_deltas": bool(cfg.vector_include_deltas),
                "vector_include_delta2": bool(cfg.vector_include_delta2),
            },
            params={
                "sr": int(pre.sr),
                "n_mfcc": int(cfg.n_mfcc),
                "n_fft": int(n_fft),
                "hop_length": int(hop_length),
                "n_mels": int(cfg.n_mels),
                "fmin": float(cfg.fmin),
                "fmax": None if cfg.fmax is None else float(cfg.fmax),
                "dct_type": int(cfg.dct_type),
                "lifter": int(cfg.lifter),
                "htk": bool(cfg.htk),
                "norm": cfg.norm,
            },
        )
