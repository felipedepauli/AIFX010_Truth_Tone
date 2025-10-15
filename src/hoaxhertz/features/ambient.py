# src/hoaxhertz/features/ambient.py
"""
Ambient (Acoustic Environment Signature) feature extractor.

Two levels:
- ambient_signature: simple fixed vector (mean+var of log-mel)
- compute_aes: robust vector (median+IQR or mean+std per mel band) + flatness,
               optionally with framewise outputs for temporal analysis.

Also provides a concrete AmbientFeatureExtractor implementing the generic interface.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from pydantic import ConfigDict

from hoaxhertz.features import (
    FeatureConfigBase,
    FeatureExtractor,
    FeatureOutput,
    PreprocInput,
)

# --------- Utility functions (building blocks) --------------------------------


def ambient_signature(y: np.ndarray, sr: int, n_mels: int = 40) -> np.ndarray:
    """
    Simple ambient signature: log-mel mean and variance per band.
    Returns a fixed-length vector of size (2 * n_mels,).
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logS = librosa.power_to_db(S + 1e-9)
    mu = logS.mean(axis=1)
    var = logS.var(axis=1)
    return np.concatenate([mu, var], axis=0)


def compute_aes(
    y: np.ndarray,                  # Audio signal (1D numpy array)
    sr: int,                        # Sample rate of audio
    n_mels: int = 40,               # Number of mel bands
    n_fft: int = 1024,              # FFT window size
    hop_length: int = 160,          # Hop length between frames
    fmin: int = 50,                 # Minimum frequency for mel filterbank
    fmax: int | None = None,        # Maximum frequency for mel filterbank
    use_robust: bool = True,        # Use robust statistics (median/IQR) if True, else mean/std
    use_vad: bool = True,           # Apply voice activity detection (VAD) based on RMS
    vad_percentile: int = 30,       # Percentile threshold for VAD (frames below this RMS are kept)
    return_framewise: bool = False, # If True, return framewise features in addition to vector
) -> dict[str, Any]:
    """
    Acoustic Environment Signature (AES) with optional robust aggregation.

    Returns (return_framewise=False):
        {
          "vector": (D,),
          "proto": {...},
          "params": {...}
        }

    Returns (return_framewise=True):
        {
          "framewise": {"logmel": (T,M), "sfm": (T,), "times": (T,)},
          "vector": (D,),
          "proto": {...},
          "params": {...}
        }
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    logS = librosa.power_to_db(S + 1e-12)  # (M, T)

    T = logS.shape[1]
    times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)

    sfm = librosa.feature.spectral_flatness(S=S + 1e-12).squeeze()  # (T,)

    if use_vad:
        rms = librosa.feature.rms(
            y=y,
            frame_length=n_fft,
            hop_length=hop_length,
            center=True,
        ).squeeze()
        # Garantir mesmo tamanho de T:
        if rms.shape[0] != T:
            # Trunca ou padroniza para T; aqui eu tronco ao menor comum
            L = min(rms.shape[0], T)
            rms = rms[:L]
            logS_used_base = logS[:, :L]
            sfm = sfm[:L]
        else:
            logS_used_base = logS
        thr = np.percentile(rms, vad_percentile)
        keep = rms <= thr
        logS_used = logS[:, keep] if keep.any() else logS
        sfm_used = sfm[keep] if keep.any() else sfm
    else:
        logS_used = logS
        sfm_used = sfm

    if use_robust:
        med = np.median(logS_used, axis=1)
        p25 = np.percentile(logS_used, 25, axis=1)
        p75 = np.percentile(logS_used, 75, axis=1)
        iqr = p75 - p25
        core = np.concatenate([med, iqr], axis=0)  # (2M,)
        agg_type = "robust"
    else:
        mu = logS_used.mean(axis=1)
        std = logS_used.std(axis=1)
        core = np.concatenate([mu, std], axis=0)  # (2M,)
        agg_type = "meanstd"

    sfm_mean = float(np.mean(sfm_used))
    sfm_std = float(np.std(sfm_used))

    vector = np.concatenate([core, np.array([sfm_mean, sfm_std], dtype=float)], axis=0)  # (2M+2,)

    out: dict[str, Any] = {
        "vector": vector,
        "proto": {
            "type": agg_type,
            "sfm_mean": sfm_mean,
            "sfm_std": sfm_std,
            "bands": n_mels,
        },
        "params": {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "fmin": fmin,
            "fmax": fmax,
            "sr": sr,
        },
    }

    if return_framewise:
        out["framewise"] = {
            "logmel": logS.T,  # (T, M)
            "sfm": sfm,  # (T,)
            "times": times,  # (T,)
        }
    return out


# --------- Config model for Ambient extractor ---------------------------------


class AmbientConfig(FeatureConfigBase):
    """
    Configuration for AmbientFeatureExtractor.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "ambient"
    # Spectral parameters
    n_mels: int = 40
    n_fft: int = 1024
    hop_length: int = 160
    fmin: int = 50
    fmax: int | None = None
    # Aggregation / robustness
    use_robust: bool = True
    use_vad: bool = True
    vad_percentile: int = 30
    # Output form
    return_framewise: bool = False
    # Fast path toggle: when True, use simple ambient_signature (2*M dims)
    fast_signature: bool = False


# --------- Concrete extractor -------------------------------------------------


class AmbientFeatureExtractor(FeatureExtractor):
    """
    Ambient/acoustic environment feature extractor that implements the generic interface.

    Steps:
        - Read preprocessed inputs (y, sr, optional frames info)
        - Depending on config, extract:
            * fast_signature -> concat(mean, var) over log-mel
            * compute_aes -> robust statistics + flatness (and optional framewise)
        - Return FeatureOutput with vector/framewise/meta/params
    """

    def configure(self, cfg: FeatureConfigBase) -> None:
        if not isinstance(cfg, AmbientConfig):
            raise TypeError("AmbientFeatureExtractor expects AmbientConfig.")
        self._cfg = cfg

    def extract(self, pre: PreprocInput) -> FeatureOutput:
        # Validate config and inputs
        cfg = self.cfg
        if pre.y is None or pre.sr is None:
            raise ValueError("PreprocInput must provide y and sr.")

        if cfg.fast_signature:
            vec = ambient_signature(pre.y, pre.sr, n_mels=cfg.n_mels)
            return FeatureOutput(
                vector=vec,
                framewise=None,
                meta={"mode": "fast_signature", "bands": cfg.n_mels},
                params={"sr": pre.sr},
            )

        # Full AES
        aes = compute_aes(
            y=pre.y,
            sr=pre.sr,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            use_robust=cfg.use_robust,
            use_vad=cfg.use_vad,
            vad_percentile=cfg.vad_percentile,
            return_framewise=cfg.return_framewise,
        )

        return FeatureOutput(
            vector=aes["vector"],
            framewise=aes.get("framewise"),
            meta=aes.get("proto", {}),
            params=aes.get("params", {}),
        )
