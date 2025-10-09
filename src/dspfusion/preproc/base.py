"""Minimal audio preprocessing helpers.

Provides:
- PreprocConfig: sampling and framing parameters.
- load_and_standardize: load mono audio, resample to target_sr, peak-normalize.
- frame_signal: slice waveform into overlapping short-time frames.
"""

from dataclasses import dataclass
import numpy as np, librosa

@dataclass
class PreprocConfig:
    """Basic parameters for preprocessing and framing.

    Attributes
    ----------
    target_sr : int
        Target sample rate (Hz).
    frame_ms : int
        Frame length in milliseconds (e.g., 25 ms).
    hop_ms : int
        Hop length in milliseconds (e.g., 10 ms).
    vad : bool
        Placeholder for Voice Activity Detection (unused here).
    """
    target_sr:int = 16000
    frame_ms:int = 25
    hop_ms:int = 10
    vad: bool = True

def load_and_standardize(path:str, cfg:PreprocConfig):
    """Load audio, resample if needed, and peak-normalize.

    Returns
    -------
    y : np.ndarray
        Mono waveform in approximately [-1, 1].
    sr : int
        Sample rate (equals cfg.target_sr).
    """
    # Load as mono at native sample rate (sr=None keeps original).
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != cfg.target_sr:
        # Resample to target sample rate.
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.target_sr)
        sr = cfg.target_sr
    # Peak normalization with small epsilon to avoid div-by-zero on silence.
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr

def frame_signal(y:np.ndarray, sr:int, cfg:PreprocConfig):
    """Frame a 1D waveform into overlapping short-time frames.

    Returns
    -------
    frames : np.ndarray
        Shape (frame_length, n_frames); often a view into y.
    n_fft : int
        Frame length in samples.
    hop : int
        Hop length in samples.
    """
    # Convert milliseconds to samples.
    n_fft = int(cfg.frame_ms*sr/1000)
    hop = int(cfg.hop_ms*sr/1000)
    # Create overlapping frames along axis 0.
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop, axis=0)
    return frames, n_fft, hop
