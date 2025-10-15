"""Simple plotting helpers for preprocessing demos.

Functions
---------
- save_waveform_with_windows: plot a waveform and overlay selected frame windows.
- save_frames_series: save images of specific framed segments.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def save_waveform_with_windows(
    y: np.ndarray,
    sr: int,
    frame_len: int,
    hop: int,
    frame_indices: Iterable[int],
    out_path: str = "waveform_with_frames.png",
    title: Optional[str] = None,
) -> str:
    """Save a plot of the waveform with vertical spans marking selected frames.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Mono waveform.
    sr : int
        Sample rate in Hz.
    frame_len : int
        Frame length in samples.
    hop : int
        Hop length in samples.
    frame_indices : Iterable[int]
        Indices of frames to highlight.
    out_path : str
        Output image path.
    title : Optional[str]
        Custom title; if None, a default with sr/frame/hop is used.

    Returns
    -------
    str
        The path where the image was saved.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    t = np.arange(len(y)) / float(sr)
    plt.figure(figsize=(10, 3))
    plt.plot(t, y, lw=0.9, color="C0")

    for idx in frame_indices:
        start = idx * hop
        end = start + frame_len
        if start >= len(y):
            continue
        end = min(end, len(y))
        t0, t1 = start / sr, end / sr
        plt.axvspan(t0, t1, color="C3", alpha=0.2)

    if title is None:
        title = f"Waveform (sr={sr} Hz) | frame={frame_len} samples, hop={hop}"
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_frames_series(
    frames: np.ndarray,
    frame_indices: Iterable[int],
    out_dir: str = "frames",
    prefix: str = "frame",
    sr: Optional[int] = None,
) -> List[str]:
    """Save individual plots for selected frames (columns of frames array).

    Parameters
    ----------
    frames : np.ndarray, shape (frame_len, n_frames)
        Framed signal where each column is one frame.
    frame_indices : Iterable[int]
        Indices of frames to save.
    out_dir : str
        Output directory to save images.
    prefix : str
        Filename prefix for saved images.
    sr : Optional[int]
        If provided, it'll appear in the figure title for context.

    Returns
    -------
    List[str]
        Paths to the saved images.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    frame_len = frames.shape[0]
    x = np.arange(frame_len)

    for idx in frame_indices:
        if idx < 0 or idx >= frames.shape[1]:
            continue
        y = frames[:, idx]

        plt.figure(figsize=(6, 2.4))
        plt.plot(x, y, lw=1.0, color="C1")
        title = f"Frame {idx} (len={frame_len})"
        if sr is not None:
            title += f" | sr={sr} Hz"
        plt.title(title)
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        path = os.path.join(out_dir, f"{prefix}_{idx:04d}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        saved.append(path)

    return saved
