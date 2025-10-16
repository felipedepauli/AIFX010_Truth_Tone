# apps/zoo/sample_mfcc.py
"""
Run MFCCFeatureExtractor, save plots and a Markdown report.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from hoaxhertz.features import PreprocInput
from hoaxhertz.features.mfcc import MFCCConfig, MFCCFeatureExtractor
from hoaxhertz.preproc.base import PreprocConfig, frame_signal, LoadAndStandardize

# ---------- plotting helpers (one chart per figure, no explicit colors) -------


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_mfcc_images(out, plots_dir: str = "../plots", prefix: str = "mfcc") -> dict:
    """
    Save MFCC, Δ, (ΔΔ if present) as images. Returns dict with saved paths.
    """
    _ensure_dir(plots_dir)
    saved = {}

    fw = out.framewise or {}
    mfcc = fw.get("mfcc", None)
    delta = fw.get("delta", None)
    delta2 = fw.get("delta2", None)
    times = fw.get("times", None)

    if mfcc is not None:
        fig = plt.figure()
        ax = plt.gca()
        if times is not None:
            extent = [float(times[0]), float(times[-1]), 0, mfcc.shape[1]]
            ax.imshow(mfcc.T, aspect="auto", origin="lower", extent=extent)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Coeff idx")
        else:
            ax.imshow(mfcc.T, aspect="auto", origin="lower")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Coeff idx")
        ax.set_title("MFCC (T x D)")
        fig.tight_layout()
        p = os.path.join(plots_dir, f"{prefix}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved["mfcc"] = p

    if delta is not None:
        fig = plt.figure()
        ax = plt.gca()
        if times is not None:
            extent = [float(times[0]), float(times[-1]), 0, delta.shape[1]]
            ax.imshow(delta.T, aspect="auto", origin="lower", extent=extent)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Coeff idx")
        else:
            ax.imshow(delta.T, aspect="auto", origin="lower")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Coeff idx")
        ax.set_title("Δ MFCC (T x D)")
        fig.tight_layout()
        p = os.path.join(plots_dir, f"{prefix}_delta.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved["delta"] = p

    if delta2 is not None:
        fig = plt.figure()
        ax = plt.gca()
        if times is not None:
            extent = [float(times[0]), float(times[-1]), 0, delta2.shape[1]]
            ax.imshow(delta2.T, aspect="auto", origin="lower", extent=extent)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Coeff idx")
        else:
            ax.imshow(delta2.T, aspect="auto", origin="lower")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Coeff idx")
        ax.set_title("ΔΔ MFCC (T x D)")
        fig.tight_layout()
        p = os.path.join(plots_dir, f"{prefix}_delta2.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved["delta2"] = p

    return saved


def save_mfcc_report(
    out, audio_path: str, saved_imgs: dict, reports_dir: str = "apps/artifacts/reports"
) -> str:
    """
    Write a Markdown report with vector stats, params, and links to images.
    """
    _ensure_dir(reports_dir)
    vec = np.asarray(out.vector) if out.vector is not None else np.array([])

    lines = []
    lines.append("# MFCC Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"**Audio:** `{audio_path}`")
    lines.append("")
    lines.append("## Clip-level vector")
    lines.append(f"- shape: `{tuple(vec.shape)}`")
    if vec.size > 0:
        lines.append(f"- min/max: `{float(np.min(vec)):.4f}` / `{float(np.max(vec)):.4f}`")
        lines.append(f"- mean/std: `{float(np.mean(vec)):.4f}` / `{float(np.std(vec)):.4f}`")
        head = ", ".join(f"{x:.3f}" for x in vec[:10])
        lines.append(f"- first 10 values: {head}")
    lines.append("")
    lines.append("## Parameters")
    for k, v in (out.params or {}).items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Meta")
    for k, v in (out.meta or {}).items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Framewise visuals")
    if "mfcc" in saved_imgs:
        rel = os.path.relpath(saved_imgs["mfcc"])
        lines.append(f"![MFCC]({rel})")
    if "delta" in saved_imgs:
        rel = os.path.relpath(saved_imgs["delta"])
        lines.append(f"![Δ MFCC]({rel})")
    if "delta2" in saved_imgs:
        rel = os.path.relpath(saved_imgs["delta2"])
        lines.append(f"![ΔΔ MFCC]({rel})")
    lines.append("")

    path = os.path.join(reports_dir, "mfcc_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ------------------------------- main -----------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MFCC extraction demo with plots and report.")
    parser.add_argument("--audio", type=str, default="ml/research/sample_data/sample.wav")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=25)
    parser.add_argument("--hop-ms", type=int, default=10)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--include-delta", action="store_true", default=True)
    parser.add_argument("--include-delta2", action="store_true", default=False)
    parser.add_argument("--cmvn", action="store_true", default=False)
    parser.add_argument(
        "--robust",
        action="store_true",
        default=False,
        help="Use median+IQR aggregation.",
    )
    args = parser.parse_args()

    # 1) preprocess
    pre_cfg = PreprocConfig(target_sr=args.sr, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    y, sr = LoadAndStandardize(args.audio, pre_cfg)
    frames, n_fft, hop = frame_signal(y, sr, pre_cfg)
    pre = PreprocInput(y=y, sr=sr, frames=frames, n_fft=n_fft, hop=hop)

    # 2) configure MFCC extractor
    mfcc_cfg = MFCCConfig(
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        use_deltas=args.include_delta,
        use_delta2=args.include_delta2,
        cmvn=args.cmvn,
        use_robust=args.robust,
        return_framewise=True,
        vector_include_deltas=args.include_delta,
        vector_include_delta2=args.include_delta2,
    )
    ext = MFCCFeatureExtractor()
    ext.configure(mfcc_cfg)

    # 3) extract
    out = ext.extract(pre)
    print("clip vector shape:", out.vector.shape)
    print("framewise keys:", list(out.framewise.keys()))
    print("mfcc framewise shape:", out.framewise["mfcc"].shape)

    # 4) plots + report
    imgs = save_mfcc_images(out, plots_dir="../plots", prefix="mfcc")
    rep = save_mfcc_report(
        out, audio_path=args.audio, saved_imgs=imgs, reports_dir="apps/artifacts/reports"
    )
    print("Saved images:", imgs)
    print("Saved report:", rep)
