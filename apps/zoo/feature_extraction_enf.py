import os

import matplotlib.pyplot as plt
import numpy as np

from hoaxhertz.features.enf import ENFConfig, ENFFeatureExtractor
from hoaxhertz.features import PreprocInput
from hoaxhertz.preproc.base import PreprocConfig, frame_signal, LoadAndStandardize

# --- Save ENF plot to artifacts/plots ---------------------------------------


def save_enf_plot(enf_out, png_path: str = "artifacts/plots/enf_curve.png") -> str:
    """
    Save ENF trajectory plot (times vs fi) to a PNG file.

    Args:
        enf_out: FeatureOutput returned by ENFFeatureExtractor (requires framewise).
        png_path: Destination PNG path.

    Returns:
        The saved file path.
    """
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    if (
        enf_out.framewise is None
        or "times" not in enf_out.framewise
        or "fi" not in enf_out.framewise
    ):
        raise ValueError("Framewise ENF data not available. Enable return_framewise in ENFConfig.")

    t = np.asarray(enf_out.framewise["times"])
    fi = np.asarray(enf_out.framewise["fi"])

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(t, fi, linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ENF (Hz)")
    nom = enf_out.meta.get("nominal", None)
    if nom is not None:
        ax.set_title(f"ENF trajectory (nominal={nom} Hz)")
    else:
        ax.set_title("ENF trajectory")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def save_enf_report(enf_out, audio_path: str, report_dir: str = "apps/artifacts/reports") -> str:
    """
    Save a simple ENF report (Markdown) with vector stats and link to the PNG curve.
    """
    os.makedirs(report_dir, exist_ok=True)
    vec = np.asarray(enf_out.vector) if enf_out.vector is not None else np.array([])
    png_rel = "../plots/enf_curve.png"  # relative path in repo
    png_abs = save_enf_plot(enf_out, png_rel)

    lines = []
    lines.append("# ENF Report\n")
    lines.append(f"**Audio:** `{audio_path}`\n")
    lines.append("## Vector (mean_dev, std_dev, min, max, slope_hz_per_min)\n")
    if vec.size == 5:
        lines.append(f"- {vec.tolist()}\n")
    else:
        lines.append("- (vector not available)\n")
    lines.append("## Curve\n")
    lines.append(f"![ENF curve]({png_rel})\n")
    lines.append("## Meta / Params\n")
    for k, v in (enf_out.meta or {}).items():
        lines.append(f"- **meta.{k}**: `{v}`")
    for k, v in (enf_out.params or {}).items():
        lines.append(f"- **param.{k}**: `{v}`")

    report_path = os.path.join(report_dir, "enf_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


# 1) preprocess
cfg_pre = PreprocConfig(target_sr=16000, frame_ms=25, hop_ms=10)
y, sr = LoadAndStandardize("ml/research/sample_data/sample.wav", cfg_pre)
frames, n_fft, hop = frame_signal(y, sr, cfg_pre)

pre = PreprocInput(y=y, sr=sr, frames=frames, n_fft=n_fft, hop=hop)

# 2) configure ENF
enf_cfg = ENFConfig(
    nominal=60.0,
    bw=1.5,
    strategy="zc",  # try "stft" for robustness
    smooth_sec=0.25,
    smooth_method="savgol",
    return_framewise=True,
)

enf = ENFFeatureExtractor()
enf.configure(enf_cfg)

# 3) extract
out = enf.extract(pre)

png_path = save_enf_plot(out, "artifacts/plots/enf_curve.png")
print("Saved plot:", png_path)

rep_path = save_enf_report(out, audio_path="ml/research/sample_data/sample.wav")
print("Saved report:", rep_path)
