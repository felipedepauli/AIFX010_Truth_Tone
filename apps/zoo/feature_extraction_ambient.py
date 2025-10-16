"""
Example: run Ambient feature extractor over a preprocessed audio
and write a human-friendly Markdown report.
"""

import json
import os
from datetime import datetime

import numpy as np

from hoaxhertz.features.ambient import AmbientConfig, AmbientFeatureExtractor
from hoaxhertz.features import PreprocInput
from hoaxhertz.preproc.base import PreprocConfig, frame_signal, LoadAndStandardize


def save_ambient_report(
    out, audio_path: str, save_dir: str = "apps/artifacts/reports", top_k: int = 5
) -> str:
    """
    Build a human-friendly Markdown report from an AmbientFeatureExtractor output.
    Saves to apps/artifacts/reports/ambient_report.md by default.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ambient_report.md")

    vec = np.asarray(out.vector) if out.vector is not None else None
    params = out.params or {}
    meta = out.meta or {}
    fw = out.framewise or {}

    # Basic stats for the clip-level vector
    if vec is not None and vec.size > 0:
        vshape = tuple(vec.shape)
        vmin = float(np.min(vec))
        vmax = float(np.max(vec))
        vmean = float(np.mean(vec))
        vstd = float(np.std(vec))
        vhead = ", ".join(f"{x:.3f}" for x in vec[:top_k])
    else:
        vshape = (0,)
        vmin = vmax = vmean = vstd = float("nan")
        vhead = "—"

    # Framewise summary (if available)
    fw_logmel = fw.get("logmel", None)
    fw_sfm = fw.get("sfm", None)
    fw_times = fw.get("times", None)
    fw_summary = []
    if fw_logmel is not None:
        fw_summary.append(f"- logmel: shape={tuple(np.asarray(fw_logmel).shape)} (T×M)")
    if fw_sfm is not None:
        arr = np.asarray(fw_sfm)
        fw_summary.append(
            f"- spectral flatness: shape={tuple(arr.shape)}, mean={float(np.mean(arr)):.4f}, std={float(np.std(arr)):.4f}"
        )
    if fw_times is not None:
        arr = np.asarray(fw_times)
        dur_s = (
            float(arr[-1] - arr[0]) if arr.size > 1 else (float(arr[0]) if arr.size == 1 else 0.0)
        )
        fw_summary.append(f"- times: shape={tuple(arr.shape)}, approx span={dur_s:.3f}s")

    # Build markdown
    lines = []
    lines.append("# Ambient Feature Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"**Audio:** `{audio_path}`")
    lines.append("")
    lines.append("## Clip-level vector")
    lines.append(f"- shape: `{vshape}`")
    lines.append(f"- min/max: `{vmin:.4f}` / `{vmax:.4f}`")
    lines.append(f"- mean/std: `{vmean:.4f}` / `{vstd:.4f}`")
    lines.append(f"- first {top_k} values: {vhead}")
    lines.append("")
    lines.append("## Parameters")
    if params:
        for k, v in params.items():
            lines.append(f"- **{k}**: `{v}`")
    else:
        lines.append("- —")
    lines.append("")
    lines.append("## Meta")
    if meta:
        for k, v in meta.items():
            # compact representation for nested structures
            vv = json.dumps(v) if isinstance(v, (dict, list)) else v
            lines.append(f"- **{k}**: `{vv}`")
    else:
        lines.append("- —")
    lines.append("")
    lines.append("## Framewise (if present)")
    if fw_summary:
        lines.extend(fw_summary)
    else:
        lines.append("- —")
    lines.append("")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return save_path


# --- CLI entrypoint -----------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Run Ambient feature extractor and write a report."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="ml/research/sample_data/sample.wav",
        help="Path to the audio file to analyze.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="apps/artifacts/reports",
        help="Directory to save the Markdown report.",
    )
    # Optional knobs for preprocessing
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--frame-ms", type=int, default=25, help="Frame length (ms).")
    parser.add_argument("--hop-ms", type=int, default=10, help="Hop length (ms).")
    # Optional knobs for ambient config
    parser.add_argument("--n-mels", type=int, default=40, help="Number of mel bands.")
    parser.add_argument(
        "--robust", action="store_true", help="Use robust aggregation (median/IQR)."
    )
    parser.add_argument("--no-vad", action="store_true", help="Disable anti-VAD (use all frames).")
    parser.add_argument("--fast", action="store_true", help="Use fast_signature (mean+var only).")
    parser.add_argument(
        "--framewise", action="store_true", help="Return framewise outputs as well."
    )
    args = parser.parse_args()

    # 1) Preprocess
    cfg_pre = PreprocConfig(target_sr=args.sr, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    y, sr = LoadAndStandardize(args.audio, cfg_pre)
    frames, n_fft, hop = frame_signal(y, sr, cfg_pre)

    pre = PreprocInput(y=y, sr=sr, frames=frames, n_fft=n_fft, hop=hop)

    # 2) Configure extractor
    ambient_cfg = AmbientConfig(
        n_mels=args.n_mels,
        use_robust=bool(args.robust),
        use_vad=not bool(args.no_vad),
        return_framewise=bool(args.framewise),
        fast_signature=bool(args.fast),
    )
    extractor = AmbientFeatureExtractor()
    extractor.configure(ambient_cfg)

    # 3) Extract
    out = extractor.extract(pre)
    print(out.vector.shape)  # e.g., (82,) for n_mels=40 (2*M+2)

    # 4) Report (timestamped rename to avoid overwrite)
    os.makedirs(args.report_dir, exist_ok=True)
    report_path = save_ambient_report(
        out, audio_path=args.audio, save_dir=args.report_dir, top_k=100
    )
    ts = time.strftime("%Y%m%d-%H%M%S")
    new_path = os.path.join(args.report_dir, f"ambient_report_{ts}.md")
    os.replace(report_path, new_path)
    print("Report saved at:", new_path)
