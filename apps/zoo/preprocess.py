"""
Example: visualize waveform windows using your preprocess (hoaxhertz.preproc.base)
- Loads audio with your LoadAndStandardize
- Frames with your frame_signal
- Saves plots into artifacts/plots/
"""

import os

from hoaxhertz.preproc.base import PreprocConfig, frame_signal, LoadAndStandardize
from hoaxhertz.preproc.utils.viz import save_frames_series, save_waveform_with_windows

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="ml/research/sample_data/sample.wav")
    # opcional: permitir mudar SR/frame/hop pela linha de comando
    parser.add_argument("--sr", type=int, default=None, help="Override target sample rate")
    parser.add_argument("--frame_ms", type=int, default=None)
    parser.add_argument("--hop_ms", type=int, default=None)
    args = parser.parse_args()

    # Config padrão do seu preprocess
    cfg = PreprocConfig(
        target_sr=args.sr if args.sr is not None else 16000,  # use int, não 16e3
        frame_ms=args.frame_ms if args.frame_ms is not None else 25,
        hop_ms=args.hop_ms if args.hop_ms is not None else 10,
        vad=True,
    )

    # 1) Load + standardize (seu preprocess)
    y, sr = LoadAndStandardize(args.audio, cfg)

    # 2) Frame (seu preprocess)
    frames, n_fft, hop = frame_signal(y, sr, cfg)  # (frame_len, n_frames), n_fft, hop

    # 3) Escolher alguns instantes-alvo e convertê-los para índices de frame
    targets_s = [0.2, 0.6, 0.9]
    indices = []
    for t in targets_s:
        idx = int((t * sr) // hop)
        if 0 <= idx < frames.shape[1]:
            indices.append(idx)

    # 4) Save plots (with sr, frame and hop in the title)
    os.makedirs("artifacts/plots", exist_ok=True)
    out_wave = save_waveform_with_windows(
        y, sr, n_fft, hop, indices, out_path="artifacts/plots/waveform_with_frames.png"
    )
    print("Saved:", out_wave)

    out_frames = save_frames_series(
        frames,
        indices,
        out_dir="artifacts/plots",
        prefix="frame",
        sr=sr,  # passa sr para aparecer no título dos frames
    )
    for p in out_frames:
        print("Saved:", p)
