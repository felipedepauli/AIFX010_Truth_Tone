from dataclasses import dataclass
import numpy as np, librosa

@dataclass
class PreprocConfig:
    target_sr:int = 16000
    frame_ms:int = 25
    hop_ms:int = 10
    vad: bool = True

def load_and_standardize(path:str, cfg:PreprocConfig):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != cfg.target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.target_sr)
        sr = cfg.target_sr
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr

def frame_signal(y:np.ndarray, sr:int, cfg:PreprocConfig):
    n_fft = int(cfg.frame_ms*sr/1000)
    hop = int(cfg.hop_ms*sr/1000)
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop, axis=0)
    return frames, n_fft, hop
