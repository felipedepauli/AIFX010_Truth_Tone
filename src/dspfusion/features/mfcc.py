import numpy as np, librosa

def mfcc_features(y:np.ndarray, sr:int, n_mfcc:int=20):
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return M.T  # (frames, n_mfcc)
