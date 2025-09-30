import numpy as np, librosa

def ambient_signature(y, sr, n_mels=40):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logS = librosa.power_to_db(S+1e-9)
    # estatísticas por banda (média, var)
    mu = logS.mean(axis=1)
    var = logS.var(axis=1)
    return np.concatenate([mu, var])  # vetor fixo
