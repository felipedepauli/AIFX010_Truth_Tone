import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def speaker_change_scores(mfcc_mat:np.ndarray, win:int=15):
    # distâncias entre janelas adjacentes (média de MFCCs por janela)
    X = []
    for i in range(0, len(mfcc_mat), win):
        X.append(mfcc_mat[i:i+win].mean(axis=0))
    X = np.array(X)
    d = cosine_distances(X[:-1], X[1:]).diagonal() if X.shape[0] > 1 else np.array([])
    return d  # maior = mudança
