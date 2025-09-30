import numpy as np

def ambient_change(sig_list):
    # correlação entre assinaturas ambientais consecutivas
    if len(sig_list)<2: return np.array([])
    corrs=[]
    for a,b in zip(sig_list[:-1], sig_list[1:]):
        num = np.dot(a,b); den = (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
        corrs.append(num/den)
    corrs = np.array(corrs)
    scores = 1.0 - corrs
    return scores
