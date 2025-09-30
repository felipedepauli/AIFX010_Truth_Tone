import numpy as np

def fuse_scores(*arrays):
    sc = [a for a in arrays if a is not None and len(a)>0]
    if not sc: return None
    L = min(map(len, sc))
    sc = [a[:L] for a in sc]
    # normalização z-score por trilha
    norm = [ (a - a.mean())/(a.std()+1e-9) for a in sc ]
    return np.mean(norm, axis=0)
