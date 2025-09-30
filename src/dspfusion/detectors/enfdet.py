import numpy as np

def enf_discontinuity(fi:np.ndarray, thr:float=0.4):
    if fi is None or len(fi)<3: return np.array([])
    dd = np.abs(np.diff(fi))
    return (dd > thr).astype(float)  # 1 onde há salto
