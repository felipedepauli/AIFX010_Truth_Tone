import numpy as np, scipy.signal as sg

def estimate_enf(y:np.ndarray, sr:int, nominal:float=60.0, bw:float=1.5):
    # filtro passa-faixa +/- bw Hz ao redor do nominal (placeholder simples)
    sos = sg.iirfilter(6, [(nominal-bw)/(sr/2),(nominal+bw)/(sr/2)], btype='band', ftype='butter', output='sos')
    z = sg.sosfilt(sos, y)
    # estimativa rudimentar de frequência instantânea por zero-crossing
    zero = np.where(np.diff(np.signbit(z)))[0]
    if len(zero) < 2: return None
    periods = np.diff(zero)/sr
    fi = 1.0/periods
    return fi  # série de freq. inst.
