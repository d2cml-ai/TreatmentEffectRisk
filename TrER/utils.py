import numpy as np


def wtdquantile(Y: np.array, sw: np.array, g):
    if g >= 1:
        return np.max(Y)
    else:
        o = np.argsort(Y)
        cum_w = np.cumsum(sw[o])
        threshold = np.sum(sw) * g
        idx = np.where(cum_w >= threshold)[0][0]
        return Y[o[idx]]
