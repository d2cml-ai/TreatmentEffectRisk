import numpy as np


def wtdquantile(y: np.array, SW: np.array, g):
    Y = y.copy()
    sw = SW.copy()
    if g >= 1:
        return np.max(Y)
    o = np.argsort(Y)
    cum_w = np.cumsum(sw[o])
    cum_w = np.array(cum_w)
    threshold = np.sum(sw) * g
    idx = np.array(o[cum_w >= threshold])[0]
    return Y[idx]


def goldsectmax(f, a, b, tol=0.001, m=100):
    iter = 0
    phi = (np.sqrt(5) - 1) / 2
    a_star = b - phi * abs(b - a)
    b_star = a + phi * abs(b - a)

    while abs(b - a) > tol:
        iter += 1
        if iter > m:
            print("Warning: iterations maximum exceeded")
            break
        if f(a_star) > f(b_star):
            b = b_star
            b_star = a_star
            a_star = b - phi * abs(b - a)
        else:
            a = a_star
            a_star = b_star
            b_star = a + phi * abs(b - a)

    return (a + b) / 2


def rearrange_cvar(cvar):
    cvar_permuted = cvar.copy()

    return np.sort(cvar_permuted)


np.random.seed(0)


def make_cvgroup(n, K, right=True):
    split = np.random.rand(n)
    return np.digitize(split, np.quantile(split, np.linspace(0, 1, K + 1)), right=right)


def make_cvgroup_balanced(data, K, form_t):
    cvgroup = np.zeros(len(data), dtype=int)
    cvgroup[data[form_t] == 1] = make_cvgroup((data[form_t] == 1).sum(), K, right=True)
    cvgroup[data[form_t] == 0] = make_cvgroup((data[form_t] == 0).sum(), K, right=False)
    return cvgroup
