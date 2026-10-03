"""Transform variants with identity and a measured causal/prefix test.

A variant is PREFIX_INVARIANT when its emitted value at t computed on the full
series equals the value computed on series[:t+1] AND replacing every value
after t with noise leaves it unchanged. Measured on real TRAIN log prices at
fixed probe points; a global (non-causal) variant must FAIL, which is the
negative control of the test itself."""
from __future__ import annotations

import numpy as np
from scipy import signal

W = 256


def _haar_modwt_last(x: np.ndarray, J: int = 5) -> np.ndarray:
    """a-trous Haar detail coefficients at the last sample, computed causally."""
    out = []
    a = x.copy()
    for j in range(J):
        s = 2 ** j
        prev = np.concatenate([np.full(s, a[0]), a[:-s]])
        na = 0.5 * (a + prev)
        out.append((a - na)[-1])
        a = na
    out.append(a[-1])
    return np.array(out)


def wavelet_modwt_haar_causal(x, t):
    seg = x[max(0, t - W + 1): t + 1]
    return _haar_modwt_last(seg - seg.mean())


def wavelet_dwt_db4_global(x, t):
    import pywt
    coeffs = pywt.wavedec(x - x.mean(), "db4", level=4)
    rec = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], "db4")[: len(x)]
    return np.array([rec[t]])


def multitaper_trailing(x, t):
    seg = x[max(0, t - W + 1): t + 1]
    seg = np.diff(seg)
    tapers = signal.windows.dpss(len(seg), NW=3, Kmax=5)
    f = np.fft.rfftfreq(len(seg), d=1.0)
    p = np.mean([np.abs(np.fft.rfft(seg * tp)) ** 2 for tp in tapers], axis=0)
    band = (f >= 1 / 48) & (f <= 1 / 6)
    return np.array([p[band].sum() / p[1:].sum()])


def hilbert_trailing_lastsample(x, t):
    seg = x[max(0, t - W + 1): t + 1]
    seg = signal.detrend(seg)
    return np.array([np.abs(signal.hilbert(seg))[-1]])


def hilbert_global(x, t):
    return np.array([np.abs(signal.hilbert(signal.detrend(x)))[t]])


def stl_trailing_lastsample(x, t):
    from statsmodels.tsa.seasonal import STL
    seg = x[max(0, t - 24 * 14 + 1): t + 1]
    r = STL(seg, period=24, robust=False).fit()
    return np.array([r.trend[-1], r.seasonal[-1]])


def stl_global(x, t):
    from statsmodels.tsa.seasonal import STL
    r = STL(x, period=24, robust=False).fit()
    return np.array([r.trend[t], r.seasonal[t]])


def _kalman(x, q, r):
    n = len(x); m = np.zeros(n); P = np.zeros(n)
    mm, pp = x[0], r
    for i in range(n):
        pp = pp + q
        k = pp / (pp + r)
        mm = mm + k * (x[i] - mm); pp = (1 - k) * pp
        m[i], P[i] = mm, pp
    return m, P


def kalman_local_level_filter(x, t, q=1e-8, r=1e-8):
    m, _ = _kalman(x[: t + 1], q, r)
    return np.array([m[-1]])


def kalman_local_level_filter_full(x, t, q=1e-8, r=1e-8):
    m, _ = _kalman(x, q, r)
    return np.array([m[t]])


def kalman_local_level_smoother(x, t, q=1e-8, r=1e-8):
    m, P = _kalman(x, q, r)
    s = m.copy()
    for i in range(len(x) - 2, -1, -1):
        Pp = P[i] + q
        s[i] = m[i] + P[i] / Pp * (s[i + 1] - m[i])
    return np.array([s[t]])


IMPLS = {
    "tv.wavelet_modwt_haar_causal": wavelet_modwt_haar_causal,
    "tv.wavelet_dwt_db4_global": wavelet_dwt_db4_global,
    "tv.multitaper_trailing": multitaper_trailing,
    "tv.hilbert_trailing_lastsample": hilbert_trailing_lastsample,
    "tv.hilbert_global": hilbert_global,
    "tv.stl_trailing_lastsample": stl_trailing_lastsample,
    "tv.stl_global": stl_global,
    "tv.kalman_local_level_filter": kalman_local_level_filter_full,
    "tv.kalman_local_level_smoother": kalman_local_level_smoother,
}


def prefix_test(fn, x: np.ndarray, probes, seed: int = 0, tol: float = 1e-10) -> dict:
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t in probes:
        full = fn(x, t)
        pre = fn(x[: t + 1], t)
        y = x.copy()
        y[t + 1:] = x[t] + np.cumsum(rng.normal(0, np.std(np.diff(x)) * 5, len(x) - t - 1))
        pert = fn(y, t)
        d = max(float(np.max(np.abs(full - pre))), float(np.max(np.abs(full - pert))))
        scale = max(1e-12, float(np.max(np.abs(full))))
        worst = max(worst, d / scale)
    return {"probes": [int(p) for p in probes], "max_rel_change": worst,
            "status": "PREFIX_INVARIANT_MEASURED" if worst <= tol else "PREFIX_VIOLATION_MEASURED"}
