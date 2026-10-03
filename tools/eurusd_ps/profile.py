"""PS1 basic profile: one state per feature x metric cell.

States: MEASURED, FAILED (exception kept), NOT_APPLICABLE (reason), PENDING
(reason, e.g. dependency missing or deferred). A failure is never a zero.
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

METRICS = ["missingness", "constant", "scale_tails", "volatility", "acf", "trend", "adf", "kpss",
           "seasonality", "spectrum", "cost"]
METRICS_VERSION = "laneA_ps1_metrics.v1"
ACF_LAGS = (1, 6, 24, 120, 168)
MAX_STAT_N = 20000   # ADF/KPSS on the most recent contiguous-in-time sample of this size (declared)

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    _SM = True
except Exception:  # pragma: no cover
    _SM = False
from scipy import signal, stats


def _cell(fid, metric, state, value=None, reason=""):
    return {"feature_id": fid, "metric": metric, "state": state, "value": value, "reason": reason,
            "metrics_version": METRICS_VERSION}


def profile_feature(fid: str, x: pd.Series, build_cost_s: float = float("nan")) -> list[dict]:
    t0 = time.perf_counter()
    cells = []
    n = len(x)
    v = x.to_numpy(dtype=float)
    fin = np.isfinite(v)
    nn = int(fin.sum())
    cells.append(_cell(fid, "missingness", "MEASURED", {"n_rows": n, "n_finite": nn, "missing_fraction": float(1 - nn / n) if n else None}))
    if nn == 0:
        for m in METRICS[1:-1]:
            cells.append(_cell(fid, m, "NOT_APPLICABLE", None, "NO_FINITE_TRAIN_VALUES"))
        cells.append(_cell(fid, "cost", "MEASURED", {"build_s": build_cost_s, "profile_s": time.perf_counter() - t0, "bytes_float64": n * 8}))
        return cells
    xv = v[fin]
    uniq = np.unique(xv)
    top_share = float(pd.Series(xv).value_counts(normalize=True).iloc[0])
    is_const = len(uniq) == 1
    cells.append(_cell(fid, "constant", "MEASURED", {"n_unique": int(min(len(uniq), 10**9)), "is_constant": bool(is_const),
                                                     "top_value_share": top_share}))
    lowcard = len(uniq) < 3

    def guard(metric, fn):
        try:
            cells.append(_cell(fid, metric, "MEASURED", fn()))
        except Exception as e:  # keep failures distinct from zeros
            cells.append(_cell(fid, metric, "FAILED", None, f"{type(e).__name__}: {e}"[:300]))

    if is_const:
        for m in ("scale_tails", "volatility", "acf", "trend", "adf", "kpss", "seasonality", "spectrum"):
            cells.append(_cell(fid, m, "NOT_APPLICABLE", None, "CONSTANT_IN_TRAIN"))
    else:
        def scale():
            q = np.percentile(xv, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
            mad = float(np.median(np.abs(xv - q[4])))
            return {"mean": float(xv.mean()), "std": float(xv.std()), "min": float(xv.min()), "max": float(xv.max()),
                    "p001": q[0], "p01": q[1], "p05": q[2], "p25": q[3], "median": q[4], "p75": q[5], "p95": q[6], "p99": q[7], "p999": q[8],
                    "mad": mad, "skew": float(stats.skew(xv)), "excess_kurtosis": float(stats.kurtosis(xv)),
                    "tail_ratio_p99_mad": float((q[7] - q[4]) / mad) if mad > 0 else None,
                    "share_beyond_5mad": float(np.mean(np.abs(xv - q[4]) > 5 * mad)) if mad > 0 else None}
        guard("scale_tails", scale)

        def vol():
            d = x.diff().to_numpy(float)
            d = d[np.isfinite(d)]
            rs = x.rolling(168, min_periods=84).std()
            rs = rs[np.isfinite(rs)]
            return {"std_first_diff": float(d.std()) if len(d) else None,
                    "rolling168_std_median": float(rs.median()) if len(rs) else None,
                    "vol_of_vol_cv": float(rs.std() / rs.mean()) if len(rs) and rs.mean() > 0 else None}
        guard("volatility", vol)

        def acf():
            return {f"lag_{k}": (float(x.autocorr(k)) if nn > k + 10 else None) for k in ACF_LAGS} | \
                   {"lag_unit": "rows of the decision grid (1 row = 1 market hour; weekend gaps not filled)"}
        guard("acf", acf)

        def trend():
            idx = np.where(fin)[0]
            tt = idx / (24 * 365.25)
            sl, ic, r, p, se = stats.linregress(tt, xv)
            sub = np.linspace(0, len(xv) - 1, min(len(xv), 3000)).astype(int)
            tau, ptau = stats.kendalltau(tt[sub], xv[sub])
            return {"ols_slope_per_year": float(sl), "slope_in_std_per_year": float(sl / xv.std()), "ols_r": float(r),
                    "kendall_tau_3000": float(tau), "kendall_p_3000": float(ptau),
                    "caveat": "OLS p-value omitted: serial correlation makes it invalid"}
        guard("trend", trend)

        sample = pd.Series(xv[-MAX_STAT_N:])
        if not _SM:
            cells.append(_cell(fid, "adf", "PENDING", None, "DEPENDENCY_MISSING:statsmodels"))
            cells.append(_cell(fid, "kpss", "PENDING", None, "DEPENDENCY_MISSING:statsmodels"))
        elif lowcard:
            cells.append(_cell(fid, "adf", "NOT_APPLICABLE", None, "FEWER_THAN_3_DISTINCT_VALUES"))
            cells.append(_cell(fid, "kpss", "NOT_APPLICABLE", None, "FEWER_THAN_3_DISTINCT_VALUES"))
        else:
            def adf():
                r = adfuller(sample.values, maxlag=24, autolag=None, regression="c")
                return {"stat": float(r[0]), "pvalue": float(r[1]), "lags": int(r[2]), "nobs": int(r[3]),
                        "crit_5pct": float(r[4]["5%"]), "H0": "unit root", "regression": "c",
                        "sample": f"last {len(sample)} finite TRAIN values (gaps compressed)"}
            guard("adf", adf)

            def kp():
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    r = kpss(sample.values, regression="c", nlags="auto")
                return {"stat": float(r[0]), "pvalue": float(r[1]), "lags": int(r[2]), "H0": "level stationary",
                        "pvalue_truncated_to_table": any("p-value" in str(i.message) for i in w),
                        "sample": f"last {len(sample)} finite TRAIN values (gaps compressed)"}
            guard("kpss", kp)

        def seas():
            ix = x.index
            s = pd.Series(v, index=ix).dropna()
            mid = s.index - pd.Timedelta(minutes=30)
            out = {}
            tot = s.var()
            for nm, key in (("hour_of_day", mid.hour), ("day_of_week", mid.dayofweek), ("month", mid.month)):
                gm = s.groupby(key).transform("mean")
                out[f"eta2_{nm}"] = float(gm.var() / tot) if tot > 0 else None
            return out
        guard("seasonality", seas)

        def spec():
            z = v - np.nanmean(v)
            fill = float(np.mean(~fin))
            z = np.where(fin, z, 0.0)
            fr, p = signal.welch(z, fs=1.0, nperseg=min(2048, len(z)))
            fr, p = fr[1:], p[1:]
            top = np.argsort(p)[::-1][:3]
            pn = p / p.sum()
            ent = float(-(pn * np.log(pn + 1e-300)).sum() / np.log(len(pn)))
            return {"top_periods_rows": [float(1 / fr[i]) for i in top], "top_power_share": [float(pn[i]) for i in top],
                    "spectral_entropy_norm": ent, "zero_filled_fraction": fill, "method": "Welch nperseg<=2048 on demeaned rows"}
        guard("spectrum", spec)
    cells.append(_cell(fid, "cost", "MEASURED", {"build_s": build_cost_s, "profile_s": float(time.perf_counter() - t0),
                                                  "bytes_float64": int(n * 8)}))
    return cells


def coverage_rows(fid: str, x: pd.Series, folds: list[dict]) -> list[dict]:
    rows = []
    v = np.isfinite(x.to_numpy(float))
    for f in folds:
        for part in ("train", "val"):
            r = f[f"{part}_rows"]
            if r is None:
                rows.append({"feature_id": fid, "fold": f["name"], "part": part, "n": 0, "coverage": None, "state": "EMPTY_BLOCK"})
                continue
            seg = v[r[0]:r[1]]
            cov = float(seg.mean()) if len(seg) else None
            st = "NO_COVERAGE" if cov == 0 else ("FULL" if cov == 1 else "PARTIAL")
            rows.append({"feature_id": fid, "fold": f["name"], "part": part, "n": int(len(seg)), "coverage": cov, "state": st})
    return rows
