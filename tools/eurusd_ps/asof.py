"""As-of join by elapsed time: value usable at t iff availability_time <= t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def asof_last(decision_t: pd.DatetimeIndex, avail_t: pd.Series, values: pd.Series,
              max_age_h: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Last value whose availability <= t, and its age in hours. NaN beyond max_age_h."""
    d = pd.DatetimeIndex(decision_t)
    order = np.argsort(avail_t.values, kind="stable")
    a = avail_t.values[order]
    v = np.asarray(values, dtype=float)[order]
    idx = np.searchsorted(a, d.values, side="right") - 1
    out = np.full(len(d), np.nan)
    age = np.full(len(d), np.nan)
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    age[ok] = (d.values[ok] - a[idx[ok]]) / np.timedelta64(1, "h")
    if max_age_h is not None:
        stale = age > max_age_h
        out[stale] = np.nan
    return out, age


def asof_count_sum(decision_t: pd.DatetimeIndex, avail_t: pd.Series, values: pd.Series | None,
                   window_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Count and sum of values with t - window_h < availability <= t."""
    d = pd.DatetimeIndex(decision_t).values
    order = np.argsort(avail_t.values, kind="stable")
    a = avail_t.values[order]
    v = np.ones(len(a)) if values is None else np.nan_to_num(np.asarray(values, dtype=float)[order])
    cs = np.concatenate([[0.0], np.cumsum(v)])
    hi = np.searchsorted(a, d, side="right")
    lo = np.searchsorted(a, d - np.timedelta64(int(window_h * 3600), "s"), side="right")
    return (hi - lo).astype(float), cs[hi] - cs[lo]


def asof_price(decision_t: pd.DatetimeIndex, bar_end: pd.DatetimeIndex, close: np.ndarray,
               offset_h: float, limit: pd.Timestamp | None = None) -> tuple[np.ndarray, np.ndarray]:
    """C_asof(t + offset): close of the last bar ending <= t + offset (offset may be negative).
    Returns NaN where t + offset >= limit (data not read) or no bar exists."""
    d = pd.DatetimeIndex(decision_t)
    q = d + pd.Timedelta(hours=offset_h)
    idx = np.searchsorted(bar_end.values, q.values, side="right") - 1
    out = np.full(len(d), np.nan)
    stale = np.full(len(d), np.nan)
    ok = idx >= 0
    if limit is not None:
        ok &= (q < limit)
    out[ok] = close[idx[ok]]
    stale[ok] = (q.values[ok] - bar_end.values[idx[ok]]) / np.timedelta64(1, "h")
    return out, stale
