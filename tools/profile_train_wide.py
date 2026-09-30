#!/usr/bin/env python3
"""M03 identity-bound TRAIN-only profiler for wide CSV resources (lake or local).

Two passes, never more:
  1. IDENTITY: stream every byte of the declared resource into SHA256 and count
     records. No value is parsed. The digest must equal the declared lake/resource
     identity or the run is refused before any statistic exists.
  2. PROFILE: read exactly the bytes of the header plus the declared TRAIN rows
     (the byte offset comes from pass 1) and parse only those. Nothing past the
     TRAIN boundary is parsed, so no holdout value can reach a statistic.

Every catalogued metric is emitted for every column as a row with a status. A
metric that could not be computed says so (NOT_RUN / FAILED / UNSUPPORTED with a
reason); it is never a missing cell. Metrics describe; they do not select.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import warnings

import numpy as np
from scipy import signal, stats

SCHEMA_IN = "feature_train_manifest.v2"
SCHEMA_OUT = "feature_train_profile.v2"
QUANTILES = (0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995)
BASE_LAGS = (1, 2, 3, 6, 12)
MISSING_TOKENS = {"", "nan", "na", "null", "none", "n/a"}
LIMITS = {"max_columns": 4096, "max_train_rows": 200_000, "max_train_cells": 25_000_000,
          "max_resource_bytes": 1 << 30, "max_pair_lag_columns": 64}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------- manifest
def validate_manifest(m: dict) -> None:
    if m.get("schema") != SCHEMA_IN or m.get("split") != "TRAIN":
        raise ValueError("An explicit feature_train_manifest.v2 TRAIN declaration is required")
    for key in ("dataset_id", "resource_sha256", "path", "split_rule", "governance", "timestamp_column"):
        if not m.get(key):
            raise ValueError(f"manifest field required: {key}")
    if m["governance"] not in ("LAKE_RESOURCE_IDENTITY", "LOCAL_FILE"):
        raise ValueError("governance must be LAKE_RESOURCE_IDENTITY or LOCAL_FILE")
    if m["governance"] == "LAKE_RESOURCE_IDENTITY" and not (m.get("lake") and m.get("resource")):
        raise ValueError("a lake resource needs lake and resource identifiers")
    b = m.get("boundaries", {}).get("train")
    if not (isinstance(b, list) and len(b) == 2 and all(type(v) is int for v in b) and b[0] == 0 < b[1]):
        raise ValueError("TRAIN must be a declared prefix [0, n) with integer bounds")
    if type(m.get("registered_rows")) is not int or m["registered_rows"] < b[1]:
        raise ValueError("registered_rows is required and must cover the TRAIN prefix")
    if m.get("default_role") not in (None, "feature", "excluded"):
        raise ValueError("default_role must be feature or excluded")
    allowed = {"feature", "target", "timestamp", "identifier", "excluded"}
    for name, spec in m.get("columns", {}).items():
        if spec.get("role") not in allowed:
            raise ValueError(f"unknown role for column {name}")
        if spec["role"] in ("excluded", "target", "identifier") and not spec.get("reason"):
            raise ValueError(f"an exclusion needs a reason: {name}")
    periods = m.get("declared_periods_rows", {})
    if any(type(v) is not int or v < 2 for v in periods.values()):
        raise ValueError("declared periods must be integers >= 2 rows")
    if periods and m.get("primary_period") not in periods:
        raise ValueError("primary_period must name one declared period")


# ----------------------------------------------------------------- pass 1
def identity_pass(path: Path, byte_cap: int, train_rows: int):
    """-> (sha256, total_bytes, data_records, train_end_offset). Values are never parsed."""
    h = hashlib.sha256()
    total = newlines = 0
    target = train_rows + 1          # header line + TRAIN rows
    end_offset = None
    last = b""
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            if total + len(chunk) > byte_cap:
                raise ValueError("resource exceeds the declared byte cap")
            h.update(chunk)
            if end_offset is None:
                pos = -1
                need = target - newlines
                for _ in range(need):
                    pos = chunk.find(b"\n", pos + 1)
                    if pos < 0:
                        break
                if pos >= 0 and chunk[:pos + 1].count(b"\n") == need:
                    end_offset = total + pos + 1
            newlines += chunk.count(b"\n")
            total += len(chunk)
            last = chunk[-1:]
    records = newlines + (1 if last not in (b"\n", b"") else 0) - 1
    return h.hexdigest(), total, records, end_offset


# ----------------------------------------------------------------- pass 2
def read_train(path: Path, end_offset: int, m: dict):
    with path.open("rb") as f:
        raw = f.read(end_offset)
    if len(raw) != end_offset:
        raise ValueError("short read of the TRAIN prefix")
    prefix_sha = sha(raw)
    lines = raw.split(b"\n")
    del raw
    if lines and lines[-1] == b"":
        lines.pop()
    reader = csv.reader((ln.decode("utf-8", errors="strict") for ln in lines), strict=True)
    header = next(reader)
    if len(header) != len(set(header)) or len(header) > LIMITS["max_columns"]:
        raise ValueError("duplicate columns or schema over the column limit")
    if m.get("columns_total") is not None and len(header) != m["columns_total"]:
        raise ValueError("header width differs from the declared resource schema")
    unknown = set(m.get("columns", {})) - set(header)
    if unknown:
        raise ValueError(f"manifest names columns absent from the header: {sorted(unknown)[:5]}")
    n = m["boundaries"]["train"][1]
    if n * len(header) > LIMITS["max_train_cells"]:
        raise ValueError("TRAIN cells exceed the declared cell cap")
    ts_col = m["timestamp_column"]
    numeric_idx = [i for i, c in enumerate(header) if c != ts_col]
    X = np.full((n, len(header)), np.nan)
    nonnumeric = np.zeros(len(header), dtype=int)
    missing = np.zeros(len(header), dtype=int)
    stamps = []
    t_index = header.index(ts_col) if ts_col in header else None
    rows = 0
    for row in reader:
        if len(row) != len(header):
            raise ValueError("malformed CSV record width")
        if t_index is not None:
            stamps.append(row[t_index])
        for i in numeric_idx:
            v = row[i]
            try:
                X[rows, i] = float(v)
            except ValueError:
                if v.strip().lower() in MISSING_TOKENS:
                    missing[i] += 1
                else:
                    nonnumeric[i] += 1
        rows += 1
    if rows != n:
        raise ValueError(f"TRAIN prefix parsed {rows} rows, declared {n}")
    return header, X, stamps, nonnumeric, missing, prefix_sha


# ----------------------------------------------------------------- metrics
def longest_run(x):
    finite = np.isfinite(x)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, 0
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


class Emitter:
    def __init__(self):
        self.rows = []

    def __call__(self, column, family, metric, value=None, status="OK", reason="", settings=None):
        if value is not None and isinstance(value, float) and not math.isfinite(value):
            status, reason, value = "FAILED", reason or "NONFINITE_VALUE", None
        self.rows.append({"column": column, "family": family, "metric": metric,
                          "value": value, "status": status, "reason": reason,
                          "settings": json.dumps(settings, sort_keys=True) if settings else ""})


def catalog(periods: dict, primary: str | None):
    """The full metric list every feature column receives, in order."""
    names = [("missingness", k) for k in ("rows", "missing_count", "missing_fraction", "nonnumeric_count",
                                          "nonfinite_count", "longest_finite_run", "constant_flag", "unique_count")]
    names += [("distribution", k) for k in ("mean", "std", "min", "max", "median", "iqr", "mad",
                                            "skewness", "excess_kurtosis", "upper_tail_ratio",
                                            "lower_tail_ratio", "robust_z_outlier_fraction_gt5",
                                            "zero_fraction")]
    names += [("distribution", f"quantile_{q}") for q in QUANTILES]
    names += [("volatility", k) for k in ("diff_std", "diff_abs_mean", "diff_std_over_std",
                                          "rolling_std_cv")]
    names += [("trend", k) for k in ("slope_per_row", "slope_per_day", "linear_r2")]
    lags = sorted(set(BASE_LAGS) | set(periods.values()))
    names += [("acf", f"acf_lag_{l}") for l in lags] + [("acf", "decorrelation_lag_1_over_e")]
    names += [("spectral", k) for k in ("peak1_period_rows", "peak1_power_fraction", "peak2_period_rows",
                                        "peak2_power_fraction", "peak3_period_rows", "peak3_power_fraction",
                                        "peak1_period_hours", "entropy_normalized",
                                        "low_frequency_power_fraction", "centroid_cycles_per_row")]
    for t in ("adf_c_aic", "kpss_c_auto", "kpss_ct_auto"):
        names += [("stationarity", f"{t}_{k}") for k in ("statistic", "pvalue", "lags")]
    names += [("stationarity", "adf_kpss_joint_reading")]
    for name, s in periods.items():
        names += [("seasonality", f"acf_at_{name}_{s}"), ("seasonality", f"seasonal_diff_var_ratio_{name}_{s}")]
    names += [("seasonality", "stl_seasonal_strength_primary"), ("seasonality", "stl_trend_strength_primary")]
    return names


def profile_column(emit, name, x, nonnumeric, missing_tokens, m, step_seconds):
    periods = m.get("declared_periods_rows", {})
    primary = m.get("primary_period")
    n = len(x)
    finite = x[np.isfinite(x)]
    nan_count = int(np.isnan(x).sum())
    constant = bool(len(finite) and np.ptp(finite) == 0)
    s0, s1 = longest_run(x)
    emit(name, "missingness", "rows", n)
    emit(name, "missingness", "missing_count", int(nan_count - nonnumeric),
         settings={"tokens": sorted(MISSING_TOKENS), "also": "NaN literals parsed by float()",
                   "empty_or_token_cells": int(missing_tokens)})
    emit(name, "missingness", "missing_fraction", (nan_count - nonnumeric) / n)
    emit(name, "missingness", "nonnumeric_count", int(nonnumeric))
    emit(name, "missingness", "nonfinite_count", int(n - len(finite)))
    emit(name, "missingness", "longest_finite_run", s1 - s0)
    emit(name, "missingness", "constant_flag", constant)
    emit(name, "missingness", "unique_count", int(len(np.unique(finite))))
    fam = catalog(periods, primary)
    if nonnumeric or not len(finite) or constant:
        reason = ("NONNUMERIC_VALUES" if nonnumeric else "NO_FINITE_VALUES" if not len(finite)
                  else "CONSTANT_IN_TRAIN")
        for family, metric in fam[8:]:
            emit(name, family, metric, status="NOT_RUN", reason=reason)
        return reason
    # distribution / scale / tails
    q = dict(zip(QUANTILES, np.quantile(finite, QUANTILES)))
    med = q[0.5]
    mad = float(np.median(np.abs(finite - med)))
    for k, v in (("mean", finite.mean()), ("std", finite.std(ddof=1) if len(finite) > 1 else math.nan),
                 ("min", finite.min()), ("max", finite.max()), ("median", med),
                 ("iqr", q[0.75] - q[0.25]), ("mad", mad)):
        emit(name, "distribution", k, float(v))
    for k, fn in (("skewness", stats.skew), ("excess_kurtosis", stats.kurtosis)):
        emit(name, "distribution", k, float(fn(finite, bias=False)) if len(finite) > 3 else None,
             status="OK" if len(finite) > 3 else "NOT_RUN", reason="" if len(finite) > 3 else "INSUFFICIENT_SAMPLE",
             settings={"bias": False})
    up, lo = q[0.75] - med, med - q[0.25]
    emit(name, "distribution", "upper_tail_ratio", float((q[0.995] - med) / up) if up > 0 else None,
         status="OK" if up > 0 else "NOT_RUN", reason="" if up > 0 else "ZERO_UPPER_QUARTILE_SPREAD",
         settings={"formula": "(q0.995-q0.5)/(q0.75-q0.5); Gaussian ~3.82"})
    emit(name, "distribution", "lower_tail_ratio", float((med - q[0.005]) / lo) if lo > 0 else None,
         status="OK" if lo > 0 else "NOT_RUN", reason="" if lo > 0 else "ZERO_LOWER_QUARTILE_SPREAD",
         settings={"formula": "(q0.5-q0.005)/(q0.5-q0.25); Gaussian ~3.82"})
    if mad > 0:
        emit(name, "distribution", "robust_z_outlier_fraction_gt5",
             float(np.mean(np.abs(finite - med) / (1.4826 * mad) > 5)), settings={"scale": "1.4826*MAD"})
    else:
        emit(name, "distribution", "robust_z_outlier_fraction_gt5", status="NOT_RUN", reason="ZERO_MAD")
    emit(name, "distribution", "zero_fraction", float(np.mean(finite == 0)))
    for qq in QUANTILES:
        emit(name, "distribution", f"quantile_{qq}", float(q[qq]), settings={"method": "numpy linear"})
    # temporal diagnostics on the longest contiguous finite run, no imputation
    y = x[s0:s1]
    seg = {"segment_rows": [s0, s1], "rule": "longest contiguous finite run; no imputation"}
    d = np.diff(y)
    sd = float(y.std(ddof=1))
    emit(name, "volatility", "diff_std", float(d.std(ddof=1)) if len(d) > 1 else None,
         status="OK" if len(d) > 1 else "NOT_RUN", reason="" if len(d) > 1 else "INSUFFICIENT_SAMPLE", settings=seg)
    emit(name, "volatility", "diff_abs_mean", float(np.abs(d).mean()) if len(d) else None,
         status="OK" if len(d) else "NOT_RUN", reason="" if len(d) else "INSUFFICIENT_SAMPLE", settings=seg)
    emit(name, "volatility", "diff_std_over_std", float(d.std(ddof=1) / sd) if len(d) > 1 and sd > 0 else None,
         status="OK" if len(d) > 1 and sd > 0 else "NOT_RUN",
         reason="" if len(d) > 1 and sd > 0 else "INSUFFICIENT_SAMPLE", settings=seg)
    w = periods.get(primary) if primary else None
    if w and len(y) >= 4 * w:
        blocks = y[: len(y) // w * w].reshape(-1, w).std(axis=1, ddof=1)
        ok = blocks.mean() > 0
        emit(name, "volatility", "rolling_std_cv", float(blocks.std(ddof=1) / blocks.mean()) if ok else None,
             status="OK" if ok else "NOT_RUN", reason="" if ok else "ZERO_BLOCK_STD",
             settings={"window_rows": w, "rule": "non-overlapping primary-period blocks; std of block std / mean"})
    else:
        emit(name, "volatility", "rolling_std_cv", status="NOT_RUN",
             reason="NO_DECLARED_PRIMARY_PERIOD" if not w else "SEGMENT_SHORTER_THAN_4_PERIODS")
    if len(y) < 8:
        for family, metric in fam:
            if family in ("trend", "acf", "spectral", "stationarity", "seasonality"):
                emit(name, family, metric, status="NOT_RUN", reason="INSUFFICIENT_SAMPLE")
        return None
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    centered = y - y.mean()
    energy = float(centered @ centered)
    emit(name, "trend", "slope_per_row", float(slope), settings=seg)
    if step_seconds:
        emit(name, "trend", "slope_per_day", float(slope * 86400 / step_seconds),
             settings={"step_seconds": step_seconds, "basis": "declared regular step; see sampling"})
    else:
        emit(name, "trend", "slope_per_day", status="UNSUPPORTED", reason="NO_VERIFIED_REGULAR_STEP")
    emit(name, "trend", "linear_r2", float(1 - (resid @ resid) / energy) if energy > 0 else None,
         status="OK" if energy > 0 else "NOT_RUN", reason="" if energy > 0 else "ZERO_ENERGY")
    # ACF by declared lag
    lags = sorted(set(BASE_LAGS) | set(periods.values()))
    nlag = min(max(lags + [64]), len(y) - 1)
    acf = signal.correlate(centered, centered, mode="full", method="fft")[len(y) - 1:len(y) + nlag] / energy
    for l in lags:
        if l <= nlag:
            emit(name, "acf", f"acf_lag_{l}", float(acf[l]), settings={"estimator": "biased, mean-centered"})
        else:
            emit(name, "acf", f"acf_lag_{l}", status="NOT_RUN", reason="LAG_EXCEEDS_SEGMENT")
    below = np.flatnonzero(acf[1:] < 1 / math.e)
    if len(below):
        emit(name, "acf", "decorrelation_lag_1_over_e", int(below[0] + 1))
    else:
        emit(name, "acf", "decorrelation_lag_1_over_e", status="NOT_RUN", reason=f"ACF_ABOVE_1_OVER_E_THROUGH_LAG_{nlag}")
    # spectrum
    f, p = signal.periodogram(y, fs=1.0, detrend="linear")
    f, p = f[1:], p[1:]
    total = float(p.sum())
    if total > np.finfo(float).eps * max(energy, 1.0):
        p = p / total
        order = np.argsort(p)[::-1]
        for k in range(3):
            i = order[k]
            emit(name, "spectral", f"peak{k + 1}_period_rows", float(1 / f[i]),
                 settings={"method": "periodogram, linear detrend, rank by power"})
            emit(name, "spectral", f"peak{k + 1}_power_fraction", float(p[i]))
        if step_seconds:
            emit(name, "spectral", "peak1_period_hours", float(step_seconds / f[order[0]] / 3600))
        else:
            emit(name, "spectral", "peak1_period_hours", status="UNSUPPORTED", reason="NO_VERIFIED_REGULAR_STEP")
        pos = p[p > 0]
        emit(name, "spectral", "entropy_normalized", float(-(pos * np.log(pos)).sum() / np.log(len(p))))
        cut = 1.0 / w if w else 0.1
        emit(name, "spectral", "low_frequency_power_fraction", float(p[f < cut].sum()),
             settings={"cutoff_cycles_per_row": cut, "basis": "primary period" if w else "0.1 cycles/row"})
        emit(name, "spectral", "centroid_cycles_per_row", float(f @ p))
    else:
        for metric in ("peak1_period_rows", "peak1_power_fraction", "peak2_period_rows", "peak2_power_fraction",
                       "peak3_period_rows", "peak3_power_fraction", "peak1_period_hours", "entropy_normalized",
                       "low_frequency_power_fraction", "centroid_cycles_per_row"):
            emit(name, "spectral", metric, status="NOT_RUN", reason="NEGLIGIBLE_DETRENDED_ENERGY")
    stationarity(emit, name, y)
    seasonality(emit, name, y, periods, primary)
    return None


def stationarity(emit, name, y):
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        for t in ("adf_c_aic", "kpss_c_auto", "kpss_ct_auto"):
            for k in ("statistic", "pvalue", "lags"):
                emit(name, "stationarity", f"{t}_{k}", status="UNSUPPORTED", reason="statsmodels not installed")
        emit(name, "stationarity", "adf_kpss_joint_reading", status="UNSUPPORTED", reason="statsmodels not installed")
        return
    results = {}
    specs = (("adf_c_aic", {"test": "ADF", "null": "unit root", "regression": "c", "autolag": "AIC",
                            "maxlag": "statsmodels default 12*(nobs/100)^(1/4)",
                            "limitation": "MacKinnon approximate p-value; low power near unity; lag choice by AIC"}),
             ("kpss_c_auto", {"test": "KPSS", "null": "level stationarity", "regression": "c", "nlags": "auto (Hobijn)",
                              "limitation": "p-value interpolated from a table bounded to [0.01, 0.10]"}),
             ("kpss_ct_auto", {"test": "KPSS", "null": "trend stationarity", "regression": "ct", "nlags": "auto (Hobijn)",
                               "limitation": "p-value interpolated from a table bounded to [0.01, 0.10]"}))
    for key, settings in specs:
        settings = dict(settings, n=len(y))
        if len(y) < 50:
            for k in ("statistic", "pvalue", "lags"):
                emit(name, "stationarity", f"{key}_{k}", status="NOT_RUN", reason="INSUFFICIENT_SAMPLE", settings=settings)
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if key.startswith("adf"):
                    a = adfuller(y, regression="c", autolag="AIC")
                    stat, pv, lags = a[0], a[1], a[2]
                else:
                    a = kpss(y, regression="c" if key == "kpss_c_auto" else "ct", nlags="auto")
                    stat, pv, lags = a[0], a[1], a[2]
            msgs = [str(w.message) for w in caught]
            status = "OK_WITH_WARNING" if msgs else "OK"
            s = dict(settings, warnings=msgs) if msgs else settings
            if not (math.isfinite(stat) and math.isfinite(pv)):
                raise FloatingPointError("nonfinite estimator result")
            results[key] = (float(stat), float(pv), bool(msgs))
            emit(name, "stationarity", f"{key}_statistic", float(stat), status=status, settings=s)
            emit(name, "stationarity", f"{key}_pvalue", float(pv), status=status,
                 reason="P_VALUE_AT_TABLE_BOUND" if msgs and key.startswith("kpss") else "", settings=s)
            emit(name, "stationarity", f"{key}_lags", int(lags), status=status, settings=s)
        except Exception as exc:
            for k in ("statistic", "pvalue", "lags"):
                emit(name, "stationarity", f"{key}_{k}", status="FAILED", reason=f"{type(exc).__name__}: {exc}"[:200],
                     settings=settings)
    if "adf_c_aic" in results and "kpss_c_auto" in results:
        adf_rej = results["adf_c_aic"][1] < 0.05
        kpss_rej = results["kpss_c_auto"][1] < 0.05
        reading = {(True, False): "ADF_REJECTS_UNIT_ROOT_KPSS_DOES_NOT_REJECT_LEVEL",
                   (False, True): "ADF_DOES_NOT_REJECT_KPSS_REJECTS_LEVEL",
                   (True, True): "BOTH_REJECT_CONFLICT", (False, False): "NEITHER_REJECTS_LOW_POWER"}[(adf_rej, kpss_rej)]
        emit(name, "stationarity", "adf_kpss_joint_reading", reading,
             settings={"alpha": 0.05, "note": "descriptive, unadjusted for multiple testing; not an eligibility rule"})
    else:
        emit(name, "stationarity", "adf_kpss_joint_reading", status="NOT_RUN", reason="A_COMPONENT_TEST_DID_NOT_COMPLETE")


def seasonality(emit, name, y, periods, primary):
    var = float(y.var())
    for pname, s in periods.items():
        if len(y) > 2 * s and var > 0:
            c = y - y.mean()
            emit(name, "seasonality", f"acf_at_{pname}_{s}", float((c[s:] @ c[:-s]) / (c @ c)),
                 settings={"declared_period_rows": s})
            emit(name, "seasonality", f"seasonal_diff_var_ratio_{pname}_{s}", float(np.var(y[s:] - y[:-s]) / var),
                 settings={"formula": "var(x_t - x_{t-s}) / var(x_t); 2 for white noise, <1 when season repeats"})
        else:
            for metric in (f"acf_at_{pname}_{s}", f"seasonal_diff_var_ratio_{pname}_{s}"):
                emit(name, "seasonality", metric, status="NOT_RUN",
                     reason="SEGMENT_SHORTER_THAN_2_PERIODS" if var > 0 else "ZERO_VARIANCE")
    if not primary:
        for metric in ("stl_seasonal_strength_primary", "stl_trend_strength_primary"):
            emit(name, "seasonality", metric, status="NOT_RUN", reason="NO_DECLARED_PRIMARY_PERIOD")
        return
    s = periods[primary]
    settings = {"method": "statsmodels STL", "period": s, "robust": False,
                "formula": "max(0, 1 - var(R)/var(S+R)) and max(0, 1 - var(R)/var(T+R)) (Wang, Smith, Hyndman 2006)",
                "note": "optional diagnostic transform; not a model input"}
    if len(y) < 2 * s + 1:
        for metric in ("stl_seasonal_strength_primary", "stl_trend_strength_primary"):
            emit(name, "seasonality", metric, status="NOT_RUN", reason="SEGMENT_SHORTER_THAN_2_PERIODS", settings=settings)
        return
    try:
        from statsmodels.tsa.seasonal import STL
        r = STL(y, period=s, robust=False).fit()
        R, S, T = r.resid, r.seasonal, r.trend
        vs, vt = np.var(S + R), np.var(T + R)
        emit(name, "seasonality", "stl_seasonal_strength_primary", float(max(0.0, 1 - np.var(R) / vs)) if vs > 0 else None,
             status="OK" if vs > 0 else "NOT_RUN", reason="" if vs > 0 else "ZERO_VARIANCE", settings=settings)
        emit(name, "seasonality", "stl_trend_strength_primary", float(max(0.0, 1 - np.var(R) / vt)) if vt > 0 else None,
             status="OK" if vt > 0 else "NOT_RUN", reason="" if vt > 0 else "ZERO_VARIANCE", settings=settings)
    except ImportError:
        for metric in ("stl_seasonal_strength_primary", "stl_trend_strength_primary"):
            emit(name, "seasonality", metric, status="UNSUPPORTED", reason="statsmodels not installed", settings=settings)
    except Exception as exc:
        for metric in ("stl_seasonal_strength_primary", "stl_trend_strength_primary"):
            emit(name, "seasonality", metric, status="FAILED", reason=f"{type(exc).__name__}: {exc}"[:200], settings=settings)


def sampling(stamps, fmt, declared_step):
    import pandas as pd
    if not stamps:
        return {"status": "NO_TIMESTAMP_COLUMN", "verified_step_seconds": None}
    ts = pd.to_datetime(pd.Series(stamps), format=fmt, errors="coerce")
    dt = ts.diff().dt.total_seconds().iloc[1:]
    steps = dt.value_counts().head(5)
    regular = bool(ts.notna().all() and len(dt) and (dt > 0).all() and dt.nunique() == 1)
    out = {"status": "REGULAR" if regular else "IRREGULAR_OR_INVALID", "invalid_count": int(ts.isna().sum()),
           "first": str(ts.iloc[0]), "last": str(ts.iloc[-1]), "median_step_seconds": float(dt.median()),
           "top_steps_seconds": {str(k): int(v) for k, v in steps.items()},
           "non_positive_steps": int((dt <= 0).sum()), "declared_step_seconds": declared_step}
    out["verified_step_seconds"] = declared_step if regular and float(dt.iloc[0]) == declared_step else None
    if not regular:
        out["consequence"] = "physical-time units (per-day slope, period hours) are UNSUPPORTED; row units only"
    return out


def redundancy(X, names, max_lag_cols, lag_max):
    """TRAIN-only Pearson redundancy (all pairs) and, when small enough, lagged cross-correlation."""
    out = {"rule": "Pearson on TRAIN rows with every selected column finite; descriptive, never merges",
           "columns": len(names)}
    if len(names) < 2:
        out.update(status="NOT_RUN", reason="FEWER_THAN_TWO_COLUMNS")
        return out
    ok = np.isfinite(X).all(axis=1)
    Z = X[ok]
    out["rows_used"] = int(ok.sum())
    C = np.corrcoef(Z, rowvar=False)
    iu = np.triu_indices(len(names), 1)
    a = np.abs(C[iu])
    out.update(status="OK", pairs=int(len(a)), pairs_abs_ge_0_95=int((a >= 0.95).sum()),
               pairs_abs_ge_0_99=int((a >= 0.99).sum()), abs_corr_quantiles={str(q): float(np.quantile(a, q)) for q in (0.5, 0.9, 0.99)})
    for thr in (0.95, 0.99):
        parent = list(range(len(names)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in zip(*iu):
            if abs(C[i, j]) >= thr:
                parent[find(i)] = find(j)
        groups = {}
        for i in range(len(names)):
            groups.setdefault(find(i), []).append(names[i])
        multi = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
        out[f"components_abs_ge_{thr}"] = {"count_multi_member": len(multi), "largest": len(multi[0]) if multi else 1,
                                             "members": multi[:50]}
    top = []
    for i, j in zip(*iu):
        if abs(C[i, j]) >= 0.95:
            top.append([names[i], names[j], float(C[i, j])])
    out["redundant_pairs_abs_ge_0_95"] = sorted(top, key=lambda r: -abs(r[2]))[:500]
    if len(names) <= max_lag_cols:
        Zc = (Z - Z.mean(0)) / Z.std(0)
        lagged = []
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                best = (0, 0.0)
                for lag in range(1, lag_max + 1):
                    r = float(np.mean(Zc[lag:, i] * Zc[:-lag, j]))
                    if abs(r) > abs(best[1]):
                        best = (lag, r)
                lagged.append([names[j], names[i], best[0], best[1]])
        out["lagged"] = {"status": "OK", "rule": f"max |corr(x_i(t), x_j(t-lag))| over lags 1..{lag_max}; leader first",
                         "pairs": sorted(lagged, key=lambda r: -abs(r[3]))[:400]}
    else:
        out["lagged"] = {"status": "NOT_RUN", "reason": f"COLUMNS_OVER_LAG_SCAN_CAP_{max_lag_cols}",
                         "consequence": "lag-informed grouping must be fitted inside inner folds (protocol step G3)"}
    return out


# ----------------------------------------------------------------- run
def versions():
    v = {"python": platform.python_version()}
    for dep in ("numpy", "scipy", "pandas", "statsmodels"):
        try:
            v[dep] = importlib.metadata.version(dep)
        except importlib.metadata.PackageNotFoundError:
            v[dep] = "NOT_INSTALLED"
    return v


def clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


def run(manifest_path, source_root, output, byte_cap=LIMITS["max_resource_bytes"]):
    started = time.monotonic()
    mbytes = Path(manifest_path).read_bytes()
    m = json.loads(mbytes)
    validate_manifest(m)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output must be a new directory")
    root = Path(source_root).resolve()
    path = (root / m["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("resource must be a regular file beneath source-root")
    n_train = m["boundaries"]["train"][1]
    if n_train > LIMITS["max_train_rows"]:
        raise ValueError("TRAIN rows exceed the declared row cap")
    digest, total_bytes, records, end_offset = identity_pass(path, byte_cap, n_train)
    if digest != m["resource_sha256"]:
        raise ValueError(f"resource identity mismatch: {digest} != declared {m['resource_sha256']}")
    if records != m["registered_rows"]:
        raise ValueError(f"record count {records} differs from registered {m['registered_rows']}")
    if end_offset is None:
        raise ValueError("TRAIN boundary not found in the resource")
    header, X, stamps, nonnumeric, missing, prefix_sha = read_train(path, end_offset, m)
    samp = sampling(stamps, m.get("timestamp_format"), m.get("step_seconds"))
    step = samp.get("verified_step_seconds")
    roles = {}
    for c in header:
        spec = m.get("columns", {}).get(c)
        if c == m["timestamp_column"]:
            roles[c] = {"role": "timestamp", "reason": "time index; never a branch input"}
        elif spec:
            roles[c] = spec
        else:
            roles[c] = {"role": m.get("default_role") or "excluded",
                        "reason": "" if m.get("default_role") == "feature" else "UNDECLARED_ROLE"}
    emit = Emitter()
    columns = []
    admitted = []
    for i, c in enumerate(header):
        spec = roles[c]
        entry = {"column": c, "position": i, "role": spec["role"], "declared_reason": spec.get("reason", ""),
                 "target_channel": c in m.get("target_channels", [])}
        if spec["role"] != "feature":
            entry.update(status="EXCLUDED", exclusion_reason=spec.get("reason") or spec["role"].upper())
            columns.append(entry)
            continue
        reason = profile_column(emit, c, X[:, i], int(nonnumeric[i]), int(missing[i]), m, step)
        miss = float(np.mean(~np.isfinite(X[:, i])))
        cap = m.get("max_missing_fraction", 0.2)
        if reason:
            entry.update(status="PROFILED_EXCLUDED", exclusion_reason=reason)
        elif miss > cap:
            entry.update(status="PROFILED_EXCLUDED", exclusion_reason=f"EXCESS_MISSING_GT_{cap}")
        else:
            entry.update(status="PROFILED_ADMISSIBLE", exclusion_reason="")
            admitted.append(i)
        columns.append(entry)
    red = redundancy(X[:, admitted], [header[i] for i in admitted], LIMITS["max_pair_lag_columns"],
                     max(m.get("declared_periods_rows", {}).values(), default=24) if len(admitted) <= 64 else 0)
    code_sha = sha(Path(__file__).read_bytes())
    report = {
        "schema": SCHEMA_OUT, "dataset_id": m["dataset_id"], "split": "TRAIN", "governance": m["governance"],
        "lake": m.get("lake"), "resource": m.get("resource"), "manifest": m, "manifest_sha256": sha(mbytes),
        "resource_identity": {"sha256": digest, "bytes": total_bytes, "records": records,
                              "verified_against_declared": True,
                              "note": "every byte hashed for identity; no value past the TRAIN boundary parsed"},
        "train_prefix": {"rows": [0, n_train], "bytes_parsed": end_offset, "prefix_sha256": prefix_sha,
                         "split_rule": m["split_rule"], "full_train_coverage": True},
        "sampling": samp, "columns": columns,
        "branches": [{"branch": k, "columns": [header[i]]} for k, i in enumerate(admitted)],
        "grouping": "one admissible feature per branch; metrics propose, inner validation decides",
        "redundancy": red, "implementation": {"tool": "tools/profile_train_wide.py", "sha256": code_sha,
                                             "versions": versions(), "limits": LIMITS},
        "assumptions": ["No imputation; temporal metrics use the longest contiguous finite run",
                        "Values are profiled as stored (no scaler); TSL loaders standardize with a TRAIN-fitted scaler later",
                        "ADF/KPSS p-values are descriptive and unadjusted for multiple testing",
                        "Point-in-time availability is not certified by this profile",
                        "No target, no model and no holdout value enters any statistic"],
        "wall_seconds": None}
    output.mkdir(parents=True)
    report["wall_seconds"] = time.monotonic() - started
    report = clean(report)
    (output / "profile.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    with (output / "metrics_long.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["column", "family", "metric", "value", "status", "reason", "settings"],
                           lineterminator="\n")
        w.writeheader()
        for r in emit.rows:
            w.writerow(clean(r))
    with (output / "columns.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["column", "position", "role", "target_channel", "status",
                                           "exclusion_reason", "declared_reason"], lineterminator="\n")
        w.writeheader()
        for c in columns:
            w.writerow({k: c.get(k, "") for k in w.fieldnames})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-bytes", type=int, default=LIMITS["max_resource_bytes"])
    a = p.parse_args()
    r = run(a.manifest, a.source_root, a.output, a.max_bytes)
    print(json.dumps({"dataset_id": r["dataset_id"], "columns": len(r["columns"]), "branches": len(r["branches"]),
                      "resource_sha256": r["resource_identity"]["sha256"], "wall_seconds": r["wall_seconds"]}))


if __name__ == "__main__":
    main()
