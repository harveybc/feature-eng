"""Train-only descriptive metrics for the 83 input features of the ETH 4h data contract.

Each feature gets one typed row of *descriptive* statistics fitted on the TRAIN rows only:
availability and missingness, variance, ADF / KPSS status (when ``statsmodels`` is installed),
autocorrelation at lags 1/6/24, normalised spectral entropy, a dominant trailing period
estimate and the association with the forward log-return of CLOSE at horizons 6/12/18/24/30/36.

What this module does not do: it never selects or drops a feature, and a number here is an
observation about the train rows, not a causal or predictive claim.

Isolation guarantee: the train boundary is found by reading only the DATE_TIME column, chunk by
chunk, and stopping at the first date after ``train_end``. The feature file is then read with
``nrows`` equal to the train row count, so validation and test rows are never parsed. The
protected ETH test rows [15895, 18085) are refused explicitly if the train boundary would ever
reach them. The target at horizon h uses only rows inside the train slice (t + h <= last train
row), so no train row's target looks into validation.

Persisted per row: row_id, fit_scope, data_digest, parameter_digest. The digests cover the train
slice and the parameters only; the output carries no timestamps, so the same train rows and
parameters give byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings

import numpy as np
import pandas as pd

SCHEMA = "feature_eng.train_feature_metrics.v1"
FIT_SCOPE = "train_only"
PROTECTED_TEST_ROWS = (15895, 18085)  # half-open; never read
DEFAULT_PARAMS = {
    "acf_lags": [1, 6, 24],
    "target_horizons": [6, 12, 18, 24, 30, 36],
    "target_column": "CLOSE",
    "target_definition": "log(close[t+h]/close[t]), t+h inside the train slice",
    "adf_maxlag": 12,
    "adf_regression": "c",
    "kpss_regression": "c",
    "kpss_nlags": "auto",
    "trailing_window": 1024,
    "min_pairs": 30,
    "date_column": "DATE_TIME",
}

try:  # optional dependency, status is reported either way
    from statsmodels.tsa.stattools import adfuller, kpss
    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    HAVE_STATSMODELS = False


class ScopeError(ValueError):
    """The requested fit would read rows that are not train rows."""


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _num(x):
    """JSON-safe float: non-finite becomes None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------------------- loading

def load_contract(manifest_path: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    feats = list(m["feature_columns"])
    if len(set(feats)) != len(feats):
        raise ScopeError("manifest feature_columns contains duplicates")
    return {"features": feats, "train_start": m["splits"]["train_start"],
            "train_end": m["splits"]["train_end"], "manifest_rows": m.get("rows"),
            "manifest_sha256": m.get("sha256")}


def train_row_count(csv_path: str, train_end: str, date_column: str, chunk: int = 2000) -> int:
    """Number of leading rows with date <= train_end; reads dates only and stops early."""
    end = pd.Timestamp(train_end)
    n = 0
    prev = None
    for part in pd.read_csv(csv_path, usecols=[date_column], chunksize=chunk):
        d = pd.to_datetime(part[date_column])
        if not d.is_monotonic_increasing or (prev is not None and d.iloc[0] <= prev):
            raise ScopeError("dates are not strictly increasing; train boundary is ambiguous")
        inside = int((d <= end).sum())
        n += inside
        if inside < len(d):
            break
        prev = d.iloc[-1]
    return n


def load_train_frame(csv_path: str, contract: dict, params: dict) -> pd.DataFrame:
    dc = params["date_column"]
    n = train_row_count(csv_path, contract["train_end"], dc)
    if n <= 0:
        raise ScopeError("no train rows found")
    if n > PROTECTED_TEST_ROWS[0]:
        raise ScopeError(f"train boundary {n} reaches the protected rows "
                         f"[{PROTECTED_TEST_ROWS[0]}, {PROTECTED_TEST_ROWS[1]})")
    cols = [dc, params["target_column"]] + contract["features"]
    wanted = set(cols)
    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted, nrows=n)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ScopeError(f"file lacks declared columns: {missing}")
    return df[cols]


# ----------------------------------------------------------------------------- statistics

def _longest_finite_run(x: np.ndarray) -> np.ndarray:
    ok = np.append(np.isfinite(x), False)
    best = (0, 0)
    start = None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return x[best[0]:best[1]]


def _longest_missing_run(x: np.ndarray) -> int:
    run = best = 0
    for v in ~np.isfinite(x):
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _acf(x: np.ndarray, lag: int, min_pairs: int):
    a, b = x[lag:], x[:-lag]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_pairs:
        return None
    a, b = a[ok], b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return _num(np.corrcoef(a, b)[0, 1])


def _periodogram(seg: np.ndarray):
    seg = seg - seg.mean()
    psd = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(len(seg), d=1.0)
    return freqs[1:], psd[1:]


def _spectral_entropy(seg: np.ndarray, min_pairs: int):
    if len(seg) < max(min_pairs, 8) or np.std(seg) == 0:
        return None
    _, psd = _periodogram(seg)
    tot = psd.sum()
    if tot <= 0 or len(psd) < 2:
        return None
    p = psd / tot
    p = p[p > 0]
    return _num(-(p * np.log2(p)).sum() / math.log2(len(psd)))


def _dominant_period(seg: np.ndarray, window: int, min_pairs: int):
    """Period in bars of the strongest non-DC periodogram peak of the trailing window."""
    seg = seg[-window:]
    if len(seg) < max(min_pairs, 8) or np.std(seg) == 0:
        return None, len(seg)
    freqs, psd = _periodogram(seg)
    return _num(1.0 / freqs[int(np.argmax(psd))]), len(seg)


def _stationarity(x: np.ndarray, p: dict) -> dict:
    out = {"adf_status": "", "adf_stat": None, "adf_pvalue": None,
           "kpss_status": "", "kpss_stat": None, "kpss_pvalue": None}
    seg = _longest_finite_run(x)
    if not HAVE_STATSMODELS:
        out["adf_status"] = out["kpss_status"] = "dependency_missing"
        return out
    if len(seg) < max(p["min_pairs"], 3 * p["adf_maxlag"]) or np.std(seg) == 0:
        out["adf_status"] = out["kpss_status"] = "skipped_constant_or_short"
        return out
    try:
        r = adfuller(seg, maxlag=p["adf_maxlag"], regression=p["adf_regression"], autolag=None)
        out.update(adf_status="ok", adf_stat=_num(r[0]), adf_pvalue=_num(r[1]))
    except Exception as e:  # report, do not hide
        out["adf_status"] = f"error:{type(e).__name__}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = kpss(seg, regression=p["kpss_regression"], nlags=p["kpss_nlags"])
        out.update(kpss_status="ok", kpss_stat=_num(r[0]), kpss_pvalue=_num(r[1]))
    except Exception as e:
        out["kpss_status"] = f"error:{type(e).__name__}"
    return out


def _target_association(x: np.ndarray, close: np.ndarray, p: dict) -> dict:
    from scipy.stats import spearmanr
    out = {}
    n = len(x)
    for h in p["target_horizons"]:
        pe = sp = None
        npairs = 0
        if n > h:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log(close[h:] / close[:-h])
            xs = x[:-h]
            ok = np.isfinite(xs) & np.isfinite(y)
            npairs = int(ok.sum())
            if npairs >= p["min_pairs"] and np.std(xs[ok]) > 0 and np.std(y[ok]) > 0:
                pe = _num(np.corrcoef(xs[ok], y[ok])[0, 1])
                sp = _num(spearmanr(xs[ok], y[ok])[0])
        out[f"target_pearson_h{h}"] = pe
        out[f"target_spearman_h{h}"] = sp
        out[f"target_pairs_h{h}"] = npairs
    return out


def feature_row(name: str, x: np.ndarray, close: np.ndarray, p: dict) -> dict:
    n = len(x)
    fin = np.isfinite(x)
    nf = int(fin.sum())
    xf = x[fin]
    first_ok = int(np.argmax(fin)) if nf else n
    row = {
        "feature": name,
        "available": bool(nf > 0),
        "n_rows": n,
        "n_finite": nf,
        "n_missing": n - nf,
        "missing_fraction": _num((n - nf) / n) if n else None,
        "leading_missing": first_ok,
        "longest_missing_run": _longest_missing_run(x),
        "n_unique": int(len(np.unique(xf))) if nf else 0,
        "mean": _num(xf.mean()) if nf else None,
        "variance": _num(xf.var(ddof=1)) if nf > 1 else None,
        "is_constant": bool(nf > 0 and np.all(xf == xf[0])),
    }
    row.update(_stationarity(x, p))
    for lag in p["acf_lags"]:
        row[f"acf_lag{lag}"] = _acf(x, lag, p["min_pairs"])
    seg = _longest_finite_run(x)
    row["spectral_entropy"] = _spectral_entropy(seg, p["min_pairs"])
    period, wlen = _dominant_period(seg, p["trailing_window"], p["min_pairs"])
    row["dominant_trailing_period_bars"] = period
    row["trailing_window_rows"] = wlen
    row.update(_target_association(x, close, p))
    return row


# ----------------------------------------------------------------------------- materializer

def data_digest(train: np.ndarray, close: np.ndarray, features: list, target: str) -> str:
    h = hashlib.sha256()
    h.update(_canon({"features": features, "target": target, "rows": int(len(close))}).encode())
    h.update(np.ascontiguousarray(close, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(train, dtype="<f8").tobytes())
    return h.hexdigest()


def materialize(train_df: pd.DataFrame, features: list, params: dict | None = None,
                row_offset: int = 0) -> dict:
    """Metrics from a frame that is already train-only. The frame is the whole input."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    n = len(train_df)
    if row_offset < 0 or row_offset + n > PROTECTED_TEST_ROWS[0]:
        raise ScopeError("train slice must lie before the protected test rows")
    X = train_df[features].to_numpy(dtype="float64")
    close = train_df[p["target_column"]].to_numpy(dtype="float64")
    d_digest = data_digest(X, close, features, p["target_column"])
    p_digest = _sha(_canon(p))
    rows = []
    for j, name in enumerate(features):
        r = feature_row(name, X[:, j], close, p)
        r = {"row_id": _sha(f"{FIT_SCOPE}|{name}|{d_digest}|{p_digest}")[:32],
             "fit_scope": FIT_SCOPE, "data_digest": d_digest, "parameter_digest": p_digest,
             **r}
        rows.append(r)
    return {"schema": SCHEMA, "fit_scope": FIT_SCOPE,
            "fit_rows": {"start": row_offset, "stop": row_offset + n},
            "protected_test_rows": list(PROTECTED_TEST_ROWS),
            "data_digest": d_digest, "parameter_digest": p_digest, "parameters": p,
            "n_features": len(features),
            "adf_kpss_dependency": "statsmodels" if HAVE_STATSMODELS else "missing",
            "note": "descriptive train-only statistics; no selection, no causal claim",
            "rows": rows}


def run(csv_path: str, manifest_path: str, out_dir: str, params: dict | None = None) -> dict:
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    contract = load_contract(manifest_path)
    df = load_train_frame(csv_path, contract, p)
    res = materialize(df.drop(columns=[p["date_column"]]), contract["features"], p)
    res["manifest_sha256_declared"] = contract["manifest_sha256"]
    res["train_end"] = contract["train_end"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_feature_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    pd.DataFrame(res["rows"]).to_csv(os.path.join(out_dir, "train_feature_metrics.csv"),
                                     index=False)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    res = run(a.csv, a.manifest, a.out_dir)
    print(f"rows={res['fit_rows']} features={res['n_features']} data_digest={res['data_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
