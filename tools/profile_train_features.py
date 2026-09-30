#!/usr/bin/env python3
"""Bounded diagnostic CSV profiling. Never opens a heldout input or fits a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_manifest(m, max_rows):
    if m.get("split") != "TRAIN" or m.get("schema") != "feature_train_manifest.v1":
        raise ValueError("An explicit feature_train_manifest.v1 TRAIN declaration is required")
    if not m.get("dataset_id") or not m.get("declaration_source"):
        raise ValueError("dataset_id and declaration_source are required")
    if not 1 <= max_rows <= 4096:
        raise ValueError("max_rows must be 1..4096")
    ranges = m.get("boundaries", {})
    for name, bounds in ranges.items():
        if (len(bounds) != 2 or any(type(v) is not int for v in bounds)
                or bounds[0] < 0 or bounds[1] <= bounds[0]):
            raise ValueError(f"Invalid half-open row boundary: {name}")
    train = ranges.get("train")
    if not train or train[0] != 0 or max_rows > train[1]:
        raise ValueError("Only a declared TRAIN prefix starting at row zero is supported")
    if m.get("layout") not in ("train_only_file", "mixed_prefix"):
        raise ValueError("layout must be train_only_file or mixed_prefix")
    if m["layout"] == "train_only_file" and set(ranges) != {"train"}:
        raise ValueError("Separate files use separate coordinate systems; declare only train here")
    ordered = sorted(ranges.values())
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Split boundaries overlap")
    if not isinstance(m.get("columns"), dict) or not m["columns"]:
        raise ValueError("Every column must have an explicit role")
    allowed = {"feature", "target", "timestamp", "identifier", "excluded"}
    if any(v.get("role") not in allowed for v in m["columns"].values()):
        raise ValueError("Unknown column role")
    if sum(v["role"] == "timestamp" for v in m["columns"].values()) > 1:
        raise ValueError("At most one timestamp column")


class ExactLines:
    """Unbuffered read(1) is deliberate: no application read crosses the row cap."""

    def __init__(self, raw, byte_cap):
        self.raw, self.byte_cap = raw, byte_cap
        self.bytes_read = 0
        self.hasher = hashlib.sha256()

    def __iter__(self):
        return self

    def __next__(self):
        line = bytearray()
        while self.bytes_read < self.byte_cap:
            b = self.raw.read(1)
            if not b:
                if line:
                    return line.decode("utf-8")
                raise StopIteration
            self.bytes_read += 1
            self.hasher.update(b)
            line.extend(b)
            if b == b"\n":
                return line.decode("utf-8")
        raise ValueError("Input byte cap reached before a complete CSV record")


def read_train(raw, m, rows, byte_cap):
    lines = ExactLines(raw, byte_cap)
    reader = csv.reader(lines, strict=True)
    header = next(reader)
    if len(header) > 2048 or len(header) != len(set(header)):
        raise ValueError("Duplicate columns or schema exceeds 2048 columns")
    if set(header) != set(m["columns"]):
        raise ValueError("CSV schema does not match manifest column roles")
    values = []
    for _ in range(rows):
        try:
            row = next(reader)
        except StopIteration as exc:
            raise ValueError("Source ended before requested TRAIN prefix") from exc
        if len(row) != len(header):
            raise ValueError("Malformed CSV record width")
        values.append(row)
    return pd.DataFrame(values, columns=header), lines.hasher.hexdigest(), lines.bytes_read


def longest_run(x):
    finite = np.isfinite(x)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, 0
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


def stationarity(x):
    out = {}
    for name in ("adf", "kpss"):
        result = {"status": "NOT_RUN", "n": len(x), "regression": "c",
                  "null": "unit_root" if name == "adf" else "level_stationarity"}
        out[name] = result
        if len(x) < 50 or np.ptp(x) == 0:
            result["reason"] = "INSUFFICIENT_SAMPLE" if len(x) < 50 else "CONSTANT"
            continue
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            result.update(status="UNAVAILABLE", reason="statsmodels not installed")
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if name == "adf":
                    lag = min(12, len(x) // 2 - 2)
                    answer = adfuller(x, regression="c", maxlag=lag, autolag=None)
                    result.update(lag=lag, autolag=None)
                else:
                    answer = kpss(x, regression="c", nlags="auto")
                    result.update(lag=int(answer[2]), lag_rule="auto")
            result.update(status="OK", statistic=float(answer[0]), pvalue=float(answer[1]),
                          warnings=[str(w.message) for w in caught])
            if not np.isfinite(answer[:2]).all():
                result.update(status="FAILED", reason="Nonfinite estimator result")
            elif caught:
                result["status"] = "OK_WITH_WARNING"
        except Exception as exc:
            result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
    return out


def temporal_metrics(x):
    start, end = longest_run(x)
    y = x[start:end]
    out = {"segment": [start, end], "segment_rule": "longest contiguous finite run; earliest tie",
           "status": "NOT_RUN", "stationarity": stationarity(y)}
    if len(y) < 8 or np.ptp(y) == 0:
        out["reason"] = "INSUFFICIENT_SAMPLE" if len(y) < 8 else "CONSTANT"
        return out
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    residual = y - (slope * t + intercept)
    centered = y - y.mean()
    energy = float(centered @ centered)
    nlag = min(64, len(y) // 2)
    acf = signal.correlate(centered, centered, mode="full", method="fft")[len(y) - 1:len(y) + nlag] / energy
    peaks, _ = signal.find_peaks(acf[1:])
    peaks = sorted((int(p + 1) for p in peaks if acf[p + 1] > 0),
                   key=lambda p: -acf[p])[:5]
    f, power = signal.periodogram(y, fs=1.0, detrend="linear")
    f, power = f[1:], power[1:]
    total = float(power.sum())
    spectral = {"status": "NOT_RUN", "reason": "NEGLIGIBLE_DETRENDED_ENERGY"}
    if total > np.finfo(float).eps * energy:
        p = power / total
        top = int(np.argmax(p))
        positive = p[p > 0]
        spectral = {"status": "OK", "peak_frequency_cycles_per_row": float(f[top]),
                    "peak_period_rows": float(1 / f[top]), "peak_power_fraction": float(p[top]),
                    "entropy_normalized": float(-(positive * np.log(positive)).sum() / np.log(len(p))),
                    "centroid_cycles_per_row": float(f @ p),
                    "low_frequency_power_fraction": float(p[f <= .1].sum())}
    out.update(status="OK", trend_slope_per_row=float(slope),
               trend_r2=float(1 - (residual @ residual) / energy),
               acf={"estimator": "biased, mean-centered, normalized lag zero",
                    "values": acf.tolist(), "positive_local_peak_lags": peaks,
                    "interpretation": "descriptive peaks, not significance tests"},
               spectral=spectral)
    return out


def profile_frame(frame, m, column_cap=64, pair_cap=0):
    features, series = [], {}
    numeric_used = 0
    sampling = {"status": "UNVERIFIED", "assumption": "row-index spacing; no physical-time claim"}
    for name in frame:
        spec = m["columns"][name]
        s = frame[name]
        missing = s.str.strip().str.lower().isin(("", "nan", "na", "null", "none"))
        converted = pd.to_numeric(s.mask(missing), errors="coerce").to_numpy(dtype=float)
        finite = converted[np.isfinite(converted)]
        item = {"column": name, "role": spec["role"], "unit": spec.get("unit", "UNDECLARED"),
                "rows": len(s), "missing_count": int(missing.sum()),
                "nonnumeric_count": int((~missing & pd.isna(converted)).sum()),
                "infinite_count": int(np.isinf(converted).sum()), "finite_count": len(finite),
                "constant": bool(len(finite) > 0 and np.ptp(finite) == 0),
                "unique_nonmissing": int(s[~missing].nunique()), "status": "EXCLUDED",
                "exclusion_reason": None}
        item["missing_fraction"] = item["missing_count"] / len(s)
        if spec["role"] == "timestamp":
            ts = pd.to_datetime(s, format=m.get("timestamp_format"), errors="coerce")
            dt = ts.diff().dt.total_seconds().iloc[1:]
            regular = bool(ts.notna().all() and len(dt) > 0 and (dt > 0).all() and dt.nunique() == 1)
            sampling = {"status": "REGULAR" if regular else "IRREGULAR_OR_INVALID",
                        "invalid_count": int(ts.isna().sum()), "first": str(ts.iloc[0]),
                        "last": str(ts.iloc[-1]), "median_step_seconds": float(dt.median()),
                        "assumption": "temporal metrics use row order; physical-time inference only if regular"}
        if spec["role"] != "feature":
            item["exclusion_reason"] = spec.get("reason", spec["role"])
        elif item["nonnumeric_count"]:
            item["exclusion_reason"] = "NONNUMERIC_VALUES"
        elif not len(finite):
            item["exclusion_reason"] = "NO_FINITE_VALUES"
        elif numeric_used >= column_cap:
            item["exclusion_reason"] = "NUMERIC_COLUMN_CAP"
        else:
            numeric_used += 1
            q25, median, q75 = np.quantile(finite, [.25, .5, .75])
            diff = np.diff(converted)
            diff = diff[np.isfinite(diff)]
            item.update(mean=float(finite.mean()), std=float(finite.std(ddof=1)) if len(finite) > 1 else None,
                        minimum=float(finite.min()), maximum=float(finite.max()), median=float(median),
                        iqr=float(q75 - q25), mad=float(np.median(np.abs(finite - median))),
                        volatility_diff_std=float(diff.std(ddof=1)) if len(diff) > 1 else None,
                        volatility_pairs=len(diff), temporal=temporal_metrics(converted), status="PROFILED")
            if item["constant"]:
                item["exclusion_reason"] = "CONSTANT_IN_PROFILE_PREFIX"
            else:
                series[name] = converted
        features.append(item)
    pairs = []
    total_pairs = len(series) * (len(series) - 1) // 2
    for a, b in itertools.islice(itertools.combinations(series, 2), pair_cap):
        mask = np.isfinite(series[a]) & np.isfinite(series[b])
        x, y = series[a][mask], series[b][mask]
        row = {"a": a, "b": b, "n": len(x), "status": "NOT_RUN"}
        if len(x) >= 8 and np.ptp(x) > 0 and np.ptp(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
            row.update(status="OK", pearson=corr, redundant_abs_ge_0_95=abs(corr) >= .95)
        else:
            row["reason"] = "INSUFFICIENT_PAIRED_VARIATION"
        pairs.append(row)
    return {"features": features, "sampling": sampling,
            "branches": [{"branch": i, "columns": [name]} for i, name in enumerate(series)],
            "grouping": "one admissible feature per branch; no automatic selection or merging",
            "redundancy": {"enabled": pair_cap > 0, "rule": "first pairs in CSV schema order; Pearson on pairwise finite TRAIN rows; diagnostic only",
                           "pairs": pairs, "possible_pairs": total_pairs, "not_computed": total_pairs - len(pairs)}}


def clean(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def write_outputs(out, report):
    out.mkdir(parents=True, exist_ok=False)
    report = clean(report)
    (out / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = []
    for f in report["features"]:
        row = {k: v for k, v in f.items() if k != "temporal"}
        t = f.get("temporal", {})
        row.update(temporal_status=t.get("status", "NOT_RUN"),
                   trend_slope=t.get("trend_slope_per_row"), trend_r2=t.get("trend_r2"),
                   acf_lag1=(t.get("acf", {}).get("values", [None, None]))[1],
                   spectral_status=t.get("spectral", {}).get("status", "NOT_RUN"),
                   peak_period_rows=t.get("spectral", {}).get("peak_period_rows"))
        for test in ("adf", "kpss"):
            result = t.get("stationarity", {}).get(test, {})
            row[test + "_status"] = result.get("status", "NOT_RUN")
            row[test + "_pvalue"] = result.get("pvalue")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "features.csv", index=False)
    title = html.escape(report["dataset_id"])
    (out / "features.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>TRAIN feature metrics</title>'
        '<style>body{font:14px sans-serif;margin:20px}table{border-collapse:collapse}'
        'td,th{padding:6px;border:1px solid #ccc;white-space:nowrap}</style>'
        f'<h1>{title}</h1><p>TRAIN rows {report["profile_range"]}; bounded descriptive profile.</p>'
        + table.to_html(index=False, escape=True))


def run(manifest_path, source_root, output, max_rows=512, column_cap=64, pair_cap=0, byte_cap=16 << 20):
    started = time.monotonic()
    manifest_bytes = Path(manifest_path).read_bytes()
    m = json.loads(manifest_bytes)
    validate_manifest(m, max_rows)
    if not 1 <= column_cap <= 64 or not 0 <= pair_cap <= 256 or not 1 <= byte_cap <= 16 << 20:
        raise ValueError("Resource bounds: columns 1..64; pairs 0..256; bytes 1..16777216")
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(repo) or output.exists():
        raise ValueError("Output must be a new directory inside this worktree")
    root = Path(source_root).resolve()
    path = (root / m["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Input must be an accessible regular file beneath source-root")
    with path.open("rb", buffering=0) as raw:
        frame, sha, read_bytes = read_train(raw, m, max_rows, byte_cap)
    report = profile_frame(frame, m, column_cap, pair_cap)
    versions = {"python": platform.python_version()}
    for dep in ("numpy", "scipy", "pandas", "statsmodels"):
        try:
            versions[dep] = importlib.metadata.version(dep)
        except importlib.metadata.PackageNotFoundError:
            versions[dep] = "NOT_INSTALLED"
    report.update(schema="feature_train_profile.v1", dataset_id=m["dataset_id"], split="TRAIN",
                  manifest=m, manifest_sha256=digest(manifest_bytes),
                  source_code_sha256=digest(Path(__file__).read_bytes()),
                  consumed_input_sha256=sha, consumed_bytes=read_bytes, profile_range=[0, len(frame)],
                  full_train_coverage=len(frame) == m["boundaries"]["train"][1],
                  limits={"max_rows": max_rows, "numeric_columns": column_cap, "pairs": pair_cap, "bytes": byte_cap},
                  assumptions=["Upstream normalization and feature causality are not certified",
                               "No missing-value imputation; temporal tests use a contiguous finite run",
                               "ADF/KPSS p-values are diagnostics, not multiple-testing-adjusted decisions",
                               "Volatility is adjacent finite first-difference sample std, not return volatility",
                               "No heldout input opened; bytes hash identifies consumed prefix, not full dataset"],
                  versions=versions, wall_seconds=time.monotonic() - started)
    write_outputs(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=512)
    p.add_argument("--max-columns", type=int, default=64)
    p.add_argument("--pair-cap", type=int, default=0)
    p.add_argument("--max-bytes", type=int, default=16 << 20)
    a = p.parse_args()
    r = run(a.manifest, a.source_root, a.output, a.max_rows, a.max_columns, a.pair_cap, a.max_bytes)
    print(json.dumps({"output": str(a.output), "columns": len(r["features"]), "branches": len(r["branches"]),
                      "profile_range": r["profile_range"], "consumed_bytes": r["consumed_bytes"]}))


if __name__ == "__main__":
    main()
