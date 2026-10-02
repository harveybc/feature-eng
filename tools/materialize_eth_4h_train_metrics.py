#!/usr/bin/env python3
"""Materialize descriptive metrics for the frozen ETHUSDT 4h TRAIN contract.

The reader stops after the declared TRAIN prefix.  Validation and test rows are
never parsed, hashed, or used to construct a target association.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy import signal


SCHEMA = "eth_4h_train_metrics.v1"
HORIZONS = (6, 12, 18, 24, 30, 36)
ACF_LAGS = (1, 6, 24)
MISSING = {"", "nan", "na", "null", "none", "n/a"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: object) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def validate_contract(contract: dict) -> None:
    if contract.get("schema") != "selected_feature_manifest.v1":
        raise ValueError("selected_feature_manifest.v1 contract required")
    if contract.get("feature_count") != 83 or len(contract.get("features", [])) != 83:
        raise ValueError("ETH 4h contract must declare exactly 83 features")
    if len(set(contract["features"])) != 83:
        raise ValueError("feature IDs must be unique")
    resource = contract.get("resource", {})
    split = contract.get("split", {}).get("train", {})
    rows = split.get("rows")
    if resource.get("asset") not in (None, "ETHUSDT spot"):
        raise ValueError("contract asset must be ETHUSDT spot")
    if resource.get("bar") not in (None, "4h"):
        raise ValueError("contract bar must be 4h")
    if not isinstance(rows, list) or rows[0] != 0 or not isinstance(rows[1], int) or rows[1] <= 36:
        raise ValueError("contract needs a TRAIN prefix longer than the largest horizon")
    if resource.get("timestamp_column") != "DATE_TIME":
        raise ValueError("contract timestamp column must be DATE_TIME")


class ExactLines:
    """Yield complete physical lines without reading a byte past TRAIN."""

    def __init__(self, raw: object) -> None:
        self.raw = raw
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = bytearray()
        while True:
            byte = self.raw.read(1)
            if not byte:
                if line:
                    return line.decode("utf-8")
                raise StopIteration
            self.hasher.update(byte)
            self.bytes_read += 1
            line.extend(byte)
            if byte == b"\n":
                return line.decode("utf-8")


def read_train(path: Path, contract: dict):
    train_rows = contract["split"]["train"]["rows"][1]
    with path.open("rb", buffering=0) as raw:
        lines = ExactLines(raw)
        reader = csv.reader(lines, strict=True)
        header = next(reader)
        required = {"DATE_TIME", "CLOSE", *contract["features"]}
        if len(header) != len(set(header)) or not required.issubset(header):
            raise ValueError("source schema does not satisfy the frozen ETH 4h contract")
        values = np.full((train_rows, len(contract["features"]) + 1), np.nan)
        positions = {name: header.index(name) for name in ("CLOSE", *contract["features"])}
        for row_number in range(train_rows):
            try:
                row = next(reader)
            except StopIteration as exc:
                raise ValueError("source ended before the declared TRAIN prefix") from exc
            if len(row) != len(header):
                raise ValueError("malformed CSV record width inside TRAIN")
            for output_index, name in enumerate(("CLOSE", *contract["features"])):
                token = row[positions[name]].strip()
                if token.lower() not in MISSING:
                    try:
                        values[row_number, output_index] = float(token)
                    except ValueError:
                        pass
        return values[:, 0], values[:, 1:], lines.hasher.hexdigest(), lines.bytes_read


def longest_finite_run(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return values[:0]
    index = int(np.argmax(ends - starts))
    return values[starts[index]:ends[index]]


def trailing_finite_window(values: np.ndarray, window: int) -> np.ndarray:
    end = len(values)
    while end and not np.isfinite(values[end - 1]):
        end -= 1
    start = end
    while start and np.isfinite(values[start - 1]):
        start -= 1
    return values[max(start, end - window):end]


def stationarity(values: np.ndarray) -> dict:
    result = {}
    for name, null in (("adf", "unit_root"), ("kpss", "level_stationarity")):
        row = {"status": "NOT_RUN", "null": null, "n": int(len(values))}
        if len(values) < 50:
            row["reason"] = "INSUFFICIENT_SAMPLE"
        elif np.ptp(values) == 0:
            row["reason"] = "CONSTANT"
        else:
            try:
                from statsmodels.tsa.stattools import adfuller, kpss
            except ImportError:
                row.update(status="UNAVAILABLE", reason="statsmodels not installed")
            else:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        answer = adfuller(values, regression="c", autolag="AIC") if name == "adf" else kpss(values, regression="c", nlags="auto")
                    row.update(status="OK_WITH_WARNING" if caught else "OK", statistic=float(answer[0]),
                               pvalue=float(answer[1]), lags=int(answer[2]))
                except Exception as exc:  # optional estimator failures must remain visible
                    row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        result[name] = row
    return result


def acf(values: np.ndarray, lag: int) -> dict:
    if len(values) <= lag or np.ptp(values) == 0:
        return {"status": "NOT_RUN", "reason": "INSUFFICIENT_SAMPLE_OR_CONSTANT", "value": None}
    value = float(np.corrcoef(values[:-lag], values[lag:])[0, 1])
    return {"status": "OK" if math.isfinite(value) else "FAILED", "value": value if math.isfinite(value) else None}


def spectral(values: np.ndarray) -> dict:
    if len(values) < 8 or np.ptp(values) == 0:
        return {"status": "NOT_RUN", "reason": "INSUFFICIENT_SAMPLE_OR_CONSTANT"}
    frequency, power = signal.periodogram(values, detrend="linear")
    frequency, power = frequency[1:], power[1:]
    total = float(power.sum())
    if total <= np.finfo(float).eps:
        return {"status": "NOT_RUN", "reason": "NEGLIGIBLE_DETRENDED_ENERGY"}
    probability = power / total
    nonzero = probability[probability > 0]
    entropy = 0.0 if len(probability) == 1 else float(-(nonzero * np.log(nonzero)).sum() / np.log(len(probability)))
    top = int(np.argmax(probability))
    return {"status": "OK", "entropy_normalized": entropy,
            "dominant_period_rows": float(1 / frequency[top]),
            "dominant_power_fraction": float(probability[top])}


def target_association(feature: np.ndarray, close: np.ndarray, horizon: int) -> dict:
    target = close[horizon:] / close[:-horizon] - 1.0
    candidate = feature[:-horizon]
    mask = np.isfinite(candidate) & np.isfinite(target)
    if mask.sum() < 3 or np.ptp(candidate[mask]) == 0 or np.ptp(target[mask]) == 0:
        return {"status": "NOT_RUN", "reason": "INSUFFICIENT_PAIRED_VARIATION", "n": int(mask.sum()), "pearson": None}
    value = float(np.corrcoef(candidate[mask], target[mask])[0, 1])
    return {"status": "OK" if math.isfinite(value) else "FAILED", "n": int(mask.sum()),
            "pearson": value if math.isfinite(value) else None,
            "target": f"CLOSE[t+{horizon}]/CLOSE[t]-1"}


def materialize(contract: dict, close: np.ndarray, features: np.ndarray, data_digest: str, consumed_bytes: int,
                parameter_digest: str, trailing_window: int) -> list[dict]:
    availability = {"resource_availability_class": contract["resource"]["availability_class"],
                    "decision_time": contract["task"]["decision_time"],
                    "point_in_time": contract["resource"]["point_in_time"]}
    scope = {"split": "TRAIN", "row_range": contract["split"]["train"]["rows"],
             "source_rows_read": int(len(close)), "consumed_bytes": consumed_bytes}
    rows = []
    for index, feature_id in enumerate(contract["features"]):
        values = features[:, index]
        finite = values[np.isfinite(values)]
        segment = longest_finite_run(values)
        trailing = trailing_finite_window(values, trailing_window)
        row = {"type": "feature_train_metric_row.v1", "row_id": f"eth_4h:feature:{feature_id}",
               "feature_id": feature_id, "availability": availability, "fit_scope": scope,
               "data_digest": data_digest, "parameter_digest": parameter_digest,
               "missingness": {"rows": int(len(values)), "missing_count": int((~np.isfinite(values)).sum()),
                               "missing_fraction": float((~np.isfinite(values)).mean()), "finite_count": int(len(finite))},
               "variance": {"status": "OK" if len(finite) > 1 else "NOT_RUN",
                            "ddof": 1, "value": float(np.var(finite, ddof=1)) if len(finite) > 1 else None},
               "stationarity": stationarity(segment), "acf": {str(lag): acf(segment, lag) for lag in ACF_LAGS},
               "spectral_entropy": {"status": spectral(segment).get("status"), "value": spectral(segment).get("entropy_normalized")},
               "trailing_dominant_period": {**spectral(trailing), "window_rows": int(len(trailing))},
               "target_association": {str(h): target_association(values, close, h) for h in HORIZONS}}
        rows.append(row)
    return rows


def clean(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def run(contract_path: Path, input_path: Path, output: Path, trailing_window: int = 252) -> dict:
    if trailing_window < 8:
        raise ValueError("trailing window must be at least 8 rows")
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    validate_contract(contract)
    repo = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to(repo) or output.exists():
        raise ValueError("output must be a new directory inside this feature-eng worktree")
    close, features, data_digest, consumed_bytes = read_train(input_path, contract)
    parameters = {"schema": SCHEMA, "contract_sha256": digest(contract_bytes), "horizons": HORIZONS,
                  "acf_lags": ACF_LAGS, "target": "CLOSE[t+h]/CLOSE[t]-1", "trailing_window": trailing_window}
    parameter_digest = canonical_digest(parameters)
    rows = clean(materialize(contract, close, features, data_digest, consumed_bytes, parameter_digest, trailing_window))
    output.mkdir(parents=True)
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    flat = []
    for row in rows:
        flat.append({"type": row["type"], "row_id": row["row_id"], "feature_id": row["feature_id"],
                     "fit_scope": json.dumps(row["fit_scope"], sort_keys=True), "data_digest": row["data_digest"],
                     "parameter_digest": row["parameter_digest"], "availability": json.dumps(row["availability"], sort_keys=True),
                     "missing_fraction": row["missingness"]["missing_fraction"], "variance": row["variance"]["value"],
                     "adf_status": row["stationarity"]["adf"]["status"], "kpss_status": row["stationarity"]["kpss"]["status"],
                     "acf_lag_1": row["acf"]["1"]["value"], "acf_lag_6": row["acf"]["6"]["value"],
                     "acf_lag_24": row["acf"]["24"]["value"], "spectral_entropy": row["spectral_entropy"]["value"],
                     "trailing_dominant_period_rows": row["trailing_dominant_period"].get("dominant_period_rows"),
                     **{f"target_association_h{h}": row["target_association"][str(h)]["pearson"] for h in HORIZONS}})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    receipt = {"schema": SCHEMA, "rows": len(rows), "fit_scope": rows[0]["fit_scope"], "data_digest": data_digest,
               "parameter_digest": parameter_digest, "contract_sha256": digest(contract_bytes),
               "not_claimed": ["feature selection", "feature dropping", "causality", "point-in-time availability certification"]}
    (output / "materialization.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"rows": rows, "receipt": receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trailing-window", type=int, default=252)
    args = parser.parse_args()
    result = run(args.contract, args.input, args.output, args.trailing_window)
    print(json.dumps({"output": str(args.output), "rows": len(result["rows"]), **result["receipt"]}, sort_keys=True))


if __name__ == "__main__":
    main()
