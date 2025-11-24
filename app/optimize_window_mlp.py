#!/usr/bin/env python3
"""Benchmark sliding-window sizes for MLP regressors on hourly data.

This script replaces the old horizon optimizer with a focused experiment that:
  * Loads a high-frequency OHLC CSV (default: 5-minute bars)
  * Builds the typical price ( (HIGH + LOW + CLOSE) / 3 )
  * Resamples the series to 1H and 4H tracks and evaluates fixed horizons
  * Trains an sklearn ``MLPRegressor`` for each window size per track
  * Compares the learned model against a naive ``last value`` baseline using MAE

CLI usage is intentionally simple: provide the path to the source CSV and (optionally)
an output CSV destination. Results are printed to stdout and saved for downstream
analysis in the same format as the legacy benchmark.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

WINDOW_SIZES: List[int] = [3, 6, 12, 24, 48, 96]
TRAIN_RATIO: float = 0.7

TRACKS: Dict[str, Dict[str, int | str]] = {
    "1H": {"freq": "1H", "horizon": 24},   # predict 24 hours ahead
    "4H": {"freq": "4H", "horizon": 30},   # predict 30×4H bars (~5 days)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sliding-window sizes for hourly / 4-hour MLP regressors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv",
        help="Path to the source OHLC CSV (must contain DATE_TIME, HIGH, LOW, CLOSE columns)",
    )
    parser.add_argument(
        "--output",
        default="optimize_window_mlp_results.csv",
        help="Destination CSV file for aggregated metrics",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Proportion of samples to keep in the training split (time-ordered)",
    )
    return parser.parse_args()


def read_source_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip().upper() for col in df.columns]
    if "DATE_TIME" not in df.columns:
        raise ValueError("CSV must include a DATE_TIME column")

    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], utc=True, errors="coerce")
    df = df.dropna(subset=["DATE_TIME"]).sort_values("DATE_TIME")
    df = df.set_index("DATE_TIME")

    required = {"OPEN", "HIGH", "LOW", "CLOSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    return df[list(required)].astype(float)


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    ohlc = df.resample(freq).agg(
        {
            "OPEN": "first",
            "HIGH": "max",
            "LOW": "min",
            "CLOSE": "last",
        }
    )
    return ohlc.dropna()


def typical_price(ohlc: pd.DataFrame) -> pd.Series:
    return ((ohlc["HIGH"] + ohlc["LOW"] + ohlc["CLOSE"]) / 3.0).rename("TYPICAL_PRICE")


def make_supervised(series: pd.Series, window: int, horizon: int) -> Optional[Dict[str, np.ndarray]]:
    values = series.to_numpy(dtype=np.float64)
    samples = len(values) - window - horizon + 1
    if samples <= 0:
        return None

    X = np.empty((samples, window), dtype=np.float64)
    y = np.empty(samples, dtype=np.float64)
    naive = np.empty(samples, dtype=np.float64)

    for i in range(samples):
        start = i
        end = start + window
        target_idx = end - 1 + horizon
        X[i] = values[start:end]
        y[i] = values[target_idx]
        naive[i] = values[end - 1]

    return {"X": X, "y": y, "naive": naive}


def split_train_val(X: np.ndarray, y: np.ndarray, naive: np.ndarray, train_ratio: float) -> Optional[Dict[str, np.ndarray]]:
    split_idx = int(len(X) * train_ratio)
    if split_idx < 2 or len(X) - split_idx < 2:
        return None

    return {
        "X_train": X[:split_idx],
        "X_val": X[split_idx:],
        "y_train": y[:split_idx],
        "y_val": y[split_idx:],
        "naive_train": naive[:split_idx],
        "naive_val": naive[split_idx:],
    }


def build_regressor(random_state: int = 42) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=64,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
    )


def evaluate_window(
    track: str,
    series: pd.Series,
    window: int,
    horizon: int,
    train_ratio: float,
) -> Optional[Dict[str, float]]:
    supervised = make_supervised(series, window, horizon)
    if supervised is None:
        print(f"[SKIP] {track} window={window}: insufficient history for horizon {horizon}")
        return None

    splits = split_train_val(supervised["X"], supervised["y"], supervised["naive"], train_ratio)
    if splits is None:
        print(f"[SKIP] {track} window={window}: split yielded fewer than 2 samples per set")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(splits["X_train"])
    X_val_scaled = scaler.transform(splits["X_val"])

    model = build_regressor()
    start_time = time.time()
    model.fit(X_train_scaled, splits["y_train"])
    fit_seconds = time.time() - start_time

    pred_train = model.predict(X_train_scaled)
    pred_val = model.predict(X_val_scaled)

    mae_train = mean_absolute_error(splits["y_train"], pred_train)
    mae_val = mean_absolute_error(splits["y_val"], pred_val)
    naive_mae_train = mean_absolute_error(splits["y_train"], splits["naive_train"])
    naive_mae_val = mean_absolute_error(splits["y_val"], splits["naive_val"])

    print(
        f"[TRACK {track}] window={window:>3} | samples train/val={len(splits['X_train'])}/{len(splits['X_val'])} "
        f"| MAE_val={mae_val:.6f} vs naive {naive_mae_val:.6f} (Δ={naive_mae_val - mae_val:.6f})"
    )

    return {
        "track": track,
        "frequency": track,
        "window": window,
        "horizon": horizon,
        "train_samples": len(splits["X_train"]),
        "val_samples": len(splits["X_val"]),
        "mae_train": mae_train,
        "mae_val": mae_val,
        "naive_mae_train": naive_mae_train,
        "naive_mae_val": naive_mae_val,
        "mae_gain_val": naive_mae_val - mae_val,
        "fit_seconds": fit_seconds,
    }


def summarize_results(results: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    print("\n================ SUMMARY ================")
    for track in df["track"].unique():
        track_df = df[df["track"] == track].sort_values("mae_val")
        if track_df.empty:
            continue
        best = track_df.iloc[0]
        print(
            f"Best {track}: window={int(best['window'])} | MAE_val={best['mae_val']:.6f} | "
            f"naive={best['naive_mae_val']:.6f} | gain={best['mae_gain_val']:.6f}"
        )
    print("========================================\n")
    return df.sort_values(["track", "window"]).reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    data = read_source_csv(csv_path)

    results: List[Dict[str, float]] = []
    for track_name, cfg in TRACKS.items():
        freq = str(cfg["freq"])
        horizon = int(cfg["horizon"])
        track_ohlc = resample_ohlc(data, freq)
        series = typical_price(track_ohlc)

        if series.empty:
            print(f"[WARN] Track {track_name} produced no data after resampling")
            continue

        for window in WINDOW_SIZES:
            result = evaluate_window(track_name, series, window, horizon, args.train_ratio)
            if result:
                results.append(result)

    if not results:
        raise RuntimeError("No results were produced. Check input data and parameters.")

    summary = summarize_results(results)
    output_path = Path(args.output)
    summary.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - surfacing full context to CLI
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
