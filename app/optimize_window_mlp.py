#!/usr/bin/env python3
"""Benchmark sliding-window sizes for Conv1D regressors on hourly data.

This script replaces the old horizon optimizer with a focused experiment that:
  * Loads a high-frequency OHLC CSV (default: 5-minute bars)
  * Builds the typical price ( (HIGH + LOW + CLOSE) / 3 )
  * Resamples the series to 1H and 4H tracks and evaluates fixed horizons
    * Trains a TensorFlow/Keras Conv1D stack for each window size per track
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
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

WINDOW_SIZES: List[int] = [3, 6, 12, 24, 48, 96, 120, 144, 240]

TRACKS: Dict[str, Dict[str, int | str]] = {
    "1H": {"freq": "1h", "horizon": 24},   # predict 24 hours ahead
    "4H": {"freq": "4h", "horizon": 30},   # predict 30×4H bars (~5 days)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sliding-window sizes for hourly / 4-hour MLP regressors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv",
        help="Path to the source OHLC CSV (must contain datetime/open/high/low/close columns)",
    )
    parser.add_argument(
        "--output",
        default="optimize_window_mlp_results.csv",
        help="Destination CSV file for aggregated metrics",
    )
    return parser.parse_args()


def read_source_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize column names to upper-case to be forgiving with casing, but accept both
    # 'DATE_TIME' and 'DATETIME' as the datetime column name.
    cols_map = {c: c.strip() for c in df.columns}
    df.columns = [c.strip() for c in df.columns]

    # Accept either 'DATE_TIME' or 'datetime' (any case). We'll check both variants.
    columns_upper = [c.upper() for c in df.columns]
    if 'DATE_TIME' in columns_upper:
        dt_col = df.columns[columns_upper.index('DATE_TIME')]
    elif 'DATETIME' in columns_upper:
        dt_col = df.columns[columns_upper.index('DATETIME')]
    else:
        raise ValueError("CSV must include a 'datetime' or 'DATE_TIME' column")

    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
    df = df.dropna(subset=[dt_col]).sort_values(dt_col)
    df = df.set_index(dt_col)

    # required OHLC names (case-insensitive)
    col_upper_map = {c.upper(): c for c in df.columns}
    required_upper = {'OPEN', 'HIGH', 'LOW', 'CLOSE'}
    missing = required_upper - set(col_upper_map.keys())
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    # return a DataFrame with standard column names
    std_cols = {u: col_upper_map[u] for u in required_upper}
    out = df[[std_cols[u] for u in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]].copy()
    out.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    return out.astype(float)


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
    n = len(values)
    samples = n - window - horizon + 1
    if samples <= 0:
        return None

    X = np.empty((samples, window), dtype=np.float64)
    y = np.empty(samples, dtype=np.float64)
    naive = np.empty(samples, dtype=np.float64)
    idx = []

    for i in range(samples):
        start = i
        end = start + window
        target_idx = end - 1 + horizon
        X[i] = values[start:end]
        y[i] = values[target_idx]
        naive[i] = values[end - 1]
        idx.append(series.index[end - 1])

    return {"X": X, "y": y, "naive": naive, "idx": pd.DatetimeIndex(idx)}


def split_train_val_test(idx: pd.DatetimeIndex, X: np.ndarray, y: np.ndarray, naive: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    # train = years -5 to -2, val = year -2 to -1, test = last year
    last_date = idx.max()
    if pd.isnull(last_date):
        return None
    test_start = last_date - pd.DateOffset(years=1)
    val_start = last_date - pd.DateOffset(years=2)
    train_start = last_date - pd.DateOffset(years=6)

    mask_test = idx > test_start
    mask_val = (idx > val_start) & (idx <= test_start)
    mask_train = (idx > train_start) & (idx <= val_start)

    if mask_train.sum() < 2 or mask_val.sum() < 2 or mask_test.sum() < 2:
        return None

    return {
        "X_train": X[mask_train],
        "X_val": X[mask_val],
        "X_test": X[mask_test],
        "y_train": y[mask_train],
        "y_val": y[mask_val],
        "y_test": y[mask_test],
        "naive_train": naive[mask_train],
        "naive_val": naive[mask_val],
        "naive_test": naive[mask_test],
    }


def build_conv_model(window: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window, 1)),
            keras.layers.Conv1D(128, kernel_size=3, activation="relu", padding="causal"),
            keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="causal"),
            keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="causal"),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def evaluate_window(
    track: str,
    series: pd.Series,
    window: int,
    horizon: int,
) -> Optional[Dict[str, float]]:
    supervised = make_supervised(series, window, horizon)
    if supervised is None:
        print(f"[SKIP] {track} window={window}: insufficient history for horizon {horizon}")
        return None

    splits = split_train_val_test(supervised['idx'], supervised["X"], supervised["y"], supervised["naive"])
    if splits is None:
        print(f"[SKIP] {track} window={window}: split yielded fewer than 2 samples per set")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(splits["X_train"])
    X_val_scaled = scaler.transform(splits["X_val"])
    X_test_scaled = scaler.transform(splits["X_test"])

    X_train_seq = X_train_scaled.reshape(-1, window, 1).astype(np.float32)
    X_val_seq = X_val_scaled.reshape(-1, window, 1).astype(np.float32)
    X_test_seq = X_test_scaled.reshape(-1, window, 1).astype(np.float32)
    y_train = splits["y_train"].astype(np.float32)
    y_val = splits["y_val"].astype(np.float32)
    y_test = splits["y_test"].astype(np.float32)

    model = build_conv_model(window)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_mae", patience=15, restore_best_weights=True)
    ]
    start_time = time.time()
    model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=2000,
        batch_size=64,
        verbose=0,
        callbacks=callbacks,
    )
    fit_seconds = time.time() - start_time

    pred_train = model.predict(X_train_seq, verbose=0).ravel()
    pred_val = model.predict(X_val_seq, verbose=0).ravel()
    pred_test = model.predict(X_test_seq, verbose=0).ravel()

    mae_train = mean_absolute_error(splits["y_train"], pred_train)
    mae_val = mean_absolute_error(splits["y_val"], pred_val)
    mae_test = mean_absolute_error(splits["y_test"], pred_test)
    naive_mae_train = mean_absolute_error(splits["y_train"], splits["naive_train"])
    naive_mae_val = mean_absolute_error(splits["y_val"], splits["naive_val"])
    naive_mae_test = mean_absolute_error(splits["y_test"], splits["naive_test"])

    print(
        f"[TRACK {track}] window={window:>3} | samples train/val/test="
        f"{len(splits['X_train'])}/{len(splits['X_val'])}/{len(splits['X_test'])} "
        f"| MAE_val={mae_val:.6f} vs naive {naive_mae_val:.6f} (Δ={naive_mae_val - mae_val:.6f}) "
        f"| MAE_test={mae_test:.6f} vs naive {naive_mae_test:.6f} (Δ={naive_mae_test - mae_test:.6f})"
    )

    return {
        "track": track,
        "frequency": track,
        "window": window,
        "horizon": horizon,
        "train_samples": len(splits["X_train"]),
        "val_samples": len(splits["X_val"]),
        "test_samples": len(splits["X_test"]),
        "mae_train": mae_train,
        "mae_val": mae_val,
        "mae_test": mae_test,
        "naive_mae_train": naive_mae_train,
        "naive_mae_val": naive_mae_val,
        "naive_mae_test": naive_mae_test,
        "mae_gain_val": naive_mae_val - mae_val,
        "mae_gain_test": naive_mae_test - mae_test,
        "fit_seconds": fit_seconds,
    }


def summarize_results(results: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    print("\n================ SUMMARY ================")
    for track in df["track"].unique():
        track_df = df[df["track"] == track].sort_values("mae_val")
        if track_df.empty:
            continue
        best_val = track_df.loc[track_df['mae_val'].idxmin()]
        best_test = track_df.loc[track_df['mae_test'].idxmin()]
        print(
            f"Best VAL {track}: window={int(best_val['window'])} | MAE={best_val['mae_val']:.6f} | "
            f"naive={best_val['naive_mae_val']:.6f} | gain={best_val['mae_gain_val']:.6f}"
        )
        print(
            f"Best TEST {track}: window={int(best_test['window'])} | MAE={best_test['mae_test']:.6f} | "
            f"naive={best_test['naive_mae_test']:.6f} | gain={best_test['mae_gain_test']:.6f}"
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
            result = evaluate_window(track_name, series, window, horizon)
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
