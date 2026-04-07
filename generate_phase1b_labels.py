#!/usr/bin/env python3
"""
Generate labeled datasets for phase_1_b in the predictor repo.

Reads the raw 4h OHLC EURUSD data, runs the oracle label generator on each
phase_1 date range (d1-d6), and saves:
  - base_d{n}.csv         (OHLC + labels)
  - normalization_config_a.json  (label statistics, not needed for binary but kept for convention)
  - normalization_config_b.json

Usage:
    cd /home/harveybc/Documents/GitHub/feature-eng
    PYTHONPATH=./ python generate_phase1b_labels.py
"""

import os
import sys
import json
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---
RAW_OHLC = os.path.join(os.path.dirname(__file__),
    "tests/data/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv")

PHASE1_DIR = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/phase_1")

OUTPUT_DIR = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/phase_1_b")

# Date ranges matching phase_1 base_d*.csv (from predictor repo)
DATE_RANGES = {
    "d1": ("2005-06-22 12:00:00", "2010-05-12 00:00:00"),
    "d2": ("2010-05-12 04:00:00", "2011-07-28 16:00:00"),
    "d3": ("2011-07-28 20:00:00", "2012-10-16 16:00:00"),
    "d4": ("2012-10-16 20:00:00", "2017-09-20 00:00:00"),
    "d5": ("2017-09-20 04:00:00", "2018-12-14 00:00:00"),
    "d6": ("2018-12-14 04:00:00", "2020-04-29 20:00:00"),
}

# Oracle params (4h bars → prediction_horizon adjusted)
# At 4h bars, ~30 bars ≈ 5 trading days (one week)
ORACLE_PARAMS = {
    'tp_pips': 131.325,
    'sl_pips': 93.33,
    'spread_pips': 30.0,
    'commission_per_lot': 10.0,
    'slippage_pips': 10.0,
    'pip_cost': 0.00001,
    'friday_close_hour': 20,
    'prediction_horizon': 30,  # 30 × 4h = 120h ≈ 1 week
}


def load_raw_ohlc(path):
    """Load raw 4h OHLC and normalize column names."""
    print(f"Loading raw 4h OHLC from {path} ...")
    df = pd.read_csv(path)
    # Rename columns: "Gmt time" → DATE_TIME, lowercase → uppercase
    df.rename(columns={
        'Gmt time': 'DATE_TIME',
        'open': 'OPEN',
        'high': 'HIGH',
        'low': 'LOW',
        'close': 'CLOSE',
        'volume': 'VOLUME',
    }, inplace=True)
    # Parse date: DD.MM.YYYY HH:MM:SS.000
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%d.%m.%Y %H:%M:%S.%f')
    df.set_index('DATE_TIME', inplace=True)
    df.sort_index(inplace=True)
    # Keep only OHLC
    df = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].astype(float)
    print(f"  {len(df)} rows: {df.index[0]} → {df.index[-1]}")
    return df


def main():
    ohlc = load_raw_ohlc(RAW_OHLC)

    # Import oracle label generator
    from app.plugins.oracle_labels import Plugin as OraclePlugin

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_stats = {}

    for dname, (start, end) in DATE_RANGES.items():
        print(f"\n--- {dname}: {start} → {end} ---")
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        # Slice OHLC to date range
        # But we need extra future bars beyond end_dt for the oracle scan
        # Add prediction_horizon bars of buffer
        horizon = ORACLE_PARAMS['prediction_horizon']
        # Find the index position for the end
        mask = (ohlc.index >= start_dt) & (ohlc.index <= end_dt)
        subset_indices = ohlc.index[mask]
        if len(subset_indices) == 0:
            print(f"  WARNING: No data found for range {start} → {end}")
            continue

        # Get extended range for oracle scanning (need future bars beyond end)
        end_idx = ohlc.index.get_loc(subset_indices[-1])
        extended_end = min(end_idx + horizon + 1, len(ohlc))
        start_idx = ohlc.index.get_loc(subset_indices[0])

        # Extended OHLC for oracle (includes future buffer)
        ohlc_extended = ohlc.iloc[start_idx:extended_end].copy()
        print(f"  OHLC slice: {len(ohlc_extended)} rows (incl. {extended_end - end_idx - 1} future buffer)")

        # Run oracle on extended data
        oracle = OraclePlugin()
        oracle.set_params(**ORACLE_PARAMS)
        labels = oracle.process(ohlc_extended)

        # Trim back to actual date range (remove future buffer rows)
        actual_len = len(subset_indices)
        ohlc_subset = ohlc_extended.iloc[:actual_len].copy()
        labels_subset = labels.iloc[:actual_len].copy()

        # Merge OHLC + labels
        result = pd.concat([ohlc_subset, labels_subset], axis=1)
        print(f"  Result: {len(result)} rows, columns: {list(result.columns)}")

        # Label stats
        for col in ['buy_entry_label', 'sell_entry_label', 'buy_exit_label', 'sell_exit_label']:
            ones = int(result[col].sum())
            total = len(result)
            pct = ones / total * 100 if total > 0 else 0
            print(f"    {col}: {ones}/{total} ({pct:.1f}%)")

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"base_{dname}.csv")
        result.to_csv(out_path, index=True)
        print(f"  Saved {out_path}")

        # Collect stats for normalization config
        for col in result.columns:
            vals = result[col].dropna()
            if col not in all_stats:
                all_stats[col] = {'values': []}
            all_stats[col]['values'].extend(vals.tolist())

    # Compute normalization configs from d1 (training set)
    print("\n--- Computing normalization configs ---")
    d1_path = os.path.join(OUTPUT_DIR, "base_d1.csv")
    if os.path.exists(d1_path):
        d1 = pd.read_csv(d1_path, index_col=0)
        norm_a = {}
        norm_b = {}
        for col in d1.columns:
            vals = d1[col].dropna()
            norm_a[col] = {"mean": float(vals.mean()), "std": float(vals.std())}
            norm_b[col] = {"min": float(vals.min()), "max": float(vals.max())}

        norm_a_path = os.path.join(OUTPUT_DIR, "normalization_config_a.json")
        with open(norm_a_path, 'w') as f:
            json.dump(norm_a, f, indent=2)
        print(f"  Saved {norm_a_path}")

        norm_b_path = os.path.join(OUTPUT_DIR, "normalization_config_b.json")
        with open(norm_b_path, 'w') as f:
            json.dump(norm_b, f, indent=2)
        print(f"  Saved {norm_b_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
