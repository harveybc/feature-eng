#!/usr/bin/env python3
"""
Generate labeled datasets for phase_1_c (direction classification) in the
predictor repo.

Reads the raw 4h OHLC EURUSD data, runs the direction label generator on
each phase_1 date range (d1-d6), and saves:
  - base_d{n}.csv   (OHLC + direction labels)
  - normalization_config_a.json  (z-score stats from d1)
  - normalization_config_b.json  (min/max stats from d1)

Usage:
    cd /home/harveybc/Documents/GitHub/feature-eng
    PYTHONPATH=./ python generate_phase1c_labels.py
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

# Where the existing phase_1 base files live (for reference)
PHASE1_DIR = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/phase_1")

OUTPUT_DIR = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/phase_1_c")

# Date ranges matching phase_1 base_d*.csv (from predictor repo)
DATE_RANGES = {
    "d1": ("2005-06-22 12:00:00", "2010-05-12 00:00:00"),
    "d2": ("2010-05-12 04:00:00", "2011-07-28 16:00:00"),
    "d3": ("2011-07-28 20:00:00", "2012-10-16 16:00:00"),
    "d4": ("2012-10-16 20:00:00", "2017-09-20 00:00:00"),
    "d5": ("2017-09-20 04:00:00", "2018-12-14 00:00:00"),
    "d6": ("2018-12-14 04:00:00", "2020-04-29 20:00:00"),
}

# Direction label params (ATR-based path-scanning, matching ideal oracle)
DIRECTION_PARAMS = {
    'atr_period': 14,
    'tp_mult': 2.0,            # TP distance = ATR * 2.0
    'sl_mult': 1.0,            # SL distance = ATR * 1.0
    'spread_pips': 15.0,       # spread in pipettes
    'commission_per_lot': 7.0, # USD per lot
    'slippage_pips': 5.0,      # slippage in pipettes
    'pip_cost': 0.00001,
    'prediction_horizon': 120, # max bars to scan
    'friday_close_hour': 20,
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

    # Import direction label generator
    from app.plugins.direction_labels import Plugin as DirectionPlugin

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    max_horizon = DIRECTION_PARAMS['prediction_horizon']

    for dname, (start, end) in DATE_RANGES.items():
        print(f"\n--- {dname}: {start} → {end} ---")
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        # Slice OHLC to date range
        # Need extra future bars beyond end_dt for label generation
        mask = (ohlc.index >= start_dt) & (ohlc.index <= end_dt)
        subset_indices = ohlc.index[mask]
        if len(subset_indices) == 0:
            print(f"  WARNING: No data found for range {start} → {end}")
            continue

        # Get extended range (need future bars beyond end)
        end_idx = ohlc.index.get_loc(subset_indices[-1])
        extended_end = min(end_idx + max_horizon + 1, len(ohlc))
        start_idx = ohlc.index.get_loc(subset_indices[0])

        # Extended OHLC for label generation (includes future buffer)
        ohlc_extended = ohlc.iloc[start_idx:extended_end].copy()
        print(f"  OHLC slice: {len(ohlc_extended)} rows (incl. {extended_end - end_idx - 1} future buffer)")

        # Run direction labels on extended data
        direction = DirectionPlugin()
        direction.set_params(**DIRECTION_PARAMS)
        labels = direction.process(ohlc_extended)

        # Trim back to actual date range (remove future buffer rows)
        actual_len = len(subset_indices)
        ohlc_subset = ohlc_extended.iloc[:actual_len].copy()
        labels_subset = labels.iloc[:actual_len].copy()

        # Drop rows where labels are NaN (last N rows without future data)
        valid_mask = labels_subset['direction_long_label'].notna()
        ohlc_subset = ohlc_subset[valid_mask]
        labels_subset = labels_subset[valid_mask]

        # Merge OHLC + labels
        result = pd.concat([ohlc_subset, labels_subset], axis=1)
        print(f"  Result: {len(result)} rows, columns: {list(result.columns)}")

        # Label stats
        for col in ['direction_long_label', 'direction_short_label']:
            ones = int(result[col].sum())
            total = len(result)
            pct = ones / total * 100 if total > 0 else 0
            print(f"    {col}: {ones}/{total} ({pct:.1f}%)")

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"base_{dname}.csv")
        result.to_csv(out_path, index=True)
        print(f"  Saved {out_path}")

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
