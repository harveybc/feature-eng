#!/usr/bin/env python3
"""
Transform phase_1_b base datasets into normalized datasets with derived features.

Reads base_d1..d6.csv (OHLC + oracle labels), computes:
  - typical_price = (HIGH + LOW + CLOSE) / 3
  - Cyclic seasonal features: hod_sin/cos, dow_sin/cos, dom_sin/cos, moy_sin/cos
  - Rolling features: rolling_std_24, rolling_ema_24, price_minus_ema
  - Keeps label columns as-is (not normalized)

Then z-score normalizes using:
  - config_a: params computed from d1 only → applied to d1, d2, d3
  - config_b: params computed from d4 only → applied to d4, d5, d6

Outputs:
  - base_d{n}.csv      (updated: typical_price + seasonals + rolling + labels)
  - normalized_d{n}.csv (z-score normalized features + raw labels)
  - normalization_config_a.json
  - normalization_config_b.json

Usage:
    python normalize_phase1b.py
"""

import json
import os
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
import pandas as pd

PHASE1B_DIR = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/phase_1_b")

ROLLING_WINDOW = 24

# Columns that get normalized (features)
FEATURE_COLS = [
    'typical_price',
    'hod_sin', 'hod_cos',
    'dow_sin', 'dow_cos',
    'dom_sin', 'dom_cos',
    'moy_sin', 'moy_cos',
    'rolling_std_24', 'rolling_ema_24', 'price_minus_ema',
]

# Columns that stay raw (targets — not normalized)
LABEL_COLS = [
    'buy_entry_label', 'sell_entry_label',
    'buy_exit_label', 'sell_exit_label',
    'bars_to_friday',
]


def add_derived_features(df):
    """Compute typical_price, seasonal, and rolling features from OHLC + DATE_TIME."""
    dt = pd.to_datetime(df['DATE_TIME'])

    # typical_price = (H + L + C) / 3
    df['typical_price'] = (df['HIGH'] + df['LOW'] + df['CLOSE']) / 3.0

    # Cyclic sinusoidal encoding (matching preprocessor formulas exactly)
    df['hod_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hod_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df['dom_sin'] = np.sin(2 * np.pi * (dt.dt.day - 1) / 31)
    df['dom_cos'] = np.cos(2 * np.pi * (dt.dt.day - 1) / 31)
    df['moy_sin'] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
    df['moy_cos'] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)

    # Rolling features from typical_price
    df['rolling_std_24'] = df['typical_price'].rolling(window=ROLLING_WINDOW).std()
    df['rolling_ema_24'] = df['typical_price'].ewm(span=ROLLING_WINDOW, adjust=False).mean()
    df['price_minus_ema'] = df['typical_price'] - df['rolling_ema_24']

    return df


def compute_norm_params(df, feature_cols):
    """Compute z-score parameters (mean, std) from a training dataset."""
    params = {}
    for col in feature_cols:
        vals = df[col].dropna()
        params[col] = {
            'mean': float(vals.mean()),
            'std': float(vals.std()),
        }
    return params


def apply_normalization(df, norm_params, feature_cols):
    """Apply z-score normalization to feature columns, leave labels raw."""
    normalized = df.copy()
    for col in feature_cols:
        if col in norm_params and col in normalized.columns:
            mean = norm_params[col]['mean']
            std = norm_params[col]['std']
            if std > 0:
                normalized[col] = (normalized[col] - mean) / std
            else:
                normalized[col] = 0.0
    return normalized


def main():
    datasets = {}

    # --- Phase 1: Load base OHLC+labels, compute derived features ---
    print("=== Phase 1: Computing derived features ===")
    for i in range(1, 7):
        dname = f"d{i}"
        path = os.path.join(PHASE1B_DIR, f"base_{dname}.csv")
        print(f"\nLoading {path} ...")
        df = pd.read_csv(path)
        print(f"  Raw: {len(df)} rows, columns: {list(df.columns)}")

        # Compute derived features
        df = add_derived_features(df)

        # Drop NaN rows from rolling window (first ROLLING_WINDOW-1 rows)
        n_before = len(df)
        df = df.dropna(subset=['rolling_std_24']).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Trimmed {n_dropped} NaN rows from rolling window")

        datasets[dname] = df
        print(f"  Final: {len(df)} rows")

    # --- Phase 2: Save updated base files (features + labels, no OHLC) ---
    print("\n=== Phase 2: Saving updated base files ===")
    output_cols = ['DATE_TIME'] + FEATURE_COLS + LABEL_COLS
    for dname, df in datasets.items():
        out_path = os.path.join(PHASE1B_DIR, f"base_{dname}.csv")
        df[output_cols].to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df)} rows)")

    # --- Phase 3: Compute normalization parameters from training sets ---
    print("\n=== Phase 3: Computing normalization parameters ===")
    norm_a = compute_norm_params(datasets['d1'], FEATURE_COLS)
    norm_b = compute_norm_params(datasets['d4'], FEATURE_COLS)

    norm_a_path = os.path.join(PHASE1B_DIR, "normalization_config_a.json")
    with open(norm_a_path, 'w') as f:
        json.dump(norm_a, f, indent=2)
    print(f"  Saved {norm_a_path} (from d1, {len(norm_a)} features)")

    norm_b_path = os.path.join(PHASE1B_DIR, "normalization_config_b.json")
    with open(norm_b_path, 'w') as f:
        json.dump(norm_b, f, indent=2)
    print(f"  Saved {norm_b_path} (from d4, {len(norm_b)} features)")

    # --- Phase 4: Normalize and save ---
    print("\n=== Phase 4: Applying normalization ===")
    config_map = {
        'd1': norm_a, 'd2': norm_a, 'd3': norm_a,
        'd4': norm_b, 'd5': norm_b, 'd6': norm_b,
    }

    for dname, df in datasets.items():
        norm_params = config_map[dname]
        normalized = apply_normalization(df, norm_params, FEATURE_COLS)

        out_path = os.path.join(PHASE1B_DIR, f"normalized_{dname}.csv")
        normalized[output_cols].to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(normalized)} rows)")

        # Quick sanity check on normalized features
        for col in ['typical_price', 'rolling_std_24']:
            vals = normalized[col].dropna()
            print(f"    {col}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
