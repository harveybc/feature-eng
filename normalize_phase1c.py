#!/usr/bin/env python3
"""
Transform phase_1_c base datasets into normalized datasets with derived features.

Reads base_d1..d6.csv (OHLC + direction labels), computes:
  - Technical indicators: ATR, RSI, MACD(3), ADX(3), Stochastic(2), CCI,
    Williams %R, ROC, Bollinger Band Width
  - Derived features: ATR_ratio, BB_position
  - Rolling features: rolling_std_24, price_minus_ema
  - Cyclic seasonal: hod_sin/cos, dow_sin/cos
  - Keeps label columns as-is (not normalized)

Then z-score normalizes using:
  - config_a: params computed from d1 only → applied to d1, d2, d3
  - config_c: params computed from d4 only → applied to d4, d5, d6

Outputs:
  - base_d{n}.csv      (updated: indicators + derived + labels)
  - normalized_d{n}.csv (z-score normalized features + raw labels)
  - normalization_config_a.json
  - normalization_config_c.json

Usage:
    cd /home/harveybc/Documents/GitHub/feature-eng
    python normalize_phase1c.py
"""

import json
import os
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta

PHASE1C_DIR = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/phase_1_c")

ROLLING_WINDOW = 24

# Oracle TP/SL params (for ATR_ratio computation — matches direction oracle)
# ATR * tp_mult = TP distance in price; ATR_ratio = tp_mult (constant)
# We keep TP_PIPS for consistency with phase_1b but compute from ATR
TP_PIPS = 131.325
PIP_COST = 0.00001

# Columns that get normalized (features) — same comprehensive set as phase_1b
FEATURE_COLS = [
    'ATR', 'RSI',
    'MACD', 'MACD_Histogram', 'MACD_Signal',
    'ADX', 'DI_plus', 'DI_minus',
    'Stochastic_K', 'Stochastic_D',
    'BB_Width', 'CCI', 'WilliamsR', 'ROC',
    'ATR_ratio', 'BB_position',
    'rolling_std_24', 'price_minus_ema',
    'hod_sin', 'hod_cos', 'dow_sin', 'dow_cos',
]

# Columns that stay raw (targets — not normalized)
LABEL_COLS = [
    'direction_long_label', 'direction_short_label',
    'bars_to_friday',
]


def add_derived_features(df):
    """Compute technical indicators and derived features from OHLC + DATE_TIME."""
    dt = pd.to_datetime(df['DATE_TIME'])
    high = df['HIGH'].astype(float)
    low = df['LOW'].astype(float)
    close = df['CLOSE'].astype(float)

    # === Tier 1: Technical Indicators ===
    df['ATR'] = ta.atr(high, low, close, length=14)
    df['RSI'] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Histogram'] = macd['MACDh_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']

    adx = ta.adx(high, low, close, length=14)
    df['ADX'] = adx['ADX_14']
    df['DI_plus'] = adx['DMP_14']
    df['DI_minus'] = adx['DMN_14']

    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    df['Stochastic_K'] = stoch['STOCHk_14_3_3']
    df['Stochastic_D'] = stoch['STOCHd_14_3_3']

    # === Tier 2 ===
    bbands = ta.bbands(close, length=20, std=2.0)
    bb_upper = bbands['BBU_20_2.0']
    bb_lower = bbands['BBL_20_2.0']
    df['BB_Width'] = bb_upper - bb_lower

    df['CCI'] = ta.cci(high, low, close, length=20)
    df['WilliamsR'] = ta.willr(high, low, close, length=14)
    df['ROC'] = ta.roc(close, length=10)

    # === Tier 3: Derived ===
    atr_in_pips = df['ATR'] / PIP_COST
    df['ATR_ratio'] = TP_PIPS / atr_in_pips.replace(0, np.nan)

    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df['BB_position'] = (close - bb_lower) / bb_range

    # === Rolling / Price-based ===
    typical_price = (high + low + close) / 3.0
    df['rolling_std_24'] = typical_price.rolling(window=ROLLING_WINDOW).std()
    rolling_ema_24 = typical_price.ewm(span=ROLLING_WINDOW, adjust=False).mean()
    df['price_minus_ema'] = typical_price - rolling_ema_24

    # === Cyclic Time Features ===
    df['hod_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hod_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # === bars_to_friday ===
    dow = dt.dt.dayofweek
    hour = dt.dt.hour
    # Approximate: bars remaining until Friday 20:00 UTC (4h bars)
    bars_per_day = 6  # 24h / 4h
    bars_to_fri = np.zeros(len(df))
    for i in range(len(df)):
        d = dow.iloc[i]
        h = hour.iloc[i]
        if d < 4:  # Mon=0..Thu=3
            days_left = 4 - d
            bars_left = days_left * bars_per_day + max(0, (20 - h) // 4)
        elif d == 4:  # Friday
            bars_left = max(0, (20 - h) // 4)
        else:  # Sat/Sun
            bars_left = 0
        bars_to_fri[i] = bars_left
    df['bars_to_friday'] = bars_to_fri

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
    print("=== Phase 1: Computing derived features (technical indicators) ===")
    for i in range(1, 7):
        dname = f"d{i}"
        path = os.path.join(PHASE1C_DIR, f"base_{dname}.csv")
        print(f"\nLoading {path} ...")
        df = pd.read_csv(path)
        print(f"  Raw: {len(df)} rows, columns: {list(df.columns)}")

        # Compute derived features (tech indicators + rolling + cyclic)
        df = add_derived_features(df)

        # Drop NaN rows from indicator warmup
        n_before = len(df)
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Trimmed {n_dropped} NaN rows from indicator warmup")

        datasets[dname] = df
        print(f"  Final: {len(df)} rows, {len(FEATURE_COLS)} features")

        # Label stats
        for col in ['direction_long_label', 'direction_short_label']:
            if col in df.columns:
                ones = int(df[col].sum())
                total = len(df)
                pct = ones / total * 100 if total > 0 else 0
                print(f"    {col}: {ones}/{total} ({pct:.1f}%)")

    # --- Phase 2: Save updated base files ---
    print("\n=== Phase 2: Saving updated base files ===")
    output_cols = ['DATE_TIME'] + FEATURE_COLS + LABEL_COLS
    for dname, df in datasets.items():
        out_path = os.path.join(PHASE1C_DIR, f"base_{dname}.csv")
        df[output_cols].to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(df)} rows)")

    # --- Phase 3: Compute normalization parameters ---
    print("\n=== Phase 3: Computing normalization parameters ===")
    norm_a = compute_norm_params(datasets['d1'], FEATURE_COLS)
    norm_c = compute_norm_params(datasets['d4'], FEATURE_COLS)

    norm_a_path = os.path.join(PHASE1C_DIR, "normalization_config_a.json")
    with open(norm_a_path, 'w') as f:
        json.dump(norm_a, f, indent=2)
    print(f"  Saved {norm_a_path} (from d1, {len(norm_a)} features)")

    norm_c_path = os.path.join(PHASE1C_DIR, "normalization_config_c.json")
    with open(norm_c_path, 'w') as f:
        json.dump(norm_c, f, indent=2)
    print(f"  Saved {norm_c_path} (from d4, {len(norm_c)} features)")

    # --- Phase 4: Normalize and save ---
    print("\n=== Phase 4: Applying normalization ===")
    config_map = {
        'd1': norm_a, 'd2': norm_a, 'd3': norm_a,
        'd4': norm_c, 'd5': norm_c, 'd6': norm_c,
    }

    for dname, df in datasets.items():
        norm_params = config_map[dname]
        normalized = apply_normalization(df, norm_params, FEATURE_COLS)

        out_path = os.path.join(PHASE1C_DIR, f"normalized_{dname}.csv")
        normalized[output_cols].to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({len(normalized)} rows)")

        # Quick sanity check
        for col in ['ATR', 'RSI', 'MACD', 'ADX', 'rolling_std_24']:
            vals = normalized[col].dropna()
            print(f"    {col}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
