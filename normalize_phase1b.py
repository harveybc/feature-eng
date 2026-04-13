#!/usr/bin/env python3
"""
Transform phase_1_b base datasets into normalized datasets with derived features.

Reads base_d1..d6.csv (OHLC + oracle labels), computes:
  - Technical indicators: ATR, RSI, MACD(3), ADX(3), Stochastic(2), CCI,
    Williams %R, ROC, Bollinger Band Width
  - Derived features: ATR_ratio (TP feasibility), BB_position
  - Rolling features: rolling_std_24, price_minus_ema
  - Cyclic seasonal: hod_sin/cos, dow_sin/cos
  - Keeps label columns as-is (not normalized)

Then z-score normalizes using:
  - config_a: params computed from d1 only → applied to d1, d2, d3
  - config_b: params computed from d4 only → applied to d4, d5, d6

Outputs:
  - base_d{n}.csv      (updated: indicators + derived + labels)
  - normalized_d{n}.csv (z-score normalized features + raw labels)
  - normalization_config_a.json
  - normalization_config_b.json

Usage:
    cd /home/harveybc/Documents/GitHub/feature-eng
    python normalize_phase1b.py
"""

import json
import os
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta

PHASE1B_DIR = os.path.expanduser(
    "~/Documents/GitHub/predictor/examples/data_downsampled/phase_1_b")

ROLLING_WINDOW = 24

# Oracle TP/SL params (for ATR_ratio computation)
TP_PIPS = 131.325
PIP_COST = 0.00001

# Columns that get normalized (features) — new comprehensive set
FEATURE_COLS = [
    # Tier 1: directly relevant to TP/SL prediction
    'ATR',                     # Average True Range (volatility)
    'RSI',                     # Relative Strength Index (momentum)
    'MACD', 'MACD_Histogram', 'MACD_Signal',  # Trend momentum
    'ADX', 'DI_plus', 'DI_minus',             # Trend strength
    'Stochastic_K', 'Stochastic_D',           # Momentum oscillator
    # Tier 2: complementary
    'BB_Width',                # Bollinger Band width (volatility regime)
    'CCI',                     # Commodity Channel Index
    'WilliamsR',               # Overbought/oversold
    'ROC',                     # Rate of Change (momentum)
    # Tier 3: derived (highly targeted)
    'ATR_ratio',               # TP_pips / ATR — TP feasibility
    'BB_position',             # Price position within Bollinger Bands
    # Rolling / price-based
    'rolling_std_24',          # 24-bar rolling volatility
    'price_minus_ema',         # Distance from EMA (mean reversion)
    # Cyclic time features (keep only hour and day-of-week)
    'hod_sin', 'hod_cos',
    'dow_sin', 'dow_cos',
]

# Columns that stay raw (targets — not normalized)
LABEL_COLS = [
    'buy_entry_label', 'sell_entry_label',
    'buy_exit_label', 'sell_exit_label',
    'bars_to_friday',
]


def add_derived_features(df):
    """Compute technical indicators and derived features from OHLC + DATE_TIME."""
    dt = pd.to_datetime(df['DATE_TIME'])
    high = df['HIGH'].astype(float)
    low = df['LOW'].astype(float)
    close = df['CLOSE'].astype(float)

    # === Tier 1: Technical Indicators ===

    # ATR (14-period) — volatility, most critical for TP/SL feasibility
    df['ATR'] = ta.atr(high, low, close, length=14)

    # RSI (14-period) — momentum oscillator
    df['RSI'] = ta.rsi(close, length=14)

    # MACD (12, 26, 9) — trend momentum
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Histogram'] = macd['MACDh_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']

    # ADX (14-period) + DI+/DI- — trend strength and direction
    adx = ta.adx(high, low, close, length=14)
    df['ADX'] = adx['ADX_14']
    df['DI_plus'] = adx['DMP_14']
    df['DI_minus'] = adx['DMN_14']

    # Stochastic (14, 3, 3) — momentum oscillator
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    df['Stochastic_K'] = stoch['STOCHk_14_3_3']
    df['Stochastic_D'] = stoch['STOCHd_14_3_3']

    # === Tier 2: Complementary Indicators ===

    # Bollinger Bands (20, 2σ) — volatility envelope
    bbands = ta.bbands(close, length=20, std=2.0)
    bb_upper = bbands['BBU_20_2.0']
    bb_lower = bbands['BBL_20_2.0']
    df['BB_Width'] = bb_upper - bb_lower

    # CCI (20-period) — momentum with different mathematical basis
    df['CCI'] = ta.cci(high, low, close, length=20)

    # Williams %R (14-period) — fast overbought/oversold
    df['WilliamsR'] = ta.willr(high, low, close, length=14)

    # ROC (10-period) — rate of change
    df['ROC'] = ta.roc(close, length=10)

    # === Tier 3: Derived Features ===

    # ATR_ratio = TP_pips / (ATR / pip_cost) — measures TP feasibility
    # Values < 1 mean TP is within a single ATR (very reachable)
    atr_in_pips = df['ATR'] / PIP_COST
    df['ATR_ratio'] = TP_PIPS / atr_in_pips.replace(0, np.nan)

    # BB_position = (price - BB_Lower) / (BB_Upper - BB_Lower)
    # 0 = at lower band (oversold), 1 = at upper band (overbought)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df['BB_position'] = (close - bb_lower) / bb_range

    # === Rolling / Price-based ===
    typical_price = (high + low + close) / 3.0
    df['rolling_std_24'] = typical_price.rolling(window=ROLLING_WINDOW).std()
    rolling_ema_24 = typical_price.ewm(span=ROLLING_WINDOW, adjust=False).mean()
    df['price_minus_ema'] = typical_price - rolling_ema_24

    # === Cyclic Time Features (keep only hour and day-of-week) ===
    df['hod_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hod_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

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
        path = os.path.join(PHASE1B_DIR, f"base_{dname}.csv")
        print(f"\nLoading {path} ...")
        df = pd.read_csv(path)
        print(f"  Raw: {len(df)} rows, columns: {list(df.columns)}")

        # Compute derived features (tech indicators + rolling + cyclic)
        df = add_derived_features(df)

        # Drop NaN rows from indicator warmup (MACD needs 26, BBands 20, etc.)
        n_before = len(df)
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Trimmed {n_dropped} NaN rows from indicator warmup")

        datasets[dname] = df
        print(f"  Final: {len(df)} rows, {len(FEATURE_COLS)} features")

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
        for col in ['ATR', 'RSI', 'MACD', 'ADX', 'rolling_std_24']:
            vals = normalized[col].dropna()
            print(f"    {col}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
