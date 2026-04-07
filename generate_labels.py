#!/usr/bin/env python3
"""
Generate labeled datasets (d1=train, d2=validation, d3=test) for binary
entry/exit predictors.

This script:
  1. Loads the full OHLC CSV (phase_2_3_base_d3.csv)
  2. Runs the oracle label generator to produce binary labels
  3. Runs the tech_indicator plugin to produce features
  4. Merges features + labels
  5. Splits chronologically into d1 (60%), d2 (20%), d3 (20%)
  6. Saves to CSV files for predictor training

Usage:
    cd /home/harveybc/Documents/GitHub/feature-eng
    PYTHONPATH=./ python generate_labels.py \\
        --input_file ../heuristic-strategy/tests/data/phase_2_3_base_d3.csv \\
        --output_dir ../prediction_provider/data/labeled

Output files:
    labeled_d1.csv  (training   — ~2017-03 to 2018-10)
    labeled_d2.csv  (validation — ~2018-10 to 2019-07)
    labeled_d3.csv  (test       — ~2019-07 to 2020-03)
"""

import argparse
import os
import sys
import numpy as np
# Fix numpy.NaN deprecation for pandas_ta compatibility
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
import pandas as pd

# Allow importing from feature-eng app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Generate labeled d1/d2/d3 datasets")
    parser.add_argument("--input_file", required=True,
                        help="Path to OHLC CSV (e.g. phase_2_3_base_d3.csv)")
    parser.add_argument("--output_dir", default="data/labeled",
                        help="Directory for output CSVs")
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    # Oracle label params (defaults match strategy worst-case)
    parser.add_argument("--tp_pips", type=float, default=131.325)
    parser.add_argument("--sl_pips", type=float, default=93.33)
    parser.add_argument("--spread_pips", type=float, default=30.0)
    parser.add_argument("--commission_per_lot", type=float, default=10.0)
    parser.add_argument("--slippage_pips", type=float, default=10.0)

    args = parser.parse_args()

    # ── 1. Load OHLC data ──
    print(f"Loading OHLC data from {args.input_file} ...")
    df = pd.read_csv(args.input_file)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df.set_index('DATE_TIME', inplace=True)
    df = df.sort_index()

    # Ensure float columns
    for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE']:
        df[col] = df[col].astype(float)

    print(f"  Loaded {len(df)} rows: {df.index[0]} → {df.index[-1]}")

    # ── 2. Generate technical indicator features ──
    print("Generating technical indicator features ...")
    from app.plugins.tech_indicator import Plugin as TechPlugin
    tech = TechPlugin()
    features = tech.process(df.copy())
    print(f"  Generated {features.shape[1]} feature columns")

    # ── 3. Generate oracle labels ──
    print("Generating oracle labels ...")
    from app.plugins.oracle_labels import Plugin as OraclePlugin
    oracle = OraclePlugin()
    oracle.set_params(
        tp_pips=args.tp_pips,
        sl_pips=args.sl_pips,
        spread_pips=args.spread_pips,
        commission_per_lot=args.commission_per_lot,
        slippage_pips=args.slippage_pips,
    )
    labels = oracle.process(df)
    print(f"  Generated label columns: {list(labels.columns)}")

    # ── 4. Merge OHLC + features + labels ──
    merged = pd.concat([df, features, labels], axis=1)

    # Drop rows with NaN from indicator warm-up period
    initial_len = len(merged)
    merged.dropna(inplace=True)
    print(f"  Dropped {initial_len - len(merged)} NaN rows (indicator warm-up)")
    print(f"  Final dataset: {len(merged)} rows")

    # Print label distribution
    for col in ['buy_entry_label', 'sell_entry_label', 'buy_exit_label', 'sell_exit_label']:
        ones = merged[col].sum()
        total = len(merged)
        print(f"  {col}: {ones}/{total} positives ({ones/total*100:.1f}%)")

    # ── 5. Chronological split ──
    n = len(merged)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))

    d1 = merged.iloc[:train_end]
    d2 = merged.iloc[train_end:val_end]
    d3 = merged.iloc[val_end:]

    print(f"\n  d1 (train):      {len(d1)} rows  {d1.index[0]} → {d1.index[-1]}")
    print(f"  d2 (validation): {len(d2)} rows  {d2.index[0]} → {d2.index[-1]}")
    print(f"  d3 (test):       {len(d3)} rows  {d3.index[0]} → {d3.index[-1]}")

    # ── 6. Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    for name, split in [("labeled_d1.csv", d1), ("labeled_d2.csv", d2), ("labeled_d3.csv", d3)]:
        path = os.path.join(args.output_dir, name)
        split.to_csv(path, index=True)
        print(f"  Saved {path} ({len(split)} rows)")

    # Also save the base OHLC splits (without features, for oracle baseline runs)
    for name, split in [("base_d1.csv", d1), ("base_d2.csv", d2), ("base_d3.csv", d3)]:
        path = os.path.join(args.output_dir, name)
        split[['OPEN', 'HIGH', 'LOW', 'CLOSE']].to_csv(path, index=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
