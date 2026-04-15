#!/usr/bin/env python3
"""
Regime Detector Module
======================
Classifies each 4h bar into a market regime based on technical indicators.
Uses thresholds derived from hierarchical clustering of 15 years of EURUSD data.

Regimes (from K=6 clustering):
  1 - HIGH_VOL_BEARISH_FADING:  High ATR, bearish DI, oversold → reversal zone
  2 - STRONG_DOWNTREND:         High ADX, very bearish DI, low RSI → persistent decline
  3 - STRONG_UPTREND:           High ADX, very bullish DI, high RSI → persistent rally
  4 - MILD_BULLISH_RANGE:       Low ADX, mild bullish, mid volatility → flat/choppy
  5 - LOW_VOL_BEARISH_PULLBACK: Low ADX, slight bearish in uptrend EMA → mean-revert buy
  6 - LOW_VOL_BULLISH_DRIFT:    Low ADX, mild bullish, low vol → trend-follow buy
"""

import numpy as np
import pandas as pd


def compute_regime_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime indicator features from OHLC DataFrame.
    
    Parameters
    ----------
    ohlc_df : pd.DataFrame
        Must have columns: open, high, low, close (lowercase)
        with a DatetimeIndex.
    
    Returns
    -------
    pd.DataFrame with regime feature columns, index aligned to ohlc_df.
    """
    c = ohlc_df['close']
    h = ohlc_df['high']
    l = ohlc_df['low']

    # --- ATR (14 bars) ---
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    atr_60 = tr.rolling(60).mean()

    # ATR percentile (rolling 120 bars ~ 20 days)
    atr_pct = atr_14.rolling(120).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    atr_ratio = atr_14 / (atr_60 + 1e-10)

    # --- ADX (14 bars) ---
    plus_dm = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0.0)
    minus_dm = minus_dm.where(~mask, 0.0)
    atr_s = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_s)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=14, adjust=False).mean()
    di_spread = plus_di - minus_di

    # --- RSI (14 bars) ---
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-10))

    # --- Bollinger Bands (20 bars) ---
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_position = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)
    bb_width_pct = bb_width.rolling(120).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # --- EMAs ---
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    price_vs_ema50 = (c - ema50) / (atr_14 + 1e-10)
    ema_alignment = (ema50 - ema200) / (atr_14 + 1e-10)

    # --- ROC ---
    roc_12 = c.pct_change(12) * 100

    # --- Stochastic ---
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = 100 * (c - low14) / (high14 - low14 + 1e-10)

    # --- MACD histogram (normalized) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    macd_hist_norm = macd_hist / (atr_14 + 1e-10)

    features = pd.DataFrame({
        'adx': adx,
        'di_spread': di_spread,
        'atr_pct': atr_pct,
        'atr_ratio': atr_ratio,
        'bb_width_pct': bb_width_pct,
        'bb_position': bb_position,
        'rsi': rsi,
        'roc_12': roc_12,
        'price_vs_ema50': price_vs_ema50,
        'ema_alignment': ema_alignment,
        'stoch_k': stoch_k,
        'macd_hist': macd_hist_norm,
        # Raw values for strategy use
        'atr_raw': atr_14,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_mid': bb_mid,
        'ema50': ema50,
        'ema200': ema200,
    }, index=ohlc_df.index)

    return features


def classify_regime(features: pd.DataFrame,
                    adx_strong: float = 35.0,
                    adx_mild: float = 25.0,
                    di_strong: float = 15.0,
                    di_mild: float = 5.0,
                    atr_pct_high: float = 0.65,
                    atr_pct_low: float = 0.35,
                    rsi_overbought: float = 65.0,
                    rsi_oversold: float = 40.0) -> pd.Series:
    """
    Classify each bar into a regime based on feature thresholds.
    
    Returns Series of regime labels (int 1-6).
    
    Regime decision tree:
    
    ADX >= adx_strong?
      ├── YES: DI_spread > +di_strong → 3 (STRONG_UPTREND)
      │        DI_spread < -di_strong → 2 (STRONG_DOWNTREND)
      │        else → check ATR
      └── NO: ATR_pct >= atr_pct_high?
              ├── YES: DI_spread < -di_mild → 1 (HIGH_VOL_BEARISH_FADING)
              │        else → 4 (MILD range, high vol)
              └── NO: ATR_pct <= atr_pct_low?
                      ├── YES: DI_spread < -di_mild → 5 (LOW_VOL_BEARISH_PULLBACK)
                      │        DI_spread > +di_mild → 6 (LOW_VOL_BULLISH_DRIFT)
                      │        else → 4 (MILD range)
                      └── NO: 4 (MILD_BULLISH_RANGE / default)
    """
    regime = pd.Series(4, index=features.index, dtype=int)  # default: mild range

    adx = features['adx']
    di = features['di_spread']
    atr = features['atr_pct']
    rsi = features['rsi']

    # Strong trends
    strong_up = (adx >= adx_strong) & (di > di_strong)
    strong_dn = (adx >= adx_strong) & (di < -di_strong)
    regime[strong_up] = 3
    regime[strong_dn] = 2

    # Non-strong ADX, high volatility
    not_strong = ~strong_up & ~strong_dn
    high_vol_bear = not_strong & (atr >= atr_pct_high) & (di < -di_mild)
    regime[high_vol_bear] = 1

    # Non-strong ADX, low volatility
    low_vol = not_strong & (atr <= atr_pct_low)
    low_vol_bear_pull = low_vol & (di < -di_mild)
    low_vol_bull_drift = low_vol & (di > di_mild)
    regime[low_vol_bear_pull] = 5
    regime[low_vol_bull_drift] = 6

    return regime


REGIME_NAMES = {
    1: "HIGH_VOL_BEARISH_FADING",
    2: "STRONG_DOWNTREND",
    3: "STRONG_UPTREND",
    4: "MILD_RANGE",
    5: "LOW_VOL_BEARISH_PULLBACK",
    6: "LOW_VOL_BULLISH_DRIFT",
}

# Strategy actions per regime (default config)
REGIME_ACTIONS = {
    1: "buy_reversal",     # High vol bearish fading → buy reversal (6-bar Sharpe +0.41)
    2: "sell_trend",       # Strong downtrend → sell with trend (Sharpe -0.85 going long)
    3: "sell_exhaustion",  # Strong uptrend exhausting → sell (6-bar Sharpe -0.32 going long)
    4: "flat",             # Mild range → no trade (Sharpe ~0)
    5: "buy_meanrevert",   # Low vol bearish pullback in uptrend → buy (Sharpe +0.54)
    6: "buy_trend",        # Low vol bullish drift → buy (Sharpe +0.57)
}


if __name__ == "__main__":
    # Quick test with EURUSD data
    import sys
    ohlc_file = sys.argv[1] if len(sys.argv) > 1 else "tests/data/eurusd_hour_2005_2020_ohlc.csv"
    
    df = pd.read_csv(ohlc_file)
    # Handle column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ('datetime', 'date_time', 'date'):
            col_map[col] = 'datetime'
        elif cl in ('open',):
            col_map[col] = 'open'
        elif cl in ('high',):
            col_map[col] = 'high'
        elif cl in ('low',):
            col_map[col] = 'low'
        elif cl in ('close',):
            col_map[col] = 'close'
    df.rename(columns=col_map, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    
    # Resample to 4h if hourly
    td = df.index[1] - df.index[0]
    if td.total_seconds() < 14400:
        print(f"Resampling from {td} to 4h...")
        df = df.resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

    features = compute_regime_features(df)
    regimes = classify_regime(features)
    features['regime'] = regimes
    
    valid = features.dropna(subset=['adx'])
    print(f"\nRegime distribution ({len(valid)} bars):")
    for r in sorted(valid['regime'].unique()):
        n = (valid['regime'] == r).sum()
        print(f"  {r} ({REGIME_NAMES.get(r, '?')}): {n} bars ({n/len(valid)*100:.1f}%)"
              f" → action: {REGIME_ACTIONS.get(r, '?')}")
    
    # Save
    out = "regime_classified.csv"
    features.to_csv(out)
    print(f"\nSaved: {out}")
