#!/usr/bin/env python3
"""
Regime Detector Module
======================
Classifies each 4h bar into a market regime based on technical indicators.

V2 (Causal-Evidence Based):
  Uses bb_position (CORE, causal score=5), atr_ratio (CORE, score=4),
  and ema_alignment (LEADING, only positive Transfer Entropy) as primary
  classification features — validated by ICP, DoWhy refutation, Causal Forest,
  and Transfer Entropy analysis on 15yr EURUSD data.

Regimes (V2 — causal evidence):
  1 - VOLATILE_OVERSOLD:      BB low + high ATR ratio → buy reversal
  2 - BEARISH_CONTINUATION:   BB low + low ATR ratio + bearish EMA → flat/sell
  3 - VOLATILE_OVERBOUGHT:    BB high + high ATR ratio → flat (exhaustion)
  4 - NEUTRAL:                BB mid or no clear signal → flat
  5 - PULLBACK_IN_UPTREND:    BB low + low ATR ratio + bullish EMA → buy mean-revert
  6 - BULLISH_DRIFT:          BB high + low ATR ratio + bullish EMA → buy trend

Legacy V1 (Cluster-Based):
  Uses adx, di_spread, atr_pct thresholds from hierarchical clustering.
  Kept for backward compatibility.
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


def classify_regime_v2(features: pd.DataFrame,
                       bb_low: float = 0.25,
                       bb_high: float = 0.75,
                       atr_ratio_high: float = 1.2,
                       ema_align_thresh: float = 0.0) -> pd.Series:
    """
    Classify each bar into a regime based on causal-evidence features.
    
    Uses bb_position (CORE), atr_ratio (CORE), and ema_alignment (LEADING)
    as validated by causal inference analysis (ICP, DoWhy, Transfer Entropy).
    
    Returns Series of regime labels (int 1-6).
    
    Decision tree:
    
    bb_position < bb_low?  (price near lower BB)
      ├── YES: atr_ratio > atr_ratio_high?
      │        ├── YES → 1 (VOLATILE_OVERSOLD → buy reversal)
      │        └── NO:  ema_alignment > ema_align_thresh?
      │                 ├── YES → 5 (PULLBACK_IN_UPTREND → buy mean-revert)
      │                 └── NO  → 2 (BEARISH_CONTINUATION → flat/sell)
      └── NO:  bb_position > bb_high?  (price near upper BB)
               ├── YES: atr_ratio > atr_ratio_high?
               │        ├── YES → 3 (VOLATILE_OVERBOUGHT → flat)
               │        └── NO:  ema_alignment > ema_align_thresh?
               │                 ├── YES → 6 (BULLISH_DRIFT → buy trend)
               │                 └── NO  → 4 (NEUTRAL → flat)
               └── NO:  → 4 (NEUTRAL → flat)
    """
    regime = pd.Series(4, index=features.index, dtype=int)  # default: NEUTRAL

    bb = features['bb_position']
    atr_r = features['atr_ratio']
    ema_a = features['ema_alignment']

    # Lower BB zone
    bb_is_low = bb < bb_low
    regime[bb_is_low & (atr_r > atr_ratio_high)] = 1  # VOLATILE_OVERSOLD
    regime[bb_is_low & (atr_r <= atr_ratio_high) & (ema_a > ema_align_thresh)] = 5  # PULLBACK_IN_UPTREND
    regime[bb_is_low & (atr_r <= atr_ratio_high) & (ema_a <= ema_align_thresh)] = 2  # BEARISH_CONTINUATION

    # Upper BB zone
    bb_is_high = bb > bb_high
    regime[bb_is_high & (atr_r > atr_ratio_high)] = 3  # VOLATILE_OVERBOUGHT
    regime[bb_is_high & (atr_r <= atr_ratio_high) & (ema_a > ema_align_thresh)] = 6  # BULLISH_DRIFT
    # bb_is_high & low atr_r & bearish ema → stays 4 (NEUTRAL)

    return regime


# ─── V3: GMM Cluster-Based Classification ─────────────────────────
# Centroids from K=9 GMM fitted on 15yr EURUSD (24K 4h bars).
# Features: [bb_position, atr_ratio, ema_alignment] (standardised).
# Scaler params from StandardScaler fit on same data.

_GMM_SCALER_MEAN = np.array([0.4920, 1.0196, 0.0049])
_GMM_SCALER_SCALE = np.array([0.2780, 0.1823, 2.9037])

# Raw centroids (unscaled) for each of the 9 GMM clusters
_GMM_CENTROIDS_RAW = np.array([
    [0.784, 0.996, 3.494],   # C0: overbought + bullish EMA
    [0.224, 0.937, -1.501],  # C1: low BB + bearish
    [0.757, 0.846, -0.736],  # C2: overbought + mild bearish
    [0.797, 1.079, -1.564],  # C3: overbought + vol + bearish
    [0.381, 0.872, -5.722],  # C4: mid BB + deeply bearish
    [0.160, 1.133, 1.921],   # C5: oversold + bullish EMA
    [0.823, 1.363, 1.323],   # C6: overbought + high vol
    [0.238, 1.230, -3.610],  # C7: oversold + high vol + bearish → BEST BUY
    [0.277, 0.877, 3.004],   # C8: low BB + bullish EMA
])

# Map GMM cluster → regime label (1-based)
# Based on forward-return analysis from clustering study:
#   Only C7 is actionable BUY (Sharpe=+0.085, WR=54.6%)
#   C4 has SELL signal but we know sell doesn't work
_GMM_CLUSTER_TO_REGIME = {
    0: 4,   # NEUTRAL (overbought + bullish → no edge)
    1: 2,   # BEARISH_CONTINUATION (low BB + bearish)
    2: 4,   # NEUTRAL (overbought + mild bearish)
    3: 3,   # VOLATILE_OVERBOUGHT (high BB + vol + bearish)
    4: 4,   # NEUTRAL (mid BB + deeply bearish → sell edge but not trading)
    5: 5,   # PULLBACK_IN_UPTREND (oversold + bullish EMA)
    6: 3,   # VOLATILE_OVERBOUGHT (very high BB + very high vol)
    7: 1,   # VOLATILE_OVERSOLD → BUY (oversold + high vol + bearish = reversal)
    8: 6,   # BULLISH_DRIFT (low BB + bullish EMA)
}

def classify_regime_v3(features: pd.DataFrame) -> pd.Series:
    """
    Classify each bar using GMM cluster centroids (nearest centroid).
    
    V3 uses data-driven cluster boundaries from 15yr EURUSD GMM analysis
    instead of hand-tuned or GA-optimized thresholds.
    
    No optimizable parameters — regime boundaries are fixed from unsupervised
    clustering on the full historical dataset.
    
    Returns Series of regime labels (int 1-6).
    """
    X_raw = features[['bb_position', 'atr_ratio', 'ema_alignment']].values
    
    # Standardise using the same scaler as the GMM was fitted with
    X_scaled = (X_raw - _GMM_SCALER_MEAN) / (_GMM_SCALER_SCALE + 1e-10)
    
    # Scaled centroids
    centroids_scaled = (_GMM_CENTROIDS_RAW - _GMM_SCALER_MEAN) / (_GMM_SCALER_SCALE + 1e-10)
    
    # Nearest centroid assignment (Euclidean in scaled space)
    # Shape: (n_bars, 9)
    dists = np.sqrt(((X_scaled[:, None, :] - centroids_scaled[None, :, :]) ** 2).sum(axis=2))
    cluster_labels = dists.argmin(axis=1)
    
    # Map cluster → regime
    regime_labels = np.array([_GMM_CLUSTER_TO_REGIME[c] for c in cluster_labels])
    
    return pd.Series(regime_labels, index=features.index, dtype=int)


REGIME_NAMES = {
    1: "VOLATILE_OVERSOLD",
    2: "BEARISH_CONTINUATION",
    3: "VOLATILE_OVERBOUGHT",
    4: "NEUTRAL",
    5: "PULLBACK_IN_UPTREND",
    6: "BULLISH_DRIFT",
}

# Legacy V1 names (for backward compatibility)
REGIME_NAMES_V1 = {
    1: "HIGH_VOL_BEARISH_FADING",
    2: "STRONG_DOWNTREND",
    3: "STRONG_UPTREND",
    4: "MILD_RANGE",
    5: "LOW_VOL_BEARISH_PULLBACK",
    6: "LOW_VOL_BULLISH_DRIFT",
}

# Strategy actions per regime (V2 causal-evidence based)
REGIME_ACTIONS = {
    1: "buy_reversal",     # Volatile oversold → buy reversal (bb_position low + high atr_ratio)
    2: "flat",             # Bearish continuation → no trade (sell edge weak per causal analysis)
    3: "flat",             # Volatile overbought → no trade (exhaustion zone)
    4: "flat",             # Neutral → no trade
    5: "buy_meanrevert",   # Pullback in uptrend → buy (bb low + bullish EMA alignment)
    6: "buy_trend",        # Bullish drift → buy trend (bb high + bullish EMA + low vol)
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
    regimes = classify_regime_v3(features)
    features['regime'] = regimes
    
    valid = features.dropna(subset=['bb_position'])
    print(f"\nRegime distribution ({len(valid)} bars):")
    for r in sorted(valid['regime'].unique()):
        n = (valid['regime'] == r).sum()
        print(f"  {r} ({REGIME_NAMES.get(r, '?')}): {n} bars ({n/len(valid)*100:.1f}%)"
              f" → action: {REGIME_ACTIONS.get(r, '?')}")
    
    # Save
    out = "regime_classified.csv"
    features.to_csv(out)
    print(f"\nSaved: {out}")
