#!/usr/bin/env python3
"""
Market Regime Analysis via Hierarchical Clustering
===================================================
Compute technical indicators from raw OHLC, resample to 4h,
then discover natural market regimes using agglomerative clustering.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & resample to 4h ─────────────────────────────────────────

print("Loading OHLC data...")
df = pd.read_csv('tests/data/eurusd_hour_2005_2020_ohlc.csv')
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
df.set_index('datetime', inplace=True)
df = df[['open','high','low','close']].sort_index()

# Resample to 4h bars
ohlc_4h = df.resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
print(f"4h bars: {len(ohlc_4h)} ({ohlc_4h.index[0]} to {ohlc_4h.index[-1]})")

# ── 2. Compute raw technical indicators ──────────────────────────────

print("Computing indicators...")
c = ohlc_4h['close']
h = ohlc_4h['high']
l = ohlc_4h['low']
o = ohlc_4h['open']

# --- ATR (14 bars) ---
tr = pd.concat([
    h - l,
    (h - c.shift(1)).abs(),
    (l - c.shift(1)).abs()
], axis=1).max(axis=1)
atr_14 = tr.rolling(14).mean()

# ATR percentile (rolling 120 bars ~ 20 days)
atr_pct = atr_14.rolling(120).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
)

# --- ADX (14 bars) ---
plus_dm = (h - h.shift(1)).clip(lower=0)
minus_dm = (l.shift(1) - l).clip(lower=0)
# zero out when the other is larger
mask = plus_dm > minus_dm
plus_dm = plus_dm.where(mask, 0.0)
minus_dm = minus_dm.where(~mask, 0.0)

atr_s = tr.ewm(span=14, adjust=False).mean()
plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_s)
minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_s)
dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
adx = dx.ewm(span=14, adjust=False).mean()

# --- RSI (14 bars) ---
delta = c.diff()
gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
rsi = 100 - 100 / (1 + gain / (loss + 1e-10))

# --- Bollinger Bands (20 bars, 2 std) ---
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)  # normalized width
bb_position = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

# BB width percentile (regime: squeeze vs expansion)
bb_width_pct = bb_width.rolling(120).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
)

# --- MACD ---
ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
macd_signal = macd.ewm(span=9, adjust=False).mean()
macd_hist = macd - macd_signal

# --- Rate of Change (12 bars) ---
roc_12 = c.pct_change(12) * 100

# --- Directional Movement Index ---
di_spread = plus_di - minus_di  # positive = bullish

# --- Trend strength composite ---
# Combines ADX level with price position relative to EMAs
ema50 = c.ewm(span=50, adjust=False).mean()
ema200 = c.ewm(span=200, adjust=False).mean()
price_vs_ema50 = (c - ema50) / (atr_14 + 1e-10)  # in ATR units
price_vs_ema200 = (c - ema200) / (atr_14 + 1e-10)
ema_alignment = (ema50 - ema200) / (atr_14 + 1e-10)  # positive = uptrend

# --- Stochastic %K (14 bars) ---
low14 = l.rolling(14).min()
high14 = h.rolling(14).max()
stoch_k = 100 * (c - low14) / (high14 - low14 + 1e-10)

# --- Volatility regime: ATR ratio (current / long-term) ---
atr_60 = tr.rolling(60).mean()
atr_ratio = atr_14 / (atr_60 + 1e-10)

# ── 3. Build regime feature matrix ──────────────────────────────────

# Select features that describe the market STATE (not predict the future)
regime_df = pd.DataFrame({
    'adx': adx,                      # trend strength
    'di_spread': di_spread,           # trend direction
    'atr_pct': atr_pct,              # volatility percentile
    'atr_ratio': atr_ratio,          # vol expansion/contraction
    'bb_width_pct': bb_width_pct,    # BB squeeze level
    'bb_position': bb_position,       # where in the BB range (0-1)
    'rsi': rsi,                       # overbought/oversold
    'roc_12': roc_12,                # momentum
    'price_vs_ema50': price_vs_ema50, # trend position (short-term)
    'ema_alignment': ema_alignment,   # EMA stack alignment
    'stoch_k': stoch_k,             # stochastic position
    'macd_hist': macd_hist / (atr_14 + 1e-10),  # MACD histogram in ATR units
}, index=ohlc_4h.index).dropna()

print(f"Regime features: {regime_df.shape[0]} bars x {regime_df.shape[1]} features")
print(f"Date range: {regime_df.index[0]} to {regime_df.index[-1]}")
print(f"\nFeature statistics:")
print(regime_df.describe().round(3))

# ── 4. Standardize & PCA ────────────────────────────────────────────

scaler = StandardScaler()
X = scaler.fit_transform(regime_df)

# PCA for visualization and noise reduction
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)
print(f"\nPCA explained variance: {pca.explained_variance_ratio_.round(3)}")
print(f"Cumulative: {pca.explained_variance_ratio_.cumsum().round(3)}")

# ── 5. Hierarchical clustering ──────────────────────────────────────

print("\nPerforming hierarchical clustering (Ward linkage)...")
# Use a random sample for the dendrogram (full dataset too large)
np.random.seed(42)
sample_idx = np.random.choice(len(X_pca), size=min(5000, len(X_pca)), replace=False)
sample_idx.sort()
Z_sample = linkage(X_pca[sample_idx], method='ward')

# Test different numbers of clusters
for n_clusters in [3, 4, 5, 6, 7, 8]:
    labels = fcluster(Z_sample, n_clusters, criterion='maxclust')
    # Map back to full dataset using nearest neighbor
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_pca[sample_idx], labels)
    all_labels = knn.predict(X_pca)
    
    # Compute per-cluster stats
    regime_df_temp = regime_df.copy()
    regime_df_temp['regime'] = all_labels
    
    # Forward returns (4h) for profitability analysis
    regime_df_temp['fwd_return'] = c.reindex(regime_df.index).pct_change(1).shift(-1)
    regime_df_temp['fwd_return_6bar'] = c.reindex(regime_df.index).pct_change(6).shift(-6)
    
    cluster_sizes = regime_df_temp.groupby('regime').size()
    cluster_pct = cluster_sizes / len(regime_df_temp) * 100
    
    print(f"\n{'='*70}")
    print(f"K={n_clusters} clusters:")
    for cl in sorted(regime_df_temp['regime'].unique()):
        subset = regime_df_temp[regime_df_temp['regime'] == cl]
        print(f"  Regime {cl} ({len(subset)} bars, {cluster_pct[cl]:.1f}%):")
        print(f"    ADX={subset['adx'].mean():.1f}  DI_spread={subset['di_spread'].mean():.1f}"
              f"  ATR_pct={subset['atr_pct'].mean():.2f}  BB_width_pct={subset['bb_width_pct'].mean():.2f}")
        print(f"    RSI={subset['rsi'].mean():.1f}  BB_pos={subset['bb_position'].mean():.2f}"
              f"  EMA_align={subset['ema_alignment'].mean():.2f}  ROC={subset['roc_12'].mean():.3f}")
        fwd1 = subset['fwd_return'].dropna()
        fwd6 = subset['fwd_return_6bar'].dropna()
        if len(fwd1) > 0:
            print(f"    Fwd_1bar: mean={fwd1.mean()*10000:.2f}pips  std={fwd1.std()*10000:.2f}pips"
                  f"  sharpe={fwd1.mean()/(fwd1.std()+1e-10)*np.sqrt(252*6):.2f}")
        if len(fwd6) > 0:
            print(f"    Fwd_6bar: mean={fwd6.mean()*10000:.2f}pips  std={fwd6.std()*10000:.2f}pips"
                  f"  sharpe={fwd6.mean()/(fwd6.std()+1e-10)*np.sqrt(252):.2f}")

# ── 6. Detailed analysis with K=5 (good default) ────────────────────

print("\n" + "="*70)
print("DETAILED ANALYSIS WITH K=5")
print("="*70)

labels_5 = fcluster(Z_sample, 5, criterion='maxclust')
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_pca[sample_idx], labels_5)
all_labels_5 = knn5.predict(X_pca)

regime_df['regime'] = all_labels_5
regime_df['fwd_return'] = c.reindex(regime_df.index).pct_change(1).shift(-1)
regime_df['fwd_return_6bar'] = c.reindex(regime_df.index).pct_change(6).shift(-6)

# Regime transition matrix
transitions = pd.crosstab(
    regime_df['regime'], 
    regime_df['regime'].shift(-1).dropna().astype(int),
    normalize='index'
).round(3)
print("\nRegime Transition Probabilities (row → col):")
print(transitions)

# Regime persistence (how many consecutive bars on average)
regime_changes = (regime_df['regime'] != regime_df['regime'].shift(1))
regime_runs = []
current_run = 0
current_regime = None
for idx, row in regime_df.iterrows():
    if row['regime'] != current_regime:
        if current_regime is not None:
            regime_runs.append((current_regime, current_run))
        current_regime = row['regime']
        current_run = 1
    else:
        current_run += 1
if current_regime is not None:
    regime_runs.append((current_regime, current_run))

runs_df = pd.DataFrame(regime_runs, columns=['regime', 'run_length'])
print("\nRegime Persistence (consecutive 4h bars):")
for r in sorted(runs_df['regime'].unique()):
    subset = runs_df[runs_df['regime'] == r]['run_length']
    print(f"  Regime {r}: mean={subset.mean():.1f} bars  median={subset.median():.0f}"
          f"  max={subset.max()}  std={subset.std():.1f}")

# ── 7. Plots ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

# Get close prices aligned to regime data
close_regime = c.reindex(regime_df.index)

# Plot 1: Price + regime colors
colors_map = {1: 'green', 2: 'red', 3: 'blue', 4: 'orange', 5: 'purple'}
ax1 = axes[0]
ax1.plot(regime_df.index, close_regime, color='black', linewidth=0.5, alpha=0.7)
for r in sorted(regime_df['regime'].unique()):
    mask = regime_df['regime'] == r
    ax1.scatter(regime_df.index[mask], close_regime[mask], 
                c=colors_map.get(r, 'gray'), s=1, alpha=0.5, label=f'Regime {r}')
ax1.set_ylabel('EUR/USD Close')
ax1.set_title('Market Regimes (K=5) - Full History 2005-2020')
ax1.legend(markerscale=10)

# Plot 2: ADX + ATR percentile
ax2 = axes[1]
ax2.plot(regime_df.index, regime_df['adx'], label='ADX', color='blue', linewidth=0.5)
ax2.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='ADX=25')
ax2_r = ax2.twinx()
ax2_r.plot(regime_df.index, regime_df['atr_pct'], label='ATR_pct', color='orange', linewidth=0.5, alpha=0.7)
ax2.set_ylabel('ADX')
ax2_r.set_ylabel('ATR Percentile')
ax2.legend(loc='upper left')
ax2_r.legend(loc='upper right')

# Plot 3: Regime timeline
ax3 = axes[2]
ax3.scatter(regime_df.index, regime_df['regime'], c=[colors_map.get(r, 'gray') for r in regime_df['regime']], 
            s=2, alpha=0.5)
ax3.set_ylabel('Regime')
ax3.set_yticks(sorted(regime_df['regime'].unique()))

# Plot 4: Cumulative returns per regime (hypothetical)
ax4 = axes[3]
for r in sorted(regime_df['regime'].unique()):
    mask = regime_df['regime'] == r
    returns_in_regime = regime_df.loc[mask, 'fwd_return'].fillna(0)
    cum_ret = returns_in_regime.cumsum() * 10000  # in pips
    ax4.plot(regime_df.index[mask], cum_ret, color=colors_map.get(r, 'gray'), 
             linewidth=0.5, alpha=0.7, label=f'Regime {r}')
ax4.set_ylabel('Cumulative Pips (long bias)')
ax4.legend()
ax4.set_xlabel('Date')

plt.tight_layout()
plt.savefig('regime_analysis_overview.png', dpi=150)
print("\nSaved: regime_analysis_overview.png")

# ── 8. PCA scatter ───────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

ax = axes2[0]
for r in sorted(regime_df['regime'].unique()):
    mask = regime_df['regime'].values == r
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_map.get(r, 'gray'), 
               s=1, alpha=0.3, label=f'Regime {r}')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Regime Clusters in PCA Space')
ax.legend(markerscale=10)

ax2 = axes2[1]
for r in sorted(regime_df['regime'].unique()):
    mask = regime_df['regime'].values == r
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 2], c=colors_map.get(r, 'gray'), 
                s=1, alpha=0.3, label=f'Regime {r}')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
ax2.set_title('PC1 vs PC3')
ax2.legend(markerscale=10)

plt.tight_layout()
plt.savefig('regime_pca_scatter.png', dpi=150)
print("Saved: regime_pca_scatter.png")

# ── 9. Dendrogram ────────────────────────────────────────────────────

fig3, ax = plt.subplots(1, 1, figsize=(16, 8))
dendrogram(Z_sample, truncate_mode='lastp', p=30, ax=ax, 
           color_threshold=Z_sample[-4, 2])  # 5 clusters
ax.set_title('Hierarchical Clustering Dendrogram (Ward)')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig('regime_dendrogram.png', dpi=150)
print("Saved: regime_dendrogram.png")

# ── 10. Save regime data ─────────────────────────────────────────────

# Save scaler, PCA, and cluster model info for online use
import pickle

model_data = {
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'pca_components': pca.components_,
    'pca_mean': pca.mean_,
    'feature_names': list(regime_df.columns[:12]),
    'n_clusters': 5,
}

# Save the labeled data
regime_df.to_csv('regime_labeled_data.csv')
print(f"Saved: regime_labeled_data.csv ({len(regime_df)} rows)")

# Print summary for strategy development
print("\n" + "="*70)
print("REGIME CHARACTERIZATION SUMMARY")
print("="*70)
for r in sorted(regime_df['regime'].unique()):
    subset = regime_df[regime_df['regime'] == r]
    fwd = subset['fwd_return'].dropna()
    fwd6 = subset['fwd_return_6bar'].dropna()
    
    # Determine regime character
    avg_adx = subset['adx'].mean()
    avg_di = subset['di_spread'].mean()
    avg_atr = subset['atr_pct'].mean()
    avg_bbw = subset['bb_width_pct'].mean()
    avg_rsi = subset['rsi'].mean()
    avg_ema = subset['ema_alignment'].mean()
    
    # Classify
    if avg_adx > 25 and avg_di > 5:
        typ = "STRONG UPTREND"
    elif avg_adx > 25 and avg_di < -5:
        typ = "STRONG DOWNTREND"
    elif avg_adx > 20 and avg_di > 0:
        typ = "MILD UPTREND"
    elif avg_adx > 20 and avg_di < 0:
        typ = "MILD DOWNTREND"
    elif avg_atr > 0.6:
        typ = "HIGH VOLATILITY RANGE"
    elif avg_atr < 0.4:
        typ = "LOW VOLATILITY RANGE"
    else:
        typ = "MIXED/TRANSITION"
    
    print(f"\nRegime {r}: {typ}")
    print(f"  Bars: {len(subset)} ({len(subset)/len(regime_df)*100:.1f}%)")
    print(f"  ADX: {avg_adx:.1f}  DI_spread: {avg_di:.1f}")
    print(f"  ATR_pct: {avg_atr:.2f}  BB_width_pct: {avg_bbw:.2f}")
    print(f"  RSI: {avg_rsi:.1f}  EMA_alignment: {avg_ema:.2f}")
    if len(fwd) > 0:
        print(f"  1-bar mean return: {fwd.mean()*10000:.2f} pips (std: {fwd.std()*10000:.2f})")
    if len(fwd6) > 0:
        print(f"  6-bar mean return: {fwd6.mean()*10000:.2f} pips (std: {fwd6.std()*10000:.2f})")

print("\nDone!")
