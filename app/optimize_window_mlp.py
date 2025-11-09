#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_horizon_mlp.py
- Finds best predictive horizons for the two winning MLP configs:
  * Short-term: 5T, feature_mode=lognorm_close,  target_mode=logret_h, horizons in 5T bars
  * Long-term: 1H, feature_mode=raw_close,      target_mode=logret_h, horizons in 1H bars
- Loads 5T CSV, resamples to 1H for long-term.
- Uses same training method/metrics as the last benchmark:
  * Time split: 2y train, 1y valid (time-based)
  * MLP (Keras if available, else sklearn), StandardScaler, regularization, early stopping
  * MAE computed strictly in PRICE space (all models + NAIVE on the same scale)
- Prints detailed progress and saves CSV with the same metric columns + deltas vs NAIVE.
"""

import argparse, sys, os, time, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression  # not used, imported to keep parity if needed

# Optional Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import regularizers
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# ---------- small print helpers ----------
def info(msg: str, quiet: bool=False):
    if not quiet: print(msg, flush=True)
def warn(msg: str):
    print(f"[ADVERTENCIA] {msg}", file=sys.stderr, flush=True)
def error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ---------- timing helpers ----------
def tnow(): return time.perf_counter()
def tsecs(t0): return f"{time.perf_counter()-t0:,.2f}s"

# ---------- resampling & time index ----------
def _normalize_rule(rule: str) -> str:
    return {'5T':'5min','15T':'15min','1H':'1h','4H':'4h','1D':'1D'}.get(rule, rule)

def ensure_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=False)
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    df = df[~df.index.duplicated(keep='last')]
    return df

def resample_close(df: pd.DataFrame, close_col: str, rule: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return df[[close_col]].resample(_normalize_rule(rule), label='right', closed='right').last().dropna()

# ---------- dataset views ----------
def make_supervised_views(close: pd.Series, window: int, horizon: int,
                          feature_mode: str, target_mode: str):
    """
    Build zero-copy sliding windows and aligned targets.

    Returns:
      X_view    : (m, window) view
      y_target  : length m   (logret_h or price)
      y_price   : true future price C_{t+h} (length m)
      last_c    : last price in each window C_t (length m)
      idx       : DatetimeIndex aligned to y (length m)
    """
    c = close.values.astype(float)
    n = c.shape[0]
    if n < (window + horizon):
        return (np.empty((0, window), float), np.empty((0,), float),
                np.empty((0,), float), np.empty((0,), float),
                close.index[:0])

    W = sliding_window_view(c, window_shape=window)  # (n - window + 1, window)
    m = n - window - horizon + 1
    Wm = W[:m]
    last_c = c[window-1: window-1+m]                     # C_t
    y_price = c[window + horizon - 1: window + horizon - 1 + m]  # C_{t+h}

    # features
    if feature_mode == 'raw_close':
        X_view = Wm
    elif feature_mode == 'lognorm_close':
        logW = np.log(Wm)
        X_view = logW - logW[:, [-1]]   # normalize by last (stationary-ish)
    else:
        raise ValueError(f"feature_mode desconocido: {feature_mode}")

    # targets
    if target_mode == 'price':
        y_target = y_price.copy()
    elif target_mode == 'logret_h':
        # log-return to horizon: log(C_{t+h}) - log(C_t)
        y_target = np.log(y_price) - np.log(last_c)
    else:
        raise ValueError(f"target_mode desconocido: {target_mode}")

    idx = close.index[window + horizon - 1: window + horizon - 1 + m]
    return X_view, y_target, y_price, last_c, idx

# ---------- time split ----------
def time_based_train_valid_split(index: pd.DatetimeIndex, years_train=2, years_valid=1):
    if index.size == 0:
        return np.zeros((0,), bool), np.zeros((0,), bool)
    end = index.max()
    valid_start = end - pd.Timedelta(days=365*years_valid)
    train_start = valid_start - pd.Timedelta(days=365*years_train)
    if train_start < index.min():
        train_start = index.min()
    is_valid = (index > valid_start) & (index <= end)
    is_train = (index > train_start) & (index <= valid_start)
    if is_train.sum() < 100: warn("Train muy pequeño (<100).")
    if is_valid.sum() < 50:  warn("Valid muy pequeño (<50).")
    return np.asarray(is_train), np.asarray(is_valid)

# ---------- optional thinning (keeps compute sane) ----------
def apply_stride_cap_indices(idx_len, stride=None, cap=None):
    all_idx = np.arange(idx_len, dtype=np.int64)
    if cap is not None and cap > 0 and idx_len > cap:
        all_idx = all_idx[-cap:]
    s = max(1, int(stride)) if stride is not None else 1
    if s > 1: all_idx = all_idx[::s]
    return all_idx

def auto_budget_for(rule: str, window_bars: int, mode: str):
    """
    Simple heuristic to keep runs fast.
    Returns stride_tr, stride_va, cap_tr, cap_va.
    """
    if mode == 'short':  # 5T, lognorm_close/logret_h
        # many samples; thin moderately
        return 8, 8, 250_000, 150_000
    else:                # long 1H, raw_close/logret_h
        # fewer rows; light/no thinning
        return 1, 1, None, None

# ---------- MLP train/eval (MAE in price space) ----------
def fit_eval_mlp(X_view, y_target, y_price, last_c, idx,
                 seed, mlp_width, mlp_second,
                 target_mode, stride_tr, stride_va, cap_tr, cap_va,
                 quiet=False) -> Dict[str, float]:

    t0 = tnow()
    is_tr, is_va = time_based_train_valid_split(idx, 2, 1)
    n_tr, n_va = int(is_tr.sum()), int(is_va.sum())
    info(f"  [split] train={n_tr:,}, valid={n_va:,} (elapsed={tsecs(t0)})", quiet)
    if n_tr == 0 or n_va == 0:
        return {k: np.nan for k in ['NAIVE_TRAIN_MAE','NAIVE_VALID_MAE',
                                    'MLP_TRAIN_MAE','MLP_VALID_MAE']}

    tr_pos = np.flatnonzero(is_tr)
    va_pos = np.flatnonzero(is_va)
    tr_idx = tr_pos[apply_stride_cap_indices(tr_pos.size, stride_tr, cap_tr)]
    va_idx = va_pos[apply_stride_cap_indices(va_pos.size, stride_va, cap_va)]
    info(f"  [thinning] train_kept={tr_idx.size:,}, valid_kept={va_idx.size:,} (elapsed={tsecs(t0)})", quiet)

    Xtr = X_view[tr_idx]; Xva = X_view[va_idx]
    ytr = y_target[tr_idx]; yva = y_target[va_idx]
    ypr_tr = y_price[tr_idx]; ypr_va = y_price[va_idx]
    last_tr = last_c[tr_idx];  last_va = last_c[va_idx]

    # inverse to price
    def inv_price(yhat, lastv):
        return yhat if target_mode == 'price' else (lastv * np.exp(yhat))

    # NAIVE baseline in price space
    yhat_naive_tr = last_tr
    yhat_naive_va = last_va
    naive_tr = mean_absolute_error(ypr_tr, yhat_naive_tr)
    naive_va = mean_absolute_error(ypr_va, yhat_naive_va)
    info(f"  [NAIVE] train={naive_tr:.6f}, valid={naive_va:.6f} (elapsed={tsecs(t0)})", quiet)

    # MLP
    if KERAS_AVAILABLE:
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xva_s = sc.transform(Xva)
        model = keras.Sequential([
            keras.layers.Input(shape=(Xtr_s.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(mlp_width, activation='relu',
                               kernel_regularizer=regularizers.l2(1e-4)),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(mlp_second, activation='relu',
                               kernel_regularizer=regularizers.l2(1e-4)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mae', metrics=['mae'])
        es = keras.callbacks.EarlyStopping(monitor='val_mae', patience=20,
                                           restore_best_weights=True, mode='min', verbose=0)
        model.fit(Xtr_s, ytr, validation_data=(Xva_s, yva),
                  epochs=200, batch_size=256, shuffle=True, verbose=0, callbacks=[es])
        yhat_tr = model.predict(Xtr_s, verbose=0).ravel()
        yhat_va = model.predict(Xva_s, verbose=0).ravel()
    else:
        mlp = MLPRegressor(hidden_layer_sizes=(mlp_width, mlp_second),
                           activation='relu', solver='adam', alpha=1e-4,
                           learning_rate='adaptive', learning_rate_init=1e-3,
                           max_iter=400, shuffle=True, random_state=seed,
                           early_stopping=True, n_iter_no_change=20, tol=1e-4, verbose=False)
        pipe = Pipeline([('sc', StandardScaler()), ('mlp', mlp)])
        pipe.fit(Xtr, ytr)
        yhat_tr = pipe.predict(Xtr); yhat_va = pipe.predict(Xva)

    mae_mlp_tr = mean_absolute_error(ypr_tr, inv_price(yhat_tr, last_tr))
    mae_mlp_va = mean_absolute_error(ypr_va, inv_price(yhat_va, last_va))
    info(f"  [MLP]   train={mae_mlp_tr:.6f}, valid={mae_mlp_va:.6f} (elapsed={tsecs(t0)})", quiet)

    return {
        'NAIVE_TRAIN_MAE': float(naive_tr),
        'NAIVE_VALID_MAE': float(naive_va),
        'MLP_TRAIN_MAE': float(mae_mlp_tr),
        'MLP_VALID_MAE': float(mae_mlp_va),
    }

# ---------- runner for a single (periodicity, window/horizon) ----------
def run_one(df5m: pd.DataFrame, rule: str, window_bars: int,
            feature_mode: str, target_mode: str,
            seed: int, mlp_width: int, mlp_second: int, quiet: bool):

    hdr = f"[{rule}][{feature_mode}/{target_mode}][window={window_bars}]"
    t0 = tnow()
    info(f"{hdr} preparar serie/resample…", quiet)
    if rule == '5T':
        df_rule = df5m
    elif rule == '1H':
        df_rule = resample_close(df5m, 'close', '1H')
    else:
        raise ValueError("This program supports only 5T and 1H for the requested experiment.")

    # Build views (horizon_bars = window_bars)
    info(f"{hdr} building views…", quiet)
    Xv, y_t, y_p, last_c, idx = make_supervised_views(
        df_rule['close'].astype(float),
        window=window_bars,
        horizon=window_bars,
        feature_mode=feature_mode,
        target_mode=target_mode
    )
    m = y_t.shape[0]
    info(f"{hdr} m={m:,} muestras (elapsed={tsecs(t0)})", quiet)
    if m == 0:
        warn(f"{hdr} sin muestras suficientes.")
        return {
            'periodicity': rule, 'feature_mode': feature_mode, 'target_mode': target_mode,
            'window_bars': int(window_bars), 'n_samples': 0,
            'NAIVE_TRAIN_MAE': np.nan, 'NAIVE_VALID_MAE': np.nan,
            'MLP_TRAIN_MAE': np.nan,   'MLP_VALID_MAE': np.nan
        }

    # Thinning heuristic per track
    mode = 'short' if rule == '5T' else 'long'
    stride_tr, stride_va, cap_tr, cap_va = auto_budget_for(rule, window_bars, mode)
    info(f"{hdr} fit/eval (stride_tr={stride_tr}, stride_va={stride_va}, cap_tr={cap_tr}, cap_va={cap_va})…", quiet)

    metrics = fit_eval_mlp(Xv, y_t, y_p, last_c, idx,
                           seed=seed, mlp_width=mlp_width, mlp_second=mlp_second,
                           target_mode=target_mode,
                           stride_tr=stride_tr, stride_va=stride_va,
                           cap_tr=cap_tr, cap_va=cap_va,
                           quiet=quiet)

    out = {
        'periodicity': rule,
        'feature_mode': feature_mode,
        'target_mode': target_mode,
        'window_bars': int(window_bars),
        'n_samples': int(m),
        **metrics
    }
    info(f"{hdr} DONE (total={tsecs(t0)})", quiet)
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Optimize horizons for the two winning MLP configs (short 5T, long 1H).")
    ap.add_argument('csv', type=str, help='Ruta al CSV base (5T).')
    ap.add_argument('--time-col', type=str, default=None)
    ap.add_argument('--close-col', type=str, default='close')
    ap.add_argument('--out', type=str, default='opt_horizon_mlp.csv')
    ap.add_argument('--mlp-width', type=int, default=128)
    ap.add_argument('--mlp-second', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-tqdm', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    info("[INIT] Cargando CSV base (5T)…", args.quiet)
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        error(f"No se pudo leer el CSV: {e}"); sys.exit(1)
    info(f"[INIT] CSV cargado: shape={df.shape}", args.quiet)

    time_col = args.time_col
    if time_col is None:
        cands = [c for c in df.columns if c.lower() in ('time','timestamp','datetime','date')]
        if not cands:
            error("No se encontró columna temporal. Use --time-col."); sys.exit(1)
        time_col = cands[0]
        info(f"[INIT] Columna temporal detectada: {time_col}", args.quiet)

    if args.close_col not in df.columns:
        error(f"No existe la columna '{args.close_col}'. Use --close-col."); sys.exit(1)

    info("[INIT] Normalizando índice temporal…", args.quiet)
    df = ensure_datetime_index(df[[time_col, args.close_col]].rename(columns={args.close_col:'close'}), time_col)
    info(f"[INIT] Serie 5T lista: n={df.shape[0]} puntos.", args.quiet)

    # Horizon sets (as bars)
    horizons_short_5T = [12, 24, 36, 48, 60, 72, 144, 288]         # 5-minute bars
    horizons_long_1H  = [24, 48, 72, 96, 120, 144, 288]            # 1-hour bars

    combos: List[Tuple[str, str, str, int]] = []
    # Short-term track (5T)
    for hb in horizons_short_5T:
        combos.append(('5T', 'lognorm_close', 'logret_h', hb))
    # Long-term track (1H)
    for hb in horizons_long_1H:
        combos.append(('1H', 'raw_close', 'logret_h', hb))

    progress = tqdm(total=len(combos), disable=args.no_tqdm, desc="Horizons progress", miniters=1)
    results: List[Dict[str, float]] = []
    for (rule, fmode, tmode, window_bars) in combos:
        info(f"\n=== RUN {progress.n+1}/{len(combos)}: {rule} | window={window_bars} | {fmode}/{tmode} ===", args.quiet)
        row = run_one(df, rule, window_bars, fmode, tmode,
                      seed=args.seed, mlp_width=args.mlp_width, mlp_second=args.mlp_second,
                      quiet=args.quiet)
        results.append(row)
        progress.update(1)
    progress.close()

    info("[PIPE] Compilando resultados…", args.quiet)
    dfr = (pd.DataFrame(results)
           .sort_values(by=['periodicity','window_bars'])
           .reset_index(drop=True))

    # deltas vs NAIVE (valid)
    dfr['MLP_VALID_DeltaMAE_vs_NAIVE']  = dfr['MLP_VALID_MAE'] - dfr['NAIVE_VALID_MAE']
    dfr['MLP_VALID_ImprovPct_vs_NAIVE'] = (dfr['NAIVE_VALID_MAE'] - dfr['MLP_VALID_MAE'])/dfr['NAIVE_VALID_MAE']*100.0

    # pretty print
    base_cols = ['periodicity','feature_mode','target_mode','window_bars','n_samples',
                 'NAIVE_TRAIN_MAE','NAIVE_VALID_MAE',
                 'MLP_TRAIN_MAE','MLP_VALID_MAE']
    delta_cols = ['MLP_VALID_DeltaMAE_vs_NAIVE','MLP_VALID_ImprovPct_vs_NAIVE']

    def _fmt(v):
        try: return f"{v:,.6f}"
        except Exception: return str(v)

    info("\n=== OPTIMIZACIÓN DE HORIZONTE — MLP (todo en PRECIO) ===", False)
    print(dfr[base_cols + delta_cols].to_string(index=False, float_format=_fmt), flush=True)

    # leaderboards per periodicity
    for rule in ['5T','1H']:
        sub = dfr[(dfr['periodicity']==rule) & (dfr['n_samples']>0)]
        if sub.empty:
            print(f"\n=== LEADERBOARD — {rule} ===\n(sin filas útiles)", flush=True)
            continue
        # pick smallest valid MLP MAE (break ties by larger improvement vs NAIVE)
        sub = sub.copy()
        sub['rank_key'] = list(zip(sub['MLP_VALID_MAE'].values, -sub['MLP_VALID_ImprovPct_vs_NAIVE'].values))
        best = sub.sort_values(by=['rank_key']).iloc[0]
        print(f"\n=== LEADERBOARD — {rule} ===", flush=True)
        print(pd.DataFrame([{
            'periodicity': rule,
            'best_window_bars': int(best['window_bars']),
            'best_valid_mae': float(best['MLP_VALID_MAE']),
            'naive_valid_mae': float(best['NAIVE_VALID_MAE']),
            'vs_naive_improv_pct': float(best['MLP_VALID_ImprovPct_vs_NAIVE'])
        }]).to_string(index=False, float_format=_fmt), flush=True)

    # save
    info(f"\n[PIPE] Guardando resultados en: {args.out}", args.quiet)
    try:
        dfr.drop(columns=['rank_key'], errors='ignore').to_csv(args.out, index=False)
        info("[OK] Archivo escrito correctamente.", False)
    except Exception as e:
        error(f"No se pudo guardar {args.out}: {e}")

if __name__ == '__main__':
    main()
