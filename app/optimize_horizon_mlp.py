#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
horizon_search_fixed_window.py

Search the best predictive horizon for two fixed-window setups:

Short-term  (5T):
  - periodicity='5T'
  - window_bars=288
  - feature_mode='lognorm_close'
  - target_mode='logret_h'
  - horizons_short_5T = [12, 24, 36, 48, 60, 72, 144, 288]

Long-term   (1H):
  - periodicity='1H'
  - window_bars=120
  - feature_mode='raw_close'
  - target_mode='logret_h'
  - horizons_long_1H = [24, 48, 72, 96, 120, 144, 288]

MAE is computed in PRICE space (model preds are inverse-transformed).
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
from sklearn.linear_model import LinearRegression  # not used, but handy for quick checks

# Optional Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import regularizers
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# ---------- small helpers ----------
def info(msg: str, quiet: bool=False):
    if not quiet:
        print(msg, flush=True)
def warn(msg: str):
    print(f"[ADVERTENCIA] {msg}", file=sys.stderr, flush=True)
def error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
def tnow(): return time.perf_counter()
def tsecs(t0): return f"{time.perf_counter()-t0:,.2f}s"

# ---------- resampling / datetime ----------
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
        return df[[close_col]].resample(_normalize_rule(rule),
                                        label='right', closed='right').last().dropna()

# ---------- dataset views ----------
def make_views(close: pd.Series, window: int, horizon: int,
               feature_mode: str, target_mode: str):
    """
    Returns:
      X_view (m, window)    - features (zero-copy or small transform)
      y_target (m,)         - model target (price or logret_h)
      y_price  (m,)         - true future price
      last_c   (m,)         - last price in window (C_t)
      idx_y    (m,)         - timestamps for each sample
    """
    c = close.values.astype(float)
    n = c.shape[0]
    if n < (window + horizon):
        return (np.empty((0, window), float), np.empty((0,), float),
                np.empty((0,), float), np.empty((0,), float),
                close.index[:0])

    W = sliding_window_view(c, window_shape=window)  # (n-window+1, window)
    m = n - window - horizon + 1
    Wm = W[:m]
    last_c = c[window-1: window-1+m]                         # C_t
    y_price = c[window+horizon-1: window+horizon-1+m]        # C_{t+h}

    if feature_mode == 'raw_close':
        X_view = Wm
    elif feature_mode == 'lognorm_close':
        logW = np.log(Wm)
        X_view = logW - logW[:, [-1]]
    else:
        raise ValueError(f"feature_mode desconocido: {feature_mode}")

    if target_mode == 'price':
        y_target = y_price.copy()
    elif target_mode == 'logret_h':
        y_target = np.log(y_price) - np.log(last_c)
    else:
        raise ValueError(f"target_mode desconocido: {target_mode}")

    idx_y = close.index[window + horizon - 1: window + horizon - 1 + m]
    return X_view, y_target, y_price, last_c, idx_y

# ---------- time split ----------
def time_based_train_valid_split(index: pd.DatetimeIndex, years_train=2, years_valid=1):
    if index.size == 0:
        return np.zeros((0,), bool), np.zeros((0,), bool)
    end = index.max()
    valid_start = end - pd.Timedelta(days=365*years_valid)
    train_start = valid_start - pd.Timedelta(days=365*years_train)
    if train_start < index.min(): train_start = index.min()
    is_valid = (index > valid_start) & (index <= end)
    is_train = (index > train_start) & (index <= valid_start)
    if is_train.sum() < 100: warn("Train muy pequeño (<100).")
    if is_valid.sum() < 50:  warn("Valid muy pequeño (<50).")
    return np.asarray(is_train), np.asarray(is_valid)

# ---------- MLP fit/eval (PRICE space MAE) ----------
def fit_eval_mlp(X_view, y_target, y_price, last_c, idx,
                 target_mode: str, seed: int, mlp_width: int, mlp_second: int,
                 quiet: bool=False) -> Dict[str, float]:
    t0 = tnow()
    is_tr, is_va = time_based_train_valid_split(idx, 2, 1)
    n_tr, n_va = int(is_tr.sum()), int(is_va.sum())
    info(f"  [split] train={n_tr:,}, valid={n_va:,} (elapsed={tsecs(t0)})", quiet)
    if n_tr==0 or n_va==0:
        return {k: np.nan for k in ("NAIVE_TRAIN_MAE","NAIVE_VALID_MAE",
                                    "MLP_TRAIN_MAE","MLP_VALID_MAE")}

    tr_idx = np.flatnonzero(is_tr)
    va_idx = np.flatnonzero(is_va)

    Xtr, Xva = X_view[tr_idx], X_view[va_idx]
    ytr, yva = y_target[tr_idx], y_target[va_idx]
    ypr_tr, ypr_va = y_price[tr_idx], y_price[va_idx]
    last_tr, last_va = last_c[tr_idx],  last_c[va_idx]

    def inv_price(yhat, lastv):
        return yhat if target_mode=='price' else (lastv * np.exp(yhat))

    # Baseline (NAIVE = persist last price)
    yhat_naive_tr, yhat_naive_va = last_tr, last_va
    mae_naive_tr = mean_absolute_error(ypr_tr, yhat_naive_tr)
    mae_naive_va = mean_absolute_error(ypr_va, yhat_naive_va)
    info(f"  [NAIVE] train={mae_naive_tr:.6f}, valid={mae_naive_va:.6f} (elapsed={tsecs(t0)})", quiet)

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
        "NAIVE_TRAIN_MAE": float(mae_naive_tr),
        "NAIVE_VALID_MAE": float(mae_naive_va),
        "MLP_TRAIN_MAE": float(mae_mlp_tr),
        "MLP_VALID_MAE": float(mae_mlp_va),
    }

# ---------- run one periodicity with fixed window ----------
def evaluate_fixed_window(df5m: pd.DataFrame, rule: str, window_bars: int,
                          horizons: List[int], feature_mode: str, target_mode: str,
                          seed: int, mlp_width: int, mlp_second: int,
                          quiet: bool) -> List[Dict[str, float]]:
    hdr = f"[{rule}][window={window_bars}][{feature_mode}/{target_mode}]"
    info(f"{hdr} START", quiet)
    df_rule = df5m if rule=='5T' else resample_close(df5m, 'close', rule)

    out_rows = []
    pbar = tqdm(total=len(horizons), desc=f"{rule} horizons", miniters=1, disable=quiet)
    for H in horizons:
        info(f"{hdr} building views: horizon={H}…", quiet)
        X_view, y_t, y_p, last_c, idx = make_views(
            df_rule['close'].astype(float),
            window=int(window_bars),
            horizon=int(H),
            feature_mode=feature_mode,
            target_mode=target_mode
        )
        m = y_t.shape[0]
        info(f"{hdr} horizon={H} -> samples={m:,}", quiet)
        if m == 0:
            warn(f"{hdr} horizon={H}: sin muestras.")
            row = dict(periodicity=rule, feature_mode=feature_mode, target_mode=target_mode,
                       window_bars=int(window_bars), horizon_bars=int(H), n_samples=0,
                       NAIVE_TRAIN_MAE=np.nan, NAIVE_VALID_MAE=np.nan,
                       MLP_TRAIN_MAE=np.nan, MLP_VALID_MAE=np.nan,
                       MLP_VALID_DeltaMAE_vs_NAIVE=np.nan, MLP_VALID_ImprovPct_vs_NAIVE=np.nan)
            out_rows.append(row)
            pbar.update(1)
            continue

        mets = fit_eval_mlp(X_view, y_t, y_p, last_c, idx,
                            target_mode, seed, mlp_width, mlp_second, quiet)
        d_mae = mets["MLP_VALID_MAE"] - mets["NAIVE_VALID_MAE"]
        imp_pct = (mets["NAIVE_VALID_MAE"] - mets["MLP_VALID_MAE"]) / mets["NAIVE_VALID_MAE"] * 100.0

        row = dict(periodicity=rule, feature_mode=feature_mode, target_mode=target_mode,
                   window_bars=int(window_bars), horizon_bars=int(H), n_samples=int(m),
                   **mets,
                   MLP_VALID_DeltaMAE_vs_NAIVE=float(d_mae),
                   MLP_VALID_ImprovPct_vs_NAIVE=float(imp_pct))
        out_rows.append(row)
        pbar.update(1)
    pbar.close()
    info(f"{hdr} DONE", quiet)
    return out_rows

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Horizon search with fixed windows for short/long setups (MLP).")
    ap.add_argument('csv', type=str)
    ap.add_argument('--time-col', type=str, default=None)
    ap.add_argument('--close-col', type=str, default='close')
    ap.add_argument('--out', type=str, default='horizon_fixed_window.csv')
    ap.add_argument('--mlp-width', type=int, default=128)
    ap.add_argument('--mlp-second', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    info("[INIT] Cargando CSV base (5T)…", args.quiet)
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        error(f"No se pudo leer el CSV: {e}")
        sys.exit(1)
    info(f"[INIT] CSV shape={df.shape}", args.quiet)

    time_col = args.time_col
    if time_col is None:
        cands = [c for c in df.columns if c.lower() in ('time','timestamp','datetime','date')]
        if not cands:
            error("No se encontró columna temporal. Use --time-col.")
            sys.exit(1)
        time_col = cands[0]
        info(f"[INIT] Columna temporal detectada: {time_col}", args.quiet)

    if args.close_col not in df.columns:
        error(f"No existe la columna '{args.close_col}'. Use --close-col.")
        sys.exit(1)

    info("[INIT] Normalizando índice temporal…", args.quiet)
    df = ensure_datetime_index(df[[time_col, args.close_col]].rename(columns={args.close_col:'close'}), time_col)
    info(f"[INIT] Serie 5T lista: n={df.shape[0]} puntos.", args.quiet)

    # Fixed windows chosen by improvement vs NAIVE:
    window_short_5T = 288        # 5T
    window_long_1H  = 144        # 1H

    horizons_short_5T = [12, 24, 36, 48, 60, 72, 144, 288]
    horizons_long_1H  = [24, 48, 72, 96, 120, 144, 288]

    # Short-term setup
    res_short = evaluate_fixed_window(
        df5m=df,
        rule='5T',
        window_bars=window_short_5T,
        horizons=horizons_short_5T,
        feature_mode='lognorm_close',
        target_mode='logret_h',
        seed=args.seed,
        mlp_width=args.mlp_width,
        mlp_second=args.mlp_second,
        quiet=args.quiet
    )

    # Long-term setup
    res_long = evaluate_fixed_window(
        df5m=df,
        rule='1H',
        window_bars=window_long_1H,
        horizons=horizons_long_1H,
        feature_mode='raw_close',
        target_mode='logret_h',
        seed=args.seed,
        mlp_width=args.mlp_width,
        mlp_second=args.mlp_second,
        quiet=args.quiet
    )

    # Compile & print
    results = pd.DataFrame(res_short + res_long)
    results = results.sort_values(by=['periodicity','horizon_bars']).reset_index(drop=True)

    info("\n=== RESULTS (MLP, MAE in PRICE) ===", False)
    cols = ['periodicity','feature_mode','target_mode','window_bars','horizon_bars','n_samples',
            'NAIVE_TRAIN_MAE','NAIVE_VALID_MAE','MLP_TRAIN_MAE','MLP_VALID_MAE',
            'MLP_VALID_DeltaMAE_vs_NAIVE','MLP_VALID_ImprovPct_vs_NAIVE']
    def _fmt(v):
        try: return f"{v:,.12f}"
        except Exception: return str(v)
    print(results[cols].to_string(index=False, float_format=_fmt), flush=True)

    # Leaderboards by periodicity
    for rule in ['5T','1H']:
        sub = results[results['periodicity']==rule]
        if sub.empty: continue
        best = sub.sort_values(by=['MLP_VALID_MAE','MLP_VALID_ImprovPct_vs_NAIVE'],
                               ascending=[True, False]).iloc[0]
        print(f"\n=== LEADERBOARD — {rule} (fixed window={int(best['window_bars'])}) ===", flush=True)
        print(f"Best horizon_bars={int(best['horizon_bars'])} | "
              f"valid_MAE={best['MLP_VALID_MAE']:.12f} | "
              f"naive_MAE={best['NAIVE_VALID_MAE']:.12f} | "
              f"improv%={best['MLP_VALID_ImprovPct_vs_NAIVE']:.6f}", flush=True)

    # Save CSV
    out = args.out
    info(f"\n[PIPE] Guardando resultados en: {out}", args.quiet)
    try:
        results.to_csv(out, index=False)
        info("[OK] Archivo escrito correctamente.", False)
    except Exception as e:
        error(f"No se pudo guardar {out}: {e}")

if __name__ == '__main__':
    main()
