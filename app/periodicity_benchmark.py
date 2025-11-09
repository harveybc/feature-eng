#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast periodicity_benchmark.py
- Vectorized sliding windows (no Python loops, no giant copies)
- Honest, single progress bar + per-stage timing logs
- Budget-aware thinning (stride/cap) applied AFTER time split (no leakage)
"""

import argparse, sys, math, os, time, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TF INFO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# XGBoost core API (most compatible)
try:
    import xgboost as xgb
    XGB_CORE_AVAILABLE = True
except Exception:
    XGB_CORE_AVAILABLE = False

# Keras (optional)
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

# ---------- time utilities ----------
def tnow(): return time.perf_counter()
def tsecs(t0): return f"{time.perf_counter()-t0:,.2f}s"

# ---------- resampling ----------
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

# ---------- horizon & window ----------
def hours_per_bar(rule: str) -> float:
    return {'5T':5/60, '15T':15/60, '1H':1.0, '4H':4.0, '1D':24.0}.get(rule, np.nan)

def to_bars_or_nan(hours: float, rule: str) -> float:
    hp = hours_per_bar(rule)
    if not np.isfinite(hp) or hp<=0 or hours<hp: return np.nan
    return int(max(1, round(hours/hp)))

# ---------- dataset building (vectorized views) ----------
def make_supervised_views(close: pd.Series, window: int, horizon: int,
                          feature_mode: str, target_mode: str):
    """
    Returns:
      W_view: view of shape (n-window+1, window) over close (zero-copy)
      y_target (len m), y_price (len m), last_c (len m), idx_y (len m)
    where m = n - window - horizon + 1
    """
    c = close.values.astype(float)
    n = c.shape[0]
    if n < (window + horizon):
        return (np.empty((0, window), float), np.empty((0,), float),
                np.empty((0,), float), np.empty((0,), float),
                close.index[:0])

    # Sliding window (zero-copy view)
    W = sliding_window_view(c, window_shape=window)  # shape: (n-window+1, window)

    # Align target so that sample i predicts c[i + window + horizon - 1]
    m = n - window - horizon + 1
    Wm = W[:m]                       # keep only those that have a target m
    last_c = c[window-1: window-1+m] # C_t
    y_price = c[window+horizon-1: window+horizon-1+m]  # C_{t+h}

    # features
    if feature_mode == 'raw_close':
        X_view = Wm  # view
    elif feature_mode == 'lognorm_close':
        # log-normalized by last price in window (stationary-ish)
        logW = np.log(Wm)
        X_view = logW - logW[:, [-1]]  # subtract last col
    else:
        raise ValueError(f"feature_mode desconocido: {feature_mode}")

    # targets
    if target_mode == 'price':
        y_target = y_price.copy()
    elif target_mode == 'logret_h':
        y_target = np.log(y_price) - np.log(last_c)
    else:
        raise ValueError(f"target_mode desconocido: {target_mode}")

    # timestamps aligned to target
    idx = close.index[window + horizon - 1: window + horizon - 1 + m]
    return X_view, y_target, y_price, last_c, idx

# ---------- time split & thinning ----------
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

def apply_stride_cap_indices(idx_len, stride=None, cap=None):
    """
    Returns integer indices [0..idx_len-1] thinned by (cap then stride).
    Keeps the most recent 'cap' first, then steps by 'stride'.
    """
    all_idx = np.arange(idx_len, dtype=np.int64)
    if cap is not None and cap>0 and idx_len>cap:
        all_idx = all_idx[-cap:]
    s = max(1, int(stride)) if stride is not None else 1
    if s>1: all_idx = all_idx[::s]
    return all_idx

def auto_budget(rule: str, horizon_hours: float, budget: str):
    # Returns stride_tr, stride_va, cap_tr, cap_va, xgb_rounds, xgb_es
    mult = {'small':1.5,'medium':1.0,'large':0.7}.get(budget,1.0)
    stride_tr=stride_va=1; cap_tr=cap_va=None; rounds=int(700/mult); es=50
    if rule=='5T' and horizon_hours>=48:  # brutal case
        stride_tr=stride_va=int(20*mult); cap_tr=int(200_000*mult); cap_va=int(120_000*mult)
        rounds=int(600/mult); es=40
    elif rule=='5T' and horizon_hours<=6:
        stride_tr=stride_va=int(8*mult); cap_tr=int(250_000*mult); cap_va=int(150_000*mult)
        rounds=int(700/mult); es=50
    elif rule=='15T' and horizon_hours>=48:
        stride_tr=stride_va=int(6*mult); cap_tr=int(200_000*mult); cap_va=int(100_000*mult)
        rounds=int(700/mult); es=50
    return max(1,stride_tr),max(1,stride_va),cap_tr,cap_va,rounds,es

# ---------- XGB core (with ES) ----------
def train_xgb_core_with_es(Xtr, ytr, Xva, yva, seed, num_boost_round, es_rounds):
    if not XGB_CORE_AVAILABLE: raise RuntimeError("XGBoost core API no disponible.")
    dtr = xgb.DMatrix(Xtr, label=ytr); dva = xgb.DMatrix(Xva, label=yva)
    params = {
        "objective":"reg:absoluteerror", "eval_metric":"mae",
        "seed":seed, "tree_method":"hist",
        "max_depth":3, "grow_policy":"lossguide", "max_leaves":64,
        "min_child_weight":10.0, "gamma":1.0,
        "reg_lambda":5.0, "reg_alpha":0.5,
        "subsample":0.6, "colsample_bytree":0.6, "colsample_bylevel":0.6,
        "learning_rate":0.03,
        "booster":"gbtree"  # dart can be slower; keep lean here
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        bst = xgb.train(params, dtr, num_boost_round=num_boost_round,
                        early_stopping_rounds=es_rounds, evals=[(dva,"valid")],
                        verbose_eval=False)
    best_it = getattr(bst, "best_iteration", None)
    def _pred(dm):
        if best_it is None: return bst.predict(dm)
        try: return bst.predict(dm, iteration_range=(0,int(best_it)+1))
        except TypeError:
            try: return bst.predict(dm, ntree_limit=int(best_it)+1)
            except TypeError: return bst.predict(dm)
    return _pred(dtr), _pred(dva)

# ---------- models (compute MAE in PRICE space) ----------
def fit_eval_modes(X_view, y_target, y_price, last_c, idx,
                   seed, mlp_width, mlp_second, target_mode,
                   stride_tr, stride_va, cap_tr, cap_va,
                   xgb_rounds, xgb_es, quiet=False) -> Dict[str,float]:

    t0 = tnow()
    is_tr, is_va = time_based_train_valid_split(idx, 2, 1)
    n_tr = int(is_tr.sum()); n_va=int(is_va.sum())
    info(f"  [split] train={n_tr:,}, valid={n_va:,} (elapsed={tsecs(t0)})", quiet)

    # Build index positions for the view (we never materialize full X)
    tr_pos = np.flatnonzero(is_tr)
    va_pos = np.flatnonzero(is_va)

    # Thin AFTER split
    tr_sel = apply_stride_cap_indices(tr_pos.size, stride_tr, cap_tr)
    va_sel = apply_stride_cap_indices(va_pos.size, stride_va, cap_va)
    tr_idx = tr_pos[tr_sel]; va_idx = va_pos[va_sel]
    info(f"  [thinning] train_kept={tr_idx.size:,}, valid_kept={va_idx.size:,} (elapsed={tsecs(t0)})", quiet)

    # Slice views/vectors
    Xtr = X_view[tr_idx]; Xva = X_view[va_idx]
    ytr = y_target[tr_idx]; yva = y_target[va_idx]
    ypr_tr = y_price[tr_idx]; ypr_va = y_price[va_idx]
    last_tr = last_c[tr_idx];  last_va = last_c[va_idx]

    # Helper to inverse-target to PRICE space
    def inv_price(yhat, lastv):
        return yhat if target_mode=='price' else (lastv*np.exp(yhat))

    # Baseline
    yhat_naive_tr = last_tr
    yhat_naive_va = last_va
    naive_tr = mean_absolute_error(ypr_tr, yhat_naive_tr)
    naive_va = mean_absolute_error(ypr_va, yhat_naive_va)
    info(f"  [NAIVE] train={naive_tr:.6f}, valid={naive_va:.6f} (elapsed={tsecs(t0)})", quiet)

    # LR
    lr = Pipeline([('sc', StandardScaler()), ('lr', LinearRegression())])
    lr.fit(Xtr, ytr)
    mae_lr_tr = mean_absolute_error(ypr_tr, inv_price(lr.predict(Xtr), last_tr))
    mae_lr_va = mean_absolute_error(ypr_va, inv_price(lr.predict(Xva), last_va))
    info(f"  [LR]    train={mae_lr_tr:.6f}, valid={mae_lr_va:.6f} (elapsed={tsecs(t0)})", quiet)

    # MLP
    if KERAS_AVAILABLE:
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xva_s=sc.transform(Xva)
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

    # XGB (core)
    if XGB_CORE_AVAILABLE and Xtr.shape[0]>0 and Xva.shape[0]>0:
        try:
            yx_tr, yx_va = train_xgb_core_with_es(Xtr, ytr, Xva, yva, seed, xgb_rounds, xgb_es)
            mae_xgb_tr = mean_absolute_error(ypr_tr, inv_price(yx_tr, last_tr))
            mae_xgb_va = mean_absolute_error(ypr_va, inv_price(yx_va, last_va))
            info(f"  [XGB]   train={mae_xgb_tr:.6f}, valid={mae_xgb_va:.6f} (elapsed={tsecs(t0)})", quiet)
        except Exception as e:
            warn(f"XGBoost falló: {e}. Se omite XGB.")
            mae_xgb_tr = np.nan; mae_xgb_va = np.nan
    else:
        mae_xgb_tr = np.nan; mae_xgb_va = np.nan
        info(f"  [XGB]   omitido (no disponible o sin filas) (elapsed={tsecs(t0)})", quiet)

    return {
        'NAIVE_TRAIN_MAE': float(naive_tr),
        'NAIVE_VALID_MAE': float(naive_va),
        'LR_TRAIN_MAE': float(mae_lr_tr),
        'LR_VALID_MAE': float(mae_lr_va),
        'MLP_TRAIN_MAE': float(mae_mlp_tr),
        'MLP_VALID_MAE': float(mae_mlp_va),
        'XGB_TRAIN_MAE': float(mae_xgb_tr),
        'XGB_VALID_MAE': float(mae_xgb_va),
    }

# ---------- one combo ----------
def run_combo(df5m: pd.DataFrame, rule: str, H: float,
              feature_mode: str, target_mode: str,
              seed: int, mlp_width: int, mlp_second: int,
              budget: str, stride_tr: int, stride_va: int,
              cap_tr: int, cap_va: int, quiet: bool) -> Dict[str, float]:

    hdr = f"[{rule}][{feature_mode}/{target_mode}][H={H}h]"
    t0 = tnow()
    info(f"{hdr} preparar serie/resample…", quiet)
    df_rule = df5m if rule=='5T' else resample_close(df5m, 'close', rule)

    window = to_bars_or_nan(H, rule)
    if not np.isfinite(window):
        warn(f"[{rule}] Horizonte {H}h < tamaño de barra; se omite.")
        return {'periodicity':rule,'horizon_hours':H,'feature_mode':feature_mode,'target_mode':target_mode,
                'window_bars':np.nan,'n_samples':0,
                'NAIVE_TRAIN_MAE':np.nan,'NAIVE_VALID_MAE':np.nan,
                'LR_TRAIN_MAE':np.nan,'LR_VALID_MAE':np.nan,
                'MLP_TRAIN_MAE':np.nan,'MLP_VALID_MAE':np.nan,
                'XGB_TRAIN_MAE':np.nan,'XGB_VALID_MAE':np.nan}

    info(f"{hdr} building views (window={int(window)})…", quiet)
    X_view, y_t, y_p, last_c, idx = make_supervised_views(
        df_rule['close'].astype(float), window=int(window), horizon=int(window),
        feature_mode=feature_mode, target_mode=target_mode
    )
    m = y_t.shape[0]
    info(f"{hdr} m={m:,} muestras (elapsed={tsecs(t0)})", quiet)
    if m==0:
        warn(f"{hdr} sin muestras suficientes."); 
        return {'periodicity':rule,'horizon_hours':H,'feature_mode':feature_mode,'target_mode':target_mode,
                'window_bars':int(window),'n_samples':0,
                'NAIVE_TRAIN_MAE':np.nan,'NAIVE_VALID_MAE':np.nan,
                'LR_TRAIN_MAE':np.nan,'LR_VALID_MAE':np.nan,
                'MLP_TRAIN_MAE':np.nan,'MLP_VALID_MAE':np.nan,
                'XGB_TRAIN_MAE':np.nan,'XGB_VALID_MAE':np.nan}

    # auto budget if not overridden
    if any(v is None for v in (stride_tr, stride_va, cap_tr, cap_va)):
        a_st, a_sv, a_ct, a_cv, rounds, es = auto_budget(rule, H, budget)
        stride_tr = a_st if stride_tr is None else stride_tr
        stride_va = a_sv if stride_va is None else stride_va
        cap_tr    = a_ct if cap_tr    is None else cap_tr
        cap_va    = a_cv if cap_va    is None else cap_va
        xgb_rounds, xgb_es = rounds, es
    else:
        xgb_rounds, xgb_es = 700, 50

    info(f"{hdr} fit/eval (stride_tr={stride_tr}, stride_va={stride_va}, cap_tr={cap_tr}, cap_va={cap_va})…", quiet)
    metrics = fit_eval_modes(
        X_view, y_t, y_p, last_c, idx,
        seed, mlp_width, mlp_second, target_mode,
        stride_tr, stride_va, cap_tr, cap_va,
        xgb_rounds, xgb_es, quiet
    )
    out = {'periodicity':rule,'horizon_hours':H,'feature_mode':feature_mode,'target_mode':target_mode,
           'window_bars':int(window),'n_samples':int(m), **metrics}
    info(f"{hdr} DONE (total={tsecs(t0)})", quiet)
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Fast LR/MLP/XGB benchmark with views + budgeted thinning.")
    ap.add_argument('csv', type=str)
    ap.add_argument('--time-col', type=str, default=None)
    ap.add_argument('--close-col', type=str, default='close')
    ap.add_argument('--out', type=str, default='periodicity_benchmark.csv')
    ap.add_argument('--mlp-width', type=int, default=128)
    ap.add_argument('--mlp-second', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-tqdm', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--budget', type=str, choices=['small','medium','large'], default='medium')
    ap.add_argument('--stride-train', type=int, default=None)
    ap.add_argument('--stride-valid', type=int, default=None)
    ap.add_argument('--cap-train', type=int, default=None)
    ap.add_argument('--cap-valid', type=int, default=None)
    args = ap.parse_args()

    info("[LEGEND] LR=Linear Regression; MLP=Multi-Layer Perceptron; XGB=XGBoost; "
         "MAE=Mean Absolute Error; NAIVE=persistencia (C_t).", args.quiet)

    info("[INIT] Cargando CSV base (5T)…", args.quiet)
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        error(f"No se pudo leer el CSV: {e}"); sys.exit(1)
    info(f"[INIT] CSV cargado: shape={df.shape}", args.quiet)

    time_col = args.time_col
    if time_col is None:
        cands = [c for c in df.columns if c.lower() in ('time','timestamp','datetime','date')]
        if not cands: error("No se encontró columna temporal. Use --time-col."); sys.exit(1)
        time_col = cands[0]
        info(f"[INIT] Columna temporal detectada: {time_col}", args.quiet)

    if args.close_col not in df.columns:
        error(f"No existe la columna '{args.close_col}'. Use --close-col."); sys.exit(1)

    info("[INIT] Normalizando índice temporal…", args.quiet)
    df = ensure_datetime_index(df[[time_col, args.close_col]].rename(columns={args.close_col:'close'}), time_col)
    info(f"[INIT] Serie 5T lista: n={df.shape[0]} puntos.", args.quiet)

    periodicities = ['5T','15T','1H','4H','1D']
    horizons = [3.0, 72.0]
    feature_modes = ['raw_close','lognorm_close']
    target_modes  = ['price','logret_h']

    # single overall progress bar
    combos = [(r,H,f,t) for r in periodicities for H in horizons for f in feature_modes for t in target_modes]
    progress = tqdm(total=len(combos), disable=args.no_tqdm, desc="Grid progress", miniters=1)

    results: List[Dict[str,float]] = []
    for (rule,H,fmode,tmode) in combos:
        info(f"\n=== COMBO {progress.n+1}/{len(combos)}: {rule} | H={H}h | {fmode}/{tmode} ===", args.quiet)
        row = run_combo(df, rule, H, fmode, tmode,
                        seed=args.seed, mlp_width=args.mlp_width, mlp_second=args.mlp_second,
                        budget=args.budget,
                        stride_tr=args.stride_train, stride_va=args.stride_valid,
                        cap_tr=args.cap_train, cap_va=args.cap_valid,
                        quiet=args.quiet)
        results.append(row)
        progress.update(1)
    progress.close()

    info("[PIPE] Compilando resultados…", args.quiet)
    dfr = (pd.DataFrame(results)
           .sort_values(by=['horizon_hours','periodicity','feature_mode','target_mode'])
           .reset_index(drop=True))

    # deltas vs naive
    for model in ['LR','MLP','XGB']:
        dfr[f'{model}_VALID_DeltaMAE_vs_NAIVE']  = dfr[f'{model}_VALID_MAE'] - dfr['NAIVE_VALID_MAE']
        dfr[f'{model}_VALID_ImprovPct_vs_NAIVE'] = (dfr['NAIVE_VALID_MAE'] - dfr[f'{model}_VALID_MAE'])/dfr['NAIVE_VALID_MAE']*100.0

    # print compact table
    base_cols = ['horizon_hours','periodicity','feature_mode','target_mode','window_bars','n_samples',
                 'NAIVE_TRAIN_MAE','NAIVE_VALID_MAE',
                 'LR_TRAIN_MAE','LR_VALID_MAE',
                 'MLP_TRAIN_MAE','MLP_VALID_MAE',
                 'XGB_TRAIN_MAE','XGB_VALID_MAE']
    delta_cols = ['LR_VALID_DeltaMAE_vs_NAIVE','LR_VALID_ImprovPct_vs_NAIVE',
                  'MLP_VALID_DeltaMAE_vs_NAIVE','MLP_VALID_ImprovPct_vs_NAIVE',
                  'XGB_VALID_DeltaMAE_vs_NAIVE','XGB_VALID_ImprovPct_vs_NAIVE']
    def _fmt(v): 
        try: return f"{v:,.6f}"
        except Exception: return str(v)
    info("\n=== RESUMEN DE MAE (train/valid) + Δ vs NAIVE ===", False)
    print(dfr[base_cols+delta_cols].to_string(index=False, float_format=_fmt), flush=True)

    # leaderboards
        # leaderboards (robust to NaNs and empty/unsupported combos)
    def best_row_for_periodicity(sub: pd.DataFrame):
        models = ['LR', 'MLP', 'XGB']
        model_cols = [f'{m}_VALID_MAE' for m in models]

        # keep only rows that have samples and at least one finite model MAE
        mask_ok = (sub['n_samples'] > 0) & np.isfinite(sub[model_cols]).any(axis=1)
        sub_ok = sub.loc[mask_ok].copy()
        if sub_ok.empty:
            return None, None

        # per-row best (ignore NaNs)
        row_best_vals = np.nanmin(sub_ok[model_cols].values, axis=1)  # shape: [n_rows]
        # pick the row with smallest best-val
        ridx_ok = int(np.nanargmin(row_best_vals))
        best_row = sub_ok.iloc[ridx_ok]

        # which model won on that row?
        vals_this = [best_row[c] for c in model_cols]
        m_idx = int(np.nanargmin(vals_this))
        return best_row, models[m_idx]

    for H in horizons:
        print(f"\n=== LEADERBOARD — VALID (H={H}h) ===", flush=True)
        rows = []
        for rule in periodicities:
            sub = dfr[(dfr['horizon_hours'] == H) & (dfr['periodicity'] == rule)]
            best, best_model = best_row_for_periodicity(sub)
            if best is None:
                print(f"- {rule}: sin muestras representables o todos los modelos NaN.")
                continue

            # NAIVE of that periodicity/H: use the median NAIVE_VALID_MAE across the remaining usable rows
            usable = sub[(sub['n_samples'] > 0)]
            naive = float(np.nanmedian(usable['NAIVE_VALID_MAE']))

            best_valid = float(best[f'{best_model}_VALID_MAE'])
            imp = (naive - best_valid) / naive * 100.0

            rows.append({
                'periodicity': rule,
                'best_model': best_model,
                'feature_mode': best['feature_mode'],
                'target_mode': best['target_mode'],
                'window_bars': int(best['window_bars']),
                'n_samples': int(best['n_samples']),
                'best_valid_mae': best_valid,
                'naive_valid_mae': naive,
                'vs_naive_improv_pct': imp
            })

        if rows:
            ldf = (pd.DataFrame(rows)
                     .sort_values(by=['best_valid_mae','vs_naive_improv_pct'],
                                  ascending=[True, False]))
            def _fmt(v):
                try: return f"{v:,.6f}"
                except Exception: return str(v)
            cols = ['periodicity','best_model','feature_mode','target_mode',
                    'window_bars','n_samples','best_valid_mae',
                    'naive_valid_mae','vs_naive_improv_pct']
            print(ldf[cols].to_string(index=False, float_format=_fmt), flush=True)
        else:
            print("(sin filas útiles)", flush=True)


    info(f"\n[PIPE] Guardando resultados en: {args.out}", args.quiet)
    try:
        dfr.to_csv(args.out, index=False); info("[OK] Archivo escrito correctamente.", False)
    except Exception as e:
        error(f"No se pudo guardar {args.out}: {e}")

if __name__ == '__main__':
    main()
