#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
periodicity_benchmark.py — budget-aware
=======================================

- Same functionality as your last version (4 data modes, LR/MLP/XGB, MAE vs NAIVE, leaderboards)
- NEW: time-aware thinning (stride + cap) AFTER time split to avoid leakage.
- NEW: auto-heurstics per (periodicity, horizon) to prevent runaway runtimes/memory.

Tip:
    Use --budget {small,medium,large} or set --stride-* / --cap-* manually.

"""

import argparse, sys, math, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# XGBoost core API (preferred for compatibility & ES)
try:
    import warnings as _warnings
    import xgboost as xgb
    XGB_CORE_AVAILABLE = True
except Exception:
    XGB_CORE_AVAILABLE = False

# Keras/TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import regularizers
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# ------------------------------
# Print helpers
# ------------------------------
def info(msg: str, quiet: bool = False) -> None:
    if not quiet: print(msg, flush=True)

def warn(msg: str) -> None:
    print(f"[ADVERTENCIA] {msg}", file=sys.stderr, flush=True)

def error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ------------------------------
# Frequencies & resample
# ------------------------------
def hours_per_bar(rule: str) -> float:
    mapping = {'5T': 5.0/60.0, '15T': 15.0/60.0, '1H': 1.0, '4H': 4.0, '1D': 24.0}
    return mapping.get(rule, np.nan)

def to_bars_or_nan(hours: float, rule: str) -> float:
    hp = hours_per_bar(rule)
    if np.isnan(hp) or hp <= 0: return np.nan
    if hours < hp: return np.nan
    return int(max(1, round(hours / hp)))

def _normalize_rule(rule: str) -> str:
    mapping = {'5T': '5min', '15T': '15min', '1H': '1h', '4H': '4h', '1D': '1D'}
    return mapping.get(rule, rule)

def ensure_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=False)
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    df = df[~df.index.duplicated(keep='last')]
    return df

def resample_close(df: pd.DataFrame, close_col: str, rule: str) -> pd.DataFrame:
    out = df[[close_col]].resample(_normalize_rule(rule), label='right', closed='right').last().dropna()
    return out

# ------------------------------
# 4-mode dataset builders
# ------------------------------
def build_feature_matrix(close: np.ndarray, window: int, mode: str) -> np.ndarray:
    n = close.shape[0]
    m = n - window
    if m <= 0: return np.empty((0, window), float)
    X = np.empty((m, window), float)
    if mode == 'raw_close':
        for i in range(m): X[i, :] = close[i:i+window]
    elif mode == 'lognorm_close':
        logc = np.log(close)
        for i in range(m):
            seg = logc[i:i+window]
            X[i, :] = seg - seg[-1]
    else:
        raise ValueError(f"feature_mode desconocido: {mode}")
    return X

def build_target_vectors(close: np.ndarray, window: int, horizon: int, mode: str):
    n = close.shape[0]
    m = n - window - horizon + 1
    if m <= 0: return np.empty((0,), float), np.empty((0,), float), np.empty((0,), float)
    y_price = np.empty((m,), float)
    last_c  = np.empty((m,), float)
    for i in range(m):
        last_c[i]  = close[i + window - 1]            # C_t
        y_price[i] = close[i + window + horizon - 1]  # C_{t+h}
    if mode == 'price':
        y_target = y_price.copy()
    elif mode == 'logret_h':
        y_target = np.log(y_price) - np.log(last_c)
    else:
        raise ValueError(f"target_mode desconocido: {mode}")
    return y_target, y_price, last_c

def make_supervised_moded(close: pd.Series, window: int, horizon: int,
                          feature_mode: str, target_mode: str):
    c = close.astype(float).values
    n = c.shape[0]
    if n < (window + horizon):
        return np.empty((0, window), float), np.empty((0,), float), np.empty((0,), float), np.empty((0,), float), close.index[:0]
    X0 = build_feature_matrix(c, window=window, mode=feature_mode)
    y_target, y_price, last_c = build_target_vectors(c, window=window, horizon=horizon, mode=target_mode)
    m = y_target.shape[0]
    X = X0[:m, :]
    idx = close.index[window + horizon - 1 : window + horizon - 1 + m]
    return X, y_target, y_price, last_c, idx

# ------------------------------
# Time split + budget thinning
# ------------------------------
def time_based_train_valid_split(index: pd.DatetimeIndex, years_train: int = 2, years_valid: int = 1):
    if index.size == 0:
        return np.zeros((0,), bool), np.zeros((0,), bool)
    end = index.max()
    valid_start = end - pd.Timedelta(days=365 * years_valid)
    train_start = valid_start - pd.Timedelta(days=365 * years_train)
    min_time = index.min()
    if train_start < min_time: train_start = min_time
    is_valid = (index > valid_start) & (index <= end)
    is_train = (index > train_start) & (index <= valid_start)
    if is_train.sum() < 100: warn("Train muy pequeño (<100); resultados inestables.")
    if is_valid.sum() < 50: warn("Valid muy pequeño (<50); resultados ruidosos.")
    return np.asarray(is_train), np.asarray(is_valid)

def _apply_stride_and_cap(X, y_t, y_p, last_c, idx, stride:int, cap:int):
    """
    Apply after selecting a split. Keeps the most recent samples (cap) and
    optionally strides (every k-th). Preserves order (no shuffling).
    """
    n = X.shape[0]
    if n == 0: return X, y_t, y_p, last_c, idx
    # Cap: keep last 'cap' rows
    if cap is not None and cap > 0 and n > cap:
        X   = X[-cap:]
        y_t = y_t[-cap:]
        y_p = y_p[-cap:]
        last_c = last_c[-cap:]
        idx = idx[-cap:]
        n = cap
    # Stride
    s = max(1, int(stride)) if stride is not None else 1
    if s > 1:
        X   = X[::s]
        y_t = y_t[::s]
        y_p = y_p[::s]
        last_c = last_c[::s]
        idx = idx[::s]
    return X, y_t, y_p, last_c, idx

def _auto_budget(rule: str, horizon_hours: float, budget: str):
    """
    Returns (stride_train, stride_valid, cap_train, cap_valid, xgb_rounds, xgb_es)
    tuned to keep runtime sane for large 5T windows.
    """
    # Base defaults (safe)
    stride_tr, stride_va = 1, 1
    cap_tr, cap_va = None, None
    rounds, es = 800, 50

    # Heuristic scale per budget
    mult = {'small': 1.5, 'medium': 1.0, 'large': 0.7}.get(budget, 1.0)

    # Very large: 5T & 72h (window=864) → millions of rows
    if rule == '5T' and horizon_hours >= 48:
        stride_tr, stride_va = int(20*mult), int(20*mult)
        cap_tr, cap_va = int(200_000*mult), int(120_000*mult)
        rounds, es = int(600/mult), 40
    # Large: 5T & 3h (window=36)
    elif rule == '5T' and horizon_hours <= 6:
        stride_tr, stride_va = int(8*mult), int(8*mult)
        cap_tr, cap_va = int(250_000*mult), int(150_000*mult)
        rounds, es = int(700/mult), 50
    # Medium: 15T & 72h (window=288)
    elif rule == '15T' and horizon_hours >= 48:
        stride_tr, stride_va = int(6*mult), int(6*mult)
        cap_tr, cap_va = int(200_000*mult), int(100_000*mult)
        rounds, es = int(700/mult), 50
    # Light for hourly/daily
    elif rule in ('1H','4H','1D'):
        stride_tr, stride_va = 1, 1
        cap_tr, cap_va = None, None
        rounds, es = int(700/mult), 60

    return max(1,stride_tr), max(1,stride_va), cap_tr, cap_va, rounds, es

# ------------------------------
# XGBoost (core) with ES and budget
# ------------------------------
def train_xgb_core_with_es(Xtr, ytr, Xva, yva, seed: int, num_boost_round: int, es_rounds: int):
    if not XGB_CORE_AVAILABLE:
        raise RuntimeError("XGBoost core API no disponible.")
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "seed": seed,
        "tree_method": "hist",
        "max_depth": 3,
        "grow_policy": "lossguide",
        "max_leaves": 64,
        "min_child_weight": 10.0,
        "gamma": 1.0,
        "reg_lambda": 5.0,
        "reg_alpha": 0.5,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "colsample_bylevel": 0.6,
        "learning_rate": 0.03,
        "booster": "dart",
        "rate_drop": 0.1,
        "skip_drop": 0.1,
        "normalize_type": "tree",
        "sample_type": "uniform",
    }
    evals = [(dva, "valid")]
    # Silence the "Pass evals as keyword args" warnings in older xgb
    try:
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=FutureWarning)
            bst = xgb.train(params, dtr, num_boost_round=num_boost_round,
                            early_stopping_rounds=es_rounds, evals=evals, verbose_eval=False)
    except Exception:
        # fallback to gbtree / squarederror if dart unsupported
        params.pop("rate_drop", None); params.pop("skip_drop", None)
        params.pop("normalize_type", None); params.pop("sample_type", None)
        params["booster"] = "gbtree"
        try:
            bst = xgb.train(params, dtr, num_boost_round=num_boost_round,
                            early_stopping_rounds=es_rounds, evals=evals, verbose_eval=False)
        except Exception:
            params["objective"] = "reg:squarederror"
            bst = xgb.train(params, dtr, num_boost_round=num_boost_round,
                            early_stopping_rounds=es_rounds, evals=evals, verbose_eval=False)

    best_it = getattr(bst, "best_iteration", None)
    def _predict(dm):
        if best_it is None: return bst.predict(dm)
        try: return bst.predict(dm, iteration_range=(0, int(best_it)+1))
        except TypeError:
            try: return bst.predict(dm, ntree_limit=int(best_it)+1)
            except TypeError: return bst.predict(dm)
    return _predict(dtr), _predict(dva)

# ------------------------------
# Modeling & evaluation (MAE in PRICE space)
# ------------------------------
def fit_and_eval_models_moded(X, y_target, y_price, last_c, idx,
                              seed, mlp_width, mlp_second, target_mode,
                              stride_train, stride_valid, cap_train, cap_valid,
                              xgb_rounds, xgb_es, quiet: bool=False) -> Dict[str, float]:
    is_train, is_valid = time_based_train_valid_split(idx, years_train=2, years_valid=1)
    if is_train.sum() == 0 or is_valid.sum() == 0:
        warn("Split temporal vacío; se omite evaluación.")
        return {k: np.nan for k in [
            'NAIVE_TRAIN_MAE','NAIVE_VALID_MAE','LR_TRAIN_MAE','LR_VALID_MAE',
            'MLP_TRAIN_MAE','MLP_VALID_MAE','XGB_TRAIN_MAE','XGB_VALID_MAE'
        ]}

    # Slice splits
    Xtr, Xva = X[is_train], X[is_valid]
    ytr, yva = y_target[is_train], y_target[is_valid]
    ypr_tr, ypr_va = y_price[is_train], y_price[is_valid]
    last_tr, last_va = last_c[is_train], last_c[is_valid]
    idx_tr, idx_va = idx[is_train], idx[is_valid]

    # Apply thinning (no leakage)
    Xtr, ytr, ypr_tr, last_tr, idx_tr = _apply_stride_and_cap(Xtr, ytr, ypr_tr, last_tr, idx_tr, stride_train, cap_train)
    Xva, yva, ypr_va, last_va, idx_va = _apply_stride_and_cap(Xva, yva, ypr_va, last_va, idx_va, stride_valid, cap_valid)

    def inv_to_price(yhat, last_vec):
        if target_mode == 'price': return yhat
        elif target_mode == 'logret_h': return last_vec * np.exp(yhat)
        else: raise ValueError("target_mode desconocido")

    # Baseline NAIVE
    yhat_naive_tr_price = last_tr
    yhat_naive_va_price = last_va
    mae_naive_tr = mean_absolute_error(ypr_tr, yhat_naive_tr_price)
    mae_naive_va = mean_absolute_error(ypr_va, yhat_naive_va_price)

    # LR
    lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    lr_pipe.fit(Xtr, ytr)
    yhat_lr_tr_price = inv_to_price(lr_pipe.predict(Xtr), last_tr)
    yhat_lr_va_price = inv_to_price(lr_pipe.predict(Xva), last_va)
    mae_lr_tr = mean_absolute_error(ypr_tr, yhat_lr_tr_price)
    mae_lr_va = mean_absolute_error(ypr_va, yhat_lr_va_price)

    # MLP
    if KERAS_AVAILABLE:
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr); Xva_s = scaler.transform(Xva)
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
        yhat_mlp_tr = model.predict(Xtr_s, verbose=0).ravel()
        yhat_mlp_va = model.predict(Xva_s, verbose=0).ravel()
    else:
        mlp = MLPRegressor(hidden_layer_sizes=(mlp_width, mlp_second),
                           activation='relu', solver='adam',
                           alpha=1e-4, learning_rate='adaptive', learning_rate_init=1e-3,
                           max_iter=400, shuffle=True, random_state=seed,
                           early_stopping=True, n_iter_no_change=20, tol=1e-4, verbose=False)
        mlp_pipe = Pipeline([('scaler', StandardScaler()), ('mlp', mlp)])
        mlp_pipe.fit(Xtr, ytr)
        yhat_mlp_tr = mlp_pipe.predict(Xtr)
        yhat_mlp_va = mlp_pipe.predict(Xva)

    yhat_mlp_tr_price = inv_to_price(yhat_mlp_tr, last_tr)
    yhat_mlp_va_price = inv_to_price(yhat_mlp_va, last_va)
    mae_mlp_tr = mean_absolute_error(ypr_tr, yhat_mlp_tr_price)
    mae_mlp_va = mean_absolute_error(ypr_va, yhat_mlp_va_price)

    # XGB (core) — budgeted
    if XGB_CORE_AVAILABLE and Xtr.shape[0] > 0 and Xva.shape[0] > 0:
        try:
            yhat_xgb_tr, yhat_xgb_va = train_xgb_core_with_es(Xtr, ytr, Xva, yva, seed=seed,
                                                              num_boost_round=xgb_rounds, es_rounds=xgb_es)
            yhat_xgb_tr_price = inv_to_price(yhat_xgb_tr, last_tr)
            yhat_xgb_va_price = inv_to_price(yhat_xgb_va, last_va)
            mae_xgb_tr = mean_absolute_error(ypr_tr, yhat_xgb_tr_price)
            mae_xgb_va = mean_absolute_error(ypr_va, yhat_xgb_va_price)
        except Exception as e:
            warn(f"XGBoost core falló: {e}. Se omite XGB.")
            mae_xgb_tr = np.nan; mae_xgb_va = np.nan
    else:
        mae_xgb_tr = np.nan; mae_xgb_va = np.nan

    return {
        'NAIVE_TRAIN_MAE': float(mae_naive_tr),
        'NAIVE_VALID_MAE': float(mae_naive_va),
        'LR_TRAIN_MAE': float(mae_lr_tr),
        'LR_VALID_MAE': float(mae_lr_va),
        'MLP_TRAIN_MAE': float(mae_mlp_tr),
        'MLP_VALID_MAE': float(mae_mlp_va),
        'XGB_TRAIN_MAE': float(mae_xgb_tr),
        'XGB_VALID_MAE': float(mae_xgb_va),
    }

# ------------------------------
# One evaluation (periodicity, horizon, modes)
# ------------------------------
def evaluate_periodicity_horizon_mode(df5m: pd.DataFrame, rule: str, horizon_hours: float,
                                      feature_mode: str, target_mode: str,
                                      seed: int, mlp_width: int, mlp_second: int,
                                      budget: str, stride_train: int, stride_valid: int,
                                      cap_train: int, cap_valid: int, quiet: bool) -> Dict[str, float]:
    info(f"[{rule}][{feature_mode}/{target_mode}] Preparando dataset para H={horizon_hours}h…", quiet)
    df_rule = resample_close(df5m, 'close', rule) if rule != '5T' else df5m.copy()
    window = to_bars_or_nan(horizon_hours, rule)
    horizon_bars = window
    if (window is np.nan) or np.isnan(window):
        warn(f"[{rule}] Horizonte {horizon_hours}h < tamaño de barra; se omite.")
        return {'periodicity': rule, 'horizon_hours': horizon_hours, 'feature_mode': feature_mode, 'target_mode': target_mode,
                'window_bars': np.nan, 'n_samples': 0,
                'NAIVE_TRAIN_MAE': np.nan, 'NAIVE_VALID_MAE': np.nan,
                'LR_TRAIN_MAE': np.nan, 'LR_VALID_MAE': np.nan,
                'MLP_TRAIN_MAE': np.nan, 'MLP_VALID_MAE': np.nan,
                'XGB_TRAIN_MAE': np.nan, 'XGB_VALID_MAE': np.nan}

    close = df_rule['close'].astype(float)
    X, y_t, y_price, last_c, idx = make_supervised_moded(close, window=int(window), horizon=int(horizon_bars),
                                                         feature_mode=feature_mode, target_mode=target_mode)
    if X.shape[0] == 0:
        warn(f"[{rule}] Sin datos suficientes (window={window}, horizon={horizon_bars}).")
        return {'periodicity': rule, 'horizon_hours': horizon_hours, 'feature_mode': feature_mode, 'target_mode': target_mode,
                'window_bars': int(window), 'n_samples': 0,
                'NAIVE_TRAIN_MAE': np.nan, 'NAIVE_VALID_MAE': np.nan,
                'LR_TRAIN_MAE': np.nan, 'LR_VALID_MAE': np.nan,
                'MLP_TRAIN_MAE': np.nan, 'MLP_VALID_MAE': np.nan,
                'XGB_TRAIN_MAE': np.nan, 'XGB_VALID_MAE': np.nan}

    # Auto budget if user didn't override
    if stride_train is None or stride_valid is None or cap_train is None or cap_valid is None:
        auto_st, auto_sv, auto_ct, auto_cv, auto_rounds, auto_es = _auto_budget(rule, horizon_hours, budget)
        if stride_train is None: stride_train = auto_st
        if stride_valid is None: stride_valid = auto_sv
        if cap_train   is None: cap_train   = auto_ct
        if cap_valid   is None: cap_valid   = auto_cv
        xgb_rounds, xgb_es = auto_rounds, auto_es
    else:
        # reasonable defaults for rounds if all strides/caps were passed manually
        xgb_rounds, xgb_es = 800, 50

    metrics = fit_and_eval_models_moded(
        X, y_t, y_price, last_c, idx,
        seed=seed, mlp_width=mlp_width, mlp_second=mlp_second,
        target_mode=target_mode,
        stride_train=stride_train, stride_valid=stride_valid,
        cap_train=cap_train, cap_valid=cap_valid,
        xgb_rounds=xgb_rounds, xgb_es=xgb_es,
        quiet=quiet
    )
    out = {'periodicity': rule, 'horizon_hours': horizon_hours,
           'feature_mode': feature_mode, 'target_mode': target_mode,
           'window_bars': int(window), 'n_samples': int(X.shape[0]), **metrics}
    return out

# ------------------------------
# Main (with leaderboards and deltas vs NAIVE)
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Benchmark LR/MLP/XGB con 4 modos (features/target) y presupuesto de cómputo (stride + cap)."
    )
    ap.add_argument('csv', type=str)
    ap.add_argument('--time-col', type=str, default=None)
    ap.add_argument('--close-col', type=str, default='close')
    ap.add_argument('--out', type=str, default='periodicity_benchmark.csv')
    ap.add_argument('--mlp-width', type=int, default=128)
    ap.add_argument('--mlp-second', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-tqdm', action='store_true')
    ap.add_argument('--quiet', action='store_true')

    # NEW: budget knobs
    ap.add_argument('--budget', type=str, choices=['small','medium','large'], default='medium',
                    help="Controla stride/cap y los rounds de XGB automáticamente (default: medium).")
    ap.add_argument('--stride-train', type=int, default=None, help="Sobrescribe stride de train (mantén cada k-ésima muestra).")
    ap.add_argument('--stride-valid', type=int, default=None, help="Sobrescribe stride de valid.")
    ap.add_argument('--cap-train', type=int, default=None, help="Sobrescribe máximo de filas en train (mantiene las más recientes).")
    ap.add_argument('--cap-valid', type=int, default=None, help="Sobrescribe máximo de filas en valid.")

    args = ap.parse_args()

    info("[LEGEND] LR=Linear Regression; MLP=Multi-Layer Perceptron; XGB=XGBoost (core API); "
         "MAE=Mean Absolute Error; NAIVE=persistencia (predice C_t).", args.quiet)

    info("[INIT] Cargando CSV base (5T)…", args.quiet)
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        error(f"No se pudo leer el CSV: {e}"); sys.exit(1)
    info(f"[INIT] CSV cargado: shape={df.shape}", args.quiet)

    time_col = args.time_col
    if time_col is None:
        candidates = [c for c in df.columns if c.lower() in ('time','timestamp','datetime','date')]
        if not candidates:
            error("No se encontró columna temporal. Use --time-col."); sys.exit(1)
        time_col = candidates[0]
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

    results: List[Dict[str, float]] = []
    it_rules = periodicities if args.no_tqdm else tqdm(periodicities, desc="Evaluando periodicidades", miniters=1)
    for rule in it_rules:
        it_H = horizons if args.no_tqdm else tqdm(horizons, desc=f"[{rule}] Horizontes", leave=False, miniters=1)
        for H in it_H:
            it_feat = feature_modes if args.no_tqdm else tqdm(feature_modes, desc=f"[{rule}][{H}h] feature_modes", leave=False)
            for fmode in it_feat:
                it_tgt = target_modes if args.no_tqdm else tqdm(target_modes, desc=f"[{rule}][{H}h] target_modes", leave=False)
                for tmode in it_tgt:
                    row = evaluate_periodicity_horizon_mode(
                        df5m=df, rule=rule, horizon_hours=H,
                        feature_mode=fmode, target_mode=tmode,
                        seed=args.seed, mlp_width=args.mlp_width, mlp_second=args.mlp_second,
                        budget=args.budget,
                        stride_train=args.stride_train, stride_valid=args.stride_valid,
                        cap_train=args.cap_train, cap_valid=args.cap_valid,
                        quiet=args.quiet
                    )
                    results.append(row)

    info("[PIPE] Compilando resultados…", args.quiet)
    dfr = (pd.DataFrame(results)
           .sort_values(by=['horizon_hours','periodicity','feature_mode','target_mode'])
           .reset_index(drop=True))

    # Deltas vs NAIVE (valid)
    for model in ['LR','MLP','XGB']:
        dfr[f'{model}_VALID_DeltaMAE_vs_NAIVE'] = dfr[f'{model}_VALID_MAE'] - dfr['NAIVE_VALID_MAE']
        dfr[f'{model}_VALID_ImprovPct_vs_NAIVE'] = (dfr['NAIVE_VALID_MAE'] - dfr[f'{model}_VALID_MAE']) / dfr['NAIVE_VALID_MAE'] * 100.0

    info("\n=== RESUMEN DE MAE (train y valid) — espacio de PRECIO ===", False)
    base_cols = [
        'horizon_hours','periodicity','feature_mode','target_mode','window_bars','n_samples',
        'NAIVE_TRAIN_MAE','NAIVE_VALID_MAE',
        'LR_TRAIN_MAE','LR_VALID_MAE',
        'MLP_TRAIN_MAE','MLP_VALID_MAE',
        'XGB_TRAIN_MAE','XGB_VALID_MAE'
    ]
    delta_cols = [
        'LR_VALID_DeltaMAE_vs_NAIVE','LR_VALID_ImprovPct_vs_NAIVE',
        'MLP_VALID_DeltaMAE_vs_NAIVE','MLP_VALID_ImprovPct_vs_NAIVE',
        'XGB_VALID_DeltaMAE_vs_NAIVE','XGB_VALID_ImprovPct_vs_NAIVE'
    ]
    def _fmt(v):
        try: return f"{v:,.6f}"
        except Exception: return str(v)
    print(dfr[base_cols + delta_cols].to_string(index=False, float_format=_fmt), flush=True)

    # Leaderboards per horizon
    def best_row_for_periodicity(subdf: pd.DataFrame) -> pd.Series:
        models = ['LR','MLP','XGB']
        tmp = subdf.copy()
        tmp['row_best_mae'] = np.nanmin(np.vstack([tmp[f'{m}_VALID_MAE'].values for m in models]), axis=0)
        best_model_idx = np.nanargmin(np.vstack([tmp[f'{m}_VALID_MAE'].values for m in models]), axis=0)
        model_names = np.array(models)[best_model_idx]
        tmp['row_best_model'] = model_names
        ridx = int(np.nanargmin(tmp['row_best_mae'].values))
        return tmp.iloc[ridx]

    for H in horizons:
        print(f"\n=== LEADERBOARD — VALID (H={H}h) ===", flush=True)
        rows = []
        for rule in periodicities:
            sub = dfr[(dfr['horizon_hours']==H) & (dfr['periodicity']==rule)]
            if sub.shape[0] == 0: continue
            best = best_row_for_periodicity(sub)
            naive_valid = float(np.nanmedian(sub['NAIVE_VALID_MAE'].values))
            best_mae = float(best['row_best_mae'])
            imp = (naive_valid - best_mae) / naive_valid * 100.0
            rows.append({
                'periodicity': rule,
                'best_model': best['row_best_model'],
                'feature_mode': best['feature_mode'],
                'target_mode': best['target_mode'],
                'window_bars': int(best['window_bars']),
                'n_samples': int(best['n_samples']),
                'best_valid_mae': best_mae,
                'naive_valid_mae': naive_valid,
                'vs_naive_improv_pct': imp
            })
        if rows:
            ldf = pd.DataFrame(rows).sort_values(by=['best_valid_mae','vs_naive_improv_pct'], ascending=[True,False])
            cols = ['periodicity','best_model','feature_mode','target_mode','window_bars','n_samples','best_valid_mae','naive_valid_mae','vs_naive_improv_pct']
            print(ldf[cols].to_string(index=False, float_format=_fmt), flush=True)
        else:
            print("(sin filas)")

    info(f"\n[PIPE] Guardando resultados en: {args.out}", args.quiet)
    try:
        dfr.to_csv(args.out, index=False); info("[OK] Archivo escrito correctamente.", False)
    except Exception as e:
        error(f"No se pudo guardar {args.out}: {e}")

if __name__ == '__main__':
    main()
