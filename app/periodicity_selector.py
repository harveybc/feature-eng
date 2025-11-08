#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
periodicity_selector_verbose.py — HF(3–6h) / LF(3–6d) horizon-focused
"""

import argparse, sys, math
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import median_abs_deviation
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

# ---------------- Logging ----------------
def info(msg: str, quiet: bool=False) -> None:
    if not quiet: print(msg, flush=True)
def warn(msg: str) -> None:
    print(f"[ADVERTENCIA] {msg}", file=sys.stderr, flush=True)
def error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ------------- Robust utils -------------
def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = median_abs_deviation(x, nan_policy='omit', scale='normal')
    if mad == 0 or np.isnan(mad): return np.zeros_like(x)
    return (x - med) / mad

def ensure_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=False, errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    df = df[~df.index.duplicated(keep='last')]
    return df

def resample_close(df: pd.DataFrame, close_col: str, rule: str) -> pd.DataFrame:
    return df[[close_col]].resample(rule, label='right', closed='right').last().dropna()

def log_returns(close: pd.Series) -> pd.Series:
    c = close.astype(float).copy()
    min_pos = c[c > 0.0].min()
    if pd.isna(min_pos) or min_pos <= 0.0:
        c = c + (abs(c.min()) + 1e-12)
    return np.log(c).diff().dropna()

# -------- Periodicity <-> scale ----------
def hours_per_bar(rule: str) -> float:
    return {'5T': 5.0/60.0, '15T': 15.0/60.0, '1H': 1.0, '4H': 4.0, '1D': 24.0}.get(rule, np.nan)

def to_bars_or_nan(hours: float, rule: str) -> float:
    """Return bars for a real horizon; NaN if horizon < bar size (not representable)."""
    hp = hours_per_bar(rule)
    if np.isnan(hp) or hp <= 0: return np.nan
    if hours < hp: return np.nan
    return int(max(1, round(hours / hp)))

# -------------- Thinning -----------------
def _thin_series_for_mi(x: np.ndarray, max_samples: int) -> np.ndarray:
    if x.size <= max_samples: return x
    step = int(np.ceil(x.size / max_samples))
    return x[::step]

def _thin_binary_sequence(s: np.ndarray, max_samples: int) -> np.ndarray:
    if s.size <= max_samples: return s
    step = int(np.ceil(s.size / max_samples))
    return s[::step]

# --------------- Metrics -----------------
def mutual_information_horizon(r: pd.Series, horizon_bars: int, k: int=5,
                               max_samples: int=800_000, use_tqdm: bool=False) -> float:
    if (horizon_bars is None) or (np.isnan(horizon_bars)) or (horizon_bars < 1):
        return np.nan
    x_full = np.asarray(r.dropna(), dtype=float)
    if x_full.size <= (horizon_bars + 10): return np.nan
    x = _thin_series_for_mi(x_full, max_samples=max_samples)
    n_eff = x.size
    if n_eff <= (horizon_bars + 10): return np.nan
    rng = range(n_eff - horizon_bars)
    if use_tqdm: rng = tqdm(rng, desc=f"MI: h={horizon_bars} barras", leave=False, miniters=1)
    X = np.empty((n_eff - horizon_bars, 1), dtype=float)
    Y = np.empty((n_eff - horizon_bars,), dtype=float)
    for i in rng:
        X[i, 0] = x[i]
        Y[i] = np.sum(x[i+1:i+1+horizon_bars])
    try:
        mi_nats = mutual_info_regression(X, Y, n_neighbors=k, random_state=42)
        return float(mi_nats[0] / math.log(2.0))
    except Exception:
        return np.nan

def acf_sum_abs(r: pd.Series, max_lag: int) -> float:
    if (max_lag is None) or (np.isnan(max_lag)) or (max_lag < 1): return np.nan
    x = np.asarray(r.dropna(), dtype=float)
    n = x.size
    if n < max_lag + 5: return np.nan
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 0: return np.nan
    fft = np.fft.rfft(x, n=2**int(np.ceil(np.log2(2*n - 1))))
    acf_full = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acf_full = acf_full / (var * np.arange(n, 0, -1))
    return float(np.sum(np.abs(acf_full[1:max_lag+1])))

def snr_smoothing_returns(r: pd.Series, smooth_window: int) -> float:
    """Safe rolling: supports window=1/2; min_periods ≤ window."""
    if (smooth_window is None) or (np.isnan(smooth_window)) or (smooth_window < 1):
        return np.nan
    x = pd.Series(r.dropna().values, index=r.dropna().index)
    if x.size < max(8, 4*smooth_window):  # need some data
        return np.nan
    window = int(smooth_window)
    minp = min(window, max(1, window//3))  # ensure ≤ window
    m = x.rolling(window=window, center=True, min_periods=minp).mean()
    resid = x - m
    var_sig = np.nanvar(m.values)
    var_noise = np.nanvar(resid.values)
    if (var_noise is None) or np.isnan(var_noise) or (var_noise <= 0): return np.nan
    return float(var_sig / var_noise)

def permutation_entropy_weighted(r: pd.Series, m: int=5, tau: int=1, use_tqdm: bool=True) -> float:
    x = np.asarray(r.dropna(), dtype=float)
    N = x.size
    L = N - (m - 1) * tau
    if L <= 0: return np.nan
    M = np.empty((L, m), dtype=float)
    for i in range(m):
        M[:, i] = x[i * tau : i * tau + L]
    order = np.argsort(M, axis=1, kind='mergesort')
    it = range(order.shape[0])
    if use_tqdm: it = tqdm(it, desc="WPE: codificando patrones", leave=False, miniters=1)
    from collections import defaultdict
    weight_map = defaultdict(float)
    for idx in it:
        pat = tuple(order[idx])
        weight_map[pat] += float(np.var(M[idx, :]))
    weights = np.array(list(weight_map.values()), dtype=float)
    if weights.size == 0: return np.nan
    weights = weights / np.sum(weights)
    weights = np.maximum(weights, 1e-300)
    Hw = -np.sum(weights * np.log(weights))
    K = math.factorial(m)
    return float(np.clip(Hw / np.log(K), 0.0, 1.0))

def lz_entropy_rate_sign(r: pd.Series, method: str='zlib', max_samples: int=100_000) -> float:
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 256: return np.nan
    s = (x > 0).astype(np.uint8)
    if method == 'zlib':
        import zlib
        b = bytes(s.tolist())
        if len(b) == 0: return np.nan
        comp = zlib.compress(b, level=6)
        return float((len(comp) * 8.0) / len(b))
    # lz76 with thinning
    s = _thin_binary_sequence(s, max_samples=max_samples)
    n = int(s.size)
    if n < 256: return np.nan
    seq = s.tolist()
    i, k, l, c = 0, 1, 1, 1
    while True:
        if i + k > n: c += 1; break
        if seq[i:i+k] == seq[l:l+k]:
            k += 1
            if l + k > n: c += 1; break
        else:
            l += 1
            if l == i + k:
                c += 1; i += k
                if i + 1 > n: break
                l = 0; k = 1
    return float((c * math.log(n)) / n)

# ---------- Ω band-limited ----------
def spectral_forecastability_bandlimited(r: pd.Series, band_hours: Tuple[float, float], rule: str) -> float:
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 256: return np.nan
    hpb = hours_per_bar(rule)
    if np.isnan(hpb) or hpb <= 0: return np.nan
    fs = 1.0 / hpb  # samples per hour
    # choose a sane nperseg for low freq coverage
    nperseg = int(max(256, min(8192, 8 * fs)))
    if nperseg > x.size: nperseg = max(128, x.size // 2)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, detrend='constant',
                   return_onesided=True, scaling='density')
    Pxx = np.maximum(np.asarray(Pxx, float), 1e-300)
    with np.errstate(divide='ignore'):
        T = 1.0 / np.maximum(f, 1e-12)  # hours
    Tmin, Tmax = band_hours
    mask = (T >= Tmin) & (T <= Tmax)
    if not np.any(mask): return np.nan
    p = Pxx[mask]
    p = p / np.sum(p)
    p = np.maximum(p, 1e-300)
    Hs = -np.sum(p * np.log(p))
    K = p.size
    denom = np.log(K) if K > 1 else 1.0
    return float(np.clip(1.0 - Hs/denom, 0.0, 1.0))

# --------- Evaluation per P ----------
def evaluate_periodicity(df_close: pd.DataFrame, rule: str,
                         mi_k: int, max_mi_samples: int,
                         lz_method: str, max_lz_samples: int,
                         use_tqdm: bool, quiet: bool,
                         lambda_penalty: float) -> Dict[str, float]:
    info(f"[{rule}] Preparando serie y retornos…", quiet)
    dfp = resample_close(df_close, 'close', rule) if rule != '5T' else df_close.copy()
    r = log_returns(dfp['close'])

    HF_hours = [3.0, 4.0, 5.0, 6.0]
    LF_hours = [72.0, 96.0, 120.0, 144.0]

    # MI lists with horizon guards (skip if horizon < bar)
    info(f"[{rule}] MI HF (3–6h)…", quiet)
    mi_hf_vals = []
    for H in (tqdm(HF_hours, desc=f"MI HF ({rule})", leave=False) if use_tqdm else HF_hours):
        hbars = to_bars_or_nan(H, rule)
        mi_hf_vals.append(np.nan if (hbars is np.nan or np.isnan(hbars)) else
                          mutual_information_horizon(r, hbars, k=mi_k, max_samples=max_mi_samples, use_tqdm=False))
    mi_hf_med = float(np.nanmedian(mi_hf_vals)) if np.any(~np.isnan(mi_hf_vals)) else np.nan

    info(f"[{rule}] MI LF (3–6d)…", quiet)
    mi_lf_vals = []
    for H in (tqdm(LF_hours, desc=f"MI LF ({rule})", leave=False) if use_tqdm else LF_hours):
        hbars = to_bars_or_nan(H, rule)
        mi_lf_vals.append(np.nan if (hbars is np.nan or np.isnan(hbars)) else
                          mutual_information_horizon(r, hbars, k=mi_k, max_samples=max_mi_samples, use_tqdm=False))
    mi_lf_med = float(np.nanmedian(mi_lf_vals)) if np.any(~np.isnan(mi_lf_vals)) else np.nan

    # Ω in bands
    info(f"[{rule}] Ω_HF 3–12h…", quiet)
    omega_hf = spectral_forecastability_bandlimited(r, band_hours=(3.0, 12.0), rule=rule)
    info(f"[{rule}] Ω_LF 2–10d…", quiet)
    omega_lf = spectral_forecastability_bandlimited(r, band_hours=(48.0, 240.0), rule=rule)

    # ACF/SNR windows (skip if horizon < bar)
    h_s_max = to_bars_or_nan(6.0, rule)
    h_l_max = to_bars_or_nan(144.0, rule)

    info(f"[{rule}] ACF_sum_abs HF (≤6h)…", quiet)
    acf_hf = acf_sum_abs(r, int(h_s_max)) if (h_s_max==h_s_max) else np.nan
    info(f"[{rule}] ACF_sum_abs LF (≤6d)…", quiet)
    acf_lf = acf_sum_abs(r, int(h_l_max)) if (h_l_max==h_l_max) else np.nan

    info(f"[{rule}] SNR HF (ventana=6h)…", quiet)
    snr_hf = snr_smoothing_returns(r, int(h_s_max)) if (h_s_max==h_s_max) else np.nan
    info(f"[{rule}] SNR LF (ventana=6d)…", quiet)
    snr_lf = snr_smoothing_returns(r, int(h_l_max)) if (h_l_max==h_l_max) else np.nan

    # Global penalties (safe)
    info(f"[{rule}] WPE global…", quiet)
    wpe = permutation_entropy_weighted(r, m=5, tau=1, use_tqdm=use_tqdm)
    info(f"[{rule}] LZ global (método={lz_method})…", quiet)
    lz_rate = lz_entropy_rate_sign(r, method=lz_method, max_samples=max_lz_samples)

    return {
        'periodicity': rule,
        'n_points': int(dfp.shape[0]),
        # HF
        'mi_hf_med_bits': mi_hf_med,
        'omega_hf': omega_hf,
        'acf_hf_sum_abs': acf_hf,
        'snr_hf': snr_hf,
        # LF
        'mi_lf_med_bits': mi_lf_med,
        'omega_lf': omega_lf,
        'acf_lf_sum_abs': acf_lf,
        'snr_lf': snr_lf,
        # penalties
        'wpe': wpe,
        'lz_entropy_rate': lz_rate,
        # MI breakdown (debug)
        'mi_hf_h3_bits': mi_hf_vals[0],
        'mi_hf_h4_bits': mi_hf_vals[1],
        'mi_hf_h5_bits': mi_hf_vals[2],
        'mi_hf_h6_bits': mi_hf_vals[3],
        'mi_lf_d3_bits': mi_lf_vals[0],
        'mi_lf_d4_bits': mi_lf_vals[1],
        'mi_lf_d5_bits': mi_lf_vals[2],
        'mi_lf_d6_bits': mi_lf_vals[3],
    }

# ---------- Composite scores ----------
def composite_scores(df_metrics: pd.DataFrame, lambda_penalty: float) -> pd.DataFrame:
    hf_plus = ['mi_hf_med_bits', 'omega_hf', 'acf_hf_sum_abs', 'snr_hf']
    lf_plus = ['mi_lf_med_bits', 'omega_lf', 'acf_lf_sum_abs', 'snr_lf']
    penalties = ['wpe', 'lz_entropy_rate']
    for c in hf_plus + lf_plus + penalties:
        df_metrics[f'z_{c}'] = robust_zscore(df_metrics[c].values)
    df_metrics['S_HF'] = sum(df_metrics[f'z_{c}'] for c in hf_plus) - \
                         lambda_penalty * (df_metrics['z_wpe'] + df_metrics['z_lz_entropy_rate'])
    df_metrics['S_LF'] = sum(df_metrics[f'z_{c}'] for c in lf_plus) - \
                         lambda_penalty * (df_metrics['z_wpe'] + df_metrics['z_lz_entropy_rate'])
    # ranks (keep independent orderings only for print purposes)
    df_metrics['rank_HF'] = df_metrics['S_HF'].rank(ascending=False, method='dense').astype(int)
    df_metrics['rank_LF'] = df_metrics['S_LF'].rank(ascending=False, method='dense').astype(int)
    return df_metrics

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Selector de periodicidad orientado a horizontes HF(3–6h) y LF(3–6d) con tqdm."
    )
    ap.add_argument('csv', type=str, help='CSV base (5 minutos).')
    ap.add_argument('--time-col', type=str, default=None, help='Columna temporal (auto).')
    ap.add_argument('--close-col', type=str, default='close', help="Columna close (default: 'close').")
    ap.add_argument('--out', type=str, default='periodicity_analysis.csv', help="CSV de salida.")
    ap.add_argument('--mi-nneighbors', type=int, default=5, help="Vecinos k para MI.")
    ap.add_argument('--max-mi-samples', type=int, default=800_000, help="Cap de muestras MI (thinning).")
    ap.add_argument('--max-lz-samples', type=int, default=100_000, help="Cap de símbolos LZ (thinning).")
    ap.add_argument('--lz-method', type=str, default='zlib', choices=['lz76','zlib'],
                    help="LZ exacto (thinning) o proxy zlib (rápido).")
    ap.add_argument('--lambda-penalty', type=float, default=0.3,
                    help="Peso penalizador para WPE y LZ en S_HF/S_LF.")
    ap.add_argument('--quiet', action='store_true', help="Menos prints (mantiene barras).")
    ap.add_argument('--no-tqdm', action='store_true', help="Desactivar tqdm.")
    ap.add_argument('--progress', action='store_true', help="Forzar tqdm.")
    args = ap.parse_args()

    info("[INIT] Cargando CSV…", args.quiet)
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        error(f"No se pudo leer el CSV: {e}"); sys.exit(1)
    info(f"[INIT] CSV cargado: shape={df.shape}", args.quiet)

    time_col = args.time_col
    if time_col is None:
        candidates = [c for c in df.columns if c.lower() in ('time', 'timestamp', 'datetime', 'date')]
        if len(candidates) == 0:
            error("No se encontró columna temporal. Use --time-col."); sys.exit(1)
        time_col = candidates[0]; info(f"[INIT] Columna temporal: {time_col}", args.quiet)
    close_col = args.close_col
    if close_col not in df.columns:
        error(f"No existe la columna '{close_col}'. Use --close-col."); sys.exit(1)

    info("[INIT] Normalizando índice temporal…", args.quiet)
    df = ensure_datetime_index(df[[time_col, close_col]].rename(columns={close_col: 'close'}), time_col)
    info(f"[INIT] Serie lista: n={df.shape[0]} puntos (5T)", args.quiet)
    if df.shape[0] < 1000: warn("Muy pocos puntos; métricas pueden ser inestables.")

    auto_use_tqdm = (df.shape[0] > 100_000) or args.progress
    use_tqdm = False if args.no_tqdm else auto_use_tqdm

    periodicities = ['5T','15T','1H','4H','1D']
    info(f"[PIPE] Periodicidades objetivo: {periodicities}", args.quiet)

    rows: List[Dict[str, float]] = []
    iterator = tqdm(periodicities, desc="Evaluando periodicidades", miniters=1) if use_tqdm else periodicities
    for rule in iterator:
        try:
            metrics = evaluate_periodicity(
                df_close=df, rule=rule,
                mi_k=args.mi_nneighbors, max_mi_samples=args.max_mi_samples,
                lz_method=args.lz_method, max_lz_samples=args.max_lz_samples,
                use_tqdm=use_tqdm, quiet=args.quiet,
                lambda_penalty=args.lambda_penalty
            )
        except Exception as e:
            error(f"Falló evaluación en {rule}: {e}")
            metrics = {'periodicity': rule, 'n_points': np.nan,
                       'mi_hf_med_bits': np.nan, 'omega_hf': np.nan, 'acf_hf_sum_abs': np.nan, 'snr_hf': np.nan,
                       'mi_lf_med_bits': np.nan, 'omega_lf': np.nan, 'acf_lf_sum_abs': np.nan, 'snr_lf': np.nan,
                       'wpe': np.nan, 'lz_entropy_rate': np.nan,
                       'mi_hf_h3_bits': np.nan, 'mi_hf_h4_bits': np.nan, 'mi_hf_h5_bits': np.nan, 'mi_hf_h6_bits': np.nan,
                       'mi_lf_d3_bits': np.nan, 'mi_lf_d4_bits': np.nan, 'mi_lf_d5_bits': np.nan, 'mi_lf_d6_bits': np.nan}
        rows.append(metrics)

    info("[PIPE] Agregando métricas y construyendo scores…", args.quiet)
    dfm = pd.DataFrame(rows)
    dfr = composite_scores(dfm.copy(), lambda_penalty=args.lambda_penalty)

    # Pretty print HF
    info("\n=== RESUMEN POR PERIODICIDAD — HF (3–6h) ===", False)
    cols_hf = ['periodicity','n_points','mi_hf_med_bits','omega_hf','acf_hf_sum_abs','snr_hf','wpe','lz_entropy_rate','S_HF']
    dfr_hf = dfr.sort_values('S_HF', ascending=False).reset_index(drop=True)
    _fmt = lambda v: f"{v:,.6f}" if isinstance(v,(int,float,np.floating)) and not np.isnan(v) else str(v)
    print(dfr_hf[cols_hf].to_string(index=True, float_format=_fmt), flush=True)

    # Pretty print LF
    info("\n=== RESUMEN POR PERIODICIDAD — LF (3–6d) ===", False)
    cols_lf = ['periodicity','n_points','mi_lf_med_bits','omega_lf','acf_lf_sum_abs','snr_lf','wpe','lz_entropy_rate','S_LF']
    dfr_lf = dfr.sort_values('S_LF', ascending=False).reset_index(drop=True)
    print(dfr_lf[cols_lf].to_string(index=True, float_format=_fmt), flush=True)

    # Save CSV
    info(f"\n[PIPE] Guardando resultados en: {args.out}", args.quiet)
    try:
        dfr.to_csv(args.out, index=False); info("[OK] Archivo escrito correctamente.", False)
    except Exception as e:
        error(f"No se pudo guardar {args.out}: {e}")

    # Rankings
    info("\n=== RANKING HF (mejor→peor) ===", False)
    for i, row in dfr_hf.iterrows():
        print(f"{i+1:2d}. {row['periodicity']}  (S_HF = {row['S_HF']:.3f})", flush=True)

    info("\n=== RANKING LF (mejor→peor) ===", False)
    for i, row in dfr_lf.iterrows():
        print(f"{i+1:2d}. {row['periodicity']}  (S_LF = {row['S_LF']:.3f})", flush=True)

if __name__ == '__main__':
    main()
