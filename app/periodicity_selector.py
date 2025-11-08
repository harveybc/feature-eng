#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
periodicity_selector_verbose.py
===============================

Selector de periodicidad orientado a HORIZONTES DE NEGOCIO:
- HF (corto plazo): 3–6 horas
- LF (largo plazo): 3–6 días

Características clave:
- Métricas MODEL-FREE focalizadas a esos horizontes:
  * MI pasada→futuro para todos los horizontes en cada rango (mediana HF/LF)
  * Forecastability Ω ESPECTRAL LIMITADA A BANDA (3–12h para HF, 2–10d para LF)
  * ACF mass y SNR a escala de horizonte (6h para HF, 6d para LF)
  * WPE y LZ como penalizadores globales (tunable con --lambda-penalty)
- Composición de dos scores y dos rankings: S_HF y S_LF
- Progreso visible con tqdm y prints con flush
- Caps de muestras para MI/LZ que evitan bloqueos en datasets gigantes

Uso:
    python periodicity_selector_verbose.py path/to/data_5m.csv \
        [--time-col TIME_COL] [--close-col CLOSE_COL] \
        [--out periodicity_analysis.csv] \
        [--mi-nneighbors 5] [--max-mi-samples 800000] \
        [--max-lz-samples 100000] [--lz-method zlib] \
        [--lambda-penalty 0.3] [--quiet] [--no-tqdm] [--progress]

Dependencias:
    numpy, pandas, scipy, scikit-learn, tqdm
"""

# ------------------------------
# Importaciones estándar / typing
# ------------------------------
import argparse                      # Parseo de argumentos de línea de comandos
import sys                           # Salida/errores y flushing
import math                          # Utilidades matemáticas
from typing import Dict, Tuple, List # Tipado de retornos para claridad

# ------------------------------
# Paquetes científicos
# ------------------------------
import numpy as np                                   # Cálculo numérico vectorizado
import pandas as pd                                  # Estructuras de datos / resampleo
from scipy.signal import welch                       # PSD por Welch para forecastability
from scipy.stats import median_abs_deviation         # Z-score robusto (MAD)
from sklearn.feature_selection import mutual_info_regression  # MI kNN continua

# ------------------------------
# Progreso
# ------------------------------
from tqdm import tqdm                # Barras de progreso

# ---------------------------------------------------------------------------
# Utilidades de logging/prints con flush inmediato para que siempre se vean
# ---------------------------------------------------------------------------
def info(msg: str, quiet: bool = False) -> None:
    """Imprime mensajes informativos con flush inmediato, si no está en quiet."""
    if not quiet:
        print(msg, flush=True)

def warn(msg: str) -> None:
    """Imprime advertencias a stderr con flush inmediato."""
    print(f"[ADVERTENCIA] {msg}", file=sys.stderr, flush=True)

def error(msg: str) -> None:
    """Imprime errores a stderr con flush inmediato."""
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Núcleo de cómputo: utilidades robustas
# ---------------------------------------------------------------------------
def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Z-score robusto por mediana/MAD, resistente a atípicos."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = median_abs_deviation(x, nan_policy='omit', scale='normal')
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return (x - med) / mad

def ensure_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Asegura índice datetime ordenado y sin duplicados."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=False, errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)
    df = df.set_index(time_col)
    df = df[~df.index.duplicated(keep='last')]
    return df

def resample_close(df: pd.DataFrame, close_col: str, rule: str) -> pd.DataFrame:
    """Downsample sin look-ahead: último close del intervalo (label/right)."""
    out = df[[close_col]].resample(rule, label='right', closed='right').last().dropna()
    return out

def log_returns(close: pd.Series) -> pd.Series:
    """Retornos logarítmicos con corrección de casos no positivos."""
    c = close.astype(float).copy()
    min_pos = c[c > 0.0].min()
    if pd.isna(min_pos) or min_pos <= 0.0:
        eps = 1e-12
        c = c + (abs(c.min()) + eps)
    r = np.log(c).diff().dropna()
    return r

# ---------------------------------------------------------------------------
# Conversión de periodicidades a escalas útiles
# ---------------------------------------------------------------------------
def bars_per_day_for_rule(rule: str) -> float:
    """Barras por día (aprox) para cada regla estándar."""
    mapping = {'5T': 288.0, '15T': 96.0, '1H': 24.0, '4H': 6.0, '1D': 1.0}
    return mapping.get(rule, np.nan)

def hours_per_bar(rule: str) -> float:
    """Tamaño de barra en horas para cada regla."""
    mapping = {'5T': 5.0/60.0, '15T': 15.0/60.0, '1H': 1.0, '4H': 4.0, '1D': 24.0}
    return mapping.get(rule, np.nan)

def to_bars(hours: float, rule: str) -> int:
    """Convierte horas reales a número de barras según periodicidad."""
    hp = hours_per_bar(rule)
    if np.isnan(hp) or hp <= 0:
        return np.nan
    return int(max(1, round(hours / hp)))

# ---------------------------------------------------------------------------
# Thinning helpers (evitar OOM/tiempos excesivos)
# ---------------------------------------------------------------------------
def _thin_series_for_mi(x: np.ndarray, max_samples: int) -> np.ndarray:
    """Submuestreo temporal uniforme para MI si la serie es masiva."""
    if x.size <= max_samples:
        return x
    step = int(np.ceil(x.size / max_samples))
    return x[::step]

def _thin_binary_sequence(s: np.ndarray, max_samples: int) -> np.ndarray:
    """Submuestreo temporal uniforme para secuencias binarias (mantiene orden)."""
    n = s.size
    if n <= max_samples:
        return s
    step = int(np.ceil(n / max_samples))
    return s[::step]

# ---------------------------------------------------------------------------
# Métricas elementales orientadas a horizonte
# ---------------------------------------------------------------------------
def mutual_information_horizon(r: pd.Series, horizon_bars: int, k: int = 5,
                               max_samples: int = 800_000,
                               use_tqdm: bool = True) -> float:
    """MI(X_t ; sum_{i=1..h} r_{t+i}) en bits, con thinning temporal si es enorme."""
    x_full = np.asarray(r.dropna(), dtype=float)
    if horizon_bars < 1 or x_full.size <= (horizon_bars + 10):
        return np.nan
    x = _thin_series_for_mi(x_full, max_samples=max_samples)
    n_eff = x.size
    if n_eff <= (horizon_bars + 10):
        return np.nan
    rng = range(n_eff - horizon_bars)
    if use_tqdm:
        rng = tqdm(rng, desc=f"MI: h={horizon_bars} barras", leave=False, miniters=1)
    X = np.empty((n_eff - horizon_bars, 1), dtype=float)
    Y = np.empty((n_eff - horizon_bars,), dtype=float)
    for i in rng:
        X[i, 0] = x[i]
        Y[i] = np.sum(x[i+1:i+1+horizon_bars])
    try:
        mi_nats = mutual_info_regression(X, Y, n_neighbors=k, random_state=42)
        mi_bits = float(mi_nats[0] / math.log(2.0))
    except Exception:
        mi_bits = np.nan
    return mi_bits

def acf_sum_abs(r: pd.Series, max_lag: int) -> float:
    """Suma de |ACF| en lag=1..max_lag (FFT)."""
    x = np.asarray(r.dropna(), dtype=float)
    n = x.size
    if n < max_lag + 5 or max_lag < 1:
        return np.nan
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 0:
        return np.nan
    fft = np.fft.rfft(x, n=2**int(np.ceil(np.log2(2*n - 1))))
    acf_full = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acf_full = acf_full / (var * np.arange(n, 0, -1))
    s = float(np.sum(np.abs(acf_full[1:max_lag+1])))
    return s

def snr_smoothing_returns(r: pd.Series, smooth_window: int) -> float:
    """SNR heurístico en retornos: var(señal_suavizada)/var(residuo)."""
    x = pd.Series(r.dropna().values, index=r.dropna().index)
    if x.size < smooth_window * 4:
        return np.nan
    m = x.rolling(window=smooth_window, center=True,
                  min_periods=max(3, smooth_window//3)).mean()
    resid = x - m
    var_sig = np.nanvar(m.values)
    var_noise = np.nanvar(resid.values)
    if var_noise <= 0 or np.isnan(var_noise):
        return np.nan
    return float(var_sig / var_noise)

# ---------------------------------------------------------------------------
# Complejidad ordinal y simbólica (penalizadores globales)
# ---------------------------------------------------------------------------
def permutation_entropy_weighted(r: pd.Series, m: int = 5, tau: int = 1,
                                 use_tqdm: bool = True) -> float:
    """WPE con manejo de ties (ponderación por varianza local)."""
    x = np.asarray(r.dropna(), dtype=float)
    N = x.size
    L = N - (m - 1) * tau
    if L <= 0:
        return np.nan
    M = np.empty((L, m), dtype=float)
    for i in range(m):
        M[:, i] = x[i * tau : i * tau + L]
    order = np.argsort(M, axis=1, kind='mergesort')
    it = range(order.shape[0])
    if use_tqdm:
        it = tqdm(it, desc="WPE: codificando patrones", leave=False, miniters=1)
    from collections import defaultdict
    weight_map = defaultdict(float)
    for idx in it:
        pat = tuple(order[idx])
        w = float(np.var(M[idx, :]))
        weight_map[pat] += max(w, 0.0)
    weights = np.array(list(weight_map.values()), dtype=float)
    if weights.size == 0:
        return np.nan
    weights = weights / np.sum(weights)
    weights = np.maximum(weights, 1e-300)
    Hw = -np.sum(weights * np.log(weights))
    K = math.factorial(m)
    wpe = float(np.clip(Hw / np.log(K), 0.0, 1.0))
    return wpe

def lz_entropy_rate_sign(r: pd.Series,
                         method: str = 'lz76',
                         max_samples: int = 100_000) -> float:
    """Tasa LZ sobre signo de retornos (con thinning) o proxy zlib ultrarrápido."""
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 256:
        return np.nan
    s = (x > 0).astype(np.uint8)
    if method == 'zlib':
        import zlib
        b = bytes(s.tolist())
        comp = zlib.compress(b, level=6)
        n = len(b)
        if n == 0:
            return np.nan
        return float((len(comp) * 8.0) / n)
    # lz76 con thinning
    s_thin = _thin_binary_sequence(s, max_samples=max_samples)
    n = int(s_thin.size)
    if n < 256:
        return np.nan
    seq = s_thin.tolist()
    i, k, l, c = 0, 1, 1, 1
    while True:
        if i + k > n:
            c += 1
            break
        if seq[i:i+k] == seq[l:l+k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            l += 1
            if l == i + k:
                c += 1
                i += k
                if i + 1 > n:
                    break
                l = 0
                k = 1
    rate = (c * math.log(n)) / n
    return float(rate)

# ---------------------------------------------------------------------------
# Forecastability Ω LIMITADA A BANDA DE INTERÉS
# ---------------------------------------------------------------------------
def spectral_forecastability_bandlimited(r: pd.Series,
                                         fs_per_bar: float,
                                         band_hours: Tuple[float, float],
                                         rule: str) -> float:
    """
    Forecastability Ω pero calculada SOLO dentro de una banda de periodos en horas.
    Pasos:
      1) Welch PSD sobre retornos.
      2) Convertir frecuencias a periodos (en horas) usando tamaño de barra.
      3) Seleccionar bins cuyo periodo T esté dentro de la banda [Tmin, Tmax].
      4) Normalizar PSD en esa banda y medir entropía espectral discreta.
      5) Ω_band = 1 - H_band / log(K_band).
    """
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 256:
        return np.nan
    # tamaño de barra en horas
    hpb = hours_per_bar(rule)
    if np.isnan(hpb) or hpb <= 0:
        return np.nan
    # frecuencia de muestreo en "barras por hora"
    fs = 1.0 / hpb
    # Welch
    nperseg = int(max(256, min(8192, 8 * fs)))  # usar algo más generoso en ventanas
    if nperseg > x.size:
        nperseg = max(128, x.size // 2)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, detrend='constant',
                   return_onesided=True, scaling='density')
    Pxx = np.maximum(np.asarray(Pxx, float), 1e-300)
    # periodos en horas por bin: T = 1/f (evitar f=0)
    with np.errstate(divide='ignore'):
        T = 1.0 / np.maximum(f, 1e-12)
    # máscara de banda en horas (ej: HF: [3,12], LF: [48,240] horas ~ 2–10 días)
    Tmin, Tmax = band_hours
    mask = (T >= Tmin) & (T <= Tmax)
    if not np.any(mask):
        return np.nan
    p = Pxx[mask]
    p = p / np.sum(p)
    p = np.maximum(p, 1e-300)
    Hs = -np.sum(p * np.log(p))
    K = p.size
    denom = np.log(K) if K > 1 else 1.0
    omega_band = float(np.clip(1.0 - Hs / denom, 0.0, 1.0))
    return omega_band

# ---------------------------------------------------------------------------
# Evaluación por periodicidad (orientada a HF/LF) con informes
# ---------------------------------------------------------------------------
def evaluate_periodicity(df_close: pd.DataFrame, rule: str,
                         mi_k: int, max_mi_samples: int,
                         lz_method: str, max_lz_samples: int,
                         use_tqdm: bool, quiet: bool,
                         lambda_penalty: float) -> Dict[str, float]:
    """Calcula todas las métricas para una periodicidad y reporta progreso orientado a HF/LF."""
    info(f"[{rule}] Preparando serie y retornos…", quiet)
    if rule != '5T':
        dfp = resample_close(df_close, 'close', rule)
    else:
        dfp = df_close.copy()
    r = log_returns(dfp['close'])

    # --- Conjuntos de horizontes en horas reales:
    HF_hours = [3.0, 4.0, 5.0, 6.0]         # 3–6 horas
    LF_hours = [72.0, 96.0, 120.0, 144.0]   # 3–6 días

    # --- MI por todos los horizontes en cada rango (agregación por mediana):
    info(f"[{rule}] MI HF (3–6h) para todos los horizontes…", quiet)
    mi_hf_values = []
    it_hf = HF_hours
    if use_tqdm:
        it_hf = tqdm(HF_hours, desc=f"MI HF ({rule})", leave=False, miniters=1)
    for H in it_hf:
        hbars = to_bars(H, rule)
        mi_bits = mutual_information_horizon(r, horizon_bars=hbars, k=mi_k,
                                             max_samples=max_mi_samples, use_tqdm=False)
        mi_hf_values.append(mi_bits)
    mi_hf_med = float(np.nanmedian(mi_hf_values)) if np.any(~np.isnan(mi_hf_values)) else np.nan

    info(f"[{rule}] MI LF (3–6d) para todos los horizontes…", quiet)
    mi_lf_values = []
    it_lf = LF_hours
    if use_tqdm:
        it_lf = tqdm(LF_hours, desc=f"MI LF ({rule})", leave=False, miniters=1)
    for H in it_lf:
        hbars = to_bars(H, rule)
        mi_bits = mutual_information_horizon(r, horizon_bars=hbars, k=mi_k,
                                             max_samples=max_mi_samples, use_tqdm=False)
        mi_lf_values.append(mi_bits)
    mi_lf_med = float(np.nanmedian(mi_lf_values)) if np.any(~np.isnan(mi_lf_values)) else np.nan

    # --- Ω band-limited para bandas de interés:
    # HF band: 3–12 horas (captura estructura alrededor del corto)
    # LF band: 48–240 horas (~2–10 días)
    info(f"[{rule}] Ω_HF en banda 3–12h…", quiet)
    omega_hf = spectral_forecastability_bandlimited(r, fs_per_bar=1.0, band_hours=(3.0, 12.0), rule=rule)
    info(f"[{rule}] Ω_LF en banda 2–10 días…", quiet)
    omega_lf = spectral_forecastability_bandlimited(r, fs_per_bar=1.0, band_hours=(48.0, 240.0), rule=rule)

    # --- ACF mass y SNR a escala:
    # ACF_HF: hasta 6h en barras; ACF_LF: hasta 6d en barras
    h_s_max = to_bars(6.0, rule)
    h_l_max = to_bars(144.0, rule)
    info(f"[{rule}] ACF_sum_abs HF (hasta 6h)…", quiet)
    acf_hf = acf_sum_abs(r, max_lag=int(h_s_max)) if not np.isnan(h_s_max) else np.nan
    info(f"[{rule}] ACF_sum_abs LF (hasta 6d)…", quiet)
    acf_lf = acf_sum_abs(r, max_lag=int(h_l_max)) if not np.isnan(h_l_max) else np.nan

    info(f"[{rule}] SNR HF (ventana=6h)…", quiet)
    snr_hf = snr_smoothing_returns(r, smooth_window=int(h_s_max)) if not np.isnan(h_s_max) else np.nan
    info(f"[{rule}] SNR LF (ventana=6d)…", quiet)
    snr_lf = snr_smoothing_returns(r, smooth_window=int(h_l_max)) if not np.isnan(h_l_max) else np.nan

    # --- Penalizadores globales (complejidad ordinal/simbólica):
    info(f"[{rule}] WPE global…", quiet)
    wpe = permutation_entropy_weighted(r, m=5, tau=1, use_tqdm=use_tqdm)
    info(f"[{rule}] LZ global (método={lz_method})…", quiet)
    lz_rate = lz_entropy_rate_sign(r, method=lz_method, max_samples=max_lz_samples)

    # --- Devolver todas las métricas (HF/LF + penalizadores + tamaños)
    return {
        'periodicity': rule,
        'n_points': int(dfp.shape[0]),
        # HF block
        'mi_hf_med_bits': mi_hf_med,
        'omega_hf': omega_hf,
        'acf_hf_sum_abs': acf_hf,
        'snr_hf': snr_hf,
        # LF block
        'mi_lf_med_bits': mi_lf_med,
        'omega_lf': omega_lf,
        'acf_lf_sum_abs': acf_lf,
        'snr_lf': snr_lf,
        # penalties
        'wpe': wpe,
        'lz_entropy_rate': lz_rate,
        # raw MI lists (optional debugging)
        'mi_hf_h3_bits': mi_hf_values[0] if len(mi_hf_values)>0 else np.nan,
        'mi_hf_h4_bits': mi_hf_values[1] if len(mi_hf_values)>1 else np.nan,
        'mi_hf_h5_bits': mi_hf_values[2] if len(mi_hf_values)>2 else np.nan,
        'mi_hf_h6_bits': mi_hf_values[3] if len(mi_hf_values)>3 else np.nan,
        'mi_lf_d3_bits': mi_lf_values[0] if len(mi_lf_values)>0 else np.nan,
        'mi_lf_d4_bits': mi_lf_values[1] if len(mi_lf_values)>1 else np.nan,
        'mi_lf_d5_bits': mi_lf_values[2] if len(mi_lf_values)>2 else np.nan,
        'mi_lf_d6_bits': mi_lf_values[3] if len(mi_lf_values)>3 else np.nan,
    }

# ---------------------------------------------------------------------------
# Composición de scores y rankings (HF y LF)
# ---------------------------------------------------------------------------
def composite_scores(df_metrics: pd.DataFrame, lambda_penalty: float) -> pd.DataFrame:
    """
    Construye z-scores robustos y dos composites:
      S_HF = +z(mi_hf_med_bits) + z(omega_hf) + z(acf_hf_sum_abs) + z(snr_hf)
             - lambda * [z(wpe) + z(lz_entropy_rate)]
      S_LF = +z(mi_lf_med_bits) + z(omega_lf) + z(acf_lf_sum_abs) + z(snr_lf)
             - lambda * [z(wpe) + z(lz_entropy_rate)]
    """
    # columnas por bloque
    hf_plus  = ['mi_hf_med_bits', 'omega_hf', 'acf_hf_sum_abs', 'snr_hf']
    lf_plus  = ['mi_lf_med_bits', 'omega_lf', 'acf_lf_sum_abs', 'snr_lf']
    penalties = ['wpe', 'lz_entropy_rate']

    # z-scores robustos
    for c in hf_plus + lf_plus + penalties:
        df_metrics[f'z_{c}'] = robust_zscore(df_metrics[c].values)

    # composites
    df_metrics['S_HF'] = 0.0
    for c in hf_plus:
        df_metrics['S_HF'] += df_metrics[f'z_{c}']
    df_metrics['S_HF'] += -lambda_penalty * (df_metrics['z_wpe'] + df_metrics['z_lz_entropy_rate'])

    df_metrics['S_LF'] = 0.0
    for c in lf_plus:
        df_metrics['S_LF'] += df_metrics[f'z_{c}']
    df_metrics['S_LF'] += -lambda_penalty * (df_metrics['z_wpe'] + df_metrics['z_lz_entropy_rate'])

    # rankings por score
    df_metrics = df_metrics.copy()
    df_metrics = df_metrics.sort_values('S_HF', ascending=False).reset_index(drop=True)
    df_metrics['rank_HF'] = np.arange(1, df_metrics.shape[0] + 1)

    df_metrics = df_metrics.sort_values('S_LF', ascending=False).reset_index(drop=True)
    df_metrics['rank_LF'] = np.arange(1, df_metrics.shape[0] + 1)

    # no devolver ordenados por uno de ellos; mantener columnas y ordenar luego al imprimir
    return df_metrics

# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------
def main():
    """Punto de entrada CLI con prints y barras de progreso."""
    ap = argparse.ArgumentParser(
        description="Selector de periodicidad (verbose) orientado a horizontes HF (3–6h) y LF (3–6d)."
    )
    ap.add_argument('csv', type=str, help='Ruta al CSV base de 5 minutos.')
    ap.add_argument('--time-col', type=str, default=None, help='Columna temporal (auto si no se da).')
    ap.add_argument('--close-col', type=str, default='close', help="Columna de cierre (default: 'close').")
    ap.add_argument('--out', type=str, default='periodicity_analysis.csv', help="CSV de salida.")
    ap.add_argument('--mi-nneighbors', type=int, default=5, help="Vecinos k para MI (default=5).")
    ap.add_argument('--max-mi-samples', type=int, default=800_000,
                    help="Máximo de muestras para MI tras thinning (default=800k).")
    ap.add_argument('--max-lz-samples', type=int, default=100_000,
                    help="Máximo de símbolos binarios para LZ tras thinning (default=100k).")
    ap.add_argument('--lz-method', type=str, default='zlib', choices=['lz76', 'zlib'],
                    help="Método para tasa LZ: 'lz76' (thinning) o 'zlib' (proxy muy rápido).")
    ap.add_argument('--lambda-penalty', type=float, default=0.3,
                    help="Peso de penalización para WPE y LZ en los composites (default=0.3).")
    ap.add_argument('--quiet', action='store_true', help="Menos mensajes (mantiene barras).")
    ap.add_argument('--no-tqdm', action='store_true', help="Desactivar barras tqdm.")
    ap.add_argument('--progress', action='store_true',
                    help="Forzar tqdm incluso en datasets pequeños (por defecto se decide automáticamente).")
    args = ap.parse_args()

    # Lectura del CSV con informe
    info("[INIT] Cargando CSV…", args.quiet)
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        error(f"No se pudo leer el CSV: {e}")
        sys.exit(1)
    info(f"[INIT] CSV cargado: shape={df.shape}", args.quiet)

    # Detección de columnas
    time_col = args.time_col
    if time_col is None:
        candidates = [c for c in df.columns if c.lower() in ('time', 'timestamp', 'datetime', 'date')]
        if len(candidates) == 0:
            error("No se encontró columna temporal. Use --time-col.")
            sys.exit(1)
        time_col = candidates[0]
        info(f"[INIT] Columna temporal detectada: {time_col}", args.quiet)
    close_col = args.close_col
    if close_col not in df.columns:
        error(f"No existe la columna '{close_col}'. Use --close-col.")
        sys.exit(1)

    # Índice temporal
    info("[INIT] Normalizando índice temporal…", args.quiet)
    df = ensure_datetime_index(df[[time_col, close_col]].rename(columns={close_col: 'close'}), time_col=time_col)
    info(f"[INIT] Serie temporal lista: n={df.shape[0]} puntos (5T)", args.quiet)
    if df.shape[0] < 1000:
        warn("Pocos puntos de 5m; las métricas pueden ser inestables.")

    # Selección de si usar tqdm
    auto_use_tqdm = (df.shape[0] > 100_000) or args.progress
    use_tqdm = False if args.no_tqdm else auto_use_tqdm

    # Periodicidades a evaluar
    periodicities = ['5T', '15T', '1H', '4H', '1D']
    info(f"[PIPE] Periodicidades objetivo: {periodicities}", args.quiet)

    # Evaluación con barra de progreso en el bucle principal
    rows: List[Dict[str, float]] = []
    iterator = periodicities
    if use_tqdm:
        iterator = tqdm(periodicities, desc="Evaluando periodicidades", miniters=1)
    for rule in iterator:
        try:
            metrics = evaluate_periodicity(
                df_close=df,
                rule=rule,
                mi_k=args.mi_nneighbors,
                max_mi_samples=args.max_mi_samples,
                lz_method=args.lz_method,
                max_lz_samples=args.max_lz_samples,
                use_tqdm=use_tqdm,
                quiet=args.quiet,
                lambda_penalty=args.lambda_penalty
            )
        except Exception as e:
            error(f"Falló evaluación en {rule}: {e}")
            metrics = {
                'periodicity': rule, 'n_points': np.nan,
                'mi_hf_med_bits': np.nan, 'omega_hf': np.nan, 'acf_hf_sum_abs': np.nan, 'snr_hf': np.nan,
                'mi_lf_med_bits': np.nan, 'omega_lf': np.nan, 'acf_lf_sum_abs': np.nan, 'snr_lf': np.nan,
                'wpe': np.nan, 'lz_entropy_rate': np.nan,
                'mi_hf_h3_bits': np.nan, 'mi_hf_h4_bits': np.nan, 'mi_hf_h5_bits': np.nan, 'mi_hf_h6_bits': np.nan,
                'mi_lf_d3_bits': np.nan, 'mi_lf_d4_bits': np.nan, 'mi_lf_d5_bits': np.nan, 'mi_lf_d6_bits': np.nan
            }
        rows.append(metrics)

    # DataFrame y scores
    info("[PIPE] Agregando métricas y construyendo scores (HF/LF)…", args.quiet)
    dfm = pd.DataFrame(rows)
    dfr = composite_scores(dfm.copy(), lambda_penalty=args.lambda_penalty)

    # Impresión de resumen HF
    info("\n=== RESUMEN POR PERIODICIDAD — HF (3–6h) ===", False)
    display_cols_hf = [
        'periodicity', 'n_points',
        'mi_hf_med_bits', 'omega_hf', 'acf_hf_sum_abs', 'snr_hf',
        'wpe', 'lz_entropy_rate',
        'S_HF'
    ]
    # Ordenar por S_HF descendente solo para imprimir esta vista
    dfr_hf = dfr.sort_values('S_HF', ascending=False).reset_index(drop=True)
    def _fmt(v):
        try: return f"{v:,.6f}"
        except Exception: return str(v)
    print(dfr_hf[display_cols_hf].to_string(index=True, float_format=_fmt), flush=True)

    # Impresión de resumen LF
    info("\n=== RESUMEN POR PERIODICIDAD — LF (3–6d) ===", False)
    display_cols_lf = [
        'periodicity', 'n_points',
        'mi_lf_med_bits', 'omega_lf', 'acf_lf_sum_abs', 'snr_lf',
        'wpe', 'lz_entropy_rate',
        'S_LF'
    ]
    dfr_lf = dfr.sort_values('S_LF', ascending=False).reset_index(drop=True)
    print(dfr_lf[display_cols_lf].to_string(index=True, float_format=_fmt), flush=True)

    # Guardado CSV completo (todas las columnas)
    info(f"\n[PIPE] Guardando resultados en: {args.out}", args.quiet)
    try:
        dfr.to_csv(args.out, index=False)
        info("[OK] Archivo escrito correctamente.", False)
    except Exception as e:
        error(f"No se pudo guardar {args.out}: {e}")

    # Rankings finales
    info("\n=== RANKING HF (mejor→peor) ===", False)
    for i, row in dfr_hf.iterrows():
        print(f"{i+1:2d}. {row['periodicity']}  (S_HF = {row['S_HF']:.3f})", flush=True)

    info("\n=== RANKING LF (mejor→peor) ===", False)
    for i, row in dfr_lf.iterrows():
        print(f"{i+1:2d}. {row['periodicity']}  (S_LF = {row['S_LF']:.3f})", flush=True)

if __name__ == '__main__':
    main()
