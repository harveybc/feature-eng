#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
periodicity_selector_verbose.py
===============================

Script CLI para seleccionar periodicidad óptima (modelo-free) con:
- Mensajes detallados en cada fase (prints con flush).
- Barras de progreso con tqdm en operaciones costosas.
- Limites seguros para evitar bloqueos en datasets muy grandes (1.3M+ filas).

Uso:
    python periodicity_selector_verbose.py path/to/data_5m.csv \
        [--time-col TIME_COL] [--close-col CLOSE_COL] \
        [--out periodicity_analysis.csv] \
        [--mi-nneighbors 5] [--max-mi-samples 800000] [--quiet] [--no-tqdm]

Dependencias:
    numpy, pandas, scipy, scikit-learn, tqdm
"""

# ------------------------------
# Importaciones estándar y typing
# ------------------------------
import argparse                      # Parseo de argumentos de línea de comandos
import sys                           # Salida/errores y flushing
import math                          # Utilidades matemáticas
from typing import Dict, Tuple       # Tipado de retornos para claridad

# ------------------------------
# Paquetes científicos
# ------------------------------
import numpy as np                   # Cálculo numérico vectorizado
import pandas as pd                  # Estructuras de datos / resampleo
from scipy.signal import welch       # PSD por Welch para forecastability
from scipy.stats import median_abs_deviation  # Z-score robusto (MAD)
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
    """Imprime errores a stderr con flush inmediato y sugiere salida controlada."""
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Núcleo de cómputo: utilidades numéricas y estadísticas
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

def bars_per_day_for_rule(rule: str) -> float:
    """Barras por día (aprox) para cada regla estándar."""
    mapping = {'5T': 288.0, '15T': 96.0, '1H': 24.0, '4H': 6.0, '1D': 1.0}
    return mapping.get(rule, np.nan)

def horizons_in_bars(rule: str) -> Tuple[int, int, int]:
    """Convierte horizontes 6h y 144h a número de barras para cada periodicidad."""
    if rule == '5T':   return (72, 1728, 72)
    if rule == '15T':  return (24, 576, 24)
    if rule == '1H':   return (6, 144, 6)
    if rule == '4H':   return (max(1, 6//4), 144//4, max(1, 6//4))
    if rule == '1D':   return (1, 6, 1)
    return (np.nan, np.nan, np.nan)

# ---------------------------------------------------------------------------
# Métricas (con progresos internos cuando aplica)
# ---------------------------------------------------------------------------
def spectral_forecastability_omega(r: pd.Series, fs_per_day: float) -> float:
    """Forecastability Ω via entropía espectral de Welch (p en frecuencias)."""
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 128:
        return np.nan
    nperseg = int(max(128, min(4096, 4 * fs_per_day)))
    if nperseg > x.size:
        nperseg = max(64, x.size // 2)
    f, Pxx = welch(x, nperseg=nperseg, detrend='constant',
                   return_onesided=True, scaling='density')
    Pxx = np.maximum(np.asarray(Pxx, float), 1e-300)
    p = Pxx / np.sum(Pxx)
    Hs = -np.sum(p * np.log(p))
    K = p.size
    denom = np.log(K) if K > 1 else 1.0
    omega = float(np.clip(1.0 - Hs / denom, 0.0, 1.0))
    return omega

def permutation_entropy_weighted(r: pd.Series, m: int = 5, tau: int = 1,
                                 use_tqdm: bool = True, quiet: bool = False) -> float:
    """WPE con manejo de ties y barra de progreso opcional."""
    x = np.asarray(r.dropna(), dtype=float)
    N = x.size
    L = N - (m - 1) * tau
    if L <= 0:
        return np.nan
    # Inicializa matriz embebida (vectorizado por columna).
    M = np.empty((L, m), dtype=float)
    for i in range(m):
        M[:, i] = x[i * tau : i * tau + L]
    # argsort estable por filas (sin bucles Python).
    order = np.argsort(M, axis=1, kind='mergesort')
    # Codifica patrones como tuplas; esto requiere iteración, aquí mostramos progreso.
    it = range(order.shape[0])
    if use_tqdm:
        it = tqdm(it, desc="WPE: codificando patrones", leave=False, miniters=1)
    from collections import defaultdict
    weight_map = defaultdict(float)
    for idx in it:
        pat = tuple(order[idx])
        # Peso por varianza local (pondera amplitud)
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

def lz_entropy_rate_sign(r: pd.Series) -> float:
    """Tasa LZ sobre signo de retornos (rápido y sin progreso necesario)."""
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 256:
        return np.nan
    s = (x > 0).astype(np.uint8).tolist()
    n = len(s)
    i, k, l, c = 0, 1, 1, 1
    while True:
        if i + k > n:
            c += 1
            break
        if s[i:i+k] == s[l:l+k]:
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

def _thin_series_for_mi(x: np.ndarray, max_samples: int) -> np.ndarray:
    """Submuestreo temporal uniforme para MI si la serie es masiva."""
    if x.size <= max_samples:
        return x
    step = int(np.ceil(x.size / max_samples))
    return x[::step]

def mutual_information_horizon(r: pd.Series, horizon_bars: int, k: int = 5,
                               max_samples: int = 800_000,
                               use_tqdm: bool = True) -> float:
    """
    MI(X_t ; sum_{i=1..h} r_{t+i}) en bits, con thinning temporal si es enorme.
    Progreso: fase de construcción de Y puede mostrar barra.
    """
    x_full = np.asarray(r.dropna(), dtype=float)
    if horizon_bars < 1 or x_full.size <= (horizon_bars + 10):
        return np.nan
    x = _thin_series_for_mi(x_full, max_samples=max_samples)
    n_eff = x.size
    if n_eff <= (horizon_bars + 10):
        return np.nan
    # Construye X e Y con progreso
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

def acf_sum_abs(r: pd.Series, max_lag: int, use_tqdm: bool = False) -> float:
    """Suma de |ACF| en lag=1..max_lag; opcionalmente muestra progreso chunked."""
    x = np.asarray(r.dropna(), dtype=float)
    n = x.size
    if n < max_lag + 5 or max_lag < 1:
        return np.nan
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 0:
        return np.nan
    # FFT grande (rápido), no vale la pena barra fina; mostramos solo un stub
    if use_tqdm:
        _ = tqdm(total=1, desc="ACF por FFT", leave=False)
    fft = np.fft.rfft(x, n=2**int(np.ceil(np.log2(2*n - 1))))
    acf_full = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acf_full = acf_full / (var * np.arange(n, 0, -1))
    if use_tqdm:
        _.update(1); _.close()
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
# Evaluación por periodicidad con informes y barras de progreso
# ---------------------------------------------------------------------------
def evaluate_periodicity(df_close: pd.DataFrame, rule: str,
                         mi_k: int, max_mi_samples: int,
                         use_tqdm: bool, quiet: bool) -> Dict[str, float]:
    """Calcula todas las métricas para una periodicidad y reporta progreso."""
    info(f"[{rule}] Preparando serie y retornos…", quiet)
    if rule != '5T':
        dfp = resample_close(df_close, 'close', rule)
    else:
        dfp = df_close.copy()
    r = log_returns(dfp['close'])
    fsd = bars_per_day_for_rule(rule)
    h_s, h_l, acf_lag = horizons_in_bars(rule)
    smooth_win = int(max(3, h_s))

    info(f"[{rule}] Cálculo Ω (forecastability espectral)…", quiet)
    omega = spectral_forecastability_omega(r, fsd)

    info(f"[{rule}] Cálculo WPE (puede tardar, depende de N)…", quiet)
    wpe = permutation_entropy_weighted(r, m=5, tau=1, use_tqdm=use_tqdm, quiet=quiet)

    info(f"[{rule}] Cálculo tasa LZ (signo retornos)…", quiet)
    lz_rate = lz_entropy_rate_sign(r)

    info(f"[{rule}] Cálculo MI 6h (h={int(h_s)} barras)…", quiet)
    mi_6h = mutual_information_horizon(r, horizon_bars=int(h_s), k=mi_k,
                                       max_samples=max_mi_samples, use_tqdm=use_tqdm) \
            if not np.isnan(h_s) else np.nan

    info(f"[{rule}] Cálculo MI 144h (h={int(h_l)} barras)…", quiet)
    mi_144h = mutual_information_horizon(r, horizon_bars=int(h_l), k=mi_k,
                                         max_samples=max_mi_samples, use_tqdm=use_tqdm) \
              if not np.isnan(h_l) else np.nan

    info(f"[{rule}] Cálculo ACF_sum_abs (hasta lag={int(acf_lag)})…", quiet)
    acf_mass = acf_sum_abs(r, max_lag=int(acf_lag), use_tqdm=use_tqdm) \
               if not np.isnan(acf_lag) else np.nan

    info(f"[{rule}] Cálculo SNR (ventana suavizado={smooth_win})…", quiet)
    snr = snr_smoothing_returns(r, smooth_window=int(smooth_win)) \
          if not np.isnan(smooth_win) else np.nan

    info(f"[{rule}] Hecho. (n_points={dfp.shape[0]})", quiet)
    return {
        'periodicity': rule,
        'n_points': int(dfp.shape[0]),
        'omega_forecastability': omega,
        'wpe': wpe,
        'lz_entropy_rate': lz_rate,
        'mi_6h_bits': mi_6h,
        'mi_144h_bits': mi_144h,
        'acf_sum_abs': acf_mass,
        'snr_smooth': snr,
    }

def composite_score(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Construye z-scores robustos y S_pred, con ranking."""
    cols_plus  = ['omega_forecastability', 'mi_6h_bits', 'mi_144h_bits', 'acf_sum_abs', 'snr_smooth']
    cols_minus = ['wpe', 'lz_entropy_rate']
    for c in cols_plus + cols_minus:
        df_metrics[f'z_{c}'] = robust_zscore(df_metrics[c].values)
    df_metrics['S_pred'] = 0.0
    for c in cols_plus:
        df_metrics['S_pred'] += df_metrics[f'z_{c}']
    for c in cols_minus:
        df_metrics['S_pred'] += -df_metrics[f'z_{c}']
    df_metrics = df_metrics.sort_values('S_pred', ascending=False).reset_index(drop=True)
    df_metrics['rank'] = np.arange(1, df_metrics.shape[0] + 1)
    return df_metrics

# ---------------------------------------------------------------------------
# CLI principal con informes detallados
# ---------------------------------------------------------------------------
def main():
    """Punto de entrada CLI con prints y barras de progreso."""
    ap = argparse.ArgumentParser(
        description="Selector de periodicidad (verbose) con métricas modelo-free sobre 'close'."
    )
    ap.add_argument('csv', type=str, help='Ruta al CSV base de 5 minutos.')
    ap.add_argument('--time-col', type=str, default=None, help='Columna temporal (auto si no se da).')
    ap.add_argument('--close-col', type=str, default='close', help="Columna de cierre (default: 'close').")
    ap.add_argument('--out', type=str, default='periodicity_analysis.csv', help="CSV de salida.")
    ap.add_argument('--mi-nneighbors', type=int, default=5, help="Vecinos k para MI (default=5).")
    ap.add_argument('--max-mi-samples', type=int, default=800_000,
                    help="Máximo de muestras para MI tras thinning (default=800k).")
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
    auto_use_tqdm = (df.shape[0] > 200_000) or args.progress
    use_tqdm = False if args.no_tqdm else auto_use_tqdm

    # Periodicidades a evaluar
    periodicities = ['5T', '15T', '1H', '4H', '1D']
    info(f"[PIPE] Periodicidades objetivo: {periodicities}", args.quiet)

    # Evaluación con barra de progreso en el bucle principal
    rows = []
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
                use_tqdm=use_tqdm,
                quiet=args.quiet
            )
        except Exception as e:
            error(f"Falló evaluación en {rule}: {e}")
            metrics = {
                'periodicity': rule, 'n_points': np.nan,
                'omega_forecastability': np.nan, 'wpe': np.nan,
                'lz_entropy_rate': np.nan, 'mi_6h_bits': np.nan,
                'mi_144h_bits': np.nan, 'acf_sum_abs': np.nan, 'snr_smooth': np.nan
            }
        rows.append(metrics)

    # DataFrame y score
    info("[PIPE] Agregando métricas y construyendo score compuesto…", args.quiet)
    dfm = pd.DataFrame(rows)
    dfr = composite_score(dfm.copy())

    # Impresión de resumen
    display_cols = [
        'rank', 'periodicity', 'n_points',
        'omega_forecastability', 'wpe', 'lz_entropy_rate',
        'mi_6h_bits', 'mi_144h_bits', 'acf_sum_abs', 'snr_smooth', 'S_pred'
    ]
    info("\n=== RESUMEN POR PERIODICIDAD ===", False)
    # Formateo robusto
    def _fmt(v):
        try:
            return f"{v:,.6f}"
        except Exception:
            return str(v)
    print(dfr[display_cols].to_string(index=False, float_format=_fmt), flush=True)

    # Guardado
    info(f"\n[PIPE] Guardando resultados en: {args.out}", args.quiet)
    try:
        dfr.to_csv(args.out, index=False)
        info("[OK] Archivo escrito correctamente.", False)
    except Exception as e:
        error(f"No se pudo guardar {args.out}: {e}")

    # Ranking final
    info("\n=== RANKING (mejor→peor) ===", False)
    for _, row in dfr.iterrows():
        print(f"{int(row['rank']):2d}. {row['periodicity']}  (S_pred = {row['S_pred']:.3f})", flush=True)

if __name__ == '__main__':
    main()
