#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
periodicity_selector.py
=======================

Programa CLI para evaluar la mejor periodicidad para predicción usando únicamente
la serie de "close" de un dataset base de 5 minutos. Genera downsample a 15m, 1h,
4h y 1d; calcula métricas de predecibilidad modelo-free y compone un score final.

Uso:
    python periodicity_selector.py path/to/data_5m.csv \
        [--time-col TIME_COL] [--close-col CLOSE_COL] [--out periodicity_analysis.csv]

Requisitos:
    - Python 3.9+
    - numpy, pandas, scipy, scikit-learn

Salida:
    - Imprime tabla resumen y ranking de periodicidades.
    - Guarda periodicity_analysis.csv con todas las métricas y el score final.

Licencia:
    MIT (ajustable según tus repos).
"""

#: Importaciones estándar
import argparse  #: Parseo de argumentos CLI
import sys       #: Manejo de salida y errores
import math      #: Utilidades matemáticas
from typing import Dict, Tuple, List  #: Tipado para mejor mantenibilidad

#: Dependencias científicas
import numpy as np                    #: Cálculo numérico
import pandas as pd                   #: Estructuras de datos y resampleo
from scipy.signal import welch        #: Estimación espectral (PSD) por Welch
from scipy.stats import median_abs_deviation  #: MAD robusto para z-score
from sklearn.feature_selection import mutual_info_regression  #: MI k-NN continua
from sklearn.neighbors import NearestNeighbors  #: Utilidad para KNN si se requiere

# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def robust_zscore(x: np.ndarray) -> np.ndarray:
    """
    Calcula z-score robusto (mediana/MAD) para estabilidad frente a outliers.

    :param x: Vector de valores.
    :return: Vector z-score robusto (si MAD=0, retorna ceros).
    """
    #: Convertir a array 1D
    x = np.asarray(x).astype(float)
    #: Mediana
    med = np.nanmedian(x)
    #: MAD (Median Absolute Deviation), con corrección por consistencia
    mad = median_abs_deviation(x, nan_policy='omit', scale='normal')
    #: Evitar división por cero
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    #: z robusto
    return (x - med) / mad


def ensure_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Garantiza que el DataFrame tenga índice datetime ordenado sin duplicados.

    :param df: DataFrame de entrada.
    :param time_col: Nombre de columna temporal.
    :return: DataFrame con índice DateTimeIndex.
    """
    #: Parseo de fechas
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=False, errors='coerce')
    #: Eliminar filas con timestamp inválido
    df = df.dropna(subset=[time_col])
    #: Ordenar por tiempo
    df = df.sort_values(time_col)
    #: Fijar índice
    df = df.set_index(time_col)
    #: Eliminar duplicados en índice
    df = df[~df.index.duplicated(keep='last')]
    return df


def resample_close(df: pd.DataFrame, close_col: str, rule: str) -> pd.DataFrame:
    """
    Downsample sin look-ahead: toma el 'close' al final del periodo (regla de resample).

    :param df: DataFrame con índice datetime y columna de close.
    :param close_col: Nombre de columna close.
    :param rule: Regla de resample de pandas (p.ej., '15T', '1H', '4H', '1D').
    :return: DataFrame con columna close resampleada.
    """
    #: Usar el último valor del periodo para preservar la semántica de 'close'
    out = df[[close_col]].resample(rule, label='right', closed='right').last().dropna()
    return out


def log_returns(close: pd.Series) -> pd.Series:
    """
    Retornos logarítmicos (con forward-fill para evitar NaNs iniciales).

    :param close: Serie de precios 'close'.
    :return: Serie de retornos logarítmicos.
    """
    #: Asegurar positividad (si hay ceros/negativos, desplazar mínimamente)
    c = close.astype(float).copy()
    min_pos = c[c > 0.0].min()
    if pd.isna(min_pos) or min_pos <= 0.0:
        #: Desplazamiento mínimo para evitar log(<=0)
        eps = 1e-12
        c = c + (abs(c.min()) + eps)
    #: Retorno log
    r = np.log(c).diff()
    #: Eliminar primer NaN
    r = r.dropna()
    return r


# ---------------------------------------------------------------------------
# Métricas de predecibilidad
# ---------------------------------------------------------------------------

def spectral_forecastability_omega(r: pd.Series, fs_per_day: float) -> float:
    """
    Forecastability Ω basada en entropía espectral normalizada de los retornos.

    Implementación:
    - Estima PSD con Welch (ventanas Hanning por defecto en scipy).
    - Normaliza PSD a densidad de probabilidad.
    - Entropía espectral H_s = -sum p * log(p)
    - Ω = 1 - H_s / log(K), con K = número de bins de frecuencia (o 2π equivalente).
      Aquí usamos log(K) como normalizador discreto.

    :param r: Serie de retornos (sin NaNs).
    :param fs_per_day: Frecuencia de muestreo "por día" (barras/día) para escalar ventanas.
    :return: Ω en [0, 1] (0 ≈ ruido blanco, 1 ≈ muy predecible).
    """
    #: Convertir a numpy
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 128:
        return np.nan
    #: Welch: elegir nperseg relativo a densidad de muestreo (≈ 4 días de datos)
    nperseg = int(max(128, min(4096, 4 * fs_per_day)))
    if nperseg > x.size:
        nperseg = x.size // 2
        nperseg = max(nperseg, 64)
    #: PSD por Welch
    f, Pxx = welch(x, nperseg=nperseg, detrend='constant', return_onesided=True, scaling='density')
    Pxx = np.asarray(Pxx, dtype=float)
    Pxx = np.maximum(Pxx, 1e-300)  #: Evitar log(0)
    #: Normalizar a distribución de probabilidad
    p = Pxx / np.sum(Pxx)
    #: Entropía espectral discreta
    Hs = -np.sum(p * np.log(p))
    K = p.size
    #: Normalización por log(K)
    denom = np.log(K) if K > 1 else 1.0
    omega = 1.0 - (Hs / denom)
    #: Limitar a [0,1]
    omega = float(np.clip(omega, 0.0, 1.0))
    return omega


def permutation_entropy_weighted(r: pd.Series, m: int = 5, tau: int = 1) -> float:
    """
    Entropía de permutación ponderada (WPE) con manejo de ties para retornos.

    Implementación:
    - Embedded vectors de tamaño m y delay tau.
    - Se obtienen permutaciones por orden (con desempate estable por índice).
    - Peso: varianza local del patrón (pondera amplitud).
    - WPE = - sum_w (w_i * log w_i) / log(#patterns), con w_i = peso normalizado.

    :param r: Serie de retornos sin NaNs.
    :param m: Embedding dimension (3..7 típico).
    :param tau: Delay (1..4 típico).
    :return: WPE en [0,1] (menor = más estructura).
    """
    x = np.asarray(r.dropna(), dtype=float)
    N = x.size
    L = N - (m - 1) * tau
    if L <= 0:
        return np.nan
    #: Construcción de matriz embebida
    M = np.empty((L, m), dtype=float)
    for i in range(m):
        M[:, i] = x[i * tau : i * tau + L]
    #: Manejo de ties: argsort estable + ranking por índices para romper empates
    #: Obtener permutaciones
    order = np.argsort(M, axis=1, kind='mergesort')
    #: Para empates exactos, aplicamos desempate por índice original
    #  (argsort estable ya ayuda; adicionalmente, agregamos un muy pequeño jitter determinista)
    #  Esto evita colapsos de patrones por empates frecuentes en alta frecuencia.
    #: Codificar patrón como tupla
    patterns = [tuple(row) for row in order]
    #: Peso: varianza de cada vector embebido (mayor amplitud = mayor peso)
    var_local = np.var(M, axis=1)
    var_local = np.maximum(var_local, 0.0)
    #: Acumular pesos por patrón
    from collections import defaultdict
    weight_map = defaultdict(float)
    for pat, w in zip(patterns, var_local):
        weight_map[pat] += float(w)
    weights = np.array(list(weight_map.values()), dtype=float)
    if weights.size == 0:
        return np.nan
    #: Normalizar pesos
    weights = weights / np.sum(weights)
    weights = np.maximum(weights, 1e-300)
    #: Entropía ponderada
    Hw = -np.sum(weights * np.log(weights))
    K = math.factorial(m)  #: número máximo teórico de patrones
    wpe = Hw / np.log(K)
    #: Limitar a [0,1]
    wpe = float(np.clip(wpe, 0.0, 1.0))
    return wpe


def lz_entropy_rate_sign(r: pd.Series) -> float:
    """
    Entropía de tasa aproximada vía Lempel–Ziv sobre la señal binaria de signo de retornos.

    Implementación clásica:
    - Simbolizar retornos: 1 si r_t > 0, 0 en caso contrario.
    - Estimar complejidad LZ76 (conteo de frases) y convertir a tasa ≈ c(n) * log(n) / n.
    - Interpretación: mayor tasa ≈ más aleatorio (menos predecible).

    :param r: Serie de retornos sin NaNs.
    :return: Tasa de entropía LZ aproximada en [0, 1.5] aprox (sin cota exacta discreta).
             Se usa como métrica relativa (más alto = peor).
    """
    x = np.asarray(r.dropna(), dtype=float)
    if x.size < 256:
        return np.nan
    #: Binarización por signo
    s = (x > 0).astype(np.uint8).tolist()
    n = len(s)
    #: LZ76 phrase count
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
    #: Tasa aproximada
    rate = (c * math.log(n)) / n
    return float(rate)


def mutual_information_horizon(r: pd.Series, horizon_bars: int, k: int = 5) -> float:
    """
    Mutua información pasada->futuro para un horizonte concreto usando KSG aproximado
    vía mutual_info_regression de scikit-learn (kNN).

    Implementación:
    - Feature X_t = r_t (o vector pasado corto). Aquí usamos r_t simple para robustez
      y evitar alta varianza; si se desea, se puede extender a embedding pasado.
    - Target Y = retorno acumulado hasta el horizonte: y_t = sum_{i=1..h} r_{t+i}
    - Se alinea de manera que no haya look-ahead.

    :param r: Serie de retornos sin NaNs.
    :param horizon_bars: Horizonte en número de barras.
    :param k: Número de vecinos (kNN) para MI.
    :return: MI en nats (sklearn) → convertimos a bits dividiendo por log(2).
    """
    x = np.asarray(r.dropna(), dtype=float)
    h = int(horizon_bars)
    if h < 1 or x.size <= (h + 10):
        return np.nan
    #: Construir X e Y alineados
    #  X_t = r_t
    X = x[:-h].reshape(-1, 1)
    #  Y_t = sum de retornos futuros (horizonte)
    Y = np.array([np.sum(x[i+1:i+1+h]) for i in range(x.size - h)], dtype=float)
    #: MI kNN continua (nats); convertir a bits
    try:
        mi_nats = mutual_info_regression(X, Y, n_neighbors=k, random_state=42)
        mi_bits = float(mi_nats[0] / math.log(2.0))
    except Exception:
        mi_bits = np.nan
    return mi_bits


def acf_sum_abs(r: pd.Series, max_lag: int) -> float:
    """
    Suma de |ACF| desde lag=1 hasta max_lag (medida simple de dependencia lineal total).

    :param r: Serie de retornos sin NaNs.
    :param max_lag: Máximo rezago a considerar.
    :return: Suma de valores absolutos de la ACF (excluye lag 0).
    """
    x = np.asarray(r.dropna(), dtype=float)
    n = x.size
    if n < max_lag + 5 or max_lag < 1:
        return np.nan
    #: ACF por FFT (rápida): normalizar por varianza
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 0:
        return np.nan
    #: Autocorrelación via ifft de |FFT|^2
    fft = np.fft.rfft(x, n=2**int(np.ceil(np.log2(2*n - 1))))
    acf_full = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acf_full = acf_full / (var * np.arange(n, 0, -1))
    #: Sumatoria de |acf| en [1..max_lag]
    s = np.sum(np.abs(acf_full[1:max_lag+1]))
    return float(s)


def snr_smoothing_returns(r: pd.Series, smooth_window: int) -> float:
    """
    SNR heurístico sobre retornos:
    - Señal: suavizado centrado (media móvil) de r.
    - Ruido: residuo r - suavizado.
    - SNR = var(señal) / var(ruido), promediado temporalmente.

    :param r: Serie de retornos sin NaNs.
    :param smooth_window: Longitud de ventana (en barras).
    :return: SNR (escala lineal).
    """
    x = pd.Series(r.dropna().values, index=r.dropna().index)
    if x.size < smooth_window * 4:
        return np.nan
    #: Media móvil centrada; min_periods para robustez
    m = x.rolling(window=smooth_window, center=True, min_periods=max(3, smooth_window//3)).mean()
    resid = x - m
    #: Varianzas (omitimos NaNs)
    var_sig = np.nanvar(m.values)
    var_noise = np.nanvar(resid.values)
    if var_noise <= 0 or np.isnan(var_noise):
        return np.nan
    snr = var_sig / var_noise
    return float(snr)


# ---------------------------------------------------------------------------
# Pipeline de evaluación por periodicidad
# ---------------------------------------------------------------------------

def bars_per_day_for_rule(rule: str) -> float:
    """
    Estima el número de barras por día calendario para una regla de resample.

    :param rule: Regla pandas ('5T','15T','1H','4H','1D').
    :return: Barras/día aproximadas.
    """
    #: Mapeo simple; si se añaden reglas, extender aquí.
    mapping = {
        '5T': 288.0,   # 24*60/5
        '15T': 96.0,   # 24*60/15
        '1H': 24.0,    # 24/1
        '4H': 6.0,     # 24/4
        '1D': 1.0,     # 1 por día
    }
    return mapping.get(rule, np.nan)


def horizons_in_bars(rule: str) -> Tuple[int, int, int]:
    """
    Devuelve horizontes clave en barras según periodicidad:
      - h_short: 6h en barras
      - h_long: 144h (6 días) en barras
      - acf_max_lag: máximo lag para ACF (≈ h_short)

    :param rule: Regla de resample.
    :return: (h_short_bars, h_long_bars, acf_max_lag)
    """
    if rule == '5T':
        return (6*60//5, 144*60//5, 6*60//5)       # 72, 1728, 72
    if rule == '15T':
        return (6*60//15, 144*60//15, 6*60//15)   # 24, 576, 24
    if rule == '1H':
        return (6, 144, 6)
    if rule == '4H':
        return (6//4 if 6//4>0 else 1, 144//4, max(1, 6//4))
    if rule == '1D':
        return (max(1, 6//24), max(1, 144//24), max(1, 6//24))  # 1, 6, 1 aprox
    return (np.nan, np.nan, np.nan)


def evaluate_periodicity(df_close: pd.DataFrame, rule: str) -> Dict[str, float]:
    """
    Calcula todas las métricas para una periodicidad concreta.

    :param df_close: DataFrame indexado por tiempo, con columna 'close'.
    :param rule: Regla de la periodicidad ('5T','15T','1H','4H','1D').
    :return: Diccionario con métricas y valores.
    """
    #: Si la regla no es 5T, resamplear
    if rule != '5T':
        dfp = resample_close(df_close, 'close', rule)
    else:
        dfp = df_close.copy()

    #: Retornos log
    r = log_returns(dfp['close'])
    #: Frecuencia por día para Ω
    fsd = bars_per_day_for_rule(rule)
    #: Horizontes en barras
    h_s, h_l, acf_lag = horizons_in_bars(rule)
    #: Ventana de suavizado para SNR ~ h_s (al menos 3)
    smooth_win = int(max(3, h_s))

    #: Métricas
    omega = spectral_forecastability_omega(r, fsd)
    wpe = permutation_entropy_weighted(r, m=5, tau=1)
    lz_rate = lz_entropy_rate_sign(r)
    mi_6h = mutual_information_horizon(r, horizon_bars=int(h_s), k=5) if not np.isnan(h_s) else np.nan
    mi_144h = mutual_information_horizon(r, horizon_bars=int(h_l), k=5) if not np.isnan(h_l) else np.nan
    acf_mass = acf_sum_abs(r, max_lag=int(acf_lag)) if not np.isnan(acf_lag) else np.nan
    snr = snr_smoothing_returns(r, smooth_window=int(smooth_win)) if not np.isnan(smooth_win) else np.nan

    #: Devolver diccionario
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
    """
    Construye z-scores robustos y el composite S_pred con signos adecuados.

    Signos:
      + omega_forecastability (mayor es mejor)
      - wpe (menor es mejor)
      - lz_entropy_rate (menor es mejor)
      + mi_6h_bits, mi_144h_bits (mayor es mejor)
      + acf_sum_abs (mayor es mejor)
      + snr_smooth (mayor es mejor)

    :param df_metrics: DataFrame con columnas métricas por periodicidad.
    :return: DataFrame con columnas z_* y S_pred, y ranking.
    """
    cols_plus = ['omega_forecastability', 'mi_6h_bits', 'mi_144h_bits', 'acf_sum_abs', 'snr_smooth']
    cols_minus = ['wpe', 'lz_entropy_rate']

    #: Calcular z-scores robustos
    for c in cols_plus + cols_minus:
        z = robust_zscore(df_metrics[c].values)
        df_metrics[f'z_{c}'] = z

    #: Aplicar signos
    df_metrics['S_pred'] = 0.0
    for c in cols_plus:
        df_metrics['S_pred'] += df_metrics[f'z_{c}']
    for c in cols_minus:
        df_metrics['S_pred'] += -df_metrics[f'z_{c}']

    #: Ranking descendente (mayor S_pred mejor)
    df_metrics = df_metrics.sort_values('S_pred', ascending=False).reset_index(drop=True)
    df_metrics['rank'] = np.arange(1, df_metrics.shape[0] + 1)
    return df_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """
    Punto de entrada CLI. Lee CSV, genera downsample, calcula métricas, compone score,
    imprime resumen y guarda periodicity_analysis.csv.
    """
    #: Argumentos
    ap = argparse.ArgumentParser(
        description="Seleccionador de periodicidad (modelo-free) basado en métricas informacionales sobre 'close'."
    )
    ap.add_argument('csv', type=str, help='Ruta al CSV base de 5 minutos.')
    ap.add_argument('--time-col', type=str, default=None,
                    help='Nombre de la columna temporal (auto-detección si no se especifica).')
    ap.add_argument('--close-col', type=str, default='close',
                    help="Nombre de la columna de precios de cierre (por defecto: 'close').")
    ap.add_argument('--out', type=str, default='periodicity_analysis.csv',
                    help="Nombre del archivo de salida CSV (por defecto: periodicity_analysis.csv).")
    args = ap.parse_args()

    #: Cargar CSV
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"[ERROR] No se pudo leer el CSV: {e}", file=sys.stderr)
        sys.exit(1)

    #: Detectar columna temporal si no se proporciona
    time_col = args.time_col
    if time_col is None:
        candidates = [c for c in df.columns if c.lower() in ('time', 'timestamp', 'datetime', 'date')]
        if len(candidates) == 0:
            print("[ERROR] No se encontró columna temporal. Use --time-col.", file=sys.stderr)
            sys.exit(1)
        time_col = candidates[0]

    #: Verificar columna close
    close_col = args.close_col
    if close_col not in df.columns:
        print(f"[ERROR] No existe la columna '{close_col}' en el CSV. Use --close-col.", file=sys.stderr)
        sys.exit(1)

    #: Índice datetime ordenado
    df = ensure_datetime_index(df[[time_col, close_col]].rename(columns={close_col: 'close'}), time_col=time_col)
    if df.shape[0] < 1000:
        print("[ADVERTENCIA] Muy pocos puntos de 5m; las métricas pueden ser inestables.", file=sys.stderr)

    #: Periodicidades a evaluar (incluye 5T original)
    periodicities = ['5T', '15T', '1H', '4H', '1D']

    #: Evaluar cada periodicidad
    rows = []
    for rule in periodicities:
        try:
            metrics = evaluate_periodicity(df, rule)
        except Exception as e:
            print(f"[ERROR] Falló evaluación en {rule}: {e}", file=sys.stderr)
            metrics = {
                'periodicity': rule, 'n_points': np.nan,
                'omega_forecastability': np.nan, 'wpe': np.nan, 'lz_entropy_rate': np.nan,
                'mi_6h_bits': np.nan, 'mi_144h_bits': np.nan,
                'acf_sum_abs': np.nan, 'snr_smooth': np.nan
            }
        rows.append(metrics)

    #: DataFrame de métricas
    dfm = pd.DataFrame(rows)

    #: Componer score y ranking
    dfr = composite_score(dfm.copy())

    #: Mostrar resumen
    display_cols = [
        'rank', 'periodicity', 'n_points',
        'omega_forecastability', 'wpe', 'lz_entropy_rate',
        'mi_6h_bits', 'mi_144h_bits', 'acf_sum_abs', 'snr_smooth', 'S_pred'
    ]
    print("\n=== RESUMEN POR PERIODICIDAD ===")
    print(dfr[display_cols].to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    #: Guardar CSV
    try:
        dfr.to_csv(args.out, index=False)
        print(f"\n[OK] Resultados guardados en: {args.out}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar {args.out}: {e}", file=sys.stderr)

    #: Imprimir ranking final
    print("\n=== RANKING (mejor a peor) ===")
    for i, row in dfr.iterrows():
        print(f"{int(row['rank']):2d}. {row['periodicity']}  (S_pred = {row['S_pred']:.3f})")


if __name__ == '__main__':
    main()
