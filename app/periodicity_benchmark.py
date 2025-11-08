#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
periodicity_benchmark.py
========================

Benchmark de modelos de regresión por periodicidad y horizonte con baseline y early stopping.
- Periodicidades: 5T, 15T, 1H, 4H, 1D (downsample desde CSV base 5T).
- Horizontes: 3 horas y 3 días (72h).
- Modelos: LinearRegression (LR), MLP de dos capas (Keras si disponible; fallback sklearn), XGBoost (si disponible).
- Ventana de entrada = número de barras equivalente al horizonte.
- Train/valid: 2 años para train, 1 año para valid (time-based split).
- Métrica: MAE sobre precio futuro close_{t+h}.
- Baseline: NAIVE (persistencia) = y_hat = último valor de la ventana.
- MLP: early stopping con paciencia=20 monitorizando val_mae (si Keras); fallback sklearn con early_stopping.

Uso:
    python periodicity_benchmark.py path/to/data_5m.csv \
      [--time-col TIME] [--close-col CLOSE] \
      [--out periodicity_benchmark.csv] \
      [--mlp-width 128] [--mlp-second 64] \
      [--seed 42] [--no-tqdm] [--quiet]
"""

# ------------------------------
# Importaciones estándar
# ------------------------------
import argparse                   # Manejo de argumentos CLI
import sys                        # Salida estándar y errores
import math                       # Utilidades matemáticas
from typing import Dict, List, Tuple  # Tipado estático opcional

# ------------------------------
# Paquetes científicos
# ------------------------------
import numpy as np                # Cálculo numérico
import pandas as pd               # Series y DataFrames
from tqdm import tqdm             # Barras de progreso

# Modelos de sklearn
from sklearn.linear_model import LinearRegression            # Regresión lineal
from sklearn.neural_network import MLPRegressor             # Fallback MLP (si no hay Keras)
from sklearn.preprocessing import StandardScaler            # Estandarización de features
from sklearn.pipeline import Pipeline                       # Encadenamiento de pasos
from sklearn.metrics import mean_absolute_error             # MAE

# XGBoost opcional
try:
    from xgboost import XGBRegressor                        # Regressor XGBoost
    XGB_AVAILABLE = True                                    # Flag si XGBoost está disponible
except Exception:
    XGB_AVAILABLE = False                                   # Si no se puede importar, se desactiva

# Keras/TensorFlow opcional (para MLP con early stopping sobre val_mae)
try:
    import tensorflow as tf                                  # Importa TensorFlow
    from tensorflow import keras                             # Alias a Keras
    KERAS_AVAILABLE = True                                   # Flag de disponibilidad
except Exception:
    KERAS_AVAILABLE = False                                  # Si falla la importación, fallback sklearn


# ------------------------------
# Utilidades de impresión con flush
# ------------------------------
def info(msg: str, quiet: bool = False) -> None:
    """Imprime mensajes informativos con flush inmediato si no está en modo silencioso."""
    if not quiet:
        print(msg, flush=True)

def warn(msg: str) -> None:
    """Imprime advertencias a stderr con flush inmediato."""
    print(f"[ADVERTENCIA] {msg}", file=sys.stderr, flush=True)

def error(msg: str) -> None:
    """Imprime errores a stderr con flush inmediato."""
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


# ------------------------------
# Mapeos y conversiones de periodicidad
# ------------------------------
def hours_per_bar(rule: str) -> float:
    """Devuelve el tamaño de barra en horas para cada regla canónica."""
    mapping = {'5T': 5.0/60.0, '15T': 15.0/60.0, '1H': 1.0, '4H': 4.0, '1D': 24.0}
    return mapping.get(rule, np.nan)

def to_bars_or_nan(hours: float, rule: str) -> float:
    """
    Convierte un horizonte en horas a número de barras para una periodicidad dada.
    Si el horizonte es menor que el tamaño de barra, retorna NaN (no representable).
    """
    hp = hours_per_bar(rule)                              # Obtiene horas por barra
    if np.isnan(hp) or hp <= 0:                           # Valida valor
        return np.nan
    if hours < hp:                                        # Si el horizonte es menor que la barra, no aplica
        return np.nan
    return int(max(1, round(hours / hp)))                 # Redondea a entero >= 1


# ------------------------------
# Lectura y resampleo
# ------------------------------
def ensure_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Asegura índice temporal ordenado, sin duplicados y tipo datetime."""
    df = df.copy()                                        # Evita modificar original
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=False)  # Convierte a datetime
    df = df.dropna(subset=[time_col])                     # Descarta filas sin tiempo válido
    df = df.sort_values(time_col).set_index(time_col)     # Ordena y define índice
    df = df[~df.index.duplicated(keep='last')]            # Quita duplicados en índice
    return df                                             # Retorna DataFrame normalizado

def resample_close(df: pd.DataFrame, close_col: str, rule: str) -> pd.DataFrame:
    """
    Realiza downsample sin look-ahead: toma el último close del intervalo (label/right).
    Esto replica la política del selector de periodicidad.
    """
    out = df[[close_col]].resample(rule, label='right', closed='right').last().dropna()
    return out


# ------------------------------
# Construcción del dataset supervisado
# ------------------------------
def make_supervised_from_close(close: pd.Series,
                               window: int,
                               horizon: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Genera X, y y los timestamps a partir de una serie de cierres:
    - X: ventanas deslizantes de tamaño 'window' con valores de close.
    - y: precio futuro close en t + horizon barras.
    - Retorna también el índice temporal de cada muestra (el del punto t + horizon).

    Notas:
    - No se mezcla el futuro en el pasado (no look-ahead).
    - Si no hay suficientes datos, se devuelven arrays vacíos.
    """
    c = close.astype(float).values                        # Extrae numpy array de precios
    n = c.shape[0]                                        # Número de puntos
    if n < (window + horizon + 1):                        # Verifica tamaño mínimo
        return np.empty((0, window), dtype=float), np.empty((0,), dtype=float), close.index[:0]
    # Número de muestras posibles
    m = n - window - horizon                              # Cuenta de ventanas válidas
    # Inicializa matrices de salida
    X = np.empty((m, window), dtype=float)                # Matriz de features
    y = np.empty((m,), dtype=float)                       # Vector de etiquetas
    # Construye ventanas deslizantes
    for i in range(m):                                    # Itera sobre posiciones de inicio
        X[i, :] = c[i:i+window]                           # Ventana de tamaño 'window'
        y[i] = c[i + window + horizon - 1]                # Precio en t+window+(h-1) => close futuro
    # Índices temporales alineados a y
    time_index = close.index[window + horizon - 1 : window + horizon - 1 + m]  # Índices de y
    return X, y, time_index                               # Retorna dataset supervisado


# ------------------------------
# Split temporal robusto (2 años train, 1 año valid)
# ------------------------------
def time_based_train_valid_split(index: pd.DatetimeIndex,
                                 years_train: int = 2,
                                 years_valid: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Masks for time-based split. Returns NumPy boolean arrays.
    """
    if index.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)

    end = index.max()
    valid_start = end - pd.Timedelta(days=365 * years_valid)
    train_start = valid_start - pd.Timedelta(days=365 * years_train)

    min_time = index.min()
    if train_start < min_time:
        train_start = min_time

    # boolean numpy arrays (no .values needed)
    is_valid = (index > valid_start) & (index <= end)
    is_train = (index > train_start) & (index <= valid_start)

    if is_train.sum() < 100:
        warn("Conjunto de entrenamiento resultó pequeño (<100 muestras). Los resultados pueden ser inestables.")
    if is_valid.sum() < 50:
        warn("Conjunto de validación resultó pequeño (<50 muestras). Los resultados pueden ser ruidosos.")

    # Return directly; they are already np.ndarray[bool]
    return np.asarray(is_train, dtype=bool), np.asarray(is_valid, dtype=bool)


# ------------------------------
# Entrenamiento y evaluación de modelos + baseline
# ------------------------------
def fit_and_eval_models(X: np.ndarray,
                        y: np.ndarray,
                        idx: pd.DatetimeIndex,
                        seed: int,
                        mlp_width: int,
                        mlp_second: int,
                        quiet: bool = False) -> Dict[str, float]:
    """
    Entrena y evalúa:
      - Baseline NAIVE (persistencia) en train y valid,
      - LinearRegression (LR) con StandardScaler,
      - MLP 2 capas con early stopping monitorizando val_mae (si Keras),
        o fallback sklearn MLPRegressor (early_stopping interno),
      - XGBoost (si está disponible).
    Retorna MAE de train y valid para cada uno.
    """
    # Obtiene máscaras de train/valid basadas en tiempo
    is_train, is_valid = time_based_train_valid_split(idx, years_train=2, years_valid=1)
    # Verifica que haya muestras en ambos conjuntos
    if is_train.sum() == 0 or is_valid.sum() == 0:
        warn("Split temporal produjo conjuntos vacíos. Saltando evaluación en este caso.")
        return {
            'NAIVE_TRAIN_MAE': np.nan, 'NAIVE_VALID_MAE': np.nan,
            'LR_TRAIN_MAE': np.nan, 'LR_VALID_MAE': np.nan,
            'MLP_TRAIN_MAE': np.nan, 'MLP_VALID_MAE': np.nan,
            'XGB_TRAIN_MAE': np.nan if XGB_AVAILABLE else np.nan,
            'XGB_VALID_MAE': np.nan if XGB_AVAILABLE else np.nan,
        }

    # Extrae subconjuntos
    Xtr, ytr = X[is_train], y[is_train]                   # Conjunto de entrenamiento
    Xva, yva = X[is_valid], y[is_valid]                   # Conjunto de validación

    # ---------------- Baseline NAIVE (persistencia) ----------------
    # Predice el futuro como el último valor de la ventana de entrada
    yhat_naive_tr = Xtr[:, -1]                            # Persistencia en train
    yhat_naive_va = Xva[:, -1]                            # Persistencia en valid
    naive_train_mae = mean_absolute_error(ytr, yhat_naive_tr)  # MAE train baseline
    naive_valid_mae = mean_absolute_error(yva, yhat_naive_va)  # MAE valid baseline

    # ---------------- Linear Regression (LR) ----------------
    lr_pipe = Pipeline(steps=[
        ('scaler', StandardScaler(with_mean=True, with_std=True)),   # Estandarización
        ('lr', LinearRegression())                                   # Regresión lineal
    ])
    lr_pipe.fit(Xtr, ytr)                                            # Entrena LR
    yhat_lr_tr = lr_pipe.predict(Xtr)                                # Predicción en train
    yhat_lr_va = lr_pipe.predict(Xva)                                # Predicción en valid
    mae_lr_tr = mean_absolute_error(ytr, yhat_lr_tr)                 # MAE train
    mae_lr_va = mean_absolute_error(yva, yhat_lr_va)                 # MAE valid

    # ---------------- MLP (2 capas) ----------------
    if KERAS_AVAILABLE:
        # Escalado estándar (fit en train, transform en train/valid)
        scaler = StandardScaler(with_mean=True, with_std=True)       # Crea scaler
        Xtr_s = scaler.fit_transform(Xtr)                            # Ajusta en train
        Xva_s = scaler.transform(Xva)                                # Aplica en valid

        # Define MLP dos capas (mlp_width, mlp_second)
        model = keras.Sequential([
            keras.layers.Input(shape=(Xtr_s.shape[1],)),             # Entrada = dimensión de ventana
            keras.layers.Dense(mlp_width, activation='relu'),        # Capa oculta 1
            keras.layers.Dense(mlp_second, activation='relu'),       # Capa oculta 2
            keras.layers.Dense(1, activation='linear')               # Salida escalar (precio futuro)
        ])
        # Compila el modelo con MAE como loss y métrica
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss='mae',
                      metrics=['mae'])

        # Early stopping: monitoriza val_mae con paciencia=20 (restore best weights)
        es = keras.callbacks.EarlyStopping(monitor='val_mae',
                                           patience=20,
                                           restore_best_weights=True,
                                           mode='min',
                                           verbose=0)

        # Entrena con validación explícita en el conjunto holdout (2y/1y)
        model.fit(Xtr_s, ytr,
                  validation_data=(Xva_s, yva),
                  epochs=200,
                  batch_size=256,
                  shuffle=True,
                  verbose=0,
                  callbacks=[es])

        # Predicciones en train/valid
        yhat_mlp_tr = model.predict(Xtr_s, verbose=0).ravel()
        yhat_mlp_va = model.predict(Xva_s, verbose=0).ravel()
        mae_mlp_tr = mean_absolute_error(ytr, yhat_mlp_tr)          # MAE train
        mae_mlp_va = mean_absolute_error(yva, yhat_mlp_va)          # MAE valid
    else:
        warn("TensorFlow/Keras no disponible. Usando fallback sklearn.MLPRegressor con early_stopping interno.")
        mlp = MLPRegressor(hidden_layer_sizes=(mlp_width, mlp_second),   # Dos capas ocultas
                           activation='relu',                            # Activación ReLU
                           solver='adam',                                # Optimizador Adam
                           alpha=1e-4,                                   # Regularización L2
                           batch_size='auto',                            # Tamaño de batch
                           learning_rate='adaptive',                     # LR adaptativo
                           learning_rate_init=1e-3,                      # LR inicial
                           max_iter=400,                                 # Más iteraciones permitidas
                           shuffle=True,                                 # Barajado por época
                           random_state=seed,                            # Semilla
                           early_stopping=True,                          # Early stopping interno (val split interno)
                           n_iter_no_change=20,                          # Paciencia aproximada
                           tol=1e-4,                                     # Tolerancia
                           verbose=False)                                # Silencioso
        mlp_pipe = Pipeline(steps=[
            ('scaler', StandardScaler(with_mean=True, with_std=True)),   # Estandarización
            ('mlp', mlp)                                                 # MLP sklearn
        ])
        mlp_pipe.fit(Xtr, ytr)                                           # Entrena MLP (con val interna)
        yhat_mlp_tr = mlp_pipe.predict(Xtr)                              # Pred train
        yhat_mlp_va = mlp_pipe.predict(Xva)                              # Pred valid
        mae_mlp_tr = mean_absolute_error(ytr, yhat_mlp_tr)               # MAE train
        mae_mlp_va = mean_absolute_error(yva, yhat_mlp_va)               # MAE valid

    # ---------------- XGBoost (opcional) ----------------
    if XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=300,              # N árboles
            max_depth=6,                   # Profundidad
            learning_rate=0.05,            # Eta
            subsample=0.8,                 # Submuestreo filas
            colsample_bytree=0.8,          # Submuestreo columnas
            reg_lambda=1.0,                # L2
            reg_alpha=0.0,                 # L1
            random_state=seed,             # Semilla
            n_jobs=0,                      # Usa todos los cores
            tree_method='hist',            # Rápido y memoria eficiente
        )
        xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)      # Entrena XGB
        yhat_xgb_tr = xgb.predict(Xtr)                               # Pred train
        yhat_xgb_va = xgb.predict(Xva)                               # Pred valid
        mae_xgb_tr = mean_absolute_error(ytr, yhat_xgb_tr)           # MAE train
        mae_xgb_va = mean_absolute_error(yva, yhat_xgb_va)           # MAE valid
    else:
        warn("XGBoost no está disponible. Se omitirá este modelo.")
        mae_xgb_tr = np.nan
        mae_xgb_va = np.nan

    # Retorna MAEs incluyendo baseline y train/valid por modelo
    return {
        'NAIVE_TRAIN_MAE': float(naive_train_mae),
        'NAIVE_VALID_MAE': float(naive_valid_mae),
        'LR_TRAIN_MAE': float(mae_lr_tr),
        'LR_VALID_MAE': float(mae_lr_va),
        'MLP_TRAIN_MAE': float(mae_mlp_tr),
        'MLP_VALID_MAE': float(mae_mlp_va),
        'XGB_TRAIN_MAE': float(mae_xgb_tr),
        'XGB_VALID_MAE': float(mae_xgb_va),
    }


# ------------------------------
# Proceso por periodicidad y horizonte
# ------------------------------
def evaluate_periodicity_and_horizon(df5m: pd.DataFrame,
                                     rule: str,
                                     horizon_hours: float,
                                     seed: int,
                                     mlp_width: int,
                                     mlp_second: int,
                                     use_tqdm: bool,
                                     quiet: bool) -> Dict[str, float]:
    """
    Para una periodicidad y un horizonte:
    - Downsamplea (si aplica).
    - Construye dataset con ventana = #barras del horizonte y target = close futuro a h barras.
    - Entrena/evalúa LR, MLP (con early stopping) y XGB (si disponible) y baseline NAIVE.
    - Retorna dict con métricas y metadatos.
    """
    info(f"[{rule}] Preparando close y construyendo dataset para H={horizon_hours}h…", quiet)  # Log con flush
    df_rule = resample_close(df5m, 'close', rule) if rule != '5T' else df5m.copy()              # Downsample

    # Determina tamaño de ventana y horizonte en barras (ventana = horizonte en barras)
    window = to_bars_or_nan(horizon_hours, rule)
    horizon_bars = window
    if (window is np.nan) or np.isnan(window):
        warn(f"[{rule}] Horizonte {horizon_hours}h < tamaño de barra. Caso no representable; se omite.")
        return {
            'periodicity': rule,
            'horizon_hours': horizon_hours,
            'window_bars': np.nan,
            'n_samples': 0,
            'NAIVE_TRAIN_MAE': np.nan, 'NAIVE_VALID_MAE': np.nan,
            'LR_TRAIN_MAE': np.nan, 'LR_VALID_MAE': np.nan,
            'MLP_TRAIN_MAE': np.nan, 'MLP_VALID_MAE': np.nan,
            'XGB_TRAIN_MAE': np.nan, 'XGB_VALID_MAE': np.nan,
        }

    close = df_rule['close'].astype(float)                                                    # Serie close
    X, y, idx = make_supervised_from_close(close, window=int(window), horizon=int(horizon_bars))  # Dataset
    if X.shape[0] == 0:
        warn(f"[{rule}] No hay suficientes datos para ventana={window} y horizonte={horizon_bars} barras.")
        return {
            'periodicity': rule,
            'horizon_hours': horizon_hours,
            'window_bars': int(window),
            'n_samples': 0,
            'NAIVE_TRAIN_MAE': np.nan, 'NAIVE_VALID_MAE': np.nan,
            'LR_TRAIN_MAE': np.nan, 'LR_VALID_MAE': np.nan,
            'MLP_TRAIN_MAE': np.nan, 'MLP_VALID_MAE': np.nan,
            'XGB_TRAIN_MAE': np.nan, 'XGB_VALID_MAE': np.nan,
        }

    metrics = fit_and_eval_models(X, y, idx, seed=seed, mlp_width=mlp_width, mlp_second=mlp_second, quiet=quiet)
    out = {
        'periodicity': rule,
        'horizon_hours': horizon_hours,
        'window_bars': int(window),
        'n_samples': int(X.shape[0]),
        **metrics
    }
    return out


# ------------------------------
# CLI principal
# ------------------------------
def main():
    """Punto de entrada principal del script CLI."""
    ap = argparse.ArgumentParser(
        description="Benchmark de LR/MLP/XGBoost por periodicidad y horizonte (3h y 3d) con baseline NAIVE y early stopping en MLP."
    )
    ap.add_argument('csv', type=str, help='Ruta al CSV base (periodicidad 5T).')
    ap.add_argument('--time-col', type=str, default=None, help='Nombre de la columna temporal.')
    ap.add_argument('--close-col', type=str, default='close', help="Nombre de la columna de cierre (default: 'close').")
    ap.add_argument('--out', type=str, default='periodicity_benchmark.csv', help='Archivo CSV de salida para resultados.')
    ap.add_argument('--mlp-width', type=int, default=128, help='Unidades de la primera capa oculta del MLP.')
    ap.add_argument('--mlp-second', type=int, default=64, help='Unidades de la segunda capa oculta del MLP.')
    ap.add_argument('--seed', type=int, default=42, help='Semilla de aleatoriedad para reproducibilidad.')
    ap.add_argument('--no-tqdm', action='store_true', help='Desactiva tqdm en loops externos.')
    ap.add_argument('--quiet', action='store_true', help='Reduce verbosidad de impresión.')
    args = ap.parse_args()

    info("[LEGEND] LR=Linear Regression; MLP=Multi-Layer Perceptron; XGB=XGBoost; "
         "MAE=Mean Absolute Error; NAIVE=Persistence baseline (predict next price = last observed in window).",
         args.quiet)

    # Carga CSV base 5T
    info("[INIT] Cargando CSV base (5T)…", args.quiet)
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        error(f"No se pudo leer el CSV: {e}")
        sys.exit(1)
    info(f"[INIT] CSV cargado: shape={df.shape}", args.quiet)

    # Detecta columna temporal si no se especificó
    time_col = args.time_col
    if time_col is None:
        candidates = [c for c in df.columns if c.lower() in ('time', 'timestamp', 'datetime', 'date')]
        if len(candidates) == 0:
            error("No se encontró una columna temporal. Use --time-col.")
            sys.exit(1)
        time_col = candidates[0]
        info(f"[INIT] Columna temporal detectada: {time_col}", args.quiet)

    # Verifica columna de cierre
    if args.close_col not in df.columns:
        error(f"No existe la columna '{args.close_col}'. Use --close-col.")
        sys.exit(1)

    # Normaliza índice temporal
    info("[INIT] Normalizando índice temporal…", args.quiet)
    df = ensure_datetime_index(df[[time_col, args.close_col]].rename(columns={args.close_col: 'close'}), time_col)
    info(f"[INIT] Serie 5T lista: n={df.shape[0]} puntos.", args.quiet)

    # Periodicidades y horizontes
    periodicities = ['5T', '15T', '1H', '4H', '1D']
    horizons = [3.0, 72.0]  # 3 horas y 3 días

    # Ejecuta evaluación
    results: List[Dict[str, float]] = []
    iterator = periodicities if args.no_tqdm else tqdm(periodicities, desc="Evaluando periodicidades", miniters=1)
    for rule in iterator:
        sub_iter = horizons if args.no_tqdm else tqdm(horizons, desc=f"[{rule}] Horizontes", leave=False, miniters=1)
        for H in sub_iter:
            row = evaluate_periodicity_and_horizon(
                df5m=df, rule=rule, horizon_hours=H,
                seed=args.seed, mlp_width=args.mlp_width, mlp_second=args.mlp_second,
                use_tqdm=not args.no_tqdm, quiet=args.quiet
            )
            results.append(row)

    # Compila resultados
    info("[PIPE] Compilando resultados…", args.quiet)
    dfr = pd.DataFrame(results).sort_values(by=['horizon_hours', 'periodicity']).reset_index(drop=True)

    # Imprime resumen formateado (incluye baseline y train/valid por modelo)
    info("\n=== RESUMEN DE MAE (train y valid) POR MODELO / PERIODICIDAD / HORIZONTE ===", False)
    show_cols = [
        'horizon_hours', 'periodicity', 'window_bars', 'n_samples',
        'NAIVE_TRAIN_MAE', 'NAIVE_VALID_MAE',
        'LR_TRAIN_MAE', 'LR_VALID_MAE',
        'MLP_TRAIN_MAE', 'MLP_VALID_MAE',
        'XGB_TRAIN_MAE', 'XGB_VALID_MAE'
    ]
    def _fmt(v):
        try:
            return f"{v:,.6f}"
        except Exception:
            return str(v)
    print(dfr[show_cols].to_string(index=False, float_format=_fmt), flush=True)

    # Guarda CSV
    info(f"\n[PIPE] Guardando resultados en: {args.out}", args.quiet)
    try:
        dfr.to_csv(args.out, index=False)
        info("[OK] Archivo escrito correctamente.", False)
    except Exception as e:
        error(f"No se pudo guardar {args.out}: {e}")


# ------------------------------
# Programa principal
# ------------------------------
if __name__ == '__main__':
    main()
