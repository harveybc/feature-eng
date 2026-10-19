"""Causal fractal descriptors of a close-price path, as a feature-eng plugin.

Port of Retsu's ``tools/causal_fractal_features.py`` (agent-multi, untracked at
the time of the port; sha256 703ffb5afb1d76073fdafeeb5f1e7f6ea27916f3b7777ee5e6f12ba72e09cef4),
with the window exposed as a plugin parameter so that the same estimators can be
computed at several trailing lengths.

Contract
--------
* The value in row ``i`` is a function of ``close[i - window + 1 : i + 1]`` only.
  No scale, slope or normalization is fit on a later row, on validation or on
  test. Altering any bar after ``i`` cannot change row ``i``.
* ``window_bars`` is a count of OBSERVED bars (the same convention as
  ``hurst_proxy_200``), not calendar time. Weekend gaps are therefore not
  interpolated: 200 hourly bars span more than 200 wall-clock hours.
* The first ``window_bars - 1`` rows of any series are NaN (the trailing window
  does not fit yet). A caller wanting a complete split must feed warm-up history
  from before the split and drop it afterwards; the plugin never fills.
* Candidate columns: ``hurst_rs``, ``fractal_dimension_graph`` (= 2 - hurst_rs),
  ``box_counting_dimension``, ``variogram_hurst``, ``hurst_spectrum_width``.
  Diagnostic-only columns (past-only but fragile on one short window):
  ``correlation_dimension``, ``spectral_slope``. Intermediate columns
  ``variogram_slope``, ``hurst_q1``, ``hurst_q2``, ``hurst_q3`` are kept so the
  candidates can be audited from their parts.

Window-dependent behaviour (recorded, not hidden)
-------------------------------------------------
* R/S uses the block scales {8, 16, 32, 64} that fit in the ``window - 1`` log
  returns: windows >= 65 use four scales, windows in [33, 64] use three. The
  minimum window is 33 (three scales); Retsu's module required 64.
* ``correlation_dimension`` embeds every fifth close and needs at least 12
  samples, so it is NaN for every row when ``window_bars < 60``.

Nothing here selects a feature, claims predictive value, or authorizes a strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 200
MIN_WINDOW = 33
SCALES = np.array([8, 16, 32, 64], dtype=int)
TAUS = np.array([1, 2, 4, 8, 16, 32], dtype=int)
QS = (1.0, 2.0, 3.0)
BOX_EPS = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32], dtype=float)
CORRDIM_STRIDE = 5
CORRDIM_MIN_SAMPLES = 12

FEATURE_COLUMNS = (
    "hurst_rs",
    "fractal_dimension_graph",
    "box_counting_dimension",
    "variogram_slope",
    "variogram_hurst",
    "hurst_q1",
    "hurst_q2",
    "hurst_q3",
    "hurst_spectrum_width",
    "correlation_dimension",
    "spectral_slope",
)
CANDIDATE_COLUMNS = (
    "hurst_rs",
    "fractal_dimension_graph",
    "box_counting_dimension",
    "variogram_hurst",
    "hurst_spectrum_width",
)
DIAGNOSTIC_COLUMNS = ("correlation_dimension", "spectral_slope")
INTERMEDIATE_COLUMNS = ("variogram_slope", "hurst_q1", "hurst_q2", "hurst_q3")


def rs_scales_for_window(window: int) -> np.ndarray:
    """Block scales usable by R/S on ``window - 1`` log returns."""
    return SCALES[SCALES <= window - 1]


def _slope(log_x: np.ndarray, log_y: np.ndarray) -> float:
    mask = np.isfinite(log_x) & np.isfinite(log_y)
    if int(mask.sum()) < 2:
        return np.nan
    x = log_x[mask]
    y = log_y[mask]
    x_center = x - x.mean()
    denom = float(np.dot(x_center, x_center))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(x_center, y - y.mean()) / denom)


def _hurst_rs(log_returns: np.ndarray) -> float:
    points = []
    scale_log = []
    for scale in SCALES:
        if scale > log_returns.size:
            continue
        blocks = log_returns.size // int(scale)
        if blocks < 1:
            continue
        usable = log_returns[: blocks * int(scale)].reshape(blocks, int(scale))
        centered = usable - usable.mean(axis=1, keepdims=True)
        profile = np.cumsum(centered, axis=1)
        span = profile.max(axis=1) - profile.min(axis=1)
        sigma = usable.std(axis=1, ddof=1)
        valid = sigma > 0.0
        if not np.any(valid):
            continue
        points.append(np.log(np.mean(span[valid] / sigma[valid])))
        scale_log.append(np.log(float(scale)))
    return _slope(np.asarray(scale_log), np.asarray(points))


def _box_dimension(prices: np.ndarray) -> float:
    low = float(prices.min())
    high = float(prices.max())
    if not np.isfinite(low) or high <= low:
        return np.nan
    x = np.linspace(0.0, 1.0, prices.size)
    y = (prices - low) / (high - low)
    counts = []
    for eps in BOX_EPS:
        ix = np.minimum((x / eps).astype(int), int(1.0 / eps) - 1)
        iy = np.minimum((y / eps).astype(int), int(1.0 / eps) - 1)
        counts.append(np.log(float(np.unique(ix * 1024 + iy).size)))
    return -_slope(np.log(BOX_EPS), np.asarray(counts))


def _variogram(prices: np.ndarray) -> tuple[float, float]:
    logs = []
    gammas = []
    for tau in TAUS:
        if tau >= prices.size:
            continue
        increment = prices[tau:] - prices[:-tau]
        gamma = 0.5 * float(np.mean(increment * increment))
        if gamma <= 0.0:
            continue
        logs.append(np.log(float(tau)))
        gammas.append(np.log(gamma))
    slope = _slope(np.asarray(logs), np.asarray(gammas))
    if not np.isfinite(slope):
        return np.nan, np.nan
    return slope, slope / 2.0


def _generalized_hurst(prices: np.ndarray) -> tuple[float, float, float, float]:
    values = []
    for q in QS:
        logs = []
        moments = []
        for tau in TAUS:
            if tau >= prices.size:
                continue
            increment = np.abs(prices[tau:] - prices[:-tau])
            moment = float(np.mean(increment ** q))
            if moment <= 0.0:
                continue
            logs.append(np.log(float(tau)))
            moments.append(np.log(moment) / q)
        values.append(_slope(np.asarray(logs), np.asarray(moments)))
    h1, h2, h3 = values
    width = h1 - h3 if np.isfinite(h1) and np.isfinite(h3) else np.nan
    return h1, h2, h3, width


def _correlation_dimension(prices: np.ndarray) -> float:
    # Delay-1 embedding of every fifth point. The radius grid is the
    # percentile range of THIS window only.
    sample = prices[::CORRDIM_STRIDE]
    if sample.size < CORRDIM_MIN_SAMPLES:
        return np.nan
    embedded = np.column_stack((sample[:-1], sample[1:]))
    centered = embedded - embedded.mean(axis=0)
    distances = []
    for i in range(centered.shape[0] - 1):
        delta = centered[i + 1 :] - centered[i]
        distances.append(np.sqrt(np.sum(delta * delta, axis=1)))
    dist = np.concatenate(distances)
    dist = dist[dist > 0.0]
    if dist.size < 8:
        return np.nan
    radii = np.quantile(dist, [0.2, 0.4, 0.6, 0.8])
    counts = [np.log(float(np.mean(dist <= radius))) for radius in radii if radius > 0.0]
    usable = np.asarray([radius for radius in radii if radius > 0.0])
    if usable.size < 2:
        return np.nan
    return _slope(np.log(usable), np.asarray(counts[: usable.size]))


def _spectral_slope(log_returns: np.ndarray) -> float:
    series = log_returns - log_returns.mean()
    if float(np.dot(series, series)) <= 0.0:
        return np.nan
    spectrum = np.abs(np.fft.rfft(series)) ** 2
    freqs = np.arange(1, spectrum.size)
    power = spectrum[1:]
    valid = power > 0.0
    if int(valid.sum()) < 4:
        return np.nan
    return _slope(np.log(freqs[valid].astype(float)), np.log(power[valid]))


def causal_fractal_row(chunk: np.ndarray) -> tuple:
    """All descriptors of ONE trailing window (the last element is row i)."""
    if not np.all(np.isfinite(chunk)) or np.any(chunk <= 0.0):
        return tuple([np.nan] * len(FEATURE_COLUMNS))
    log_returns = np.diff(np.log(chunk))
    hurst = _hurst_rs(log_returns)
    var_slope, var_hurst = _variogram(chunk)
    h1, h2, h3, width = _generalized_hurst(chunk)
    return (
        hurst,
        (2.0 - hurst) if np.isfinite(hurst) else np.nan,
        _box_dimension(chunk),
        var_slope,
        var_hurst,
        h1,
        h2,
        h3,
        width,
        _correlation_dimension(chunk),
        _spectral_slope(log_returns),
    )


def causal_fractal_frame(prices, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """One row per price. Rows before a full trailing window are NaN."""
    series = np.asarray(prices, dtype=float)
    if series.ndim != 1:
        raise ValueError("prices must be one series")
    window = int(window)
    if window < MIN_WINDOW:
        raise ValueError(f"window must be >= {MIN_WINDOW} observed bars (three R/S scales)")
    rows = np.full((series.size, len(FEATURE_COLUMNS)), np.nan, dtype=float)
    for end in range(window - 1, series.size):
        rows[end] = causal_fractal_row(series[end - window + 1 : end + 1])
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))


class Plugin:
    """feature-eng plugin: trailing-window fractal descriptors of CLOSE."""

    plugin_params = {
        "window_bars": DEFAULT_WINDOW,     # observed bars, not calendar time
        "close_column": "CLOSE",
        "include_diagnostic": True,        # keep correlation_dimension, spectral_slope
        "include_intermediate": True,      # keep variogram_slope, hurst_q1..q3
        "suffix_window": True,             # name columns <feature>_w<window_bars>
    }

    plugin_debug_vars = ["window_bars", "close_column", "include_diagnostic",
                         "include_intermediate", "suffix_window"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {v: self.params.get(v) for v in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def output_columns(self) -> list:
        cols = list(FEATURE_COLUMNS)
        if not self.params["include_diagnostic"]:
            cols = [c for c in cols if c not in DIAGNOSTIC_COLUMNS]
        if not self.params["include_intermediate"]:
            cols = [c for c in cols if c not in INTERMEDIATE_COLUMNS]
        return [self.column_name(c) for c in cols]

    def column_name(self, feature: str) -> str:
        if self.params["suffix_window"]:
            return f"{feature}_w{int(self.params['window_bars'])}"
        return feature

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the descriptor columns aligned on ``data.index``."""
        close_col = self.params["close_column"]
        if close_col not in data.columns:
            raise KeyError(f"close column '{close_col}' not found in data")
        window = int(self.params["window_bars"])
        frame = causal_fractal_frame(data[close_col].to_numpy(dtype=float), window)
        frame.index = data.index
        frame.columns = [self.column_name(c) for c in FEATURE_COLUMNS]
        return frame[self.output_columns()]

    def process_additional_datasets(self, data, config):
        """No auxiliary datasets; same contract as the label plugins."""
        return pd.DataFrame(index=data.index), None, None
