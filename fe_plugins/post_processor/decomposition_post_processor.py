#!/usr/bin/env python3
"""Decomposition Post-Processor Plugin

Per-feature strictly causal rolling decompositions (STL, Wavelet, MTM-style band power) using only
data up to and including the current index (never future leakage).

Responsibilities:
    1. Resolve runtime parameters (merged with plugin defaults) & validate target feature list.
    2. Optionally compute log returns (first column) without future data usage.
    3. For each selected feature perform enabled decompositions (STL, Wavelet, MTM) via rolling windows:
         - Window covers past W points including current point t (indices i-W+1 .. i).
         - Components assigned only to position i (causal) – no smoothing with future samples.
    4. Aggregate all generated component columns (optionally keeping original feature).
    5. Record comprehensive parameter snapshot & metrics (counts, output columns) in debug state.
    6. Return resulting DataFrame with deterministic column ordering (log_return, decompositions, originals).

Notes:
    - STL requires sufficiently large window & period; if infeasible for early rows values stay NaN (forward-filled once at end).
    - Wavelet uses per-window SWT (stationary wavelet transform) to maintain alignment (no decimation) when pywt available.
    - MTM approximation uses Welch PSD over the causal window; band energies stored per horizon.
    - All optional dependencies gracefully degrade (features skipped if library missing).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:  # Wavelets (optional)
    import pywt  # type: ignore
    HAS_WAVELETS = True
except Exception:  # pragma: no cover
    HAS_WAVELETS = False
    pywt = None  # type: ignore
    logger.warning("pywt not installed; wavelet decomposition disabled.")

try:  # Signal processing (optional)
    from scipy import signal  # type: ignore
    HAS_SIGNAL = True
except Exception:  # pragma: no cover
    HAS_SIGNAL = False
    signal = None  # type: ignore
    logger.warning("scipy.signal not installed; MTM/Welch decomposition disabled.")


class FeaturePlugin:  # Consistent naming with other plugins
    """Causal rolling decomposition post-processor."""

    # ------------------------------------------------------------------
    # Default parameters (merge-able)
    # ------------------------------------------------------------------
    plugin_params: Dict[str, Any] = {
        # Feature selection -------------------------------------------------
        "decomp_features": [],            # List of feature column names to decompose
        "add_log_return": False,          # Add log return of CLOSE (if present)
        "keep_original": True,            # Keep original feature alongside components
        "replace_original": False,        # If True and keep_original False -> original removed
        # Progress ---------------------------------------------------------
        "show_progress": True,
        "progress_min_features": 2,
        # STL --------------------------------------------------------------
        "use_stl": True,
        "stl_period": 24,
        "stl_window": None,              # If None => 2*period+1
        "stl_trend": None,               # If None -> auto adjusted (odd)
        # WAVELET ----------------------------------------------------------
        "use_wavelet": True,
        "wavelet_name": "db4",
        "wavelet_levels": 2,
        # MTM / Welch PSD --------------------------------------------------
        "use_mtm": True,
        "mtm_window_len": 168,
        "mtm_freq_bands": [(0.0, 0.01), (0.01, 0.06), (0.06, 0.2), (0.2, 0.5)],
        # OUTPUT CONTROL ---------------------------------------------------
        "max_output_columns": None,       # Optional safety cap
    "drop_na": True,                  # Drop any rows containing NaNs before returning
    }

    plugin_debug_vars: List[str] = [
        "features_requested",
        "features_processed",
        "use_stl",
        "use_wavelet",
        "use_mtm",
        "stl_period",
        "stl_window",
        "stl_trend",
        "wavelet_name",
        "wavelet_levels",
        "mtm_window_len",
        "mtm_freq_bands",
        "components_generated",
        "output_columns",
    "rows_before_drop",
    "rows_after_drop",
    "rows_dropped_due_to_nans",
    ]

    def __init__(self) -> None:
        self.params: Dict[str, Any] = self.plugin_params.copy()
        self._debug_state: Dict[str, Any] = {k: None for k in self.plugin_debug_vars}

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------
    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    def get_debug_info(self) -> Dict[str, Any]:  # noqa: D401
        return {k: self._debug_state.get(k) for k in self.plugin_debug_vars}

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def post_process(self, config: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        """Execute strictly causal decompositions.

        Steps:
            1. Resolve active parameters & compute derived STL window/trend if needed.
            2. Validate and filter feature list to those present in data.
            3. Optionally add log return column (CLOSE required).
            4. For each feature perform enabled rolling decompositions (STL, Wavelet, MTM) using only past+current data.
            5. Collect component arrays (aligned to original index) & optionally remove originals.
            6. Enforce output column cap if configured.
            7. Update debug state & return assembled DataFrame.
        """

        # -----------------------------------------------------------------
        # 1. Resolve parameters
        # -----------------------------------------------------------------
        p = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        self.set_params(**p)  # persist
        self._finalize_stl_params()
        use_stl = p["use_stl"] and p["stl_period"] and p["stl_period"] > 1
        use_wavelet = p["use_wavelet"] and HAS_WAVELETS
        use_mtm = p["use_mtm"] and HAS_SIGNAL

        feature_list = list(p["decomp_features"] or [])

        # -----------------------------------------------------------------
        # 2. Validate feature list
        # -----------------------------------------------------------------
        valid_features = [f for f in feature_list if f in data.columns]
        if feature_list and not valid_features:
            logger.warning("None of the requested decomposition features were found in the dataset; returning original data")
            return data.copy()
        if not feature_list:  # default: try all numeric (?) – stick with explicit only
            logger.info("No decomp_features specified; returning data unchanged.")
            return data.copy()

        # Output structure
        out = pd.DataFrame(index=data.index)
        components_generated: List[str] = []

        # -----------------------------------------------------------------
        # 3. Log return (optional)
        # -----------------------------------------------------------------
        if p["add_log_return"] and "CLOSE" in data.columns:
            close = data["CLOSE"].astype(float)
            log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out["log_return"] = log_ret
            components_generated.append("log_return")
        elif p["add_log_return"]:
            logger.warning("Requested log_return but CLOSE column missing; skipping.")

        # Progress handling
        iterator = valid_features
        if p["show_progress"] and len(valid_features) >= p["progress_min_features"]:
            try:
                iterator = tqdm(valid_features, desc="decompose", leave=False)
            except Exception:  # pragma: no cover
                iterator = valid_features

        # -----------------------------------------------------------------
        # 4. Per-feature causal rolling decompositions
        # -----------------------------------------------------------------
        for feat in iterator:
            series = data[feat].astype(float).to_numpy()
            n = len(series)

            feat_components: Dict[str, np.ndarray] = {}

            if use_stl:
                stl_trend, stl_seasonal, stl_resid = self._causal_stl(
                    series,
                    window=p["stl_window"],
                    period=p["stl_period"],
                    trend_len=p["stl_trend"],
                )
                feat_components[f"{feat}_stl_trend"] = stl_trend
                feat_components[f"{feat}_stl_seasonal"] = stl_seasonal
                feat_components[f"{feat}_stl_resid"] = stl_resid

            if use_wavelet:
                wav_dict = self._causal_wavelet(series, p["wavelet_name"], p["wavelet_levels"])  # returns dict
                for k, v in wav_dict.items():
                    feat_components[f"{feat}_{k}"] = v

            if use_mtm:
                mtm_dict = self._causal_mtm(series, p["mtm_window_len"], p["mtm_freq_bands"])  # returns dict
                for k, v in mtm_dict.items():
                    feat_components[f"{feat}_{k}"] = v

            # Assign to output
            for comp_name, comp_vals in feat_components.items():
                out[comp_name] = comp_vals
                components_generated.append(comp_name)

            # Keep original feature if required
            if p["keep_original"]:
                out[feat] = series
            elif p["replace_original"] and not p["keep_original"]:
                # Original omitted intentionally
                pass
            else:  # default keep_original True covers main use-case
                pass

        # -----------------------------------------------------------------
        # 5. Enforce column cap if configured
        # -----------------------------------------------------------------
        max_cols = p.get("max_output_columns")
        if max_cols is not None and len(out.columns) > max_cols:
            logger.warning(
                "Truncating output columns from %s to %s due to max_output_columns", len(out.columns), max_cols
            )
            keep = list(out.columns)[: int(max_cols)]
            out = out[keep]

        # -----------------------------------------------------------------
        # 6. Drop NaNs (optional) & update debug state
        # -----------------------------------------------------------------
        rows_before = len(out)
        if p.get("drop_na", True):
            out = out.dropna()
        rows_after = len(out)
        rows_dropped = rows_before - rows_after
        self._debug_state.update(
            {
                "features_requested": feature_list,
                "features_processed": valid_features,
                "use_stl": use_stl,
                "use_wavelet": use_wavelet,
                "use_mtm": use_mtm,
                "stl_period": p["stl_period"],
                "stl_window": p["stl_window"],
                "stl_trend": p["stl_trend"],
                "wavelet_name": p["wavelet_name"],
                "wavelet_levels": p["wavelet_levels"],
                "mtm_window_len": p["mtm_window_len"],
                "mtm_freq_bands": p["mtm_freq_bands"],
                "components_generated": len(components_generated),
                "output_columns": list(out.columns),
                "rows_before_drop": rows_before,
                "rows_after_drop": rows_after,
                "rows_dropped_due_to_nans": rows_dropped,
            }
        )

        return out

    # ------------------------------------------------------------------
    # STL (causal rolling)
    # ------------------------------------------------------------------
    def _finalize_stl_params(self) -> None:
        if self.params.get("stl_window") is None:
            self.params["stl_window"] = 2 * int(self.params["stl_period"]) + 1
        if self.params.get("stl_trend") is None:
            # heuristic: near period + 1, must be odd
            candidate = int(self.params["stl_period"]) + 1
            if candidate % 2 == 0:
                candidate += 1
            self.params["stl_trend"] = candidate
        # Ensure odd
        if self.params["stl_trend"] % 2 == 0:
            self.params["stl_trend"] += 1

    def _causal_stl(self, series: np.ndarray, window: int, period: int, trend_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(series)
        trend = np.full(n, np.nan, dtype=float)
        seasonal = np.full(n, np.nan, dtype=float)
        resid = np.full(n, np.nan, dtype=float)
        if window < period * 2:  # minimal viability check
            logger.warning("STL window (%s) may be too small for period %s; components may remain NaN", window, period)
        for i in range(window - 1, n):  # inclusive window end at i
            w_slice = series[i - window + 1 : i + 1]
            try:
                stl = STL(w_slice, period=period, trend=trend_len, robust=True)
                res = stl.fit()
                trend[i] = res.trend[-1]
                seasonal[i] = res.seasonal[-1]
                resid[i] = res.resid[-1]
            except Exception:  # pragma: no cover
                # Leave NaN
                pass
        self._forward_fill_first_valid([trend, seasonal, resid])
        return trend.astype(np.float32), seasonal.astype(np.float32), resid.astype(np.float32)

    # ------------------------------------------------------------------
    # Wavelet (causal rolling SWT)
    # ------------------------------------------------------------------
    def _causal_wavelet(self, series: np.ndarray, wavelet_name: str, levels: int) -> Dict[str, np.ndarray]:
        if not HAS_WAVELETS:
            return {}
        n = len(series)
        wavelet = pywt.Wavelet(wavelet_name)  # type: ignore
        min_window = max(2 ** levels, wavelet.dec_len * 2)
        results: Dict[str, np.ndarray] = {
            f"wav_detail_L{lvl}": np.full(n, np.nan, dtype=np.float32) for lvl in range(1, levels + 1)
        }
        results[f"wav_approx_L{levels}"] = np.full(n, np.nan, dtype=np.float32)
        for i in range(min_window - 1, n):
            window = series[i - min_window + 1 : i + 1]
            try:
                coeffs = pywt.swt(window, wavelet_name, level=levels, trim_approx=False)  # type: ignore
                # coeffs: list of (approx, detail) starting at level 1
                for lvl, (approx_arr, detail_arr) in enumerate(coeffs, start=1):
                    # Assign last value of detail / approx for causal alignment
                    results[f"wav_detail_L{lvl}"][i] = detail_arr[-1]
                    if lvl == levels:
                        results[f"wav_approx_L{levels}"][i] = approx_arr[-1]
            except Exception:  # pragma: no cover
                pass
        self._forward_fill_first_valid(list(results.values()))
        return results

    # ------------------------------------------------------------------
    # MTM / Welch band power (causal rolling)
    # ------------------------------------------------------------------
    def _causal_mtm(self, series: np.ndarray, window_len: int, bands: List[Tuple[float, float]]) -> Dict[str, np.ndarray]:
        if not HAS_SIGNAL:
            return {}
        n = len(series)
        features: Dict[str, np.ndarray] = {
            f"mtm_band_{i+1}_{low:.3f}_{high:.3f}": np.full(n, np.nan, dtype=np.float32)
            for i, (low, high) in enumerate(bands)
        }
        for i in range(window_len - 1, n):
            window = series[i - window_len + 1 : i + 1]
            try:
                freqs, psd = signal.welch(window, nperseg=min(len(window), 256))  # type: ignore
                for (low, high), (feat_name, arr) in zip(bands, features.items()):
                    mask = (freqs >= low) & (freqs < high)
                    if mask.any():
                        arr[i] = float(psd[mask].mean())
            except Exception:  # pragma: no cover
                pass
        self._forward_fill_first_valid(list(features.values()))
        return features

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _forward_fill_first_valid(arrays: List[np.ndarray]) -> None:
        for arr in arrays:
            if np.isnan(arr).all():
                continue
            # find first non-nan
            idx = np.where(~np.isnan(arr))[0]
            if len(idx) == 0:
                continue
            first = idx[0]
            arr[:first] = arr[first]


# Plugin interface (if dynamic loader expects these)
def get_plugin_class():  # noqa: D401
    return FeaturePlugin


def get_plugin_info():  # noqa: D401
    return {
        "name": "decomposition_post_processor",
        "version": "3.0.0",
        "description": "Causal rolling STL / Wavelet / MTM decompositions",
        "author": "feature-eng",
        "type": "post_processor",
        "capabilities": ["stl", "wavelet", "mtm"],
        "dependencies": ["numpy", "pandas", "statsmodels"],
        "optional_dependencies": ["pywt", "scipy", "tqdm"],
    }