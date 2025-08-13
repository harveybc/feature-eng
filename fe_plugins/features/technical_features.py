#!/usr/bin/env python3
"""Technical Features Plugin

Computes a configurable set of technical indicators from OHLC price data.

Responsibilities:
    1. Load input OHLC data from CSV (respecting max rows) OR use provided in-memory DataFrame.
    2. Parse / standardize the date/time column (supports optional explicit format) and map OHLC columns case-insensitively.
    3. Calculate the selected technical indicators using configured parameters (all defaults in plugin_params).
    4. Aggregate indicator outputs into a single feature DataFrame (retaining the timestamp column only once).
    5. Record indicator list, parameter snapshot, counts, data source in debug info for perfect replicability.
    6. Return the resulting feature DataFrame.

Notes:
    - Mirrors the unified step/comment style used across other feature plugins (base, seasonal, fundamental).
    - The process method only uses the merged runtime configuration passed in; plugin_params provide default values
      for the two-pass configuration merge in main.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalFeaturePlugin:  # Consistent plugin class name
    """Technical indicator feature plugin."""

    # ------------------------------------------------------------------
    # Default (merge-able) plugin parameters
    # ------------------------------------------------------------------
    plugin_params: Dict[str, Any] = {
        # INPUT SOURCE -------------------------------------------------
        "tech_features_input_file": "tests/data/eurusd_hour_2005_2020_ohlc.csv",
        "tech_features_max_rows": 1000000,
        # DATETIME / COLUMN CONFIG ------------------------------------
        "date_time_col": "DATE_TIME",
        "date_time_format": None,
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
    # PROGRESS -----------------------------------------------------
    "show_progress": True,
    "progress_min_rows": 5000,
        # INDICATOR SELECTION -----------------------------------------
        "indicators": [
            "rsi",
            "macd",
            "ema",
            "stoch",
            "adx",
            "atr",
            "cci",
            "bbands",
            "williams",
            "momentum",
            "roc",
        ],
        # PER-INDICATOR PARAMETERS ------------------------------------
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ema_period": 20,
        "stoch_k_period": 14,
        "stoch_d_period": 3,
        "stoch_smooth": 3,
        "adx_period": 14,
        "atr_period": 14,
        "cci_period": 20,
        "bbands_period": 20,
        "bbands_std": 2.0,
        "williams_period": 14,
        "momentum_period": 4,
        "roc_period": 12,
    }

    plugin_debug_vars: List[str] = [
        "rows_read",
        "rows_exported",
        "features_exported",
        "indicators_used",
        "indicator_params",
        "data_source",
    ]

    def __init__(self) -> None:
        self.params: Dict[str, Any] = self.plugin_params.copy()
        self._debug_state: Dict[str, Any] = {k: None for k in self.plugin_debug_vars}

    # ------------------------------------------------------------------
    # Standard plugin interface methods
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
    def process(self, config: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> pd.DataFrame:  # noqa: D401
        """Execute technical indicator feature generation.

        Steps:
            1. Resolve active parameters from merged config.
            2. Load dataset (CSV or provided DataFrame) & parse datetime.
            3. Normalize / map OHLC columns.
            4. Compute requested technical indicators.
            5. Assemble final DataFrame (timestamp + indicator columns).
            6. Update debug info & return feature set.
        """

        # -----------------------------------------------------------------
        # 1. Resolve parameters from merged configuration
        # -----------------------------------------------------------------
        params = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        input_file = params["tech_features_input_file"]
        max_rows = params["tech_features_max_rows"]
        dt_col = params["date_time_col"]
        dt_fmt = params["date_time_format"]
        o_col = params["open_col"]
        h_col = params["high_col"]
        l_col = params["low_col"]
        c_col = params["close_col"]
        indicators = params["indicators"]

        indicator_param_keys = [
            "rsi_period",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "ema_period",
            "stoch_k_period",
            "stoch_d_period",
            "stoch_smooth",
            "adx_period",
            "atr_period",
            "cci_period",
            "bbands_period",
            "bbands_std",
            "williams_period",
            "momentum_period",
            "roc_period",
        ]
        indicator_params_used = {k: params[k] for k in indicator_param_keys if k in params}

        # -----------------------------------------------------------------
        # 2. Load data (file OR provided DataFrame) & parse datetime
        # -----------------------------------------------------------------
        if input_file and input_file.lower() != "none":
            try:
                df = pd.read_csv(input_file, nrows=max_rows)
                data_source = "file"
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to read technical features input file '{input_file}': {exc}") from exc
        else:
            if data is None:
                raise ValueError(
                    "tech_features_input_file is None and no 'data' parameter provided to process(); cannot proceed."
                )
            df = data.copy()
            if max_rows is not None:
                df = df.head(max_rows)
            data_source = "data_param"

        rows_read = len(df)

        lower_map = {c.lower(): c for c in df.columns}
        resolved_dt = lower_map.get(dt_col.lower())
        resolved_o = lower_map.get(o_col.lower())
        resolved_h = lower_map.get(h_col.lower())
        resolved_l = lower_map.get(l_col.lower())
        resolved_c = lower_map.get(c_col.lower())
        missing = [name for name, val in [
            (dt_col, resolved_dt),
            (o_col, resolved_o),
            (h_col, resolved_h),
            (l_col, resolved_l),
            (c_col, resolved_c),
        ] if val is None]
        if missing:
            raise ValueError(f"Missing required columns in technical features input: {missing}")

        if dt_fmt:
            try:
                ts = pd.to_datetime(df[resolved_dt], format=dt_fmt, errors="coerce")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to parse '%s' with format '%s': %s; falling back to generic parsing.",
                    dt_col,
                    dt_fmt,
                    exc,
                )
                ts = pd.to_datetime(df[resolved_dt], errors="coerce")
        else:
            ts = pd.to_datetime(df[resolved_dt], errors="coerce")

        valid_mask = ts.notna()
        if not valid_mask.all():
            dropped = (~valid_mask).sum()
            logger.warning("Dropped %s rows with invalid timestamps in technical_features", dropped)
        df = df.loc[valid_mask].reset_index(drop=True)
        ts = ts[valid_mask].reset_index(drop=True)

        opens = df[resolved_o].astype(float).reset_index(drop=True)
        highs = df[resolved_h].astype(float).reset_index(drop=True)
        lows = df[resolved_l].astype(float).reset_index(drop=True)
        closes = df[resolved_c].astype(float).reset_index(drop=True)

        out = pd.DataFrame({dt_col: ts})

        # -----------------------------------------------------------------
        # 3. Helper functions
        # -----------------------------------------------------------------
        def ema(series: pd.Series, period: int) -> pd.Series:
            return series.ewm(span=period, adjust=False).mean()

        # -----------------------------------------------------------------
        # 4. Compute requested indicators STRICTLY CAUSALLY (loop per tick)
        # -----------------------------------------------------------------
        n = len(closes)
        use_progress = bool(params.get("show_progress", True)) and n >= int(params.get("progress_min_rows", 5000))
        try:
            if use_progress:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(range(n), desc="technical_indicators", mininterval=0.5)
            else:
                iterator = range(n)
        except Exception:  # pragma: no cover
            iterator = range(n)

        # Pre-allocate storage lists for each indicator
        want = set(ind.lower() for ind in indicators)
        data_store: Dict[str, List[Optional[float]]] = {}
        def init_slot(name: str):
            if name not in data_store:
                data_store[name] = [None] * n

        if "rsi" in want:
            init_slot("RSI")
        if "macd" in want:
            init_slot("MACD")
            init_slot("MACD_SIGNAL")
            init_slot("MACD_HIST")
        if "ema" in want:
            init_slot("EMA")
        if "stoch" in want:
            init_slot("STOCH_K")
            init_slot("STOCH_D")
        if "adx" in want:
            init_slot("ADX")
        if "atr" in want:
            init_slot("ATR")
        if "cci" in want:
            init_slot("CCI")
        if "bbands" in want:
            init_slot("BB_MID")
            init_slot("BB_UP")
            init_slot("BB_LOW")
            init_slot("BB_WIDTH")
        if "williams" in want:
            init_slot("WILLR")
        if "momentum" in want:
            init_slot("MOM")
        if "roc" in want:
            init_slot("ROC")

        # State variables for recursive indicators
        # EMA & MACD
        ema_period = params["ema_period"]
        ema_alpha = 2 / (ema_period + 1)
        ema_value = None

        macd_fast = params["macd_fast"]
        macd_slow = params["macd_slow"]
        macd_signal_p = params["macd_signal"]
        macd_alpha_fast = 2 / (macd_fast + 1)
        macd_alpha_slow = 2 / (macd_slow + 1)
        macd_alpha_signal = 2 / (macd_signal_p + 1)
        macd_fast_val = None
        macd_slow_val = None
        macd_signal_val = None

        # RSI state
        rsi_period = params["rsi_period"]
        gains_window: List[float] = []
        losses_window: List[float] = []
        prev_close_val = None

        # ATR / ADX state (Wilder smoothing)
        adx_period = params["adx_period"]
        atr_period = params["atr_period"]
        atr_val = None
        tr14 = None  # smoothed true range
        plus_dm14 = None
        minus_dm14 = None
        adx_list: List[Optional[float]] = []  # store for intermediate to compute ADX

        # Stochastic state (need rolling windows)
        stoch_k_period = params["stoch_k_period"]
        stoch_d_period = params["stoch_d_period"]
        stoch_smooth = params["stoch_smooth"]
        k_values: List[float] = []  # smoothed %K history to derive %D

        # CCI state
        cci_period = params["cci_period"]

        # Bollinger Bands state
        bb_period = params["bbands_period"]
        bb_std_mult = params["bbands_std"]

        # Williams %R state
        will_p = params["williams_period"]

        # Momentum & ROC
        mom_p = params["momentum_period"]
        roc_p = params["roc_period"]

        for i in iterator:  # per-tick strictly causal updates
            o = opens.iloc[i]
            h = highs.iloc[i]
            l = lows.iloc[i]
            c = closes.iloc[i]

            # --- EMA ---------------------------------------------------
            if "ema" in want:
                if ema_value is None:
                    ema_value = c
                else:
                    ema_value = ema_alpha * c + (1 - ema_alpha) * ema_value
                data_store["EMA"][i] = ema_value

            # --- MACD --------------------------------------------------
            if "macd" in want:
                if macd_fast_val is None:
                    macd_fast_val = c
                else:
                    macd_fast_val = macd_alpha_fast * c + (1 - macd_alpha_fast) * macd_fast_val
                if macd_slow_val is None:
                    macd_slow_val = c
                else:
                    macd_slow_val = macd_alpha_slow * c + (1 - macd_alpha_slow) * macd_slow_val
                macd_line = macd_fast_val - macd_slow_val
                if macd_signal_val is None:
                    macd_signal_val = macd_line
                else:
                    macd_signal_val = macd_alpha_signal * macd_line + (1 - macd_alpha_signal) * macd_signal_val
                macd_hist = macd_line - macd_signal_val
                data_store["MACD"][i] = macd_line
                data_store["MACD_SIGNAL"][i] = macd_signal_val
                data_store["MACD_HIST"][i] = macd_hist

            # --- RSI ---------------------------------------------------
            if "rsi" in want:
                if prev_close_val is not None:
                    change = c - prev_close_val
                    gains_window.append(max(change, 0))
                    losses_window.append(abs(min(change, 0)))
                    if len(gains_window) > rsi_period:
                        gains_window.pop(0)
                        losses_window.pop(0)
                if len(gains_window) == rsi_period:
                    avg_gain = sum(gains_window) / rsi_period
                    avg_loss = sum(losses_window) / rsi_period
                    if avg_loss == 0:
                        rsi_val = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi_val = 100 - (100 / (1 + rs))
                    data_store["RSI"][i] = rsi_val
                prev_close_val = c

            # --- ATR & ADX --------------------------------------------
            if "atr" in want or "adx" in want:
                if i == 0:
                    tr = h - l
                    plus_dm = 0.0
                    minus_dm = 0.0
                else:
                    prev_h = highs.iloc[i - 1]
                    prev_l = lows.iloc[i - 1]
                    prev_c = closes.iloc[i - 1]
                    move_up = h - prev_h
                    move_down = prev_l - l
                    plus_dm = move_up if (move_up > move_down and move_up > 0) else 0.0
                    minus_dm = move_down if (move_down > move_up and move_down > 0) else 0.0
                    tr = max(
                        h - l,
                        abs(h - prev_c),
                        abs(l - prev_c),
                    )
                # Wilder smoothing
                if atr_val is None:
                    atr_val = tr
                    tr14 = tr
                    plus_dm14 = plus_dm
                    minus_dm14 = minus_dm
                else:
                    atr_val = (atr_val * (atr_period - 1) + tr) / atr_period
                    tr14 = (tr14 * (adx_period - 1) + tr) / adx_period
                    plus_dm14 = (plus_dm14 * (adx_period - 1) + plus_dm) / adx_period
                    minus_dm14 = (minus_dm14 * (adx_period - 1) + minus_dm) / adx_period
                if "atr" in want:
                    data_store["ATR"][i] = atr_val
                if "adx" in want and i >= adx_period:
                    plus_di = 100 * (plus_dm14 / tr14 if tr14 else 0)
                    minus_di = 100 * (minus_dm14 / tr14 if tr14 else 0)
                    denom = plus_di + minus_di
                    dx_val = 0 if denom == 0 else (100 * abs(plus_di - minus_di) / denom)
                    adx_list.append(dx_val)
                    # Average of first adx_period DX values forms initial ADX, then Wilder smoothing
                    if len(adx_list) == adx_period:
                        adx_current = sum(adx_list) / adx_period
                        data_store["ADX"][i] = adx_current
                    elif len(adx_list) > adx_period:
                        prev_adx = data_store["ADX"][i - 1]
                        if prev_adx is None:
                            prev_adx = sum(adx_list[-adx_period:]) / adx_period
                        adx_current = (prev_adx * (adx_period - 1) + dx_val) / adx_period
                        data_store["ADX"][i] = adx_current

            # --- Stochastic (raw then smoothed) -----------------------
            if "stoch" in want:
                if i + 1 >= stoch_k_period:
                    window_slice = slice(i - stoch_k_period + 1, i + 1)
                    ll = lows.iloc[window_slice].min()
                    hh = highs.iloc[window_slice].max()
                    if hh != ll:
                        k_val_raw = 100 * (c - ll) / (hh - ll)
                    else:
                        k_val_raw = 0.0
                    # Smooth %K
                    k_values.append(k_val_raw)
                    if len(k_values) < stoch_smooth:
                        k_smoothed = None
                    else:
                        k_smoothed = sum(k_values[-stoch_smooth:]) / stoch_smooth
                    if k_smoothed is not None:
                        data_store["STOCH_K"][i] = k_smoothed
                        # %D
                        recent_k_for_d = [v for v in k_values if v is not None]
                        if len(recent_k_for_d) >= stoch_smooth + stoch_d_period - 1:
                            k_for_d = [
                                data_store["STOCH_K"][j]
                                for j in range(i - stoch_d_period + 1, i + 1)
                                if j >= 0 and data_store["STOCH_K"][j] is not None
                            ]
                            if len(k_for_d) == stoch_d_period:
                                data_store["STOCH_D"][i] = sum(k_for_d) / stoch_d_period

            # --- CCI ---------------------------------------------------
            if "cci" in want:
                if i + 1 >= cci_period:
                    window_slice = slice(i - cci_period + 1, i + 1)
                    tp_window = (highs.iloc[window_slice] + lows.iloc[window_slice] + closes.iloc[window_slice]) / 3.0
                    sma_tp = tp_window.mean()
                    mad = (tp_window - sma_tp).abs().mean()
                    cci_val = (tp_window.iloc[-1] - sma_tp) / (0.015 * mad) if mad != 0 else 0.0
                    data_store["CCI"][i] = cci_val

            # --- Bollinger Bands --------------------------------------
            if "bbands" in want:
                if i + 1 >= bb_period:
                    window_slice = slice(i - bb_period + 1, i + 1)
                    close_window = closes.iloc[window_slice]
                    mid = close_window.mean()
                    std_v = close_window.std(ddof=0)
                    up = mid + bb_std_mult * std_v
                    low_v = mid - bb_std_mult * std_v
                    data_store["BB_MID"][i] = mid
                    data_store["BB_UP"][i] = up
                    data_store["BB_LOW"][i] = low_v
                    data_store["BB_WIDTH"][i] = up - low_v

            # --- Williams %R -----------------------------------------
            if "williams" in want:
                if i + 1 >= will_p:
                    window_slice = slice(i - will_p + 1, i + 1)
                    hh = highs.iloc[window_slice].max()
                    ll = lows.iloc[window_slice].min()
                    denom = (hh - ll)
                    willr = -100 * (hh - c) / denom if denom != 0 else 0.0
                    data_store["WILLR"][i] = willr

            # --- Momentum ---------------------------------------------
            if "momentum" in want and i >= mom_p:
                data_store["MOM"][i] = c - closes.iloc[i - mom_p]

            # --- ROC --------------------------------------------------
            if "roc" in want and i >= roc_p:
                prev_val = closes.iloc[i - roc_p]
                data_store["ROC"][i] = ((c / prev_val) - 1) * 100.0 if prev_val != 0 else 0.0

        # After loop populate out DataFrame
        for col_key, values in data_store.items():
            # Use indicator-specific suffixing like earlier vectorized version
            if col_key == "RSI":
                out[f"RSI_{rsi_period}"] = values
            elif col_key.startswith("MACD"):
                if col_key == "MACD":
                    out[f"MACD_{macd_fast}_{macd_slow}"] = values
                elif col_key == "MACD_SIGNAL":
                    out[f"MACD_SIGNAL_{macd_signal_p}"] = values
                elif col_key == "MACD_HIST":
                    out[f"MACD_HIST_{macd_fast}_{macd_slow}_{macd_signal_p}"] = values
            elif col_key == "EMA":
                out[f"EMA_{ema_period}"] = values
            elif col_key == "STOCH_K":
                out[f"STOCH_K_{stoch_k_period}"] = values
            elif col_key == "STOCH_D":
                out[f"STOCH_D_{stoch_d_period}"] = values
            elif col_key == "ADX":
                out[f"ADX_{adx_period}"] = values
            elif col_key == "ATR":
                out[f"ATR_{atr_period}"] = values
            elif col_key == "CCI":
                out[f"CCI_{cci_period}"] = values
            elif col_key.startswith("BB_"):
                suffix_map = {"BB_MID": "MID", "BB_UP": "UP", "BB_LOW": "LOW", "BB_WIDTH": "WIDTH"}
                base = col_key
                out[f"BB_{suffix_map[base]}_{bb_period}_{int(bb_std_mult)}"] = values
            elif col_key == "WILLR":
                out[f"WILLR_{will_p}"] = values
            elif col_key == "MOM":
                out[f"MOM_{mom_p}"] = values
            elif col_key == "ROC":
                out[f"ROC_{roc_p}"] = values

        # -----------------------------------------------------------------
        # 5. Finalize output frame
        # -----------------------------------------------------------------
        feature_df = out

        # -----------------------------------------------------------------
        # 6. Finalize & debug tracking
        # -----------------------------------------------------------------
        self._debug_state.update(
            {
                "rows_read": rows_read,
                "rows_exported": len(feature_df),
                "features_exported": len(feature_df.columns) - 1,
                "indicators_used": indicators,
                "indicator_params": indicator_params_used,
                "data_source": data_source,
            }
        )

        return feature_df


