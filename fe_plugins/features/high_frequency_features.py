#!/usr/bin/env python3
"""High Frequency Features Plugin

Constructs strictly causal high-frequency lag features from subordinate 15-minute and 30-minute
datasets aligned to an hourly backbone timeline.

Responsibilities:
    1. Load three datasets (hourly backbone, 15m, 30m) from CSV OR from an in-memory dict provided to process().
    2. Parse & standardize datetime columns independently (with optional explicit per-source formats).
    3. Keep ONLY the hourly datetime column from the hourly dataset (no numeric features taken from it).
    4. For every hourly timestamp, collect the last N (default 8) <= timestamp 15m closes and last N 30m closes
       strictly using historical (and current timestamp) data only (never future rows).
    5. Produce 16 high-frequency lag feature columns: 8 for 15m, 8 for 30m (padding with NaN if insufficient history).
    6. Provide debug information (row counts, features exported, data source mode) for reproducibility.
    7. Return the resulting DataFrame: [HOURLY_DATETIME_COL + lag feature columns].

Notes:
    - Follows the unified step/comment style used by other plugins (e.g., base_features).
    - If file paths are set to "none"/None, the process() data argument MUST be a dict containing keys
      'hourly', 'm15', 'm30' with DataFrames.
    - Target columns (value columns) for 15m & 30m default to CLOSE (case-insensitive mapping handled).
    - Assumption: The phrase "previous 125min ticks" in the request refers to 8 prior 15m ticks (8 * 15 = 120m);
      implemented as last 8 15m rows up to the hourly timestamp (inclusive). Adjust 'high_freq_num_lags_15m'
      if a different depth is required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeaturePlugin:  # Consistent naming pattern
    """High-frequency lag feature construction plugin."""

    # ------------------------------------------------------------------
    # Default (merge-able) plugin parameters
    # ------------------------------------------------------------------
    plugin_params: Dict[str, Any] = {
        # Input file paths (set to "none" or None to force data dict usage)
        "high_freq_hourly_file": "tests/data/eurusd_hour_2005_2020_ohlc.csv",
        "high_freq_15m_file": "tests/data/EURUSD-2000-2020-15m.csv",
        "high_freq_30m_file": None,
    # Automatic synthesis: when 30m file missing/None, derive 30m by taking every 2nd 15m row
    "high_freq_generate_30m_from_15m": True,
        # Optional per-source explicit datetime format strings
        "high_freq_hourly_date_time_format": None,
        "high_freq_15m_date_time_format": None,
        "high_freq_30m_date_time_format": None,
        # Column names (case-insensitive resolution)
        "high_freq_hourly_date_time_col": "DATE_TIME",
        "high_freq_15m_date_time_col": "DATE_TIME",
        "high_freq_30m_date_time_col": "DATE_TIME",
        # Target (value) column names for subordinate granularities
        "high_freq_15m_target_col": "CLOSE",
        "high_freq_30m_target_col": "CLOSE",
        # Lag depths
        "high_freq_num_lags_15m": 8,
        "high_freq_num_lags_30m": 8,
        # Row limits
        "high_freq_max_rows_hourly": 1000000,
        "high_freq_max_rows_15m": 1000000,
        "high_freq_max_rows_30m": 1000000,
    }

    plugin_debug_vars: List[str] = [
        "rows_hourly",
        "rows_15m",
        "rows_30m",
        "rows_exported",
        "features_exported",
        "data_source_mode",
        "lags_15m",
        "lags_30m",
    "rows_dropped_due_to_nans",
    "synthetic_30m",
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
    def process(self, config: Dict[str, Any], data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:  # noqa: D401
        """Generate high-frequency lag features.

        Steps:
            1. Resolve parameters from merged configuration.
            2. Load three datasets (hourly backbone + 15m + 30m) OR consume provided data dict.
            3. Parse datetime columns (with optional formats), drop invalid rows & sort ascending.
            4. Build hourly output frame retaining only the hourly datetime column.
            5. For each hourly timestamp, collect strictly historical (<= timestamp) last N 15m and last N 30m closes.
            6. Assemble lag feature columns (padding with NaN when insufficient history).
            7. Update debug info and return the feature DataFrame.
        """

        # -----------------------------------------------------------------
        # 1. Resolve parameters
        # -----------------------------------------------------------------
        p = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        file_hourly = p["high_freq_hourly_file"]
        file_15m = p["high_freq_15m_file"]
        file_30m = p["high_freq_30m_file"]
        synth_enabled = bool(p.get("high_freq_generate_30m_from_15m", True))
        fmt_hourly = p["high_freq_hourly_date_time_format"]
        fmt_15m = p["high_freq_15m_date_time_format"]
        fmt_30m = p["high_freq_30m_date_time_format"]
        col_hourly_dt = p["high_freq_hourly_date_time_col"]
        col_15m_dt = p["high_freq_15m_date_time_col"]
        col_30m_dt = p["high_freq_30m_date_time_col"]
        col_15m_target = p["high_freq_15m_target_col"]
        col_30m_target = p["high_freq_30m_target_col"]
        lags_15m = int(p["high_freq_num_lags_15m"])
        lags_30m = int(p["high_freq_num_lags_30m"])
        max_h = p["high_freq_max_rows_hourly"]
        max_15 = p["high_freq_max_rows_15m"]
        max_30 = p["high_freq_max_rows_30m"]
        need_synth_30m = (file_30m is None or str(file_30m).lower() == "none") and synth_enabled
        # We'll use files if at least hourly or 15m file is provided (we can synthesize 30m if missing)
        use_files = any([
            file_hourly is not None and str(file_hourly).lower() != "none",
            file_15m is not None and str(file_15m).lower() != "none",
            (not need_synth_30m) and file_30m is not None and str(file_30m).lower() != "none",
        ])

        # -----------------------------------------------------------------
        # 2. Load datasets
        # -----------------------------------------------------------------
        if use_files:
            try:
                hourly_df = pd.read_csv(file_hourly, usecols=[col_hourly_dt], nrows=max_h)
                df_15m = pd.read_csv(file_15m, nrows=max_15)
                if need_synth_30m:
                    # Synthetic 30m series: PURE SAMPLING of every 2nd 15m bar (no averaging/aggregation)
                    # This preserves exact timestamp alignment at 30m cadence without mixing prices.
                    df_30m = df_15m.iloc[::2].copy().reset_index(drop=True)
                    data_source_mode = "files+synthetic30m"
                else:
                    df_30m = pd.read_csv(file_30m, nrows=max_30)
                    data_source_mode = "files"
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to load one or more high frequency input files: {exc}") from exc
        else:
            if data is None:
                raise ValueError(
                    "All high frequency file paths are None/'none' and no data dict provided to process()."
                )
            if need_synth_30m and "m15" not in data:
                raise ValueError("Synthetic 30m generation requested but 15m dataset not provided in data dict.")
            if "hourly" not in data or "m15" not in data:
                raise ValueError("High frequency data dict missing required 'hourly' or 'm15' dataset.")
            hourly_df = data["hourly"][ [ self._resolve_ci(data["hourly"], col_hourly_dt) ] ]
            df_15m = data["m15"].copy()
            if need_synth_30m or "m30" not in data:
                # Synthetic 30m series: sampling every second 15m observation (no aggregation)
                df_30m = df_15m.iloc[::2].copy().reset_index(drop=True)
                data_source_mode = "data_param+synthetic30m"
            else:
                df_30m = data["m30"].copy()
                data_source_mode = "data_param"
            if max_h is not None:
                hourly_df = hourly_df.head(max_h)
            if max_15 is not None:
                df_15m = df_15m.head(max_15)
            if max_30 is not None:
                df_30m = df_30m.head(max_30)

        rows_hourly_raw = len(hourly_df)
        rows_15m_raw = len(df_15m)
        rows_30m_raw = len(df_30m)

        # -----------------------------------------------------------------
        # 3. Parse datetime columns & normalize case-insensitive mappings
        # -----------------------------------------------------------------
        hourly_dt_col_res = self._resolve_ci(hourly_df, col_hourly_dt)
        m15_dt_col_res = self._resolve_ci(df_15m, col_15m_dt)
        m30_dt_col_res = self._resolve_ci(df_30m, col_30m_dt)
        m15_target_col_res = self._resolve_ci(df_15m, col_15m_target)
        m30_target_col_res = self._resolve_ci(df_30m, col_30m_target)

        if hourly_dt_col_res is None or m15_dt_col_res is None or m30_dt_col_res is None:
            raise ValueError("Failed to resolve required datetime columns for high frequency processing.")
        if m15_target_col_res is None or m30_target_col_res is None:
            raise ValueError("Failed to resolve target (value) columns for 15m or 30m datasets.")

        hourly_df[hourly_dt_col_res] = self._parse_dt(hourly_df[hourly_dt_col_res], fmt_hourly)
        df_15m[m15_dt_col_res] = self._parse_dt(df_15m[m15_dt_col_res], fmt_15m)
        df_30m[m30_dt_col_res] = self._parse_dt(df_30m[m30_dt_col_res], fmt_30m)

        # Drop invalid rows (NaT) and sort
        hourly_df = hourly_df[hourly_df[hourly_dt_col_res].notna()].sort_values(hourly_dt_col_res).reset_index(drop=True)
        df_15m = df_15m[df_15m[m15_dt_col_res].notna()].sort_values(m15_dt_col_res).reset_index(drop=True)
        df_30m = df_30m[df_30m[m30_dt_col_res].notna()].sort_values(m30_dt_col_res).reset_index(drop=True)

        # -----------------------------------------------------------------
        # 4. Build hourly output base DataFrame
        # -----------------------------------------------------------------
        out = pd.DataFrame({col_hourly_dt: hourly_df[hourly_dt_col_res]})

        # Pre-extract value Series for speed
        m15_times = df_15m[m15_dt_col_res].to_numpy()
        m15_vals = df_15m[m15_target_col_res].astype(float).to_numpy()
        m30_times = df_30m[m30_dt_col_res].to_numpy()
        m30_vals = df_30m[m30_target_col_res].astype(float).to_numpy()

        # -----------------------------------------------------------------
        # 5. Collect causal lag windows per hourly timestamp
        # -----------------------------------------------------------------
        from bisect import bisect_right

        m15_features: List[List[Optional[float]]] = []  # each inner list size = lags_15m
        m30_features: List[List[Optional[float]]] = []  # each inner list size = lags_30m

        for ts in out[col_hourly_dt].to_numpy():
            # 15m
            pos_15 = bisect_right(m15_times, ts)  # index AFTER last <= ts
            window_15 = m15_vals[max(0, pos_15 - lags_15m):pos_15]
            # pad on left if fewer than lags
            if len(window_15) < lags_15m:
                pad = [float('nan')] * (lags_15m - len(window_15))
                window_15_list = pad + window_15.tolist()
            else:
                window_15_list = window_15.tolist()
            m15_features.append(window_15_list)

            # 30m
            pos_30 = bisect_right(m30_times, ts)
            window_30 = m30_vals[max(0, pos_30 - lags_30m):pos_30]
            if len(window_30) < lags_30m:
                pad = [float('nan')] * (lags_30m - len(window_30))
                window_30_list = pad + window_30.tolist()
            else:
                window_30_list = window_30.tolist()
            m30_features.append(window_30_list)

        # -----------------------------------------------------------------
        # 6. Assemble lag feature columns
        # -----------------------------------------------------------------
        # Column naming convention: M15_LAG_1 .. M15_LAG_N (1 = oldest in window, N = most recent <= timestamp)
        for idx in range(lags_15m):
            out[f"M15_LAG_{idx + 1}"] = [row[idx] for row in m15_features]
        for idx in range(lags_30m):
            out[f"M30_LAG_{idx + 1}"] = [row[idx] for row in m30_features]

        # -----------------------------------------------------------------
        # 7. Drop rows containing any NaN (remove early rows lacking full history)
        # -----------------------------------------------------------------
        pre_drop_rows = len(out)
        out = out.dropna(how="any").reset_index(drop=True)
        rows_after_drop = len(out)

        # -----------------------------------------------------------------
        # 8. Finalize debug info
        # -----------------------------------------------------------------
        self._debug_state.update(
            {
                "rows_hourly": rows_hourly_raw,
                "rows_15m": rows_15m_raw,
                "rows_30m": rows_30m_raw,
                "rows_exported": rows_after_drop,
                "features_exported": len(out.columns) - 1,
                "data_source_mode": data_source_mode,
                "lags_15m": lags_15m,
                "lags_30m": lags_30m,
                "rows_dropped_due_to_nans": pre_drop_rows - rows_after_drop,
                "synthetic_30m": bool(need_synth_30m or data_source_mode.endswith("synthetic30m")),
            }
        )

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_ci(df: pd.DataFrame, name: str) -> Optional[str]:
        lower_map = {c.lower(): c for c in df.columns}
        return lower_map.get(name.lower())

    @staticmethod
    def _parse_dt(series: pd.Series, fmt: Optional[str]) -> pd.Series:
        if fmt:
            try:
                return pd.to_datetime(series, format=fmt, errors="coerce")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse datetime with format '%s': %s; falling back to generic parsing", fmt, exc)
                return pd.to_datetime(series, errors="coerce")
        return pd.to_datetime(series, errors="coerce")

# Backward compatibility alias if loader expects 'Plugin'
Plugin = FeaturePlugin
