#!/usr/bin/env python3
"""Fundamental Features Plugin

Generates aligned hourly fundamental features derived from two market series:
SP500 (index) and VIX (volatility). Always returns HOURLY rows even if inputs are
daily (expands per-hour either by replication or interpolation).

Responsibilities:
    1. Load SP500 & VIX CSV inputs (independent date/time + target columns & formats).
    2. Parse datetimes (using explicit formats when supplied) & normalize column cases.
    3. Align both series to their common overlapping time window (trim non-overlap).
    4. If use_daily_data=True, expand daily points to 24 hourly points per day:
         - Replicate target value each hour OR
         - Interpolate hourly values between prior day close and current day close (if use_interpolation).
    5. If use_daily_data=False, treat inputs as already hourly & inner join on hourly timestamps.
    6. Limit merged output to fundamental_features_max_rows if configured.
    7. Provide debug information (rows read per source, rows exported, overlap window, flags).

Notes:
    - Matches unified step/comment style of base_features.
    - process() relies exclusively on the merged runtime config.
      plugin_params only supply defaults for the two-pass merge system.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FundamentalFeaturePlugin:
    """Fundamental feature extraction plugin (SP500 + VIX hourly alignment)."""

    # ---------------------------------------------------------------------
    # Default (merge-able) plugin parameters
    # ---------------------------------------------------------------------
    plugin_params: Dict[str, Any] = {
        # INPUT FILES ----------------------------------------------------
        "sp500_input_file": "tests/data/sp_500_day_1927_2020_ohlc.csv",
        "vix_input_file": "tests/data/vix_day_1990_2024.csv",
        # DATETIME CONFIG ------------------------------------------------
        "sp500_date_time_col": "DATE_TIME",
        "vix_date_time_col": "DATE_TIME",
        "sp500_date_time_format": None,  # Optional explicit format
        "vix_date_time_format": None,
        # TARGET COLUMNS -------------------------------------------------
        "sp500_target_col": "CLOSE",
        "vix_target_col": "CLOSE",
        # CONTROL FLAGS --------------------------------------------------
        "use_daily_data": True,          # Treat input rows as daily values
        "use_interpolation": False,      # Interpolate hourly if daily
        "fundamental_features_max_rows": 1000000,
    }

    # Debug vars to expose via get_debug_info
    plugin_debug_vars: List[str] = [
        "rows_read_sp500",
        "rows_read_vix",
        "rows_exported",
        "features_exported",
        "overlap_start",
        "overlap_end",
        "daily_input",
        "interpolation_used",
    ]

    def __init__(self) -> None:
        self.params: Dict[str, Any] = self.plugin_params.copy()
        self._debug_state: Dict[str, Any] = {k: None for k in self.plugin_debug_vars}

    # ---------------------------------------------------------------------
    # Standard plugin interface methods
    # ---------------------------------------------------------------------
    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    def get_debug_info(self) -> Dict[str, Any]:  # noqa: D401
        return {k: self._debug_state.get(k) for k in self.plugin_debug_vars}

    # ---------------------------------------------------------------------
    # Core processing
    # ---------------------------------------------------------------------
    def process(self, config: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> pd.DataFrame:  # noqa: D401
        """Execute fundamental feature extraction producing hourly SP500 & VIX.

        Notes:
            The optional 'data' parameter is ignored (two-file design) but accepted
            for interface uniformity.

        Steps:
            1. Resolve active parameters from merged config.
            2. Load SP500 & VIX CSV inputs.
            3. Parse & normalize datetime + select target columns.
            4. Align to common overlapping date/time window.
            5. Convert daily inputs to hourly (replicate or interpolate) OR merge hourly directly.
            6. Limit output rows if fundamental_features_max_rows is set.
            7. Update debug info & return hourly DataFrame.
        """

        # -----------------------------------------------------------------
        # 1. Resolve parameters from merged configuration
        # -----------------------------------------------------------------
        p = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        sp500_file = p["sp500_input_file"]
        vix_file = p["vix_input_file"]
        sp500_dt_col = p["sp500_date_time_col"]
        vix_dt_col = p["vix_date_time_col"]
        sp500_dt_fmt = p["sp500_date_time_format"]
        vix_dt_fmt = p["vix_date_time_format"]
        sp500_target_col = p["sp500_target_col"]
        vix_target_col = p["vix_target_col"]
        use_daily = p["use_daily_data"]
        use_interp = p["use_interpolation"]
        max_rows = p["fundamental_features_max_rows"]

        # -----------------------------------------------------------------
        # 2. Load CSV inputs
        # -----------------------------------------------------------------
        def _read_csv(path: str, label: str) -> pd.DataFrame:
            try:
                return pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to read {label} file '{path}': {exc}") from exc

        sp_df_raw = _read_csv(sp500_file, "SP500")
        vix_df_raw = _read_csv(vix_file, "VIX")

        rows_sp = len(sp_df_raw)
        rows_vix = len(vix_df_raw)

        # -----------------------------------------------------------------
        # 3. Parse & normalize datetime + select target columns
        # -----------------------------------------------------------------
        def _prepare(df: pd.DataFrame, dt_col: str, fmt: Optional[str], target_col: str, label: str) -> pd.DataFrame:
            # Case-insensitive column resolution
            lower_map = {c.lower(): c for c in df.columns}
            resolved_dt = lower_map.get(dt_col.lower())
            resolved_target = lower_map.get(target_col.lower())
            if resolved_dt is None:
                raise ValueError(f"{label}: date/time column '{dt_col}' not found")
            if resolved_target is None:
                raise ValueError(f"{label}: target column '{target_col}' not found")

            if fmt:
                try:
                    ts = pd.to_datetime(df[resolved_dt], format=fmt, errors="coerce")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "%s: failed to parse datetime with format '%s': %s; falling back to generic parse",
                        label,
                        fmt,
                        exc,
                    )
                    ts = pd.to_datetime(df[resolved_dt], errors="coerce")
            else:
                ts = pd.to_datetime(df[resolved_dt], errors="coerce")

            valid_mask = ts.notna()
            if not valid_mask.all():
                dropped = (~valid_mask).sum()
                logger.warning("%s: dropped %s rows with invalid timestamps", label, dropped)
            df2 = pd.DataFrame({"date_time": ts[valid_mask], label: df.loc[valid_mask, resolved_target]})
            df2 = df2.sort_values("date_time").reset_index(drop=True)
            return df2

        sp_df = _prepare(sp_df_raw, sp500_dt_col, sp500_dt_fmt, sp500_target_col, "SP500")
        vix_df = _prepare(vix_df_raw, vix_dt_col, vix_dt_fmt, vix_target_col, "VIX")

        # -----------------------------------------------------------------
        # 4. Align to overlapping window (trim non-overlap)
        # -----------------------------------------------------------------
        overlap_start = max(sp_df["date_time"].min(), vix_df["date_time"].min())
        overlap_end = min(sp_df["date_time"].max(), vix_df["date_time"].max())
        if overlap_start >= overlap_end:
            raise ValueError("No overlapping date/time range between SP500 and VIX series")

        sp_df = sp_df[(sp_df["date_time"] >= overlap_start) & (sp_df["date_time"] <= overlap_end)]
        vix_df = vix_df[(vix_df["date_time"] >= overlap_start) & (vix_df["date_time"] <= overlap_end)]

        # -----------------------------------------------------------------
        # 5. Convert to hourly data
        # -----------------------------------------------------------------
        if use_daily:
            # Collapse to daily on date component; keep last value per day (already sorted)
            sp_daily = sp_df.groupby(sp_df["date_time"].dt.date).tail(1).copy()
            vix_daily = vix_df.groupby(vix_df["date_time"].dt.date).tail(1).copy()

            # Re-align daily sets in case trimming created mismatched days
            common_days = sorted(set(sp_daily["date_time"].dt.date) & set(vix_daily["date_time"].dt.date))
            sp_daily = sp_daily[sp_daily["date_time"].dt.date.isin(common_days)]
            vix_daily = vix_daily[vix_daily["date_time"].dt.date.isin(common_days)]

            if not common_days:
                raise ValueError("No overlapping daily dates after alignment for SP500 and VIX")

            # Build hourly expansion
            hourly_rows: List[Dict[str, Any]] = []
            sp_daily = sp_daily.reset_index(drop=True)
            vix_daily = vix_daily.reset_index(drop=True)

            for idx, day in enumerate(common_days):
                sp_val = sp_daily.loc[sp_daily["date_time"].dt.date == day, "SP500"].iloc[0]
                vix_val = vix_daily.loc[vix_daily["date_time"].dt.date == day, "VIX"].iloc[0]

                # For interpolation we need previous day's values
                if use_interp and idx > 0:
                    prev_sp = sp_daily.loc[sp_daily["date_time"].dt.date == common_days[idx - 1], "SP500"].iloc[0]
                    prev_vix = vix_daily.loc[vix_daily["date_time"].dt.date == common_days[idx - 1], "VIX"].iloc[0]
                else:
                    prev_sp = sp_val
                    prev_vix = vix_val

                for hour in range(24):
                    ts = pd.Timestamp.combine(pd.to_datetime(day).date(), pd.Timestamp(hour=hour, minute=0).time())
                    if use_interp and idx > 0:
                        # Linear interpolation from previous day's close (prev) to current day's close (curr)
                        frac = (hour + 1) / 24.0  # Assumption: hour 23 ~ full day progression
                        sp_hour = prev_sp + (sp_val - prev_sp) * frac
                        vix_hour = prev_vix + (vix_val - prev_vix) * frac
                    else:
                        sp_hour = sp_val
                        vix_hour = vix_val
                    hourly_rows.append({"date_time": ts, "SP500": sp_hour, "VIX": vix_hour})

            merged = pd.DataFrame(hourly_rows).sort_values("date_time").reset_index(drop=True)
        else:
            # Treat inputs as already hourly; floor to hour and inner join
            sp_df["date_time"] = sp_df["date_time"].dt.floor("H")
            vix_df["date_time"] = vix_df["date_time"].dt.floor("H")
            merged = pd.merge(sp_df, vix_df, on="date_time", how="inner", suffixes=("_sp", "_vix"))
            # After merge, columns: date_time, SP500, VIX (due to identical labels) – ensure only needed
            merged = merged[["date_time", "SP500", "VIX"]].sort_values("date_time").reset_index(drop=True)

        # -----------------------------------------------------------------
        # 6. Limit output rows
        # -----------------------------------------------------------------
        if max_rows is not None and len(merged) > max_rows:
            merged = merged.head(max_rows)

        # -----------------------------------------------------------------
        # 7. Finalize & debug tracking
        # -----------------------------------------------------------------
        self._debug_state.update(
            {
                "rows_read_sp500": rows_sp,
                "rows_read_vix": rows_vix,
                "rows_exported": len(merged),
                "features_exported": 2,  # SP500 + VIX
                "overlap_start": overlap_start.isoformat(),
                "overlap_end": overlap_end.isoformat(),
                "daily_input": bool(use_daily),
                "interpolation_used": bool(use_interp and use_daily),
            }
        )

        return merged
