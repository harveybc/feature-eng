#!/usr/bin/env python3
"""Seasonal Features Plugin

Generates calendar / intraday seasonal features from a timestamp column with optional
cyclic (sine/cosine) encoding for model-friendly representation while avoiding one-hot
expansion.

Responsibilities:
    1. Load input data from CSV (respecting max rows) OR use provided in-memory DataFrame.
    2. Locate and parse the configured date/time column (supports optional explicit format).
    3. Drop rows whose timestamps fail to parse (logged via warning).
    4. Derive raw seasonal components (day_of_week, day_of_month, hour_of_day).
    5. Optionally apply cyclic encoding (sin/cos) and remove raw component integers.
    6. Return a DataFrame containing the timestamp plus derived seasonal feature columns.
    7. Provide debug information (rows read, valid rows, feature count, source type, cyclic flag).

Notes:
    - Mirrors the unified step/comment style used across base_features, main, and pipeline.
    - The process method uses only the merged configuration passed at runtime.
      plugin_params solely supply defaults for the two-pass merging system.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SeasonalFeaturePlugin:  # Name expected by loader
    # ------------------------------------------------------------------
    # Default parameters (merged during second pass)
    # ------------------------------------------------------------------
    plugin_params: Dict[str, Any] = {
        # INPUT SOURCE --------------------------------------------------
        "seasonal_features_input_file": "tests/data/eurusd_hour_2005_2020_ohlc.csv",  # Optional CSV path
        "seasonal_features_max_rows": 1000000,  # Optional row cap
        # DATETIME HANDLING ---------------------------------------------
        "date_time_col": "date_time",  # Name of timestamp column
        "date_time_format": None,  # Optional explicit strftime format
        # FEATURE FLAGS -------------------------------------------------
        "generate_cyclic_features": True,  # If True produce sin/cos pairs
    }

    def __init__(self) -> None:
        self.params: Dict[str, Any] = self.plugin_params.copy()
        self._debug_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Configuration setter (called after merge)
    # ------------------------------------------------------------------
    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    # ------------------------------------------------------------------
    # Accessor for pipeline debug collection
    # ------------------------------------------------------------------
    def get_debug_info(self) -> Dict[str, Any]:
        return {**self._debug_state}

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def process(self, config: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> pd.DataFrame:  # noqa: D401
        """Generate seasonal features from either an input file or provided data.

        Steps:
            1. Resolve active parameters from merged config.
            2. Load data from CSV (if configured) or provided DataFrame.
            3. Parse timestamps (explicit format if provided, else generic).
            4. Derive base seasonal integers (day_of_week/day_of_month/hour_of_day).
            5. Optionally perform cyclic encoding and drop raw integer columns.
            6. Update debug info & return final seasonal feature DataFrame.
        """

        # -----------------------------------------------------------------
        # 1. Resolve parameters from merged configuration
        # -----------------------------------------------------------------
        params = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        input_file = params["seasonal_features_input_file"]
        max_rows = params["seasonal_features_max_rows"]
        date_time_col = params["date_time_col"]
        date_time_format = params["date_time_format"]
        generate_cyclic = params["generate_cyclic_features"]

        # -----------------------------------------------------------------
        # 2. Load data (file OR provided DataFrame)
        # -----------------------------------------------------------------
        if input_file and str(input_file).lower() != "none":
            try:
                df = pd.read_csv(input_file, nrows=max_rows)
                data_source = "file"
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to read seasonal features input file '{input_file}': {exc}") from exc
        else:
            if data is None:
                raise ValueError(
                    "seasonal_features_input_file is None and no 'data' provided to seasonal_features.process()."
                )
            df = data.copy()
            if max_rows is not None:
                df = df.head(max_rows)
            data_source = "data_param"

        row_count = len(df)

        if date_time_col not in df.columns:
            # Attempt a case-insensitive match
            lower_map = {c.lower(): c for c in df.columns}
            resolved = lower_map.get(date_time_col.lower())
            if resolved:
                date_time_col = resolved
            else:
                raise ValueError(f"Timestamp column '{date_time_col}' not found in data.")

        # -----------------------------------------------------------------
        # 3. Parse datetime (explicit format if provided)
        # -----------------------------------------------------------------
        if date_time_format:
            try:
                ts = pd.to_datetime(df[date_time_col], format=date_time_format, errors="coerce")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to parse '%s' with format '%s': %s; falling back to generic parsing.",
                    date_time_col,
                    date_time_format,
                    exc,
                )
                ts = pd.to_datetime(df[date_time_col], errors="coerce")
        else:
            ts = pd.to_datetime(df[date_time_col], errors="coerce")

        # -----------------------------------------------------------------
        # 4. Drop rows with invalid timestamps & init working frame
        # -----------------------------------------------------------------
        valid_mask = ts.notna()
        if not valid_mask.all():
            dropped = (~valid_mask).sum()
            logger.warning("Dropped %s rows with unparseable timestamps in seasonal_features", dropped)
        ts = ts[valid_mask]
        work_df = pd.DataFrame({date_time_col: ts})

        # -----------------------------------------------------------------
        # 5. Derive raw seasonal integer components
        # -----------------------------------------------------------------
        work_df["day_of_week"] = ts.dt.dayofweek  # 0=Mon
        work_df["day_of_month"] = ts.dt.day
        work_df["hour_of_day"] = ts.dt.hour

        # -----------------------------------------------------------------
        # 6. Optional cyclic encoding (replace raw ints with sin/cos pairs)
        # -----------------------------------------------------------------
        if generate_cyclic:
            def add_cyclic(src_col: str, period: int, prefix: str) -> None:
                angle = 2 * np.pi * work_df[src_col] / period
                work_df[f"{prefix}_sin"] = np.sin(angle)
                work_df[f"{prefix}_cos"] = np.cos(angle)

            add_cyclic("day_of_week", 7, "dow")
            add_cyclic("day_of_month", 31, "dom")  # 31 for stable periodic frame
            add_cyclic("hour_of_day", 24, "hod")

            # Remove raw integer columns to return only transformed features plus timestamp
            work_df = work_df.drop(columns=["day_of_week", "day_of_month", "hour_of_day"])

        # -----------------------------------------------------------------
        # 7. Finalize & debug tracking
        # -----------------------------------------------------------------
        self._debug_state.update(
            {
                "rows_read": row_count,
                "rows_valid": len(work_df),
                "features_exported": len(work_df.columns) - 1,  # exclude timestamp
                "data_source": data_source,
                "cyclic": bool(generate_cyclic),
            }
        )

        return work_df
