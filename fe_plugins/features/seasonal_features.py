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
        "date_time_col": "date_time",  # Preferred timestamp column name
        "date_time_format": None,  # Optional explicit strftime format
        "date_time_fail_fast": True,  # Raise immediately if timestamp column missing
        "date_time_fail_fast_invalid": True,  # Raise immediately on first invalid timestamp
        "date_time_additional_synonyms": ["datetime", "timestamp", "time"],  # Extra names to try
        "date_time_dayfirst_fallback": True,  # Try dayfirst if >90% NA
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
            5. Optionally perform cyclic encoding (sin/cos) while keeping raw integer columns.
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

        # Flexible timestamp column resolution (case + normalization + synonyms)
        def _normalize(s: str) -> str:
            return ''.join(ch for ch in s.lower() if ch.isalnum())
        lower_map = {c.lower(): c for c in df.columns}
        norm_map = {_normalize(c): c for c in df.columns}
        synonyms = [date_time_col, *self.params.get("date_time_additional_synonyms", [])]
        resolved_col = None
        for cand in synonyms:
            # direct lower
            if cand.lower() in lower_map:
                resolved_col = lower_map[cand.lower()]
                break
            # normalized
            n = _normalize(cand)
            if n in norm_map:
                resolved_col = norm_map[n]
                break
        if resolved_col is None:
            msg = (
                f"seasonal_features: timestamp column '{date_time_col}' not found. "
                f"Available: {list(df.columns)[:20]}"
            )
            if params.get("date_time_fail_fast", True):
                raise RuntimeError(msg)
            else:
                logger.error(msg)
                raise RuntimeError(msg)  # still abort – seasonal requires timestamp
        date_time_col = resolved_col

        # -----------------------------------------------------------------
        # 3. Parse datetime (explicit format if provided)
        # -----------------------------------------------------------------
        raw_ts = df[date_time_col]
        if date_time_format:
            try:
                ts = pd.to_datetime(raw_ts, format=date_time_format, errors="coerce")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed explicit parse for '%s' (%s); fallback generic: %s", date_time_col, date_time_format, exc
                )
                ts = pd.to_datetime(raw_ts, errors="coerce")
        else:
            ts = pd.to_datetime(raw_ts, errors="coerce")

        # Adaptive dayfirst attempt: if any invalids and improvement possible choose best
        if params.get("date_time_dayfirst_fallback", True) and ts.isna().any():
            alt = pd.to_datetime(raw_ts, errors="coerce", dayfirst=True)
            base_valid = ts.notna().sum()
            alt_valid = alt.notna().sum()
            if alt_valid > base_valid:
                logger.info(
                    "seasonal_features: switching to dayfirst=True parse (valid %s -> %s, improvement %d)",
                    base_valid,
                    alt_valid,
                    alt_valid - base_valid,
                )
                ts = alt

        # -----------------------------------------------------------------
        # 4. Drop rows with invalid timestamps & init working frame
        # -----------------------------------------------------------------
        valid_mask = ts.notna()
        if not valid_mask.all():
            invalid_mask = ~valid_mask
            invalid_count = int(invalid_mask.sum())
            first_bad = int(invalid_mask[invalid_mask].index[0])
            raw_value = raw_ts.iloc[first_bad]
            row_ctx = df.iloc[first_bad].to_dict()
            msg = (
                "seasonal_features: invalid timestamp encountered "
                f"count={invalid_count} first_index={first_bad} raw_value={raw_value!r} row={row_ctx}"
            )
            if params.get("date_time_fail_fast_invalid", True):
                raise RuntimeError(msg)
            logger.warning(msg)
        ts_valid = ts[valid_mask]
        work_df = pd.DataFrame({date_time_col: ts_valid})

        # -----------------------------------------------------------------
        # 5. Derive raw seasonal integer components
        # -----------------------------------------------------------------
        work_df["day_of_week"] = ts_valid.dt.dayofweek  # 0=Mon
        work_df["day_of_month"] = ts_valid.dt.day
        work_df["hour_of_day"] = ts_valid.dt.hour

        # -----------------------------------------------------------------
        # 6. Optional cyclic encoding (add sin/cos pairs; keep raw ints)
        # -----------------------------------------------------------------
        if generate_cyclic:
            def add_cyclic(src_col: str, period: int, prefix: str) -> None:
                angle = 2 * np.pi * work_df[src_col] / period
                work_df[f"{prefix}_sin"] = np.sin(angle)
                work_df[f"{prefix}_cos"] = np.cos(angle)

            add_cyclic("day_of_week", 7, "dow")
            add_cyclic("day_of_month", 31, "dom")  # 31 for stable periodic frame
            add_cyclic("hour_of_day", 24, "hod")
            # Keep raw integer columns as canonical fields expected by aligner

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
