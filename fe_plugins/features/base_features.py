#!/usr/bin/env python3
"""Base Features Plugin

Generates foundational OHLC-based features with optional candlestick differentials.

Responsibilities:
    1. Load input data from CSV (respecting max rows) OR use provided in-memory DataFrame.
    2. Normalize and map column names (case-insensitive) to configured OHLC identifiers.
    3. Optionally compute candlestick differential features (BC-BO, BH-BL, BH-BO, BO-BL).
    4. Optionally drop OPEN/HIGH/LOW columns when close_only is enabled.
    5. Return a DataFrame containing the base feature set (including DATE_TIME if present).
    6. Provide debug information (rows read/exported, feature counts, source type).

Notes:
    - Uses the unified step/comment style applied across main.py and pipeline.
    - The process method operates solely with the merged runtime configuration passed in.
      plugin_params only supplies default values for the merging system.
"""


from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseFeaturePlugin:
    """Base feature extraction plugin (OHLC + optional candlestick differentials)."""

    # ---------------------------------------------------------------------
    # Default (merge-able) plugin parameters
    # ---------------------------------------------------------------------
    plugin_params: Dict[str, Any] = {
        "base_features_input_file": "tests/data/eurusd_hour_2005_2020_ohlc.csv",  # Default path (can be overridden)
        "base_features_max_rows": 1000000,
        "date_time_col": "DATE_TIME",
        "date_time_format": None,  # Optional explicit datetime format for parsing
        # Datetime parsing controls
        "date_time_dayfirst_fallback": True,
        "date_time_additional_formats": ["%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"],
        "date_time_fail_fast": True,
        # Flexible resolution of datetime column name (case/underscore/punctuation-insensitive)
        "date_time_synonyms": [
            "DATE_TIME",
            "DATETIME",
            "date_time",
            "datetime",
            "timestamp",
            "time",
            "date",
            "date time",
            "date/time",
        ],
        "open_col": "OPEN",
        "high_col": "HIGH",
        "low_col": "LOW",
        "close_col": "CLOSE",
        "use_candlestick": True,
        "close_only": False,
    }

    # Debug vars to expose via get_debug_info
    plugin_debug_vars: List[str] = [
        "rows_read",
        "features_read",
        "rows_exported",
        "features_exported",
        "data_source",
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
    def process(self, config: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Execute base feature extraction.

        Steps:
            1. Resolve active parameters from merged config.
            2. Load data from CSV (if configured) or use provided DataFrame.
            3. Normalize column names and map to configured OHLC schema.
            4. Select/export core columns (respecting close_only flag).
            5. Optionally compute candlestick differential features.
            6. Assemble final DataFrame & update debug information.
        """

        # -----------------------------------------------------------------
        # 1. Resolve parameters from merged configuration
        # -----------------------------------------------------------------
        params = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        input_file = params["base_features_input_file"]
        max_rows = params["base_features_max_rows"]
        date_time_col = params["date_time_col"]
        date_time_format = params["date_time_format"]
        open_col = params["open_col"]
        high_col = params["high_col"]
        low_col = params["low_col"]
        close_col = params["close_col"]
        use_candlestick = params["use_candlestick"]
        close_only = params["close_only"]

        # -----------------------------------------------------------------
        # 2. Load data (file OR provided DataFrame)
        # -----------------------------------------------------------------
        if input_file and input_file.lower() != "none":
            try:
                df = pd.read_csv(input_file, nrows=max_rows)
                data_source = "file"
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to read base features input file '{input_file}': {exc}") from exc
        else:
            if data is None:
                raise ValueError(
                    "base_features_input_file is None and no 'data' parameter provided to process(); cannot proceed."
                )
            df = data.copy()
            if max_rows is not None:
                df = df.head(max_rows)
            data_source = "data_param"

        original_feature_count = len(df.columns)
        original_row_count = len(df)

        # -----------------------------------------------------------------
        # 3. Normalize column names (case-insensitive mapping)
        # -----------------------------------------------------------------
        lower_map = {c.lower(): c for c in df.columns}

        def _norm(name: str) -> str:
            return "".join(ch for ch in name.lower() if ch.isalnum())

        norm_map = {_norm(c): c for c in df.columns}

        def resolve(col_name: str) -> Optional[str]:
            return lower_map.get(col_name.lower())

        # Resolve DATE_TIME flexibly (handles 'datetime', 'date_time', 'timestamp', etc.)
        dt_resolved = resolve(date_time_col)
        if dt_resolved is None:
            # Try synonyms with normalization
            synonyms: List[str] = [date_time_col] + (params.get("date_time_synonyms", []) or [])
            for syn in synonyms:
                cand = norm_map.get(_norm(syn))
                if cand is not None:
                    dt_resolved = cand
                    break
        if dt_resolved is None:
            # Fallback: any column containing 'date' or 'time'
            candidates = [c for c in df.columns if ("date" in c.lower() or "time" in c.lower())]
            if candidates:
                dt_resolved = candidates[0]

        mapped_cols = {
            "date_time": dt_resolved,
            "open": resolve(open_col),
            "high": resolve(high_col),
            "low": resolve(low_col),
            "close": resolve(close_col),
        }

        missing_required = [k for k, v in mapped_cols.items() if k != "date_time" and v is None]
        if missing_required:
            raise ValueError(f"Missing required OHLC columns in data: {missing_required}")

        # -----------------------------------------------------------------
        # 4. Select core columns (respect close_only flag)
        # -----------------------------------------------------------------
        selected_frames = []
        if mapped_cols["date_time"]:
            raw_dt = df[mapped_cols["date_time"]].copy()
            # Always attempt to parse to datetime dtype
            if date_time_format is not None:
                try:
                    dt_series = pd.to_datetime(raw_dt, format=date_time_format, errors="coerce")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to parse date_time with format '%s': %s; falling back to generic parsing",
                        date_time_format,
                        exc,
                    )
                    dt_series = pd.to_datetime(raw_dt, errors="coerce")
            else:
                dt_series = pd.to_datetime(raw_dt, errors="coerce")

            # Adaptive dayfirst if it improves valid count
            if params.get("date_time_dayfirst_fallback", True) and dt_series.isna().any():
                alt = pd.to_datetime(raw_dt, errors="coerce", dayfirst=True)
                if alt.notna().sum() > dt_series.notna().sum():
                    dt_series = alt

            # Try additional explicit formats to maximize valid parses
            if dt_series.isna().any():
                for fmt in params.get("date_time_additional_formats", []) or []:
                    alt = pd.to_datetime(raw_dt, format=fmt, errors="coerce")
                    if alt.notna().sum() > dt_series.notna().sum():
                        dt_series = alt
                    if not dt_series.isna().any():
                        break

            # Fail-fast on first invalid timestamp
            if params.get("date_time_fail_fast", True) and dt_series.isna().any():
                bad_idx = int(dt_series[dt_series.isna()].index[0])
                raise RuntimeError(
                    f"base_features: invalid timestamp at index {bad_idx} raw_value={raw_dt.iloc[bad_idx]!r} row={df.iloc[bad_idx].to_dict()}"
                )

            # Always standardize the output column name to params['date_time_col'] (e.g., 'DATE_TIME')
            selected_frames.append(dt_series.to_frame(name=date_time_col))
        else:
            # No datetime-like column found; fail early with context to help debugging
            available_cols = list(df.columns)
            raise ValueError(
                "BaseFeaturePlugin: Could not resolve a datetime column. Looked for '%s' and synonyms %s. Available columns: %s"
                % (date_time_col, params.get("date_time_synonyms", []), available_cols)
            )

        if close_only:
            selected_frames.append(df[[mapped_cols["close"]]].rename(columns={mapped_cols["close"]: close_col}))
        else:
            ohlc_map = {
                mapped_cols["open"]: open_col,
                mapped_cols["high"]: high_col,
                mapped_cols["low"]: low_col,
                mapped_cols["close"]: close_col,
            }
            selected_frames.append(df[list(ohlc_map.keys())].rename(columns=ohlc_map))

        base_df = pd.concat(selected_frames, axis=1)

        # -----------------------------------------------------------------
        # 5. Candlestick differential features (optional)
        # -----------------------------------------------------------------
        if use_candlestick:
            if not close_only:  # All OHLC available
                base_df["BC-BO"] = base_df[close_col] - base_df[open_col]
                base_df["BH-BL"] = base_df[high_col] - base_df[low_col]
                base_df["BH-BO"] = base_df[high_col] - base_df[open_col]
                base_df["BO-BL"] = base_df[open_col] - base_df[low_col]
            else:  # Only CLOSE kept; cannot compute all candlesticks, skip with warning
                logger.warning(
                    "Candlestick features requested but open/high/low removed (close_only=True); skipping candlestick generation."
                )

        # -----------------------------------------------------------------
        # 6. Finalize & debug tracking
        # -----------------------------------------------------------------
        self._debug_state.update(
            {
                "rows_read": original_row_count,
                "features_read": original_feature_count,
                "rows_exported": len(base_df),
                "features_exported": len(base_df.columns),
                "data_source": data_source,
            }
        )

        return base_df
