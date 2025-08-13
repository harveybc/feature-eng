#!/usr/bin/env python3
"""Default Aligner Plugin

Merges multiple feature plugin output DataFrames on a common strictly shared datetime index.

Responsibilities:
    1. Accept a collection (dict or list) of feature DataFrames from previous feature plugins.
    2. Detect and normalize each dataset's datetime column (case-insensitive; supports explicit format).
    3. Determine a common intersection date range & exact shared timestamps across ALL datasets.
    4. Slice each dataset to that intersection (dropping rows outside or with missing timestamps).
    5. Optionally prefix non-datetime columns with the dataset key to avoid collisions.
    6. Concatenate columns side-by-side aligned perfectly row-by-row on the datetime.
    7. Enforce optional max row export limit (head truncation after alignment).
    8. Provide debug information (row counts per input, rows aligned, start/end, columns exported, drops).

Notes:
    - Only timestamps present in every dataset survive (strict inner alignment).
    - Datetime parsing uses provided format when given; otherwise generic pandas parsing.
    - Duplicate datetimes inside an input are de-duplicated keeping the first occurrence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Union
import re

import pandas as pd

logger = logging.getLogger(__name__)


class AlignerPlugin:
    """Dataset alignment & merge plugin."""

    # ------------------------------------------------------------------
    # Default (merge-able) parameters
    # ------------------------------------------------------------------
    plugin_params: Dict[str, Any] = {
        "aligner_date_time_col": "DATE_TIME",          # Standardized output datetime column name
        "aligner_date_time_format": None,               # Optional explicit datetime format (applied if provided)
        "aligner_max_rows": None,                       # Optional cap on exported aligned rows
        "aligner_prefix_columns": False,                # No prefixes by default; we'll canonicalize names instead
        "aligner_sort_output": True,                    # Sort final output by datetime ascending
        # Canonicalization & schema control
        "aligner_strip_prefixes": True,                 # Remove "Plugin__" prefixes if present
        "aligner_enforce_output_schema": True,          # Enforce final columns & ordering
        "aligner_strict_schema": True,                  # Drop extra columns not in schema
        "aligner_add_missing_columns": True,            # Add missing columns as NaN to match exact schema
        "aligner_output_columns": [
            "DATE_TIME",
            "OPEN","LOW","HIGH","CLOSE",
            "RSI","MACD","MACD_Histogram","MACD_Signal","EMA",
            "Stochastic_%K","Stochastic_%D",
            "ADX","DI+","DI-","ATR","CCI","WilliamsR","Momentum","ROC",
            "BC-BO","BH-BL","BH-BO","BO-BL",
            "S&P500_Close","vix_close",
            "CLOSE_15m_tick_1","CLOSE_15m_tick_2","CLOSE_15m_tick_3","CLOSE_15m_tick_4",
            "CLOSE_15m_tick_5","CLOSE_15m_tick_6","CLOSE_15m_tick_7","CLOSE_15m_tick_8",
            "CLOSE_30m_tick_1","CLOSE_30m_tick_2","CLOSE_30m_tick_3","CLOSE_30m_tick_4",
            "CLOSE_30m_tick_5","CLOSE_30m_tick_6","CLOSE_30m_tick_7","CLOSE_30m_tick_8",
            "day_of_month","hour_of_day","day_of_week",
        ],
    }

    plugin_debug_vars: List[str] = [
        "input_datasets",
        "rows_input_total",
        "rows_aligned",
        "columns_exported",
        "start_timestamp",
        "end_timestamp",
        "dropped_empty_or_invalid",
        "per_dataset_row_counts",
        "missing_output_columns",
        "extra_columns_dropped",
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

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    # ------------------------------------------------------------------
    # Core alignment
    # ------------------------------------------------------------------
    def align(self, config: Dict[str, Any], datasets: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]]) -> pd.DataFrame:  # noqa: D401
        """Align multiple feature DataFrames on a shared datetime index and merge columns.

        Parameters
        ----------
        config : dict
            Runtime configuration (merged before call) used to override plugin_params.
        datasets : dict[str, DataFrame] | list[DataFrame]
            Collection of feature DataFrames produced by feature plugins. If a list is provided,
            anonymous keys dataset_1 ... dataset_N are assigned.

        Returns
        -------
        DataFrame
            Aligned & merged feature dataset with a single datetime column and all feature columns.
        """

        # 1) Resolve parameters
        params = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        dt_col_standard = params["aligner_date_time_col"]
        dt_format = params["aligner_date_time_format"]
        max_rows = params["aligner_max_rows"]
        prefix_cols = params["aligner_prefix_columns"]
        sort_output = params["aligner_sort_output"]
        strip_prefixes = params.get("aligner_strip_prefixes", True)
        enforce_schema = params.get("aligner_enforce_output_schema", True)
        strict_schema = params.get("aligner_strict_schema", True)
        add_missing = params.get("aligner_add_missing_columns", True)
        desired_cols: List[str] = params.get("aligner_output_columns", [])

        # 2) Normalize datasets input to dict[str, DataFrame]
        if isinstance(datasets, list):
            ds_dict: Dict[str, pd.DataFrame] = {f"dataset_{i+1}": df for i, df in enumerate(datasets)}
        elif isinstance(datasets, dict):
            ds_dict = datasets.copy()
        else:
            raise TypeError("'datasets' must be a list or dict of DataFrames")

        if not ds_dict:
            raise ValueError("No datasets provided to align()")

        per_dataset_row_counts: Dict[str, int] = {}
        dropped_invalid: Dict[str, int] = {}
        parsed_frames: Dict[str, pd.DataFrame] = {}

        # 3) Detect & parse datetime column in each dataset
        for name, df in ds_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Dataset '{name}' is not a pandas DataFrame")
            original_rows = len(df)
            per_dataset_row_counts[name] = original_rows
            if original_rows == 0:
                dropped_invalid[name] = 0
                continue

            lower_map = {c.lower(): c for c in df.columns}
            dt_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            resolved = lower_map.get(dt_col_standard.lower())
            if resolved is None:
                resolved = dt_candidates[0] if dt_candidates else None
            if resolved is None:
                raise ValueError(f"Could not locate a datetime-like column in dataset '{name}'")

            series = df[resolved]
            if dt_format:
                try:
                    parsed = pd.to_datetime(series, format=dt_format, errors="coerce")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to parse datetime in '%s' with format '%s': %s; falling back to generic parsing.",
                        name,
                        dt_format,
                        exc,
                    )
                    parsed = pd.to_datetime(series, errors="coerce")
            else:
                parsed = pd.to_datetime(series, errors="coerce")

            valid_mask = parsed.notna()
            dropped_invalid[name] = int((~valid_mask).sum())
            parsed_df = df.loc[valid_mask].copy()
            parsed_df[dt_col_standard] = parsed.loc[valid_mask].values

            # drop other datetime-like columns
            drop_dt_like = [
                c for c in parsed_df.columns if c != dt_col_standard and ("date" in c.lower() or "time" in c.lower())
            ]
            if drop_dt_like:
                parsed_df = parsed_df.drop(columns=drop_dt_like, errors="ignore")

            parsed_df = parsed_df.sort_values(dt_col_standard).drop_duplicates(subset=dt_col_standard, keep="first")
            parsed_frames[name] = parsed_df

        # 4) Intersection of timestamps
        timestamp_sets: List[set] = [set(df[dt_col_standard].unique()) for df in parsed_frames.values() if len(df)]
        if not timestamp_sets:
            raise ValueError("All provided datasets are empty after datetime parsing")
        common_timestamps = set.intersection(*timestamp_sets)
        if not common_timestamps:
            raise ValueError("No shared timestamps across all datasets; cannot align")

        common_index = sorted(common_timestamps)
        start_ts = common_index[0]
        end_ts = common_index[-1]

        # 5) Slice each dataset to the common timestamps
        aligned_parts: List[pd.DataFrame] = []
        total_input_rows = sum(per_dataset_row_counts.values())
        for name, df in parsed_frames.items():
            aligned_df = df[df[dt_col_standard].isin(common_timestamps)].copy()
            aligned_df = aligned_df.sort_values(dt_col_standard)
            if prefix_cols:
                rename_map = {c: f"{name}__{c}" for c in aligned_df.columns if c != dt_col_standard}
                aligned_df = aligned_df.rename(columns=rename_map)
            aligned_parts.append(aligned_df.set_index(dt_col_standard))

        # 6) Concatenate on datetime index
        merged = pd.concat(aligned_parts, axis=1, join="inner")
        merged.index.name = dt_col_standard
        if sort_output:
            merged = merged.sort_index()
        merged = merged.reindex(common_index)
        merged.reset_index(inplace=True)

        # 7) Standardize column names
        def _strip_plugin_prefix(col: str) -> str:
            if strip_prefixes and "__" in col:
                return col.split("__", 1)[-1]
            return col

        def _map_to_canonical(col: str) -> str:
            if col == dt_col_standard:
                return "DATE_TIME"
            base = _strip_plugin_prefix(col)
            direct = {
                "OPEN": "OPEN",
                "HIGH": "HIGH",
                "LOW": "LOW",
                "CLOSE": "CLOSE",
                "BC-BO": "BC-BO",
                "BH-BL": "BH-BL",
                "BH-BO": "BH-BO",
                "BO-BL": "BO-BL",
                "S&P500_Close": "S&P500_Close",
                "SP500": "S&P500_Close",
                "S&P500": "S&P500_Close",
                "VIX": "vix_close",
                "VIX_Close": "vix_close",
                "vix_close": "vix_close",
                "day_of_month": "day_of_month",
                "hour_of_day": "hour_of_day",
                "day_of_week": "day_of_week",
            }
            if base in direct:
                return direct[base]

            upper = base.upper()
            if re.fullmatch(r"RSI(_\d+)?", upper):
                return "RSI"
            if re.fullmatch(r"EMA(_\d+)?", upper):
                return "EMA"
            if re.fullmatch(r"MACD(_\d+_?\d*)?", upper):
                return "MACD"
            if "MACD" in upper and "HIST" in upper:
                return "MACD_Histogram"
            if ("MACD" in upper and "SIGNAL" in upper) or re.fullmatch(r"MACD_SIGNAL(_\d+)?", upper):
                return "MACD_Signal"
            if re.fullmatch(r"(STOCH|STOCHASTIC)[_%]?K(_\d+)?", upper):
                return "Stochastic_%K"
            if re.fullmatch(r"(STOCH|STOCHASTIC)[_%]?D(_\d+)?", upper):
                return "Stochastic_%D"
            if re.fullmatch(r"ADX(_\d+)?", upper):
                return "ADX"
            if re.search(r"(PLUS_DI|DI\+|DI_PLUS|PDI)(_?\d+)?", upper):
                return "DI+"
            if re.search(r"(MINUS_DI|DI-|DI_MINUS|NDI)(_?\d+)?", upper):
                return "DI-"
            if re.fullmatch(r"ATR(_\d+)?", upper):
                return "ATR"
            if re.fullmatch(r"CCI(_\d+)?", upper):
                return "CCI"
            if re.fullmatch(r"(WILLR|WILLIAMSR)(_?\d+)?", upper):
                return "WilliamsR"
            if re.fullmatch(r"(MOM|MOMENTUM)(_?\d+)?", upper):
                return "Momentum"
            if re.fullmatch(r"ROC(_\d+)?", upper):
                return "ROC"

            m15 = re.fullmatch(r"M15_LAG_(\d+)", upper)
            if m15:
                return f"CLOSE_15m_tick_{int(m15.group(1))}"
            m30 = re.fullmatch(r"M30_LAG_(\d+)", upper)
            if m30:
                return f"CLOSE_30m_tick_{int(m30.group(1))}"

            return base

        merged = merged.rename(columns={c: _map_to_canonical(c) for c in merged.columns})

        # 8) Max rows
        if max_rows is not None:
            merged = merged.head(int(max_rows))

        # 9) Enforce final schema
        extra_cols = [c for c in merged.columns if desired_cols and c not in desired_cols]
        missing_cols = [c for c in desired_cols if c not in merged.columns] if desired_cols else []
        if enforce_schema and desired_cols:
            if add_missing:
                for mc in missing_cols:
                    merged[mc] = pd.NA
            extra_cols = [c for c in merged.columns if c not in desired_cols]
            if strict_schema and extra_cols:
                merged = merged.drop(columns=extra_cols, errors="ignore")
            keep_in_order = [c for c in desired_cols if c in merged.columns]
            merged = merged.loc[:, keep_in_order]

        # 10) Debug info
        self._debug_state.update(
            {
                "input_datasets": len(ds_dict),
                "rows_input_total": total_input_rows,
                "rows_aligned": len(merged),
                "columns_exported": len(merged.columns) - 1,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "dropped_empty_or_invalid": dropped_invalid,
                "per_dataset_row_counts": per_dataset_row_counts,
                "missing_output_columns": missing_cols if enforce_schema and desired_cols else [],
                "extra_columns_dropped": extra_cols if enforce_schema and strict_schema else [],
            }
        )

        return merged


# Backward compatibility alias (if framework expects a different class reference)
Plugin = AlignerPlugin
