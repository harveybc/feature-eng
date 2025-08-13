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
        "aligner_prefix_columns": True,                 # Prefix columns with dataset key (avoid collisions)
        "aligner_sort_output": True,                    # Sort final output by datetime ascending
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

        # -----------------------------------------------------------------
        # 1. Resolve parameters
        # -----------------------------------------------------------------
        params = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        dt_col_standard = params["aligner_date_time_col"]
        dt_format = params["aligner_date_time_format"]
        max_rows = params["aligner_max_rows"]
        prefix_cols = params["aligner_prefix_columns"]
        sort_output = params["aligner_sort_output"]

        # Normalize datasets input to dict[str, DataFrame]
        if isinstance(datasets, list):
            ds_dict: Dict[str, pd.DataFrame] = {f"dataset_{i+1}": df for i, df in enumerate(datasets)}
        elif isinstance(datasets, dict):
            ds_dict = datasets.copy()
        else:  # pragma: no cover - defensive
            raise TypeError("'datasets' must be a list or dict of DataFrames")

        if not ds_dict:
            raise ValueError("No datasets provided to align()")

        per_dataset_row_counts: Dict[str, int] = {}
        dropped_invalid: Dict[str, int] = {}
        parsed_frames: Dict[str, pd.DataFrame] = {}

        # -----------------------------------------------------------------
        # 2. Detect & parse datetime column in each dataset
        # -----------------------------------------------------------------
        for name, df in ds_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Dataset '{name}' is not a pandas DataFrame")
            original_rows = len(df)
            per_dataset_row_counts[name] = original_rows
            if original_rows == 0:
                dropped_invalid[name] = 0
                continue

            # Resolve datetime column case-insensitively
            lower_map = {c.lower(): c for c in df.columns}
            dt_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            resolved = lower_map.get(dt_col_standard.lower())
            if resolved is None:
                # Fallback: choose first candidate containing date/time
                resolved = dt_candidates[0] if dt_candidates else None
            if resolved is None:
                raise ValueError(f"Could not locate a datetime-like column in dataset '{name}'")

            series = df[resolved]
            if dt_format:
                try:
                    parsed = pd.to_datetime(series, format=dt_format, errors='coerce')
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to parse datetime in '%s' with format '%s': %s; falling back to generic parsing.",
                        name,
                        dt_format,
                        exc,
                    )
                    parsed = pd.to_datetime(series, errors='coerce')
            else:
                parsed = pd.to_datetime(series, errors='coerce')

            valid_mask = parsed.notna()
            dropped = (~valid_mask).sum()
            dropped_invalid[name] = int(dropped)
            parsed_df = df.loc[valid_mask].copy()
            parsed_df[dt_col_standard] = parsed.loc[valid_mask].values

            # Drop duplicate timestamps keeping first (strict causality / uniqueness requirement)
            parsed_df = parsed_df.sort_values(dt_col_standard).drop_duplicates(subset=dt_col_standard, keep='first')

            parsed_frames[name] = parsed_df

        # -----------------------------------------------------------------
        # 3. Determine common intersection of timestamps
        # -----------------------------------------------------------------
        timestamp_sets: List[set] = [set(df[dt_col_standard].unique()) for df in parsed_frames.values() if len(df)]
        if not timestamp_sets:
            raise ValueError("All provided datasets are empty after datetime parsing")
        common_timestamps = set.intersection(*timestamp_sets)
        if not common_timestamps:
            raise ValueError("No shared timestamps across all datasets; cannot align")

        # Convert to sorted list
        common_index = sorted(common_timestamps)
        start_ts = common_index[0]
        end_ts = common_index[-1]

        # -----------------------------------------------------------------
        # 4. Slice each dataset to the common timestamps & rename columns
        # -----------------------------------------------------------------
        aligned_parts: List[pd.DataFrame] = []
        total_input_rows = sum(per_dataset_row_counts.values())
        for name, df in parsed_frames.items():
            aligned_df = df[df[dt_col_standard].isin(common_timestamps)].copy()
            # Ensure sorted
            aligned_df = aligned_df.sort_values(dt_col_standard)
            # Optionally prefix columns (except datetime)
            if prefix_cols:
                rename_map = {
                    c: f"{name}__{c}" for c in aligned_df.columns if c != dt_col_standard
                }
                aligned_df = aligned_df.rename(columns=rename_map)
            aligned_parts.append(aligned_df.set_index(dt_col_standard))

        # -----------------------------------------------------------------
        # 5. Concatenate features side-by-side on datetime index
        # -----------------------------------------------------------------
        merged = pd.concat(aligned_parts, axis=1, join='inner')
        merged.index.name = dt_col_standard
        if sort_output:
            merged = merged.sort_index()

        # Align to the exact common_index ordering
        merged = merged.reindex(common_index)
        merged.reset_index(inplace=True)

        # -----------------------------------------------------------------
        # 6. Apply max row cap if specified
        # -----------------------------------------------------------------
        if max_rows is not None:
            merged = merged.head(int(max_rows))

        # -----------------------------------------------------------------
        # 7. Finalize debug info
        # -----------------------------------------------------------------
        self._debug_state.update(
            {
                "input_datasets": len(ds_dict),
                "rows_input_total": total_input_rows,
                "rows_aligned": len(merged),
                "columns_exported": len(merged.columns) - 1,  # exclude datetime
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "dropped_empty_or_invalid": dropped_invalid,
                "per_dataset_row_counts": per_dataset_row_counts,
            }
        )

        return merged


# Backward compatibility alias (if framework expects a different class reference)
Plugin = AlignerPlugin
