#!/usr/bin/env python3
"""Fundamental Features Plugin (clean rewrite)

Generates aligned hourly SP500 & VIX features with optional daily->hourly expansion.
Adds robust, flexible datetime/target column resolution (case & punctuation insensitive
with synonyms) similar to the technical features plugin.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalFeaturePlugin:
    """Extract hourly SP500 & VIX features.

    Steps:
        1. Load SP500 & VIX CSVs.
        2. Resolve datetime / target columns flexibly.
        3. Parse timestamps with fallback strategies (format -> generic -> dayfirst).
        4. Trim to overlapping window.
        5. If use_daily_data: expand to 24 hourly rows per day (replicate or interpolate).
        6. Else: treat inputs as hourly and inner join on floored hour.
        7. Truncate to max rows if configured.
        8. Populate debug state.
    """

    plugin_params: Dict[str, Any] = {
        "sp500_input_file": "tests/data/sp_500_day_1927_2020_ohlc.csv",
        "vix_input_file": "tests/data/vix_day_1990_2024.csv",
        "sp500_date_time_col": "DATE_TIME",
        "vix_date_time_col": "DATE_TIME",
        "sp500_date_time_format": None,
        "vix_date_time_format": None,
        "sp500_target_col": "CLOSE",
        "vix_target_col": "CLOSE",
        "use_daily_data": True,
        "use_interpolation": False,
        "fundamental_features_max_rows": 1_000_000,
    # Fail-fast controls
    "fail_fast_on_invalid_timestamp": True,
    }

    plugin_debug_vars: List[str] = [
        "rows_read_sp500",
        "rows_read_vix",
        "rows_exported",
        "features_exported",
        "overlap_start",
        "overlap_end",
        "daily_input",
        "interpolation_used",
        "sp500_ts_dropped",
        "vix_ts_dropped",
    ]

    def __init__(self) -> None:
        self.params = self.plugin_params.copy()
        self._debug_state = {k: None for k in self.plugin_debug_vars}

    def set_params(self, **kwargs) -> None:  # noqa: D401
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    def get_debug_info(self) -> Dict[str, Any]:  # noqa: D401
        return {k: self._debug_state.get(k) for k in self.plugin_debug_vars}

    @staticmethod
    def _normalize(name: str) -> str:
        return ''.join(ch for ch in name.lower() if ch.isalnum())

    def _resolve_column(self, df: pd.DataFrame, desired: str, label: str, kind: str, extra_synonyms: Optional[List[str]] = None) -> str:
        lower_map = {c.lower(): c for c in df.columns}
        norm_map = {self._normalize(c): c for c in df.columns}
        synonyms = [desired, desired.lower(), desired.upper(), desired.replace('-', '_')]
        if extra_synonyms:
            synonyms.extend(extra_synonyms)
        norms = {self._normalize(s) for s in synonyms}
        for s in synonyms:
            if s.lower() in lower_map:
                return lower_map[s.lower()]
        for n in norms:
            if n in norm_map:
                return norm_map[n]
        raise ValueError(f"{label}: {kind} column '{desired}' not found. Available (first 15): {list(df.columns)[:15]}")

    def _parse_datetime(self, series: pd.Series, fmt: Optional[str], label: str) -> pd.Series:
        if fmt:
            try:
                ts = pd.to_datetime(series, format=fmt, errors='coerce')
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: explicit format failed (%s); fallback generic (%s)", label, fmt, exc)
                ts = pd.to_datetime(series, errors='coerce')
        else:
            ts = pd.to_datetime(series, errors='coerce')
            if ts.isna().mean() > 0.9:
                ts = pd.to_datetime(series, errors='coerce', dayfirst=True)
        return ts

    def process(self, config: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> pd.DataFrame:  # noqa: D401
        p = {**self.params, **{k: config.get(k, v) for k, v in self.params.items()}}
        sp500_file = p['sp500_input_file']
        vix_file = p['vix_input_file']
        try:
            sp_raw = pd.read_csv(sp500_file)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed reading SP500 file {sp500_file}: {exc}") from exc
        try:
            vix_raw = pd.read_csv(vix_file)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed reading VIX file {vix_file}: {exc}") from exc

        sp_dt_col = self._resolve_column(sp_raw, p['sp500_date_time_col'], 'SP500', 'datetime', ['date', 'datetime', 'date_time'])
        vix_dt_col = self._resolve_column(vix_raw, p['vix_date_time_col'], 'VIX', 'datetime', ['date', 'datetime', 'date_time'])
        sp_target_col = self._resolve_column(sp_raw, p['sp500_target_col'], 'SP500', 'target', ['close', 'adjclose'])
        vix_target_col = self._resolve_column(vix_raw, p['vix_target_col'], 'VIX', 'target', ['close', 'adjclose'])

        sp_ts = self._parse_datetime(sp_raw[sp_dt_col], p['sp500_date_time_format'], 'SP500')
        vix_ts = self._parse_datetime(vix_raw[vix_dt_col], p['vix_date_time_format'], 'VIX')
        sp_valid = sp_ts.notna(); vix_valid = vix_ts.notna()
        sp_dropped = int((~sp_valid).sum()); vix_dropped = int((~vix_valid).sum())
        if (sp_dropped or vix_dropped) and p.get('fail_fast_on_invalid_timestamp', True):
            # capture first invalid details
            if sp_dropped:
                bad_idx = int((~sp_valid).idxmax())
                raise RuntimeError(
                    f"fundamental_features: invalid SP500 timestamp at index {bad_idx} raw_value={sp_raw.loc[bad_idx, sp_dt_col]!r} row={sp_raw.loc[bad_idx].to_dict()}"
                )
            if vix_dropped:
                bad_idx = int((~vix_valid).idxmax())
                raise RuntimeError(
                    f"fundamental_features: invalid VIX timestamp at index {bad_idx} raw_value={vix_raw.loc[bad_idx, vix_dt_col]!r} row={vix_raw.loc[bad_idx].to_dict()}"
                )
        else:
            if sp_dropped: logger.warning("SP500: dropped %s invalid timestamp rows", sp_dropped)
            if vix_dropped: logger.warning("VIX: dropped %s invalid timestamp rows", vix_dropped)
        sp_df = pd.DataFrame({'date_time': sp_ts[sp_valid], 'SP500': sp_raw.loc[sp_valid, sp_target_col].astype(float)}).sort_values('date_time').reset_index(drop=True)
        vix_df = pd.DataFrame({'date_time': vix_ts[vix_valid], 'VIX': vix_raw.loc[vix_valid, vix_target_col].astype(float)}).sort_values('date_time').reset_index(drop=True)

        overlap_start = max(sp_df.date_time.min(), vix_df.date_time.min())
        overlap_end = min(sp_df.date_time.max(), vix_df.date_time.max())
        if overlap_start >= overlap_end:
            raise ValueError('No overlapping date/time range between SP500 and VIX series')
        sp_df = sp_df[(sp_df.date_time >= overlap_start) & (sp_df.date_time <= overlap_end)]
        vix_df = vix_df[(vix_df.date_time >= overlap_start) & (vix_df.date_time <= overlap_end)]

        use_daily = bool(p['use_daily_data']); use_interp = bool(p['use_interpolation'])
        if use_daily:
            sp_daily = sp_df.groupby(sp_df.date_time.dt.date).tail(1).reset_index(drop=True)
            vix_daily = vix_df.groupby(vix_df.date_time.dt.date).tail(1).reset_index(drop=True)
            common_days = sorted(set(sp_daily.date_time.dt.date) & set(vix_daily.date_time.dt.date))
            if not common_days:
                raise ValueError('No overlapping daily dates after trimming for SP500 & VIX')
            sp_daily = sp_daily[sp_daily.date_time.dt.date.isin(common_days)].reset_index(drop=True)
            vix_daily = vix_daily[vix_daily.date_time.dt.date.isin(common_days)].reset_index(drop=True)
            rows = []
            for idx, day in enumerate(common_days):
                sp_val = sp_daily.loc[sp_daily.date_time.dt.date == day, 'SP500'].iloc[0]
                vix_val = vix_daily.loc[vix_daily.date_time.dt.date == day, 'VIX'].iloc[0]
                if use_interp and idx > 0:
                    prev_sp = sp_daily.loc[sp_daily.date_time.dt.date == common_days[idx - 1], 'SP500'].iloc[0]
                    prev_vix = vix_daily.loc[vix_daily.date_time.dt.date == common_days[idx - 1], 'VIX'].iloc[0]
                else:
                    prev_sp = sp_val; prev_vix = vix_val
                base_day = pd.Timestamp(day).normalize()
                for hour in range(24):
                    ts = base_day + pd.Timedelta(hours=hour)
                    if use_interp and idx > 0:
                        frac = (hour + 1) / 24.0
                        sp_hour = prev_sp + (sp_val - prev_sp) * frac
                        vix_hour = prev_vix + (vix_val - prev_vix) * frac
                    else:
                        sp_hour = sp_val; vix_hour = vix_val
                    rows.append({'date_time': ts, 'SP500': sp_hour, 'VIX': vix_hour})
            merged = pd.DataFrame(rows)
        else:
            sp_df['date_time'] = sp_df.date_time.dt.floor('H')
            vix_df['date_time'] = vix_df.date_time.dt.floor('H')
            merged = pd.merge(sp_df, vix_df, on='date_time', how='inner')
        merged = merged.sort_values('date_time').reset_index(drop=True)

        max_rows = p['fundamental_features_max_rows']
        if max_rows and len(merged) > max_rows:
            merged = merged.head(int(max_rows))

        self._debug_state.update({
            'rows_read_sp500': len(sp_raw),
            'rows_read_vix': len(vix_raw),
            'rows_exported': len(merged),
            'features_exported': 2,
            'overlap_start': overlap_start.isoformat(),
            'overlap_end': overlap_end.isoformat(),
            'daily_input': use_daily,
            'interpolation_used': use_interp and use_daily,
            'sp500_ts_dropped': sp_dropped,
            'vix_ts_dropped': vix_dropped,
        })
        return merged
