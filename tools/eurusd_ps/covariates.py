"""Batch 002 covariates: Yahoo daily, FRED daily market series, FX cross pairs.

Availability rules (declared, conservative, never earlier than the source can
have published):
* Yahoo daily bar of local trading date D: Close only (Adj Close is revised by
  later dividends -> not point-in-time); available at (D+1) 00:00 UTC, which is
  after every listed exchange's close on D.
* FRED daily of observation date D: available at (D+2) 00:00 UTC (H.15-type
  series are published the next business day in the US afternoon); the lake
  copy is the current vintage (realtime_start 2026-05-01) -- revisions of these
  market-observed series are rare but not excluded; declared NO_VINTAGE.
* FX cross pair hourly bars: same measured-clock conversion as EURUSD; available
  at the UTC bar end.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import contract as C
from . import sources as S
from .asof import asof_last
from .features import _meta
from .inventory import FRED_LIC, HISTDATA_LIC, YAHOO_LIC


def _read_cut(path: str, date_col: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    return d


def yahoo_series(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["Date", "Close"])
    local_date = pd.to_datetime(d["Date"].astype(str).str[:10])
    out = pd.DataFrame({"date": local_date, "close": d["Close"].astype(float)})
    out["avail_utc"] = (out["date"] + pd.Timedelta(days=1)).dt.tz_localize("UTC")
    out = out[out["avail_utc"] < C.READ_END]          # nothing available at/after READ_END is kept
    return out.sort_values("avail_utc").reset_index(drop=True)


def fred_series(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["date", "value"])
    out = pd.DataFrame({"date": pd.to_datetime(d["date"]), "value": pd.to_numeric(d["value"], errors="coerce")})
    out["avail_utc"] = (out["date"] + pd.Timedelta(days=2)).dt.tz_localize("UTC")
    out = out[out["avail_utc"] < C.READ_END]
    return out.sort_values("avail_utc").reset_index(drop=True)


def daily_features(name: str, s: pd.DataFrame, value_col: str, decision: pd.DatetimeIndex, kind: str, licence: str,
                   source: str, levels: bool) -> tuple[pd.DataFrame, list[dict]]:
    f = pd.DataFrame(index=decision)
    meta = []
    x = s.dropna(subset=[value_col])
    ev_t = "observation/trading date D"
    av_t = "(D+1) 00:00 UTC" if kind == "yahoo" else "(D+2) 00:00 UTC"
    max_age = 24 * 7     # beyond one week without a new observation the value is stale -> NaN
    if levels:
        v, _ = asof_last(decision, x["avail_utc"], x[value_col], max_age_h=max_age)
        f[f"{name}.level"] = v
        meta.append(_meta(f"{name}.level", f"{kind}_level", source, "source unit", max_age, ev_t, av_t,
                          "last available level, age <= 168 h", licence, frequency="1d"))
    pos = x[value_col] > 0
    if kind == "yahoo" or pos.all():
        lv = np.log(x[value_col].where(pos))
        diff_name, unit, tr = "logret", "log-return", "ln(V_D / V_prev)"
    else:
        lv = x[value_col]
        diff_name, unit, tr = "diff", "source unit", "V_D - V_prev"
    for k in (1, 5):
        dv = lv - lv.shift(k)
        v, _ = asof_last(decision, x["avail_utc"], dv, max_age_h=max_age)
        f[f"{name}.{diff_name}_{k}d"] = v
        meta.append(_meta(f"{name}.{diff_name}_{k}d", f"{kind}_change", source, unit, max_age + 24 * k * 1.5, ev_t, av_t,
                          f"{tr} over {k} observation(s), last available, age <= 168 h", licence, frequency="1d"))
    return f, meta


def fx_pair_features(pair: str, path: str, decision: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[dict], dict]:
    raw = pd.read_parquet(path, filters=[("timestamp", ">=", (C.WARMUP_START - pd.Timedelta(days=1))),
                                         ("timestamp", "<", (C.READ_END + pd.Timedelta(days=1)))])
    clk = S.infer_fx_clock(raw["timestamp"])
    if clk.get("clock") is None:
        return pd.DataFrame(index=decision), [], clk
    naive = pd.DatetimeIndex(raw["timestamp"]).tz_localize(None)
    if clk["clock"].startswith("NEW_YORK_LOCAL"):
        st = naive.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    else:
        st = naive.tz_localize("UTC")
    end = st + pd.Timedelta(hours=1) if clk["clock"].endswith("BAR_START") else st
    h = pd.DataFrame({"close": raw["close"].to_numpy(float)}, index=end)
    h = h[~h.index.isna()]
    h = h[(h.index <= C.READ_END)].sort_index()
    h = h[~h.index.duplicated(keep="first")]
    lc = np.log(h["close"])
    src = f"lake:features/trading_asset_data/{pair}/1h.parquet"
    f = pd.DataFrame(index=decision)
    meta = []
    r1 = lc.diff()
    feats = {f"fx.{pair}.logret_1h": r1, f"fx.{pair}.logret_24h": lc - lc.shift(24),
             f"fx.{pair}.ewma_vol_24": np.sqrt((r1 ** 2).ewm(halflife=24, adjust=False).mean())}
    sup = {f"fx.{pair}.logret_1h": 1, f"fx.{pair}.logret_24h": 24, f"fx.{pair}.ewma_vol_24": 120}
    for k, ser in feats.items():
        # exact-time join: the pair's bar must end exactly at t (no carry-over across missing hours)
        f[k] = ser.reindex(decision).to_numpy()
        meta.append(_meta(k, "fx_cross", src, "log-return", sup[k], "UTC hourly bar", "bar end",
                          k.split(".")[-1] + " (bar-indexed)", HISTDATA_LIC))
    clk["file"] = src
    return f, meta, clk
