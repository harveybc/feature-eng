"""Causal cross-asset candidate features for EURUSD H1 (feature-eng plugin ``cross_asset_h1``).

Owner decision 2026-10-20: look for direction information OUTSIDE the EURUSD price pool. Every feature is defined
on the admitted EURUSD hourly grid (UTC bar START ``s``; the decision is taken at the bar close ``s + 1 h``) and
uses only source values whose AVAILABILITY time is <= that decision time. Sources and their availability rules:

  hourly FX (HistData G10)  bar start re-localized from the lake clock (Europe/London wall - 5 h) to UTC; a bar is
                            available at its close (start + 1 h). WEEKLY clock gate: the HistData EURUSD, after the
                            same re-localization, must align with the admitted EURUSD at lag 0 in >= 97 % of the
                            weeks with a median weekly correlation >= 0.98, otherwise the builder refuses.
  hourly BTCUSDT (Binance)  ``open_time`` is a true UTC bar start; available at start + 1 h.
  daily market series       (FRED market yields, Yahoo VIX / indices / futures): the value dated D is available from
                            D + ``daily_lag_days`` days 00:00 UTC (declared conservative lag: 2 days).
  CFTC COT (EUR, weekly)    as-of Tuesday T; public on Friday afternoon (US); available from T + 4 days 00:00 UTC.

Each source's features are computed on the source's OWN series with trailing operations only, then joined to the
grid with ``merge_asof`` on availability (backward) with a staleness bound. No centred or two-sided operation is
used anywhere. Nothing here is a label and nothing reads TEST rows (callers filter before 2025-01-01).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
HIST_FX_PAIRS = ("gbpusd", "usdchf", "usdjpy", "eurgbp", "eurjpy", "audusd")
USD_SIGN = {"gbpusd": -1.0, "audusd": -1.0, "usdchf": 1.0, "usdjpy": 1.0}       # + = USD stronger
EUR_SIGN = {"eurgbp": 1.0, "eurjpy": 1.0}                                         # + = EUR stronger
HOURLY_WINDOWS = (1, 4, 24)


class ClockGateError(ValueError):
    pass


# ----------------------------------------------------------------------------- clocks
def relocalize_lake_clock(df: pd.DataFrame, time_col: str = "datetime", shift_hours: int = 5) -> pd.DataFrame:
    """HistData stamps carry a UTC label but are the LAKE clock: Europe/London wall time minus 5 h (identity
    EU_DST_UTC_MINUS_5, the same clock found for the lake, K15). Measured 2026-10-20: a New York re-localization is
    exact in summer and winter but off by one hour in the weeks when only the US is on daylight time, which is the
    signature of an EU-DST clock. Label + 5 h -> localize Europe/London (ambiguous / nonexistent -> dropped) -> UTC."""
    t = pd.DatetimeIndex(pd.to_datetime(df[time_col]))
    wall = t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t
    london = (wall + pd.Timedelta(hours=shift_hours)).tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    out = df.drop(columns=[time_col]).copy()
    out.index = london.tz_convert("UTC")
    out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.index.name = "bar_start_utc"
    return out


def relocalize_new_york(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    """REJECTED hypothesis kept for the record (one-hour error in the US-only DST weeks); not used by the builder."""
    t = pd.DatetimeIndex(pd.to_datetime(df[time_col]))
    wall = t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t
    ny = wall.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    out = df.drop(columns=[time_col]).copy()
    out.index = ny.tz_convert("UTC")
    out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.index.name = "bar_start_utc"
    return out


def clock_gate(hist_close: pd.Series, admitted_close: pd.Series, *, lags=range(-2, 3), min_share_lag0: float = 0.97,
               min_median_corr: float = 0.98, min_rows: int = 40) -> dict:
    """WEEKLY alignment test of two UTC-indexed close series (hourly log returns): for every ISO week with enough
    joint rows, the lag (hours) maximising the correlation. Refuses unless lag 0 wins in >= ``min_share_lag0`` of the
    weeks and the median weekly lag-0 correlation is >= ``min_median_corr``. A week where the vendors' VALUES differ
    lowers its correlation at every lag but does not move the best lag; a clock error moves it."""
    a = np.log(admitted_close).diff()
    h = np.log(hist_close).diff()
    weeks, best = [], []
    iso = a.index.isocalendar()
    key = pd.Series(iso.year.to_numpy().astype(int) * 100 + iso.week.to_numpy().astype(int), index=a.index)
    grp = pd.Series(range(len(a)), index=a.index).groupby(key.to_numpy())
    for yw, g in grp:
        y, w = divmod(int(yw), 100)
        lo, hi = a.index[g.iloc[0]], a.index[g.iloc[-1]]
        cs = {}
        for L in lags:
            hs = h[lo - 3 * HOUR: hi + 3 * HOUR].copy()
            hs.index = hs.index + L * HOUR
            jj = pd.concat([a[lo:hi], hs], axis=1, join="inner").dropna()
            if len(jj) >= min_rows:
                cs[int(L)] = float(jj.iloc[:, 0].corr(jj.iloc[:, 1]))
        if 0 in cs:
            b = max(cs, key=cs.get)
            weeks.append({"week": f"{y}-W{int(w):02d}", "best_lag": b, "corr_lag0": cs[0], "corr_best": cs[b]})
            best.append(b)
    share0 = float(np.mean([b == 0 for b in best])) if best else 0.0
    med = float(np.median([w["corr_lag0"] for w in weeks])) if weeks else float("nan")
    res = {"weeks": len(weeks), "share_best_lag0": share0, "median_corr_lag0": med, "min_share_lag0": min_share_lag0,
           "min_median_corr": min_median_corr, "weeks_not_lag0": [w for w in weeks if w["best_lag"] != 0],
           "weeks_lag0_below_0p9": [w["week"] for w in weeks if w["corr_lag0"] < 0.9]}
    if not (share0 >= min_share_lag0 and med >= min_median_corr):
        raise ClockGateError(f"CLOCK_GATE: lag 0 best in {share0:.3f} of {len(weeks)} weeks, median corr {med:.4f}")
    res["passed"] = True
    return res


# ----------------------------------------------------------------------------- per-source trailing features
def hourly_features(close: pd.Series, prefix: str) -> pd.DataFrame:
    """Trailing log returns over 1/4/24 observed bars and 24-bar realized volatility, on the source's own bars.
    Index = source bar start (UTC); the value is known at bar close (start + 1 h)."""
    lc = np.log(close.astype(float))
    r1 = lc.diff()
    out = {f"{prefix}.ret_{w}": lc.diff(w) for w in HOURLY_WINDOWS}
    out[f"{prefix}.rv_24"] = r1.pow(2).rolling(24, min_periods=24).sum().pow(0.5)
    return pd.DataFrame(out, index=close.index)


def composite(feats: dict[str, pd.DataFrame], signs: dict[str, float], name: str) -> pd.DataFrame:
    """Signed mean of the legs' trailing returns on the union of their bar starts (each leg forward-filled at most
    one bar so a single missing leg-bar does not void the composite; values stay trailing)."""
    cols = {}
    for w in HOURLY_WINDOWS:
        legs = [signs[p] * feats[p][f"fx.{p}.ret_{w}"] for p in signs if p in feats]
        frame = pd.concat(legs, axis=1).sort_index().ffill(limit=1)
        cols[f"fx.{name}.ret_{w}"] = frame.mean(axis=1, skipna=False)
    return pd.DataFrame(cols)


def btc_features(df: pd.DataFrame) -> pd.DataFrame:
    """BTCUSDT hourly: trailing returns, rv_24, and the 24-bar mean taker-buy share of volume."""
    close = df["close"].astype(float)
    f = hourly_features(close, "btc")
    share = (df["taker_buy_base_volume"].astype(float) / df["volume"].astype(float).replace(0.0, np.nan))
    f["btc.taker_buy_share_24"] = share.rolling(24, min_periods=24).mean()
    return f


def daily_features(series: pd.Series, name: str, kind: str) -> pd.DataFrame:
    """Daily market series indexed by its calendar date. ``kind``: 'yield' -> level, 1d and 5d change;
    'price' -> 1d and 5d log return; 'vol' -> level and 1d change."""
    s = series.astype(float).dropna().sort_index()
    if kind == "yield":
        return pd.DataFrame({f"dly.{name}.level": s, f"dly.{name}.chg_1d": s.diff(1), f"dly.{name}.chg_5d": s.diff(5)})
    if kind == "vol":
        return pd.DataFrame({f"dly.{name}.level": s, f"dly.{name}.chg_1d": s.diff(1)})
    lp = np.log(s)
    return pd.DataFrame({f"dly.{name}.ret_1d": lp.diff(1), f"dly.{name}.ret_5d": lp.diff(5)})


def cot_features(df: pd.DataFrame) -> pd.DataFrame:
    """CFTC EUR weekly (as-of Tuesday): net long, weekly change, net long / open interest proxy, 156-week z-score."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
    net = d["EUR_Net_Long"].astype(float)
    gross = (d["EUR_Long"].astype(float) + d["EUR_Short"].astype(float)).replace(0.0, np.nan)
    mu = net.rolling(156, min_periods=52).mean()
    sd = net.rolling(156, min_periods=52).std()
    return pd.DataFrame({"cot.eur.net_long": net, "cot.eur.net_change": net.diff(1), "cot.eur.net_share": net / gross,
                         "cot.eur.net_z156": (net - mu) / sd})


# ----------------------------------------------------------------------------- availability joins
def asof_join(grid_start: pd.DatetimeIndex, feats: pd.DataFrame, available_at: pd.DatetimeIndex,
              max_stale: pd.Timedelta) -> pd.DataFrame:
    """For each grid bar (decision at start + 1 h) the latest feature row with available_at <= decision, if it is
    not older than ``max_stale``; else NaN."""
    decision = pd.DataFrame({"decision": (pd.DatetimeIndex(grid_start) + HOUR).as_unit("ns")}, index=np.arange(len(grid_start)))
    f = feats.copy()
    f["available_at"] = pd.DatetimeIndex(available_at).as_unit("ns")
    f = f.sort_values("available_at").dropna(subset=["available_at"])
    j = pd.merge_asof(decision.sort_values("decision"), f, left_on="decision", right_on="available_at",
                      direction="backward")
    stale = (j["decision"] - j["available_at"]) > max_stale
    cols = [c for c in feats.columns]
    j.loc[stale | j["available_at"].isna(), cols] = np.nan
    j = j.sort_index()
    out = j[cols]
    out.index = grid_start
    return out


def hourly_available(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index) + HOUR


def daily_available(dates, lag_days: int = 2) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    d = d.tz_localize(None) if d.tz is not None else d
    return (d.normalize() + pd.Timedelta(days=lag_days)).tz_localize("UTC")


def cot_available(asof_tuesdays) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(pd.to_datetime(asof_tuesdays))
    d = d.tz_localize(None) if d.tz is not None else d
    return (d.normalize() + pd.Timedelta(days=4)).tz_localize("UTC")


# ----------------------------------------------------------------------------- feature metadata
def feature_meta(columns) -> list[dict]:
    out = []
    for c in columns:
        fam = c.split(".")[0]
        rule = {"fx": "source bar close (start + 1 h), lake clock (London wall - 5 h) re-localized to UTC, weekly clock gate passed",
                "btc": "source bar close (start + 1 h), Binance UTC open_time",
                "dly": "daily value dated D available from D + 2 days 00:00 UTC",
                "cot": "CFTC as-of Tuesday T available from T + 4 days 00:00 UTC"}[fam]
        out.append({"feature_id": c, "family": f"cross_asset_{fam}", "availability_time": "timestamp",
                    "availability_rule": rule, "transform": "trailing differences / rolling sums of the source's own past values",
                    "unit": "log return" if ".ret_" in c else ("percentage points" if c.startswith("dly.") and (".level" in c or ".chg_" in c) else "ratio/level")})
    return out


class Plugin:
    """feature-eng plugin entry: builds the cross-asset frame on a given UTC hourly grid from pre-loaded sources."""

    plugin_params = {"daily_lag_days": 2, "hourly_max_stale_hours": 3, "daily_max_stale_days": 7, "cot_max_stale_days": 14,
                     "min_share_lag0": 0.97, "min_median_corr": 0.98}
    plugin_debug_vars = list(plugin_params)

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    def get_debug_info(self):
        return {v: self.params.get(v) for v in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def build(self, grid_start: pd.DatetimeIndex, *, fx: dict[str, pd.DataFrame] | None = None, btc: pd.DataFrame | None = None,
              daily: dict[str, tuple[pd.Series, str]] | None = None, cot: pd.DataFrame | None = None) -> pd.DataFrame:
        """``fx``: pair -> UTC-indexed OHLC frame (already re-localized); ``btc``: UTC-indexed Binance frame;
        ``daily``: name -> (date-indexed series, kind); ``cot``: raw CFTC frame. Returns grid-indexed features."""
        p = self.params
        parts = []
        hstale = pd.Timedelta(hours=p["hourly_max_stale_hours"])
        if fx:
            per = {k: hourly_features(v["close"], f"fx.{k}") for k, v in fx.items()}
            for k, f in per.items():
                parts.append(asof_join(grid_start, f, hourly_available(f.index), hstale))
            for name, signs in (("usd_strength", USD_SIGN), ("eur_strength", EUR_SIGN)):
                comp = composite(per, signs, name)
                parts.append(asof_join(grid_start, comp, hourly_available(comp.index), hstale))
        if btc is not None:
            f = btc_features(btc)
            parts.append(asof_join(grid_start, f, hourly_available(f.index), hstale))
        if daily:
            for name, (s, kind) in daily.items():
                f = daily_features(s, name, kind)
                parts.append(asof_join(grid_start, f, daily_available(f.index, p["daily_lag_days"]),
                                       pd.Timedelta(days=p["daily_max_stale_days"])))
        if cot is not None:
            f = cot_features(cot)
            parts.append(asof_join(grid_start, f, cot_available(f.index), pd.Timedelta(days=p["cot_max_stale_days"])))
        return pd.concat(parts, axis=1)
