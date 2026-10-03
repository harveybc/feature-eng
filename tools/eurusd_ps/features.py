"""Batch features on the decision grid. Every column is registered with its
unit, frequency, event/availability clock, window support and identity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .asof import asof_count_sum, asof_last, asof_price

LAKE_LICENCE = "HistData.com free data (lake copy; research use; redistribution of raw bytes not granted)"


def _meta(fid, family, source, unit, support_h, event_time, availability_time, transform, licence=LAKE_LICENCE,
          frequency="1h", role="feature", note=""):
    return {"feature_id": fid, "family": family, "source": source, "unit": unit, "frequency": frequency,
            "support_h": support_h, "event_time": event_time, "availability_time": availability_time,
            "transform": transform, "licence": licence, "role": role, "note": note}


PRICE_SRC = "lake:features/trading_asset_data/eurusd/5m.parquet"
BAR_EVT = "UTC hourly bar [t-1h, t)"
BAR_AV = "t (bar end)"


def price_features(hourly: pd.DataFrame, decision: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[dict]]:
    h = hourly
    c = h["close"]
    lc = np.log(c)
    f = pd.DataFrame(index=h.index)
    meta = []
    end = h.index
    cl = c.to_numpy(float)
    for k in (1, 2, 4, 6, 12, 24, 48, 120, 168):
        past, _ = asof_price(h.index, end, cl, -k)
        f[f"px.logret_{k}h"] = np.log(cl / past)
        meta.append(_meta(f"px.logret_{k}h", "returns", PRICE_SRC, "log-return", k, BAR_EVT, BAR_AV,
                          f"ln(C(t)/C_asof(t-{k}h)) elapsed time"))
    f["px.log_hl"] = np.log(h["high"] / h["low"]); meta.append(_meta("px.log_hl", "range", PRICE_SRC, "log-ratio", 1, BAR_EVT, BAR_AV, "ln(H/L)"))
    rngp = (h["high"] - h["low"]).replace(0, np.nan)
    f["px.close_loc"] = (c - h["low"]) / rngp; meta.append(_meta("px.close_loc", "range", PRICE_SRC, "fraction", 1, BAR_EVT, BAR_AV, "(C-L)/(H-L)"))
    f["px.log_co"] = np.log(c / h["open"]); meta.append(_meta("px.log_co", "range", PRICE_SRC, "log-ratio", 1, BAR_EVT, BAR_AV, "ln(C/O)"))
    f["px.rv5"] = h["rv5"]; meta.append(_meta("px.rv5", "volatility", PRICE_SRC, "log-return", 1, BAR_EVT, BAR_AV, "sqrt(sum within-hour 5m log-return^2)"))
    r1 = lc.diff()
    for hl in (24, 168):
        f[f"px.ewma_vol_{hl}"] = np.sqrt((r1 ** 2).ewm(halflife=hl, adjust=False).mean())
        meta.append(_meta(f"px.ewma_vol_{hl}", "volatility", PRICE_SRC, "log-return", 5 * hl, BAR_EVT, BAR_AV,
                          f"EWMA std of 1h log returns, halflife {hl} bars", note="support ~5 halflives"))
    pk = (np.log(h["high"] / h["low"]) ** 2) / (4 * np.log(2))
    f["px.parkinson_24"] = np.sqrt(pk.rolling(24, min_periods=24).mean())
    meta.append(_meta("px.parkinson_24", "volatility", PRICE_SRC, "log-return", 24, BAR_EVT, BAR_AV, "Parkinson over 24 bars"))
    gap = pd.Series(h.index, index=h.index).diff() / pd.Timedelta(hours=1)
    f["px.hours_since_prev_bar"] = gap
    meta.append(_meta("px.hours_since_prev_bar", "market_state", PRICE_SRC, "hours", 72, BAR_EVT, BAR_AV, "elapsed hours since previous bar end"))
    for n in (24, 168):
        m = c.rolling(n, min_periods=n).mean(); s = c.rolling(n, min_periods=n).std()
        f[f"px.zclose_{n}"] = (c - m) / s
        meta.append(_meta(f"px.zclose_{n}", "returns", PRICE_SRC, "z", n, BAR_EVT, BAR_AV, f"(C-mean_{n})/std_{n} over bars"))
    f["px.log_close"] = lc; meta.append(_meta("px.log_close", "level", PRICE_SRC, "log-price", 0, BAR_EVT, BAR_AV, "ln(C)"))
    f["q.n5"] = h["n5"]; meta.append(_meta("q.n5", "quality", PRICE_SRC, "count", 1, BAR_EVT, BAR_AV, "5m bars in the hour",
                                           role="quality_excluded", note="DATA_COMPLETENESS_COUNT: profiled, not a market feature"))
    # technical indicators: feature-eng tech_indicator defaults (pandas_ta lengths), explicit formulas
    tf, tm = technicals(h)
    f = pd.concat([f, tf], axis=1)
    meta += tm
    return f.reindex(decision), meta


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def technicals(h: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    c, hi, lo = h["close"], h["high"], h["low"]
    f = pd.DataFrame(index=h.index)
    meta = []
    note = "formula of feature-eng app/plugins/tech_indicator.py defaults (pandas_ta); pandas_ta parity NOT_VERIFIED (absent on worker_a)"
    d = c.diff()
    up, dn = _wilder(d.clip(lower=0), 14), _wilder((-d).clip(lower=0), 14)
    f["ta.rsi_14"] = 100 - 100 / (1 + up / dn)
    ema = lambda x, n: x.ewm(span=n, adjust=False, min_periods=n).mean()
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    f["ta.macd_n"] = macd / c
    f["ta.macd_hist_n"] = (macd - sig) / c
    f["ta.ema20_dev"] = np.log(c / ema(c, 20))
    ll, hh = lo.rolling(14, min_periods=14).min(), hi.rolling(14, min_periods=14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    f["ta.stoch_k"] = k.rolling(3, min_periods=3).mean()
    f["ta.stoch_d"] = f["ta.stoch_k"].rolling(3, min_periods=3).mean()
    f["ta.willr_14"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    f["ta.atr_14_n"] = atr / c
    upm, dnm = hi.diff(), -lo.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=h.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=h.index)
    pdi, ndi = 100 * _wilder(pdm, 14) / atr, 100 * _wilder(ndm, 14) / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    f["ta.adx_14"] = _wilder(dx, 14)
    f["ta.dmp_14"], f["ta.dmn_14"] = pdi, ndi
    tp = (hi + lo + c) / 3
    md = tp.rolling(20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    f["ta.cci_20"] = (tp - tp.rolling(20, min_periods=20).mean()) / (0.015 * md)
    m20, s20 = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std(ddof=0)
    f["ta.bb_pctb_20"] = (c - (m20 - 2 * s20)) / (4 * s20)
    f["ta.bb_width_20"] = 4 * s20 / m20
    f["ta.mom_10_n"] = (c - c.shift(10)) / c
    f["ta.roc_10"] = 100 * (c / c.shift(10) - 1)
    sup = {"ta.rsi_14": 70, "ta.macd_n": 130, "ta.macd_hist_n": 175, "ta.ema20_dev": 100, "ta.stoch_k": 16,
           "ta.stoch_d": 18, "ta.willr_14": 14, "ta.atr_14_n": 70, "ta.adx_14": 140, "ta.dmp_14": 70, "ta.dmn_14": 70,
           "ta.cci_20": 20, "ta.bb_pctb_20": 20, "ta.bb_width_20": 20, "ta.mom_10_n": 10, "ta.roc_10": 10}
    for col in f.columns:
        meta.append(_meta(col, "technical", PRICE_SRC, "indicator", sup[col], BAR_EVT, BAR_AV,
                          f"{col} on UTC hourly bars (bar-indexed window)", note=note))
    return f, meta


def calendar_features(decision: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[dict]]:
    t = pd.DatetimeIndex(decision)
    s = t - pd.Timedelta(minutes=30)  # midpoint of the bar ending at t
    f = pd.DataFrame(index=t)
    hod = s.hour + s.minute / 60
    f["cal.hour_sin"], f["cal.hour_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    dow = s.dayofweek + hod / 24
    f["cal.dow_sin"], f["cal.dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    doy = s.dayofyear
    f["cal.doy_sin"], f["cal.doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    ny = s.tz_convert("America/New_York"); ld = s.tz_convert("Europe/London"); tk = s.tz_convert("Asia/Tokyo")
    f["cal.us_dst"] = np.array([bool(x.dst()) for x in ny], dtype=float)
    f["cal.sess_london"] = ((ld.hour >= 8) & (ld.hour < 17)).astype(float)
    f["cal.sess_newyork"] = ((ny.hour >= 8) & (ny.hour < 17)).astype(float)
    f["cal.sess_tokyo"] = ((tk.hour >= 9) & (tk.hour < 18)).astype(float)
    meta = []
    for col in f.columns:
        meta.append(_meta(col, "calendar_known", "derived:clock", "dimensionless", 0, "bar midpoint t-30min", "known in advance",
                          f"{col} at bar midpoint", licence="derived (no third-party data)",
                          note="session = published exchange-centre hours (fixed rule); holidays not included"))
    return f, meta


ARCHIVE_SRC = "feature-eng:tests/data/economic_calendar_2011_2021.csv"
ARCHIVE_LIC = "UNKNOWN_PROVENANCE (no provenance sidecar; registered in data-gov with 12 absences)"
GROUPS = {"USD": ["United States"], "EUR": ["Euro Zone", "Germany", "France", "Italy", "Spain"]}
TIERS = {"high": "High Volatility Expected", "moderate": "Moderate Volatility Expected", "low": "Low Volatility Expected"}


def archive_event_table(a: pd.DataFrame, latency_min: int = 1) -> pd.DataFrame:
    """One row per release with numeric actual/forecast/previous, a causal
    surprise z (expanding MAD of past surprises of the same description) and a
    revision (this release's 'previous' minus the prior release's 'actual')."""
    from .sources import parse_num
    e = a.copy()
    e["actual_v"], e["forecast_v"], e["previous_v"] = parse_num(e["actual"]), parse_num(e["forecast"]), parse_num(e["previous"])
    e["avail_utc"] = e["scheduled_utc"] + pd.Timedelta(minutes=latency_min)
    e["group"] = None
    for g, cs in GROUPS.items():
        e.loc[e["country"].isin(cs), "group"] = g
    e = e.sort_values("avail_utc").reset_index(drop=True)
    e["surprise"] = e["actual_v"] - e["forecast_v"]
    key = e["country"] + "|" + e["description"]
    e["key"] = key
    def causal_z(s: pd.Series) -> pd.Series:
        mad = s.abs().expanding(min_periods=6).median().shift(1)
        return s / mad.replace(0, np.nan)
    e["surprise_z"] = e.groupby("key", group_keys=False)["surprise"].apply(causal_z)
    prior_actual = e.groupby("key")["actual_v"].shift(1)
    e["revision"] = e["previous_v"] - prior_actual
    e["revision_z"] = e.groupby("key", group_keys=False)["revision"].apply(causal_z)
    return e


def coverage_mask(decision: pd.DatetimeIndex, support_h: float, valid_windows) -> np.ndarray:
    """True iff the whole look-back (t - support_h, t] lies inside one window in
    which the source is known to be complete. Outside it a count of 0 or an
    age of 168 h would be invented, so the value is NaN instead."""
    d = pd.DatetimeIndex(decision)
    lo = d - pd.Timedelta(hours=support_h)
    ok = np.zeros(len(d), dtype=bool)
    for a, b in valid_windows:
        ok |= (lo >= a) & (d <= b)
    return ok


def event_features(ev: pd.DataFrame, decision: pd.DatetimeIndex, top_k: int = 8,
                   valid_windows=None) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    if valid_windows is None:
        valid_windows = [(ev["avail_utc"].min(), ev["avail_utc"].max())]
    f = pd.DataFrame(index=decision)
    meta, inv = [], []
    evt_t = "scheduled release instant (archive clock, measured era) "
    av_t = "scheduled instant + 1 min (ASSUMED_SCHEDULED_PUBLICATION_LOCALIZED; no receipt clock)"
    for g in GROUPS:
        for tier, label in TIERS.items():
            sub = ev[(ev["group"] == g) & (ev["volatility"] == label)]
            p = f"ev.{g}.{tier}"
            cnt, _ = asof_count_sum(decision, sub["avail_utc"], None, 24)
            f[f"{p}.count_24h"] = cnt
            _, age = asof_last(decision, sub["avail_utc"], pd.Series(np.zeros(len(sub))))
            f[f"{p}.hours_since"] = np.minimum(age, 168.0)
            meta.append(_meta(f"{p}.count_24h", "event_calendar", ARCHIVE_SRC, "count", 24, evt_t, av_t, "releases in (t-24h, t]", ARCHIVE_LIC))
            meta.append(_meta(f"{p}.hours_since", "event_calendar", ARCHIVE_SRC, "hours", 168, evt_t, av_t, "min(168, hours since last release)", ARCHIVE_LIC))
            if tier == "low":
                continue
            wc = sub[sub["surprise_z"].notna()]
            v, _ = asof_last(decision, wc["avail_utc"], wc["surprise_z"], max_age_h=168)
            f[f"{p}.last_surprise_z"] = v
            _, s24 = asof_count_sum(decision, wc["avail_utc"], wc["surprise_z"].clip(-10, 10), 24)
            f[f"{p}.sum_surprise_z_24h"] = s24
            wr = sub[sub["revision_z"].notna()]
            rv, _ = asof_last(decision, wr["avail_utc"], wr["revision_z"], max_age_h=168)
            f[f"{p}.last_revision_z"] = rv
            for nm, u, tr in (("last_surprise_z", "z", "causal z of (actual-forecast), age<=168h"),
                              ("sum_surprise_z_24h", "z", "sum of clipped surprise z in (t-24h,t]"),
                              ("last_revision_z", "z", "causal z of (previous - prior actual), age<=168h")):
                meta.append(_meta(f"{p}.{nm}", "event_surprise", ARCHIVE_SRC, u, 168, evt_t, av_t, tr, ARCHIVE_LIC))
    # per-description features for the most frequent high-impact releases per group (counts measured in the archive's TRAIN part)
    for g in GROUPS:
        hi = ev[(ev["group"] == g) & (ev["volatility"] == TIERS["high"]) & ev["surprise_z"].notna()]
        top = hi["key"].value_counts().head(top_k)
        for key, n in top.items():
            sub = hi[hi["key"] == key]
            slug = "".join(ch if ch.isalnum() else "_" for ch in key.split("|")[1].lower())[:40].strip("_")
            fid = f"ev.{g}.desc.{slug}.last_surprise_z"
            v, _ = asof_last(decision, sub["avail_utc"], sub["surprise_z"], max_age_h=24 * 35)
            f[fid] = v
            meta.append(_meta(fid, "event_surprise_by_release", ARCHIVE_SRC, "z", 24 * 35, evt_t, av_t,
                              f"last causal surprise z of '{key}' within 35 days ({int(n)} releases with consensus)", ARCHIVE_LIC))
    sup = {m["feature_id"]: m["support_h"] for m in meta}
    for col in f.columns:
        f.loc[~coverage_mask(decision, sup[col], valid_windows), col] = np.nan
    for m in meta:
        m["note"] = (m["note"] + "; " if m["note"] else "") + "NaN where (t-support, t] leaves the source's complete-coverage windows"
    return f, meta, inv
