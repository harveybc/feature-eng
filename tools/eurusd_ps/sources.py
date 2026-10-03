"""Loaders that never return values at or after READ_END, and the FX clock.

Lake FX bytes (HistData-derived) carry NAIVE New York wall-clock stamps even
though the parquet column is typed UTC. ``infer_fx_clock`` measures it from
the weekly close/open; ``lake_5m_to_utc`` converts to true UTC bar starts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import contract as C

NY = "America/New_York"


def infer_fx_clock(stamps: pd.Series, min_share: float = 0.8) -> dict:
    """Weekly FX close is Friday 17:00 New York, open Sunday 17:00 New York.

    For each weekly gap (>= 30 h) record the last stamp before and the first
    after, split by US DST state. Hypotheses: NY local or UTC, stamp = bar
    start or bar end. A hypothesis passes if its predicted (close, open)
    wall-clock pair holds the modal share >= min_share in both DST states.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(stamps)).tz_localize(None) if getattr(pd.to_datetime(stamps).dt, "tz", None) is None \
        else pd.DatetimeIndex(pd.to_datetime(stamps).dt.tz_localize(None))
    ts = ts.sort_values()
    step = pd.Series(ts).diff().median()
    gaps = np.where(pd.Series(ts).shift(-1).values - ts.values >= np.timedelta64(30, "h"))[0]
    if len(gaps) < 10:
        return {"status": "UNDETERMINED", "reason": "FEWER_THAN_10_WEEKS"}
    last = ts[gaps]
    first = ts[gaps + 1]
    # DST state judged on the UTC instant under each hypothesis is circular; use the calendar date's NY state at noon
    dst = np.array([bool(pd.Timestamp(d.date()).tz_localize(NY).replace(hour=12).dst()) for d in last])
    res = {"step": str(step), "weeks": int(len(gaps)), "by_dst": {}}
    for s, nm in ((True, "US_DST"), (False, "US_STD")):
        m = dst == s
        lc = pd.Series(last[m].strftime("%a %H:%M")).value_counts(normalize=True)
        fo = pd.Series(first[m].strftime("%a %H:%M")).value_counts(normalize=True)
        res["by_dst"][nm] = {"n": int(m.sum()), "close_mode": lc.index[0], "close_share": float(lc.iloc[0]),
                             "open_mode": fo.index[0], "open_share": float(fo.iloc[0])}
    stepm = int(step / pd.Timedelta(minutes=1))
    def fmt(day, h, m):
        return f"{day} {h:02d}:{m:02d}"
    hyps = {}
    for tz in ("NEW_YORK_LOCAL", "UTC"):
        for conv in ("BAR_START", "BAR_END"):
            pred = {}
            for nm, off in (("US_DST", 4), ("US_STD", 5)):
                ch = 17 if tz == "NEW_YORK_LOCAL" else 17 + off
                close_end = pd.Timestamp(2000, 1, 7, ch % 24)
                close_stamp = close_end - pd.Timedelta(minutes=stepm) if conv == "BAR_START" else close_end
                open_start = pd.Timestamp(2000, 1, 9, ch % 24)
                open_stamp = open_start if conv == "BAR_START" else open_start + pd.Timedelta(minutes=stepm)
                pred[nm] = (fmt(close_stamp.strftime("%a"), close_stamp.hour, close_stamp.minute),
                            fmt(open_stamp.strftime("%a"), open_stamp.hour, open_stamp.minute))
            ok = all(res["by_dst"][nm]["close_mode"] == pred[nm][0] and res["by_dst"][nm]["open_mode"] == pred[nm][1]
                     and res["by_dst"][nm]["close_share"] >= min_share and res["by_dst"][nm]["open_share"] >= min_share
                     for nm in pred)
            hyps[f"{tz}/{conv}"] = {"predicted": pred, "pass": bool(ok)}
    passing = [k for k, v in hyps.items() if v["pass"]]
    res["hypotheses"] = hyps
    res["status"] = "DETERMINED" if len(passing) == 1 else "UNDETERMINED"
    res["clock"] = passing[0] if len(passing) == 1 else None
    return res


def lake_5m_to_utc(df: pd.DataFrame, clock: str) -> pd.DataFrame:
    """Return 5m bars with true UTC start/end. Stamps that do not exist or are
    ambiguous in New York local time (DST transitions, Sunday 02:00, market
    closed) become NaT and are dropped and counted."""
    naive = pd.DatetimeIndex(df["timestamp"]).tz_localize(None) if pd.DatetimeIndex(df["timestamp"]).tz is not None \
        else pd.DatetimeIndex(df["timestamp"])
    if clock.startswith("NEW_YORK_LOCAL"):
        loc = naive.tz_localize(NY, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    elif clock.startswith("UTC"):
        loc = naive.tz_localize("UTC")
    else:
        raise ValueError(f"clock {clock!r} not determined; refusing to convert")
    step = pd.Timedelta(minutes=5)
    start = loc if clock.endswith("BAR_START") else loc - step
    out = df.drop(columns=["timestamp"]).copy()
    out["start_utc"] = start
    out = out[~out["start_utc"].isna()].copy()
    out["end_utc"] = out["start_utc"] + step
    out.attrs["dropped_unmappable"] = int(len(df) - len(out))
    return out.sort_values("start_utc").reset_index(drop=True)


def load_lake_5m(path: str, clock: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Read EURUSD 5m lake bytes, convert to UTC, and cut at READ_END.

    The raw read is bounded by a parquet filter with a one-day margin (the
    naive stamps are up to 5 h behind UTC); the exact cut is applied after the
    conversion, so no 5m bar ending at or after READ_END is returned.
    """
    margin = pd.Timedelta(days=1)
    lo = (C.WARMUP_START - margin).tz_localize(None)
    hi = (C.READ_END + margin).tz_localize(None)
    raw = pd.read_parquet(path, filters=[("timestamp", ">=", lo.tz_localize("UTC")),
                                         ("timestamp", "<", hi.tz_localize("UTC"))])
    clk = infer_fx_clock(raw["timestamp"])
    use = clock or clk.get("clock")
    if use is None:
        raise ValueError("FX clock UNDETERMINED; refusing to convert")
    bars = lake_5m_to_utc(raw, use)
    bars = bars[(bars["end_utc"] <= C.READ_END) & (bars["start_utc"] >= C.WARMUP_START)].reset_index(drop=True)
    assert bars["end_utc"].max() <= C.READ_END
    clk["applied"] = use
    clk["dropped_unmappable"] = bars.attrs.get("dropped_unmappable", 0)
    return bars, clk


def hourly_from_5m(b5: pd.DataFrame) -> pd.DataFrame:
    """UTC hourly bars [H, H+1h) indexed by bar END t = H+1h."""
    h = b5["start_utc"].dt.floor("1h")
    g = b5.assign(h=h).groupby("h", sort=True)
    lr = np.log(b5["close"]).diff()
    same = h.eq(h.shift(1))
    rv = (lr.where(same) ** 2).groupby(h).sum(min_count=1)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "n5": g["close"].size(),
        "rv5": np.sqrt(rv),
    })
    out.index = out.index + pd.Timedelta(hours=1)
    out.index.name = "t_utc"
    return out


def load_archive_calendar(path: str, eras: list[dict]) -> tuple[pd.DataFrame, dict]:
    """2011-2021 calendar archive (no header). Clock eras are the measured ones
    registered in data-gov (CALENDAR_REGISTRATION_2026_09_26). UNDETERMINED
    months are excluded (never given a neighbour's offset). Rows whose UTC
    instant is >= READ_END are dropped (none exist; asserted)."""
    cols = ["event_date", "event_time", "country", "volatility", "description", "evaluation",
            "data_format", "actual", "forecast", "previous"]
    a = pd.read_csv(path, header=None, names=cols, dtype=str, keep_default_na=False)
    for c in cols:
        a[c] = a[c].str.strip()
    local = pd.to_datetime(a["event_date"] + " " + a["event_time"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    utc = pd.Series(pd.NaT, index=a.index, dtype="datetime64[ns, UTC]")
    era_of = pd.Series("", index=a.index)
    for e in eras:
        m = (local >= pd.Timestamp(e["from"])) & (local < pd.Timestamp(e["to"]) + pd.Timedelta(days=1))
        era_of[m] = e["status"]
        if e["status"] != "DETERMINED":
            continue
        if e.get("tz") == NY:
            utc[m] = pd.DatetimeIndex(local[m]).tz_localize(NY, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
        else:
            utc[m] = (pd.DatetimeIndex(local[m]) - pd.Timedelta(seconds=e["utc_offset_seconds"])).tz_localize("UTC")
    a["scheduled_utc"] = utc
    a["clock_era_status"] = era_of.replace("", "OUTSIDE_ERAS")
    stats = {"rows": int(len(a)), "unparseable_local": int(local.isna().sum()),
             "excluded_clock_undetermined": int((a["clock_era_status"] == "UNDETERMINED").sum()),
             "outside_eras": int((a["clock_era_status"] == "OUTSIDE_ERAS").sum())}
    a = a[a["scheduled_utc"].notna()].copy()
    assert (a["scheduled_utc"] < C.READ_END).all()
    stats["kept_rows"] = int(len(a))
    # complete-coverage windows in UTC: each DETERMINED era, start shifted +5 h and end +4 h (local midnight
    # bounds converted conservatively), the last one cut at the last release present in the bytes
    win = []
    for e in eras:
        if e["status"] != "DETERMINED":
            continue
        s0 = pd.Timestamp(e["from"], tz="UTC") + pd.Timedelta(hours=5)
        s1 = pd.Timestamp(e["to"], tz="UTC") + pd.Timedelta(days=1, hours=4)
        win.append([s0, min(s1, a["scheduled_utc"].max())])
    merged = []
    for w in sorted(win):
        if merged and w[0] <= merged[-1][1] + pd.Timedelta(hours=10):
            merged[-1][1] = max(merged[-1][1], w[1])
        else:
            merged.append(w)
    stats["valid_windows_utc"] = [[str(x), str(y)] for x, y in merged]
    a.attrs["valid_windows"] = [(x, y) for x, y in merged]
    return a.reset_index(drop=True), stats


_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_num(s: pd.Series) -> pd.Series:
    """'2.2', '-0.13', '1.5%', '200K', '' -> float; magnitude suffix applied."""
    x = s.astype(str).str.strip().str.replace(",", "", regex=False)
    suf = x.str.extract(r"([KMBT])$", expand=False)
    core = x.str.replace(r"[%KMBT]$", "", regex=True)
    v = pd.to_numeric(core, errors="coerce")
    mult = suf.map(_MULT).fillna(1.0)
    return v * mult
