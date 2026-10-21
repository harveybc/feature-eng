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


def one_hour_returns(close: pd.Series) -> pd.Series:
    """Log return of each bar over EXACTLY one hour: NaN unless the previous observation is exactly 1 h earlier
    (a missing source bar must never turn a multi-hour move into a "1 h" return; Musashi finding 2026-10-20)."""
    c = close.sort_index()
    r = np.log(c).diff()
    gap = c.index.to_series().diff()
    return r.where(gap == HOUR)


def clock_gate(hist_close: pd.Series, admitted_close: pd.Series, *, lags=range(-2, 3), min_share_lag0: float = 0.97,
               min_median_corr: float = 0.98, min_rows: int = 40) -> dict:
    """WEEKLY alignment test of two UTC-indexed close series (log returns over EXACTLY one hour in each feed): for every ISO week with enough
    joint rows, the lag (hours) maximising the correlation. Refuses unless lag 0 wins in >= ``min_share_lag0`` of the
    weeks and the median weekly lag-0 correlation is >= ``min_median_corr``. A clock error moves the best lag."""
    a = one_hour_returns(admitted_close)
    h = one_hour_returns(hist_close)
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
def on_grid(close: pd.Series, grid_start: pd.DatetimeIndex) -> pd.Series:
    """The source close at EXACTLY each grid bar start (UTC); a source bar missing at that start stays NaN (no fill)."""
    c = close[~close.index.duplicated(keep="first")]
    return c.reindex(pd.DatetimeIndex(grid_start))


def hourly_features(close: pd.Series, prefix: str) -> pd.DataFrame:
    """On the EURUSD admitted grid (observed market bars; closures are part of the grid, not gaps): trailing log
    returns over 1/4/24 grid bars and the 24-bar realized volatility. A missing source bar at either end of a
    return makes it NaN; rv_24 needs all 24 one-bar returns. The value at grid bar t uses the source bar that
    starts at t (complete at the decision t + 1 h)."""
    lc = np.log(close.astype(float))
    r1 = lc.diff()
    out = {f"{prefix}.ret_{w}": lc.diff(w) for w in HOURLY_WINDOWS}
    out[f"{prefix}.rv_24"] = r1.pow(2).rolling(24, min_periods=24).sum().pow(0.5)
    return pd.DataFrame(out, index=close.index)


def composite(feats: dict[str, pd.DataFrame], signs: dict[str, float], name: str) -> pd.DataFrame:
    """Signed mean of the legs' grid returns; NaN when any leg is missing at that bar (no fill)."""
    cols = {}
    for w in HOURLY_WINDOWS:
        legs = [signs[p] * feats[p][f"fx.{p}.ret_{w}"] for p in signs if p in feats]
        cols[f"fx.{name}.ret_{w}"] = pd.concat(legs, axis=1).mean(axis=1, skipna=False)
    return pd.DataFrame(cols)


def btc_features(df: pd.DataFrame, grid_start: pd.DatetimeIndex) -> pd.DataFrame:
    """BTCUSDT on the EURUSD grid: returns over grid bars (a weekend grid step spans the FX closure, declared), rv_24,
    and the mean taker-buy share of the 24 source hours ending with the grid bar (24 consecutive source hours)."""
    close = on_grid(df["close"].astype(float), grid_start)
    f = hourly_features(close, "btc")
    share = (df["taker_buy_base_volume"].astype(float) / df["volume"].astype(float).replace(0.0, np.nan))
    share = share[~share.index.duplicated(keep="first")].asfreq("h")          # BTC trades 24/7: true hourly grid
    f["btc.taker_buy_share_24"] = on_grid(share.rolling(24, min_periods=24).mean(), grid_start)
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


def daily_available(dates, lag_days: int = 2, offset_hours: float | None = None) -> pd.DatetimeIndex:
    """r2 default: date D + ``lag_days`` 00:00 UTC. r3 (D2, release time): D 00:00 UTC + ``offset_hours`` (declared
    per source from its publication time, e.g. US close data 22 h, FRED H.15 yields 45 h = next day 21:00 UTC)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    d = d.tz_localize(None) if d.tz is not None else d
    if offset_hours is not None:
        return (d.normalize() + pd.Timedelta(hours=float(offset_hours))).tz_localize("UTC")
    return (d.normalize() + pd.Timedelta(days=lag_days)).tz_localize("UTC")


def cot_available(asof_tuesdays, offset_hours: float | None = None) -> pd.DatetimeIndex:
    """r2: Tuesday + 4 days 00:00 UTC. r3: Tuesday 00:00 UTC + ``offset_hours`` (Friday 21:00 UTC = 93 h; the CFTC
    publishes Friday 15:30 ET = 19:30/20:30 UTC)."""
    d = pd.DatetimeIndex(pd.to_datetime(asof_tuesdays))
    d = d.tz_localize(None) if d.tz is not None else d
    if offset_hours is not None:
        return (d.normalize() + pd.Timedelta(hours=float(offset_hours))).tz_localize("UTC")
    return (d.normalize() + pd.Timedelta(days=4)).tz_localize("UTC")


# ----------------------------------------------------------------------------- D3 / D6: factors and causal Kalman channels
def kalman_local_trend(y: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Causal local-linear-trend Kalman FILTER (never smoothed) on a series with unit observation noise and state
    noise q (level and slope): returns the filtered level and slope at each t using y[:t+1] only. NaN observations
    are skipped (prediction only)."""
    n = len(y)
    lvl, slp = np.full(n, np.nan), np.full(n, np.nan)
    x = None
    P = None
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q, q * q])
    H = np.array([1.0, 0.0])
    for t in range(n):
        if x is None:
            if np.isfinite(y[t]):
                x, P = np.array([y[t], 0.0]), np.eye(2)
            continue
        x = F @ x
        P = F @ P @ F.T + Q
        if np.isfinite(y[t]):
            S = H @ P @ H + 1.0
            K = P @ H / S
            x = x + K * (y[t] - H @ x)
            P = P - np.outer(K, H) @ P
        lvl[t], slp[t] = x
    return lvl, slp


def kalman_q_mle(y: np.ndarray, grid=(1e-4, 1e-3, 1e-2, 1e-1)) -> float:
    """Pick q on TRAIN data by the one-step Gaussian prediction likelihood (unit observation noise scale)."""
    best, bq = -np.inf, grid[0]
    yy = y[np.isfinite(y)]
    sd = np.std(np.diff(yy)) or 1.0
    z = yy / sd
    for q in grid:
        lvl, slp = kalman_local_trend(z, q)
        pred = (lvl + slp)[:-1]
        e = z[1:] - pred
        e = e[np.isfinite(e)]
        ll = -0.5 * np.sum(np.log(2 * np.pi) + e ** 2)
        if ll > best:
            best, bq = ll, q
    return float(bq)


def pca_fit(R_fit: pd.DataFrame, k: int) -> dict:
    """PCA loadings fitted on a PRE-GRID period (rows with every column present); frozen afterwards."""
    X = R_fit.to_numpy(float)
    ok = np.isfinite(X).all(1)
    mu, sd = X[ok].mean(0), X[ok].std(0)
    Z = (X[ok] - mu) / np.where(sd > 0, sd, 1)
    _, sv, vt = np.linalg.svd(Z, full_matrices=False)
    W = vt[:k].T
    for j in range(k):                                   # sign convention: largest |loading| positive
        if W[np.argmax(np.abs(W[:, j])), j] < 0:
            W[:, j] = -W[:, j]
    return {"columns": list(R_fit.columns), "mu": mu, "sd": sd, "W": W, "rows": int(ok.sum()),
            "explained": (sv[:k] ** 2 / (sv ** 2).sum()).round(4).tolist()}


def pca_apply(R: pd.DataFrame, fit: dict) -> pd.DataFrame:
    X = R[fit["columns"]].to_numpy(float)
    F = ((X - fit["mu"]) / np.where(fit["sd"] > 0, fit["sd"], 1)) @ fit["W"]
    F[~np.isfinite(X).all(1)] = np.nan
    return pd.DataFrame(F, index=R.index, columns=[f"f{j + 1}" for j in range(fit["W"].shape[1])])


def factor_features(grid_start, per_pair: dict, fit_returns: pd.DataFrame, eur_close_grid: pd.Series,
                    daily_grid: pd.DataFrame | None, daily_fit: pd.DataFrame | None, k_fx: int = 3, k_daily: int = 2):
    """D3: PCA factors of the six pairs' one-bar grid returns and of the daily cross-asset innovations, with loadings
    and Kalman noise ratios fitted ONLY on the pre-grid period (sources' own history, before the first grid bar), so
    every grid value is a function of data available at its decision; 4/24-bar sums and a causal Kalman slope of the
    cumulated factor. D6: causal Kalman local-trend slope of the EURUSD grid log close at three fixed speeds."""
    out, info = {}, {}
    fx_fit = pca_fit(fit_returns, k_fx)
    info["fx_pca"] = {k: v for k, v in fx_fit.items() if k in ("columns", "rows", "explained")} | {"W": fx_fit["W"].round(4).tolist()}
    R1 = pd.DataFrame({k: per_pair[k][f"fx.{k}.ret_1"] for k in fx_fit["columns"]})
    F = pca_apply(R1, fx_fit)
    Ff = pca_apply(fit_returns, fx_fit)
    for c in F.columns:
        f = F[c]
        out[f"fac.fx_{c}.ret_1"] = f
        out[f"fac.fx_{c}.ret_4"] = f.rolling(4, min_periods=4).sum()
        out[f"fac.fx_{c}.ret_24"] = f.rolling(24, min_periods=24).sum()
        q = kalman_q_mle(Ff[c].fillna(0.0).cumsum().to_numpy()[-20000:])
        cum = f.fillna(0.0).cumsum().where(f.notna())
        _, slope = kalman_local_trend(cum.to_numpy() / (np.nanstd(Ff[c]) or 1.0), q)
        out[f"fac.fx_{c}.kalman_slope"] = slope
        info[f"fx_{c}_q"] = q
    if daily_grid is not None and daily_fit is not None and len(daily_fit.columns):
        d_fit = pca_fit(daily_fit, k_daily)
        info["daily_pca"] = {k: v for k, v in d_fit.items() if k in ("columns", "rows", "explained")} | {"W": d_fit["W"].round(4).tolist()}
        Fd = pca_apply(daily_grid, d_fit)
        for c in Fd.columns:
            out[f"fac.daily_{c}"] = Fd[c]
    lc = np.log(eur_close_grid.to_numpy(float))
    scale = np.nanstd(np.diff(lc)[:24 * 250]) or 1.0          # first grid year of one-bar moves (scale only)
    for name, q in (("fast", 1e-1), ("mid", 1e-2), ("slow", 1e-3)):
        _, slope = kalman_local_trend(lc / scale, q)
        out[f"kal.eurusd.slope_{name}"] = slope
    return pd.DataFrame(out, index=grid_start), info


# ----------------------------------------------------------------------------- feature metadata
def feature_meta(columns) -> list[dict]:
    out = []
    for c in columns:
        fam = c.split(".")[0]
        rule = {"fac": "TRAIN-frozen PCA loadings applied to inputs available at the decision; causal Kalman filter (never smoothed)",
                "kal": "causal Kalman local-trend filter of the EURUSD grid closes <= t (never smoothed)","fx": "source bar starting at the grid bar (complete at the decision), lake clock re-localized to UTC; returns over EURUSD grid bars, NaN when a source bar is missing",
                "btc": "source bar starting at the grid bar (complete at the decision), Binance UTC; returns over EURUSD grid bars",
                "dly": "daily value dated D available from D + 2 days 00:00 UTC",
                "cot": "CFTC as-of Tuesday T available from T + 4 days 00:00 UTC"}[fam]
        out.append({"feature_id": c, "family": f"cross_asset_{fam}", "availability_time": "timestamp",
                    "availability_rule": rule, "transform": "trailing differences / rolling sums over observed EURUSD grid bars (never filled across a missing source bar)",
                    "unit": "log return" if ".ret_" in c else ("percentage points" if c.startswith("dly.") and (".level" in c or ".chg_" in c) else "ratio/level")})
    return out


class Plugin:
    """feature-eng plugin entry: builds the cross-asset frame on a given UTC hourly grid from pre-loaded sources."""

    plugin_params = {"daily_lag_days": 2, "daily_max_stale_days": 7, "cot_max_stale_days": 14, "release_offset_hours": None,
                     "factors": False, "factor_fit_start": "2015-01-01",
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
              daily: dict[str, tuple[pd.Series, str]] | None = None, cot: pd.DataFrame | None = None,
              eur_close: pd.Series | None = None) -> pd.DataFrame:
        """Hourly sources are joined by EXACT bar start on the grid (missing bar -> NaN); daily / COT by availability.
        ``fx``: pair -> UTC-indexed OHLC frame (already re-localized); ``btc``: UTC-indexed Binance frame;
        ``daily``: name -> (date-indexed series, kind); ``cot``: raw CFTC frame. Returns grid-indexed features."""
        p = self.params
        parts = []
        if fx:
            per = {k: hourly_features(on_grid(v["close"], grid_start), f"fx.{k}") for k, v in fx.items()}
            parts += list(per.values())
            for name, signs in (("usd_strength", USD_SIGN), ("eur_strength", EUR_SIGN)):
                parts.append(composite(per, signs, name))
        if btc is not None:
            parts.append(btc_features(btc, grid_start))
        offsets = p.get("release_offset_hours") or {}
        dly_parts = []
        if daily:
            for name, (s, kind) in daily.items():
                f = daily_features(s, name, kind)
                avail = daily_available(f.index, p["daily_lag_days"], offsets.get(name))
                j = asof_join(grid_start, f, avail, pd.Timedelta(days=p["daily_max_stale_days"]))
                parts.append(j)
                dly_parts.append(j)
        if cot is not None:
            f = cot_features(cot)
            parts.append(asof_join(grid_start, f, cot_available(f.index, offsets.get("cot")), pd.Timedelta(days=p["cot_max_stale_days"])))
        if p.get("factors") and fx and eur_close is not None:
            g0 = pd.DatetimeIndex(grid_start)[0]
            f0 = pd.Timestamp(p["factor_fit_start"], tz="UTC")
            own = {k: one_hour_returns(v["close"]) for k, v in fx.items()}
            fit_ret = pd.concat(own, axis=1, join="inner")
            fit_ret = fit_ret[(fit_ret.index >= f0) & (fit_ret.index < g0)]
            dgrid = dfit = None
            if dly_parts and daily:
                D = pd.concat(dly_parts, axis=1)
                dgrid = D[[c for c in D.columns if c.endswith(("chg_1d", "ret_1d"))]]
                dd = pd.concat([daily_features(s_, nm, kd) for nm, (s_, kd) in daily.items()], axis=1)
                dd = dd[[c for c in dgrid.columns]]
                di = pd.DatetimeIndex(dd.index)
                dfit = dd[(di >= f0.tz_localize(None)) & (di < g0.tz_localize(None))]
            ff, self.factor_info = factor_features(grid_start, per, fit_ret, eur_close, dgrid, dfit)
            self.factor_info["fit_period"] = [str(f0), str(g0)]
            parts.append(ff)
        return pd.concat(parts, axis=1)
