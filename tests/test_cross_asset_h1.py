"""cross_asset_h1: clock re-localization, availability rules, causality (future perturbation), composites, gate."""
import numpy as np
import pandas as pd
import pytest

from app.plugins import cross_asset_h1 as CA


def test_lake_clock_relocalization_handles_eu_and_us_dst_weeks():
    # label = London wall - 5 h. Winter: London = UTC -> label 07:00 -> 12:00 UTC.
    # Summer (BST): London = UTC+1 -> label 07:00 -> London 12:00 -> 11:00 UTC.
    # US-only DST week (2024-03-12; US on EDT, UK still GMT): London = UTC -> 12:00 UTC (a New York rule gives 11:00).
    lab = pd.to_datetime(["2024-01-15 07:00", "2024-07-15 07:00", "2024-03-12 07:00"]).tz_localize("UTC")
    df = pd.DataFrame({"datetime": lab, "close": [1.0, 1.1, 1.2]})
    out = CA.relocalize_lake_clock(df)
    assert list(out.index.strftime("%Y-%m-%d %H:%M")) == ["2024-01-15 12:00", "2024-03-12 12:00", "2024-07-15 11:00"]   # sorted
    ny = CA.relocalize_new_york(df)
    assert ny.index[1].strftime("%H:%M") == "11:00"                    # the rejected rule differs exactly there


def test_clock_gate_passes_aligned_and_refuses_shifted():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=24 * 7 * 30, freq="h", tz="UTC")
    c = pd.Series(1.1 * np.exp(rng.normal(0, 5e-4, len(idx)).cumsum()), index=idx)
    g = CA.clock_gate(c, c)
    assert g["passed"] and g["share_best_lag0"] == 1.0
    shifted = c.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    with pytest.raises(CA.ClockGateError):
        CA.clock_gate(shifted, c)


def test_availability_rules_daily_and_cot():
    d = CA.daily_available(pd.to_datetime(["2024-03-04"]))           # Monday close
    assert str(d[0]) == "2024-03-06 00:00:00+00:00"
    t = CA.cot_available(pd.to_datetime(["2024-03-05"]))             # Tuesday as-of
    assert str(t[0]) == "2024-03-09 00:00:00+00:00"                  # Saturday 00:00 UTC, after the Friday release
    grid = pd.date_range("2024-03-05 20:00", periods=8, freq="h", tz="UTC")
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-03-03", "2024-03-04"]))
    f = CA.daily_features(s, "x", "vol")
    j = CA.asof_join(grid, f, CA.daily_available(f.index), pd.Timedelta(days=7))
    # decisions before 2024-03-06 00:00 see the value dated 03-03 (available 03-05 00:00), not 03-04
    assert j["dly.x.level"].iloc[0] == 1.0 and j["dly.x.level"].iloc[-1] == 2.0


def test_hourly_join_uses_only_bars_closed_by_the_decision_and_staleness():
    src_idx = pd.date_range("2024-01-02 00:00", periods=10, freq="h", tz="UTC")
    f = pd.DataFrame({"v": np.arange(10.0)}, index=src_idx)
    grid = pd.DatetimeIndex(["2024-01-02 03:00", "2024-01-02 15:00"], tz="UTC")
    j = CA.asof_join(grid, f, CA.hourly_available(f.index), pd.Timedelta(hours=3))
    assert j["v"].iloc[0] == 3.0                                     # bar 03:00 closes at the 04:00 decision
    assert np.isnan(j["v"].iloc[1])                                  # last source bar is too old


def test_future_perturbation_never_changes_values_known_at_or_before_the_cut():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-02", periods=24 * 60, freq="h", tz="UTC")
    fx = {p: pd.DataFrame({"close": 1.2 * np.exp(rng.normal(0, 4e-4, len(idx)).cumsum())}, index=idx) for p in CA.HIST_FX_PAIRS}
    btc = pd.DataFrame({"close": 2e4 * np.exp(rng.normal(0, 3e-3, len(idx)).cumsum()), "volume": rng.uniform(10, 20, len(idx)),
                        "taker_buy_base_volume": rng.uniform(4, 10, len(idx))}, index=idx)
    days = pd.date_range("2022-10-01", "2023-03-31", freq="B")
    daily = {"us10y": (pd.Series(3 + rng.normal(0, 0.05, len(days)).cumsum(), index=days), "yield"),
             "spx": (pd.Series(4000 * np.exp(rng.normal(0, 0.01, len(days)).cumsum()), index=days), "price")}
    tues = pd.date_range("2019-01-01", "2023-03-31", freq="W-TUE")
    cot = pd.DataFrame({"Date": tues.strftime("%Y-%m-%d"), "EUR_Long": rng.integers(1e5, 2e5, len(tues)),
                        "EUR_Short": rng.integers(5e4, 1.5e5, len(tues))})
    cot["EUR_Net_Long"] = cot.EUR_Long - cot.EUR_Short
    cot["EUR_Net_Change"] = cot.EUR_Net_Long.diff()
    grid = idx[24 * 3:]
    pl = CA.Plugin()
    X = pl.build(grid, fx=fx, btc=btc, daily=daily, cot=cot)
    cut = pd.Timestamp("2023-02-01 10:00", tz="UTC")
    fx2 = {k: v.assign(close=np.where(v.index + CA.HOUR > cut, v.close * 1.05, v.close)) for k, v in fx.items()}
    btc2 = btc.assign(close=np.where(btc.index + CA.HOUR > cut, btc.close * 0.9, btc.close))
    daily2 = {k: (s.where(~(CA.daily_available(s.index) > cut), s * 1.1), kind) for k, (s, kind) in daily.items()}
    cot2 = cot.copy()
    m = np.asarray(CA.cot_available(cot2["Date"]) > cut)
    cot2.loc[m, "EUR_Net_Long"] += 99999
    X2 = pl.build(grid, fx=fx2, btc=btc2, daily=daily2, cot=cot2)
    before = np.asarray(grid + CA.HOUR <= cut)
    A, B = X.to_numpy(float)[before], X2.to_numpy(float)[before]
    assert np.array_equal(np.isnan(A), np.isnan(B)) and np.allclose(A[~np.isnan(A)], B[~np.isnan(B)], rtol=0, atol=0)
    assert not np.allclose(np.nan_to_num(X.to_numpy(float)[~before]), np.nan_to_num(X2.to_numpy(float)[~before]))
    assert len(X.columns) == 6 * 4 + 6 + 5 + 3 + 2 + 4


def test_composite_signs():
    idx = pd.date_range("2024-01-02", periods=30, freq="h", tz="UTC")
    up = pd.Series(np.exp(np.linspace(0, 0.03, 30)), index=idx)
    feats = {"gbpusd": CA.hourly_features(1 / up, "fx.gbpusd"), "audusd": CA.hourly_features(1 / up, "fx.audusd"),
             "usdchf": CA.hourly_features(up, "fx.usdchf"), "usdjpy": CA.hourly_features(up, "fx.usdjpy")}
    comp = CA.composite(feats, CA.USD_SIGN, "usd_strength")
    assert (comp["fx.usd_strength.ret_1"].dropna() > 0).all()        # USD rising on every leg -> positive
