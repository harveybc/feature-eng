"""Lane A (EURUSD PS0/PS1) contract tests on synthetic bytes only."""
import numpy as np
import pandas as pd
import pytest

from tools.eurusd_ps import contract as C
from tools.eurusd_ps import features as F
from tools.eurusd_ps import profile as P
from tools.eurusd_ps import sources as S
from tools.eurusd_ps import targets as T
from tools.eurusd_ps import variants as V
from tools.eurusd_ps.asof import asof_count_sum, asof_last, asof_price


def synth_5m_ny(start="2023-10-01", end="2025-03-01", seed=0, poison_from=None):
    """FX-like 5m bars, NY wall clock, bar start; market closed Fri 17:00 -> Sun 17:00 NY."""
    idx = pd.date_range(start, end, freq="5min")
    wd, hr = idx.dayofweek, idx.hour
    closed = (wd == 5) | ((wd == 4) & (hr >= 17)) | ((wd == 6) & (hr < 17))
    idx = idx[~closed]
    rng = np.random.default_rng(seed)
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 3e-4, len(idx))))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + 1e-4); lo = np.minimum(o, c) * (1 - 1e-4)
    df = pd.DataFrame({"timestamp": idx.tz_localize("UTC"), "open": o, "high": hi, "low": lo, "close": c})
    if poison_from is not None:
        m = df["timestamp"] >= pd.Timestamp(poison_from, tz="UTC")
        df.loc[m, ["open", "high", "low", "close"]] = 1e6
    return df


def test_clock_inference_recovers_ny_bar_start_and_refuses_scramble():
    df = synth_5m_ny("2022-01-01", "2023-06-01")
    r = S.infer_fx_clock(df["timestamp"])
    assert r["status"] == "DETERMINED" and r["clock"] == "NEW_YORK_LOCAL/BAR_START"
    utc = pd.DatetimeIndex(df["timestamp"]).tz_localize(None).tz_localize("America/New_York", ambiguous="NaT",
                                                                          nonexistent="NaT").tz_convert("UTC")
    r2 = S.infer_fx_clock(pd.Series(utc[~utc.isna()] + pd.Timedelta(minutes=5)))
    assert r2["clock"] == "UTC/BAR_END"
    rng = np.random.default_rng(1)
    scr = df["timestamp"] + pd.to_timedelta(rng.integers(0, 12, len(df)), unit="h")
    assert S.infer_fx_clock(scr)["status"] == "UNDETERMINED"


def test_asof_join_uses_only_available_values_ties_admitted_and_max_age():
    t = pd.DatetimeIndex(pd.to_datetime(["2020-01-01 10:00", "2020-01-01 11:00", "2020-01-03 11:00"], utc=True))
    av = pd.Series(pd.to_datetime(["2020-01-01 10:00", "2020-01-01 10:30", "2020-01-01 11:00:01"], utc=True, format="ISO8601"))
    v = pd.Series([1.0, 2.0, 3.0])
    out, age = asof_last(t, av, v, max_age_h=24)
    assert out[0] == 1.0 and age[0] == 0.0          # tie at equality admitted
    assert out[1] == 2.0                             # 11:00:01 not yet available at 11:00
    assert np.isnan(out[2])                          # older than 24 h -> NaN, not carried forward
    cnt, s = asof_count_sum(t, av, v, window_h=1)
    assert cnt.tolist() == [1, 1, 0] and s.tolist() == [1, 2, 0]  # window (t-1h, t]: left edge excluded


def test_asof_join_future_perturbation_does_not_change_past_rows():
    t = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    av = pd.Series(pd.date_range("2020-01-01 00:30", periods=50, freq="3h", tz="UTC"))
    v = pd.Series(np.arange(50.0))
    a0, _ = asof_last(t, av, v)
    v2 = v.copy(); v2[av > t[100]] = 999.0
    a1, _ = asof_last(t, av, v2)
    assert np.array_equal(a0[:101], a1[:101], equal_nan=True)
    assert not np.array_equal(a0, a1, equal_nan=True)  # the perturbation is real


def test_price_and_event_features_invariant_to_future_perturbation():
    df = synth_5m_ny("2022-11-01", "2023-03-01")
    b5 = S.lake_5m_to_utc(df, "NEW_YORK_LOCAL/BAR_START")
    h = S.hourly_from_5m(b5)
    cut = h.index[len(h) // 2]
    b5p = b5.copy(); m = b5p["end_utc"] > cut
    b5p.loc[m, ["open", "high", "low", "close"]] *= 1.05
    hp = S.hourly_from_5m(b5p)
    d = h.index[h.index <= cut]
    f0, _ = F.price_features(h, d); f1, _ = F.price_features(hp, d)
    pd.testing.assert_frame_equal(f0, f1)
    ev = pd.DataFrame({"avail_utc": pd.date_range("2022-11-02", periods=60, freq="37h", tz="UTC")})
    ev["group"] = "USD"; ev["volatility"] = F.TIERS["high"]; ev["key"] = "United States|NFP"
    ev["surprise_z"] = np.linspace(-2, 2, 60); ev["revision_z"] = 0.5
    e0, _, _ = F.event_features(ev, d, top_k=1)
    ev2 = ev.copy(); ev2.loc[ev2["avail_utc"] > cut, "surprise_z"] = 50.0
    e1, _, _ = F.event_features(ev2, d, top_k=1)
    pd.testing.assert_frame_equal(e0, e1)


def test_loader_never_returns_test_or_validation_period(tmp_path):
    df = synth_5m_ny("2023-10-01", "2025-03-01", poison_from="2024-01-01")
    p = tmp_path / "x.parquet"; df.to_parquet(p)
    b5, clk = S.load_lake_5m(str(p))
    assert clk["applied"] == "NEW_YORK_LOCAL/BAR_START"
    assert b5["end_utc"].max() <= C.READ_END
    assert b5["close"].max() < 1e5                      # no poisoned (post-TRAIN) value reached the frame
    h = S.hourly_from_5m(b5)
    d = h.index[(h.index >= pd.Timestamp("2023-12-20", tz="UTC")) & (h.index < C.TRAIN_END)]
    tg = T.build_targets(h, b5, d)
    late = d + pd.Timedelta(hours=144) >= C.READ_END
    assert tg.loc[late, "Y_l_144h"].isna().all()       # censored, never filled
    assert (tg.loc[late, "Y_b_l144_state"] == "CENSORED").all()
    with pytest.raises(ValueError):
        C.inner_folds(pd.DatetimeIndex([C.TRAIN_END]))


def test_purge_is_derived_from_supports_and_folds_respect_it():
    pg = C.derive_purge({"a": 720})
    assert pg["label_purge_h"] == max(C.Y_L_HOURS + [s["timeout_h"] for s in C.Y_B_SPECS]) == 144
    assert pg["feature_embargo_h"] == 0 and pg["max_feature_support_h"] == 720
    t = pd.date_range("2012-05-01", "2023-12-31 23:00", freq="h", tz="UTC")
    for f in C.inner_folds(t):
        tr_last = t[f["train_rows"][1] - 1]; va_first = t[f["val_rows"][0]]
        assert tr_last + pd.Timedelta(hours=144) < va_first
        assert t[f["val_rows"][1] - 1] < C.TRAIN_END


def test_barrier_labels_tp_sl_timeout_ambiguous():
    s = pd.date_range("2020-01-06 00:00", periods=12 * 10, freq="5min", tz="UTC")
    c = np.full(len(s), 1.0)
    b5 = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "start_utc": s, "end_utc": s + pd.Timedelta(minutes=5)})
    b5.loc[30, "high"] = 1.5                      # TP touch in the 3rd hour
    h = S.hourly_from_5m(b5)
    h["close"] = 1.0
    d = pd.DatetimeIndex([h.index[0]])
    import tools.eurusd_ps.targets as TT
    orig = TT.sigma_t
    TT.sigma_t = lambda hh: pd.Series(0.01, index=hh.index)
    try:
        r = TT.build_targets(h, b5, d)
        assert r["Y_b_s6_state"].iloc[0] == "TP" and r["Y_b_s6"].iloc[0] == 1.0
        b5.loc[30, "low"] = 0.5                   # both inside one 5m bar
        r = TT.build_targets(h, b5, d)
        assert r["Y_b_s6_state"].iloc[0] == "AMBIGUOUS" and np.isnan(r["Y_b_s6"].iloc[0])
        b5.loc[30, ["high", "low"]] = 1.0
        r = TT.build_targets(h, b5, d)
        assert r["Y_b_s6_state"].iloc[0] == "TIMEOUT" and r["Y_b_s6"].iloc[0] == 0.0
    finally:
        TT.sigma_t = orig


def test_coverage_states_every_metric_once_failure_is_not_zero(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=3000, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    for x, expect in ((pd.Series(np.nan, index=idx), "NOT_APPLICABLE"), (pd.Series(1.0, index=idx), "NOT_APPLICABLE"),
                      (pd.Series(rng.normal(size=3000), index=idx), "MEASURED")):
        cells = P.profile_feature("f", x)
        assert sorted(c["metric"] for c in cells) == sorted(P.METRICS)
        assert all(c["state"] in ("MEASURED", "FAILED", "NOT_APPLICABLE", "PENDING") for c in cells)
        assert {c["metric"]: c["state"] for c in cells}["acf"] == expect
    def boom(*a, **k):
        raise RuntimeError("forced")
    monkeypatch.setattr(P, "adfuller", boom)
    cells = {c["metric"]: c for c in P.profile_feature("f", pd.Series(rng.normal(size=3000), index=idx))}
    assert cells["adf"]["state"] == "FAILED" and cells["adf"]["value"] is None and "forced" in cells["adf"]["reason"]
    monkeypatch.setattr(P, "_SM", False)
    cells = {c["metric"]: c for c in P.profile_feature("f", pd.Series(rng.normal(size=3000), index=idx))}
    assert cells["kpss"]["state"] == "PENDING" and "DEPENDENCY_MISSING" in cells["kpss"]["reason"]
    cov = P.coverage_rows("f", pd.Series(np.r_[np.full(1500, np.nan), np.ones(1500)], index=idx),
                          [{"name": "k", "train_rows": [0, 1500], "val_rows": [1500, 3000]}])
    assert [r["state"] for r in cov] == ["NO_COVERAGE", "FULL"]


def test_transform_variant_prefix_test_separates_causal_from_global():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 1e-3, 1200))
    probes = [400, 700, 1000]
    assert V.prefix_test(V.IMPLS["tv.kalman_local_level_filter"], x, probes)["status"] == "PREFIX_INVARIANT_MEASURED"
    assert V.prefix_test(V.IMPLS["tv.kalman_local_level_smoother"], x, probes)["status"] == "PREFIX_VIOLATION_MEASURED"
    assert V.prefix_test(V.IMPLS["tv.hilbert_trailing_lastsample"], x, probes)["status"] == "PREFIX_INVARIANT_MEASURED"
    assert V.prefix_test(V.IMPLS["tv.hilbert_global"], x, probes)["status"] == "PREFIX_VIOLATION_MEASURED"


def test_event_features_are_nan_outside_source_coverage_not_zero():
    d = pd.date_range("2021-04-01", "2021-05-15", freq="h", tz="UTC")
    ev = pd.DataFrame({"avail_utc": pd.date_range("2021-04-01 12:30", "2021-04-26 12:30", freq="12h", tz="UTC")})
    ev["group"] = "USD"; ev["volatility"] = F.TIERS["high"]; ev["key"] = "United States|NFP"
    ev["surprise_z"] = 1.0; ev["revision_z"] = 0.0
    vw = [(pd.Timestamp("2021-04-01", tz="UTC"), pd.Timestamp("2021-04-27", tz="UTC"))]
    f, meta, _ = F.event_features(ev, d, top_k=1, valid_windows=vw)
    after = d > vw[0][1]
    assert f.loc[after].isna().all().all()                       # no invented 0 counts / 168 h ages
    early = d - pd.Timedelta(hours=24) < vw[0][0]
    assert f.loc[early, "ev.USD.high.count_24h"].isna().all()
    assert f.loc[(~early) & (~after), "ev.USD.high.count_24h"].notna().all()


def test_daily_covariates_respect_availability_and_read_end(tmp_path):
    from tools.eurusd_ps import covariates as CV
    dates = pd.date_range("2023-12-01", "2024-02-01", freq="B", tz="America/Chicago")
    df = pd.DataFrame({"Date": dates, "Close": np.where(dates >= pd.Timestamp("2024-01-01", tz="America/Chicago"), 1e9, 10.0 + np.arange(len(dates)))})
    p = tmp_path / "y.parquet"; df.to_parquet(p)
    s = CV.yahoo_series(str(p))
    assert s["avail_utc"].max() < C.READ_END and s["close"].max() < 1e8      # test/validation values never read
    d = pd.date_range("2023-12-04 00:00", "2023-12-06 00:00", freq="h", tz="UTC")
    f, _ = CV.daily_features("yh.x", s, "close", d, "yahoo", "lic", "src", levels=True)
    # the bar of trading date 2023-12-04 becomes usable only at 2023-12-05 00:00 UTC
    assert f.loc[pd.Timestamp("2023-12-04 23:00", tz="UTC"), "yh.x.level"] == s.loc[s["date"] == "2023-12-01", "close"].iloc[0]
    assert f.loc[pd.Timestamp("2023-12-05 00:00", tz="UTC"), "yh.x.level"] == s.loc[s["date"] == "2023-12-04", "close"].iloc[0]


def test_calendar_columns_are_selector_episode_sources_not_model_inputs():
    from tools.eurusd_ps import inventory as INV
    d = pd.date_range("2021-04-01", "2021-04-10", freq="h", tz="UTC")
    ev = pd.DataFrame({"avail_utc": pd.date_range("2021-04-01 12:30", periods=10, freq="12h", tz="UTC")})
    ev["group"] = "EUR"; ev["volatility"] = F.TIERS["high"]; ev["key"] = "Germany|GDP"; ev["surprise_z"] = 1.0; ev["revision_z"] = 0.0
    _, meta, _ = F.event_features(ev, d, top_k=1)
    assert meta and all(m["role"] == "SELECTOR_EPISODE_SOURCE" for m in meta)
    _, cm = F.calendar_features(d)
    assert all(m["role"] == "feature" for m in cm)          # known time encodings unaffected
    for fam in ("calendar_archive", "fxmacrodata_announcements", "fxmacrodata_calendar", "pit_capture"):
        assert INV.source_role(fam) == "SELECTOR_EPISODE_SOURCE"
    assert INV.source_role("yahoo_daily") == "MODEL_INPUT_CANDIDATE_SOURCE"
