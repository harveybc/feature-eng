"""cross_asset_h1 r4: per-source clock option (default = lake behaviour, r3 unchanged), UTC bar-start sources are
NOT re-localized, an additional non-G10 cross (XAUUSD) gets fx.* features without entering the six-pair factors,
and r3's config shape still produces the same output on a fixture."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.plugins import cross_asset_h1 as CA

SPEC = importlib.util.spec_from_file_location("build_store", Path(__file__).parents[1] / "tools/cross_asset_h1/build_store.py")
BS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BS)


def lake_labels(utc_index):
    """Inverse of the lake rule: UTC bar start -> Europe/London wall - 5 h, stamped as if UTC."""
    london = utc_index.tz_convert("Europe/London").tz_localize(None)
    return (london - pd.Timedelta(hours=5)).tz_localize("UTC")


def test_utc_bar_start_keeps_labels_and_lake_rule_relocalizes():
    idx = pd.DatetimeIndex(["2024-01-15 12:00", "2024-07-15 11:00", "2024-03-12 12:00"], tz="UTC")
    df = pd.DataFrame({"timestamp": idx, "close": [1.0, 1.1, 1.2]})
    out = CA.utc_bar_start(df, "timestamp")
    assert list(out.index) == sorted(idx) and list(out.columns) == ["close"]
    lake = pd.DataFrame({"datetime": lake_labels(idx), "close": [1.0, 1.1, 1.2]})
    assert CA.relocalize_lake_clock(lake).index.equals(out.index)           # same instants through the lake rule
    assert CA.SOURCE_CLOCKS["lake"] is CA.relocalize_lake_clock and CA.SOURCE_CLOCKS["utc_bar_start"] is CA.utc_bar_start


def test_r3_config_shape_reads_the_lake_exactly_as_before_and_r4_shape_is_not_relocalized(tmp_path, monkeypatch):
    idx = pd.date_range("2024-03-04", periods=24 * 10, freq="h", tz="UTC")
    close = 1.1 * np.exp(np.random.default_rng(0).normal(0, 5e-4, len(idx)).cumsum())
    # lake file: r3 shape (datetime on the lake clock, no clock key)
    lake_dir = tmp_path / "lake" / "market_data" / "forex" / "g10" / "gbpusd"
    lake_dir.mkdir(parents=True)
    pd.DataFrame({"datetime": lake_labels(idx), "open": close, "high": close, "low": close, "close": close}).to_parquet(lake_dir / "1h.parquet", index=False)
    # recovered file: r4 shape (timestamp = UTC bar start, source column)
    rec_dir = tmp_path / "recovered"
    rec_dir.mkdir()
    pd.DataFrame({"timestamp": idx, "open": close, "high": close, "low": close, "close": close, "m30_exact": True,
                  "source": "independent-hf:abc"}).to_parquet(rec_dir / "gbpusd.parquet", index=False)
    monkeypatch.setenv("LAKE_ROOT_T", str(tmp_path / "lake"))
    monkeypatch.setenv("REC_ROOT_T", str(rec_dir))
    cfg = {"financial_data_root": "${LAKE_ROOT_T}", "roots": {"recovered": "${REC_ROOT_T}"}}
    r3_spec = {"file": "market_data/forex/g10/gbpusd/1h.parquet", "sha256": BS.sha256(lake_dir / "1h.parquet")}
    r4_spec = {"root": "recovered", "file": "gbpusd.parquet", "sha256": BS.sha256(rec_dir / "gbpusd.parquet"),
               "clock": "utc_bar_start", "time_column": "timestamp"}
    r3_df, r3_id = BS.load_hourly_fx(cfg, r3_spec)
    # the r3 code path, verbatim from the previous build_store
    raw = pd.read_parquet(lake_dir / "1h.parquet")
    t = pd.DatetimeIndex(pd.to_datetime(raw["datetime"]))
    old = CA.relocalize_lake_clock(BS._before_test(raw, t), "datetime")
    old = old[old.index < BS.TEST_START]
    pd.testing.assert_frame_equal(r3_df, old)
    assert r3_id["clock"] == "lake" and r3_id["file"] == r3_spec["file"] and BS.pin_key(r3_spec) == r3_spec["file"]
    r4_df, r4_id = BS.load_hourly_fx(cfg, r4_spec)
    assert r4_df.index.equals(idx) and r4_id["clock"] == "utc_bar_start" and r4_id["source_ids"] == ["independent-hf:abc"]
    assert BS.pin_key(r4_spec) == "recovered:gbpusd.parquet"
    pd.testing.assert_series_equal(r4_df["close"], r3_df["close"], check_names=False)      # same instants, same prices
    # a lake-clock relocalization applied to the recovered file would move summer bars by one hour: refused by design
    wrong = CA.relocalize_lake_clock(pd.read_parquet(rec_dir / "gbpusd.parquet"), "timestamp")
    assert not wrong.index.equals(idx)
    with pytest.raises(SystemExit, match="UNKNOWN_SOURCE_CLOCK"):
        BS.load_hourly_fx(cfg, r4_spec | {"clock": "nope"})
    with pytest.raises(SystemExit, match="ROOT_UNRESOLVED"):
        BS.expand_root("${THIS_VARIABLE_IS_NOT_SET_X}")


def _fx_world(pairs, n=24 * 120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-06-03", periods=n, freq="h", tz="UTC")
    return idx, {p: pd.DataFrame({"close": 1.2 * np.exp(rng.normal(0, 4e-4, n).cumsum())}, index=idx) for p in pairs}


def test_factor_pairs_default_keeps_r3_behaviour_and_restricts_factors_for_an_added_cross():
    idx, fx6 = _fx_world(CA.HIST_FX_PAIRS)
    grid = idx[24 * 60:]
    eur = pd.Series(1.1 * np.exp(np.random.default_rng(2).normal(0, 4e-4, len(grid)).cumsum()), index=grid)
    base = CA.Plugin()
    base.set_params(factors=True, factor_fit_start="2019-06-01")
    X6 = base.build(grid, fx=fx6, eur_close=eur)
    explicit = CA.Plugin()
    explicit.set_params(factors=True, factor_fit_start="2019-06-01", factor_pairs=list(CA.HIST_FX_PAIRS))
    pd.testing.assert_frame_equal(explicit.build(grid, fx=fx6, eur_close=eur), X6)       # None == all six
    fx7 = dict(fx6)
    fx7["xauusd"] = pd.DataFrame({"close": 1800 * np.exp(np.random.default_rng(3).normal(0, 1e-3, len(idx)).cumsum())}, index=idx)
    r4 = CA.Plugin()
    r4.set_params(factors=True, factor_fit_start="2019-06-01", factor_pairs=list(CA.HIST_FX_PAIRS))
    X7 = r4.build(grid, fx=fx7, eur_close=eur)
    fac = [c for c in X6.columns if c.startswith(("fac.", "kal.", "fx.usd_strength", "fx.eur_strength"))]
    pd.testing.assert_frame_equal(X7[fac], X6[fac])                                        # factors / composites untouched
    assert r4.factor_info["fx_pca"]["columns"] == list(CA.HIST_FX_PAIRS)
    assert {f"fx.xauusd.ret_{w}" for w in (1, 4, 24)} | {"fx.xauusd.rv_24"} <= set(X7.columns)
    assert len(X7.columns) == len(X6.columns) + 4
    r3_like = CA.Plugin()
    r3_like.set_params(factors=True, factor_fit_start="2019-06-01")                          # None: the 7th enters
    assert r3_like.build(grid, fx=fx7, eur_close=eur).shape[1] == len(X7.columns)
    assert r3_like.factor_info["fx_pca"]["columns"] == list(CA.HIST_FX_PAIRS) + ["xauusd"]


def test_feature_meta_fx_rule_override_only_touches_the_fx_family():
    cols = ["fx.xauusd.ret_1", "fac.fx_f1.ret_1", "btc.ret_1"]
    default = {m["feature_id"]: m["availability_rule"] for m in CA.feature_meta(cols)}
    custom = {m["feature_id"]: m["availability_rule"] for m in CA.feature_meta(cols, "recovered UTC bar start")}
    assert custom["fx.xauusd.ret_1"] == "recovered UTC bar start" and "lake clock" in default["fx.xauusd.ret_1"]
    assert custom["fac.fx_f1.ret_1"] == default["fac.fx_f1.ret_1"] and custom["btc.ret_1"] == default["btc.ret_1"]


def test_r4_config_pins_recovered_sources_with_utc_clock_and_six_pair_factors():
    cfg = json.loads((Path(__file__).parents[1] / "examples/config/cross_asset_eurusd_h1_r4.json").read_text())
    r3 = json.loads((Path(__file__).parents[1] / "examples/config/cross_asset_eurusd_h1_r3.json").read_text())
    assert set(cfg["fx"]) == set(CA.HIST_FX_PAIRS) | {"xauusd"}
    for spec in list(cfg["fx"].values()) + [cfg["clock_gate"]]:
        assert spec["root"] == "recovered" and spec["clock"] == "utc_bar_start" and spec["time_column"] == "timestamp"
        assert len(spec["sha256"]) == 64
    assert cfg["params"]["factor_pairs"] == list(CA.HIST_FX_PAIRS)
    assert {k: v for k, v in cfg["params"].items() if k != "factor_pairs"} == r3["params"]
    for key in ("btc", "daily", "cot", "admitted", "grid_start", "perturbation_cuts"):
        assert cfg[key] == r3[key]
    assert cfg["clock_gate"]["window"] == r3["clock_gate"]["window"]
    text = json.dumps(cfg)
    assert "/home/" not in text and "/mnt/" not in text                                   # no host path committed
