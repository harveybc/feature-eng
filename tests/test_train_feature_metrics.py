"""Train-only feature metrics: changing validation/test cannot change the output."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import train_feature_metrics as tfm  # noqa: E402

N_TRAIN, N_VAL, N_TEST = 400, 60, 60
FEATS = ["f_walk", "f_noise", "f_const", "f_missing", "f_sine"]
SMALL = {"trailing_window": 120, "min_pairs": 20, "adf_maxlag": 4}


def _make(tmp, seed=0, val_scale=1.0):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    dates = pd.date_range("2020-01-01", periods=n, freq="4h")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"DATE_TIME": dates, "CLOSE": close,
                       "f_walk": np.cumsum(rng.normal(size=n)),
                       "f_noise": rng.normal(size=n), "f_const": 1.0,
                       "f_missing": rng.normal(size=n),
                       "f_sine": np.sin(2 * np.pi * np.arange(n) / 24)})
    df.loc[:30, "f_missing"] = np.nan
    df.loc[N_TRAIN:, ["f_walk", "f_noise", "CLOSE"]] *= val_scale  # validation/test only
    csv = os.path.join(tmp, "d.csv")
    df.to_csv(csv, index=False)
    man = os.path.join(tmp, "m.json")
    with open(man, "w") as fh:
        json.dump({"feature_columns": FEATS, "rows": n, "sha256": "x",
                   "splits": {"train_start": str(dates[0]),
                              "train_end": str(dates[N_TRAIN - 1])}}, fh)
    return csv, man


def _run(tmp, name, **kw):
    csv, man = _make(str(tmp), **kw)
    return tfm.run(csv, man, str(tmp / name), SMALL)


def test_validation_and_test_cannot_change_output(tmp_path):
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b", val_scale=7.5)
    assert a == b
    assert (tmp_path / "a" / "train_feature_metrics.json").read_bytes() == \
        (tmp_path / "b" / "train_feature_metrics.json").read_bytes()
    assert a["fit_rows"] == {"start": 0, "stop": N_TRAIN}


def test_changing_train_changes_digest(tmp_path):
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b", seed=1)
    assert a["data_digest"] != b["data_digest"]
    assert a["rows"][0]["row_id"] != b["rows"][0]["row_id"]


def test_rows_after_boundary_are_not_parsed(tmp_path):
    csv, man = _make(str(tmp_path))
    with open(csv, "a") as fh:
        fh.write("2030-01-01 00:00:00,zzz,zzz,zzz,zzz,zzz,zzz\n")
    res = tfm.run(csv, man, str(tmp_path / "o"), SMALL)
    assert res["fit_rows"]["stop"] == N_TRAIN


def test_typed_rows_and_values(tmp_path):
    r = _run(tmp_path, "a")
    rows = {x["feature"]: x for x in r["rows"]}
    assert set(rows) == set(FEATS)
    for x in r["rows"]:
        assert x["fit_scope"] == "train_only"
        assert x["data_digest"] == r["data_digest"]
        assert x["parameter_digest"] == r["parameter_digest"]
        for h in (6, 12, 18, 24, 30, 36):
            assert f"target_spearman_h{h}" in x and f"target_pearson_h{h}" in x
        for lag in (1, 6, 24):
            assert f"acf_lag{lag}" in x
    assert rows["f_const"]["is_constant"] and rows["f_const"]["variance"] == 0
    assert rows["f_const"]["adf_status"] == "skipped_constant_or_short"
    assert rows["f_missing"]["n_missing"] == 31 and rows["f_missing"]["leading_missing"] == 31
    assert rows["f_walk"]["acf_lag1"] > 0.9
    assert abs(rows["f_noise"]["acf_lag1"]) < 0.3
    assert abs(rows["f_sine"]["dominant_trailing_period_bars"] - 24) < 1.5
    assert rows["f_sine"]["spectral_entropy"] < rows["f_noise"]["spectral_entropy"]
    if tfm.HAVE_STATSMODELS:
        assert rows["f_walk"]["adf_status"] == "ok" and rows["f_walk"]["kpss_status"] == "ok"


def test_protected_rows_refused():
    df = pd.DataFrame({"CLOSE": [1.0] * 5, "a": [1.0] * 5})
    assert tfm.PROTECTED_TEST_ROWS == (15895, 18085)
    with pytest.raises(tfm.ScopeError):
        tfm.materialize(df, ["a"], row_offset=15893)


def test_boundary_guard_on_load(tmp_path, monkeypatch):
    csv, man = _make(str(tmp_path))
    monkeypatch.setattr(tfm, "PROTECTED_TEST_ROWS", (100, 200))
    with pytest.raises(tfm.ScopeError):
        tfm.run(csv, man, str(tmp_path / "o"), SMALL)
