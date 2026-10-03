"""B2 selection table: validation/test cannot change it, association alone never rejects."""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import b2_selection_table as b2  # noqa: E402
from app import train_feature_metrics as tfm  # noqa: E402

N_TRAIN, N_VAL, N_TEST = 400, 60, 60
FEATS = ["f_walk", "f_walk_dup", "f_noise", "f_const", "f_lead", "f_sine", "obv",
         "vol_regime_high", "f_target_next"]
SMALL = {"trailing_window": 120, "min_pairs": 20, "adf_maxlag": 4}


def _make(tmp, seed=0, val_scale=1.0):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    dates = pd.date_range("2020-01-01", periods=n, freq="4h")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    walk = np.cumsum(rng.normal(size=n))
    df = pd.DataFrame({"DATE_TIME": dates, "CLOSE": close, "f_walk": walk,
                       "f_walk_dup": 2 * walk + 3, "f_noise": rng.normal(size=n),
                       "f_const": 1.0, "f_lead": rng.normal(size=n),
                       "f_sine": np.sin(2 * np.pi * np.arange(n) / 24),
                       "obv": np.cumsum(rng.normal(size=n)) * 50,
                       "vol_regime_high": (rng.random(n) > 0.7).astype(float),
                       "f_target_next": rng.normal(size=n)})
    df.loc[: int(0.3 * N_TRAIN), "f_lead"] = np.nan
    df.loc[N_TRAIN:, FEATS + ["CLOSE"]] *= val_scale  # validation/test rows only
    df.loc[N_TRAIN:, "f_noise"] = np.nan if val_scale != 1.0 else df.loc[N_TRAIN:, "f_noise"]
    csv = os.path.join(tmp, "d.csv")
    df.to_csv(csv, index=False)
    man = os.path.join(tmp, "m.json")
    with open(man, "w") as fh:
        json.dump({"feature_columns": FEATS, "rows": n, "sha256": "x",
                   "splits": {"train_start": str(dates[0]),
                              "train_end": str(dates[N_TRAIN - 1])}}, fh)
    return csv, man


def _pipeline(tmp, name, params=None, **kw):
    d = tmp / name
    d.mkdir()
    csv, man = _make(str(d), **kw)
    tfm.run(csv, man, str(d / "m"), SMALL)
    res = b2.run(str(d / "m" / "train_feature_metrics.json"), csv, man, str(d / "o"),
                 dict({"min_pairs": 20}, **(params or {})))
    return res, d


def test_validation_and_test_cannot_change_table(tmp_path):
    a, da = _pipeline(tmp_path, "a")
    b, db = _pipeline(tmp_path, "b", val_scale=9.0)
    assert (da / "o" / "B2_SELECTION_TABLE_v1.json").read_bytes() == \
        (db / "o" / "B2_SELECTION_TABLE_v1.json").read_bytes()
    assert (da / "o" / "B2_SELECTION_TABLE_v1.csv").read_bytes() == \
        (db / "o" / "B2_SELECTION_TABLE_v1.csv").read_bytes()
    assert a["fit_rows"] == {"start": 0, "stop": N_TRAIN}


def test_statuses_and_reasons(tmp_path):
    r, _ = _pipeline(tmp_path, "a")
    rows = {x["feature"]: x for x in r["rows"]}
    assert [x["feature"] for x in r["rows"]] == FEATS and r["control_list_all"] == FEATS
    assert rows["f_const"]["status"] == "REJECT" and "R_CONSTANT" in rows["f_const"]["reasons"]
    assert rows["f_lead"]["status"] == "REJECT" and "R_LEADING_MISSING" in rows["f_lead"]["reasons"]
    assert rows["obv"]["reasons"] == ["R_ORIGIN_DEPENDENT_CUMSUM"]
    assert rows["f_target_next"]["status"] == "REJECT"
    assert rows["vol_regime_high"]["status"] == "PENDING"
    assert rows["vol_regime_high"]["reasons"] == ["P_WARMUP_ENCODED_AS_ZERO"]
    # exact monotone duplicate: one representative (earlier manifest index), the other pending
    assert rows["f_walk"]["status"] == "SURVIVOR"
    assert rows["f_walk_dup"]["status"] == "PENDING"
    assert rows["f_walk_dup"]["reasons"] == ["P_REDUNDANT"]
    assert rows["f_walk_dup"]["representative"] == "f_walk"
    assert rows["f_walk_dup"]["rho_to_representative"] == pytest.approx(1.0)
    assert rows["f_noise"]["status"] == "SURVIVOR" and rows["f_sine"]["status"] == "SURVIVOR"
    assert sum(r["counts"].values()) == len(FEATS)
    for x in r["rows"]:
        assert "assoc_spearman_h6" in x and "assoc_pearson_h36" in x


def _inputs(tmp_path):
    d = tmp_path / "i"
    d.mkdir()
    csv, man = _make(str(d))
    mt = tfm.run(csv, man, str(d / "m"), SMALL)
    df = tfm.load_train_frame(csv, tfm.load_contract(man), dict(tfm.DEFAULT_PARAMS, **SMALL))
    return mt, df


def _statuses(t):
    return [(x["feature"], x["status"], tuple(x["reasons"])) for x in t["rows"]]


@pytest.mark.parametrize("value", [0.0, 0.99, -0.99, None])
def test_association_alone_never_changes_status(tmp_path, value):
    mt, df = _inputs(tmp_path)
    base = b2.build(mt, df, FEATS, {"min_pairs": 20})
    mt2 = copy.deepcopy(mt)
    for r in mt2["rows"]:
        for k in list(r):
            if k.startswith("target_pearson_h") or k.startswith("target_spearman_h"):
                r[k] = value
    alt = b2.build(mt2, df, FEATS, {"min_pairs": 20})
    assert _statuses(alt) == _statuses(base)


def test_zero_association_feature_is_not_rejected(tmp_path):
    mt, df = _inputs(tmp_path)
    for r in mt["rows"]:
        if r["feature"] == "f_noise":
            for h in (6, 12, 18, 24, 30, 36):
                r[f"target_spearman_h{h}"] = 0.0
                r[f"target_pearson_h{h}"] = 0.0
    t = {x["feature"]: x for x in b2.build(mt, df, FEATS, {"min_pairs": 20})["rows"]}
    assert t["f_noise"]["status"] == "SURVIVOR"
    assert not any("ASSOC" in c for x in t.values() for c in x["reasons"])


def test_digest_mismatch_refused(tmp_path):
    mt, df = _inputs(tmp_path)
    df = df.copy()
    df.loc[5, "f_noise"] += 1.0
    with pytest.raises(b2.B2Error):
        b2.build(mt, df, FEATS, {"min_pairs": 20})


def test_threshold_and_parameter_digest(tmp_path):
    mt, df = _inputs(tmp_path)
    a = b2.build(mt, df, FEATS, {"min_pairs": 20})
    b = b2.build(mt, df, FEATS, {"min_pairs": 20, "redundancy_threshold": 0.999999})
    assert a["parameter_digest"] != b["parameter_digest"]
    c = b2.build(mt, df, FEATS, {"min_pairs": 20, "redundancy_threshold": 1.0 + 1e-9})
    assert {x["feature"]: x["status"] for x in c["rows"]}["f_walk_dup"] == "SURVIVOR"
