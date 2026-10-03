"""PS2 reversible priority: isolation, association never rejects, exploration recorded, FS16."""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import ps2_selection as ps2  # noqa: E402

N_TRAIN, N_VAL, N_TEST = 2600, 400, 400
SMALL = {"targets": {"Y_s": [1, 3], "Y_l": [24], "Y_b": [24]}, "n_null": 19,
         "null_min_shift_rows": 60, "min_fit_rows": 200, "min_eval_rows": 80}
FEATS = ["sig_level", "sig_dup", "syn_a", "syn_b", "noise_1", "noise_2", "noise_3", "noise_4",
         "noise_5", "noise_6", "f_const", "f_empty", "next_return_label", "bad_declared"]
DOMAINS = {"syn_a": "pair", "syn_b": "pair"}
DECLARED = {"bad_declared": ("T_INVALID_DECLARED", "PS1 marked invalid vintage")}


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    # hourly stamps with a weekend gap so elapsed time differs from row count
    t = pd.date_range("2020-01-06", periods=int(n * 1.4), freq="h", tz="UTC")
    t = t[t.dayofweek < 5][:n]
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.9 * s[i - 1] + rng.normal()
    a, b = rng.normal(size=n), rng.normal(size=n)
    r = np.zeros(n)
    r[1:] = 0.002 * (0.35 * s[:-1] / 2.3 + 0.9 * a[:-1] * b[:-1]) + 0.002 * rng.normal(size=n - 1)
    price = 1.1 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"ts": t, "close": price, "sig_level": s, "sig_dup": 2 * s + 1,
                       "syn_a": a, "syn_b": b, "f_const": 3.0, "f_empty": np.nan,
                       "next_return_label": rng.normal(size=n),
                       "bad_declared": rng.normal(size=n)})
    for k in range(1, 7):
        df[f"noise_{k}"] = rng.normal(size=n)
    return df


def _train_end(df):
    return df["ts"].iloc[N_TRAIN]


def _run(df, tmp, name, params=None, write=True):
    b = ps2.batch_from_frame(df, "ts", "close", FEATS, _train_end(df), DOMAINS, DECLARED, "t")
    res = ps2.build(b, dict(SMALL, **(params or {})))
    out = tmp / name
    w = ps2.write(res, str(out)) if write else None
    return res, out, w


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("base")
    return _run(_frame(), tmp, "a") + (tmp,)


def _bytes(d):
    return {f: (d / f).read_bytes() for f in sorted(os.listdir(d)) if f != "ps2_cost.json"}


# ------------------------------------------------------------------ isolation

def test_validation_and_test_rows_cannot_change_outputs(base, tmp_path):
    res, out, w, _ = base
    df = _frame()
    rng = np.random.default_rng(7)
    m = df.index >= N_TRAIN
    df.loc[m, "close"] = df.loc[m, "close"] * np.exp(rng.normal(0, 0.2, m.sum()))
    for f in FEATS:
        if f not in ("f_const", "f_empty"):
            df.loc[m, f] = rng.normal(size=m.sum()) * 1e3
    df.loc[m & (df.index % 3 == 0), "noise_1"] = np.nan
    _, out2, w2 = _run(df, tmp_path, "b")
    assert _bytes(out) == _bytes(out2)
    assert w["manifest_canonical_sha256"] == w2["manifest_canonical_sha256"]
    # dropping validation/test rows entirely changes nothing either
    _, out3, _ = _run(_frame().iloc[: N_TRAIN + 5].copy(), tmp_path, "c")
    assert _bytes(out) == _bytes(out3)


def test_future_perturbation_inside_train_cannot_change_earlier_fold(base, tmp_path):
    res, _, _, _ = base
    f1 = res["folds"][0]
    df = _frame()
    cut = f1["eval_end_row"]  # rows after the first inner fold's evaluation end
    rng = np.random.default_rng(3)
    m = (df.index > cut) & (df.index < N_TRAIN)
    df.loc[m, "close"] = df.loc[m, "close"] * np.exp(rng.normal(0, 0.3, m.sum()))
    df.loc[m, "sig_level"] = rng.normal(size=m.sum())
    res2, _, _ = _run(df, tmp_path, "p", write=False)
    key = lambda c: (c["feature"], c["target"], c["horizon"])  # noqa: E731
    a = {key(c): c for c in res["cells"] if c["fold"] == f1["fold"]}
    b = {key(c): c for c in res2["cells"] if c["fold"] == f1["fold"]}
    assert a == b
    ga = [g for g in res["groups"] if g["fold"] == f1["fold"]]
    gb = [g for g in res2["groups"] if g["fold"] == f1["fold"]]
    assert ga == gb
    assert res["fold_clusters"][f1["fold"]] == res2["fold_clusters"][f1["fold"]]
    # and the perturbation is real: a later fold does change
    assert [c for c in res["cells"] if c["fold"] != f1["fold"]] != \
        [c for c in res2["cells"] if c["fold"] != f1["fold"]]


def test_fold_fit_labels_end_before_eval_and_eval_labels_inside(base):
    res, _, _, _ = base
    df = _frame()
    b = ps2.batch_from_frame(df, "ts", "close", FEATS, _train_end(df), DOMAINS, DECLARED)
    p = dict(ps2.DEFAULT_PARAMS, **SMALL)
    tg = ps2.build_targets(b.ts, b.price, p)
    for fd in ps2.inner_folds(b.ts, p):
        for (tn, h), (y, end) in tg.items():
            fr, er = ps2.fold_rows(fd, b.ts, y, end)
            assert end[fr].max() < fd["eval_start_time"]
            assert end[er].max() <= fd["eval_end_time"]
            assert fr.max() < er.min()
    assert np.all(np.isfinite(b.X[:, FEATS.index("sig_level")]))
    assert b.ts[-1] < int(ps2.epoch_seconds([_train_end(df)])[0])


def test_labels_use_elapsed_time_and_epoch_seconds():
    t = pd.to_datetime(["2020-01-03 22:00", "2020-01-03 23:00", "2020-01-06 00:00",
                        "2020-01-06 01:00"], utc=True)
    ts = ps2.epoch_seconds(t)
    assert ts[1] - ts[0] == 3600  # seconds, not kiloseconds (pandas 3 unit defect guard)
    price = np.array([1.0, 2.0, 4.0, 8.0])
    tg = ps2.build_targets(ts, price, {"targets": {"Y_s": [1, 2]}, "barrier": {}})
    y1, _ = tg[("Y_s", 1)]
    assert y1[0] == pytest.approx(np.log(2.0))
    assert y1[1] == 0.0  # t+1h falls in the weekend: as-of price is the Friday bar itself
    assert y1[2] == pytest.approx(np.log(2.0))
    assert np.isnan(y1[3])  # support leaves the data: no label
    y2, _ = tg[("Y_s", 2)]
    assert y2[0] == pytest.approx(np.log(2.0)) and np.isnan(y2[2])


# ------------------------------------------------------------------ association never rejects

def test_technical_reject_only_from_technical_codes(base):
    res, _, _, _ = base
    rej = {r["feature"] for r in res["status"] if r["status"] == "TECHNICAL_REJECT"}
    assert rej == {"f_const", "f_empty", "next_return_label", "bad_declared"}
    for r in res["status"]:
        if r["status"] == "TECHNICAL_REJECT":
            assert r["reasons"] and all(c in ps2.TECHNICAL_CODES for c in r["reasons"])
        else:
            assert not any(c.startswith("T_") for c in r["reasons"])
    assert res["control_all_admissible"] == [f for f in FEATS if f not in rej]


def test_association_alone_never_rejects(tmp_path, monkeypatch):
    df = _frame()
    # a target with no relation to any feature: every association is null
    df["close"] = 1.1 * np.exp(np.cumsum(np.random.default_rng(99).normal(0, 0.002, len(df))))
    res, _, _ = _run(df, tmp_path, "null", write=False)
    rej = {r["feature"] for r in res["status"] if r["status"] == "TECHNICAL_REJECT"}
    assert rej == {"f_const", "f_empty", "next_return_label", "bad_declared"}
    # force extreme association results: all p = 1, then all p = 0; reject set never moves
    for forced in (1.0, 1.0 / 20):
        monkeypatch.setattr(ps2, "emp_p", lambda o, n, v=forced: v)
        r2, _, _ = _run(_frame(), tmp_path, f"forced_{forced}", write=False)
        assert {r["feature"] for r in r2["status"] if r["status"] == "TECHNICAL_REJECT"} == rej
        monkeypatch.undo()
    # a pure-noise feature is low priority or exploration, never rejected
    for r in res["status"]:
        if r["feature"].startswith("noise_"):
            assert r["status"] in ("PROVISIONAL_LOW_PRIORITY", "EXPLORATION",
                                   "PROVISIONAL_SURVIVOR")
            if r["status"] != "PROVISIONAL_SURVIVOR":
                assert "reincorporation" in r


def test_low_priority_carries_reasons_and_reincorporation(base):
    res, _, _, _ = base
    for r in res["status"]:
        assert r["status"] in ps2.STATUSES
        assert r["reasons"], r
        if r["status"] in ("PROVISIONAL_LOW_PRIORITY", "EXPLORATION"):
            assert r["reincorporation"] == ps2.REINCORPORATION


# ------------------------------------------------------------------ exploration

def test_exploration_sample_non_empty_and_recorded(base):
    res, out, _, _ = base
    ex = json.loads((out / "ps2_exploration.json").read_text())
    assert ex["sample"], "exploration sample must be non-empty"
    assert ex["seed"] == ps2.DEFAULT_PARAMS["exploration_seed"] and "rule" in ex
    assert set(ex["u"]) == set(res["control_all_admissible"])
    assert any(r["status"] == "EXPLORATION" for r in res["status"])
    for r in res["status"]:
        assert r["exploration_sample"] == (r["feature"] in ex["sample"])
    man = json.loads((out / "ps2_manifest.json").read_text())
    assert man["exploration_sample"] == ex["sample"]
    ready = json.loads((out / "READY").read_text())
    assert len(ready["manifest_canonical_sha256"]) == 64


def test_exploration_stage1_never_reads_the_ranking(tmp_path, base):
    res, _, _, _ = base
    df = _frame()
    df["close"] = df["close"].iloc[::-1].to_numpy()  # a different target and ranking
    r2, _, _ = _run(df, tmp_path, "rank", write=False)
    assert r2["exploration"]["stage1_drawn"] == res["exploration"]["stage1_drawn"]


def test_exploration_topup_when_draw_misses_pool():
    adm = [f"f{i}" for i in range(10)]
    p = dict(ps2.DEFAULT_PARAMS, exploration_fraction=0.0, exploration_min=2)
    sample, drawn, topup, rule = ps2.exploration_draw(adm, ["f3", "f7", "f9"], p)
    assert drawn == [] and len(topup) == 2 and set(topup) <= {"f3", "f7", "f9"}
    assert sample == sorted(topup)


# ------------------------------------------------------------------ FS16 synergy, redundancy

def test_fs16_pair_useful_only_jointly_resurfaces(base):
    res, _, _, _ = base
    st = {(r["feature"], r["target"], r["horizon"]): r for r in res["status"]}
    a, b = st[("syn_a", "Y_s", 1)], st[("syn_b", "Y_s", 1)]
    for r, other in ((a, "syn_b"), (b, "syn_a")):
        assert "S_SYNERGY_PAIR" in r["reasons"], r
        assert r["status"] == "PROVISIONAL_SURVIVOR"
        assert other in r["synergy_partners"]
        # individually they carry no robust utility: the pair is what resurfaces them
        assert "S_OOF_UTILITY" not in r["reasons"]


def test_signal_survives_and_duplicate_is_grouped_not_dropped(base):
    res, _, _, _ = base
    st = {(r["feature"], r["target"], r["horizon"]): r for r in res["status"]}
    s, d = st[("sig_level", "Y_s", 1)], st[("sig_dup", "Y_s", 1)]
    assert s["status"] == "PROVISIONAL_SURVIVOR" or d["status"] == "PROVISIONAL_SURVIVOR"
    assert sorted(s["dependence_cluster"]) == ["sig_dup", "sig_level"]
    assert d["status"] != "TECHNICAL_REJECT" and s["status"] != "TECHNICAL_REJECT"
    for fc in res["fold_clusters"].values():
        assert ["sig_dup", "sig_level"] in fc["dependence"]


def test_small_differences_kept_at_full_precision(base):
    res, out, _, _ = base
    txt = (out / "ps2_fold_cells.csv").read_text()
    c = next(c for c in res["cells"] if c.get("cell_status") == "MEASURED")
    assert repr(c["oof_delta"]) in txt


# ------------------------------------------------------------------ statistics units

def test_spearman_null_matches_direct_roll():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=500), rng.normal(size=500)
    sh = np.array([50, 100, 250])
    obs, null = ps2.spearman_null(x, y, sh)
    assert obs == pytest.approx(spearmanr(x, y)[0])
    for s, v in zip(sh, null):
        assert v == pytest.approx(spearmanr(np.roll(x, -s), y)[0], abs=1e-9)


def test_bh_matches_reference_and_counts_tests():
    q, m = ps2.bh([0.01, None, 0.04, 0.03, 0.5])
    assert m == 4 and q[1] is None
    assert q[0] == pytest.approx(0.04)
    assert q[3] == pytest.approx(0.04 * 4 / 3) and q[2] == pytest.approx(0.04 * 4 / 3)
    assert q[4] == pytest.approx(0.5)


def test_multiplicity_bookkeeping_recorded(base):
    res, _, _, _ = base
    meas = [c for c in res["cells"] if c.get("cell_status") == "MEASURED"]
    assert res["multiplicity"]["tests_total"] == 2 * len(meas)
    assert all(c["spearman_family_m"] >= 1 and c["n_null"] >= 10 for c in meas)


# ------------------------------------------------------------------ mutations (the tests can fail)

def test_mutation_without_pair_check_the_pair_stays_low_priority(tmp_path, monkeypatch):
    monkeypatch.setattr(ps2, "_synergy", lambda *a, **k: [])
    res, _, _ = _run(_frame(), tmp_path, "nosyn", write=False)
    st = {(r["feature"], r["target"], r["horizon"]): r for r in res["status"]}
    assert st[("syn_a", "Y_s", 1)]["status"] != "PROVISIONAL_SURVIVOR" or \
        st[("syn_b", "Y_s", 1)]["status"] != "PROVISIONAL_SURVIVOR"


def test_mutation_without_train_restriction_outputs_move(tmp_path, monkeypatch):
    df = _frame()
    df.loc[df.index >= N_TRAIN, "close"] *= 1.5
    real = ps2.restrict_to_train
    monkeypatch.setattr(ps2, "restrict_to_train",
                        lambda ts, X, price, te: real(ts, X, price, 2 ** 62))
    a, _, _ = _run(_frame(), tmp_path, "m1", write=False)
    b, _, _ = _run(df, tmp_path, "m2", write=False)
    assert a["cells"] != b["cells"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fs16_holds_across_seeds_and_noise_pairs_rarely_promoted(tmp_path, seed):
    df = _frame(seed)
    res, _, _ = _run(df, tmp_path, f"s{seed}", write=False)
    st = {(r["feature"], r["target"], r["horizon"]): r for r in res["status"]}
    for f in ("syn_a", "syn_b"):
        assert "S_SYNERGY_PAIR" in st[(f, "Y_s", 1)]["reasons"]
    noise_pairs = [r for r in res["status"] if r["feature"].startswith("noise")
                   and any(o.startswith("noise") for o in r.get("synergy_partners", []))]
    assert len(noise_pairs) <= 2
