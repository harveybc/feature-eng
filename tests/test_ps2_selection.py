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


# ------------------------------------------------------------------ contracts and CLI

def test_not_evaluated_is_not_zero_and_not_reject(tmp_path):
    df = _frame()
    df.loc[df.index < int(N_TRAIN * 0.9), "noise_6"] = np.nan  # almost no TRAIN support
    res, _, _ = _run(df, tmp_path, "sparse", write=False)
    cs = [c for c in res["cells"] if c["feature"] == "noise_6"]
    assert cs and all(c["cell_status"] == "NOT_EVALUATED" for c in cs if c["fold"] == "inner_1")
    for c in cs:
        if c["cell_status"] == "NOT_EVALUATED":
            assert "oof_delta" not in c and "spearman" not in c and c["cell_reason"]
    for r in res["status"]:
        if r["feature"] == "noise_6":
            assert r["status"] != "TECHNICAL_REJECT"


def test_barrier_label_support_inside_train_and_classes():
    df = _frame()
    b = ps2.batch_from_frame(df, "ts", "close", FEATS, _train_end(df), DOMAINS, DECLARED)
    p = dict(ps2.DEFAULT_PARAMS, **SMALL)
    y, end = ps2.build_targets(b.ts, b.price, p)[("Y_b", 24)]
    fin = np.isfinite(y)
    assert set(np.unique(y[fin])) <= {-1.0, 0.0, 1.0} and len(np.unique(y[fin])) == 3
    assert end[fin].max() <= b.ts[-1]
    assert not fin[: p["barrier"]["min_vol_returns"]].any()  # no label without past volatility


def test_ready_marker_names_the_manifest_digest(base):
    _, out, w, _ = base
    ready = json.loads((out / "READY").read_text())
    assert ready["manifest_canonical_sha256"] == w["manifest_canonical_sha256"]
    man = json.loads((out / "ps2_manifest.json").read_text())
    assert ps2._sha(ps2._canon(man).encode()) == w["manifest_canonical_sha256"]
    for fn, d in man["output_sha256"].items():
        assert ps2._sha((out / fn).read_bytes()) == d
    assert not (out / ".READY.tmp").exists()


def test_cli_end_to_end_matches_library(tmp_path, base):
    _, out, _, _ = base
    df = _frame()
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    spec = tmp_path / "f.json"
    spec.write_text(json.dumps({"features": FEATS, "domains": DOMAINS,
                                "declared": {k: list(v) for k, v in DECLARED.items()}}))
    prm = tmp_path / "p.json"
    prm.write_text(json.dumps(SMALL))
    assert ps2.main(["--data", str(csv), "--ts-col", "ts", "--price-col", "close",
                     "--train-end", str(_train_end(df)), "--features-json", str(spec),
                     "--params-json", str(prm), "--batch-id", "t", "--out-dir",
                     str(tmp_path / "cli")]) == 0
    a = (out / "ps2_status.csv").read_text()
    assert (tmp_path / "cli" / "ps2_status.csv").read_text() == a


# ------------------------------------------------------------------ lane A batch adapter

def _lane_a_dir(tmp, tamper=False, late_row=False):
    import hashlib
    df = _frame().iloc[:N_TRAIN].copy()
    d = tmp / "laneA"
    d.mkdir()
    ts = pd.DatetimeIndex(df["ts"])
    read_end = ts[-1] + pd.Timedelta(hours=1)
    if late_row:
        read_end = ts[-10]
    b = ps2.batch_from_frame(df, "ts", "close", FEATS, ts[-1] + pd.Timedelta(hours=1))
    tg = ps2.build_targets(b.ts, b.price, dict(ps2.DEFAULT_PARAMS, **SMALL))
    feats = [f for f in FEATS if f != "next_return_label"]
    fx = df[feats].copy()
    fx.insert(0, "row_id", np.arange(len(df)))
    fx.insert(0, "t_decision_utc", ts)
    fx.to_parquet(d / "features_train.parquet", index=False)
    t = pd.DataFrame({"t_decision_utc": ts, "row_id": np.arange(len(df)),
                      "Y_s_1h": tg[("Y_s", 1)][0], "Y_s_3h": tg[("Y_s", 3)][0],
                      "Y_l_24h": tg[("Y_l", 24)][0], "Y_b_l24": tg[("Y_b", 24)][0]})
    t.to_parquet(d / "targets_train.parquet", index=False)
    meta = [{"feature_id": f, "family": DOMAINS.get(f, f.split("_")[0]),
             "admissibility": "ADMISSIBLE"} for f in feats]
    meta[feats.index("bad_declared")]["admissibility"] = "EXCLUDED_ROLE:quality_excluded"
    meta[feats.index("noise_5")]["family"] = "event_surprise"  # economic-calendar column
    json.dump({"features": meta}, open(d / "admissible_features.json", "w"))
    n = len(df)
    folds = []
    for k, (vs, ve) in enumerate([(int(n * .55), int(n * .7)), (int(n * .7), int(n * .85)),
                                  (int(n * .85), n)]):
        folds.append({"name": f"inner_{k}", "train_rows": [0, vs - 30], "val_rows": [vs, ve],
                      "label_purge_h": 24})
    json.dump({"folds": folds}, open(d / "folds.json", "w"))
    json.dump({"contract_sha256": "c", "periods": {"read_end_for_ps0_ps1": str(read_end)},
               "targets": {"Y_b": {"specs": [{"name": "Y_b_l24", "timeout_h": 24}]}}},
              open(d / "contract.json", "w"))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    arts = {fn: sha(d / fn) for fn in sorted(os.listdir(d))}
    json.dump({"artifacts_sha256": arts, "code_commit": "x"}, open(d / "digests.json", "w"))
    (d / "READY").write_text(json.dumps({"batch": "batch_t", "digests_sha256": sha(d / "digests.json")}))
    if tamper:
        fx.loc[5, "noise_1"] = 99.0
        fx.to_parquet(d / "features_train.parquet", index=False)
    return d


def test_lane_a_adapter_runs_and_maps_exclusions(tmp_path):
    d = _lane_a_dir(tmp_path)
    res, w = ps2.run_lane_a(str(d), str(tmp_path / "out"), dict(SMALL, fold_majority=2,
                                                                 synergy_fold_min=3))
    st = {(r["feature"], r["target"], r["horizon"]): r for r in res["status"]}
    assert st[("bad_declared", "Y_s", 1)]["status"] == "TECHNICAL_REJECT"
    assert "EXCLUDED_ROLE" in st[("bad_declared", "Y_s", 1)]["reason_detail"][0]
    assert {k[1] for k in st} == {"Y_s", "Y_l", "Y_b_l24"}
    assert "S_SYNERGY_PAIR" in st[("syn_a", "Y_s", 1)]["reasons"]
    assert res["provenance"]["lane_a_batch"] == "batch_t"
    assert (tmp_path / "out" / "READY").exists()
    assert any(c["target"] == "Y_b_l24" and c.get("loss_name") == "logloss" for c in res["cells"])


def test_lane_a_adapter_refuses_tampered_or_late_rows(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    with pytest.raises(ps2.PS2Error, match="digest"):
        ps2.load_lane_a_batch(str(_lane_a_dir(tmp_path / "x", tamper=True)))
    with pytest.raises(ps2.PS2Error, match="READ_END"):
        ps2.load_lane_a_batch(str(_lane_a_dir(tmp_path / "y", late_row=True)))


def test_selector_episode_sources_excluded_and_reported(tmp_path):
    d = _lane_a_dir(tmp_path)
    b = ps2.load_lane_a_batch(str(d))
    assert "noise_5" not in b.names
    sel = b.provenance["selector_episode_sources"]
    assert [x["feature_id"] for x in sel] == ["noise_5"]
    assert sel[0]["status"] == "SELECTOR_EPISODE_SOURCE"


def test_ps2_batch_v1_contract_for_extractor_lanes(tmp_path):
    import hashlib
    d = _lane_a_dir(tmp_path)
    out = tmp_path / "batch_001"
    res, w = ps2.run_lane_a(str(d), str(out), dict(SMALL, fold_majority=2, synergy_fold_min=3))
    man = json.loads((out / "batch_manifest.json").read_text())
    assert man["schema"] == "ps2_batch.v1" and man["sampling_period_seconds"] == 3600
    for part in ("series", "targets"):
        assert hashlib.sha256((out / man[part]["file"]).read_bytes()).hexdigest() == man[part]["sha256"]
    s, t = np.load(out / "series.npz"), np.load(out / "targets.npz")
    ts = s["timestamps"]
    assert np.all(np.diff(ts) == 3600) and np.array_equal(t["timestamps"], ts)
    assert ts[-1] == man["train_end_ts"]
    assert t["Y_s"].shape == (len(ts), 2) and t["Y_l"].shape == (len(ts), 1)
    assert t["Y_b"].dtype.kind == "i" and set(np.unique(t["Y_b"])) <= {-1, 0, 1, 2}
    assert (t["Y_b"] == 0).any()  # SL-first class survives the re-encoding (raw -1 is not 'no support')
    surv = {r["feature"] for r in res["status"] if r["status"] in ("PROVISIONAL_SURVIVOR", "EXPLORATION")}
    assert set(man["features"]) == surv and "noise_5" not in man["features"]
    for f in man["features"]:
        assert ("x__" + f) in s.files and s["x__" + f].dtype == np.float32
    for fd in man["folds"]:
        fi = np.nonzero((ts >= fd["fit"][0]) & (ts <= fd["fit"][1]))[0]
        vi = np.nonzero((ts >= fd["val"][0]) & (ts <= fd["val"][1]))[0]
        assert fd["split"] == "train" and fd["val"][1] <= man["train_end_ts"]
        assert vi.min() - fi.max() > 720 and fi.min() >= 720
    ready = json.loads((out / "READY").read_text())
    assert ready["batch_manifest_sha256"] == hashlib.sha256(
        (out / "batch_manifest.json").read_bytes()).hexdigest()
    lc = json.loads((out / "ps2_candidates_lane_c.json").read_text())
    assert {c["feature_id"] for c in lc["candidates"]} == surv


def test_synergy_pair_budget_is_a_hard_cap(tmp_path):
    res, _, _ = _run(_frame(), tmp_path, "cap", {"synergy_max_pairs": 5, "synergy_top_k": 2},
                     write=False)
    rules = {s["pair_rule"] for s in res["synergy"]}
    assert all(s["pairs_evaluated_in_cell"] <= 5 for s in res["synergy"])
    assert rules <= {r for r in rules if "truncated" in r}


def test_incremental_batch_uses_digest_pinned_base(tmp_path):
    import hashlib
    import shutil
    base = _lane_a_dir(tmp_path)
    root = tmp_path / "ps1"
    root.mkdir()
    shutil.move(str(base), str(root / "batch_001"))
    inc = root / "batch_002"
    inc.mkdir()
    fx = pd.read_parquet(root / "batch_001" / "features_train.parquet")
    fx[["t_decision_utc", "row_id", "noise_1", "noise_2", "syn_a", "syn_b"]].rename(
        columns=lambda c: c.replace("noise_", "x_n").replace("syn_", "x_s")).to_parquet(
        inc / "features_train.parquet", index=False)
    json.dump({"features": [{"feature_id": f, "family": "x", "admissibility": "ADMISSIBLE"}
                            for f in ("x_n1", "x_n2", "x_sa", "x_sb")]},
              open(inc / "admissible_features.json", "w"))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    base_d = json.loads((root / "batch_001" / "READY").read_text())["digests_sha256"]
    json.dump({"artifacts_sha256": {fn: sha(inc / fn) for fn in sorted(os.listdir(inc))},
               "base_batch_digests_sha256": base_d}, open(inc / "digests.json", "w"))
    (inc / "READY").write_text(json.dumps({"batch": "batch_002",
                                           "digests_sha256": sha(inc / "digests.json")}))
    b = ps2.load_lane_a_batch(str(inc))
    assert b.names == ["x_n1", "x_n2", "x_sa", "x_sb"] and len(b.folds) == 3
    assert b.provenance["lane_a_base_batch_digests_sha256"] == base_d
    # tampering with the base targets refuses the incremental batch
    t = pd.read_parquet(root / "batch_001" / "targets_train.parquet")
    t.loc[3, "Y_s_1h"] = 0.5
    t.to_parquet(root / "batch_001" / "targets_train.parquet", index=False)
    with pytest.raises(ps2.PS2Error, match="digest"):
        ps2.load_lane_a_batch(str(inc))


def test_mi_only_does_not_promote_returns_but_is_recorded():
    rows = None
    p = dict(SMALL)
    df = _frame()
    b = ps2.batch_from_frame(df, "ts", "close", FEATS, _train_end(df), DOMAINS, DECLARED)
    real = ps2.emp_p
    # make every MI p tiny and every Spearman p large: MI-only evidence everywhere
    import unittest.mock as um
    with um.patch.object(ps2, "spearman_null", lambda x, y, sh: (0.0, np.ones(len(sh)))), \
            um.patch.object(ps2, "mi_null", lambda x, y, sh, B, c=False: (1.0, np.zeros(len(sh)))), \
            um.patch.object(ps2, "_synergy", lambda *a, **k: []), \
            um.patch.object(ps2, "_groups", lambda *a, **k: []):
        res = ps2.build(b, dict(p, util_null_p=0.0))
    rows = [r for r in res["status"] if r["status"] != "TECHNICAL_REJECT"]
    for r in rows:
        if r["target"].startswith("Y_b"):
            continue
        assert "S_ASSOC_MI_ONLY" not in r["reasons"]
        if "LP_ASSOC_MI_ONLY_SCALE_DEPENDENCE" in r["reasons"]:
            assert r["status"] != "PROVISIONAL_SURVIVOR"
    assert real is ps2.emp_p
    assert any("LP_ASSOC_MI_ONLY_SCALE_DEPENDENCE" in r["reasons"] for r in rows)
    assert any("S_ASSOC_MI_ONLY" in r["reasons"] for r in rows if r["target"].startswith("Y_b"))


def test_extractor_priority_tiers_never_change_status(base):
    res, _, _, _ = base
    before = [(r["feature"], r["target"], r["horizon"], r["status"]) for r in res["status"]]
    pri = ps2.extractor_priority(res["status"], 0.10, 2)
    after = [(r["feature"], r["target"], r["horizon"], r["status"]) for r in res["status"]]
    assert before == after and pri["statuses_changed"] is False
    allt = pri["tier_1"] + pri["tier_2"] + pri["tier_3"]
    assert len(allt) == len(set(allt))
    surv = {r["feature"] for r in res["status"] if r["status"] == "PROVISIONAL_SURVIVOR"}
    assert set(allt) == surv and pri["exploration"] == res["exploration"]["sample"]
    assert "sig_level" in pri["tier_1"]
