"""Regime fitting cannot cross the fold boundary -- written RED first, against `app/regime_detector.py` at `d081d0f`.

Owner addendum `SATOSHI_SOURCE_TRANSFORM_COVERAGE_ADDENDUM_2026_10_01.md` (predictor master 256c61a6), finding 6, and
the dossier `financial-data/docs/METHOD_SEMANTICS_DOSSIER_2026_10_01.md` (regime row). Source observations, each with
its line, made executable:

* `_GMM_SCALER_MEAN` / `_GMM_SCALER_SCALE` (:252-253) and `_GMM_CENTROIDS_RAW` (:256-266) are module constants whose
  comment says "fitted on 15yr EURUSD (24K 4h bars)" (:248) and "Scaler params from StandardScaler fit on same data"
  (:250); no fit period, no fold, no data digest travels with them.
* `_GMM_CLUSTER_TO_REGIME` (:272-282) is "Based on forward-return analysis from clustering study" (:269-271): the
  label mapping was informed by returns realized AFTER each bar.
* `classify_regime_v3` (:284-312) takes only `features`; it cannot be told which fold it serves, and its docstring
  says the boundaries are "fixed from unsupervised clustering on the full historical dataset" (:291-292).

These are source-code observations, not a measured leakage rate on any retained model. `test_observed_*` are GREEN
today and pin those observations; `test_required_*` are RED today and name the mechanism that admits the detector
into a temporal fold: a fit bound to a fold (`fit_regime_v3(features, fold)` refusing rows past `train_end`), a label
mapping with its own provenance bound to the same fold, and an admission check that refuses a detector whose fit
period or mapping is not linked to the fold it is used in. Causality here is the absence of unavailable future
inputs, not an economic causal effect.

CPU only, synthetic bars, standard library plus numpy, pandas and pytest.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import regime_detector as rd

SOURCE = Path(rd.__file__)


def bars(n=800, seed=11, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    index = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=index)


def fold(train_end):
    return {"fold_id": "synthetic-fold-1", "train_end": pd.Timestamp(train_end, tz="UTC")}


def _missing(what, how):
    pytest.fail(f"MECHANISM_MISSING: {what}. Turns green when {how}.")


def test_observed_v3_constants_are_literals_with_no_fit_provenance():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_GMM_"):
                    names[target.id] = node.value
    assert {"_GMM_SCALER_MEAN", "_GMM_SCALER_SCALE", "_GMM_CENTROIDS_RAW", "_GMM_CLUSTER_TO_REGIME"} <= set(names)
    # every one of them is a literal (np.array(...) over list literals, or a dict literal): nothing is fitted at import
    for name, value in names.items():
        if isinstance(value, ast.Call):
            assert all(isinstance(a, (ast.List, ast.Constant)) for a in value.args), name
        else:
            assert isinstance(value, ast.Dict), name
    text = SOURCE.read_text(encoding="utf-8")
    assert "fitted on 15yr EURUSD" in text and "forward-return analysis" in text
    assert not any(k in text for k in ("fit_period", "train_end", "fold_id")), "no fold or period travels with them"


def test_observed_classify_v3_uses_the_same_centroids_for_any_input():
    """Two unrelated synthetic histories are classified against identical, fixed centroids: nothing is refitted."""
    a = rd.compute_regime_features(bars(seed=1)).dropna()
    b = rd.compute_regime_features(bars(seed=2, start="2015-06-01")).dropna()
    before = rd._GMM_CENTROIDS_RAW.copy()
    rd.classify_regime_v3(a)
    rd.classify_regime_v3(b)
    np.testing.assert_array_equal(rd._GMM_CENTROIDS_RAW, before)
    assert "features" in inspect.signature(rd.classify_regime_v3).parameters
    assert len(inspect.signature(rd.classify_regime_v3).parameters) == 1


def test_required_fit_is_bound_to_a_fold_and_refuses_rows_past_train_end():
    fit = getattr(rd, "fit_regime_v3", None)
    if fit is None:
        _missing("no `fit_regime_v3` exists; the scaler and centroids are module constants fitted outside any fold",
                 "fit_regime_v3(features, fold) fits scaler and centroids on rows <= fold['train_end'] only, "
                 "returns a RegimeFit carrying fold_id, train_end, n_rows, data_sha256 and the fitted arrays, and "
                 "refuses REGIME_FIT_CROSSES_FOLD when given rows after train_end")
    features = rd.compute_regime_features(bars(n=1200)).dropna()
    f = fold(features.index[800])
    fitted = fit(features.loc[: f["train_end"]], f)
    assert fitted["fold_id"] == f["fold_id"] and fitted["train_end"] == f["train_end"]
    with pytest.raises(Exception) as trouble:
        fit(features, f)
    assert "REGIME_FIT_CROSSES_FOLD" in str(trouble.value)


def test_required_label_mapping_carries_its_own_fold_provenance():
    mapping_of = getattr(rd, "label_mapping_for", None)
    if mapping_of is None:
        _missing("the cluster-to-regime mapping is a literal dict 'based on forward-return analysis' with no period",
                 "label_mapping_for(fitted, outcomes_train) derives the mapping from TRAIN-only forward returns of "
                 "the same fold and records fold_id, train_end and the outcome horizon it used")
    features = rd.compute_regime_features(bars(n=1200)).dropna()
    f = fold(features.index[800])
    fitted = rd.fit_regime_v3(features.loc[: f["train_end"]], f)
    mapping = mapping_of(fitted, features["close"].pct_change(6).shift(-6).loc[: f["train_end"]])
    assert mapping["fold_id"] == f["fold_id"] and mapping["train_end"] == f["train_end"]
    assert set(mapping["cluster_to_regime"].values()) <= set(rd.REGIME_NAMES)


def test_required_admission_links_fit_period_and_mapping_to_the_fold():
    admit = getattr(rd, "admit_regime_feature", None)
    if admit is None:
        _missing("nothing checks that a regime detector's fit period and label mapping belong to the fold it is "
                 "used in; classify_regime_v3 takes only `features`",
                 "admit_regime_feature(fitted, mapping, fold) returns ADMITTED only when both carry the fold's id and "
                 "a train_end <= fold['train_end'], and otherwise refuses by name: REGIME_FIT_NOT_LINKED_TO_FOLD or "
                 "REGIME_MAPPING_NOT_LINKED_TO_FOLD; the retained constants of d081d0f are refused "
                 "REGIME_FIT_PERIOD_UNKNOWN")
    f = fold("2021-01-01")
    legacy = {"fold_id": None, "train_end": None, "centroids": rd._GMM_CENTROIDS_RAW}
    verdict = admit(legacy, {"fold_id": None, "train_end": None}, f)
    assert verdict["state"] == "REFUSED" and verdict["reason"] == "REGIME_FIT_PERIOD_UNKNOWN"
