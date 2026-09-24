"""The workbench's one envelope, answered by this provider under the retained demo reference.

Every number an answer carries is computed from the supplied rows under the frozen transform: distributions from the
actual assignments, centroids as the mean of the caller's own rows, silhouette from sklearn over the same scaled rows
and labels. What the reference does not have -- another k, another feature list, a column the rows do not carry -- is
refused by name, question by question, and never repaired on the caller's behalf.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import silhouette_score

from feature_eng_m5phet.provider import Provider
from feature_eng_m5phet.questions import OPTIMAL_K_BASIS
from feature_eng_m5phet.regimes import HierarchicalRegimes, validate_rows

DEMO = Path(os.environ.get("FEATURE_ENG_REGIMES_DEMO_DIR")
            or Path.home() / ".local" / "state" / "m5phet" / "examples" / "regimes")


def workbench():
    """`m5phet.questions`, imported normally or from the source tree this machine keeps it in."""
    source = os.environ.get("M5PHET_SRC_PATH") or str(
        Path.home() / "Documents" / "GitHub" / ".worktrees" / "m5phet-chat" / "src")
    if Path(source, "m5phet", "questions.py").is_file() and source not in sys.path:
        sys.path.insert(0, source)
    return pytest.importorskip("m5phet.questions"), pytest.importorskip("m5phet.runtime")


@pytest.fixture
def demo(monkeypatch):
    """The retained reference, under an interpreter carrying the dependency versions it was fitted with.

    `HierarchicalRegimes.load` refuses any other dependency set, and that refusal is not weakened here: the reference
    is not refitted for a test run, so under another interpreter these tests are skipped naming the reason."""
    if not (DEMO / "reference.joblib").is_file() or not (DEMO / "query.json").is_file():
        pytest.skip(f"no retained demo reference under {DEMO}")
    monkeypatch.setenv("FEATURE_ENG_REGIMES_DEMO_DIR", str(DEMO))
    monkeypatch.delenv("FEATURE_ENG_REGIMES_STATE_PATH", raising=False)
    try:
        HierarchicalRegimes.load(str(DEMO / "reference.joblib"))
    except ValueError as exc:
        pytest.skip(f"{exc}: the retained reference declares {_manifest()['metadata']['dependencies']}")
    return DEMO


def _manifest():
    return json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def rows(demo):
    return json.loads((demo / "query.json").read_text(encoding="utf-8"))["rows"]


@pytest.fixture
def provider(demo):
    return Provider()


@pytest.fixture
def model(provider, demo):
    return provider.load(str((demo / "reference.joblib").resolve()))["model"]


@pytest.fixture
def run(provider, rows):
    questions, runtime = workbench()
    registry = runtime.Registry()
    registry.register(provider)

    def _run(asked, *, features=None, data=None):
        state = {"dataset_id": "ohlc-demo", "features": list(model_features(provider) if features is None else features)}
        return questions.run_task({"area": "unsupervised", "state": state, "questions": asked}, registry,
                                  data={"rows": rows} if data is None else data)
    return _run


def model_features(provider):
    return _manifest()["metadata"]["features"]


def labels_at(model, rows, level):
    paths = np.asarray([r["cluster_path"] for r in model.assign(rows)["rows"]])
    return paths[:, model.metadata["levels"].index(level) + 1]


# --- clustering: assignment, not selection ------------------------------------------------------------------------------

def test_a_clustering_question_returns_a_distribution_over_the_supplied_rows(run, model, rows):
    out = run({"segmentacion": {"type": "clustering", "method": "auto", "expected_clusters": "2-5"}})
    answer = out["answers"]["segmentacion"]
    assert answer["status"] == "OK" and answer["type"] == "clustering", answer
    assert answer["optimal_k"] == model.metadata["levels"] and answer["optimal_k_basis"] == OPTIMAL_K_BASIS
    assert answer["rows"] == len(rows)
    for level in model.metadata["levels"]:
        shares = answer["cluster_distribution"][str(level)]
        assert shares and abs(sum(shares.values()) - 1.0) < 1e-12
        expected = {str(int(k)): int(v) for k, v in zip(*np.unique(labels_at(model, rows, level), return_counts=True))}
        assert answer["cluster_counts"][str(level)] == expected, "the distribution is the actual assignment"
        assert answer["clusters_occupied"][str(level)] == len(expected) <= level
    assert answer["reference"]["model_version"] == model.model_version
    assert out["state_ref"] == str((DEMO / "reference.joblib").resolve())


def test_silhouette_is_sklearns_over_the_scaled_supplied_rows_or_is_omitted_with_a_reason(run, model, rows):
    answer = run({"s": {"type": "clustering"}})["answers"]["s"]
    _, raw = validate_rows(rows, model.metadata["features"])
    scaled = model.scaler.transform(raw)
    for level in model.metadata["levels"]:
        labels = labels_at(model, rows, level)
        distinct = len(set(labels.tolist()))
        if 2 <= distinct <= len(rows) - 1:
            assert answer["silhouette_score"][str(level)] == pytest.approx(
                silhouette_score(scaled, labels, metric="euclidean"))
        else:
            assert str(level) in answer["silhouette_omitted"]
            assert str(level) not in answer.get("silhouette_score", {})
    assert "silhouette_score" in answer or "silhouette_omitted" in answer


def test_a_k_the_reference_is_not_fitted_with_is_refused(run, model):
    q, _ = workbench()
    answer = run({"seg": {"type": "clustering", "expected_clusters": "6-9"}})["answers"]["seg"]
    assert answer["status"] == "REFUSED" and answer["refusal"] == q.NOT_ESTIMABLE
    assert "6-9" in answer["why"] and str(model.metadata["levels"]) in answer["why"]
    assert not any(isinstance(v, (int, float)) and k not in ("type",) for k, v in answer.items()), "no number rides a refusal"


def test_a_k_within_the_fitted_levels_reports_only_those_levels(run, model):
    answer = run({"seg": {"type": "clustering", "expected_clusters": 2}})["answers"]["seg"]
    assert answer["status"] == "OK" and answer["optimal_k"] == [2] and list(answer["cluster_distribution"]) == ["2"]


def test_a_method_that_is_not_the_fitted_engine_is_refused(run):
    q, _ = workbench()
    answer = run({"seg": {"type": "clustering", "method": "kmeans"}})["answers"]["seg"]
    assert answer["refusal"] == q.NOT_ESTIMABLE and "kmeans" in answer["why"] and "refit" in answer["why"]


# --- cluster_description: the matched cluster, described in the caller's units ----------------------------------------

def test_a_cluster_description_returns_centroids_in_original_units_for_the_matched_cluster(run, model, rows):
    features = model.metadata["features"]
    metric = f"{features[1]} > 600"
    answer = run({"perfil": {"type": "cluster_description", "target_metric": metric}})["answers"]["perfil"]
    assert answer["status"] == "OK" and answer["type"] == "cluster_description", answer
    level = model.metadata["levels"][-1]
    assert answer["level"] == level
    labels = labels_at(model, rows, level)
    raw = np.asarray([[r[f] for f in features] for r in rows], dtype=float)
    satisfied = raw[:, 1] > 600
    shares = {int(c): satisfied[labels == c].mean() for c in set(labels.tolist())}
    assert shares[answer["matched_cluster"]] == max(shares.values())
    member = labels == answer["matched_cluster"]
    for i, name in enumerate(features):
        assert answer["centroid_features"][name] == pytest.approx(raw[member, i].mean())
    assert list(answer["centroid_features"]) == features
    # the scaled coordinates are NOT what is reported
    scaled_centroid = model.scaler.transform(raw)[member].mean(axis=0)
    assert not np.allclose(list(answer["centroid_features"].values()), scaled_centroid)
    assert answer["rows_in_cluster"] == int(member.sum())


def test_a_level_may_be_named_when_the_reference_has_it(run, model, rows):
    q, _ = workbench()
    coarse = model.metadata["levels"][0]
    ok = run({"p": {"type": "cluster_description", "target_metric": f"{model.metadata['features'][0]} < 0",
                    "level": coarse}})["answers"]["p"]
    assert ok["status"] == "OK" and ok["level"] == coarse
    bad = run({"p": {"type": "cluster_description", "target_metric": f"{model.metadata['features'][0]} < 0",
                     "level": 3}})["answers"]["p"]
    assert bad["refusal"] == q.NOT_ESTIMABLE and "3" in bad["why"]


def test_a_metric_on_a_column_the_rows_do_not_carry_is_refused_by_name(run):
    q, _ = workbench()
    answer = run({"perfil": {"type": "cluster_description", "target_metric": "gasto_anual > 1000"}})["answers"]["perfil"]
    assert answer["status"] == "REFUSED" and answer["refusal"] == q.NOT_ESTIMABLE
    assert "gasto_anual" in answer["why"]


def test_a_metric_that_is_not_a_comparison_is_malformed(run):
    q, _ = workbench()
    answer = run({"p": {"type": "cluster_description", "target_metric": "the big spenders"}})["answers"]["p"]
    assert answer["refusal"] == q.MALFORMED_QUESTION


# --- the envelope as a whole -----------------------------------------------------------------------------------------------

def test_a_request_with_both_questions_answers_both(run, model):
    out = run({"segmentacion": {"type": "clustering", "method": "auto", "expected_clusters": "2-4",
                                "instructions": "describe the market regimes"},
               "perfil": {"type": "cluster_description", "target_metric": f"{model.metadata['features'][0]} > 0"}})
    assert out["answered"] == 2 and out["refused"] == 0
    assert list(out["answers"]) == ["segmentacion", "perfil"]
    assert out["execution_authorized"] is False and out["provider"] == Provider.name


def test_a_feature_mismatch_is_refused_by_name_on_every_question(run, model):
    q, _ = workbench()
    fitted = model.metadata["features"]
    out = run({"seg": {"type": "clustering"}, "perfil": {"type": "cluster_description", "target_metric": f"{fitted[0]} > 0"}},
              features=["edad", "gasto_anual"])
    for answer in out["answers"].values():
        assert answer["refusal"] == q.NOT_ESTIMABLE
        assert "edad" in answer["why"] and "gasto_anual" in answer["why"] and fitted[0] in answer["why"]


def test_the_fitted_features_in_another_order_are_refused_not_reordered(run, model):
    q, _ = workbench()
    answer = run({"seg": {"type": "clustering"}}, features=list(reversed(model.metadata["features"])))["answers"]["seg"]
    assert answer["refusal"] == q.NOT_ESTIMABLE and "order" in answer["why"]


def test_a_missing_feature_list_is_refused_as_state(run):
    q, _ = workbench()
    questions, runtime = workbench()
    registry = runtime.Registry()
    registry.register(Provider())
    out = questions.run_task({"area": "unsupervised", "state": {"dataset_id": "x"},
                              "questions": {"seg": {"type": "clustering"}}}, registry, data={"rows": [{"row_id": 1}]})
    assert out["answers"]["seg"]["refusal"] == q.STATE_REQUIRED


def test_rows_that_do_not_match_the_fitted_shape_are_refused(run):
    q, _ = workbench()
    answer = run({"seg": {"type": "clustering"}}, data={"rows": [{"row_id": 1, "edad": 3.0}]})["answers"]["seg"]
    assert answer["refusal"] == q.STATE_REQUIRED


def test_a_state_ref_that_is_not_the_configured_reference_is_refused(provider, rows):
    questions, runtime = workbench()
    q = questions
    registry = runtime.Registry()
    registry.register(provider)
    out = questions.run_task({"area": "unsupervised",
                              "state": {"features": model_features(provider), "state_ref": "/nowhere/reference.joblib"},
                              "questions": {"seg": {"type": "clustering"}}}, registry, data={"rows": rows})
    assert out["answers"]["seg"]["refusal"] == q.STATE_REQUIRED and "/nowhere" in out["answers"]["seg"]["why"]


def test_the_catalog_declares_the_two_types(provider):
    questions, runtime = workbench()
    registry = runtime.Registry()
    registry.register(provider)
    declared = questions.catalog(registry)["unsupervised"]
    assert declared["provider"] == Provider.name
    assert set(declared["question_types"]) == {"clustering", "cluster_description"}
    assert declared["question_types"]["cluster_description"]["required"] == ["target_metric"]
