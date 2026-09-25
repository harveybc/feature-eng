"""WP19, first half: the declared method space, the two decisions made over it, and the spec they produce.

Everything here runs against a FAKE classification provider, in the style of `M5PHET/tests/test_decide.py`. It
proves what reaches the model and what the model may not do to the answer. It establishes nothing about whether a
chosen method is any good: the indices are computed in `test_evaluate_regimes.py`, and `regime_accuracy` stays
refused everywhere.
"""

import json
import os
from pathlib import Path
import sys

import pytest

from feature_eng_m5phet import choose_regimes, regime_space
from feature_eng_m5phet.provider import Provider


def workbench():
    """`m5phet`, imported normally or from the source tree this machine keeps it in (as the other tests do)."""
    source = os.environ.get("M5PHET_SRC_PATH") or str(
        Path.home() / "Documents" / "GitHub" / ".worktrees" / "m5phet-chat" / "src")
    if Path(source, "m5phet", "decide.py").is_file() and source not in sys.path:
        sys.path.insert(0, source)
    return pytest.importorskip("m5phet.decide")


class FakeLaya:
    """A classification provider with Laya's answer shape. `backend` decides whether a decision may exist at all."""

    name, area = "laya_news", "classification"

    def __init__(self, choices, *, backend="laya", probabilities=None, fixture=False):
        self.choices, self.backend, self.fixture = dict(choices), backend, fixture
        self.probabilities = probabilities
        self.seen = []

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["classification"],
                "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
                "supported": [{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}],
                "known_states": ["laya-checkpoint:fake"], "backend": self.backend}

    def question_types(self):
        return {"choice": {"required": ["options"], "optional": ["instructions"]}}

    def answer_questions(self, state, questions, data, as_of):
        self.seen.append({"state": state, "questions": json.loads(json.dumps(questions)), "as_of": as_of})
        answers = {}
        for name, question in questions.items():
            keys = [key for key, _label in question["options"]]
            chosen = self.choices.get(name, keys[0])
            probabilities = self.probabilities or {key: round(1.0 / len(keys), 4) for key in keys}
            answers[name] = {"type": "choice", "status": "OK", "label": chosen, "backend": self.backend,
                             "instructions": question.get("instructions"),
                             "options": [list(option) for option in question["options"]],
                             "uncalibrated_probabilities": dict(probabilities), "probability_decimals": 4,
                             "calibration": "UNCALIBRATED", "execution_authorized": False,
                             **({"non_model_fixture": True} if self.fixture else {})}
        answers["__state_ref__"] = "laya-checkpoint:fake"
        return answers


def engine_with(provider):
    decide = workbench()
    runtime = pytest.importorskip("m5phet.runtime")
    interpret = pytest.importorskip("m5phet.interpret")
    engine_module = pytest.importorskip("m5phet.web.engine")

    class Mute(interpret.Interpreter):
        def __init__(self):
            super().__init__(command="fixture", model="fixture-v1", environ={})

        @property
        def available(self):
            return False

    registry = runtime.Registry()
    registry.register(provider)
    engine = engine_module.Engine(registry=registry, environ={})
    engine.interpreter = Mute()
    return decide, engine


PROFILE = {
    "dataset": {"rows": 40320, "columns_clustered": 2, "sampling_step_seconds": 60.0,
                "sampling_regular_fraction": 1.0},
    "feature_line": choose_regimes.FEATURE_LINE,
    "features": {"a": "STATIONARY 0.100 -0.200 1 0.000", "b": "NON_STATIONARY 1.400 3.100 2 0.001"},
    "pairs": {"measured": 1, "max_abs_pearson": 0.42, "median_abs_pearson": 0.42},
    "scaler": regime_space.SCALER_POLICY,
    "holdout": "the last 0.20 of the rows is held out and not seen by the fit",
    "measured_by": "m5phet.feature_metrics.v1",
}


def _dataset():
    return {"path": "/dev/null/x.csv", "sha256": "0" * 64, "rows_read": 40320, "time_column": "timestamp"}


def _hand_spec(**overrides):
    spec = choose_regimes.build_spec(task_id="t", features=["a", "b"], method="kmeans",
                                     parameters={"n_clusters": 3},
                                     holdout={"rule": "last_fraction", "fraction": 0.2},
                                     dataset=_dataset(), chosen_by="HAND")
    spec.update(overrides)
    return spec


# --- the space ------------------------------------------------------------------------------------------------------

def test_the_space_declares_the_four_methods_and_their_grids_as_key_label_pairs():
    assert [key for key, _label in regime_space.method_options()] == list(regime_space.available_methods())
    assert regime_space.available_methods() == ("agglomerative", "kmeans", "dbscan", "gaussian_mixture")
    assert regime_space.parameter_options("kmeans") == [[f"k{k}", f"k={k}"] for k in range(2, 7)]
    assert len(regime_space.parameter_options("agglomerative")) == 15
    assert len(regime_space.parameter_options("dbscan")) == 6
    for method in regime_space.available_methods():
        for key, label in regime_space.parameter_options(method):
            assert isinstance(key, str) and key.strip() and isinstance(label, str) and label.strip()
        assert regime_space.point_parameters(method, regime_space.parameter_options(method)[0][0])


def test_a_method_whose_class_this_installation_cannot_import_is_not_in_the_space(monkeypatch):
    import sklearn.cluster

    monkeypatch.delattr(sklearn.cluster, "DBSCAN")
    assert "dbscan" not in regime_space.available_methods()
    assert "dbscan" not in [key for key, _label in regime_space.method_options()]
    capability = regime_space.as_capability()
    assert "dbscan" not in [entry["method"] for entry in capability["methods"]]
    assert capability["not_importable"] == ["dbscan"]
    with pytest.raises(regime_space.RegimeSpaceError, match=regime_space.METHOD_NOT_IMPORTABLE):
        regime_space.parameter_options("dbscan")
    # and the other three are untouched: one absent class does not take the space with it
    assert regime_space.available_methods() == ("agglomerative", "kmeans", "gaussian_mixture")


def test_a_method_nobody_declared_is_refused_by_name():
    with pytest.raises(regime_space.RegimeSpaceError, match=regime_space.METHOD_NOT_DECLARED):
        regime_space.parameter_options("spectral")
    with pytest.raises(regime_space.RegimeSpaceError, match=regime_space.PARAMETERS_NOT_DECLARED):
        regime_space.validate_parameters("kmeans", {"n_clusters": 9})
    with pytest.raises(regime_space.RegimeSpaceError, match=regime_space.PARAMETERS_NOT_DECLARED):
        regime_space.point_parameters("kmeans", "k9")


def test_the_provider_publishes_the_space_and_a_callers_copy_is_its_own():
    provider = Provider()
    published = provider.capabilities()["regime_space"]
    assert published["schema"] == regime_space.SCHEMA
    assert [entry["method"] for entry in published["methods"]] == list(regime_space.available_methods())
    published["methods"].append({"method": "untrusted"})
    assert [entry["method"] for entry in provider.capabilities()["regime_space"]["methods"]] == \
           list(regime_space.available_methods())


# --- what reaches the model -------------------------------------------------------------------------------------------

def test_each_decision_is_asked_with_exactly_the_declared_options_and_the_second_is_bound_to_the_first():
    provider = FakeLaya({choose_regimes.METHOD_DECISION: "agglomerative",
                         f"{choose_regimes.PARAMETER_DECISION}_linkage": "average",
                         f"{choose_regimes.PARAMETER_DECISION}_n_clusters": "k3"})
    decide, engine = engine_with(provider)
    made = choose_regimes.choose(engine, PROFILE, as_of="2026-09-25T00:00:00+00:00")

    first, second = provider.seen
    assert list(first["questions"]) == [choose_regimes.METHOD_DECISION]
    assert first["questions"][choose_regimes.METHOD_DECISION]["options"] == regime_space.method_options()
    # agglomerative's grid has fifteen points, more than one choice may carry, so it is asked axis by axis -- each
    # axis with exactly the values the space declares, in one envelope
    assert sorted(second["questions"]) == choose_regimes.parameter_decision_names("agglomerative")
    for parameter, options in regime_space.parameter_axes("agglomerative"):
        assert second["questions"][f"{choose_regimes.PARAMETER_DECISION}_{parameter}"]["options"] == options
    # the parameter decision's state names the method that was just chosen, so its digest moves when the first moves
    assert "method: agglomerative" in second["state"]["news"]
    linkage = made["parameters"][f"{choose_regimes.PARAMETER_DECISION}_linkage"]["decision"]
    assert linkage["state_sha256"] != made["method"]["decision"]["state_sha256"]
    assert made["method"]["decision"]["chosen"] == "agglomerative"
    assert made["parameter_point"] == "average_k3"
    assert made["parameters_chosen"] == {"linkage": "average", "n_clusters": 3}
    assert made["method"]["decision"]["execution_authorized"] is False


def test_no_question_ever_carries_more_options_than_a_choice_may_hold():
    """The provider refuses a choice with more than twelve options by name; a grid larger than that is asked one
    declared axis at a time, and never as a subset of itself."""
    decide = workbench()
    for method in regime_space.available_methods():
        questions = choose_regimes.parameter_questions(method)
        offered = set()
        for question in questions.values():
            assert 2 <= len(question["options"]) <= regime_space.MAX_OPTIONS_PER_CHOICE
            assert decide._check_question(question) is None
            offered.update(key for key, _label in question["options"])
        if len(questions) == 1:
            assert offered == {point["key"] for point in regime_space.parameter_points(method)}
        else:
            # every declared value of every axis is offered: the composition covers the whole grid
            for parameter, options in regime_space.parameter_axes(method):
                assert {key for key, _label in options} <= offered
    assert len(choose_regimes.parameter_questions("kmeans")) == 1
    assert len(choose_regimes.parameter_questions("agglomerative")) == 2


def test_the_state_carries_measurements_and_no_rows_and_is_the_same_text_twice():
    decide = workbench()
    one = decide.decision_state(choose_regimes.METHOD_DECISION, PROFILE, decimals=choose_regimes.STATE_DECIMALS)
    two = decide.decision_state(choose_regimes.METHOD_DECISION, dict(reversed(list(PROFILE.items()))),
                                decimals=choose_regimes.STATE_DECIMALS)
    assert one == two and decide.state_sha256(one) == decide.state_sha256(two)
    assert "row_id" not in one and "40320" in one
    with pytest.raises(decide.DecisionError, match=decide.ROWS_IN_STATE):
        decide.decision_state("k", {"rows": [{"a": 1.0} for _ in range(decide.MAX_LIST_ITEMS + 1)]})


def test_an_answer_outside_the_declared_options_is_refused_and_no_spec_is_produced():
    provider = FakeLaya({choose_regimes.METHOD_DECISION: "spectral"})
    _decide, engine = engine_with(provider)
    with pytest.raises(choose_regimes.RegimeSpecError, match="CHOICE_OUTSIDE_OPTIONS"):
        choose_regimes.choose(engine, PROFILE)


def test_a_parameter_point_outside_the_chosen_methods_grid_is_refused():
    provider = FakeLaya({choose_regimes.METHOD_DECISION: "kmeans",
                         choose_regimes.PARAMETER_DECISION: "ward_k2"})
    _decide, engine = engine_with(provider)
    with pytest.raises(choose_regimes.RegimeSpecError, match="CHOICE_OUTSIDE_OPTIONS"):
        choose_regimes.choose(engine, PROFILE)


def test_a_decision_from_a_fixture_backend_is_refused_and_never_becomes_a_spec():
    provider = FakeLaya({choose_regimes.METHOD_DECISION: "kmeans"}, backend="fixture", fixture=True)
    _decide, engine = engine_with(provider)
    with pytest.raises(choose_regimes.RegimeSpecError, match="NON_MODEL_FIXTURE"):
        choose_regimes.choose(engine, PROFILE)


def test_the_spec_a_real_pair_of_decisions_produces_carries_their_digests(tmp_path):
    provider = FakeLaya({choose_regimes.METHOD_DECISION: "kmeans", choose_regimes.PARAMETER_DECISION: "k4"})
    decide, engine = engine_with(provider)
    made = choose_regimes.choose(engine, PROFILE, as_of="2026-09-25T00:00:00+00:00",
                                 record_dir=str(tmp_path / "records"))
    spec = choose_regimes.build_spec(
        task_id="t", features=["a", "b"], method=made["chosen_method"], parameters=made["parameters_chosen"],
        holdout={"rule": "last_fraction", "fraction": 0.2}, dataset=_dataset(),
        decisions={choose_regimes.METHOD_DECISION: made["method"], **made["parameters"]},
        chosen_by="LAYA_DECISION")
    assert spec["parameters"] == {"n_clusters": 4} and spec["parameter_point"] == "k4"
    entry = spec["decisions"][choose_regimes.METHOD_DECISION]
    assert entry["decision_sha256"] == decide.decision_sha256(made["method"]["decision"])
    assert entry["backend"] == "laya" and entry["calibration"].startswith("UNCALIBRATED")
    # the record on disk is the record the spec names, and it still verifies against its own file name
    written = decide.load(entry["record_path"])
    assert decide.decision_sha256(written) == entry["decision_sha256"]
    assert choose_regimes.write_spec(spec, tmp_path / "spec.json").is_file()
    assert choose_regimes.read_spec(tmp_path / "spec.json") == spec


# --- what a spec may not be ---------------------------------------------------------------------------------------------

def test_a_spec_naming_a_method_or_a_point_outside_the_space_is_refused_by_name():
    with pytest.raises(choose_regimes.RegimeSpecError, match=regime_space.METHOD_NOT_DECLARED):
        choose_regimes.validate_regime_spec(_hand_spec(method="spectral"))
    with pytest.raises(choose_regimes.RegimeSpecError, match=regime_space.PARAMETERS_NOT_DECLARED):
        choose_regimes.validate_regime_spec(_hand_spec(parameters={"n_clusters": 42}))
    with pytest.raises(choose_regimes.RegimeSpecError, match=regime_space.PARAMETERS_NOT_DECLARED):
        choose_regimes.validate_regime_spec(_hand_spec(parameter_point="k9"))


def test_a_spec_with_no_features_or_no_declared_holdout_or_no_scaler_policy_is_refused_by_name():
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.FEATURES_REQUIRED):
        choose_regimes.validate_regime_spec(_hand_spec(features=[]))
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.FEATURES_REQUIRED):
        choose_regimes.validate_regime_spec(_hand_spec(features=["a", "a"]))
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.HOLDOUT_NOT_DECLARED):
        choose_regimes.validate_regime_spec(_hand_spec(holdout={}))
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.HOLDOUT_NOT_DECLARED):
        choose_regimes.validate_regime_spec(_hand_spec(holdout={"rule": "last_fraction", "fraction": 1.0}))
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.SCALER_POLICY_NOT_DECLARED):
        choose_regimes.validate_regime_spec(_hand_spec(scaler={"policy": "whatever"}))
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.SCHEMA_MISMATCH):
        choose_regimes.validate_regime_spec(_hand_spec(schema="m5phet.regime_spec.v0"))


def test_a_spec_claiming_a_model_chose_it_must_carry_the_decisions_that_did():
    spec = _hand_spec()
    spec["provenance"] = dict(spec["provenance"], chosen_by="LAYA_DECISION")
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.DECISIONS_REQUIRED):
        choose_regimes.validate_regime_spec(spec)
    # and a hand-written spec may not carry decision records it did not come from
    invented = _hand_spec()
    invented["decisions"] = {choose_regimes.METHOD_DECISION: {"chosen": "kmeans"}}
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.DECISIONS_REQUIRED):
        choose_regimes.validate_regime_spec(invented)


def test_a_spec_whose_method_is_not_the_one_the_decision_chose_is_refused(tmp_path):
    provider = FakeLaya({choose_regimes.METHOD_DECISION: "kmeans", choose_regimes.PARAMETER_DECISION: "k4"})
    _decide, engine = engine_with(provider)
    made = choose_regimes.choose(engine, PROFILE, as_of="2026-09-25T00:00:00+00:00")
    with pytest.raises(choose_regimes.RegimeSpecError, match=choose_regimes.DECISIONS_REQUIRED):
        choose_regimes.build_spec(task_id="t", features=["a", "b"], method="agglomerative",
                                  parameters={"linkage": "ward", "n_clusters": 4},
                                  holdout={"rule": "last_fraction", "fraction": 0.2}, dataset=_dataset(),
                                  decisions={choose_regimes.METHOD_DECISION: made["method"], **made["parameters"]},
                                  chosen_by="LAYA_DECISION")
