"""What a person's ordinary words may select here, and what they may not.

The workbench resolves plain phrasing against the values a provider DECLARES and hands the result back as `parameters`.
The retained reference is one task fitted at one version, so that is exactly what is declared -- read from the manifest,
never by deserializing the fitted state -- and any other task or version is refused by name instead of being served by
the only reference there is. The hierarchy levels the manifest records are deliberately not declared: assignment returns
the whole cluster path, so a level would be understood and then ignored.
"""

import copy
import json
import os
from pathlib import Path

import joblib
import pytest

from feature_eng_m5phet.provider import (NO_DESCRIPTION, Provider, chat_request, chat_slots,
                                        metric_value, metric_vocabulary)
from feature_eng_m5phet.regimes import HierarchicalRegimes

TASK_ID = "test-regimes"


@pytest.fixture(autouse=True)
def isolated_operator_config(monkeypatch):
    monkeypatch.delenv("FEATURE_ENG_REGIMES_DEMO_DIR", raising=False)
    monkeypatch.delenv("FEATURE_ENG_REGIMES_STATE_PATH", raising=False)


@pytest.fixture
def rows():
    return [dict(row_id=i, x=x, y=y) for i, (x, y) in enumerate(
        [(-6, -3), (-5, -2), (-3, 2), (-2, 3), (3, -4), (4, -2), (8, 3), (9, 5)])]


@pytest.fixture
def model(rows):
    return HierarchicalRegimes.fit_reference(rows, features=["x", "y"], levels=[2, 4], task_id=TASK_ID)


@pytest.fixture
def demo(model, tmp_path, monkeypatch):
    """A retained reference as the demo CLI leaves it: the fitted state plus the receipt that describes it."""
    directory = tmp_path / "regimes"
    directory.mkdir()
    model.save(directory / "reference.joblib")
    (directory / "manifest.json").write_text(json.dumps(
        {"state_ref": str((directory / "reference.joblib").resolve()), "model_version": model.model_version,
         "metadata": model.metadata}), encoding="utf-8")
    monkeypatch.setenv("FEATURE_ENG_REGIMES_DEMO_DIR", str(directory))
    return directory


def config(model, path):
    return dict(provider=Provider.name, family="representation_unsupervised", output_kind="hierarchical_regimes",
                state=str(path), parameters=dict(model_version=model.model_version, task_id=TASK_ID),
                as_of="2026-09-24T00:00:00Z", input="json")


def interpreter():
    """The workbench's own resolver, wherever this machine keeps it; without it the contract cannot be exercised here."""
    path = os.environ.get("M5PHET_INTERPRET_PATH")
    if path and Path(path).is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location("m5phet_interpret_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return pytest.importorskip("m5phet.interpret")


# --- what is declared, and from which artifact ----------------------------------------------------------------------

def test_it_declares_the_task_and_version_the_manifest_records(demo, model):
    slots = chat_slots()
    assert [slot["name"] for slot in slots] == ["task_id", "model_version", "target_metric"]
    assert slots[0]["allowed"] == [TASK_ID]
    assert slots[1]["allowed"] == [model.model_version]
    assert all(slot["type"] == "string" for slot in slots)


def test_it_declares_a_metric_for_every_fitted_feature_in_the_forms_the_engine_reads(demo, model):
    """The person never types a column expression, so every admissible one is declared -- and only those."""
    from feature_eng_m5phet.questions import metric_problem
    metric = chat_slots()[-1]
    assert metric["name"] == "target_metric" and metric["required"] is False
    features = list(model.metadata["features"])
    assert metric["allowed"] == [NO_DESCRIPTION] + [metric_value(f, form) for f in features
                                                    for form in ("highest", "lowest", "> 0", "< 0")]
    for value in metric["allowed"][1:]:
        assert metric_problem(value, features) is None, f"{value!r} is declared but the engine would refuse it"
    assert metric["aliases"][NO_DESCRIPTION], "an assignment command must settle this slot by its own verb"


def test_a_manifest_that_does_not_say_what_was_fitted_declares_no_metric(demo, model):
    manifest = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
    manifest["metadata"] = {k: v for k, v in manifest["metadata"].items() if k != "features"}
    (demo / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert [slot["name"] for slot in chat_slots()] == ["task_id", "model_version"]


def test_the_hierarchy_levels_are_not_declared_because_nothing_could_honour_them(demo, model):
    """The manifest records levels 2 and 4; assignment returns every level, so declaring a choice would be a promise."""
    assert model.metadata["levels"] == [2, 4]
    assert "level" not in {slot["name"] for slot in chat_slots()}
    assert all(slot["type"] != "integer" for slot in chat_slots())


def test_declaring_a_vocabulary_never_deserializes_the_fitted_state(demo, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("chat_slots must read the manifest, not the joblib")

    monkeypatch.setattr(joblib, "load", forbidden)
    assert chat_slots()[0]["allowed"] == [TASK_ID]


def test_nothing_is_declared_without_a_retained_reference(tmp_path, monkeypatch):
    assert chat_slots() == [] and Provider().chat_slots() == []
    monkeypatch.setenv("FEATURE_ENG_REGIMES_DEMO_DIR", str(tmp_path))
    assert chat_slots() == [], "an empty directory declares nothing"
    assert Provider().chat_slots() == []


def test_an_unusable_manifest_declares_nothing(demo, monkeypatch):
    (demo / "manifest.json").write_text("{not json", encoding="utf-8")
    assert chat_slots() == []
    (demo / "manifest.json").write_text(json.dumps({"model_version": "short", "metadata": {"task_id": "t"}}),
                                        encoding="utf-8")
    assert chat_slots() == [], "a version that is not a sha256 is not a version this provider can be asked for"


def test_the_provider_declares_only_for_a_state_it_may_load(demo):
    assert Provider().chat_slots() == chat_slots()
    (demo / "reference.joblib").unlink()
    assert Provider().chat_slots() == [], "no loadable reference, no vocabulary"


# --- the declaration against the workbench's own rules ----------------------------------------------------------------

def test_the_declaration_satisfies_the_resolvers_contract(demo):
    assert interpreter()._check_slots(chat_slots())


@pytest.mark.parametrize("prompt", ["Assign hierarchical regimes", "assign regimes", "asigna regimenes",
                                    "muestra regímenes jerárquicos"])
def test_the_bounded_commands_resolve_both_slots_without_a_language_model(demo, model, prompt):
    """Every accepted command names the reference in words, so the deterministic pass settles it and no model is asked.

    `target_metric` is settled too, by the command's own verb, and settled to NO_DESCRIPTION: an assignment asks for no
    description. Leaving it unresolved would send every one of these sentences to the interpreter, which would then be
    choosing a metric nobody named."""
    interpret = interpreter()
    report = interpret.interpret(prompt, chat_slots())
    assert report["status"] == interpret.STATUS_OK, report.get("why")
    assert report["parameters"] == {"task_id": TASK_ID, "model_version": model.model_version,
                                   "target_metric": NO_DESCRIPTION}
    assert set(report["sources"].values()) == {"QUESTION_TEXT"}
    assert report["interpreter"] is None


def test_a_resolver_cannot_introduce_a_version_this_reference_was_not_fitted_at(demo):
    interpret = interpreter()

    class Inventing:
        available = True

        def identity(self):
            return {"command": "stub", "model": "stub", "available": True, "reading": "test double"}

        def propose(self, prompt, slots):
            return {slot["name"]: "f" * 64 for slot in slots}

    report = interpret.interpret("what does this say about the data?", chat_slots(), interpreter=Inventing())
    assert report["status"] == interpret.STATUS_UNSUPPORTED


# --- what resolved parameters may and may not do ------------------------------------------------------------------------

def without_request_id(request):
    request = copy.deepcopy(request)
    request.pop("request_id")
    return request


def test_no_parameters_keeps_the_current_path(model, rows, tmp_path):
    request = chat_request("assign hierarchical regimes", {"rows": rows}, config(model, tmp_path / "reference.joblib"))
    assert request["task_id"] == TASK_ID
    assert request["output_schema"] == {"targets": ["regimes"], "model_version": model.model_version}


def test_the_resolved_values_change_nothing_when_they_are_the_declared_ones(model, rows, tmp_path):
    settings = config(model, tmp_path / "reference.joblib")
    resolved = {"task_id": TASK_ID, "model_version": model.model_version}
    assert (without_request_id(chat_request("assign regimes", {"rows": rows}, settings, resolved))
            == without_request_id(chat_request("assign regimes", {"rows": rows}, settings)))


def test_resolved_values_complete_a_config_that_declares_none(model, rows, tmp_path):
    settings = config(model, tmp_path / "reference.joblib")
    settings.pop("parameters")
    request = chat_request("assign regimes", {"rows": rows}, settings,
                           {"task_id": TASK_ID, "model_version": model.model_version})
    assert request["task_id"] == TASK_ID
    assert request["output_schema"]["model_version"] == model.model_version


def test_a_config_that_declares_none_and_no_resolved_values_is_still_refused(model, rows, tmp_path):
    settings = config(model, tmp_path / "reference.joblib")
    settings.pop("parameters")
    with pytest.raises(ValueError, match="config must declare"):
        chat_request("assign regimes", {"rows": rows}, settings)


@pytest.mark.parametrize("name,value", [("task_id", "someone-elses-regimes-v2"), ("model_version", "a" * 64)])
def test_a_task_or_version_that_is_not_this_reference_is_refused_by_name(model, rows, tmp_path, name, value):
    settings = config(model, tmp_path / "reference.joblib")
    with pytest.raises(ValueError) as refusal:
        chat_request("assign regimes", {"rows": rows}, settings, {name: value})
    assert value in str(refusal.value), "the value that was asked for must appear in the refusal"
    assert settings["parameters"][name] in str(refusal.value), "and so must the one this reference actually has"


def test_an_undeclared_parameter_is_refused(model, rows, tmp_path):
    with pytest.raises(ValueError, match="undeclared parameters"):
        chat_request("assign regimes", {"rows": rows}, config(model, tmp_path / "reference.joblib"), {"level": 2})


def test_resolved_parameters_must_be_a_mapping(model, rows, tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        chat_request("assign regimes", {"rows": rows}, config(model, tmp_path / "reference.joblib"), [TASK_ID])


def test_resolved_parameters_cannot_widen_an_unsupported_prompt(model, rows, tmp_path):
    """The prompt is checked first; resolved values are not a way in for a command this provider does not accept."""
    with pytest.raises(ValueError, match="prompt"):
        chat_request("fit a new model", {"rows": rows}, config(model, tmp_path / "reference.joblib"),
                     {"task_id": TASK_ID, "model_version": model.model_version})


# --- end to end over the retained artifacts ---------------------------------------------------------------------------

def test_the_declared_values_drive_a_request_the_provider_accepts(demo, model, rows):
    provider = Provider()
    slots = provider.chat_slots()
    resolved = {slot["name"]: slot["allowed"][0] for slot in slots}
    settings = config(model, demo / "reference.joblib")
    request = provider.chat_request("assign hierarchical regimes", {"rows": rows}, settings, resolved)
    state = provider.load(str((demo / "reference.joblib").resolve()))
    assert provider.infer(request, state)["outputs"]["regimes"]["payload"] == model.assign(rows)


# --- a command must be named, not typed exactly ------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "assign hierarchical regimes to these rows",
    "please assign regimes for this data",
    "asigna los regimenes jerarquicos a estas filas",
    "\u00bfpuedes asignar regimenes a estos datos?",
    "Assign Hierarchical Regimes",
])
def test_an_ordinary_sentence_naming_the_command_is_accepted(prompt, demo, model, rows, tmp_path):
    """Requiring the whole sentence to EQUAL a command refused every ordinary way of asking for the same thing, which
    reads as a broken product rather than as a boundary."""
    request = chat_request(prompt, {"rows": rows}, config(model, tmp_path / "regimes" / "reference.joblib"))
    assert request["operation"] == "infer" and request["output_kind"] == "hierarchical_regimes"


@pytest.mark.parametrize("prompt", ["forecast the price tomorrow", "train a new model on this data", "", "   "])
def test_a_prompt_naming_no_command_is_still_refused(prompt, demo, model, rows, tmp_path):
    """The boundary this check exists for is untouched: one operation, and a sentence cannot ask for another."""
    with pytest.raises(ValueError, match="this adapter performs one operation"):
        chat_request(prompt, {"rows": rows}, config(model, tmp_path / "regimes" / "reference.joblib"))


# --- a person's own words, in either language, instead of a column expression ------------------------------------------
#
# The product this serves is one sentence long: "describe el grupo de velas con cuerpo alto" must be understood, by the
# words, without anyone knowing that the column is called `body_pipettes`. So these tests use a reference fitted on the
# demo's own feature names -- the phrasings are built from THOSE, and a fixture named x and y could not show it.

OHLC_TASK = "ohlc-regimes-words"
OHLC_FEATURES = ["body_pipettes", "range_pipettes"]
SPANISH = "describe el grupo de velas con cuerpo alto"
ENGLISH = "describe the cluster with a large body"


@pytest.fixture
def ohlc_rows():
    return [dict(row_id=f"r{i}", body_pipettes=body, range_pipettes=span) for i, (body, span) in enumerate(
        [(-320.0, 700.0), (-280.0, 660.0), (-150.0, 400.0), (-90.0, 360.0),
         (120.0, 380.0), (180.0, 420.0), (340.0, 690.0), (410.0, 740.0)])]


@pytest.fixture
def ohlc(ohlc_rows, tmp_path, monkeypatch):
    """A retained reference fitted on the demo's own feature names, with the receipt the declaration is read from."""
    model = HierarchicalRegimes.fit_reference(ohlc_rows, features=OHLC_FEATURES, levels=[2, 4], task_id=OHLC_TASK)
    directory = tmp_path / "ohlc-regimes"
    directory.mkdir()
    model.save(directory / "reference.joblib")
    (directory / "manifest.json").write_text(json.dumps(
        {"state_ref": str((directory / "reference.joblib").resolve()), "model_version": model.model_version,
         "metadata": model.metadata}), encoding="utf-8")
    monkeypatch.setenv("FEATURE_ENG_REGIMES_DEMO_DIR", str(directory))
    return directory, model


def ohlc_config(directory, model):
    return dict(provider=Provider.name, family="representation_unsupervised", output_kind="hierarchical_regimes",
                state=str((directory / "reference.joblib").resolve()), input="json", as_of="2026-09-24T00:00:00Z",
                parameters=dict(task_id=OHLC_TASK, model_version=model.model_version))


def test_two_wordings_in_two_languages_settle_on_the_same_metric_by_words_alone(ohlc):
    """The point of the whole package: nobody types `body_pipettes > 0`, and no model is consulted to avoid it."""
    interpret = interpreter()
    slots = chat_slots()
    spanish = interpret.interpret(SPANISH, slots)
    english = interpret.interpret(ENGLISH, slots)
    assert spanish["status"] == interpret.STATUS_OK, spanish.get("why")
    assert english["status"] == interpret.STATUS_OK, english.get("why")
    assert spanish["parameters"]["target_metric"] == english["parameters"]["target_metric"] == "highest body_pipettes"
    assert set(spanish["sources"].values()) == {"QUESTION_TEXT"}
    assert set(english["sources"].values()) == {"QUESTION_TEXT"}
    assert spanish["interpreter"] is None and english["interpreter"] is None, "a model was consulted for nothing"


@pytest.mark.parametrize("prompt,expected", [
    ("describe el grupo de velas con cuerpo alto", "highest body_pipettes"),
    ("describe the cluster with a large body", "highest body_pipettes"),
    ("describe the cluster with the largest body", "highest body_pipettes"),
    ("describe el grupo con cuerpo bajo", "lowest body_pipettes"),
    ("describe los grupos con rango amplio", "highest range_pipettes"),
    ("describe the cluster with the narrowest range", "lowest range_pipettes"),
    ("describe el grupo con cuerpo positivo", "body_pipettes > 0"),
    ("describe the cluster with a negative body", "body_pipettes < 0"),
])
def test_ordinary_phrasings_resolve_to_the_metric_they_name(ohlc, prompt, expected):
    interpret = interpreter()
    report = interpret.interpret(prompt, chat_slots())
    assert report["status"] == interpret.STATUS_OK, report.get("why")
    assert report["parameters"]["target_metric"] == expected
    assert report["interpreter"] is None


def test_a_feature_the_reference_was_not_fitted_with_is_refused_by_name(ohlc):
    """Asked for the volume of a reference fitted on body and range, an interpreter would pick one of the two."""
    interpret = interpreter()
    report = interpret.interpret("describe el grupo con mayor volumen", chat_slots())
    assert report["status"] == interpret.STATUS_UNSUPPORTED
    assert "volumen" in report["why"] and "body_pipettes" in report["why"]


def test_a_metric_naming_a_column_the_rows_do_not_carry_is_refused_while_the_request_is_built(ohlc, ohlc_rows):
    directory, model = ohlc
    with pytest.raises(ValueError) as refusal:
        chat_request("describe cluster", {"rows": ohlc_rows}, ohlc_config(directory, model),
                     {"task_id": OHLC_TASK, "model_version": model.model_version, "target_metric": "highest volume"})
    assert "volume" in str(refusal.value) and "body_pipettes" in str(refusal.value)


def test_the_resolved_metric_is_used_and_the_engine_answers_it(ohlc, ohlc_rows):
    """A value resolved from a person's words and then dropped is exactly the quiet mistake this adapter refuses."""
    directory, model = ohlc
    provider = Provider()
    resolved = interpreter().interpret(SPANISH, provider.chat_slots())["parameters"]
    request = provider.chat_request(SPANISH, {"rows": ohlc_rows}, ohlc_config(directory, model), resolved)
    assert request["output_schema"]["description"] == {"target_metric": "highest body_pipettes"}
    state = provider.load(str((directory / "reference.joblib").resolve()))
    payload = provider.infer(request, state)["outputs"]["regimes"]["payload"]
    described = payload["cluster_description"]
    assert described["type"] == "cluster_description" and described["metric_form"] == "highest"
    assert described["target_metric"] == "highest body_pipettes"
    assert payload["rows"] == model.assign(ohlc_rows)["rows"], "the assignment is unchanged by the description"
    from feature_eng_m5phet.questions import describe_cluster
    assert described == describe_cluster(state["model"], ohlc_rows, "highest body_pipettes")


def test_an_assignment_command_is_unchanged_and_carries_no_description(ohlc, ohlc_rows):
    directory, model = ohlc
    provider = Provider()
    resolved = interpreter().interpret("assign hierarchical regimes to these rows", provider.chat_slots())["parameters"]
    assert resolved["target_metric"] == NO_DESCRIPTION
    request = provider.chat_request("assign hierarchical regimes to these rows", {"rows": ohlc_rows},
                                    ohlc_config(directory, model), resolved)
    assert request["output_schema"] == {"targets": ["regimes"], "model_version": model.model_version}
    payload = provider.infer(request, provider.load(request["fitted_state_ref"]))["outputs"]["regimes"]["payload"]
    assert "cluster_description" not in payload


def test_a_description_command_with_no_metric_is_refused_rather_than_given_a_metric(ohlc, ohlc_rows):
    directory, model = ohlc
    with pytest.raises(ValueError, match="no target_metric was resolved"):
        chat_request("describe the cluster", {"rows": ohlc_rows}, ohlc_config(directory, model),
                     {"task_id": OHLC_TASK, "model_version": model.model_version})


def test_an_assignment_command_with_a_metric_is_refused_as_two_requests(ohlc, ohlc_rows):
    directory, model = ohlc
    with pytest.raises(ValueError) as refusal:
        chat_request("assign regimes", {"rows": ohlc_rows}, ohlc_config(directory, model),
                     {"task_id": OHLC_TASK, "model_version": model.model_version,
                      "target_metric": "highest body_pipettes"})
    assert "two requests" in str(refusal.value)


def test_a_sentence_asking_for_both_is_refused_with_both_candidates_named(ohlc):
    """"muestra el grupo con cuerpo alto" names an assignment AND a description; the second is not added silently."""
    interpret = interpreter()
    report = interpret.interpret("muestra el grupo con cuerpo alto", chat_slots())
    assert report["status"] == interpret.STATUS_AMBIGUOUS
    assert NO_DESCRIPTION in report["why"] and "highest body_pipettes" in report["why"]


def test_the_words_that_named_the_metric_do_not_make_the_sentence_a_second_request(ohlc, ohlc_rows):
    """The FILLER rule is untouched: the metric's own words are accounted for, anything else still leaves a word."""
    directory, model = ohlc
    identity = {"task_id": OHLC_TASK, "model_version": model.model_version,
                "target_metric": "highest body_pipettes"}
    request = chat_request(SPANISH, {"rows": ohlc_rows}, ohlc_config(directory, model), identity)
    assert request["operation"] == "infer"
    with pytest.raises(ValueError, match="this adapter performs one operation"):
        chat_request("describe el grupo con cuerpo alto y predice el precio", {"rows": ohlc_rows},
                     ohlc_config(directory, model), identity)


def test_a_phrasing_that_could_name_two_metrics_is_not_declared_at_all():
    """Two features sharing a first word keep their full names and lose the short one; a refusal is not a vocabulary."""
    vocabulary = metric_vocabulary(["body_pipettes", "body_pct"])
    assert "large body" not in vocabulary["highest body_pipettes"]
    assert "large body pipettes" in vocabulary["highest body_pipettes"]
    assert "large body pct" in vocabulary["highest body_pct"]
