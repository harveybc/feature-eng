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

from feature_eng_m5phet.provider import Provider, chat_request, chat_slots
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
    assert [slot["name"] for slot in slots] == ["task_id", "model_version"]
    assert slots[0]["allowed"] == [TASK_ID]
    assert slots[1]["allowed"] == [model.model_version]
    assert all(slot["type"] == "string" for slot in slots)


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
    """Every accepted command names the reference in words, so the deterministic pass settles it and no model is asked."""
    interpret = interpreter()
    report = interpret.interpret(prompt, chat_slots())
    assert report["status"] == interpret.STATUS_OK, report.get("why")
    assert report["parameters"] == {"task_id": TASK_ID, "model_version": model.model_version}
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
