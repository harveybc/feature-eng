"""WP19, the explicit fit: a new fitted reference, in the served format, with the demo one left exactly as it was."""

import csv
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from feature_eng_m5phet import choose_regimes, design, fit_regimes
from feature_eng_m5phet.provider import Provider
from feature_eng_m5phet.regimes import HierarchicalRegimes, SpecRegimes

DEMO = Path(os.environ.get("FEATURE_ENG_REGIMES_DEMO_DIR")
            or Path.home() / ".local" / "state" / "m5phet" / "examples" / "regimes")


def write_blobs(path, *, rows=600, centres=((0.0, 0.0), (8.0, 8.0), (16.0, 0.0)), spread=0.4, seed=7):
    """A dataset with three separated blobs and a regular one-minute clock. Synthetic on purpose: the point of these
    tests is the plumbing, and a real series would let a reader mistake an index here for a finding."""
    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 1)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "a", "b"])
        for index in range(rows):
            centre = centres[index % len(centres)]
            writer.writerow([(start + timedelta(minutes=index)).strftime("%Y-%m-%d %H:%M:%S"),
                             f"{centre[0] + rng.normal(0, spread):.6f}", f"{centre[1] + rng.normal(0, spread):.6f}"])
    return Path(path)


def hand_spec(data, *, method="kmeans", parameters=None, fraction=0.2, task_id="blobs"):
    table = design.read_table(data)
    return choose_regimes.build_spec(
        task_id=task_id, features=["a", "b"], method=method,
        parameters=parameters or {"n_clusters": 3},
        holdout={"rule": "last_fraction", "fraction": fraction},
        dataset={"path": table["path"], "sha256": table["sha256"], "rows_read": table["rows_read"],
                 "time_column": table["time_column"]},
        chosen_by="HAND")


@pytest.fixture
def blobs(tmp_path):
    return write_blobs(tmp_path / "blobs.csv")


def digest_tree(directory):
    """Every file under a directory, by name and content. What "untouched" means, checked instead of asserted."""
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


# --- what the fit writes ------------------------------------------------------------------------------------------

def test_the_fit_writes_a_reference_the_provider_loads_and_a_manifest_that_names_its_spec(tmp_path, blobs,
                                                                                          monkeypatch):
    spec = hand_spec(blobs)
    manifest = fit_regimes.fit(spec, blobs, tmp_path / "ref")
    reference = tmp_path / "ref" / fit_regimes.REFERENCE_FILE

    assert reference.is_file()
    model = HierarchicalRegimes.load(str(reference))
    assert isinstance(model, SpecRegimes)
    assert model.model_version == manifest["model_version"] == model._fingerprint()
    assert model.metadata["method"] == "kmeans" and model.metadata["parameters"] == {"n_clusters": 3}
    assert model.metadata["levels"] == [3]
    assert manifest["spec"] == spec and manifest["spec_sha256"] == fit_regimes.spec_sha256(spec)
    assert manifest["holdout"]["rows"] == 120 and manifest["fit_rows"]["training_rows"] == 480

    # and the provider serves it once the operator configures it, through its own loading gate
    monkeypatch.setenv("FEATURE_ENG_REGIMES_STATE_PATH", str(reference))
    monkeypatch.delenv("FEATURE_ENG_REGIMES_DEMO_DIR", raising=False)
    loaded = Provider().load(str(reference.resolve()))
    assert loaded["digest"] == manifest["model_version"] and loaded["task_id"] == "blobs"


def test_the_fit_reads_a_stride_over_the_training_portion_and_never_the_holdout(tmp_path, blobs):
    spec = hand_spec(blobs)
    spec["fit_rows"] = {"rule": choose_regimes.FIT_ROW_RULE, "limit": 100}
    manifest = fit_regimes.fit(choose_regimes.validate_regime_spec(spec), blobs, tmp_path / "ref")
    selection = manifest["fit_rows"]
    assert selection["training_rows"] == 480 and selection["stride"] == 5 and selection["fitted_on"] == 96
    # the holdout is the last fifth of the file and the reference's rows all come from before it
    assert manifest["holdout"]["first_row_id"] == "480" and manifest["holdout"]["last_row_id"] == "599"
    assert max(int(row_id) for row_id in manifest["metadata"]["reference_row_ids"]) < 480


def test_a_decision_backed_spec_puts_its_digests_into_the_reference_itself(tmp_path, blobs):
    spec = hand_spec(blobs)
    spec["provenance"] = dict(spec["provenance"], chosen_by="LAYA_DECISION")
    spec["decisions"] = {
        choose_regimes.METHOD_DECISION: {"chosen": "kmeans", "decision_sha256": "a" * 64, "state_sha256": "b" * 64},
        choose_regimes.PARAMETER_DECISION: {"chosen": "k3", "decision_sha256": "c" * 64, "state_sha256": "d" * 64}}
    manifest = fit_regimes.fit(choose_regimes.validate_regime_spec(spec), blobs, tmp_path / "ref")
    assert manifest["metadata"]["decisions"] == {choose_regimes.METHOD_DECISION: "a" * 64,
                                                 choose_regimes.PARAMETER_DECISION: "c" * 64}
    assert manifest["metadata"]["chosen_by"] == "m5phet.decide"
    # a reference whose decisions changed is a different reference: the digest covers them
    other = dict(spec, decisions=dict(spec["decisions"]))
    other["decisions"][choose_regimes.METHOD_DECISION] = dict(spec["decisions"][choose_regimes.METHOD_DECISION],
                                                              decision_sha256="e" * 64)
    assert fit_regimes.fit(other, blobs, tmp_path / "other")["model_version"] != manifest["model_version"]


def test_every_declared_method_that_imports_can_actually_be_fitted(tmp_path, blobs):
    fitted = {}
    for method, parameters in (("agglomerative", {"linkage": "average", "n_clusters": 3}),
                               ("kmeans", {"n_clusters": 3}),
                               ("dbscan", {"eps": 0.5, "min_samples": 5}),
                               ("gaussian_mixture", {"n_components": 3})):
        spec = hand_spec(blobs, method=method, parameters=parameters, task_id=f"blobs-{method}")
        manifest = fit_regimes.fit(spec, blobs, tmp_path / method)
        fitted[method] = manifest["metadata"]["cluster_labels"]
    assert all(len(labels) >= 2 for labels in fitted.values()), fitted


def test_a_fit_that_puts_every_row_in_one_cluster_is_refused_by_name(tmp_path):
    """A reference that assigns one regime to everything describes nothing, and is refused rather than served."""
    flat = write_blobs(tmp_path / "flat.csv", centres=((0.0, 0.0),), spread=0.0)
    spec = hand_spec(flat, method="dbscan", parameters={"eps": 1.0, "min_samples": 3}, task_id="flat")
    with pytest.raises(ValueError, match="DEGENERATE_FIT"):
        fit_regimes.fit(spec, flat, tmp_path / "degenerate")
    assert not (tmp_path / "degenerate" / fit_regimes.REFERENCE_FILE).exists()


# --- what the fit refuses -------------------------------------------------------------------------------------------

def test_the_demo_reference_is_protected_by_name_and_is_not_touched_by_a_fit_elsewhere(tmp_path, blobs):
    spec = hand_spec(blobs)
    with pytest.raises(fit_regimes.FitRefusal, match=fit_regimes.DEMO_REFERENCE_PROTECTED):
        fit_regimes.fit(spec, blobs, fit_regimes.DEFAULT_DEMO_DIR)
    with pytest.raises(fit_regimes.FitRefusal, match=fit_regimes.DEMO_REFERENCE_PROTECTED):
        fit_regimes.fit(spec, blobs, fit_regimes.DEFAULT_DEMO_DIR / "deeper")
    if not (DEMO / fit_regimes.REFERENCE_FILE).is_file():
        pytest.skip(f"no retained demo reference under {DEMO} to hash")
    before = digest_tree(DEMO)
    fit_regimes.fit(spec, blobs, tmp_path / "elsewhere")
    assert digest_tree(DEMO) == before


def test_a_fitted_reference_is_never_overwritten(tmp_path, blobs):
    spec = hand_spec(blobs)
    fit_regimes.fit(spec, blobs, tmp_path / "ref")
    with pytest.raises(fit_regimes.FitRefusal, match=fit_regimes.REFERENCE_EXISTS):
        fit_regimes.fit(spec, blobs, tmp_path / "ref")


def test_a_file_that_is_not_the_one_the_spec_was_written_against_is_refused(tmp_path, blobs):
    spec = hand_spec(blobs)
    other = write_blobs(tmp_path / "other.csv", seed=11)
    with pytest.raises(fit_regimes.FitRefusal, match=fit_regimes.DATASET_MISMATCH):
        fit_regimes.fit(spec, other, tmp_path / "ref")


def test_a_spec_the_space_refuses_never_reaches_a_fit(tmp_path, blobs):
    spec = hand_spec(blobs)
    spec["parameters"] = {"n_clusters": 42}
    with pytest.raises(choose_regimes.RegimeSpecError, match="PARAMETERS_NOT_DECLARED"):
        fit_regimes.fit(spec, blobs, tmp_path / "ref")
    assert not (tmp_path / "ref").exists()


# --- two references, one envelope ---------------------------------------------------------------------------------------

def test_two_fitted_references_coexist_and_answer_the_same_envelope(tmp_path, blobs, monkeypatch):
    """The smallest change that lets two references coexist is none at all: the provider already reads two operator
    variables, and an envelope naming `state_ref` says which reference answers."""
    first = fit_regimes.fit(hand_spec(blobs, task_id="blobs-kmeans"), blobs, tmp_path / "kmeans")
    second = fit_regimes.fit(hand_spec(blobs, method="agglomerative",
                                       parameters={"linkage": "average", "n_clusters": 4}, task_id="blobs-ward"),
                             blobs, tmp_path / "agglomerative")
    monkeypatch.setenv("FEATURE_ENG_REGIMES_DEMO_DIR", str(tmp_path / "kmeans"))
    monkeypatch.setenv("FEATURE_ENG_REGIMES_STATE_PATH", second["state_ref"])
    provider = Provider()
    assert sorted(provider.capabilities()["known_states"]) == sorted([first["state_ref"], second["state_ref"]])

    rows, _table = fit_regimes.read_rows(blobs, choose_regimes.validate_regime_spec(first["spec"]))
    envelope = {"seg": {"type": "clustering", "method": "auto"}}
    answers = {}
    for name, state_ref in (("kmeans", first["state_ref"]), ("agglomerative", second["state_ref"])):
        answer = provider.answer_questions({"state_ref": state_ref, "features": ["a", "b"]}, envelope,
                                           {"rows": rows[:200]}, None)["seg"]
        assert answer["status"] == "OK", answer
        answers[name] = answer
    assert answers["kmeans"]["reference"]["model_version"] == first["model_version"]
    assert answers["agglomerative"]["reference"]["model_version"] == second["model_version"]
    assert answers["kmeans"]["optimal_k"] == [3] and answers["agglomerative"]["optimal_k"] == [4]
    # with two states configured, an envelope that names neither is refused rather than served by whichever is first
    refused = provider.answer_questions({"features": ["a", "b"]}, envelope, {"rows": rows[:10]}, None)["seg"]
    assert refused["status"] == "REFUSED" and refused["refusal"] == "STATE_REQUIRED"


def test_a_reference_answers_the_method_word_it_was_fitted_with_and_refuses_another(tmp_path, blobs, monkeypatch):
    manifest = fit_regimes.fit(hand_spec(blobs), blobs, tmp_path / "kmeans")
    monkeypatch.setenv("FEATURE_ENG_REGIMES_STATE_PATH", manifest["state_ref"])
    monkeypatch.delenv("FEATURE_ENG_REGIMES_DEMO_DIR", raising=False)
    provider = Provider()
    rows, _table = fit_regimes.read_rows(blobs, choose_regimes.validate_regime_spec(manifest["spec"]))
    state = {"state_ref": manifest["state_ref"], "features": ["a", "b"]}
    for word, expected in (("kmeans", "OK"), ("auto", "OK"), ("ward", "REFUSED")):
        answer = provider.answer_questions(state, {"seg": {"type": "clustering", "method": word}},
                                           {"rows": rows[:50]}, None)["seg"]
        assert answer["status"] == expected, (word, answer)
