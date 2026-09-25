"""WP19, the evaluation: internal indices on the holdout, and no accuracy anywhere.

The dataset is three synthetic blobs, so the indices have a known right answer as *plumbing* -- separated blobs give a
high silhouette and a stability of one. That is a check on the code, not a finding about regimes: `regime_accuracy`
is refused in every report this module writes, because no row carries a correct regime.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from feature_eng_m5phet import choose_regimes, evaluate_regimes, fit_regimes

from tests.test_fit_regimes import hand_spec, write_blobs                                        # noqa: F401


M5PHET = Path(os.environ.get("M5PHET_PATH") or Path.home() / "Documents" / "GitHub" / "M5PHET")


@pytest.fixture
def blobs(tmp_path):
    return write_blobs(tmp_path / "blobs.csv")


@pytest.fixture
def reference(tmp_path, blobs):
    manifest = fit_regimes.fit(hand_spec(blobs), blobs, tmp_path / "ref")
    return {"dir": tmp_path / "ref", "manifest": manifest, "data": blobs}


def report_of(reference, **kwargs):
    return evaluate_regimes.evaluate(reference["dir"], reference["data"], generated_at="2026-09-25T00:00:00Z",
                                     sealed_at="2026-09-25T00:00:00Z", **kwargs)


# --- the indices ------------------------------------------------------------------------------------------------------

def test_three_separated_blobs_give_a_high_silhouette_and_a_stability_of_one(reference):
    values = report_of(reference)["metric_sets"][0]["values"]
    assert values["silhouette"] > 0.5
    assert values["davies_bouldin"] < 1.0
    assert values["stability_index"] > 0.99 and values["adjusted_stability_index"] > 0.99
    assert set(values) == set(evaluate_regimes.DECLARED_METRICS)


def test_only_the_holdout_rows_are_scored(reference):
    report = report_of(reference)
    counts = report["metric_sets"][0]["counts"]
    assert counts["scored_rows"] == counts["holdout_rows"] == 120 == report["sealed_row_count"]
    assert counts["dropped_nonfinite"] == 0
    # the rows the fit saw are not among them: the seal is over the last fifth of the file
    reference_rows = {int(row_id) for row_id in reference["manifest"]["metadata"]["reference_row_ids"]}
    assert max(reference_rows) < 480


def test_the_stability_index_comes_from_two_refits_of_the_same_spec_on_the_two_halves(reference):
    detail = report_of(reference)["stability"]
    assert detail["status"] == "OK"
    assert [half["rows"] for half in detail["halves"]] == [60, 60]
    assert detail["halves"][0]["model_version"] != detail["halves"][1]["model_version"]
    assert "reproducibility, not correctness" in detail["rule"]


def test_an_index_that_is_not_defined_on_these_rows_is_omitted_with_its_reason(tmp_path):
    """A reference that lands every holdout row in one cluster has no silhouette; the report says so and carries no
    number in its place."""
    flat = write_blobs(tmp_path / "flat.csv", centres=((0.0, 0.0), (0.0, 0.0), (20.0, 20.0)), spread=0.01)
    manifest = fit_regimes.fit(hand_spec(flat, method="kmeans", parameters={"n_clusters": 2}, task_id="flat"),
                               flat, tmp_path / "ref")
    report = evaluate_regimes.evaluate(tmp_path / "ref", flat, generated_at="2026-09-25T00:00:00Z")
    values = report["metric_sets"][0]["values"]
    if "silhouette" not in values:
        assert evaluate_regimes.INDEX_NOT_DEFINED in report["indices_omitted"]["silhouette"]
    assert manifest["metadata"]["levels"] == [2]


# --- what the report says it is not ---------------------------------------------------------------------------------

def test_regime_accuracy_is_refused_by_name_everywhere_in_the_report(reference):
    report = report_of(reference)
    assert report["regime_accuracy"] == evaluate_regimes.REGIME_ACCURACY == "REFUSED_NO_GROUND_TRUTH"
    assert "no ground truth" in report["regime_accuracy_reason"]
    assert any("regime_accuracy" in statement for statement in report["statements"])
    assert evaluate_regimes.REGIME_ACCURACY_REASON in report["metric_sets"][0]["notes"]
    assert not any(key in report["metric_sets"][0]["values"] for key in ("accuracy", "regime_accuracy", "skill"))
    assert report["label_provenance"] == "AUTHOR_WRITTEN_SMOKE" and report["label_source"] is None
    assert "AUTHOR_WRITTEN_SMOKE" in report["flags"]


def test_a_directory_that_is_not_a_reference_is_refused_by_name(tmp_path, blobs):
    with pytest.raises(evaluate_regimes.EvaluationRefusal, match=evaluate_regimes.NOT_A_REFERENCE_DIRECTORY):
        evaluate_regimes.evaluate(tmp_path, blobs)


def test_a_dataset_that_is_not_the_sealed_one_is_refused(tmp_path, reference):
    other = write_blobs(tmp_path / "other.csv", seed=99)
    with pytest.raises(fit_regimes.FitRefusal, match=fit_regimes.DATASET_MISMATCH):
        evaluate_regimes.evaluate(reference["dir"], other)


# --- the shape the stage table reads ---------------------------------------------------------------------------------

def test_two_references_on_the_same_holdout_carry_the_same_seal_and_differ_only_in_what_they_fitted(tmp_path, blobs):
    first = fit_regimes.fit(hand_spec(blobs, task_id="kmeans"), blobs, tmp_path / "a")
    second = fit_regimes.fit(hand_spec(blobs, method="agglomerative",
                                       parameters={"linkage": "average", "n_clusters": 3}, task_id="agglomerative"),
                             blobs, tmp_path / "b")
    reports = [evaluate_regimes.evaluate(tmp_path / name, blobs, stage=stage,
                                         generated_at="2026-09-25T00:00:00Z", sealed_at="2026-09-25T00:00:00Z")
               for name, stage in (("a", "laya_chosen"), ("b", "hand_baseline"))]
    assert reports[0]["corpus_seal"] == reports[1]["corpus_seal"]
    assert reports[0]["protocol_digest"] == reports[1]["protocol_digest"]
    assert reports[0]["reference"]["model_version"] != reports[1]["reference"]["model_version"]
    assert [report["stage"] for report in reports] == ["laya_chosen", "hand_baseline"]
    assert first["model_version"] != second["model_version"]


def test_the_report_is_read_by_the_stage_comparison_generator(tmp_path, reference):
    """The real generator, not a copy of its rules: it either reads this report or the test fails."""
    generator = M5PHET / "evaluation" / "compare_stages.py"
    if not generator.is_file():
        pytest.skip(f"no M5PHET checkout at {M5PHET}")
    path = tmp_path / "laya_chosen.json"
    path.write_text(json.dumps(report_of(reference, stage="laya_chosen"), indent=2, sort_keys=True))
    done = subprocess.run([sys.executable, "-m", "evaluation.compare_stages", f"--report=laya_chosen={path}"],
                          cwd=str(M5PHET), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "## Area: regimes" in done.stdout
    # the area's quality stays refused in the table, which is the point of writing the report this way
    assert "NO_NEW_MEASUREMENT" in done.stdout and "regime_accuracy is refused" in done.stdout


def test_the_protocol_digest_and_the_seal_are_the_ones_the_evaluation_package_computes(reference):
    """Guard against drift: this module rebuilds the protocol and the seal rather than importing a package that is
    not installed here. If the real one is reachable, the two must agree byte for byte."""
    source = os.environ.get("M5PHET_EVALUATION_SRC_PATH") or str(M5PHET / "evaluation" / "src")
    if not Path(source, "m5phet_evaluation", "protocol.py").is_file():
        pytest.skip(f"no m5phet_evaluation source at {source}")
    if source not in sys.path:
        sys.path.insert(0, source)
    protocol_module = pytest.importorskip("m5phet_evaluation.protocol")
    freeze = pytest.importorskip("m5phet_evaluation.freeze")

    report = report_of(reference)
    spec = choose_regimes.validate_regime_spec(reference["manifest"]["spec"])
    rows, _table = fit_regimes.read_rows(reference["data"], spec)
    _portion, held = fit_regimes.holdout_split(rows, spec)
    declared = evaluate_regimes._protocol(held, spec=spec,
                                         minimum_rows=evaluate_regimes.DEFAULT_MINIMUM_ROWS)
    real = protocol_module.EvaluationProtocol(**declared)
    assert real.digest == report["protocol_digest"]
    labels = {row["row_id"]: [row[name] for name in spec["features"]] for row in held}
    seal = freeze.seal_corpus(labels, protocol=real, sealed_at="2026-09-25T00:00:00Z")
    assert seal.seal == report["corpus_seal"] and seal.row_count == report["sealed_row_count"]
