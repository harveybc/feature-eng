"""The evaluation reports of an event study: two stages over one sealed corpus, and what is refused instead.

The point of these reports is not the numbers -- the projections document already carries the held-out errors. It is
that the two numbers can be shown to have been computed over the SAME rows with the SAME labels, which is what a
protocol digest and a corpus seal are for. So the tests here are about identity: the two stages share a seal, the
seal covers exactly the held-out events, the naive stage's error is the baseline's error by construction, the
reports are byte-identical across runs, and a projection that never fitted produces no report at all.

The study under test is the synthetic one from `test_event_projections.py`: a market with a planted response, so the
projection genuinely has something to be better at than the naive.
"""

import json

import pytest

from feature_eng_m5phet import evaluate_events

from tests.test_event_projections import additive_planter, run as estimate_run, world

pytest.importorskip("m5phet_evaluation",
                    reason="the evaluation package is not installed in this environment; the module refuses by name")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("study")
    document, bars, _calendar = world(tmp_path, planter=additive_planter)
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(document), encoding="utf-8")
    projections = estimate_run(tmp_path, document, bars, name="eval")
    projections_path = tmp_path / "projections.json"
    projections_path.write_text(json.dumps(projections), encoding="utf-8")
    return tmp_path, rows_path, projections_path, projections


def test_every_fitted_triple_yields_one_report_per_stage_over_one_sealed_corpus(study):
    _tmp, rows, projections, document = study
    result = evaluate_events.evaluate(str(rows), str(projections))
    assert result["counts"]["reports"] == 2 * result["counts"]["triples"]
    assert result["counts"]["triples"] > 0
    by_triple = {}
    for entry in result["reports"]:
        by_triple.setdefault((entry["event_type"], entry["horizon_minutes"], entry["outcome"]), []).append(entry)
    for key, pair in by_triple.items():
        assert sorted(entry["stage"] for entry in pair) == ["local_projection", "naive"]
        seals = {entry["corpus_seal"] for entry in pair}
        digests = {entry["protocol_digest"] for entry in pair}
        assert len(seals) == 1 and len(digests) == 1, f"{key} was scored under two different corpora"


def test_the_seal_covers_exactly_the_held_out_events_and_no_fitting_event(study):
    _tmp, rows, projections, document = study
    result = evaluate_events.evaluate(str(rows), str(projections))
    written = {(entry["event_type"], entry["horizon_minutes"], entry["outcome"], entry["stage"]): entry
               for entry in result["_written"]}
    entry = next(e for e in document["projections"] if e.get("status") == "OK")
    key = (entry["event_type"], entry["horizon_minutes"], entry["outcome"], "local_projection")
    payload = written[key]["payload"]
    assert payload["sealed_row_count"] == len(entry["holdout_event_keys"])
    assert payload["family"] == "forecast"
    assert payload["label_provenance"] == "REALISED_OUTCOME"


def test_the_naive_stage_scores_the_declared_baseline_against_itself_so_its_skill_is_zero(study):
    _tmp, rows, projections, _document = study
    result = evaluate_events.evaluate(str(rows), str(projections))
    naive = [entry for entry in result["reports"] if entry["stage"] == "naive"]
    assert naive
    for entry in naive:
        assert entry["skill_mae"] == pytest.approx(0.0, abs=1e-12)


def test_the_projection_beats_the_naive_on_the_planted_world_at_the_planted_horizon(study):
    """Not a claim about markets: in a world where a response WAS planted, the estimator that models it must beat
    the sign-mean out of sample, or the reports are not measuring what they say they measure."""
    _tmp, rows, projections, _document = study
    result = evaluate_events.evaluate(str(rows), str(projections))
    projection = [entry for entry in result["reports"]
                  if entry["stage"] == "local_projection" and entry["outcome"] == "log_return"]
    assert projection, "no log_return projection was evaluated at all"
    assert max(entry["skill_mae"] for entry in projection) > 0.0


def test_the_annotations_are_the_ones_the_stage_comparison_reads(study):
    _tmp, rows, projections, _document = study
    result = evaluate_events.evaluate(str(rows), str(projections))
    payload = result["_written"][0]["payload"]
    for key in evaluate_events.ANNOTATIONS:
        assert key in payload, f"the report carries no {key!r} annotation"
    assert payload["horizon"].startswith("h+")
    assert payload["target"] in ("log_return", "realized_vol")


def test_two_runs_over_the_same_artifacts_write_the_same_bytes(study, tmp_path):
    _tmp, rows, projections, _document = study
    first = evaluate_events.write(evaluate_events.evaluate(str(rows), str(projections)), tmp_path / "a")
    second = evaluate_events.write(evaluate_events.evaluate(str(rows), str(projections)), tmp_path / "b")
    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), f"{name} differs between two runs"


def test_a_projection_that_never_fitted_is_listed_rather_than_scored(study, tmp_path):
    _tmp, rows, _projections, document = study
    broken = json.loads(json.dumps(document))
    for entry in broken["projections"]:
        entry["status"] = "NO_FITTING_EVENTS"
    path = tmp_path / "unfitted.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    result = evaluate_events.evaluate(str(rows), str(path))
    assert result["counts"]["reports"] == 0
    assert result["counts"]["not_evaluated"] == len(broken["projections"])
    assert all("NOT_FITTED" in entry["why"] for entry in result["not_evaluated"])


def test_a_document_of_another_schema_is_refused_by_name(study, tmp_path):
    _tmp, rows, _projections, document = study
    other = json.loads(json.dumps(document))
    other["schema"] = "something.else.v1"
    path = tmp_path / "other.json"
    path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(evaluate_events.EvaluationRefusal) as refusal:
        evaluate_events.evaluate(str(rows), str(path))
    assert refusal.value.code == "NOT_A_PROJECTIONS_DOCUMENT"


def test_a_fit_whose_columns_are_not_in_the_held_out_design_is_refused_rather_than_predicted(study, tmp_path):
    _tmp, rows, _projections, document = study
    tampered = json.loads(json.dumps(document))
    entry = next(e for e in tampered["projections"] if e.get("status") == "OK")
    entry["columns"] = list(entry["columns"]) + ["a_column_nobody_built"]
    entry["coefficients"]["a_column_nobody_built"] = {"value": 1.0, "std_error": 0.0,
                                                      "ci_lower": 0.0, "ci_upper": 0.0}
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(evaluate_events.EvaluationRefusal) as refusal:
        evaluate_events.evaluate(str(rows), str(path))
    assert refusal.value.code == "COLUMNS_DO_NOT_MATCH_THE_FIT"


def test_the_cli_writes_the_reports_and_the_index(study, tmp_path):
    _tmp, rows, projections, _document = study
    out = tmp_path / "reports"
    assert evaluate_events.main(["--rows", str(rows), "--projections", str(projections),
                                 "--out-dir", str(out)]) == 0
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["schema"] == evaluate_events.SCHEMA
    assert index["counts"]["reports"] == len(list(out.glob("*.json"))) - 1
    first = json.loads((out / index["reports"][0]["file"]).read_text(encoding="utf-8"))
    assert first["version"] == "m5phet-evaluation-report/1"
