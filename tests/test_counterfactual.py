"""The counterfactual path, checked on the same planted market the projections were checked on.

The question this file answers is the only one that matters about a model-based counterfactual: if a response was
planted by hand and then the release that carries it is set to zero, does the model's own subtraction give the
planted response back? So the world is imported from `test_event_projections` rather than rebuilt -- the same
releases, the same planted step of `BETA * s` at +30 minutes, the same two-pass standardization -- and the whole
file is about what happens when `A`'s surprise is taken away over a past window.

Everything else here is refusals. A superposition verdict that is not `ADDITIVE_HOLDS` makes the subtraction
meaningless and the cell is refused by that name; an event type nobody fitted, a window with no release of it, a
window that ends before it starts and a document of another schema are each refused by their own name. And the
assumed publication clock is checked to be a FLAG and not a refusal: the owner asked to see the path, labelled.
"""

import json

import pytest

from feature_eng_m5phet import counterfactual as cf
from feature_eng_m5phet import local_projections as lp

from .test_event_projections import (BETA_A, additive_planter, interaction_planter, _build, _minutes, _noise, _plan,
                                     _standardized, _write_bars, _write_calendar_without_a_clock, world)

HORIZONS = (5, 30, 60)


def study(tmp_path, *, planter=additive_planter, name="", clock=None, **kwargs):
    """A planted world, its rows document on disk and its projections document on disk, ready to be asked."""
    if clock is None:
        document, bars, _ = world(tmp_path, planter=planter, name=name)
    else:
        releases = _plan()
        calendar = _write_calendar_without_a_clock(tmp_path / f"calendar_nc{name}.csv", releases)
        minutes = _minutes(releases)
        flat = _write_bars(tmp_path / f"flat_nc{name}.csv", _noise(minutes))
        standardized = _standardized(_build(flat, calendar, publication_clock=clock))
        bars = _write_bars(tmp_path / f"bars_nc{name}.csv", planter(minutes, releases, standardized))
        document = _build(bars, calendar, publication_clock=clock)
    rows_path = tmp_path / f"rows_cf{name}.json"
    rows_path.write_text(json.dumps(document), encoding="utf-8")
    arguments = {"outcomes": ("log_return",), "horizons": HORIZONS, "bars_path": str(bars), "placebo_n": 30,
                 "chunk_bytes": 1 << 16}
    arguments.update(kwargs)
    projections = lp.estimate(str(rows_path), **arguments)
    projections_path = tmp_path / f"projections_cf{name}.json"
    projections_path.write_text(json.dumps(projections), encoding="utf-8")
    return document, rows_path, projections_path, projections


def window_of(document, *, first=40, last=70):
    """Two instants bracketing a slice of the releases, taken from the rows themselves."""
    instants = sorted({row["published_at"] for row in document["rows"]})
    return instants[first], instants[last]


def rows_of(result, event_type, horizon):
    return [row for row in result["paths"]
            if row["event_type"] == event_type and row["horizon_minutes"] == horizon]


def surprise_of(document, event_key):
    for row in document["rows"]:
        if row["event_key"] == event_key:
            return row["surprise"]
    raise AssertionError(f"no row for {event_key}")


# ------------------------------------------------------------------- zeroing the planted event removes the response

def test_zeroing_the_planted_event_gives_the_planted_response_back_inside_the_interval(tmp_path):
    document, rows_path, projections_path, projections = study(tmp_path)
    start, end = window_of(document)
    result = cf.paths(str(projections_path), str(rows_path), window=(start, end), zero_out="A")

    assert result["schema"] == cf.SCHEMA
    assert result["label"] == "MODEL_BASED_COUNTERFACTUAL"
    assert result["window"]["releases_of_the_zeroed_type_in_window"] > 0
    assert result["counters"]["paths"] > 0

    for horizon in (30, 60):
        answered = [row for row in rows_of(result, "A", horizon) if row["status"] == "OK"]
        assert answered, f"no answered path at h={horizon}"
        for row in answered:
            planted = BETA_A * surprise_of(document, row["event_key"])
            low, high = row["attributed_transient_ci_95"]
            assert low <= planted <= high, f"h={horizon} {row['event_key']}: {planted} outside {[low, high]}"
            assert row["attributed_transient"] == pytest.approx(
                row["predicted_observed"] - row["predicted_counterfactual"], rel=1e-12, abs=1e-18)
            assert row["surprise_counterfactual"] == 0.0 and row["zeroed"] is True
            assert row["observed_outcome"] is not None


def test_before_the_planted_horizon_the_attributed_transient_is_zero_inside_its_interval(tmp_path):
    document, rows_path, projections_path, _ = study(tmp_path)
    start, end = window_of(document)
    result = cf.paths(str(projections_path), str(rows_path), window=(start, end), zero_out="A")
    early = [row for row in rows_of(result, "A", 5) if row["status"] == "OK"]
    assert early
    for row in early:
        low, high = row["attributed_transient_ci_95"]
        assert low <= 0.0 <= high, f"h=5 {row['event_key']}: {[low, high]} excludes zero"


def test_a_release_of_another_type_keeps_its_surprise_and_only_its_neighbour_control_moves(tmp_path):
    document, rows_path, projections_path, _ = study(tmp_path)
    start, end = window_of(document)
    result = cf.paths(str(projections_path), str(rows_path), window=(start, end), zero_out="A")
    others = [row for row in rows_of(result, "B", 30) if row["status"] == "OK"]
    assert others
    for row in others:
        assert row["zeroed"] is False
        assert row["surprise_counterfactual"] == row["surprise_observed"]
        assert set(row["columns_that_moved"]) <= {"other_surprises_in_window_negative_offsets"}


def test_the_document_carries_the_identification_block_and_the_declared_interval_method(tmp_path):
    document, rows_path, projections_path, projections = study(tmp_path)
    start, end = window_of(document)
    result = cf.paths(str(projections_path), str(rows_path), window=(start, end), zero_out="A")
    assert result["identification"] == projections["identification"]
    assert result["identification_reasons"] == projections["identification_reasons"]
    assert result["identification_caveat"] == projections["publication_clock"]["identification_caveat"]
    assert "delta method" in result["interval_method"]
    assert result["interval_caveat"].startswith("COVARIANCE_OFF_DIAGONAL_NOT_AVAILABLE")
    assert result["projections_document"]["sha256"] == cf._digest(projections_path)
    assert result["execution_authorized"] is False
    assert "NO_NEW_MEASUREMENT" in result["reading"]
    assert "MODEL_BASED_COUNTERFACTUAL" in result["reading"]


# ------------------------------------------------------------------- the assumed clock is a flag, not a refusal

def test_an_assumed_publication_clock_travels_as_a_flag_and_the_path_is_still_computed(tmp_path):
    document, rows_path, projections_path, projections = study(tmp_path, name="a", clock="scheduled")
    assert document["publication_clock"]["mode"] == "ASSUMED_SCHEDULED_PUBLICATION"
    start, end = window_of(document)
    result = cf.paths(str(projections_path), str(rows_path), window=(start, end), zero_out="A")
    assert result["flags"] == [cf.ASSUMED_CLOCK_FLAG]
    assert result["counters"]["paths"] > 0, "the path was withheld; the owner asked to see it, labelled"
    assert all(row["flags"] == [cf.ASSUMED_CLOCK_FLAG] for row in result["paths"])
    assert result["identification_caveat"] == projections["publication_clock"]["identification_caveat"]
    assert cf.ASSUMED_CLOCK_FLAG in result["reading"]


# --------------------------------------------------------------------------------------- refusals, by name only

def test_a_superposition_that_did_not_hold_refuses_the_cell_by_name(tmp_path):
    document, rows_path, projections_path, projections = study(tmp_path, planter=interaction_planter, name="x",
                                                               horizons=(30,))
    failed = [(test["horizon_minutes"], test["outcome"]) for test in projections["superposition"]["tests"]
              if test["verdict"] != "ADDITIVE_HOLDS"]
    assert failed, projections["superposition"]["tests"]
    start, end = window_of(document)
    result = cf.paths(str(projections_path), str(rows_path), window=(start, end), zero_out="A")
    refused = [row for row in result["paths"]
               if (row["horizon_minutes"], row["outcome"]) in failed]
    assert refused
    assert all(row["status"] == "REFUSED" and row["refusal"] == "SUPERPOSITION_FAILED" for row in refused)
    assert all("attributed_transient" not in row for row in refused)
    assert result["counters"]["by_refusal"]["SUPERPOSITION_FAILED"] == len(refused)


def test_an_event_type_nobody_fitted_is_refused_by_name(tmp_path):
    document, rows_path, projections_path, _ = study(tmp_path)
    start, end = window_of(document)
    with pytest.raises(cf.CounterfactualRefusal) as refusal:
        cf.paths(str(projections_path), str(rows_path), window=(start, end), zero_out="Neverland | Nothing")
    assert refusal.value.code == "EVENT_TYPE_NOT_FITTED"


def test_a_window_with_no_release_of_the_named_type_is_refused_by_name(tmp_path):
    document, rows_path, projections_path, _ = study(tmp_path)
    instants = sorted({row["published_at"] for row in document["rows"]})
    with pytest.raises(cf.CounterfactualRefusal) as refusal:
        cf.paths(str(projections_path), str(rows_path), window=(instants[-1], "2099-01-01T00:00:00+00:00"),
                 zero_out="A")
    assert refusal.value.code == "EVENT_NOT_IN_WINDOW"


def test_a_window_that_ends_before_it_starts_and_one_without_a_zone_are_refused_by_name(tmp_path):
    document, rows_path, projections_path, _ = study(tmp_path)
    start, end = window_of(document)
    with pytest.raises(cf.CounterfactualRefusal) as refusal:
        cf.paths(str(projections_path), str(rows_path), window=(end, start), zero_out="A")
    assert refusal.value.code == "BAD_WINDOW"
    with pytest.raises(cf.CounterfactualRefusal) as refusal:
        cf.paths(str(projections_path), str(rows_path), window=("2024-01-05T00:00:00", end), zero_out="A")
    assert refusal.value.code == "BAD_WINDOW"


def test_a_document_of_another_schema_is_refused_by_name(tmp_path):
    document, rows_path, projections_path, _ = study(tmp_path)
    other = tmp_path / "not_projections.json"
    other.write_text(json.dumps({"schema": "m5phet.event_rows.v1"}), encoding="utf-8")
    with pytest.raises(cf.CounterfactualRefusal) as refusal:
        cf.paths(str(other), str(rows_path), window=window_of(document), zero_out="A")
    assert refusal.value.code == "WRONG_SCHEMA"


def test_an_unknown_horizon_or_outcome_is_refused_by_name(tmp_path):
    document, rows_path, projections_path, _ = study(tmp_path)
    window = window_of(document)
    with pytest.raises(cf.CounterfactualRefusal) as refusal:
        cf.paths(str(projections_path), str(rows_path), window=window, zero_out="A", horizons=[7])
    assert refusal.value.code == "UNKNOWN_HORIZON"
    with pytest.raises(cf.CounterfactualRefusal) as refusal:
        cf.paths(str(projections_path), str(rows_path), window=window, zero_out="A", outcomes=("profit",))
    assert refusal.value.code == "UNKNOWN_OUTCOME"


# --------------------------------------------------------------------------------------------------------- the CLI

def test_the_cli_writes_the_document_and_refuses_an_unfitted_event_type(tmp_path, capsys):
    document, rows_path, projections_path, _ = study(tmp_path)
    start, end = window_of(document)
    out = tmp_path / "counterfactual.json"
    code = cf.main(["--projections", str(projections_path), "--rows", str(rows_path),
                    "--window", start, end, "--zero-out", "A", "--horizons", "30", "--out", str(out)])
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == cf.SCHEMA and written["zero_out"] == "A"
    assert {row["horizon_minutes"] for row in written["paths"]} == {30}
    assert "MODEL_BASED_COUNTERFACTUAL" in capsys.readouterr().out

    code = cf.main(["--projections", str(projections_path), "--rows", str(rows_path),
                    "--window", start, end, "--zero-out", "C"])
    assert code == 2
    assert "EVENT_TYPE_NOT_FITTED" in capsys.readouterr().err
