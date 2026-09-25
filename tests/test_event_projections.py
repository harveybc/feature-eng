"""The local projections, checked against a market whose response to each release was planted by hand.

Every number this module reports is only as good as the question "would it find a response that is not there, and
would it miss one that is?", so the whole file is built on a synthetic market where the answer is known before the
job runs. The planting is done in two passes because the standardized surprise is not a free choice: it is the raw
surprise divided by the dispersion of the releases published before it, which `events.py` computes from the calendar
alone. So pass one builds the rows over a market that does nothing, reads the standardized surprises out of them,
and pass two plants a response proportional to exactly those numbers. The recovered `beta` is then comparable with a
planted constant instead of with a moving target.

The market has a tiny deterministic noise walk under it, because a price that never moves except when a release
lands gives a regression with zero residuals, and a zero-width interval is not a test of anything. After the last
release the market keeps moving and no release lands again for ten days: that quiet tail is where the pseudo-events
of the placebo come from, and it is the only part of this file that exists for the placebo's sake.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from feature_eng_m5phet import association, events
from feature_eng_m5phet import local_projections as lp

START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

#: the planted level response, per unit of STANDARDIZED surprise, that arrives all at once at +30 minutes
BETA_A = 0.0020
BETA_B = 0.0011
#: the planted product term of the superposition tests
GAMMA = 0.0015
JUMP_MINUTE = 30

SPACING_MINUTES = 240                  # 4 h between consecutive releases, types alternating
WINDOW_HOURS = 4.0                     # so exactly ONE other release sits at a negative offset in every window
FIRST_EVENT_MINUTE = 600
N_RELEASES = 120
QUIET_TAIL_MINUTES = 10 * 24 * 60      # ten days with no release: the placebo's only source of instants
NOISE_SIGMA = 1e-5

HORIZONS = (5, 30, 60)
CALENDAR_HEADER = ("event_type,event_time,published_at,consensus_published_at,actual,consensus,previous,"
                   "historical_availability")


def _plan(n=N_RELEASES):
    raw = np.round(np.random.default_rng(20260925).uniform(-2.0, 2.0, n), 4)
    return [("A" if i % 2 == 0 else "B", FIRST_EVENT_MINUTE + i * SPACING_MINUTES, float(value))
            for i, value in enumerate(raw)]


def _minutes(releases):
    return releases[-1][1] + QUIET_TAIL_MINUTES


def _noise(minutes):
    steps = np.random.default_rng(4242).normal(0.0, NOISE_SIGMA, minutes)
    steps[0] = 0.0
    return np.cumsum(steps)


def _write_calendar(path, releases):
    lines = [CALENDAR_HEADER]
    for kind, minute, raw in releases:
        moment = START + timedelta(minutes=minute)
        lines.append(",".join([kind, moment.isoformat(), moment.isoformat(),
                               (moment - timedelta(days=1)).isoformat(),
                               repr(100.0 + raw), repr(100.0), repr(100.0), "KNOWN"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_bars(path, log_price):
    lines = ["timestamp,close"]
    for i, value in enumerate(log_price):
        lines.append(f"{(START + timedelta(minutes=i)).isoformat()},{float(np.exp(value))!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build(bars, calendar, **kwargs):
    arguments = {"horizons_minutes": HORIZONS, "window_hours": WINDOW_HOURS, "pre_event_minutes": 60,
                 "min_prior_releases": 8}
    arguments.update(kwargs)
    return events.build(str(bars), str(calendar), events.CalendarMapping(), **arguments)


def _standardized(document):
    """published_at -> the standardized surprise `events.py` computed for the release at that instant."""
    return {row["published_at"]: row["surprise"] for row in document["rows"]}


def world(tmp_path, *, planter, name="", **kwargs):
    """Two passes: read the standardized surprises off a market that does nothing, then plant against them."""
    releases = _plan()
    calendar = _write_calendar(tmp_path / f"calendar{name}.csv", releases)
    minutes = _minutes(releases)
    flat = _write_bars(tmp_path / f"flat{name}.csv", _noise(minutes))
    standardized = _standardized(_build(flat, calendar, **kwargs))
    planted = planter(minutes, releases, standardized)
    bars = _write_bars(tmp_path / f"bars{name}.csv", planted)
    return _build(bars, calendar, **kwargs), bars, calendar


def _surprise_of(standardized, minute):
    return standardized.get((START + timedelta(minutes=minute)).isoformat())


def additive_planter(minutes, releases, standardized):
    """A step of BETA * s at +30 minutes, per release, and nothing else. Superposition holds by construction."""
    path = _noise(minutes)
    for kind, minute, _ in releases:
        s = _surprise_of(standardized, minute)
        if s is None:
            continue
        jump = (BETA_A if kind == "A" else BETA_B) * s
        path[minute + JUMP_MINUTE:] += jump
    return path


def interaction_planter(minutes, releases, standardized):
    """The additive world plus the product of an A release's surprise with the B release eight hours before it."""
    path = _noise(minutes)
    previous = None
    for kind, minute, _ in releases:
        s = _surprise_of(standardized, minute)
        if s is not None:
            jump = (BETA_A if kind == "A" else BETA_B) * s
            if kind == "A" and previous is not None:
                jump += GAMMA * s * previous
            path[minute + JUMP_MINUTE:] += jump
        previous = s
    return path


def time_noise_planter(minutes, releases, standardized):
    """No response at all: a slow deterministic swell that depends on the clock and on nothing anybody released."""
    grid = np.arange(minutes, dtype=np.float64)
    return _noise(minutes) + 0.01 * np.sin(2.0 * np.pi * grid / (3.0 * 1440.0))


def run(tmp_path, document, bars, **kwargs):
    path = tmp_path / f"rows{kwargs.pop('name', '')}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    arguments = {"placebo_n": 200, "seed": 1729, "bars_path": str(bars), "chunk_bytes": 1 << 16}
    arguments.update(kwargs)
    return lp.estimate(str(path), **arguments)


def entry_of(document, event_type, horizon, outcome="log_return"):
    for entry in document["projections"]:
        if (entry["event_type"], entry["horizon_minutes"], entry["outcome"]) == (event_type, horizon, outcome):
            return entry
    raise AssertionError(f"no projection for {event_type} h={horizon} {outcome}")


# ------------------------------------------------------------------------------------- the streaming reader is exact

def test_the_streaming_reader_returns_every_key_but_the_rows_and_yields_each_row_once(tmp_path):
    document, _, _ = world(tmp_path, planter=additive_planter)
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    seen = []
    header = lp.read_rows_document(str(path), seen.append, chunk_bytes=64)
    assert "rows" not in header
    assert set(header) == set(document) - {"rows"}
    assert header["parameters"] == document["parameters"]
    assert header["counts"] == document["counts"]
    assert seen == document["rows"]


def test_the_streaming_reader_refuses_a_document_that_is_not_the_rows_schema(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "something.else.v1", "rows": []}), encoding="utf-8")
    with pytest.raises(lp.ProjectionRefusal) as refusal:
        lp.load(str(path))
    assert refusal.value.code == "WRONG_SCHEMA"


# --------------------------------------------------------------------------------------- the planted beta is read back

def test_the_planted_beta_is_inside_its_interval_at_the_planted_horizons_and_zero_before_them(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    result = run(tmp_path, document, bars, outcomes=("log_return",))
    assert result["estimator"]["statsmodels"], "the HAC covariance needs statsmodels and it was not importable"
    for event_type, planted in (("A", BETA_A), ("B", BETA_B)):
        for horizon in (30, 60):
            entry = entry_of(result, event_type, horizon)
            assert entry["status"] == "OK", entry
            low, high = entry["beta_ci_95"]
            assert low <= planted <= high, f"{event_type} h={horizon}: {planted} outside {entry['beta_ci_95']}"
            assert entry["beta"] == pytest.approx(planted, rel=0.05)
        early = entry_of(result, event_type, 5)
        assert early["status"] == "OK"
        assert early["beta_ci_95"][0] <= 0.0 <= early["beta_ci_95"][1], early["beta_ci_95"]
        assert abs(early["beta"]) < 0.1 * planted


def test_the_controls_are_the_declared_ones_and_the_positive_half_of_the_window_is_excluded(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    result = run(tmp_path, document, bars, outcomes=("log_return",))
    entry = entry_of(result, "A", 30)
    assert "pre_event_realized_vol" in entry["controls"]
    assert "other_surprises_in_window_negative_offsets" in entry["controls"]
    assert any(name.startswith("day_of_week=") for name in entry["controls"])
    assert "NEGATIVE offsets only" in result["estimator"]["controls_reading"]
    assert "EXCLUDED" in result["estimator"]["controls_reading"]
    assert entry["hac_maxlags"] == lp._hac_lags(entry["n"])
    assert "Newey-West" in entry["covariance"]


def test_the_neighbour_control_reconstructed_from_the_index_equals_the_listed_neighbours(tmp_path):
    document, _, _ = world(tmp_path, planter=additive_planter)
    path = tmp_path / "rows_listed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = lp.load(str(path))
    window = float(document["parameters"]["window_hours"]) * 3600.0
    listed = {}
    for row in document["rows"]:
        total = sum(neighbour["surprise"] for neighbour in row["other_releases_in_window"]
                    if neighbour["offset_minutes"] < 0 and neighbour["surprise"] is not None)
        listed[row["event_key"]] = total
    for row in loaded["rows"]:
        _, other = lp._window_surprises(loaded["index"], row["published_epoch"], window,
                                        exclude_key=row["event_key"])
        assert other == pytest.approx(listed[row["event_key"]], abs=1e-12)


# -------------------------------------------------------------------------------------------- the held-out events

def test_the_held_out_events_never_enter_a_fit_and_are_the_last_fifth_by_time(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    result = run(tmp_path, document, bars, outcomes=("log_return",))
    for event_type in ("A", "B"):
        keys = sorted({(row["published_at"], row["event_key"]) for row in document["rows"]
                       if row["event_type"] == event_type})
        held = int(np.ceil(0.20 * len(keys)))
        expected = {key for _, key in keys[len(keys) - held:]}
        assert set(result["held_out"]["by_event_type"][event_type]["holdout_release_keys"]) == expected
        for horizon in HORIZONS:
            entry = entry_of(result, event_type, horizon)
            assert set(entry["fit_event_keys"]).isdisjoint(expected)
            assert set(entry["holdout_event_keys"]) <= expected
            assert set(entry["fit_event_keys"]) & set(entry["holdout_event_keys"]) == set()
            assert entry["n_fit_events"] == len(entry["fit_event_keys"])


def test_the_closure_row_scores_the_projection_and_the_naive_reference_on_the_same_events(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    result = run(tmp_path, document, bars, outcomes=("log_return",))
    rows = [row for row in result["closure_table"] if row["event_type"] == "A" and row["horizon_minutes"] == 30]
    assert len(rows) == 1
    row = rows[0]
    assert row["comparability"] == "COMPARABLE"
    assert row["local_projection_mse"] is not None and row["naive_error"] is not None
    assert row["skill"] == pytest.approx(1.0 - row["local_projection_mse"] / row["naive_error"])
    # the planted response is linear in the surprise, so a linear projection must beat a mean by sign
    assert row["skill"] > 0.0
    assert row["naive_reference"].startswith("association.naive_response_by_sign")


def test_the_naive_reference_is_rung_ones_own_table_and_not_a_second_implementation(tmp_path):
    surprise = np.asarray([-2.0, -1.0, 0.0, 1.0, 3.0])
    outcome = np.asarray([-1.0, -3.0, 7.0, 2.0, 4.0])
    table = association.naive_response_by_sign(surprise, outcome)
    assert table["negative"]["mean"] == pytest.approx(-2.0) and table["negative"]["n"] == 2
    assert table["positive"]["mean"] == pytest.approx(3.0) and table["zero"]["mean"] == pytest.approx(7.0)
    assert association.sign_bin(-0.5) == "negative" and association.sign_bin(0.0) == "zero"
    value, reason = association.naive_sign_prediction(table, -1.5)
    assert reason is None and value == pytest.approx(-2.0)
    empty = association.naive_response_by_sign(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))
    value, reason = association.naive_sign_prediction(empty, -1.0, fallback=0.5)
    assert value == 0.5 and reason.startswith("EMPTY_SIGN_BIN")


# ------------------------------------------------------------------------------------------- the superposition test

def test_superposition_holds_on_additive_data(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    result = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,))
    assert result["superposition"]["verdict"] == "ADDITIVE_HOLDS", result["superposition"]["tests"]
    test = result["superposition"]["tests"][0]
    assert test["pair"] == ["A", "B"]
    assert test["additive"]["held_out_mse"] is not None


def test_superposition_fails_when_a_product_term_was_planted(tmp_path):
    document, bars, _ = world(tmp_path, planter=interaction_planter, name="x")
    result = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), name="x")
    assert result["superposition"]["verdict"] == "INTERACTIONS_IMPROVE", result["superposition"]["tests"]
    test = result["superposition"]["tests"][0]
    assert test["relative_improvement"] > result["superposition"]["margin"]


# ------------------------------------------------------------------------------------------------------ the placebo

def test_the_placebo_passes_on_the_planted_event(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    result = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), event_types=["A"])
    placebo = result["placebo"]
    assert placebo["status"] == "OK", placebo.get("status")
    assert placebo["eligible_instants"] > 0 and placebo["n_drawn"] == 200
    verdict = placebo["verdicts"][0]
    assert verdict["verdict"] == "PLACEBO_PASSES", verdict
    assert verdict["placebo_interval_contains_zero"] and not verdict["intervals_overlap"]
    assert result["identification"] == "PLACEBO_CONSISTENT", result["identification_reasons"]
    assert "is NOT a claim of identification" in result["identification_reading"]


def test_the_placebo_fails_when_the_outcome_is_a_swell_that_depends_only_on_the_clock(tmp_path):
    document, bars, _ = world(tmp_path, planter=time_noise_planter, name="t")
    result = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), event_types=["A"], name="t")
    verdict = result["placebo"]["verdicts"][0]
    assert verdict["verdict"] == "NOT_IDENTIFIED", verdict
    assert verdict["intervals_overlap"]
    assert result["identification"] == "NOT_IDENTIFIED"


def test_the_placebo_is_reproducible_from_the_seed(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    first = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), event_types=["A"])
    second = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), event_types=["A"])
    assert first["placebo"]["verdicts"] == second["placebo"]["verdicts"]
    assert json.dumps(first["projections"]) == json.dumps(second["projections"])


# ------------------------------------------------------------------------------------------ the clock has the last word

def _write_calendar_without_a_clock(path, releases):
    """The same calendar with the publication instants taken away: the archive the operator has to assume about."""
    lines = ["event_type,event_time,actual,consensus,previous,historical_availability"]
    for kind, minute, raw in releases:
        moment = START + timedelta(minutes=minute)
        lines.append(",".join([kind, moment.isoformat(), repr(100.0 + raw), repr(100.0), repr(100.0), "KNOWN"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_an_assumed_publication_clock_is_not_identified_however_the_placebo_went(tmp_path):
    releases = _plan()
    calendar = _write_calendar_without_a_clock(tmp_path / "calendar_a.csv", releases)
    minutes = _minutes(releases)
    flat = _write_bars(tmp_path / "flat_a.csv", _noise(minutes))
    assumed = {"publication_clock": "scheduled"}
    standardized = _standardized(_build(flat, calendar, **assumed))
    bars = _write_bars(tmp_path / "bars_a.csv", additive_planter(minutes, releases, standardized))
    document = _build(bars, calendar, **assumed)
    assert document["publication_clock"]["mode"] == "ASSUMED_SCHEDULED_PUBLICATION"
    result = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), event_types=["A"], name="a")
    assert result["placebo"]["verdicts"][0]["verdict"] == "PLACEBO_PASSES"
    assert result["identification"] == "NOT_IDENTIFIED"
    assert any(reason.startswith("ASSUMED_PUBLICATION_CLOCK") for reason in result["identification_reasons"])
    assert result["provenance"] == "DEVELOPMENT_ASSUMED_CLOCK"
    assert result["publication_clock"]["identification_caveat"] == document["publication_clock"][
        "identification_caveat"]
    assert "NO_NEW_MEASUREMENT" in result["reading"]
    assert result["execution_authorized"] is False


# --------------------------------------------------------------------------------------------- refusals, by name only

def test_the_cli_writes_the_document_and_refuses_an_unknown_event_type(tmp_path, capsys):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    path = tmp_path / "rows_cli.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    out = tmp_path / "projections.json"
    code = lp.main(["--rows", str(path), "--out", str(out), "--event-types", "A", "--horizons", "30",
                    "--outcomes", "log_return", "--bars", str(bars), "--placebo-n", "50"])
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == lp.SCHEMA and written["seed"] == lp.DEFAULT_SEED
    code = lp.main(["--rows", str(path), "--event-types", "Neverland | Nothing"])
    assert code == 2
    assert "EVENT_TYPE_NOT_IN_THE_ROWS" in capsys.readouterr().err


def test_a_missing_bars_file_refuses_the_placebo_by_name_and_keeps_the_study_unidentified(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    result = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), event_types=["A"],
                 bars_path=str(tmp_path / "not_here.csv"))
    assert result["placebo"]["status"].startswith("BARS_NOT_READABLE")
    assert result["identification"] == "NOT_IDENTIFIED"
    assert any(reason.startswith("PLACEBO_NOT_RUN") for reason in result["identification_reasons"])


def test_bars_that_are_not_the_ones_the_rows_were_built_from_refuse_the_placebo(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    other = _write_bars(tmp_path / "other_bars.csv", _noise(_minutes(_plan())) + 0.01)
    result = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), event_types=["A"],
                 bars_path=str(other))
    assert result["placebo"]["status"].startswith("BARS_DIGEST_MISMATCH")


def test_statsmodels_is_refused_by_name_rather_than_replaced_by_a_plain_ols(tmp_path, monkeypatch):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    monkeypatch.setattr(lp, "_statsmodels", lambda: None)
    result = run(tmp_path, document, bars, outcomes=("log_return",), horizons=(30,), event_types=["A"])
    entry = entry_of(result, "A", 30)
    assert entry["status"] == "STATSMODELS_NOT_AVAILABLE"
    assert "beta" not in entry
    assert "labelled HAC" in entry["why"]
    assert result["closure_table"][0]["comparability"] == "NOT_COMPARABLE"
    assert result["identification"] == "NOT_IDENTIFIED"


def test_a_bad_holdout_fraction_and_an_unknown_outcome_are_refused_by_name(tmp_path):
    document, bars, _ = world(tmp_path, planter=additive_planter)
    path = tmp_path / "rows_bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(lp.ProjectionRefusal) as refusal:
        lp.estimate(str(path), holdout_fraction=1.5)
    assert refusal.value.code == "BAD_HOLDOUT_FRACTION"
    with pytest.raises(lp.ProjectionRefusal) as refusal:
        lp.estimate(str(path), outcomes=("profit",))
    assert refusal.value.code == "UNKNOWN_OUTCOME"


def test_a_rows_document_with_no_rows_is_refused_by_name_rather_than_failing_inside_the_placebo(tmp_path):
    """A joined calendar that matched nothing produces an empty rows document. The estimator must name that, not
    raise somewhere in the placebo, where the traceback says nothing about which input was empty."""
    document, bars, _ = world(tmp_path, planter=additive_planter, name="empty")
    document["rows"] = []
    document["counts"] = {"rows": 0, "releases_read": 0}
    document["excluded"] = {"counts": {"NO_CONSENSUS": 7}}
    with pytest.raises(lp.ProjectionRefusal) as refusal:
        run(tmp_path, document, bars, name="empty")
    assert refusal.value.code == "NO_EVENT_ROWS"
    assert "NO_CONSENSUS" in refusal.value.why
