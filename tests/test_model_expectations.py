"""The declared MODEL_BASED_EXPECTATION: what it may see, what it must never see, and what it must never be called.

WP28 replaces a consensus nobody published with a model's forecast, which is a smaller claim and a more dangerous
one: a model can see the future by accident in five different places, and a reader who finds the word "consensus"
next to it has been told something false. The tests here are exactly those two worries.

* **No look-ahead, planted.** The whole history after a release is rewritten into something else and every artifact
  at or before that release must come back byte for byte identical -- the expectation, the model chosen for the row,
  and that model's own out-of-sample error. That covers all three places the future could enter: the fit, the
  selection, and the error the selection is made on.
* **The words.** The rows, the calendar document and the projections fitted on them carry
  `MODEL_BASED_EXPECTATION` and never the string `consensus` in a field name; a calendar that declares both is
  refused; the identification block gains `EXPECTATION_IS_MODEL_BASED` and, under an OBSERVED clock, loses
  `ASSUMED_PUBLICATION_CLOCK`.

Nothing here reads a file outside its own tmp_path, nothing needs a GPU, and no market claim is made anywhere.
"""

import csv
import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from feature_eng_m5phet import events, expectations

START = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

ANNOUNCEMENT_HEADER = ["currency", "indicator", "date", "val", "announcement_datetime_utc"]


def series_values(n, *, drift=0.35, wobble=1.7):
    """A deterministic series with a trend and a wobble: AR(p) beats the previous value on it, and both are fittable."""
    return [100.0 + drift * i + wobble * math.sin(i / 3.0) for i in range(n)]


def write_announcements(path, values, *, indicator="cpi", currency="USD", start=START, spacing_days=30):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANNOUNCEMENT_HEADER)
        for i, value in enumerate(values):
            moment = start + timedelta(days=spacing_days * i)
            writer.writerow([currency, indicator, (moment - timedelta(days=2)).date().isoformat(),
                             repr(float(value)), moment.isoformat()])
    return path


def build(tmp_path, values, *, name="", **kwargs):
    archive = write_announcements(tmp_path / f"announcements{name}.csv", values)
    return expectations.build(str(archive), min_history=kwargs.pop("min_history", 6),
                              min_oos=kwargs.pop("min_oos", 4), **kwargs)


# --------------------------------------------------------------------------------------------- it cannot look ahead

def test_no_expectation_model_or_error_moves_when_the_whole_future_is_replaced(tmp_path):
    values = series_values(48)
    early = build(tmp_path, values, name="a")
    # everything from release 30 on becomes a different series entirely; nothing at or before it may notice
    rewritten = list(values[:30]) + [v * -3.0 + 500.0 for v in values[30:]]
    later = build(tmp_path, rewritten, name="b")
    kept = [row for row in early["rows"] if row["archive_row"] <= 31]
    assert kept, "the fixture produced no row early enough to test look-ahead on"
    found = {row["archive_row"]: row for row in later["rows"]}
    for row in kept:
        twin = found[row["archive_row"]]
        for field in ("expectation", "expectation_model", "expectation_model_oos_mae",
                      "expectation_model_oos_rmse", "expectation_model_oos_n", "previous", "actual"):
            assert twin[field] == row[field], (
                f"{field} of archive row {row['archive_row']} moved when only LATER releases changed: "
                f"{row[field]!r} became {twin[field]!r}")


def test_the_first_releases_are_refused_by_name_rather_than_expected_from_nothing(tmp_path):
    document = build(tmp_path, series_values(40))
    codes = document["excluded"]["counts"]
    assert codes["INSUFFICIENT_VINTAGE_HISTORY"] > 0
    assert all(row["expectation_model_oos_n"] >= 4 for row in document["rows"])
    first_kept = min(row["archive_row"] for row in document["rows"])
    assert first_kept > 2, "a release with no history behind it was given an expectation"


def test_every_candidate_is_scored_out_of_sample_and_the_chosen_one_is_recorded(tmp_path):
    document = build(tmp_path, series_values(48))
    assert document["expectation_kind"] == expectations.EXPECTATION_KIND
    models = document["counts"]["chosen_model"]
    assert models, "no model was ever chosen"
    assert set(models) <= {"SEASONAL_NAIVE(m=1)", "SEASONAL_NAIVE(m=12)", "AR(p<=4,BIC)"}
    for row in document["rows"]:
        assert row["expectation_model"].split("->")[0] in row["expectation_candidates"].split(";")
        assert row["expectation_model_oos_mae"] >= 0.0


def test_an_autoregression_beats_the_previous_value_on_a_series_built_to_be_predictable(tmp_path):
    document = build(tmp_path, series_values(60))
    chosen = document["counts"]["chosen_model"]
    assert chosen.get("AR(p<=4,BIC)", 0) > 0, (
        f"on a trending, wobbling series the autoregression was never preferred: {chosen}")


def test_a_release_outside_the_emission_window_still_feeds_the_history_of_the_ones_inside_it(tmp_path):
    values = series_values(48)
    whole = build(tmp_path, values, name="w")
    cut = (START + timedelta(days=30 * 35)).isoformat()
    bounded = build(tmp_path, values, name="x", emit_from=cut)
    assert bounded["excluded"]["counts"]["OUTSIDE_THE_DECLARED_EMISSION_WINDOW"] > 0
    inside = {row["archive_row"]: row for row in whole["rows"]}
    assert bounded["rows"], "the emission window removed every row"
    for row in bounded["rows"]:
        assert row["expectation"] == inside[row["archive_row"]]["expectation"], (
            "bounding what is EMITTED changed what a model was allowed to see")


# ------------------------------------------------------------------------------------- it is never called a consensus

def _event_rows(tmp_path, values, **kwargs):
    archive = write_announcements(tmp_path / "announcements.csv", values)
    document = expectations.build(str(archive), min_history=6, min_oos=4)
    calendar = expectations.write_csv(document, tmp_path / "calendar.csv")
    first = datetime.fromisoformat(document["rows"][0]["published_at"])
    last = datetime.fromisoformat(document["rows"][-1]["published_at"])
    lines = ["timestamp,close"]
    moment, index = first - timedelta(days=1), 0
    while moment <= last + timedelta(hours=8):
        lines.append(f"{moment.isoformat()},{1.1 + 0.00001 * (index % 7)!r}")
        moment, index = moment + timedelta(minutes=5), index + 1
    bars = tmp_path / "bars.csv"
    bars.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mapping = events.CalendarMapping(event=["event_type"], event_time=["event_time"], published="published_at",
                                     actual="actual", expectation="expectation",
                                     expectation_model="expectation_model",
                                     expectation_oos_error="expectation_model_oos_mae",
                                     expectation_oos_n="expectation_model_oos_n",
                                     previous="previous", availability="historical_availability")
    return events.build(str(bars), str(calendar), mapping, horizons_minutes=(5, 15), min_prior_releases=2,
                        publication_clock="observed", **kwargs)


def test_the_event_rows_carry_the_expectation_and_never_a_consensus(tmp_path):
    document = _event_rows(tmp_path, series_values(60))
    assert document["rows"], "no event row survived"
    assert document["expectation"]["kind"] == expectations.EXPECTATION_KIND
    assert document["provenance"].endswith("_MODEL_EXPECTATION")
    assert document["publication_clock"]["mode"] == "OBSERVED_ACTUAL_PUBLICATION"
    for row in document["rows"]:
        assert "consensus" not in row and "consensus_published_at" not in row
        assert row["expectation_kind"] == expectations.EXPECTATION_KIND
        assert row["expectation"] is not None
        assert row["expectation_model"]
        assert row["expectation_oos_error"] >= 0.0
        assert row["surprise_raw"] == pytest.approx(row["actual"] - row["expectation"], rel=1e-9, abs=1e-12)
    assert "consensus_column" not in document["calendar"]
    assert "NO_EXPECTATION" in document["excluded"]["counts"]


def test_a_calendar_that_declares_both_a_consensus_and_an_expectation_is_refused(tmp_path):
    archive = write_announcements(tmp_path / "announcements.csv", series_values(40))
    document = expectations.build(str(archive), min_history=6, min_oos=4)
    calendar = expectations.write_csv(document, tmp_path / "calendar.csv")
    bars = tmp_path / "bars.csv"
    bars.write_text("timestamp,close\n2025-01-01T00:00:00+00:00,1.1\n2025-01-01T00:05:00+00:00,1.1\n",
                    encoding="utf-8")
    mapping = events.CalendarMapping(event=["event_type"], event_time=["event_time"], published="published_at",
                                     actual="actual", consensus="previous", expectation="expectation",
                                     availability="historical_availability")
    with pytest.raises(events.EventsRefusal) as refusal:
        events.build(str(bars), str(calendar), mapping, publication_clock="observed")
    assert refusal.value.code == "TWO_EXPECTATIONS_DECLARED"


def test_the_identification_block_names_the_model_based_expectation_and_not_a_missing_clock(tmp_path):
    from feature_eng_m5phet import local_projections as lp

    document = _event_rows(tmp_path, series_values(120))
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    try:
        projections = lp.estimate(str(path), horizons=[5], outcomes=("log_return",), placebo_n=20)
    except lp.ProjectionRefusal as refusal:
        pytest.skip(f"the synthetic fixture produced too few events to project: {refusal}")
    reasons = " ".join(projections["identification_reasons"])
    assert lp.MODEL_BASED_EXPECTATION_REASON in reasons
    assert "ASSUMED_PUBLICATION_CLOCK" not in reasons, (
        "the release instants were observed and the study still blamed a missing clock")
    assert projections["expectation"]["kind"] == expectations.EXPECTATION_KIND
    assert projections["identification"] == "NOT_IDENTIFIED", (
        "a surprise measured against a model's expectation is not identified, whatever the placebo says")
