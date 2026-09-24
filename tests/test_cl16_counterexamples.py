"""CL16: the calendar counterexamples, frozen before repair.

Three findings, one theme: a rule that lives in a helper nobody has to call, or in a copy the caller can edit, is not a rule.

* publication and local receipt are different boundaries. Freezing the consensus at OUR receipt of the number let a consensus
  published AFTER the release be treated as what the market expected, turning a 0.4 surprise into 0.0;
* `availability_checked` returned the right answer to a question the ingestion path never asked, so an UNKNOWN row was
  ingested and produced a number anyway;
* public reads handed out the stored dictionaries themselves, so a caller could change a value that had already been hashed
  and the vintage identity would not move.
"""

import pytest

from app.economic_calendar import SCHEMA, CalendarRefusal, PointInTimeCalendar

KEY = "US.CPI.2026-02"


def arrival(kind, observed_at, **over):
    row = {"schema": SCHEMA, "event_key": KEY, "kind": kind, "observed_at": observed_at,
           "event_time": "2026-03-03T13:30:00Z", "unit": "percent_yoy", "period": "2026-02",
           "historical_availability": "KNOWN"}
    row.update(over)
    return row


def book(*rows):
    calendar = PointInTimeCalendar()
    calendar.add_all(rows)
    return calendar


# --- F4: publication and receipt are two boundaries, and both must survive --------------------------------------------------

def test_a_consensus_published_after_the_release_does_not_become_the_expectation():
    """Prior consensus 2.5. Actual 2.9 published 13:30, received 13:45. A 2.9 consensus published 13:34 arrives 13:35."""
    calendar = book(
        arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
        arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9),
        arrival("CONSENSUS", "2026-03-03T13:35:00Z", published_at="2026-03-03T13:34:00Z", consensus=2.9))
    result = calendar.surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["release_surprise"] == pytest.approx(0.4), "the expectation is what stood before the number was PUBLISHED"
    assert result["release_consensus"] == 2.5
    assert result["release_consensus_published_at"] == "2026-03-02T12:00:00+00:00"


def test_the_available_and_release_boundaries_are_both_reported_and_never_merged():
    calendar = book(
        arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
        arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9),
        arrival("CONSENSUS", "2026-03-03T13:35:00Z", published_at="2026-03-03T13:34:00Z", consensus=2.9))
    result = calendar.surprise(KEY, "2026-03-03T18:00:00Z")
    assert "release_surprise" in result and "available_surprise" in result
    assert result["available_consensus"] == 2.9, "what our system had in hand by the time it could see the number"
    assert result["available_surprise"] == pytest.approx(0.0)
    assert result["release_surprise"] != result["available_surprise"]
    assert "boundaries" in result and result["boundaries"]["release"] != result["boundaries"]["available"]


def test_nothing_is_knowable_before_local_receipt_under_either_boundary():
    calendar = book(
        arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
        arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9))
    early = calendar.surprise(KEY, "2026-03-03T13:40:00Z")
    assert early["release_surprise"] is None and early["available_surprise"] is None
    assert early["status"] != "RELEASED"


def test_a_later_revision_is_named_apart_from_the_original_release():
    calendar = book(
        arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
        arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9),
        arrival("REVISION", "2026-04-01T13:30:00Z", published_at="2026-04-01T13:30:00Z", actual=3.1))
    later = calendar.surprise(KEY, "2026-04-02T00:00:00Z")
    assert later["release_surprise"] == pytest.approx(0.4), "the original release surprise does not change with a revision"
    assert later["release_actual"] == 2.9
    assert later["revised_actual"] == 3.1
    assert later["revision_surprise"] == pytest.approx(0.6)


# --- F5: the availability rule must be on the path, not beside it -------------------------------------------------------------

def test_an_unknown_availability_row_cannot_produce_a_number():
    calendar = PointInTimeCalendar()
    calendar.add(arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5))
    calendar.add(arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9,
                         historical_availability="UNKNOWN"))
    result = calendar.surprise(KEY, "2026-03-04T00:00:00Z")
    assert result["release_surprise"] is None and result["available_surprise"] is None
    assert "UNKNOWN_HISTORICAL_AVAILABILITY" in str(result.get("reason", "")) or result["status"] != "RELEASED"
    view = calendar.view(KEY, "2026-03-04T00:00:00Z")
    assert "actual" not in view


def test_an_unknown_availability_row_is_retained_as_archive():
    calendar = PointInTimeCalendar()
    calendar.add(arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9,
                         historical_availability="UNKNOWN"))
    assert calendar.archive_rows() and calendar.archive_rows()[0]["actual"] == 2.9
    assert calendar.known_at("2026-03-04T00:00:00Z") == []


def test_an_undeclared_availability_is_refused_at_ingestion():
    calendar = PointInTimeCalendar()
    row = arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9)
    row.pop("historical_availability")
    with pytest.raises(CalendarRefusal, match="AVAILABILITY"):
        calendar.add(row)


# --- F6: a public read must not be a handle on history --------------------------------------------------------------------------

def test_a_returned_arrival_cannot_change_what_is_stored():
    calendar = book(arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9))
    rows = calendar.known_at("2026-03-04T00:00:00Z")
    rows[0]["actual"] = 99.0
    assert calendar.view(KEY, "2026-03-04T00:00:00Z")["actual"] == 2.9


def test_a_returned_view_cannot_change_the_next_one():
    calendar = book(arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9))
    view = calendar.view(KEY, "2026-03-04T00:00:00Z")
    view["actual"] = 99.0
    assert calendar.view(KEY, "2026-03-04T00:00:00Z")["actual"] == 2.9


def test_the_arrival_returned_by_add_is_also_detached():
    calendar = PointInTimeCalendar()
    stored, _disposition = calendar.add(arrival("ACTUAL", "2026-03-03T13:45:00Z",
                                                published_at="2026-03-03T13:30:00Z", actual=2.9))
    stored["actual"] = 99.0
    assert calendar.view(KEY, "2026-03-04T00:00:00Z")["actual"] == 2.9


# --- reordered delivery cannot change a value under one vintage -------------------------------------------------------------------

def test_reordered_delivery_gives_one_vintage_and_one_value():
    rows = [arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
            arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9, sequence=1),
            arrival("REVISION", "2026-04-01T13:30:00Z", published_at="2026-04-01T13:30:00Z", actual=3.1, sequence=2)]
    forward, backward = book(*rows), book(*reversed(rows))
    as_of = "2026-04-02T00:00:00Z"
    assert forward.vintage_identity(as_of) == backward.vintage_identity(as_of)
    assert forward.view(KEY, as_of) == backward.view(KEY, as_of)
    assert forward.surprise(KEY, as_of) == backward.surprise(KEY, as_of)


def test_tied_arrivals_with_a_declared_sequence_are_deterministic_in_any_delivery_order():
    rows = [arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
            arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9, sequence=1),
            arrival("REVISION", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:44:00Z", actual=3.1, sequence=2)]
    forward, backward = book(*rows), book(*reversed(rows))
    as_of = "2026-03-04T00:00:00Z"
    assert forward.view(KEY, as_of)["actual"] == backward.view(KEY, as_of)["actual"] == 3.1
