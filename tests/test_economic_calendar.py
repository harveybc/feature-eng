"""CAL01-CAL12 on deterministic fixtures: what was knowable, when, and what must be refused instead of invented.

Every case here is the counterexample from the specification written as the outcome it demands. They use a handful of rows
and no data provider, because each one is about a rule, not about a dataset -- and a rule that only holds on the data you
happened to have is not a rule.
"""

from datetime import datetime, timezone

import pytest

from app.economic_calendar import (SCHEMA, CalendarRefusal, PointInTimeCalendar, availability_checked, freshness,
                                   validate_arrival)

MONDAY = "2026-03-02T12:00:00Z"
TUESDAY = "2026-03-03T12:00:00Z"
WEDNESDAY = "2026-03-04T12:00:00Z"


def arrival(kind, observed_at, **over):
    row = {"schema": SCHEMA, "event_key": "US.CPI.2026-02", "kind": kind, "observed_at": observed_at,
           "event_time": "2026-03-03T13:30:00Z", "unit": "percent_yoy", "period": "2026-02"}
    row.update(over)
    return row


def calendar(*rows):
    book = PointInTimeCalendar()
    book.add_all(rows)
    return book


# --- CAL01: a schedule is knowable before its number is -------------------------------------------------------------------

def test_CAL01_tomorrows_schedule_is_available_and_tomorrows_actual_is_not():
    book = calendar(arrival("SCHEDULE", MONDAY))
    view = book.view("US.CPI.2026-02", MONDAY)
    assert view["status"] == "SCHEDULED"
    assert view["event_time"] == "2026-03-03T13:30:00+00:00"
    assert "actual" not in view
    assert book.surprise("US.CPI.2026-02", MONDAY)["surprise"] is None


# --- CAL02: published is not received ---------------------------------------------------------------------------------------

def test_CAL02_a_release_is_not_knowable_between_publication_and_receipt():
    book = calendar(arrival("SCHEDULE", MONDAY),
                    arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9))
    between = book.view("US.CPI.2026-02", "2026-03-03T13:35:00Z")
    assert between["status"] == "SCHEDULED", "published 13:30, received 13:45: at 13:35 nobody had it"
    assert book.surprise("US.CPI.2026-02", "2026-03-03T13:35:00Z")["surprise"] is None
    after = book.surprise("US.CPI.2026-02", "2026-03-03T13:50:00Z")
    assert after["surprise"] == pytest.approx(0.4)


def test_CAL02_an_arrival_observed_before_it_was_published_is_refused():
    with pytest.raises(CalendarRefusal, match="RECEIVED_BEFORE_PUBLISHED"):
        validate_arrival(arrival("ACTUAL", "2026-03-03T13:00:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9))


# --- CAL03: the surprise is frozen at what was known before the release ------------------------------------------------------

def test_CAL03_a_later_consensus_cannot_move_the_frozen_surprise():
    book = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=2.9),
                    arrival("CONSENSUS", "2026-03-03T14:00:00Z", consensus=2.9))
    later = book.surprise("US.CPI.2026-02", WEDNESDAY)
    assert later["consensus"] == 2.5, "the consensus published after the number is not what anyone was surprised against"
    assert later["surprise"] == pytest.approx(0.4)
    assert later["consensus_observed_at"] == "2026-03-02T12:00:00+00:00"


# --- CAL04: a revision changes later views only -----------------------------------------------------------------------------

def test_CAL04_a_late_revision_leaves_every_earlier_view_unchanged():
    book = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=2.9))
    before = book.view("US.CPI.2026-02", "2026-03-03T18:00:00Z")
    book.add(arrival("REVISION", "2026-04-01T13:30:00Z", actual=3.1))
    assert book.view("US.CPI.2026-02", "2026-03-03T18:00:00Z") == before, "the past did not change"
    after = book.view("US.CPI.2026-02", "2026-04-02T00:00:00Z")
    assert after["actual"] == 3.1 and after["revisions_known"] == 1
    assert book.surprise("US.CPI.2026-02", "2026-03-03T18:00:00Z")["surprise"] == pytest.approx(0.4)
    assert book.surprise("US.CPI.2026-02", "2026-04-02T00:00:00Z")["surprise"] == pytest.approx(0.6)


# --- CAL05: ambiguity, mixed units and incomparable periods refuse ------------------------------------------------------------

@pytest.mark.parametrize("stamp", ["2026-11-01T01:30:00", "2026-03-03 13:30:00", "not-a-time", ""])
def test_CAL05_an_ambiguous_or_naive_timestamp_refuses_before_any_tensor(stamp):
    with pytest.raises(CalendarRefusal):
        validate_arrival(arrival("SCHEDULE", stamp))


def test_CAL05_mixed_units_or_periods_refuse_rather_than_subtract():
    book = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=104.2, unit="index_level"))
    with pytest.raises(CalendarRefusal, match="INCOMPARABLE_SERIES"):
        book.view("US.CPI.2026-02", WEDNESDAY)
    other = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                     arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=2.9, period="2026-01"))
    with pytest.raises(CalendarRefusal, match="INCOMPARABLE_SERIES"):
        other.view("US.CPI.2026-02", WEDNESDAY)


def test_CAL05_a_number_without_its_unit_or_period_is_refused():
    for field in ("unit", "period"):
        with pytest.raises(CalendarRefusal, match=field.upper()):
            validate_arrival(arrival("ACTUAL", MONDAY, actual=2.9, **{field: None}))


# --- CAL06: missing consensus and a zero scale are reported, never invented ----------------------------------------------------

def test_CAL06_a_missing_consensus_gives_no_surprise_and_says_so():
    book = calendar(arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=2.9))
    result = book.surprise("US.CPI.2026-02", WEDNESDAY)
    assert result["surprise"] is None
    assert "NO_CONSENSUS_BEFORE_RELEASE" in result["reason"]
    assert "invented" in result["reason"]


@pytest.mark.parametrize("scale", [0, 0.0, -1.5])
def test_CAL06_a_non_positive_residual_scale_is_not_a_standardized_surprise(scale):
    book = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=2.9))
    result = book.surprise("US.CPI.2026-02", WEDNESDAY, scale=scale)
    assert result["surprise"] == pytest.approx(0.4), "the raw surprise is still reported"
    assert result["standardized"] is None
    assert "NON_POSITIVE_RESIDUAL_SCALE" in result["standardized_reason"]


def test_CAL06_a_positive_scale_standardizes():
    book = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=2.9))
    assert book.surprise("US.CPI.2026-02", WEDNESDAY, scale=0.2)["standardized"] == pytest.approx(2.0)


# --- CAL07: the future cannot reach backwards ---------------------------------------------------------------------------------

def test_CAL07_an_arrival_from_the_future_cannot_change_an_earlier_feature():
    book = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=2.9))
    before_identity = book.vintage_identity(WEDNESDAY)
    before_view = book.view("US.CPI.2026-02", WEDNESDAY)
    book.add(arrival("REVISION", "2026-05-01T00:00:00Z", actual=9.9))
    assert book.vintage_identity(WEDNESDAY) == before_identity
    assert book.view("US.CPI.2026-02", WEDNESDAY) == before_view


# --- CAL08: duplicates, reordering and restart give the same vintage -----------------------------------------------------------

def test_CAL08_duplicate_and_reordered_arrivals_reproduce_one_vintage():
    rows = [arrival("SCHEDULE", MONDAY), arrival("CONSENSUS", MONDAY, consensus=2.5),
            arrival("ACTUAL", "2026-03-03T13:30:00Z", actual=2.9)]
    forward = calendar(*rows)
    backward = calendar(*reversed(rows))
    duplicated = calendar(*(rows + rows))
    assert forward.vintage_identity(WEDNESDAY) == backward.vintage_identity(WEDNESDAY) == duplicated.vintage_identity(WEDNESDAY)
    assert forward.view("US.CPI.2026-02", WEDNESDAY) == backward.view("US.CPI.2026-02", WEDNESDAY)
    assert duplicated.add(rows[0])[1] == "DUPLICATE"
    restarted = calendar(*rows)                                  # a fresh process, same arrivals
    assert restarted.surprise("US.CPI.2026-02", WEDNESDAY) == forward.surprise("US.CPI.2026-02", WEDNESDAY)


# --- CAL09: unknown availability keeps the archive and refuses point-in-time use ------------------------------------------------

def test_CAL09_unknown_historical_availability_refuses_point_in_time_use():
    unknown = availability_checked(arrival("ACTUAL", MONDAY, actual=2.9), historical_availability="UNKNOWN")
    assert unknown["point_in_time_usable"] is False
    assert unknown["archive_metadata_retained"] is True
    assert "nobody verified" in unknown["reason"]
    known = availability_checked(arrival("ACTUAL", MONDAY, actual=2.9), historical_availability="KNOWN")
    assert known["point_in_time_usable"] is True
    with pytest.raises(CalendarRefusal, match="AVAILABILITY_MUST_BE_DECLARED"):
        availability_checked({}, historical_availability="probably")


# --- CAL10: equal clocks respect observed order, or exclude conservatively --------------------------------------------------------

def test_CAL10_two_values_at_the_same_instant_with_an_observed_order_respect_it():
    book = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", TUESDAY, actual=2.9, sequence=1),
                    arrival("REVISION", TUESDAY, actual=3.1, sequence=2))
    view = book.view("US.CPI.2026-02", WEDNESDAY)
    assert view["actual"] == 3.1 and view["actual_kind"] == "REVISION"


def test_CAL10_two_values_at_the_same_instant_without_an_order_are_not_silently_chosen():
    book = calendar(arrival("CONSENSUS", MONDAY, consensus=2.5),
                    arrival("ACTUAL", TUESDAY, actual=2.9),
                    arrival("REVISION", TUESDAY, actual=3.1))
    view = book.view("US.CPI.2026-02", WEDNESDAY)
    assert view["status"] == "AMBIGUOUS_SEQUENCE"
    assert "actual" not in view
    assert len(view["tied_arrivals"]) == 2


# --- CAL11: simultaneous events stay separate; cancellations and reschedules are respected ------------------------------------------

def test_CAL11_two_events_at_the_same_instant_remain_two_events():
    book = PointInTimeCalendar()
    book.add(arrival("ACTUAL", TUESDAY, event_key="US.CPI.2026-02", actual=2.9))
    book.add(arrival("ACTUAL", TUESDAY, event_key="EA.HICP.2026-02", actual=1.8))
    assert book.view("US.CPI.2026-02", WEDNESDAY)["actual"] == 2.9
    assert book.view("EA.HICP.2026-02", WEDNESDAY)["actual"] == 1.8
    assert len(book.known_at(WEDNESDAY)) == 2


def test_CAL11_a_cancellation_and_a_reschedule_are_respected():
    book = calendar(arrival("SCHEDULE", MONDAY))
    book.add(arrival("SCHEDULE_UPDATE", TUESDAY, event_time="2026-03-05T13:30:00Z"))
    updated = book.view("US.CPI.2026-02", WEDNESDAY)
    assert updated["event_time"] == "2026-03-05T13:30:00+00:00" and updated["schedule_updates"] == 1
    assert book.view("US.CPI.2026-02", MONDAY)["event_time"] == "2026-03-03T13:30:00+00:00"
    book.add(arrival("CANCELLATION", "2026-03-04T18:00:00Z"))
    assert book.view("US.CPI.2026-02", "2026-03-05T00:00:00Z")["status"] == "CANCELLED"
    assert book.view("US.CPI.2026-02", WEDNESDAY)["status"] == "SCHEDULED", "the cancellation was not yet known"


# --- CAL12: a late result is late -------------------------------------------------------------------------------------------------

def test_CAL12_a_computation_finishing_after_the_deadline_is_stale_not_backdated():
    late = freshness("2026-03-03T13:31:00Z", "2026-03-03T13:30:00Z")
    assert late["status"] == "STALE" and late["usable_for_that_decision"] is False
    assert late["late_by_seconds"] == 60.0
    assert late["computed_at"] == "2026-03-03T13:31:00+00:00", "the timestamp is not moved back to the deadline"
    in_time = freshness("2026-03-03T13:29:00Z", "2026-03-03T13:30:00Z")
    assert in_time["status"] == "IN_TIME" and in_time["usable_for_that_decision"] is True


# --- the boundary this module keeps ------------------------------------------------------------------------------------------------

def test_a_price_reaction_is_never_accepted_as_a_surprise():
    """There is deliberately no path from a market move to a `surprise` here: the input is a consensus, or there is none."""
    book = calendar(arrival("ACTUAL", TUESDAY, actual=2.9))
    result = book.surprise("US.CPI.2026-02", WEDNESDAY)
    assert result["surprise"] is None
    assert not hasattr(book, "surprise_from_returns")
