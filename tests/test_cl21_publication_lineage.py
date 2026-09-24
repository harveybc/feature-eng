"""CL21: the publication clock is a clock of its own, and the release boundary is the only thing it governs.

Three findings, one theme: the module kept saying "PUBLISHED" while computing with the order things happened to reach us.
Delivery order is a property of our plumbing -- a retry, a slow feed, a backfill -- and it is not the chronology of the
source. Where the two disagree, the code below must follow the source.

* CL21-a: the release boundary picked the LAST CONSENSUS TO ARRIVE before the release instead of the last one PUBLISHED.
  A 2.7 published 13:20 that arrives before a stale 2.5 published 13:00 leaves the 2.5 as "what the market expected", and a
  0.2 surprise is reported as 0.4. Reception is eligibility to know something; it is not the source's chronology.
* CL21-b: an arrival with no publication clock had one invented for it -- its receipt. A consensus received before the
  release but published nobody-knows-when then entered the release boundary and turned a real surprise into zero. A missing
  clock is a missing clock: the row is excluded from the release boundary and the exclusion is reported by name.
* CL21-c: when a revision arrived before the delayed original, the revision was labelled the release and the original the
  revision -- exactly backwards. The original release is the earliest-PUBLISHED actual whatever order the two arrived in.

Every fixture here is a real delivery pattern: a slow provider, a backfilled row, a revision that beats the original
through a different feed. None of them needs a data provider, because each is about a rule.
"""

import pytest

from app.economic_calendar import SCHEMA, PointInTimeCalendar

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


# --- CL21-a: the release boundary is ordered by publication, the available boundary by receipt -------------------------------

def _out_of_order_consensus_rows():
    """A 2.7 published 13:20 reaches us at 13:21; a stale 2.5 published 13:00 only lands at 13:30, after it."""
    return [arrival("CONSENSUS", "2026-03-03T13:21:00Z", published_at="2026-03-03T13:20:00Z", consensus=2.7),
            arrival("CONSENSUS", "2026-03-03T13:30:00Z", published_at="2026-03-03T13:00:00Z", consensus=2.5),
            arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9)]


def test_the_release_boundary_takes_the_last_consensus_published_not_the_last_to_arrive():
    result = book(*_out_of_order_consensus_rows()).surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["release_consensus"] == 2.7, "the market expected the newest number its source had published, not the one"\
                                               " our feed happened to hand over last"
    assert result["release_consensus_published_at"] == "2026-03-03T13:20:00+00:00"
    assert result["release_surprise"] == pytest.approx(0.2)


def test_the_available_boundary_still_follows_receipt_order():
    """The other half of the same fixture: what OUR system had in hand is exactly what arrived last, stale or not."""
    result = book(*_out_of_order_consensus_rows()).surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["available_consensus"] == 2.5
    assert result["available_consensus_observed_at"] == "2026-03-03T13:30:00+00:00"
    assert result["available_surprise"] == pytest.approx(0.4)
    assert result["release_surprise"] != result["available_surprise"], "two boundaries, never merged"


def test_delivery_order_cannot_change_either_boundary():
    rows = _out_of_order_consensus_rows()
    as_of = "2026-03-03T18:00:00Z"
    assert book(*rows).surprise(KEY, as_of) == book(*reversed(rows)).surprise(KEY, as_of)


# --- CL21-b: no publication clock, no place in the release boundary -----------------------------------------------------------

def _clockless_consensus_rows():
    """A 2.9 consensus with no publication clock lands at 13:20, ten minutes before the number is published."""
    return [arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
            arrival("CONSENSUS", "2026-03-03T13:20:00Z", consensus=2.9),
            arrival("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9)]


def test_a_consensus_without_a_publication_clock_cannot_enter_the_release_boundary():
    result = book(*_clockless_consensus_rows()).surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["release_consensus"] == 2.5, "guessing the missing clock from our receipt turns a real 0.4 into a 0.0"
    assert result["release_surprise"] == pytest.approx(0.4)


def test_the_excluded_arrival_is_reported_by_name_rather_than_dropped_in_silence():
    result = book(*_clockless_consensus_rows()).surprise(KEY, "2026-03-03T18:00:00Z")
    excluded = result["release_boundary_excluded"]
    assert len(excluded) == 1 and excluded[0]["consensus"] == 2.9
    assert "MISSING_PUBLICATION_CLOCK" in excluded[0]["reason"]


def test_a_clockless_consensus_is_still_eligible_for_the_available_boundary():
    """We did receive it before we could see the number, so our own system could indeed have computed with it."""
    result = book(*_clockless_consensus_rows()).surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["available_consensus"] == 2.9
    assert result["available_consensus_published_at"] is None
    assert result["available_surprise"] == pytest.approx(0.0)


def test_an_actual_without_a_publication_clock_has_no_release_boundary_at_all():
    """Without the release instant there is no window to pick an expectation from, and our receipt is not that instant:
    it would stretch the window forward and admit consensus published after the number."""
    calendar = book(arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
                    arrival("CONSENSUS", "2026-03-03T13:35:00Z", published_at="2026-03-03T13:34:00Z", consensus=2.7),
                    arrival("ACTUAL", "2026-03-03T13:45:00Z", actual=2.9))
    result = calendar.surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["release_surprise"] is None
    assert "MISSING_PUBLICATION_CLOCK" in result["release_reason"]
    assert result["available_surprise"] == pytest.approx(0.2), "what we could compute is unaffected by the missing clock"


# --- CL21-c: the original release is the earliest PUBLISHED actual ---------------------------------------------------------------

def _revision_overtakes_the_original():
    """The original is published 13:30 and stalls in a slow feed until 16:00; the revision, published 14:30 on another
    feed, is already here at 14:35. Arrival order says the revision came first; the source says it is the second word."""
    return [arrival("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5),
            arrival("REVISION", "2026-03-03T14:35:00Z", published_at="2026-03-03T14:30:00Z", actual=3.1),
            arrival("ACTUAL", "2026-03-03T16:00:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9)]


def test_a_revision_that_arrives_first_is_not_relabelled_the_release():
    result = book(*_revision_overtakes_the_original()).surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["release_actual"] == 2.9, "the release is the earliest number the source published, not the first to land"
    assert result["release_actual_published_at"] == "2026-03-03T13:30:00+00:00"
    assert result["release_surprise"] == pytest.approx(0.4)


def test_the_delayed_original_is_not_reported_as_the_revision_of_the_revision():
    result = book(*_revision_overtakes_the_original()).surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["revised_actual"] == 3.1 and result["revised_actual_kind"] == "REVISION"
    assert result["revision_surprise"] == pytest.approx(0.6), "the revised value against the SAME pre-release expectation"


def test_the_available_boundary_reports_the_actual_that_reached_us_first():
    """Our system saw 3.1 at 14:35 and nothing else; reporting the release value here would claim a number we did not have."""
    result = book(*_revision_overtakes_the_original()).surprise(KEY, "2026-03-03T18:00:00Z")
    assert result["available_actual"] == 3.1
    assert result["available_actual_observed_at"] == "2026-03-03T14:35:00+00:00"
    assert result["available_surprise"] == pytest.approx(0.6)


def test_the_lineage_does_not_depend_on_delivery_order_either():
    rows = _revision_overtakes_the_original()
    as_of = "2026-03-03T18:00:00Z"
    assert book(*rows).surprise(KEY, as_of) == book(*reversed(rows)).surprise(KEY, as_of)
