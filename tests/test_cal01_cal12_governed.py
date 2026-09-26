"""CAL01-CAL12 against the REAL economic-calendar resources, through the real as-of entrypoint.

`tests/test_economic_calendar.py` already implements CAL01-CAL12 on deterministic fixtures, and it should: a rule that
only holds on the data you happened to have is not a rule. This file asks the other half of RP150's question -- can each
case be EVALUATED on the bytes this machine actually holds? -- and the answer is per case and per field.

The rule this file follows, and the reason it exists rather than a shorter document:

* a case whose fields are all present in a resource RUNS, on that resource's real values and real clocks, with the
  arithmetic done by hand in the assertion;
* a case whose fields are NOT present is present here anyway and SKIPPED BY NAME, and the skip message names the missing
  field role and the resource it is missing from. A case that is quietly absent cannot be distinguished later from a
  case somebody decided not to test, which is how an unevaluated contract comes to be described as an implemented one.

Which roles each resource has is not asserted here. It is MEASURED by `feature_eng_m5phet.calendar_inventory`, from the
same bytes, in the same process, and the skip decision reads that measurement -- so the skips cannot drift away from the
data. `docs/CALENDAR_DATASET_INVENTORY.md` is that measurement written out.

One declaration is made visibly rather than hidden. No resource on this machine carries a `historical_availability`
column, so `app.economic_calendar` would archive every real row and admit none to a point-in-time view. CAL09-governed
proves exactly that, on real rows. The other cases need a row INSIDE a view to have anything to assert, so they declare
`historical_availability="KNOWN"` through `_DECLARED_BY_THIS_TEST_NOT_BY_THE_BYTES` -- a declaration this test makes so
that the as-of MACHINERY can be exercised on real values and real instants. It certifies nothing about the dataset, and
CAL09-governed is the case that shows what the honest declaration costs.

No network, no provider call, no model. Everything is read from named files whose digests the inventory records.
"""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

import pytest

from app.economic_calendar import (SCHEMA, CalendarRefusal, PointInTimeCalendar, availability_checked, freshness,
                                   instant, validate_arrival)
from feature_eng_m5phet.calendar_inventory import RESOURCES, measure_resource

#: the sibling-repository root the resource paths hang from
ROOT = Path(__file__).resolve().parents[3]

#: A declaration this FILE makes, because the bytes do not make it. See the module docstring, and CAL09-governed.
_DECLARED_BY_THIS_TEST_NOT_BY_THE_BYTES = "KNOWN"

#: Which resource each governed case is evaluated against, and which field roles it needs to be evaluable at all.
#: One row per case: this table IS the one-to-one map from the specification's case list to what runs here.
GOVERNED_CASES = {
    "CAL01": ("fxmacrodata_release_calendar", ("schedule_instant",)),
    "CAL02": ("fxmacrodata_announcements", ("publication_instant", "receipt_instant")),
    "CAL03": ("fxmacrodata_announcements", ("consensus", "publication_instant")),
    "CAL04": ("fxmacrodata_announcements", ("revision_marker", "vintage_version")),
    "CAL05": ("archive_2011_2021", ("unit", "reference_period")),
    "CAL06": ("fred_cpi_yoy_actuals", ("actual", "unit", "reference_period")),
    "CAL07": ("fxmacrodata_announcements", ("receipt_instant",)),
    "CAL08": ("fxmacrodata_announcements", ("receipt_instant",)),
    "CAL09": ("fxmacrodata_announcements", ()),
    "CAL10": ("fxmacrodata_announcements", ("observed_sequence",)),
    "CAL11": ("fxmacrodata_announcements", ("cancellation_state",)),
    "CAL12": ("fxmacrodata_announcements", ()),
}

_MEASURED: dict[str, dict] = {}


def inventory(resource):
    """The measured record for one resource, measured once per session from the bytes themselves."""
    if resource not in _MEASURED:
        _MEASURED[resource] = measure_resource(resource, ROOT)
    return _MEASURED[resource]


def governed(case):
    """Return the measured record, or skip BY NAME listing every field role the resource does not carry.

    The skip is never a bare `pytest.skip("no data")`: it names the case, the resource and each missing role, because
    "CAL04 is untested because `fxmacrodata_announcements` carries no revision_marker" is a finding and "skipped" is not.
    """
    resource, needed = GOVERNED_CASES[case]
    record = inventory(resource)
    if record["status"] != "MEASURED":
        pytest.skip(f"{case}: the resource {resource} ({RESOURCES[resource]['path']}) is not on this machine")
    missing = [role for role in needed if not record["field_roles"][role]["present"]]
    if missing:
        pytest.skip(f"{case}: NOT EVALUABLE against {resource} -- missing field role(s) "
                    f"{', '.join(missing)}. {RESOURCES[resource]['path']} carries "
                    f"{sorted(r for r, f in record['field_roles'].items() if f['present'])}")
    return record


def file_grain_receipt(resource):
    """The resource's own `acquired_at`: the ONE receipt clock it has, and it dates the download, not a release."""
    received = inventory(resource)["provenance"]["file_grain_receipt_instant"]
    assert received, f"{resource} has no provenance acquired_at to use as a receipt clock"
    return instant(received, "file_grain_receipt_instant")


def read_parquet(resource):
    pd = pytest.importorskip("pandas")
    return pd.read_parquet(ROOT / RESOURCES[resource]["path"])


def arrival(kind, *, event_key, observed_at, event_time, **over):
    row = {"schema": SCHEMA, "event_key": event_key, "kind": kind,
           "observed_at": observed_at, "event_time": event_time,
           "historical_availability": _DECLARED_BY_THIS_TEST_NOT_BY_THE_BYTES}
    row.update(over)
    return row


# --- CAL01: a real future schedule is knowable, and its actual is not --------------------------------------------------

def test_CAL01_governed_a_real_scheduled_release_is_known_and_its_actual_is_absent():
    """The forward calendar's own rows, received when the file was acquired. Nothing about the outcome is knowable."""
    governed("CAL01")
    frame = read_parquet("fxmacrodata_release_calendar")
    received = file_grain_receipt("fxmacrodata_release_calendar")
    future = frame[frame["announcement_datetime_utc"] > received.isoformat()]
    assert len(future) > 0, "the forward calendar must carry a release later than the instant the file arrived"
    row = future.iloc[0]
    event_key = f"{row['currency']}.{row['release']}.{row['announcement_datetime_utc'].date().isoformat()}"
    book = PointInTimeCalendar()
    book.add(arrival("SCHEDULE", event_key=event_key, observed_at=received,
                     event_time=row["announcement_datetime_utc"].to_pydatetime()))
    view = book.view(event_key, received + timedelta(seconds=1))
    assert view["status"] == "SCHEDULED"
    assert view["event_time"] == row["announcement_datetime_utc"].to_pydatetime().isoformat()
    assert "actual" not in view, "a scheduled release has no value, and the resource carries none"
    assert book.surprise(event_key, received + timedelta(seconds=1))["release_surprise"] is None


# --- CAL02: published is not received ---------------------------------------------------------------------------------

def test_CAL02_governed_publication_and_receipt_are_two_clocks_the_resource_does_not_both_carry():
    governed("CAL02")
    pytest.fail("unreachable: CAL02 needs a per-release receipt clock and the skip above must fire")


# --- CAL03: the frozen pre-release consensus --------------------------------------------------------------------------

def test_CAL03_governed_a_frozen_pre_release_consensus_needs_a_consensus_and_a_publication_clock():
    governed("CAL03")
    pytest.fail("unreachable: no resource on this machine carries a consensus AND a publication instant")


# --- CAL04: a revision changes later views only -----------------------------------------------------------------------

def test_CAL04_governed_a_revision_that_leaves_earlier_views_alone_needs_a_revision_marker():
    governed("CAL04")
    pytest.fail("unreachable: CAL04 needs a revision marker or a vintage version and the skip above must fire")


# --- CAL05: the archive's own wall clock, and its own units -----------------------------------------------------------

def test_CAL05_governed_the_archives_real_wall_clock_is_refused_before_any_tensor():
    """The 2011-2021 archive's clock is measured naive here, from its own first rows, and the real entrypoint refuses it.

    This limb needs no field the archive lacks: the refusal IS the outcome the case demands.
    """
    record = inventory("archive_2011_2021")
    if record["status"] != "MEASURED":
        pytest.skip(f"CAL05: {RESOURCES['archive_2011_2021']['path']} is not on this machine")
    assert record["clock"]["measured_timezone_aware"] is False
    path = ROOT / RESOURCES["archive_2011_2021"]["path"]
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        first = next(csv.reader(handle))
    columns = RESOURCES["archive_2011_2021"]["columns"]
    row = dict(zip(columns, (cell.strip() for cell in first)))
    stamp = f"{row['event_date'].replace('/', '-')}T{row['event_time']}"        # the archive's own two columns, joined
    with pytest.raises(CalendarRefusal, match="AMBIGUOUS_LOCAL_TIME|UNREADABLE_TIMESTAMP"):
        validate_arrival(arrival("SCHEDULE", event_key=f"{row['country']}.{row['description']}",
                                 observed_at=stamp, event_time=stamp))


def test_CAL05_governed_comparing_two_of_the_archives_units_needs_a_reference_period():
    governed("CAL05")
    pytest.fail("unreachable: the archive carries six units and no reference period, and the skip above must fire")


# --- CAL06: a missing consensus and a zero scale ----------------------------------------------------------------------

def test_CAL06_governed_a_real_release_with_an_empty_consensus_column_yields_no_surprise():
    """FRED's CPI YoY actuals: real value, real unit, real reference period -- and `consensus_estimate` null in every row.

    That is CAL06's condition met by the data rather than by a fixture, so the case runs and the arithmetic is checked
    by hand: no consensus was ever received, therefore there is no surprise, and zero is not reported in its place.
    """
    record = governed("CAL06")
    assert record["field_roles"]["consensus"]["non_null_rows"] == 0, \
        "this case is about a consensus column that is present and entirely empty"
    frame = read_parquet("fred_cpi_yoy_actuals")
    received = file_grain_receipt("fred_cpi_yoy_actuals")
    row = frame.iloc[-1]
    assert row["consensus_estimate"] is None or row["consensus_estimate"] != row["consensus_estimate"]
    event_key = f"{row['fred_series']}.{row['date'].date().isoformat()}"
    book = PointInTimeCalendar()
    book.add(arrival("ACTUAL", event_key=event_key, observed_at=received,
                     event_time=received, actual=float(row["actual"]),
                     unit=str(row["transform"]), period=row["date"].date().isoformat()))
    result = book.surprise(event_key, received + timedelta(seconds=1))
    assert result["available_actual"] == pytest.approx(float(row["actual"]))
    assert result["release_surprise"] is None and result["available_surprise"] is None
    assert "NO_CONSENSUS_BEFORE_THE_BOUNDARY" in result["available_reason"]
    assert "invented" in result["available_reason"]
    scaled = book.surprise(event_key, received + timedelta(seconds=1), scale=0.0)
    assert scaled["standardized"] is None
    assert "NON_POSITIVE_RESIDUAL_SCALE" in scaled["standardized_reason"]


# --- CAL07: the future cannot reach backwards -------------------------------------------------------------------------

def test_CAL07_governed_prefix_invariance_needs_a_per_release_receipt_clock():
    governed("CAL07")
    pytest.fail("unreachable: CAL07 needs a per-release receipt clock and the skip above must fire")


def test_CAL07_governed_a_file_grain_receipt_clock_collapses_the_whole_file_into_one_as_of_view():
    """What the file-grain clock actually buys, measured: one view, not a sequence of them.

    Every row of `announcements.parquet` reached this machine in one download, so under the only receipt clock the
    resource has, all of its releases became knowable at the same instant. The as-of machinery then has exactly two
    answers -- nothing, and everything -- which is why CAL07's real case is skipped above rather than approximated here.
    """
    record = inventory("fxmacrodata_announcements")
    if record["status"] != "MEASURED":
        pytest.skip("CAL07: announcements.parquet is not on this machine")
    frame = read_parquet("fxmacrodata_announcements")
    received = file_grain_receipt("fxmacrodata_announcements")
    series = frame[(frame["currency"] == "EUR") & (frame["indicator"] == "risk_free_rate")].head(5)
    assert len(series) == 5
    book = PointInTimeCalendar()
    for _, row in series.iterrows():
        book.add(arrival("ACTUAL", event_key=f"EUR.risk_free_rate.{row['date']}", observed_at=received,
                         event_time=row["announcement_datetime_utc"].to_pydatetime(),
                         published_at=row["announcement_datetime_utc"].to_pydatetime(),
                         actual=float(row["val"]),
                         unit="UNIT_NOT_IN_THE_RESOURCE", period=str(row["date"])))
    def known(cutoff):
        return sorted(row["arrival_sha256"] for row in book.known_at(cutoff))

    assert known(received - timedelta(seconds=1)) == [], "before the download nothing was knowable"
    assert len(known(received)) == 5, "at the download everything was knowable at once"
    # a year later the knowable set is byte-identical: the file has exactly one non-empty as-of view. (The digest from
    # `vintage_identity` stamps the cutoff into itself by design, so the comparison is over the arrivals themselves.)
    assert known(received + timedelta(days=365)) == known(received)
    assert book.vintage_identity(received) != book.vintage_identity(received + timedelta(days=365))


# --- CAL08: duplicates, reordering and restart ------------------------------------------------------------------------

def test_CAL08_governed_reordering_and_restart_on_real_rows_reproduce_one_vintage_identity():
    """A real slice, delivered forwards, backwards and twice. This limb needs no clock the resource lacks, because the
    identity is over CONTENT: whatever order the rows arrive in, the vintage is the same, and a restart reproduces it."""
    record = inventory("fxmacrodata_announcements")
    if record["status"] != "MEASURED":
        pytest.skip("CAL08: announcements.parquet is not on this machine")
    frame = read_parquet("fxmacrodata_announcements")
    received = file_grain_receipt("fxmacrodata_announcements")
    slice_ = frame[(frame["currency"] == "EUR") & (frame["indicator"] == "risk_free_rate")].head(4)
    rows = [arrival("ACTUAL", event_key=f"EUR.risk_free_rate.{row['date']}", observed_at=received,
                    event_time=row["announcement_datetime_utc"].to_pydatetime(),
                    actual=float(row["val"]), unit="UNIT_NOT_IN_THE_RESOURCE", period=str(row["date"]))
            for _, row in slice_.iterrows()]

    def book_of(sequence):
        book = PointInTimeCalendar()
        book.add_all(sequence)
        return book

    forward, backward, twice, restarted = (book_of(rows), book_of(list(reversed(rows))),
                                           book_of(rows + rows), book_of(rows))
    cutoff = received + timedelta(seconds=1)
    assert forward.vintage_identity(cutoff) == backward.vintage_identity(cutoff) == twice.vintage_identity(cutoff)
    assert restarted.vintage_identity(cutoff) == forward.vintage_identity(cutoff)
    assert forward.add(rows[0])[1] == "DUPLICATE"


def test_CAL08_governed_a_per_release_receipt_clock_is_still_what_a_replay_would_need():
    governed("CAL08")
    pytest.fail("unreachable: CAL08's replay case needs a per-release receipt clock")


# --- CAL09: unknown availability keeps the archive and refuses the view -----------------------------------------------

def test_CAL09_governed_no_resource_declares_historical_availability_so_a_real_row_is_archived_not_used():
    """The case that governs all the others, run on a real row.

    Not one of the five resources carries a `historical_availability` column. The honest declaration for every row in
    them is therefore UNKNOWN, and the real entrypoint keeps the row and refuses the point-in-time use -- which is why
    every other governed case above had to declare KNOWN in its own name.
    """
    governed("CAL09")
    for name in RESOURCES:
        record = inventory(name)
        if record["status"] == "MEASURED":
            assert record["field_roles"]["historical_availability"]["present"] is False, name
    frame = read_parquet("fxmacrodata_announcements")
    received = file_grain_receipt("fxmacrodata_announcements")
    row = frame.iloc[0]
    event_key = f"{row['currency']}.{row['indicator']}.{row['date']}"
    honest = {"schema": SCHEMA, "event_key": event_key, "kind": "ACTUAL", "observed_at": received,
              "event_time": row["announcement_datetime_utc"].to_pydatetime(),
              "actual": float(row["val"]), "unit": "UNIT_NOT_IN_THE_RESOURCE", "period": str(row["date"]),
              "historical_availability": "UNKNOWN"}
    book = PointInTimeCalendar()
    stored, status = book.add(honest)
    assert status == "ARCHIVED_NOT_POINT_IN_TIME"
    assert book.known_at(received + timedelta(days=1)) == []
    assert len(book.archive_rows()) == 1
    assert book.archive_rows()[0]["actual"] == pytest.approx(float(row["val"])), "the archive keeps the value"
    verdict = availability_checked(honest, historical_availability="UNKNOWN")
    assert verdict["point_in_time_usable"] is False and verdict["archive_metadata_retained"] is True
    with pytest.raises(CalendarRefusal, match="AVAILABILITY_MUST_BE_DECLARED"):
        validate_arrival({**honest, "historical_availability": "probably"})


# --- CAL10: equal clocks, observed sequence or conservative exclusion --------------------------------------------------

def test_CAL10_governed_real_rows_collide_at_one_instant_and_neither_is_silently_chosen():
    """Measured, then exercised. 45 (currency, indicator, reference period) keys of `announcements.parquet` carry more
    than one value at ONE publication instant, one of them fourteen values between 21 and 1.1e9 -- so `indicator` is not
    a series identity in this resource. The resource has no observed sequence to break the tie, and the real entrypoint
    reports neither value as current rather than picking one."""
    record = inventory("fxmacrodata_announcements")
    if record["status"] != "MEASURED":
        pytest.skip("CAL10: announcements.parquet is not on this machine")
    rungs = {tuple(r["key"]): r for r in record["vintages"]["key_ladder"]}
    tied = rungs[("currency", "indicator", "date", "announcement_datetime_utc")]
    assert tied["keys_whose_rows_disagree_about_the_value"] > 0, "this case needs a real collision to exercise"
    assert record["field_roles"]["observed_sequence"]["present"] is False

    frame = read_parquet("fxmacrodata_announcements")
    grouped = frame.groupby(["currency", "indicator", "date", "announcement_datetime_utc"])["val"].nunique()
    key = grouped[grouped > 1].index[0]
    rows = frame[(frame["currency"] == key[0]) & (frame["indicator"] == key[1])
                 & (frame["date"] == key[2]) & (frame["announcement_datetime_utc"] == key[3])]
    values = sorted({float(v) for v in rows["val"]})
    assert len(values) > 1
    received = file_grain_receipt("fxmacrodata_announcements")
    event_key = f"{key[0]}.{key[1]}.{key[2]}"
    book = PointInTimeCalendar()
    for value in values[:2]:
        book.add(arrival("ACTUAL", event_key=event_key, observed_at=received,
                         event_time=key[3].to_pydatetime(), published_at=key[3].to_pydatetime(),
                         actual=value, unit="UNIT_NOT_IN_THE_RESOURCE", period=str(key[2])))
    view = book.view(event_key, received + timedelta(seconds=1))
    assert view["status"] == "AMBIGUOUS_SEQUENCE"
    assert "actual" not in view, "with no observed order, neither real value is reported as the current one"
    assert len(view["tied_arrivals"]) == 2


def test_CAL10_governed_respecting_an_observed_order_needs_an_observed_sequence():
    governed("CAL10")
    pytest.fail("unreachable: the resource carries no observed sequence and the skip above must fire")


# --- CAL11: simultaneous events stay separate; cancellations are respected ---------------------------------------------

def test_CAL11_governed_two_real_events_at_one_instant_remain_two_events():
    """2,566 publication instants in `announcements.parquet` carry more than one distinct event key. One of them is
    exercised here: both survive, and neither is silently selected as the row for that instant."""
    record = inventory("fxmacrodata_announcements")
    if record["status"] != "MEASURED":
        pytest.skip("CAL11: announcements.parquet is not on this machine")
    frame = read_parquet("fxmacrodata_announcements")
    received = file_grain_receipt("fxmacrodata_announcements")
    counts = frame.groupby("announcement_datetime_utc")[["currency", "indicator"]].nunique().sum(axis=1)
    moment = counts[counts > 2].index[0]
    together = frame[frame["announcement_datetime_utc"] == moment].drop_duplicates(
        subset=["currency", "indicator", "date"]).head(2)
    assert len(together) == 2
    book = PointInTimeCalendar()
    keys = []
    for _, row in together.iterrows():
        key = f"{row['currency']}.{row['indicator']}.{row['date']}"
        keys.append((key, float(row["val"])))
        book.add(arrival("ACTUAL", event_key=key, observed_at=received, event_time=moment.to_pydatetime(),
                         actual=float(row["val"]), unit="UNIT_NOT_IN_THE_RESOURCE", period=str(row["date"])))
    assert len({k for k, _ in keys}) == 2, "two distinct events, published at one instant"
    cutoff = received + timedelta(seconds=1)
    for key, value in keys:
        assert book.view(key, cutoff)["actual"] == pytest.approx(value)
    assert len(book.known_at(cutoff)) == 2


def test_CAL11_governed_a_cancellation_or_a_reschedule_needs_a_cancellation_state():
    governed("CAL11")
    pytest.fail("unreachable: no resource carries a cancellation state and the skip above must fire")


# --- CAL12: a late result is late -------------------------------------------------------------------------------------

def test_CAL12_governed_a_real_release_instant_as_a_deadline_makes_a_late_result_stale():
    """Clock arithmetic against a real publication instant: it needs no field the resource lacks, so it runs."""
    governed("CAL12")
    frame = read_parquet("fxmacrodata_announcements")
    deadline = frame["announcement_datetime_utc"].iloc[0].to_pydatetime()
    late = freshness(deadline + timedelta(seconds=90), deadline)
    assert late["status"] == "STALE" and late["usable_for_that_decision"] is False
    assert late["late_by_seconds"] == 90.0
    assert late["computed_at"] == (deadline + timedelta(seconds=90)).isoformat(), "not moved back to the deadline"
    early = freshness(deadline - timedelta(seconds=1), deadline)
    assert early["status"] == "IN_TIME" and early["usable_for_that_decision"] is True


# --- the map itself ----------------------------------------------------------------------------------------------------

def test_every_specified_case_has_a_governed_test_named_after_it():
    """CAL01-CAL12, each by its number. A case that had been dropped would fail here rather than go unnoticed."""
    source = Path(__file__).read_text(encoding="utf-8")
    for number in range(1, 13):
        case = f"CAL{number:02d}"
        assert case in GOVERNED_CASES, f"{case} is not in the governed case table"
        assert f"def test_{case}_governed" in source, f"{case} has no governed test named after it"


def test_the_inventory_the_skips_read_is_the_inventory_that_is_published():
    """The published document and these skips must not drift: both come from the same measurement of the same bytes."""
    published = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "calendar_dataset_inventory.json"
    if not published.is_file():
        pytest.skip("the published inventory JSON is not in this checkout")
    recorded = {r["resource"]: r for r in json.loads(published.read_text(encoding="utf-8"))["resources"]}
    for name in RESOURCES:
        measured = inventory(name)
        if measured["status"] != "MEASURED" or recorded.get(name, {}).get("status") != "MEASURED":
            continue
        assert measured["sha256"] == recorded[name]["sha256"], f"{name}: the published inventory is stale"
        assert ({r: f["present"] for r, f in measured["field_roles"].items()}
                == {r: f["present"] for r, f in recorded[name]["field_roles"].items()}), name
