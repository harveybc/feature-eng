"""Measuring what an archive's naive wall clock means, and refusing to guess it where it cannot be measured.

Every test here builds a synthetic pair of archives in which the answer is known before the job runs: an
announcement archive whose observed instants encode a publication convention, and a consensus archive whose wall
clock is a chosen offset from those instants -- one offset for one span, another for the next. What the module
writes must be those offsets, at those dates, and `UNDETERMINED` everywhere the evidence was not there.

The last group is the one that matters most: a series whose publication convention CHANGED between the two archives'
spans carries a constant bias, and it must be thrown out of the estimate by name rather than averaged into it.
"""

import csv
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from feature_eng_m5phet import calendar_clock

NEW_YORK = ZoneInfo("America/New_York")
SYDNEY = ZoneInfo("Australia/Sydney")

#: the two economies the synthetic archives carry. Their daylight-saving calendars are in opposite seasons, which is
#: the whole point: agreement between them cannot be an artefact of one convention.
SERIES = (("United States", "USD", "initial_jobless_claims", "Initial Jobless Claims", NEW_YORK, 8, 30, 3),
          ("United States", "USD", "non_farm_payrolls", "Nonfarm Payrolls", NEW_YORK, 8, 30, 4),
          ("Australia", "AUD", "retail_sales", "Retail Sales", SYDNEY, 11, 30, 1),
          ("Australia", "AUD", "building_approvals", "Building Approvals", SYDNEY, 11, 30, 2))

ANNOUNCEMENT_HEADER = ["currency", "indicator", "date", "val", "announcement_datetime_utc"]


def announcement_rows(start, weeks):
    """One announcement per week per series, at the convention's fixed local time, as an observed UTC instant."""
    out = []
    for country, currency, indicator, _label, zone, hour, minute, weekday in SERIES:
        day = start
        while day.weekday() != weekday:
            day += timedelta(days=1)
        for index in range(weeks):
            local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
            out.append([currency, indicator, day.date().isoformat(), "1.0",
                        local.astimezone(timezone.utc).isoformat()])
            day += timedelta(days=7)
    return out


def archive_rows(start, weeks, offsets, *, shifted=()):
    """The same releases, written as the archive writes them: a naive wall clock, `offsets(date)` from UTC.

    A series named in `shifted` is written an hour away from its convention, as a release whose publication time
    changed between the two archives' spans would be.
    """
    out = []
    for country, _currency, _indicator, label, zone, hour, minute, weekday in SERIES:
        day = start
        while day.weekday() != weekday:
            day += timedelta(days=1)
        for _index in range(weeks):
            local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
            instant = local.astimezone(timezone.utc)
            wall = instant + timedelta(seconds=offsets(day.date()))
            if label in shifted:
                wall += timedelta(hours=1)
            out.append([wall.strftime("%Y/%m/%d"), wall.strftime("%H:%M:%S"), country,
                        "Moderate Volatility Expected", label, "", "% ", "1.0", "0.9", "0.8"])
            day += timedelta(days=7)
    return out


def write(tmp_path, *, start=datetime(2016, 1, 4), weeks=200, offsets=None, shifted=(), announce_weeks=60,
          announce_start=datetime(2025, 1, 6), name=""):
    offsets = offsets or (lambda day: -5 * 3600)
    archive = tmp_path / f"archive{name}.csv"
    with archive.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(archive_rows(start, weeks, offsets, shifted=shifted))
    announcements = tmp_path / f"announcements{name}.csv"
    with announcements.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANNOUNCEMENT_HEADER)
        writer.writerows(announcement_rows(announce_start, announce_weeks))
    return archive, announcements


def determined(document):
    return [period for period in document["periods"] if period["status"] == calendar_clock.STATUS_DETERMINED]


# ------------------------------------------------------------------------------- the offset is measured, not typed

def test_one_fixed_offset_is_measured_from_the_archive_and_the_conventions(tmp_path):
    archive, announcements = write(tmp_path)
    document = calendar_clock.measure(str(archive), str(announcements))
    assert document["schema"] == calendar_clock.SCHEMA
    periods = determined(document)
    assert periods, "no period was determined at all"
    assert {period["utc_offset"] for period in periods} == {"UTC-05:00"}
    assert {period["utc_offset_seconds"] for period in periods} == {-5 * 3600}
    assert document["counts"]["cluster_zones"] == ["America/New_York", "Australia/Sydney"]


def test_the_convention_is_read_off_the_announcements_and_names_the_zone(tmp_path):
    archive, announcements = write(tmp_path)
    document = calendar_clock.measure(str(archive), str(announcements))
    convention = document["conventions"]["USD|initial jobless claims"]
    assert convention["zone"] == "America/New_York"
    assert convention["local_time"] == "08:30"
    assert convention["share"] == 1.0
    assert convention["dst_spanning"] is True
    # the daylight-saving span is what makes a fixed local time distinguishable from a fixed UTC time
    assert convention["zones_tried"]["UTC"]["share"] < 1.0


def test_a_clock_that_changes_gives_two_periods_at_the_date_it_changed(tmp_path):
    change = datetime(2018, 3, 1).date()

    def offsets(day):
        return -5 * 3600 if day < change else -4 * 3600

    archive, announcements = write(tmp_path, offsets=offsets, name="two")
    document = calendar_clock.measure(str(archive), str(announcements))
    periods = determined(document)
    assert {period["utc_offset"] for period in periods} == {"UTC-05:00", "UTC-04:00"}
    before = [period for period in periods if period["utc_offset"] == "UTC-05:00"]
    after = [period for period in periods if period["utc_offset"] == "UTC-04:00"]
    assert max(period["end_date"] for period in before) < "2018-03-01"
    assert min(period["start_date"] for period in after) >= "2018-03-01"
    # the month the change happened in is not silently assigned to either side
    straddling = [period for period in document["periods"]
                  if period["start_date"] <= "2018-03-01" <= period["end_date"]]
    assert all(period["status"] == calendar_clock.STATUS_DETERMINED for period in straddling) or \
        any(period["status"] == calendar_clock.STATUS_UNDETERMINED for period in straddling)


def test_an_archive_on_a_local_clock_that_observes_daylight_saving_is_measured_as_two_alternating_offsets(tmp_path):
    """A wall clock that IS the economy's local time shows up as an offset that changes with the season -- which is a
    different fact from a fixed offset, and the document must be able to say it."""
    def offsets(day):
        moment = datetime(day.year, day.month, day.day, 12, tzinfo=NEW_YORK)
        return int(moment.utcoffset().total_seconds())

    archive, announcements = write(tmp_path, offsets=offsets, name="local")
    document = calendar_clock.measure(str(archive), str(announcements))
    offsets_found = {period["utc_offset"] for period in determined(document)}
    assert offsets_found == {"UTC-05:00", "UTC-04:00"}
    assert len(determined(document)) > 2, "a seasonal clock must produce more than two periods"


# ---------------------------------------------------------------------------- what is refused rather than guessed

def test_a_series_whose_convention_changed_is_excluded_from_the_estimate_by_name(tmp_path):
    archive, announcements = write(tmp_path, shifted=("Retail Sales",), name="shift")
    document = calendar_clock.measure(str(archive), str(announcements))
    rejected = {entry["series"]: entry for entry in document["cluster"]["rejected"]}
    assert "Australia | Retail Sales" in rejected
    assert rejected["Australia | Retail Sales"]["reason"] == "CONVENTION_DISAGREES_WITH_THE_CLUSTER"
    assert "United States | Initial Jobless Claims" in document["cluster"]["members"]
    # and with the biased series gone the cluster no longer spans two zones, so nothing is concluded at all
    assert document["counts"]["series_excluded_from_the_estimate"] == 1


def test_a_cluster_of_one_zone_concludes_nothing(tmp_path):
    """With every series of one economy biased, what is left agrees only with itself -- and one economy's agreement
    with itself is not evidence about a clock, so the job refuses instead of reporting the offset it could see."""
    archive, announcements = write(tmp_path, shifted=("Retail Sales", "Building Approvals"), name="onezone")
    with pytest.raises(calendar_clock.ClockRefusal) as refusal:
        calendar_clock.measure(str(archive), str(announcements))
    assert refusal.value.code == "NO_MULTI_ZONE_CLUSTER"
    assert "America/New_York" in refusal.value.why or "Australia/Sydney" in refusal.value.why


def test_a_month_whose_estimates_disagree_is_undetermined_and_names_why(tmp_path):
    """Half the rows of one month written an hour out: the month cannot say what the wall clock meant."""
    def offsets(day):
        return -5 * 3600 + (3600 if day.year == 2017 and day.month == 6 and day.day < 15 else 0)

    archive, announcements = write(tmp_path, offsets=offsets, name="mixed")
    document = calendar_clock.measure(str(archive), str(announcements))
    month = document["months"]["2017-06"]
    assert month["status"] == calendar_clock.STATUS_UNDETERMINED
    assert month["reason"] in calendar_clock.UNDETERMINED_REASONS
    undetermined = [period for period in document["periods"]
                    if period["status"] == calendar_clock.STATUS_UNDETERMINED
                    and period["start_date"] <= "2017-06-15" <= period["end_date"]]
    assert undetermined, "the mixed month did not become an UNDETERMINED period"
    assert undetermined[0]["utc_offset"] is None


def test_an_undetermined_period_never_borrows_a_neighbours_offset(tmp_path):
    def offsets(day):
        return -5 * 3600 + (3600 if day.year == 2017 and day.month == 6 and day.day < 15 else 0)

    archive, announcements = write(tmp_path, offsets=offsets, name="borrow")
    document = calendar_clock.measure(str(archive), str(announcements))
    for period in document["periods"]:
        if period["status"] == calendar_clock.STATUS_UNDETERMINED:
            assert period["utc_offset"] is None and period["utc_offset_seconds"] is None
            assert "CLOCK_PERIOD_UNDETERMINED" in period["reading"]


def test_an_archive_no_convention_matches_is_refused_by_name(tmp_path):
    archive, announcements = write(tmp_path, name="none")
    empty = tmp_path / "empty_announcements.csv"
    with empty.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ANNOUNCEMENT_HEADER)
        writer.writerow(["XYZ", "nothing_at_all", "2025-01-01", "1.0", "2025-01-02 12:00:00+00:00"])
    with pytest.raises(calendar_clock.ClockRefusal) as refusal:
        calendar_clock.measure(str(archive), str(empty))
    assert refusal.value.code == "NO_CONVENTION_MATCHED_THE_ARCHIVE"


def test_a_convention_that_never_crosses_a_daylight_saving_boundary_is_not_used(tmp_path):
    """Six winter weeks cannot tell a fixed New York time from a fixed UTC time, and the job says so by name rather
    than picking the zone that happened to score first."""
    archive, announcements = write(tmp_path, announce_start=datetime(2025, 1, 6), announce_weeks=6, name="nodst")
    _meta, rows = calendar_clock.read_announcements(str(announcements))
    found = calendar_clock.conventions(rows, min_announcements=4)
    statuses = {entry["status"] for entry in found.values()}
    assert statuses == {"NOT_DST_DISCRIMINATING"}
    # and with nothing usable left, measuring the archive against them is refused rather than attempted
    with pytest.raises(calendar_clock.ClockRefusal) as refusal:
        calendar_clock.measure(str(archive), str(announcements), min_announcements=4)
    assert refusal.value.code == "NO_CONVENTION_MATCHED_THE_ARCHIVE"


# ------------------------------------------------------------------------------------------ reading it back

def test_the_document_is_loaded_with_its_digest_and_a_date_finds_its_period(tmp_path):
    archive, announcements = write(tmp_path, name="load")
    document = calendar_clock.measure(str(archive), str(announcements))
    path = tmp_path / "clock.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = calendar_clock.load(str(path))
    assert loaded["sha256"] and len(loaded["sha256"]) == 64
    period = calendar_clock.period_of(loaded, datetime(2017, 6, 15).date())
    assert period is not None and period["status"] == calendar_clock.STATUS_DETERMINED
    assert calendar_clock.period_of(loaded, datetime(1990, 1, 1).date()) is None


def test_a_document_of_another_schema_is_refused_by_name(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "something.else.v1", "periods": []}), encoding="utf-8")
    with pytest.raises(calendar_clock.ClockRefusal) as refusal:
        calendar_clock.load(str(path))
    assert refusal.value.code == "NOT_A_CLOCK_DOCUMENT"


def test_the_cli_writes_the_clock(tmp_path):
    archive, announcements = write(tmp_path, name="cli")
    out = tmp_path / "clock.json"
    assert calendar_clock.main(["--archive", str(archive), "--announcements", str(announcements),
                                "--out", str(out)]) == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["schema"] == calendar_clock.SCHEMA
    assert document["counts"]["periods_determined"] >= 1
