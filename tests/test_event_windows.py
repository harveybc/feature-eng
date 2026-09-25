"""The event window builder, checked against a price path whose response to each release was planted by hand.

A synthetic market is the only place where the right answer is known before the job runs. Here two event types are
planted into an otherwise flat log price: event A adds `b * s` to the log price linearly over the thirty minutes that
follow it and leaves it there, so `log_return[t_k, t_k + h]` is `b * s * h / 30` up to thirty minutes and `b * s`
after; event B adds a one-minute zigzag of amplitude `a` for thirty minutes and returns to where it started, so it
moves `realized_vol` and not the level. Whatever the builder writes into its rows must be those numbers.

The rest of the file is about what the builder must REFUSE, because every one of those refusals is a way the table
would otherwise carry a number nobody could have had: a consensus that is not there, a scale computed from the
future, a path with a hole in it, a release whose source never said when it published.

Nothing here fits a model, nothing here needs a GPU, and no test reads a file outside its own tmp_path.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from feature_eng_m5phet import events

START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

#: the planted level response of event A, per unit of RAW surprise, reached linearly over RAMP_MINUTES
B_LEVEL = 0.002
RAMP_MINUTES = 30

#: the planted zigzag of event B: amplitude `BURST * (3 + s)`, so it is positive for every surprise here and rises
#: with the surprise -- which is what step 2 must be able to see
BURST = 0.0004
BURST_MINUTES = 30

BASE_LOG_PRICE = float(np.log(1.1))
FIRST_EVENT_MINUTE = 600
SPACING_MINUTES = 480                       # 8 h, so no two planted responses overlap at any declared horizon
N_EVENTS = 40
TAIL_MINUTES = 300


def surprises(n=N_EVENTS):
    """A fixed, machine-independent set of raw surprises. No test here depends on their particular values."""
    return np.round(np.random.default_rng(20260925).uniform(-2.0, 2.0, n), 4)


def plan(n=N_EVENTS):
    """(event type, minute, raw surprise) for every planted release, alternating the two types."""
    return [("A" if i % 2 == 0 else "B", FIRST_EVENT_MINUTE + i * SPACING_MINUTES, float(s))
            for i, s in enumerate(surprises(n))]


def planted_log_price(minutes, releases):
    """The flat base plus every planted response, evaluated on a one-minute grid."""
    path = np.full(minutes, BASE_LOG_PRICE, dtype=np.float64)
    for kind, minute, s in releases:
        if kind == "A":
            for offset in range(1, minutes - minute):
                path[minute + offset] += B_LEVEL * s * min(1.0, offset / RAMP_MINUTES)
        else:
            amplitude = BURST * (3.0 + s)
            for offset in range(1, BURST_MINUTES + 1):
                if minute + offset < minutes and offset % 2 == 1:
                    path[minute + offset] += amplitude
    return path


def write_bars(path, releases, *, minutes=None, drop=(), step_minutes=1):
    minutes = minutes or FIRST_EVENT_MINUTE + (N_EVENTS - 1) * SPACING_MINUTES + TAIL_MINUTES
    log_price = planted_log_price(minutes, releases)
    dropped = set(drop)
    lines = ["timestamp,close"]
    for i in range(0, minutes, step_minutes):
        if i in dropped:
            continue
        lines.append(f"{(START + timedelta(minutes=i)).isoformat()},{float(np.exp(log_price[i]))!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


CALENDAR_HEADER = ("event_type,event_time,published_at,consensus_published_at,actual,consensus,previous,"
                   "historical_availability")


def calendar_line(kind, minute, s, *, consensus=100.0, published=True, consensus_published=True):
    moment = START + timedelta(minutes=minute)
    return ",".join([
        kind,
        moment.isoformat(),
        moment.isoformat() if published else "",
        (moment - timedelta(days=1)).isoformat() if consensus_published else "",
        repr(consensus + s) if consensus is not None else repr(s),
        repr(consensus) if consensus is not None else "",
        repr(consensus) if consensus is not None else "",
        "KNOWN",
    ])


def write_calendar(path, releases, extra=()):
    lines = [CALENDAR_HEADER] + [calendar_line(kind, minute, s) for kind, minute, s in releases] + list(extra)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build(tmp_path, *, releases=None, extra=(), drop=(), name="", **kwargs):
    releases = plan() if releases is None else releases
    bars = write_bars(tmp_path / f"bars{name}.csv", releases, drop=drop)
    calendar = write_calendar(tmp_path / f"calendar{name}.csv", releases, extra=extra)
    return events.build(str(bars), str(calendar), events.CalendarMapping(), **kwargs)


def rows_of(document, event_type=None, horizon=None):
    return [row for row in document["rows"]
            if (event_type is None or row["event_type"] == event_type)
            and (horizon is None or row["horizon_minutes"] == horizon)]


# ------------------------------------------------------------------------------------ the planted response is read

def test_the_planted_level_response_is_recovered_at_every_declared_horizon(tmp_path):
    document = build(tmp_path)
    assert document["schema"] == events.SCHEMA
    rows = rows_of(document, "A")
    assert rows, "no row survived for the level-response event type"
    for row in rows:
        h = row["horizon_minutes"]
        expected = B_LEVEL * row["surprise_raw"] * min(1.0, h / RAMP_MINUTES)
        assert row["log_return"] == pytest.approx(expected, abs=1e-12), (
            f"horizon {h} of {row['event_key']} reads {row['log_return']} and the planted response is {expected}")


def test_the_planted_volatility_burst_is_recovered_and_leaves_the_level_where_it_was(tmp_path):
    document = build(tmp_path)
    rows = rows_of(document, "B", horizon=BURST_MINUTES)
    assert rows, "no row survived for the volatility event type"
    for row in rows:
        amplitude = BURST * (3.0 + row["surprise_raw"])
        assert row["realized_vol"] == pytest.approx(BURST_MINUTES * amplitude ** 2, rel=1e-9)
        assert row["log_return"] == pytest.approx(0.0, abs=1e-12)


def test_the_standardized_surprise_is_the_raw_one_over_the_scale_the_row_reports(tmp_path):
    document = build(tmp_path)
    for row in document["rows"]:
        assert row["surprise"] == pytest.approx(row["surprise_raw"] / row["surprise_scale"], rel=1e-12)
        assert row["surprise_boundary"] == "release"
        assert row["consensus_published_at"] is not None


def test_the_pre_event_volatility_covers_the_declared_span_and_is_zero_on_a_flat_base(tmp_path):
    document = build(tmp_path, pre_event_minutes=60)
    for row in rows_of(document, "A"):
        assert row["pre_event_minutes"] == 60
        assert row["pre_event_status"] == "OK"
        # the base is flat and the previous response ended long before, so nothing moved in that hour
        assert row["pre_event_realized_vol"] == pytest.approx(0.0, abs=1e-20)
        assert (datetime.fromisoformat(row["anchor_time"])
                - datetime.fromisoformat(row["pre_event_from"])) == timedelta(minutes=60)


def test_hour_of_day_and_day_of_week_are_the_release_instant_in_utc(tmp_path):
    document = build(tmp_path)
    for row in document["rows"]:
        moment = datetime.fromisoformat(row["published_at"]).astimezone(timezone.utc)
        assert row["hour_of_day"] == moment.hour
        assert row["day_of_week"] == moment.weekday()


def test_the_other_releases_in_the_window_carry_signed_offsets_in_minutes(tmp_path):
    document = build(tmp_path, window_hours=24.0)
    row = rows_of(document, "A")[5]
    neighbours = row["other_releases_in_window"]
    assert neighbours, "a release with others eight hours away lists none of them"
    assert all(abs(n["offset_minutes"]) <= 24 * 60 for n in neighbours)
    assert any(n["offset_minutes"] < 0 for n in neighbours) and any(n["offset_minutes"] > 0 for n in neighbours)
    assert all(n["event_key"] != row["event_key"] for n in neighbours)
    assert {n["offset_minutes"] for n in neighbours} <= {float(k * SPACING_MINUTES) for k in range(-3, 4)}


def test_the_window_is_declared_and_narrowing_it_drops_the_far_neighbours(tmp_path):
    wide = build(tmp_path, window_hours=24.0, name="wide")
    narrow = build(tmp_path, window_hours=9.0, name="narrow")
    assert wide["parameters"]["window_hours"] == 24.0 and narrow["parameters"]["window_hours"] == 9.0
    assert len(rows_of(wide, "A")[5]["other_releases_in_window"]) > \
        len(rows_of(narrow, "A")[5]["other_releases_in_window"])


# ------------------------------------------------------------------------------------------ what must be refused

def test_a_release_without_a_consensus_is_excluded_by_name(tmp_path):
    releases = plan()
    orphan = calendar_line("A", FIRST_EVENT_MINUTE + N_EVENTS * SPACING_MINUTES, 1.5, consensus=None)
    document = build(tmp_path, releases=releases, extra=(orphan,))
    assert document["excluded"]["counts"]["NO_CONSENSUS"] == 1
    named = [entry for entry in document["excluded"]["releases"] if entry["code"] == "NO_CONSENSUS"]
    assert len(named) == 1 and "NO_CONSENSUS_BEFORE_THE_BOUNDARY" in named[0]["why"]
    assert all(row["consensus"] is not None for row in document["rows"])


def test_a_release_whose_source_never_said_when_it_published_is_excluded_by_name(tmp_path):
    releases = plan()
    blind = calendar_line("A", FIRST_EVENT_MINUTE + N_EVENTS * SPACING_MINUTES, 1.5, published=False)
    document = build(tmp_path, releases=releases, extra=(blind,))
    assert document["excluded"]["counts"]["MISSING_PUBLICATION_CLOCK"] == 1
    named = [entry for entry in document["excluded"]["releases"] if entry["code"] == "MISSING_PUBLICATION_CLOCK"]
    assert len(named) == 1 and "MISSING_PUBLICATION_CLOCK" in named[0]["why"]


def test_the_first_releases_of_each_type_are_refused_for_want_of_history(tmp_path):
    document = build(tmp_path, min_prior_releases=8)
    assert document["parameters"]["min_prior_releases"] == 8
    assert document["excluded"]["counts"]["INSUFFICIENT_HISTORY"] == 16          # eight of each of the two types
    assert all(row["surprise_scale_n"] >= 8 for row in document["rows"])
    why = next(e["why"] for e in document["excluded"]["releases"] if e["code"] == "INSUFFICIENT_HISTORY")
    assert "published before this one" in why


def test_the_scale_reads_only_releases_published_before_the_row(tmp_path):
    """A huge surprise inserted AFTER a row must leave that row untouched, or the scale came from the future."""
    releases = plan()
    plain = build(tmp_path, releases=releases, name="plain")
    later = calendar_line("A", FIRST_EVENT_MINUTE + (N_EVENTS - 1) * SPACING_MINUTES + 55, 900.0)
    contaminated = build(tmp_path, releases=releases, extra=(later,), name="later")
    before = {(r["event_key"], r["horizon_minutes"]): (r["surprise"], r["surprise_scale"]) for r in plain["rows"]}
    after = {(r["event_key"], r["horizon_minutes"]): (r["surprise"], r["surprise_scale"])
             for r in contaminated["rows"] if (r["event_key"], r["horizon_minutes"]) in before}
    assert after == before, "a release published later changed the scale of a row published earlier"
    assert len(contaminated["rows"]) > len(plain["rows"]), "the later release did not become a row at all"


def test_a_horizon_whose_bars_have_a_hole_is_refused_by_name_and_the_shorter_ones_survive(tmp_path):
    releases = plan()
    victim = next(minute for kind, minute, _ in releases[20:] if kind == "A")
    document = build(tmp_path, releases=releases, drop=(victim + 45,))
    excluded = [entry for entry in document["excluded"]["event_horizons"]
                if entry["code"] == "BARS_MISSING_AT_HORIZON"]
    assert {entry["horizon_minutes"] for entry in excluded} == {60, 240}
    assert document["excluded"]["counts"]["BARS_MISSING_AT_HORIZON"] == 2
    assert "interpolating" in excluded[0]["why"]
    survivors = {row["horizon_minutes"] for row in document["rows"]
                 if row["event_key"] == excluded[0]["event_key"]}
    assert survivors == {5, 15, 30}


def test_a_release_that_falls_into_a_gap_in_the_bars_loses_every_horizon_by_name(tmp_path):
    releases = plan()
    victim = next(minute for kind, minute, _ in releases[20:] if kind == "A")
    document = build(tmp_path, releases=releases, drop=tuple(range(victim - 3, victim + 1)))
    excluded = [entry for entry in document["excluded"]["event_horizons"] if entry["horizon_minutes"] in (5, 240)]
    assert excluded, "a release inside a hole in the series still produced rows"
    assert all(entry["code"] == "BARS_MISSING_AT_HORIZON" for entry in excluded)


def test_a_horizon_that_is_not_a_whole_number_of_bars_is_refused_before_anything_is_built(tmp_path):
    """The bars this repository actually holds for EUR/USD sit on a five-minute grid, and a path cannot end between
    two bars. The refusal is by name, before any row is built, rather than a horizon quietly rounded to a bar."""
    releases = plan(20)
    bars = write_bars(tmp_path / "bars5.csv", releases, step_minutes=5)
    calendar = write_calendar(tmp_path / "calendar.csv", releases)
    with pytest.raises(events.EventsRefusal) as refusal:
        events.build(str(bars), str(calendar), events.CalendarMapping(), horizons_minutes=(7,))
    assert refusal.value.code == "HORIZON_OFF_THE_BAR_GRID"
    coarse = events.build(str(bars), str(calendar), events.CalendarMapping(), horizons_minutes=(30, 240))
    assert coarse["parameters"]["realized_vol_step_seconds"] == 300
    assert "300-second bars" in coarse["parameters"]["realized_vol"]


def test_a_horizon_that_is_not_a_positive_number_of_minutes_is_refused_by_name(tmp_path):
    with pytest.raises(events.EventsRefusal) as refusal:
        build(tmp_path, horizons_minutes=(0, 30))
    assert refusal.value.code == "BAD_HORIZONS"


def test_a_naive_calendar_timestamp_with_no_declared_zone_is_refused_by_name(tmp_path):
    releases = plan(12)
    bars = write_bars(tmp_path / "bars.csv", releases)
    lines = [CALENDAR_HEADER]
    for kind, minute, s in releases:
        lines.append(calendar_line(kind, minute, s).replace("+00:00", ""))
    (tmp_path / "calendar.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(events.EventsRefusal) as refusal:
        events.build(str(bars), str(tmp_path / "calendar.csv"), events.CalendarMapping())
    assert refusal.value.code == "AMBIGUOUS_LOCAL_TIME"


def test_a_calendar_that_never_says_whether_its_rows_were_available_is_refused_by_name(tmp_path):
    releases = plan(12)
    bars = write_bars(tmp_path / "bars.csv", releases)
    header = CALENDAR_HEADER.replace(",historical_availability", "")
    lines = [header] + [",".join(calendar_line(k, m, s).split(",")[:-1]) for k, m, s in releases]
    (tmp_path / "calendar.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(events.EventsRefusal) as refusal:
        events.build(str(bars), str(tmp_path / "calendar.csv"), events.CalendarMapping())
    assert refusal.value.code == "AVAILABILITY_NOT_DECLARED"


def test_bars_that_are_not_in_time_order_are_refused_rather_than_sorted(tmp_path):
    releases = plan(12)
    bars = write_bars(tmp_path / "bars.csv", releases)
    lines = bars.read_text(encoding="utf-8").splitlines()
    lines[10], lines[20] = lines[20], lines[10]
    bars.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(events.EventsRefusal) as refusal:
        events.build(str(bars), str(write_calendar(tmp_path / "calendar.csv", releases)),
                     events.CalendarMapping())
    assert refusal.value.code == "BARS_NOT_INCREASING"


# ----------------------------------------------------------------------------------- the document and the counts

def test_every_declared_exclusion_code_is_counted_even_when_it_did_not_fire(tmp_path):
    document = build(tmp_path)
    assert set(document["excluded"]["counts"]) == set(events.EXCLUSION_CODES)
    for code in ("MISSING_PUBLICATION_CLOCK", "NO_CONSENSUS", "INSUFFICIENT_HISTORY", "BARS_MISSING_AT_HORIZON"):
        assert code in document["excluded"]["counts"]


def test_the_counts_add_up_to_the_releases_that_were_read(tmp_path):
    """On this fixture every surviving release keeps at least one horizon, so the release-level arithmetic closes
    exactly. On a real series some releases lose EVERY horizon to a gap, and then the difference is those releases:
    the identity below is an arithmetic check of these counts, not a claim that it holds on any archive."""
    document = build(tmp_path)
    counts = document["excluded"]["counts"]
    dropped = sum(counts[code] for code in events.EXCLUSION_CODES if code != "BARS_MISSING_AT_HORIZON")
    assert document["counts"]["releases_with_a_row"] + dropped == document["counts"]["releases_read"]
    assert document["counts"]["rows"] == len(document["rows"])
    assert set(document["counts"]["by_event_type"]) == {"A", "B"}


def test_the_document_says_what_it_did_not_fit_and_what_the_rows_are_not(tmp_path):
    document = build(tmp_path)
    assert document["fitted"].startswith("NOTHING")
    assert "not an effect" in document["reading"]
    assert "not pre-release information" in document["reading"]
    assert document["parameters"]["realized_vol_step_seconds"] == 60
    assert document["bars"]["step_seconds"] == 60 and "times" not in document["bars"]
    assert document["calendar"]["units_declared"] is False
    assert "INCOMPARABLE_SERIES" in document["calendar"]["units_reading"]


def test_the_same_inputs_build_the_same_bytes(tmp_path):
    releases = plan(20)
    bars = write_bars(tmp_path / "bars.csv", releases)
    calendar = write_calendar(tmp_path / "calendar.csv", releases)
    first = json.dumps(events.build(str(bars), str(calendar), events.CalendarMapping()), sort_keys=False)
    second = json.dumps(events.build(str(bars), str(calendar), events.CalendarMapping()), sort_keys=False)
    assert first == second


def test_the_cli_writes_the_rows_and_refuses_with_a_code(tmp_path, capsys):
    releases = plan(20)
    bars = write_bars(tmp_path / "bars.csv", releases)
    calendar = write_calendar(tmp_path / "calendar.csv", releases)
    out = tmp_path / "event_rows.json"
    assert events.main(["--bars", str(bars), "--calendar", str(calendar), "--out", str(out)]) == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["schema"] == events.SCHEMA and document["rows"]
    assert events.main(["--bars", str(bars), "--calendar", str(tmp_path / "absent.csv"), "--out", str(out)]) == 2
    assert "REFUSED NO_SUCH_FILE" in capsys.readouterr().err


# ------------------------------------------------------- the declared assumption, and what it never stops being

SCHEDULED_HEADER = "event_type,event_time,actual,consensus,previous,historical_availability"


def write_scheduled_calendar(path, releases, *, consensus=100.0):
    """An archive shaped like the one this repository actually holds: a scheduled instant, and no clock of any kind
    saying when anybody published anything."""
    lines = [SCHEDULED_HEADER]
    for kind, minute, s in releases:
        moment = START + timedelta(minutes=minute)
        lines.append(",".join([kind, moment.isoformat(), repr(consensus + s), repr(consensus), repr(consensus),
                               "KNOWN"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def scheduled_inputs(tmp_path, releases=None, name=""):
    releases = plan() if releases is None else releases
    return (write_bars(tmp_path / f"bars{name}.csv", releases),
            write_scheduled_calendar(tmp_path / f"scheduled{name}.csv", releases))


def test_an_archive_with_no_clock_is_still_refused_by_name_without_the_declared_assumption(tmp_path):
    bars, calendar = scheduled_inputs(tmp_path)
    document = events.build(str(bars), str(calendar), events.CalendarMapping())
    assert document["rows"] == []
    assert document["excluded"]["counts"]["MISSING_PUBLICATION_CLOCK"] == N_EVENTS
    assert document["provenance"] == "DEVELOPMENT_OBSERVED_CLOCK"
    assert document["publication_clock"]["mode"] == "OBSERVED_PUBLICATION_CLOCK"
    assert document["publication_clock"]["tolerance_seconds"] is None


def test_the_declared_assumption_lets_the_same_archive_build_rows_and_stamps_every_one_of_them(tmp_path):
    bars, calendar = scheduled_inputs(tmp_path)
    document = events.build(str(bars), str(calendar), events.CalendarMapping(),
                            publication_clock="scheduled", assumed_tolerance_seconds=60)
    assert document["rows"], "the declared assumption produced no rows at all"
    assert document["excluded"]["counts"]["MISSING_PUBLICATION_CLOCK"] == 0
    assert document["provenance"] == "DEVELOPMENT_ASSUMED_CLOCK"
    assert document["publication_clock"] == {
        "mode": "ASSUMED_SCHEDULED_PUBLICATION",
        "tolerance_seconds": 60,
        "declared_by": "operator",
        "identification_caveat": ("the surprise's publication instant is assumed equal to the scheduled instant; no "
                                  "receipt or publication timestamp was observed; results are DEVELOPMENT and not "
                                  "identified until a publication clock exists"),
        "consensus": ("the consensus is taken as published tolerance_seconds before the scheduled instant, under the "
                      "same declared assumption; no consensus publication timestamp was observed either"),
    }
    assert "not identified until a publication clock exists" in document["reading"]
    assert document["reading"].startswith("PROVENANCE DEVELOPMENT_ASSUMED_CLOCK")
    for row in document["rows"]:
        assert row["provenance"] == "DEVELOPMENT_ASSUMED_CLOCK"
        assert row["publication_clock_mode"] == "ASSUMED_SCHEDULED_PUBLICATION"


def test_under_the_assumption_the_release_instant_is_the_scheduled_one_and_the_consensus_stands_before_it(tmp_path):
    bars, calendar = scheduled_inputs(tmp_path)
    document = events.build(str(bars), str(calendar), events.CalendarMapping(),
                            publication_clock="scheduled", assumed_tolerance_seconds=90)
    assert document["publication_clock"]["tolerance_seconds"] == 90
    for row in document["rows"]:
        assert row["published_at"] == row["event_time"]
        gap = datetime.fromisoformat(row["published_at"]) - datetime.fromisoformat(row["consensus_published_at"])
        assert gap == timedelta(seconds=90)


def test_the_assumption_moves_no_value_only_the_instants(tmp_path):
    """The planted response must come out of the assumed run exactly as it does out of a run with real clocks."""
    releases = plan()
    bars = write_bars(tmp_path / "bars.csv", releases)
    assumed = events.build(str(bars), str(write_scheduled_calendar(tmp_path / "sched.csv", releases)),
                           events.CalendarMapping(), publication_clock="scheduled")
    for row in [r for r in assumed["rows"] if r["event_type"] == "A"]:
        h = row["horizon_minutes"]
        assert row["log_return"] == pytest.approx(B_LEVEL * row["surprise_raw"] * min(1.0, h / RAMP_MINUTES),
                                                  abs=1e-12)


def test_the_assumption_refuses_to_overwrite_a_clock_the_dataset_already_declares(tmp_path):
    releases = plan(12)
    bars = write_bars(tmp_path / "bars.csv", releases)
    calendar = write_calendar(tmp_path / "calendar.csv", releases)      # this one DOES carry published_at
    with pytest.raises(events.EventsRefusal) as refusal:
        events.build(str(bars), str(calendar), events.CalendarMapping(), publication_clock="scheduled")
    assert refusal.value.code == "PUBLICATION_CLOCK_CONFLICT"


def test_a_tolerance_that_is_not_a_positive_span_is_refused_by_name(tmp_path):
    bars, calendar = scheduled_inputs(tmp_path, plan(12))
    with pytest.raises(events.EventsRefusal) as refusal:
        events.build(str(bars), str(calendar), events.CalendarMapping(),
                     publication_clock="scheduled", assumed_tolerance_seconds=0)
    assert refusal.value.code == "BAD_ASSUMED_TOLERANCE"


def test_the_neighbour_listing_can_be_capped_while_its_count_stays_exact(tmp_path):
    bars, calendar = scheduled_inputs(tmp_path)
    full = events.build(str(bars), str(calendar), events.CalendarMapping(), publication_clock="scheduled")
    capped = events.build(str(bars), str(calendar), events.CalendarMapping(), publication_clock="scheduled",
                          max_neighbours_listed=2)
    assert capped["parameters"]["max_neighbours_listed"] == 2
    by_key = {(r["event_key"], r["horizon_minutes"]): r for r in full["rows"]}
    crowded = 0
    for row in capped["rows"]:
        reference = by_key[(row["event_key"], row["horizon_minutes"])]
        assert row["other_releases_in_window_count"] == reference["other_releases_in_window_count"]
        assert len(row["other_releases_in_window"]) == row["other_releases_in_window_listed"] <= 2
        if reference["other_releases_in_window_count"] > 2:
            crowded += 1
            assert row["other_releases_in_window_listed"] == 2
    assert crowded, "no row in this fixture had more neighbours than the cap, so the cap was never exercised"


def test_the_cli_carries_the_declared_assumption_into_the_file(tmp_path):
    bars, calendar = scheduled_inputs(tmp_path, plan(20))
    out = tmp_path / "rows.json"
    assert events.main(["--bars", str(bars), "--calendar", str(calendar), "--out", str(out),
                        "--publication-clock", "scheduled",
                        "--assume-publication-tolerance-seconds", "60",
                        "--max-neighbours-listed", "4"]) == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["provenance"] == "DEVELOPMENT_ASSUMED_CLOCK"
    assert document["publication_clock"]["declared_by"] == "operator"
    assert document["parameters"]["max_neighbours_listed"] == 4
