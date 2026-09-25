"""Measure the UTC offset an archive's naive wall clock actually carries, period by period, from the data.

An economic-calendar archive with no zone is not a set of instants. Somebody must declare what its wall clock means,
and the declaration that costs nothing -- "read it as UTC" -- is the one that silently moves every release by hours
and leaves the arithmetic downstream looking perfectly healthy. In the archive this repository holds, that mistake is
worth four to five hours: every response measured from it was read from the wrong place in the price path.

This module refuses to let that be a matter of declaration. It MEASURES the offset, and the measurement rests on one
idea: a macro release has a **publication convention** -- a fixed local time of day in the economy's own zone -- and
that convention is itself observable in an archive whose instants somebody did record. Two archives, two jobs:

1. **The convention** comes from the announcement archive (observed `announcement_datetime_utc`). For each
   (currency, release) the job reads every announcement in each declared candidate zone for that currency and asks in
   which zone the local time of day is most nearly constant. A release published at a fixed wall clock in
   `America/New_York` has a UTC instant that moves by an hour across a daylight-saving boundary and a New York time
   of day that does not; a release whose UTC instant never moves is on UTC. The zone with the highest agreement wins
   and the agreement is reported. A convention whose announcements do not straddle a daylight-saving boundary cannot
   tell a fixed local time from a fixed UTC time, so it is marked `NOT_DST_DISCRIMINATING` and is not used.

2. **The offset** is then arithmetic on the archive. For a row of a release whose convention is known, the instant it
   was published is the convention's local time on that date, converted to UTC by the zone's own rules. The archive's
   wall clock minus that instant IS the offset the archive carries, to the nearest quarter hour. Thousands of rows
   give thousands of estimates; they are counted per calendar month, and the modal offset of a month is that month's
   offset when enough rows agree.

**Nothing here is interpolated.** A month with too few estimates, or with estimates that do not agree, or in which
two different economies' releases imply two different offsets, is `UNDETERMINED`. Undetermined months do not borrow
their neighbours' offset, and the rows inside them are excluded by name (`CLOCK_PERIOD_UNDETERMINED`) rather than
localized by a guess -- because a guessed hour is exactly the defect this module exists to remove, wearing a more
confident label.

The result is `m5phet.calendar_clock.v1`: a list of periods, each with its UTC offset, the counts that established
it, and a confidence; plus the per-year evidence (the wall clock's time of day per release, per year) somebody can
read to see that the periods are not an artefact of the code. `events.py --calendar-clock` consumes it.

Deterministic and CPU only: the standard library, and this package's own declared normalisation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date as _date, datetime, timedelta, timezone as _timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .calendar_join import (COUNTRY_CURRENCY, JoinRefusal, SYNONYMS, normalise, read_announcements, read_archive)
from .design import MISSING_TOKENS, _file_digest, _time_parser

SCHEMA = "m5phet.calendar_clock.v1"

#: the zones a release of each currency may be published on a fixed clock in. This is the SEARCH SPACE of the
#: convention, declared here; which of them a release actually uses is measured, never assumed. UTC is always a
#: candidate, because a release published at a fixed UTC time exists and must be distinguishable from the rest.
ZONE_CANDIDATES = {
    "USD": ("America/New_York", "America/Chicago"),
    "EUR": ("Europe/Berlin", "Europe/Paris", "Europe/Brussels"),
    "GBP": ("Europe/London",),
    "JPY": ("Asia/Tokyo",),
    "CAD": ("America/Toronto",),
    "AUD": ("Australia/Sydney",),
    "NZD": ("Pacific/Auckland",),
    "CHF": ("Europe/Zurich",),
    "CNY": ("Asia/Shanghai",),
    "SEK": ("Europe/Stockholm",),
    "DKK": ("Europe/Copenhagen",),
    "PLN": ("Europe/Warsaw",),
    "BRL": ("America/Sao_Paulo",),
    "SGD": ("Asia/Singapore",),
}

UNIVERSAL_ZONES = ("UTC",)

#: the grid every offset is rounded to. Quarter hours, because real zone offsets come in quarter hours (and because
#: a finer grid would turn a one-minute recording jitter into a distinct offset).
OFFSET_GRID_SECONDS = 900

#: an offset further from UTC than this is not a zone, it is a mismatched release
MAX_ABS_OFFSET_SECONDS = 14 * 3600

#: the fewest announcements of one release before its convention means anything
DEFAULT_MIN_ANNOUNCEMENTS = 8

#: the share of announcements that must land on the same local time of day before a zone is called the convention
DEFAULT_CONVENTION_AGREEMENT = 0.75

#: the fewest offset estimates a month needs before its modal offset is that month's offset
DEFAULT_MIN_MONTH_ROWS = 8

#: the share of a month's estimates that must agree on one offset
DEFAULT_MONTH_AGREEMENT = 0.90

#: every reason a month does not get an offset. They are the exact words the periods carry.
UNDETERMINED_REASONS = ("TOO_FEW_ESTIMATES", "ESTIMATES_DISAGREE", "ECONOMIES_DISAGREE", "NO_ESTIMATES")

STATUS_DETERMINED = "DETERMINED"
STATUS_UNDETERMINED = "UNDETERMINED"


class ClockRefusal(ValueError):
    """An input this job will not measure a clock from, carrying the code a caller matches on."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise ClockRefusal(code, why)


def _zone(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        _refuse("UNKNOWN_TIMEZONE", f"{name!r} is not a zone this machine knows ({exc})")


def _round(seconds):
    return int(round(seconds / OFFSET_GRID_SECONDS)) * OFFSET_GRID_SECONDS


def _offset_text(seconds):
    sign = "-" if seconds < 0 else "+"
    seconds = abs(int(seconds))
    return f"UTC{sign}{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


# ------------------------------------------------------------------------------------------- the convention

def conventions(announcements, *, min_announcements=DEFAULT_MIN_ANNOUNCEMENTS,
                agreement=DEFAULT_CONVENTION_AGREEMENT, zone_candidates=None):
    """Per (currency, release), the zone and local time of day the announcements agree on -- or why they do not.

    The comparison is between whole declared zones, not between offsets: what distinguishes `America/New_York` from
    UTC-5 is precisely what happens at a daylight-saving boundary, and a release whose announcements never cross one
    cannot be told apart from a fixed-UTC release. Those are reported and not used.
    """
    candidates = dict(ZONE_CANDIDATES if zone_candidates is None else zone_candidates)
    grouped = {}
    for announcement in announcements:
        key = (announcement["currency"], normalise(announcement["indicator"]))
        grouped.setdefault(key, []).append(announcement["published_at"])
    found = {}
    for (currency, release), instants in sorted(grouped.items()):
        entry = {"currency": currency, "release": release, "n": len(instants)}
        if len(instants) < min_announcements:
            entry.update({"status": "TOO_FEW_ANNOUNCEMENTS",
                          "why": f"{len(instants)} announcement(s) and {min_announcements} are declared as the "
                                 f"fewest a publication convention can be read from"})
            found[(currency, release)] = entry
            continue
        best = None
        tried = {}
        for name in tuple(candidates.get(currency, ())) + UNIVERSAL_ZONES:
            zone = _zone(name)
            times = Counter()
            offsets = set()
            for instant in instants:
                local = instant.astimezone(zone)
                times[local.strftime("%H:%M")] += 1
                offsets.add(local.utcoffset())
            time_of_day, count = times.most_common(1)[0]
            share = count / len(instants)
            tried[name] = {"local_time": time_of_day, "share": round(share, 6), "distinct_offsets": len(offsets)}
            if best is None or share > best["share"]:
                best = {"zone": name, "local_time": time_of_day, "share": share,
                        "dst_spanning": len(offsets) > 1}
        entry["zones_tried"] = tried
        if best is None or best["share"] < agreement:
            entry.update({"status": "NO_CONVENTION",
                          "why": f"no declared zone puts {int(agreement * 100)} % of these announcements on one "
                                 f"local time of day; the best was "
                                 f"{best['zone'] if best else None} at {round(best['share'], 3) if best else None}"})
        elif not best["dst_spanning"]:
            entry.update({"status": "NOT_DST_DISCRIMINATING", "zone": best["zone"],
                          "local_time": best["local_time"], "share": round(best["share"], 6),
                          "why": "these announcements all fall on one side of a daylight-saving boundary, so a fixed "
                                 "local time and a fixed UTC time are indistinguishable here and neither is chosen"})
        else:
            entry.update({"status": "OK", "zone": best["zone"], "local_time": best["local_time"],
                          "share": round(best["share"], 6), "dst_spanning": True})
        found[(currency, release)] = entry
    return found


# ------------------------------------------------------------------------------------------------ the offset

def _wall(row):
    text = " ".join(part for part in ((row.get("event_date") or "").strip(),
                                      (row.get("event_time") or "").strip()) if part)
    if not text or text in MISSING_TOKENS:
        return None
    try:
        _, parser = _time_parser(text)
        moment = parser(text)
    except ValueError:
        return None
    return moment if moment.tzinfo is None else None      # a row that already carries an offset needs no measuring


def _expected(convention, zone, day):
    """The instant the convention says a release of this day was published, or None when the local time is ambiguous.

    Three candidate days are tried, because a wall clock hours away from the zone can put the archive's date one day
    either side of the zone's; the nearest is taken and the distance IS the offset being measured.
    """
    hour, minute = (int(part) for part in convention["local_time"].split(":"))
    out = []
    for delta in (-1, 0, 1):
        local = datetime.combine(day + timedelta(days=delta), datetime.min.time(), tzinfo=zone)
        local = local.replace(hour=hour, minute=minute)
        if local.replace(fold=0).utcoffset() != local.replace(fold=1).utcoffset():
            continue                                      # inside a fold this wall clock names two instants
        out.append(local.astimezone(_timezone.utc).replace(tzinfo=None))
    return out


def estimates(archive_rows, found, *, country_currency=None, synonyms=None):
    """One offset estimate per archive row whose release has a usable convention, with what it was read from."""
    countries = dict(COUNTRY_CURRENCY if country_currency is None else country_currency)
    table = dict(SYNONYMS if synonyms is None else synonyms)
    vocabulary = {release for (_currency, release), entry in found.items() if entry.get("status") == "OK"}
    zones, out, series = {}, [], {}
    for row in archive_rows:
        country = (row.get("country") or "").strip()
        description = (row.get("description") or "").strip()
        currency = countries.get(normalise(country))
        if currency is None:
            continue
        name = normalise(description)
        if name not in vocabulary:
            name = table.get(name, name)
        convention = found.get((currency, name))
        if convention is None or convention.get("status") != "OK":
            continue
        wall = _wall(row)
        if wall is None:
            continue
        zone = zones.setdefault(convention["zone"], _zone(convention["zone"]))
        candidates = _expected(convention, zone, wall.date())
        if not candidates:
            continue
        deltas = [(wall - instant).total_seconds() for instant in candidates]
        delta = min(deltas, key=abs)
        if abs(delta) > MAX_ABS_OFFSET_SECONDS:
            continue
        offset = _round(delta)
        month = f"{wall.year:04d}-{wall.month:02d}"
        out.append({"month": month, "year": wall.year, "currency": currency, "offset_seconds": offset,
                    "series": f"{country} | {description}", "zone": convention["zone"]})
        entry = series.setdefault(f"{country} | {description}",
                                  {"currency": currency, "zone": convention["zone"],
                                   "convention_local_time": convention["local_time"], "by_year": {}})
        year = entry["by_year"].setdefault(str(wall.year), {"n": 0, "wall_time_of_day": Counter(),
                                                            "offsets": Counter()})
        year["n"] += 1
        year["wall_time_of_day"][wall.strftime("%H:%M")] += 1
        year["offsets"][_offset_text(offset)] += 1
    return out, series


#: how far two series' monthly offsets may disagree and still be called the same clock. One recording slip in a
#: hundred months is a slip; a systematic five-minute or one-hour difference is a different convention.
DEFAULT_SERIES_CONSISTENCY = 0.95

#: the fewest months two series must both have estimates in before their agreement means anything
DEFAULT_SHARED_MONTHS = 6

#: the fewest DISTINCT zones the agreeing cluster must span. Two economies whose daylight-saving calendars are not
#: the same (New York and Sydney move in opposite seasons) cannot agree on an offset by accident of one convention.
DEFAULT_MIN_CLUSTER_ZONES = 2


def _by_series_month(values, *, agreement):
    """Per (series, month) the series' own modal offset, when its own rows that month agree on one."""
    grouped = {}
    for value in values:
        grouped.setdefault(value["series"], {}).setdefault(value["month"], Counter())[value["offset_seconds"]] += 1
    out = {}
    for name, months in grouped.items():
        picked = {}
        for month, counts in months.items():
            offset, count = counts.most_common(1)[0]
            if count / sum(counts.values()) >= agreement:
                picked[month] = offset
        out[name] = picked
    return out


def cluster(values, *, series_agreement=DEFAULT_MONTH_AGREEMENT, consistency=DEFAULT_SERIES_CONSISTENCY,
            shared_months=DEFAULT_SHARED_MONTHS):
    """The largest set of series that agree with each other about what the archive's clock was, month by month.

    One archive has one clock, so two series that imply different offsets in the same month cannot both be measuring
    it. What separates them is not a vote: a release whose publication convention CHANGED between the archive's span
    and the announcements' span carries a constant bias -- an hour, a quarter of an hour -- that shows up as a
    systematic disagreement with every series whose convention did not change. Those are found here and reported with
    their own signature, so the contradiction is in the document rather than averaged away inside it.

    The cluster is grown greedily from the series with the most estimates, and a candidate joins only if it agrees
    with EVERY member over the months they share. Series are ordered by (estimates, name), so the result does not
    depend on dictionary order.
    """
    signatures = _by_series_month(values, agreement=series_agreement)
    counts = Counter(value["series"] for value in values)
    zones = {value["series"]: value["zone"] for value in values}

    def compare(a, b):
        left, right = signatures[a], signatures[b]
        shared = sorted(set(left) & set(right))
        if len(shared) < shared_months:
            return None, len(shared), None          # not comparable: no agreement is claimed either way
        same = sum(1 for month in shared if left[month] == right[month])
        return same / len(shared) >= consistency, len(shared), round(same / len(shared), 6)

    ordered = sorted(signatures, key=lambda name: (-counts[name], name))
    members, rejected = [], []
    for candidate in ordered:
        if not members:
            members.append(candidate)
            continue
        verdicts = [(member, *compare(candidate, member)) for member in members]
        bad = [entry for entry in verdicts if entry[1] is not True]
        if bad:
            member, ok, shared, share = bad[0]
            rejected.append({"series": candidate, "zone": zones.get(candidate), "estimates": counts[candidate],
                             "against": member, "shared_months": shared, "agreement": share,
                             "reason": ("CONVENTION_NOT_COMPARABLE" if ok is None
                                        else "CONVENTION_DISAGREES_WITH_THE_CLUSTER"),
                             "why": (f"this series and {member!r} share {shared} month(s), fewer than the "
                                     f"{shared_months} declared as the fewest an agreement can be read from"
                                     if ok is None else
                                     f"this series and {member!r} agree on the offset in {share:.0%} of the "
                                     f"{shared} month(s) they share, below the declared {consistency:.0%}; a "
                                     f"release whose publication convention changed between the two archives' "
                                     f"spans carries exactly this kind of constant bias, and it is excluded from "
                                     f"the estimate rather than averaged into it")})
        else:
            members.append(candidate)
    return {"members": sorted(members), "rejected": sorted(rejected, key=lambda entry: entry["series"]),
            "zones": sorted({zones[name] for name in members}),
            "estimates": sum(counts[name] for name in members),
            "rule": (f"grown greedily from the series with the most estimates; a candidate joins only if it agrees "
                     f"with every member on at least {consistency:.0%} of the months they both cover, and they must "
                     f"cover at least {shared_months} months together"),
            "parameters": {"series_agreement": float(series_agreement), "consistency": float(consistency),
                           "shared_months": int(shared_months)},
            "signatures": {name: {"months": len(signatures[name]), "estimates": counts[name],
                                  "offsets": {_offset_text(offset): count for offset, count
                                              in Counter(signatures[name].values()).most_common()}}
                           for name in sorted(signatures)}}


def _months(values, *, min_rows, agreement):
    """Per calendar month: the modal offset and whether the evidence supports calling it the month's offset."""
    grouped = {}
    for value in values:
        entry = grouped.setdefault(value["month"], {"offsets": Counter(), "by_currency": {}})
        entry["offsets"][value["offset_seconds"]] += 1
        entry["by_currency"].setdefault(value["currency"], Counter())[value["offset_seconds"]] += 1
    out = {}
    for month, entry in sorted(grouped.items()):
        total = sum(entry["offsets"].values())
        offset, count = entry["offsets"].most_common(1)[0]
        share = count / total
        # every economy that supplied enough rows must imply the same offset: one archive has one clock, and two
        # economies disagreeing means the wall clock is not what this job is modelling it as
        economies = {currency: counts.most_common(1)[0][0]
                     for currency, counts in entry["by_currency"].items() if sum(counts.values()) >= 3}
        record = {"n": total, "modal_offset_seconds": offset, "share": round(share, 6),
                  "offsets": {_offset_text(key): value for key, value in sorted(entry["offsets"].items())},
                  "economies": {currency: _offset_text(value) for currency, value in sorted(economies.items())}}
        if total < min_rows:
            record.update({"status": STATUS_UNDETERMINED, "reason": "TOO_FEW_ESTIMATES"})
        elif len(set(economies.values())) > 1:
            record.update({"status": STATUS_UNDETERMINED, "reason": "ECONOMIES_DISAGREE"})
        elif share < agreement:
            record.update({"status": STATUS_UNDETERMINED, "reason": "ESTIMATES_DISAGREE"})
        else:
            record.update({"status": STATUS_DETERMINED, "reason": None})
        out[month] = record
    return out


def _month_start(month):
    year, index = (int(part) for part in month.split("-"))
    return _date(year, index, 1)


def _month_end(month):
    start = _month_start(month)
    return (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)


def _periods(months):
    """Consecutive months that agree become one period. An undetermined month is its own period, never absorbed."""
    out = []
    for month in sorted(months):
        record = months[month]
        key = (record["status"], record.get("modal_offset_seconds") if record["status"] == STATUS_DETERMINED else None,
               record.get("reason"))
        if out and out[-1]["_key"] == key and _month_start(month) == out[-1]["end_date"] + timedelta(days=1):
            out[-1]["end_date"] = _month_end(month)
            out[-1]["months"].append(month)
            out[-1]["n"] += record["n"]
            out[-1]["_shares"].append(record["share"])
        else:
            out.append({"_key": key, "start_date": _month_start(month), "end_date": _month_end(month),
                        "months": [month], "n": record["n"], "_shares": [record["share"]],
                        "status": record["status"],
                        "utc_offset_seconds": (record["modal_offset_seconds"]
                                               if record["status"] == STATUS_DETERMINED else None),
                        "reason": record.get("reason")})
    periods = []
    for entry in out:
        shares = entry.pop("_shares")
        entry.pop("_key")
        entry["start_date"] = entry["start_date"].isoformat()
        entry["end_date"] = entry["end_date"].isoformat()
        entry["months"] = [entry["months"][0], entry["months"][-1]] if len(entry["months"]) > 1 else entry["months"]
        entry["confidence"] = round(min(shares), 6) if shares else 0.0
        entry["utc_offset"] = _offset_text(entry["utc_offset_seconds"]) if entry["status"] == STATUS_DETERMINED \
            else None
        entry["reading"] = (
            f"every naive wall clock in this period is read as {entry['utc_offset']}: the instant is the wall clock "
            f"minus that offset. {entry['n']} estimate(s), the weakest month agreeing at "
            f"{entry['confidence']:.3f}" if entry["status"] == STATUS_DETERMINED else
            f"CLOCK_PERIOD_UNDETERMINED ({entry['reason']}): {entry['n']} estimate(s) in this period do not "
            f"establish one offset, and no neighbouring period's offset is borrowed for it")
        periods.append(entry)
    return periods


def measure(archive_path, announcements_path, *, archive_columns=None, archive_has_header=False,
            announcement_columns=None, min_announcements=DEFAULT_MIN_ANNOUNCEMENTS,
            convention_agreement=DEFAULT_CONVENTION_AGREEMENT, min_month_rows=DEFAULT_MIN_MONTH_ROWS,
            month_agreement=DEFAULT_MONTH_AGREEMENT, series_consistency=DEFAULT_SERIES_CONSISTENCY,
            shared_months=DEFAULT_SHARED_MONTHS, min_cluster_zones=DEFAULT_MIN_CLUSTER_ZONES):
    """The archive's clock, period by period, with the evidence that established each period."""
    archive_meta, archive_rows = read_archive(archive_path, columns=archive_columns, has_header=archive_has_header)
    announce_meta, announcements = read_announcements(announcements_path, names=announcement_columns)
    found = conventions(announcements, min_announcements=min_announcements, agreement=convention_agreement)
    values, series = estimates(archive_rows, found)
    if not values:
        _refuse("NO_CONVENTION_MATCHED_THE_ARCHIVE",
                f"none of this archive's releases matched a release whose publication convention the announcement "
                f"archive establishes, so there is nothing to measure an offset against. "
                f"{sum(1 for entry in found.values() if entry.get('status') == 'OK')} convention(s) were "
                f"established from {announce_meta['announcements_with_an_instant']} announcement(s)")
    agreeing = cluster(values, series_agreement=month_agreement, consistency=series_consistency,
                       shared_months=shared_months)
    if len(agreeing["zones"]) < int(min_cluster_zones):
        _refuse("NO_MULTI_ZONE_CLUSTER",
                f"the series that agree about this archive's clock span {agreeing['zones']}, fewer than the "
                f"{min_cluster_zones} distinct zones declared as the fewest that make the agreement independent. "
                f"Economies whose daylight-saving calendars differ cannot agree on an offset by accident of one "
                f"convention; one zone alone can, so nothing is concluded from it")
    members = set(agreeing["members"])
    used = [value for value in values if value["series"] in members]
    months = _months(used, min_rows=min_month_rows, agreement=month_agreement)
    periods = _periods(months)
    determined = [period for period in periods if period["status"] == STATUS_DETERMINED]
    usable = {entry["status"] for entry in found.values()}
    return {
        "schema": SCHEMA,
        "provenance": "MEASURED_FROM_THE_ARCHIVE_AGAINST_OBSERVED_PUBLICATION_CONVENTIONS",
        "archive": archive_meta,
        "announcements": announce_meta,
        "method": {
            "convention": ("per (currency, release), the declared candidate zone in which the announcement "
                           "archive's observed instants fall most often on one local time of day; a release whose "
                           "announcements do not cross a daylight-saving boundary is NOT_DST_DISCRIMINATING and is "
                           "not used, because a fixed local time and a fixed UTC time are the same thing there"),
            "offset": ("per archive row of such a release, the archive's wall clock minus the instant the "
                       f"convention implies for that date, rounded to {OFFSET_GRID_SECONDS} seconds; the nearest of "
                       "the three candidate days is taken, so a wall clock hours from the zone is measured rather "
                       "than refused"),
            "cluster": ("one archive has one clock, so the series that disagree with each other about it cannot all "
                        "be measuring it. The largest mutually agreeing set is grown greedily from the series with "
                        "the most estimates and must span at least "
                        f"{min_cluster_zones} distinct zones; every series outside it is reported with its own "
                        "signature and the member it contradicts, and contributes nothing to the estimate"),
            "month": (f"a month's offset is the modal estimate when it has at least {min_month_rows} estimates, at "
                      f"least {month_agreement:.0%} of them agree, and every economy with at least three rows "
                      f"implies the same offset; otherwise the month is {STATUS_UNDETERMINED} and names which"),
            "period": ("consecutive months with the same verdict are one period. An UNDETERMINED month is never "
                       "absorbed into a neighbouring period: the rows inside it are excluded by name, because "
                       "borrowing a neighbour's offset is the guess this job exists to avoid"),
            "parameters": {"min_announcements": int(min_announcements),
                           "convention_agreement": float(convention_agreement),
                           "min_month_rows": int(min_month_rows), "month_agreement": float(month_agreement),
                           "offset_grid_seconds": OFFSET_GRID_SECONDS},
        },
        "conventions": {f"{currency}|{release}": entry for (currency, release), entry in sorted(found.items())
                        if entry.get("status") == "OK"},
        "conventions_not_used": {f"{currency}|{release}": {"status": entry.get("status"), "n": entry.get("n"),
                                                           "why": entry.get("why")}
                                 for (currency, release), entry in sorted(found.items())
                                 if entry.get("status") != "OK"},
        "evidence_by_series_and_year": {
            name: {"currency": entry["currency"], "zone": entry["zone"],
                   "convention_local_time": entry["convention_local_time"],
                   "by_year": {year: {"n": year_entry["n"],
                                      "wall_time_of_day": dict(year_entry["wall_time_of_day"].most_common(6)),
                                      "offsets": dict(year_entry["offsets"].most_common(4))}
                               for year, year_entry in sorted(entry["by_year"].items())}}
            for name, entry in sorted(series.items())},
        "cluster": agreeing,
        "months": months,
        "periods": periods,
        "counts": {
            "archive_rows_read": archive_meta["rows_read"],
            "conventions_established": sum(1 for entry in found.values() if entry.get("status") == "OK"),
            "convention_statuses": {status: sum(1 for entry in found.values() if entry.get("status") == status)
                                    for status in sorted(usable)},
            "offset_estimates": len(values),
            "offset_estimates_in_the_agreeing_cluster": len(used),
            "series_with_estimates": len(series),
            "series_in_the_agreeing_cluster": len(agreeing["members"]),
            "series_excluded_from_the_estimate": len(agreeing["rejected"]),
            "cluster_zones": agreeing["zones"],
            "months_measured": len(months),
            "months_determined": sum(1 for record in months.values() if record["status"] == STATUS_DETERMINED),
            "periods": len(periods),
            "periods_determined": len(determined),
            "distinct_offsets": sorted({period["utc_offset"] for period in determined}),
        },
        "fitted": "NOTHING: this job measures an offset from two archives; no model is fitted and no value is moved",
        "reading": ("each DETERMINED period says what the archive's naive wall clock means inside it, and the "
                    "evidence that says so is in this document. Each UNDETERMINED period says why it could not be "
                    "established; a row inside one is excluded CLOCK_PERIOD_UNDETERMINED by the row builder rather "
                    "than localized by the neighbouring period's offset"),
    }


# ---------------------------------------------------------------------------------------- reading it back

def load(path):
    """The clock document and the digest of the file it came from, or a refusal by name."""
    path = Path(path)
    if not path.is_file():
        _refuse("NO_SUCH_FILE", f"{path} is not a file this job can read")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        _refuse("CLOCK_UNREADABLE", f"{path} cannot be read as JSON ({exc})")
    if document.get("schema") != SCHEMA:
        _refuse("NOT_A_CLOCK_DOCUMENT",
                f"{path} carries schema {document.get('schema')!r} and a clock is {SCHEMA!r}; a document of another "
                f"schema is not a weaker input but an unknown one")
    periods = document.get("periods") or []
    if not periods:
        _refuse("CLOCK_HAS_NO_PERIODS", f"{path} establishes no period at all, so it localizes nothing")
    for period in periods:
        if period.get("status") == STATUS_DETERMINED and period.get("utc_offset_seconds") is None:
            _refuse("CLOCK_PERIOD_WITHOUT_AN_OFFSET",
                    f"{path} carries a DETERMINED period {period.get('start_date')}..{period.get('end_date')} with "
                    f"no offset in it")
    return {"path": str(path), "sha256": _file_digest(path), "document": document,
            "periods": sorted(periods, key=lambda entry: entry["start_date"])}


def period_of(clock, day):
    """The period a date falls in, or None. A date outside every period is not localized by the nearest one."""
    stamp = day.isoformat() if hasattr(day, "isoformat") else str(day)
    for period in clock["periods"]:
        if period["start_date"] <= stamp <= period["end_date"]:
            return period
    return None


# --------------------------------------------------------------------------------------------------- the CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.calendar_clock",
        description="Measure the UTC offset a naive economic-calendar archive's wall clock carries, per period, "
                    "against publication conventions read from an archive of observed announcement instants.")
    parser.add_argument("--archive", required=True, help="the naive consensus archive (CSV)")
    parser.add_argument("--announcements", required=True,
                        help="an archive of OBSERVED announcement instants, the conventions are read from it")
    parser.add_argument("--out", help="where to write the clock document; stdout when absent")
    parser.add_argument("--archive-columns", help="comma-separated column names for a headerless archive")
    parser.add_argument("--archive-has-header", action="store_true")
    parser.add_argument("--min-announcements", type=int, default=DEFAULT_MIN_ANNOUNCEMENTS)
    parser.add_argument("--convention-agreement", type=float, default=DEFAULT_CONVENTION_AGREEMENT)
    parser.add_argument("--min-month-rows", type=int, default=DEFAULT_MIN_MONTH_ROWS)
    parser.add_argument("--month-agreement", type=float, default=DEFAULT_MONTH_AGREEMENT)
    args = parser.parse_args(argv)
    try:
        document = measure(args.archive, args.announcements,
                           archive_columns=[part.strip() for part in args.archive_columns.split(",")]
                           if args.archive_columns else None,
                           archive_has_header=args.archive_has_header,
                           min_announcements=args.min_announcements,
                           convention_agreement=args.convention_agreement,
                           min_month_rows=args.min_month_rows, month_agreement=args.month_agreement)
    except (ClockRefusal, JoinRefusal) as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=False, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"{document['counts']['periods_determined']} determined period(s) of "
              f"{document['counts']['periods']} written to {args.out}: "
              f"{document['counts']['distinct_offsets']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
