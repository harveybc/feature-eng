"""Give a consensus an observed publication instant, by joining the two archives that each hold half of one release.

The owner's question -- how EUR/USD responds to a calendar surprise -- needs three numbers per release: what was
expected (the consensus), what arrived (the actual), and **when the actual became public**. No archive on this
machine carries all three. Two carry halves of it:

* a consensus archive: `event_date`, `event_time`, `country`, `description`, `actual`, `forecast`, `previous` -- one
  SCHEDULED instant per row, in the archive's own wall clock, with no zone and no publication timestamp;
* an announcement archive: one **observed** `announcement_datetime_utc` per release, with the released value, and no
  consensus at all.

This module joins them. The result is a calendar whose **actual's publication instant is observed** rather than
assumed, and whose only remaining clock assumption is the standard, much weaker one of the macro-announcement
literature: that the consensus stood before the release. `events.py` reads the result under
`publication_clock.mode: OBSERVED_ACTUAL_PUBLICATION` with `consensus_clock: ASSUMED_BEFORE_RELEASE` -- both written
into every artifact, because a reader must be able to see which of the two instants was measured.

**What a match is, and why it is this and nothing cleverer.** A consensus row and an announcement are the same
release when three declared things agree:

1. the **economy**, through a declared country-to-currency table (the consensus archive names countries, the
   announcement archive names currencies). Only exact correspondences are in the table: `Germany` is not `EUR`,
   because a German release is not a euro-area release, and joining them would attach one country's number to
   another's instant;
2. the **release**, through a declared normalisation (lowercase, every character that is not a letter or a digit
   becomes a space, runs of spaces collapse) plus a small **synonym table** used only where the two archives spell
   the same release differently. Every synonym that actually matched something is listed in the output, so the join
   can be read without reading this file;
3. the **calendar date**. The consensus archive's date is its own; the announcement's is the UTC date of the observed
   instant. When they differ -- a release near midnight, or an archive whose wall clock sits hours from UTC -- there
   is **no match**: the row is dropped and counted. Reaching to the neighbouring day would silently pair a release
   with another day's announcement, which is exactly the error the observed clock exists to remove.

**What is dropped, and counted by name.** `NO_OBSERVED_ANNOUNCEMENT`: a consensus row for which no announcement
carries that (economy, release, date) -- the reason is broken down further (no country in the table, no release name
in either archive's vocabulary, nothing announced that day) but the code is one, because they are one fact: this row
has no observed instant. `AMBIGUOUS_MATCH`: a key that more than one row on either side answers to -- one
announcement matching two consensus rows, or one consensus row matching two announcements. Both sides are dropped.
Choosing between two candidates by position, by proximity or by "the first one" would put an invented instant on a
real number, and there is no way to tell afterwards that it was invented.

The join rate is reported per event type, and so is the count of announcements no consensus row claimed. A join rate
is a finding: it says how much of the question the archives on this machine can answer at all.

Deterministic and CPU only: the standard library, plus pandas for a parquet announcement file when pandas and a
parquet engine are installed -- and a refusal by name when they are not.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import timezone as _timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .design import MISSING_TOKENS, _delimiter, _file_digest, _time_parser

SCHEMA = "m5phet.joined_calendar.v1"

#: the consensus archive's columns when it has no header row. These are the names `app/data_handler.py` assigns to
#: this file positionally; they are declared here rather than guessed from the file.
DEFAULT_ARCHIVE_COLUMNS = ("event_date", "event_time", "country", "volatility", "description", "evaluation",
                           "data_format", "actual", "forecast", "previous")

#: the announcement archive's column names, per its own data dictionary
DEFAULT_ANNOUNCEMENT_COLUMNS = {"currency": "currency", "indicator": "indicator",
                                "published": "announcement_datetime_utc", "value": "val", "period": "date"}

#: economy -> currency, exact correspondences only. A country whose releases are not the currency's releases is
#: absent on purpose, and its rows are dropped rather than joined to somebody else's instant.
COUNTRY_CURRENCY = {
    "united states": "USD",
    "euro zone": "EUR",
    "united kingdom": "GBP",
    "japan": "JPY",
    "canada": "CAD",
    "australia": "AUD",
    "new zealand": "NZD",
    "switzerland": "CHF",
    "china": "CNY",
    "sweden": "SEK",
    "brazil": "BRL",
    "denmark": "DKK",
    "poland": "PLN",
    "singapore": "SGD",
}

#: the declared synonym table: NORMALISED consensus description -> NORMALISED announcement indicator, used only
#: where the two archives spell the same release differently. Nothing here renames a release into another release:
#: each entry is one publication under two vocabularies. Entries that match nothing are listed as unused in the
#: output, so a synonym that quietly stopped applying is visible.
SYNONYMS = {
    "nonfarm payrolls": "non farm payrolls",
    "cpi": "inflation",
    "core cpi": "core inflation",
    "cpi mom": "inflation mom",
    "unemployment rate": "unemployment",
    "michigan consumer sentiment": "consumer sentiment",
    "michigan consumer expectations": "consumer expectations",
    "average hourly earnings": "average hourly earnings",
    "gdp": "gdp",
    "manufacturing pmi": "pmi",
    "core pce price index": "core pce",
    "jolts job openings": "job openings",
    "industrial production": "industrial production",
}

#: every reason a consensus row does not reach the joined calendar. Counted exactly; the count is the finding.
EXCLUSION_CODES = ("NO_OBSERVED_ANNOUNCEMENT", "AMBIGUOUS_MATCH", "NO_CONSENSUS", "NO_ACTUAL")

#: the finer reasons under NO_OBSERVED_ANNOUNCEMENT. They are a breakdown, never a second code: all of them mean the
#: same thing for the row, which is that nobody observed when it was published.
NO_ANNOUNCEMENT_REASONS = ("COUNTRY_NOT_IN_TABLE", "RELEASE_NAME_NOT_IN_THE_ANNOUNCEMENT_ARCHIVE",
                           "NOTHING_ANNOUNCED_ON_THAT_DATE")

#: how many dropped rows are described one by one. The COUNTS are always exact; only the examples are capped.
MAX_EXCLUDED_DETAIL = 200

#: the columns of the joined CSV, in this order. They are the names `events.py` reads without being told: the
#: observed instant is `published_at`, and the scheduled instant it replaces travels beside it as `event_time`.
JOINED_COLUMNS = ("event_type", "country", "currency", "event_time", "published_at", "actual", "consensus",
                  "previous", "historical_availability", "match_method", "synonym", "release_key",
                  "announcement_indicator", "announcement_value", "archive_row", "announcement_row")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class JoinRefusal(ValueError):
    """An input this job will not join, carrying the code a caller matches on and what was refused."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise JoinRefusal(code, why)


def normalise(text):
    """The declared normalisation, and the whole of it: lowercase, non-alphanumerics to spaces, runs collapsed.

    It is deliberately blunt. Anything cleverer -- stemming, dropping words it considers noise, fuzzy distance --
    would match releases that are not the same release, and a wrong match here attaches a real number to the wrong
    instant, which no later stage can detect.
    """
    return _NON_ALNUM.sub(" ", str(text or "").lower()).strip()


NORMALISATION = ("lowercase; every character that is not a letter or a digit becomes a space; runs of spaces "
                 "collapse; leading and trailing spaces are removed. No stemming, no stop-word removal, no fuzzy "
                 "distance: a near match is not a match")


def _zone(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        _refuse("UNKNOWN_TIMEZONE", f"{name!r} is not a zone this machine knows ({exc})")


# ------------------------------------------------------------------------------------------- the consensus archive

def read_archive(path, *, columns=None, has_header=False):
    """The consensus archive's rows as dictionaries of raw strings, with their row numbers kept."""
    path = Path(path)
    if not path.is_file():
        _refuse("NO_SUCH_FILE", f"{path} is not a file this job can read")
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        reader = csv.reader(handle, delimiter=_delimiter(sample))
        if has_header:
            try:
                header = [name.strip() for name in next(reader)]
            except StopIteration:
                _refuse("EMPTY_FILE", f"{path} has no header row")
        else:
            header = list(columns or DEFAULT_ARCHIVE_COLUMNS)
        if len(set(header)) != len(header):
            _refuse("DUPLICATE_COLUMN", f"the archive columns repeat a name: {header}")
        rows = []
        for number, row in enumerate(reader, start=2 if has_header else 1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                _refuse("RAGGED_ROW",
                        f"archive row {number} has {len(row)} fields and {len(header)} names were declared")
            rows.append({"row_number": number, **{name: row[i].strip() for i, name in enumerate(header)}})
    return {"path": str(path), "sha256": _file_digest(path), "columns": header, "rows_read": len(rows)}, rows


# ---------------------------------------------------------------------------------------- the announcement archive

def _announcement_rows_csv(path, names):
    with Path(path).open("r", newline="", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=_delimiter(sample))
        header = reader.fieldnames or []
        missing = [name for name in names.values() if name and name not in header]
        if missing:
            _refuse("COLUMN_NOT_IN_DATASET",
                    f"the announcement archive's columns are {header} and {missing} were declared but are not there")
        return header, [{"row_number": number, **{key: (row.get(name) or "").strip()
                                                  for key, name in names.items() if name}}
                        for number, row in enumerate(reader, start=2)]


def _announcement_rows_parquet(path, names):
    try:
        import pandas
    except ImportError:
        _refuse("PARQUET_READER_UNAVAILABLE",
                f"{path} is a parquet file and pandas is not installed in this environment; export it to CSV and "
                f"pass that, or install pandas. Nothing is guessed from the file name")
    try:
        frame = pandas.read_parquet(path)
    except ImportError as exc:
        _refuse("PARQUET_ENGINE_UNAVAILABLE",
                f"pandas is installed but cannot read {path}: {exc}. Export the file to CSV with an environment that "
                f"has a parquet engine and pass that CSV; its digest is recorded either way")
    header = [str(name) for name in frame.columns]
    missing = [name for name in names.values() if name and name not in header]
    if missing:
        _refuse("COLUMN_NOT_IN_DATASET",
                f"the announcement archive's columns are {header} and {missing} were declared but are not there")
    rows = []
    for number, (_, record) in enumerate(frame.iterrows(), start=2):
        rows.append({"row_number": number,
                     **{key: ("" if record[name] is None or str(record[name]) in ("NaT", "nan")
                              else str(record[name]))
                        for key, name in names.items() if name}})
    return header, rows


def read_announcements(path, *, names=None):
    """The announcements, each with the instant its source published it. Nothing here is computed."""
    path = Path(path)
    if not path.is_file():
        _refuse("NO_SUCH_FILE", f"{path} is not a file this job can read")
    names = dict(DEFAULT_ANNOUNCEMENT_COLUMNS, **(names or {}))
    read = _announcement_rows_parquet if path.suffix.lower() in (".parquet", ".pq") else _announcement_rows_csv
    header, rows = read(path, names)
    parsed = []
    for row in rows:
        text = row.get("published") or ""
        if text in MISSING_TOKENS:
            continue                    # an announcement with no instant is not an observed publication clock
        try:
            _, parser = _time_parser(text)
            moment = parser(text)
        except ValueError:
            _refuse("TIMESTAMP_UNPARSEABLE",
                    f"announcement row {row['row_number']} reads {text!r} as its publication instant, which no "
                    f"declared format reads")
        if moment.tzinfo is None or moment.utcoffset() is None:
            _refuse("ANNOUNCEMENT_INSTANT_WITHOUT_A_ZONE",
                    f"announcement row {row['row_number']} reads {text!r}, a wall clock with no offset. The whole "
                    f"point of this archive is that its instants are observed; a naive one is not an instant")
        moment = moment.astimezone(_timezone.utc)
        parsed.append({"row_number": row["row_number"], "currency": (row.get("currency") or "").strip().upper(),
                       "indicator": row.get("indicator") or "", "published_at": moment,
                       "value": row.get("value") or "", "period": row.get("period") or ""})
    parsed.sort(key=lambda r: (r["published_at"], r["row_number"]))
    return {"path": str(path), "sha256": _file_digest(path), "columns": header, "rows_read": len(rows),
            "announcements_with_an_instant": len(parsed), "column_names": names,
            "first": parsed[0]["published_at"].isoformat() if parsed else None,
            "last": parsed[-1]["published_at"].isoformat() if parsed else None}, parsed


# ------------------------------------------------------------------------------------------------------- the join

class _Excluded:
    """Exact counts, capped examples, and the finer reasons kept as a breakdown of the one code."""

    def __init__(self):
        self.counts = {code: 0 for code in EXCLUSION_CODES}
        self.reasons = {reason: 0 for reason in NO_ANNOUNCEMENT_REASONS}
        self.rows, self.truncated = [], {code: 0 for code in EXCLUSION_CODES}

    def drop(self, code, row, why, *, reason=None):
        self.counts[code] += 1
        if reason is not None:
            self.reasons[reason] += 1
        if len(self.rows) < MAX_EXCLUDED_DETAIL:
            self.rows.append({"archive_row": row.get("row_number"), "event_type": row.get("event_type"),
                              "date": row.get("date"), "code": code, "reason": reason, "why": why})
        else:
            self.truncated[code] += 1

    def document(self):
        return {"counts": self.counts,
                "no_observed_announcement_by_reason": self.reasons,
                "counts_reading": ("every count is of CONSENSUS ROWS. NO_OBSERVED_ANNOUNCEMENT means no announcement "
                                   "carries that (economy, release, calendar date); AMBIGUOUS_MATCH means the key is "
                                   "answered by more than one row on one side or the other, and both sides are "
                                   "dropped rather than paired by position"),
                "rows": self.rows, "examples_truncated_at": MAX_EXCLUDED_DETAIL,
                "examples_not_listed": {code: n for code, n in self.truncated.items() if n}}


def _archive_date(row, zone_name, zone_declared):
    """The calendar date the join keys on, and the instant the archive scheduled the release for.

    With no zone declared the date is the archive's own, exactly as written: reading a wall clock as UTC in order to
    take a date from it is a conversion nobody asked for. With a zone declared the wall clock becomes an instant and
    the date is that instant's UTC date -- which is the only way a row near midnight can meet its announcement.
    """
    text = " ".join(part for part in (row.get("event_date") or "", row.get("event_time") or "") if part).strip()
    if not text:
        return None, None
    try:
        _, parser = _time_parser(text)
        moment = parser(text)
    except ValueError:
        _refuse("TIMESTAMP_UNPARSEABLE",
                f"archive row {row['row_number']} reads {text!r} as its scheduled instant, which no declared format "
                f"reads")
    if moment.tzinfo is None or moment.utcoffset() is None:
        if not zone_declared:
            return moment, moment.date().isoformat()
        moment = moment.replace(tzinfo=_zone(zone_name))
    moment = moment.astimezone(_timezone.utc)
    return moment, moment.date().isoformat()


def _number(text):
    if text is None or text.strip() in MISSING_TOKENS:
        return None
    value = text.strip().replace(",", "") if text.count(",") == 1 and "." not in text else text.strip()
    try:
        return float(value)
    except ValueError:
        return None


def join(archive_path, announcements_path, *, archive_columns=None, archive_has_header=False,
         archive_timezone=None, announcement_columns=None, synonyms=None, country_currency=None):
    """One row per consensus row that has an observed publication instant, and an exact count of every one that has
    not."""
    synonyms = dict(SYNONYMS if synonyms is None else synonyms)
    countries = dict(COUNTRY_CURRENCY if country_currency is None else country_currency)
    archive_meta, archive_rows = read_archive(archive_path, columns=archive_columns, has_header=archive_has_header)
    announce_meta, announcements = read_announcements(announcements_path, names=announcement_columns)

    # the announcements, indexed by the key a consensus row will present. A key more than one announcement answers to
    # is kept as a list, because the ambiguity is the finding and collapsing it here would hide it.
    index, vocabulary = {}, set()
    for announcement in announcements:
        name = normalise(announcement["indicator"])
        vocabulary.add(name)
        key = (announcement["currency"], name, announcement["published_at"].date().isoformat())
        index.setdefault(key, []).append(announcement)

    zone_declared = archive_timezone is not None
    zone_name = archive_timezone or "UTC"
    excluded = _Excluded()
    synonym_hits = {}
    candidates, by_key = [], {}
    for row in archive_rows:
        country = (row.get("country") or "").strip()
        description = (row.get("description") or "").strip()
        event_type = f"{country} | {description}".strip(" |")
        scheduled, date = _archive_date(row, zone_name, zone_declared)
        row = {**row, "event_type": event_type, "date": date}
        actual = _number(row.get("actual"))
        consensus = _number(row.get("forecast"))
        if actual is None:
            continue                    # a scheduled row with no number is not a release; it is not a drop either
        if consensus is None:
            excluded.drop("NO_CONSENSUS", row,
                          "this row carries no consensus, so there is nothing for an observed instant to date")
            continue
        if date is None:
            excluded.drop("NO_OBSERVED_ANNOUNCEMENT", row, "this row carries no date at all",
                          reason="NOTHING_ANNOUNCED_ON_THAT_DATE")
            continue
        currency = countries.get(normalise(country))
        if currency is None:
            excluded.drop("NO_OBSERVED_ANNOUNCEMENT", row,
                          f"{country!r} has no exact currency correspondence in the declared table, and joining it "
                          f"to a neighbouring economy's announcements would attach one country's number to "
                          f"another's instant", reason="COUNTRY_NOT_IN_TABLE")
            continue
        normalised = normalise(description)
        method, synonym, name = "EXACT", None, normalised
        if normalised not in vocabulary and normalised in synonyms:
            name, method, synonym = synonyms[normalised], "SYNONYM", normalised
        key = (currency, name, date)
        matches = index.get(key)
        if not matches:
            reason = ("RELEASE_NAME_NOT_IN_THE_ANNOUNCEMENT_ARCHIVE" if name not in vocabulary
                      else "NOTHING_ANNOUNCED_ON_THAT_DATE")
            excluded.drop("NO_OBSERVED_ANNOUNCEMENT", row,
                          f"no announcement carries ({currency}, {name!r}, {date}); the release is not matched to a "
                          f"neighbouring day, because a neighbouring day's instant is another release's instant",
                          reason=reason)
            continue
        if len(matches) > 1:
            excluded.drop("AMBIGUOUS_MATCH", row,
                          f"{len(matches)} announcements carry ({currency}, {name!r}, {date}); choosing one of them "
                          f"would put an invented instant on a real number")
            continue
        entry = {"row": row, "announcement": matches[0], "scheduled": scheduled, "actual": actual,
                 "consensus": consensus, "previous": _number(row.get("previous")), "method": method,
                 "synonym": synonym, "release_key": f"{currency}|{name}|{date}", "key": key}
        candidates.append(entry)
        by_key.setdefault(key, []).append(entry)

    joined = []
    for entry in candidates:
        siblings = by_key[entry["key"]]
        if len(siblings) > 1:
            excluded.drop("AMBIGUOUS_MATCH", entry["row"],
                          f"{len(siblings)} consensus rows answer to ({entry['release_key']}), so the one observed "
                          f"announcement would have to be given to more than one of them")
            continue
        if entry["synonym"]:
            synonym_hits[entry["synonym"]] = synonym_hits.get(entry["synonym"], 0) + 1
        announcement = entry["announcement"]
        joined.append({
            "event_type": entry["row"]["event_type"],
            "country": (entry["row"].get("country") or "").strip(),
            "currency": entry["key"][0],
            "event_time": entry["scheduled"].isoformat() if entry["scheduled"] else "",
            "published_at": announcement["published_at"].isoformat(),
            "actual": entry["actual"], "consensus": entry["consensus"], "previous": entry["previous"],
            "historical_availability": "KNOWN",
            "match_method": entry["method"], "synonym": entry["synonym"] or "",
            "release_key": entry["release_key"],
            "announcement_indicator": announcement["indicator"],
            "announcement_value": announcement["value"],
            "archive_row": entry["row"]["row_number"], "announcement_row": announcement["row_number"],
        })
    joined.sort(key=lambda row: (row["published_at"], row["event_type"], row["archive_row"]))

    matched_announcements = {row["announcement_row"] for row in joined}
    by_event_type = {}
    # the join rate per event type is over every consensus row that could have been joined -- one that carried a
    # consensus and an actual -- and not over the ones that happened to join, which would make every rate 100 %
    considered = {}
    for row in archive_rows:
        country = (row.get("country") or "").strip()
        description = (row.get("description") or "").strip()
        event_type = f"{country} | {description}".strip(" |")
        if _number(row.get("actual")) is None or _number(row.get("forecast")) is None:
            continue
        considered[event_type] = considered.get(event_type, 0) + 1
    for row in joined:
        entry = by_event_type.setdefault(row["event_type"], {"joined": 0})
        entry["joined"] += 1
    rates = {}
    for event_type, total in sorted(considered.items()):
        got = by_event_type.get(event_type, {}).get("joined", 0)
        rates[event_type] = {"consensus_rows_with_an_actual": total, "joined": got,
                             "join_rate": round(got / total, 6) if total else None}

    return {
        "schema": SCHEMA,
        "provenance": "DEVELOPMENT_OBSERVED_CLOCK",
        "publication_clock": {
            "mode": "OBSERVED_ACTUAL_PUBLICATION",
            "consensus_clock": "ASSUMED_BEFORE_RELEASE",
            "declared_by": "the announcement archive for the actual; the operator for the consensus",
            "identification_caveat": (
                "the instant the ACTUAL became public is the one the announcement archive observed. The instant the "
                "consensus became public was not observed by anyone here, and is assumed only to precede the "
                "release -- the standard assumption of the macro-announcement literature, and a far weaker one than "
                "assuming the release happened when it was scheduled"),
        },
        "archive": archive_meta,
        "announcements": announce_meta,
        "matching": {
            "keys": ["currency (from the country, through the declared table)",
                     "normalised release name (with the declared synonyms)",
                     "calendar date (the archive's own date, or the UTC date of its instant when a zone is declared)"],
            "normalisation": NORMALISATION,
            "archive_timezone": archive_timezone,
            "archive_timezone_reading": (
                "declared: the archive's wall clock was read in this zone and the join date is the UTC date of the "
                "resulting instant" if zone_declared else
                "not declared: the join date is the archive's own calendar date, exactly as written, and no wall "
                "clock was reinterpreted as an instant"),
            "country_currency": dict(sorted(countries.items())),
            "synonyms_declared": dict(sorted(synonyms.items())),
            "synonyms_used": dict(sorted(synonym_hits.items())),
            "synonyms_declared_but_unused": sorted(set(synonyms) - set(synonym_hits)),
            "date_boundary": ("a consensus row whose date is not the UTC date of the announcement does not match; it "
                              "is dropped NO_OBSERVED_ANNOUNCEMENT rather than paired with the neighbouring day"),
        },
        "counts": {
            "archive_rows_read": archive_meta["rows_read"],
            "announcements_read": announce_meta["rows_read"],
            "announcements_with_an_instant": announce_meta["announcements_with_an_instant"],
            "consensus_rows_with_an_actual_and_a_consensus": sum(considered.values()),
            "joined": len(joined),
            "join_rate": (round(len(joined) / sum(considered.values()), 6) if considered else None),
            "announcements_matched": len(matched_announcements),
            "announcements_no_consensus_row_claimed": (announce_meta["announcements_with_an_instant"]
                                                       - len(matched_announcements)),
            "by_event_type": rates,
        },
        "excluded": excluded.document(),
        "rows": joined,
        "fitted": "NOTHING: this job joins two archives; no model is fitted and no value is transformed",
        "reading": ("every row's `published_at` is an OBSERVED instant from the announcement archive, and its "
                    "`consensus` and `actual` are the consensus archive's own numbers. `event_time` is the instant "
                    "the consensus archive scheduled the release for, kept beside the observed one so the two can "
                    "be compared, and never used as a publication clock. A join rate below 1 is not a failure of "
                    "this job: it is the measure of how much of the question these two archives can answer"),
    }


def write_csv(document, path):
    """The joined calendar, in the columns `events.py` reads without being told which is which."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(JOINED_COLUMNS)
        for row in document["rows"]:
            writer.writerow(["" if row.get(name) is None else row.get(name) for name in JOINED_COLUMNS])
    return path


# --------------------------------------------------------------------------------------------------------- the CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.calendar_join",
        description="Join a consensus archive to an archive of observed announcement instants. Nothing is fitted.")
    parser.add_argument("--archive", required=True, help="the consensus archive (CSV)")
    parser.add_argument("--announcements", required=True,
                        help="the observed announcement archive (CSV, or parquet when a parquet engine is installed)")
    parser.add_argument("--out", required=True, help="where to write the joined calendar (CSV)")
    parser.add_argument("--report", help="where to write the join document (JSON); stdout when absent")
    parser.add_argument("--archive-columns", help="comma-separated column names for a headerless consensus archive")
    parser.add_argument("--archive-has-header", action="store_true")
    parser.add_argument("--archive-timezone",
                        help="the IANA zone the consensus archive's wall clock is read in before its date is taken; "
                             "absent means the archive's own calendar date is used as written")
    parser.add_argument("--announcement-currency-column")
    parser.add_argument("--announcement-indicator-column")
    parser.add_argument("--announcement-published-column")
    parser.add_argument("--announcement-value-column")
    args = parser.parse_args(argv)
    names = {key: value for key, value in (("currency", args.announcement_currency_column),
                                           ("indicator", args.announcement_indicator_column),
                                           ("published", args.announcement_published_column),
                                           ("value", args.announcement_value_column)) if value}
    try:
        document = join(args.archive, args.announcements,
                        archive_columns=[part.strip() for part in args.archive_columns.split(",")]
                        if args.archive_columns else None,
                        archive_has_header=args.archive_has_header,
                        archive_timezone=args.archive_timezone,
                        announcement_columns=names or None)
    except JoinRefusal as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    write_csv(document, args.out)
    text = json.dumps(document, indent=2, sort_keys=False, allow_nan=False)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    print(f"{document['counts']['joined']} row(s) joined of "
          f"{document['counts']['consensus_rows_with_an_actual_and_a_consensus']} consensus row(s) with a number "
          f"(join rate {document['counts']['join_rate']}) written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
