"""Event windows: what the market did after each calendar release, and what it had been told to expect.

WP22 step 1. A macro release is a pulse: at one instant a number arrives that somebody had already forecast, and the
price path afterwards is the response. To estimate that response the way the literature does -- an event study with
local projections (Jorda 2005), the impulse read against the *standardized surprise*
`(actual - consensus) / sigma`, identified conditional on the pre-release information set
(Andersen-Bollerslev-Diebold-Vega 2003; Gurkaynak-Sack-Swanson 2005) -- somebody must first write down, per release,
the surprise, the outcome paths at each horizon, and the other pulses that landed inside the same window. That table
is what this module builds, and nothing more: no model is fitted here and no causal word is used.

The whole difficulty is that three of those quantities are trivially corruptible by information nobody had:

* **The scale.** `sigma_k` is the dispersion of that event type's *past* surprises. Computed over the whole sample it
  is a number from the future, and every standardized surprise built on it is contaminated. Here it is recomputed at
  every release from releases **published strictly before** it, and a release with fewer than a declared minimum of
  prior releases is refused `INSUFFICIENT_HISTORY` instead of being standardized by a scale nobody could have had.
* **The surprise itself.** It is `app/economic_calendar.py`'s `release_surprise` -- the first actual its SOURCE
  published, against the last consensus its source had PUBLISHED before it -- computed by that module, through its
  own arrival store, so the release boundary this table is built on is the audited one. A dataset that carries a
  consensus but never says when the consensus was published cannot support that boundary: the module reports
  `MISSING_PUBLICATION_CLOCK` for the arrival and no release surprise, and the release is excluded and counted here.
  Substituting our receipt for the source's clock would move rows in and out of the window invisibly.
* **The outcome.** A gap in the bars inside `[t_k, t_k + h]` is a horizon whose path was not observed. Interpolating
  across it invents the very movement the response is supposed to measure, so the (event, horizon) is refused
  `BARS_MISSING_AT_HORIZON` and counted. The same rule covers a release that falls into a market closure: there is no
  bar at or before it within one sampling step, so every one of its horizons is refused by that name.

Everything else is declared rather than defaulted: the horizons, the window `W` in which the other releases are
listed, the pre-event volatility span, the minimum history, the sampling step the realized volatility is summed over,
and the zone a naive timestamp is read in. A naive calendar timestamp with no declared zone is refused
`AMBIGUOUS_LOCAL_TIME`, in the same words and for the same reason as the calendar module.

Deterministic and CPU only: no seed is drawn, no model is fitted, and the same inputs with the same arguments produce
the same bytes. numpy and the standard library, plus pandas for parquet bars when pandas is installed -- and a
refusal by name when it is not.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone as _timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from app.economic_calendar import SCHEMA as ARRIVAL_SCHEMA, CalendarRefusal, PointInTimeCalendar

from .design import MISSING_TOKENS, TIME_COLUMN_NAMES, _delimiter, _file_digest, _time_parser

SCHEMA = "m5phet.event_rows.v1"

#: the horizons, in minutes, at which the response is read. Declared, configurable, and written into the document so
#: a later stage joins on them rather than assuming them.
DEFAULT_HORIZONS_MINUTES = (5, 15, 30, 60, 240)

#: `W`: how far either side of a release the OTHER releases are listed. 24 h is the plan's declared value.
DEFAULT_WINDOW_HOURS = 24.0

#: the span of the pre-event realized volatility, one of the pre-release controls the identification rests on
DEFAULT_PRE_EVENT_MINUTES = 60

#: the fewest prior releases of the same event type that make a dispersion mean anything. Below it the release is
#: refused rather than standardized by a scale estimated from three numbers.
DEFAULT_MIN_PRIOR_RELEASES = 8

#: every reason a release, or one of its horizons, does not become a row. Each is counted exactly, and the counts are
#: part of the document: a row that disappears without a reason is indistinguishable from a row nobody ever sent.
EXCLUSION_CODES = (
    "MISSING_PUBLICATION_CLOCK",
    "NO_CONSENSUS",
    "INSUFFICIENT_HISTORY",
    "NON_POSITIVE_RESIDUAL_SCALE",
    "BARS_MISSING_AT_HORIZON",
)

#: how many excluded rows are described one by one before the list says it was truncated. The COUNTS are always exact;
#: only the examples are capped, so a calendar that drops fifty thousand rows does not produce a JSON nobody can open.
MAX_EXCLUDED_DETAIL = 200

#: column names read as the bars' price when `--bars-price-column` is not given, in this order
PRICE_COLUMN_NAMES = ("close", "adj_close", "price", "mid_close", "c")

#: column names read as the calendar's event-type label when it is not named
EVENT_COLUMN_NAMES = ("event_type", "event", "event_name", "indicator", "description", "release", "name")

#: column names read as the calendar's consensus when it is not named. `forecast` is what the archive in
#: `tests/data/economic_calendar_2011_2021.csv` calls it, per `app/data_handler.py`.
CONSENSUS_COLUMN_NAMES = ("consensus", "forecast", "consensus_estimate", "expected")

ACTUAL_COLUMN_NAMES = ("actual", "value", "val")

#: column names read as the calendar's event instant when it is not named
EVENT_TIME_COLUMN_NAMES = ("event_time", "announcement_datetime_local") + TIME_COLUMN_NAMES

#: the OPTIONAL roles, and the column names that already say what they are. Nothing here is inferred from a file
#: name or a position: a column is read as a publication clock only when it is called one, or when it is named.
OPTIONAL_COLUMN_NAMES = {
    "published": ("published_at", "publication_time", "announcement_datetime_utc", "announcement_datetime"),
    "consensus_published": ("consensus_published_at", "consensus_publication_time"),
    "received": ("received_at", "observed_at"),
    "previous": ("previous", "prior"),
    "unit": ("unit",),
    "period": ("period",),
    "availability": ("historical_availability",),
}

AVAILABILITY = ("KNOWN", "UNKNOWN")

#: which clock the release boundary is read off. `observed` is the only one that is a measurement: the instants the
#: dataset itself declares. `scheduled` is an ASSUMPTION a person makes out loud -- that each release was published at
#: the instant it was scheduled for -- and it exists so that an archive with no publication clock can be worked on at
#: all. It is never a default, it is written into every artifact it touches, and nothing estimated under it is
#: identified.
PUBLICATION_CLOCK_MODES = ("observed", "scheduled")

#: how long before the scheduled instant the consensus is assumed to have stood, under the same declared assumption
DEFAULT_ASSUMED_TOLERANCE_SECONDS = 60

#: the provenance of the rows, per clock mode. It travels on the document AND on every row, so a row separated from
#: its document still says what its instants were.
PROVENANCE = {"observed": "DEVELOPMENT_OBSERVED_CLOCK", "scheduled": "DEVELOPMENT_ASSUMED_CLOCK"}


def publication_clock_block(mode, tolerance_seconds):
    """What a reader must be told about the instants underneath every number in this document."""
    if mode == "observed":
        return {"mode": "OBSERVED_PUBLICATION_CLOCK", "tolerance_seconds": None, "declared_by": "the dataset",
                "identification_caveat": ("the publication instants are the ones the dataset declares; a release that "
                                          "declares none is excluded MISSING_PUBLICATION_CLOCK and never assumed")}
    return {"mode": "ASSUMED_SCHEDULED_PUBLICATION",
            "tolerance_seconds": int(tolerance_seconds),
            "declared_by": "operator",
            "identification_caveat": ("the surprise's publication instant is assumed equal to the scheduled instant; "
                                      "no receipt or publication timestamp was observed; results are DEVELOPMENT and "
                                      "not identified until a publication clock exists"),
            "consensus": ("the consensus is taken as published tolerance_seconds before the scheduled instant, under "
                          "the same declared assumption; no consensus publication timestamp was observed either")}

#: what is written into `unit` and `period` when the dataset declares neither. It is a constant, so the calendar
#: module's INCOMPARABLE_SERIES check cannot fire -- and the document says exactly that, because a check that cannot
#: fail is not a check and nobody should read it as one.
UNDECLARED = "NOT_DECLARED_BY_THE_DATASET"


class EventsRefusal(ValueError):
    """An input this job will not build a table from, carrying the code a caller matches on and what was refused."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise EventsRefusal(code, why)


def _zone(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        _refuse("UNKNOWN_TIMEZONE", f"{name!r} is not a zone this machine knows ({exc})")


def _aware(moment, zone, *, where, zone_declared):
    """One instant, or a refusal. A naive wall clock is not an instant until somebody declares the zone, and inside a
    daylight-saving fold it names two instants even then -- which is the calendar module's own rule, in its words."""
    if moment.tzinfo is not None and moment.utcoffset() is not None:
        return moment.astimezone(_timezone.utc)
    if not zone_declared:
        _refuse("AMBIGUOUS_LOCAL_TIME",
                f"{where} carries no offset and no zone was declared. During a daylight-saving fold one local wall "
                f"clock names two instants, and choosing one of them is a guess about when something was knowable "
                f"-- declare the zone")
    early, late = moment.replace(tzinfo=zone, fold=0), moment.replace(tzinfo=zone, fold=1)
    if early.utcoffset() != late.utcoffset():
        _refuse("AMBIGUOUS_LOCAL_TIME",
                f"{where} falls inside a daylight-saving fold of the declared zone, where one wall clock names two "
                f"instants; neither is chosen for you")
    return early.astimezone(_timezone.utc)


# ------------------------------------------------------------------------------------------------------- the bars

def _pandas():
    """Kept behind a function so a test can remove parquet support without removing pandas."""
    try:
        import pandas
    except ImportError:
        return None
    return pandas


def _pick(header, declared, candidates, *, what):
    if declared is not None:
        if declared not in header:
            _refuse("COLUMN_NOT_IN_DATASET", f"{what} {declared!r} is not a column of this file; its columns are "
                                             f"{list(header)}")
        return declared
    lowered = {name.lower(): name for name in header}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    _refuse("NO_SUCH_COLUMN",
            f"none of this file's columns {list(header)} is named as {what} ({list(candidates)}); name it explicitly")


def _step_seconds(seconds):
    """The grid the bars actually sit on: the modal positive step, with everything that does not honour it counted."""
    deltas = np.diff(seconds)
    if deltas.size == 0:
        _refuse("TOO_FEW_BARS", "two bars are needed before a sampling step exists")
    if np.any(deltas <= 0):
        first = int(np.argmax(deltas <= 0)) + 1
        _refuse("BARS_NOT_INCREASING",
                f"bar {first} is not later than the one before it; a path read off rows that are not in time order "
                f"is not a path")
    values, counts = np.unique(np.round(deltas).astype(np.int64), return_counts=True)
    step = int(values[int(np.argmax(counts))])
    if step <= 0:
        _refuse("BAD_SAMPLING_STEP", f"the modal step between bars is {step} seconds")
    return step, int(np.count_nonzero(np.round(deltas).astype(np.int64) != step))


def _read_bars_csv(path, *, time_column, price_column, zone, zone_declared, max_rows):
    with Path(path).open("r", newline="", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        reader = csv.reader(handle, delimiter=_delimiter(sample))
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration:
            _refuse("EMPTY_FILE", f"{path} has no header row")
        if len(set(header)) != len(header):
            _refuse("DUPLICATE_COLUMN", f"the bars header repeats a column name: {header}")
        time_column = _pick(header, time_column, TIME_COLUMN_NAMES, what="the bars' timestamp column")
        price_column = _pick(header, price_column, PRICE_COLUMN_NAMES, what="the bars' price column")
        ti, pi = header.index(time_column), header.index(price_column)
        times, prices, parser, fmt_name = [], [], None, None
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                _refuse("RAGGED_ROW", f"bars row {row_number} has {len(row)} fields and the header has {len(header)}")
            stamp, price = row[ti].strip(), row[pi].strip()
            if stamp in MISSING_TOKENS:
                _refuse("MISSING_TIMESTAMP", f"bars row {row_number} has no value in {time_column!r}")
            if price in MISSING_TOKENS:
                _refuse("MISSING_PRICE", f"bars row {row_number} has no value in {price_column!r}; a gap in the price "
                                         f"column is not a price, and filling it would invent the movement this table "
                                         f"is meant to measure")
            if parser is None:
                fmt_name, parser = _time_parser(stamp)
            try:
                moment = parser(stamp)
            except ValueError:
                _refuse("TIMESTAMP_UNPARSEABLE",
                        f"bars row {row_number} reads {stamp!r}, which the parser chosen from the first row "
                        f"({fmt_name}) cannot read")
            times.append(_aware(moment, zone, where=f"bars row {row_number}", zone_declared=zone_declared).timestamp())
            try:
                prices.append(float(price))
            except ValueError:
                _refuse("PRICE_NOT_A_NUMBER", f"bars row {row_number} reads {price!r} in {price_column!r}")
            if max_rows is not None and len(times) >= max_rows:
                break
    return header, time_column, price_column, fmt_name, np.asarray(times, dtype=np.float64), \
        np.asarray(prices, dtype=np.float64)


def _read_bars_parquet(path, *, time_column, price_column, zone, zone_declared, max_rows):
    pandas = _pandas()
    if pandas is None:
        _refuse("PARQUET_READER_UNAVAILABLE",
                f"{path} is a parquet file and pandas is not installed in this environment; install it or supply the "
                f"bars as CSV. Nothing is guessed from the file name")
    try:
        frame = pandas.read_parquet(path)
    except ImportError as exc:
        _refuse("PARQUET_ENGINE_UNAVAILABLE",
                f"pandas is installed but cannot read {path}: {exc}. Install a parquet engine, or supply the bars as "
                f"CSV; nothing here reads the file by guessing its layout")
    if time_column is None and not any(str(name).lower() in TIME_COLUMN_NAMES for name in frame.columns):
        # a parquet often carries its instants in the index rather than a column; that is the only case in which the
        # index is read, and it is read as itself, never as row order standing in for a clock
        if not isinstance(frame.index, pandas.DatetimeIndex):
            _refuse("NO_TIMESTAMP_COLUMN",
                    f"the parquet's columns are {[str(n) for n in frame.columns]} and none of them is named as a "
                    f"timestamp, nor is its index a datetime index")
        frame = frame.reset_index()
    header = [str(name) for name in frame.columns]
    time_column = _pick(header, time_column, TIME_COLUMN_NAMES, what="the bars' timestamp column")
    price_column = _pick(header, price_column, PRICE_COLUMN_NAMES, what="the bars' price column")
    if max_rows is not None:
        frame = frame.iloc[:max_rows]
    stamps = pandas.to_datetime(frame[time_column])
    moments = [value.to_pydatetime() if hasattr(value, "to_pydatetime") else value for value in stamps]
    times = np.asarray([_aware(m, zone, where=f"bars row {i + 2}", zone_declared=zone_declared).timestamp()
                        for i, m in enumerate(moments)], dtype=np.float64)
    prices = np.asarray(frame[price_column].to_numpy(dtype="float64"), dtype=np.float64)
    return header, time_column, price_column, "parquet", times, prices


def read_bars(path, *, time_column=None, price_column=None, timezone_name="UTC", timezone_declared=False,
              max_rows=None):
    """The price path, as instants and logarithms. The only thing here that touches the caller's bars."""
    path = Path(path)
    if not path.is_file():
        _refuse("NO_SUCH_FILE", f"{path} is not a file this job can read")
    zone = _zone(timezone_name)
    read = _read_bars_parquet if path.suffix.lower() in (".parquet", ".pq") else _read_bars_csv
    header, time_column, price_column, fmt_name, times, prices = read(
        path, time_column=time_column, price_column=price_column, zone=zone, zone_declared=timezone_declared,
        max_rows=max_rows)
    if times.size < 2:
        _refuse("TOO_FEW_BARS", f"{path} yielded {times.size} bar(s)")
    if not np.all(np.isfinite(prices)):
        _refuse("NON_FINITE_PRICE", f"{path} carries a price that is not a finite number")
    if np.any(prices <= 0):
        _refuse("NON_POSITIVE_PRICE",
                f"{path} carries a price <= 0, and a log return of it does not exist")
    step, off_grid = _step_seconds(times)
    return {"path": str(path), "sha256": _file_digest(path), "columns": header, "time_column": time_column,
            "price_column": price_column, "time_format": fmt_name, "rows": int(times.size),
            "step_seconds": step, "bars_off_the_modal_step": off_grid,
            "first": datetime.fromtimestamp(float(times[0]), _timezone.utc).isoformat(),
            "last": datetime.fromtimestamp(float(times[-1]), _timezone.utc).isoformat(),
            "timezone": timezone_name, "timezone_declared": bool(timezone_declared),
            "times": times, "log_price": np.log(prices)}


# --------------------------------------------------------------------------------------------------- the calendar

def _calendar_rows(path, *, columns):
    """The calendar's rows as dictionaries of raw strings. `columns` declares the names of a headerless file."""
    path = Path(path)
    if not path.is_file():
        _refuse("NO_SUCH_FILE", f"{path} is not a file this job can read")
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        reader = csv.reader(handle, delimiter=_delimiter(sample))
        if columns is None:
            try:
                header = [name.strip() for name in next(reader)]
            except StopIteration:
                _refuse("EMPTY_FILE", f"{path} has no header row and --calendar-columns was not declared")
        else:
            header = list(columns)
        if len(set(header)) != len(header):
            _refuse("DUPLICATE_COLUMN", f"the calendar columns repeat a name: {header}")
        rows = []
        for row_number, row in enumerate(reader, start=1 if columns is not None else 2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                _refuse("RAGGED_ROW",
                        f"calendar row {row_number} has {len(row)} fields and {len(header)} names were declared")
            rows.append({"row_number": row_number,
                         **{name: row[i].strip() for i, name in enumerate(header)}})
    return header, rows


def _cell(row, column):
    if column is None:
        return None
    value = row.get(column)
    return None if value is None or value in MISSING_TOKENS else value


def _float_cell(row, column, *, where):
    text = _cell(row, column)
    if text is None:
        return None
    try:
        value = float(text.replace(",", "") if "," in text and text.count(",") == 1 and "." not in text else text)
    except ValueError:
        return None
    return value if math.isfinite(value) else _refuse("NON_FINITE_VALUE", f"{where} reads {text!r}")


def _parse_instant(texts, zone, *, where, zone_declared):
    """One instant from one or two cells, read by the formats `design` declares and by no other."""
    joined = " ".join(text for text in texts if text)
    if not joined:
        return None
    try:
        _, parser = _time_parser(joined)
        moment = parser(joined)
    except ValueError:
        _refuse("TIMESTAMP_UNPARSEABLE", f"{where} reads {joined!r}, which no declared format reads")
    except Exception as exc:                                # design._time_parser refuses with its own error type
        _refuse("TIMESTAMP_UNPARSEABLE", f"{where} reads {joined!r}: {exc}")
    return _aware(moment, zone, where=where, zone_declared=zone_declared)


#: every role a calendar column can play. `event` and `event_time` take a list of column names (a label may be spread
#: over a country and a description, an instant over a date and a time); the rest take one name or none.
MAPPING_ROLES = ("event", "event_time", "published", "consensus_published", "received", "actual", "consensus",
                 "previous", "unit", "period", "availability")


class CalendarMapping:
    """Which of the caller's columns mean what. Every one of them is declared; none is inferred from a file name."""

    def __init__(self, **names):
        unknown = sorted(set(names) - set(MAPPING_ROLES))
        if unknown:
            _refuse("UNKNOWN_MAPPING_ROLE", f"{unknown} is not a calendar column role; the roles are "
                                            f"{list(MAPPING_ROLES)}")
        for role in MAPPING_ROLES:
            setattr(self, role, names.get(role))


def read_calendar(path, mapping, *, timezone_name="UTC", timezone_declared=False, columns=None,
                  historical_availability=None, publication_clock="observed",
                  assumed_tolerance_seconds=DEFAULT_ASSUMED_TOLERANCE_SECONDS):
    """The releases, as the arrival store's own arrivals. Nothing is computed here; the module does the arithmetic."""
    zone = _zone(timezone_name)
    header, rows = _calendar_rows(path, columns=columns)
    event_columns = [name for name in (mapping.event or []) if name]
    if not event_columns:
        event_columns = [_pick(header, None, EVENT_COLUMN_NAMES, what="the calendar's event-type column")]
    for name in event_columns:
        if name not in header:
            _refuse("COLUMN_NOT_IN_DATASET", f"the event-type column {name!r} is not among {header}")
    time_columns = list(mapping.event_time or [])
    if not time_columns:
        time_columns = [_pick(header, None, EVENT_TIME_COLUMN_NAMES, what="the calendar's event-time column")]
    for name in time_columns:
        if name not in header:
            _refuse("COLUMN_NOT_IN_DATASET", f"the event-time column {name!r} is not among {header}")
    actual_column = _pick(header, mapping.actual, ACTUAL_COLUMN_NAMES, what="the calendar's actual column")
    consensus_column = _pick(header, mapping.consensus, CONSENSUS_COLUMN_NAMES,
                             what="the calendar's consensus column")
    resolved = {}
    lowered = {name.lower(): name for name in header}
    for optional, candidates in OPTIONAL_COLUMN_NAMES.items():
        name = getattr(mapping, optional, None)
        if name is not None:
            if name not in header:
                _refuse("COLUMN_NOT_IN_DATASET", f"the {optional} column {name!r} is not among {header}")
        else:
            name = next((lowered[c] for c in candidates if c in lowered), None)
        resolved[optional] = name
    mapping = CalendarMapping(event=event_columns, event_time=time_columns, actual=actual_column,
                              consensus=consensus_column, **resolved)
    if mapping.availability is None and historical_availability is None:
        _refuse("AVAILABILITY_NOT_DECLARED",
                "this calendar does not say whether its rows were historically available at their timestamps, and "
                "nothing here will assume it: declare --historical-availability, or name the column that carries it")
    if historical_availability is not None and historical_availability not in AVAILABILITY:
        _refuse("AVAILABILITY_MUST_BE_DECLARED", f"{historical_availability!r} is not one of {list(AVAILABILITY)}")
    if publication_clock not in PUBLICATION_CLOCK_MODES:
        _refuse("UNKNOWN_PUBLICATION_CLOCK_MODE",
                f"{publication_clock!r} is not one of {list(PUBLICATION_CLOCK_MODES)}")
    assumed = publication_clock == "scheduled"
    if assumed:
        if int(assumed_tolerance_seconds) <= 0:
            _refuse("BAD_ASSUMED_TOLERANCE",
                    f"the consensus must be assumed to stand some positive time before the release, got "
                    f"{assumed_tolerance_seconds}")
        observed = [role for role in ("published", "consensus_published") if getattr(mapping, role) is not None]
        if observed:
            _refuse("PUBLICATION_CLOCK_CONFLICT",
                    f"this calendar declares an observed publication clock in {observed}, and the scheduled instant "
                    f"was asked to stand in for it; an assumption does not overwrite a measurement -- drop the "
                    f"assumption, or drop the column")
    tolerance = timedelta(seconds=int(assumed_tolerance_seconds)) if assumed else None

    releases = []
    unit_declared = mapping.unit is not None
    period_declared = mapping.period is not None
    for row in rows:
        number = row["row_number"]
        event_type = " | ".join((_cell(row, name) or "") for name in event_columns).strip(" |")
        if not event_type:
            _refuse("EVENT_TYPE_MISSING", f"calendar row {number} has no event-type label in {event_columns}")
        event_time = _parse_instant([_cell(row, name) or "" for name in time_columns], zone,
                                    where=f"calendar row {number} event time", zone_declared=timezone_declared)
        if event_time is None:
            _refuse("EVENT_TIME_MISSING", f"calendar row {number} has no event time in {time_columns}")
        published = _parse_instant([_cell(row, mapping.published) or ""], zone,
                                   where=f"calendar row {number} published_at", zone_declared=timezone_declared)
        consensus_published = _parse_instant([_cell(row, mapping.consensus_published) or ""], zone,
                                             where=f"calendar row {number} consensus published_at",
                                             zone_declared=timezone_declared)
        received = _parse_instant([_cell(row, mapping.received) or ""], zone,
                                  where=f"calendar row {number} received_at", zone_declared=timezone_declared)
        actual = _float_cell(row, actual_column, where=f"calendar row {number} actual")
        consensus = _float_cell(row, consensus_column, where=f"calendar row {number} consensus")
        previous = _float_cell(row, mapping.previous, where=f"calendar row {number} previous")
        availability = _cell(row, mapping.availability) or historical_availability
        if availability not in AVAILABILITY:
            _refuse("AVAILABILITY_MUST_BE_DECLARED",
                    f"calendar row {number} declares historical availability {availability!r}, which is not one of "
                    f"{list(AVAILABILITY)}")
        unit = _cell(row, mapping.unit) if unit_declared else UNDECLARED
        period = _cell(row, mapping.period) if period_declared else UNDECLARED
        if actual is None:
            continue                                        # a scheduled event with no number is not a release yet
        if assumed:
            # the ONE place the assumption enters. It moves no value: it declares the instants the dataset never did,
            # and every artifact downstream carries the block that says so.
            published, consensus_published = event_time, event_time - tolerance
        arrivals = []
        # one event_key per RELEASE, not per event type: the arrival store models one scheduled release with its own
        # consensus, actual and revisions, and monthly releases of the same indicator are different events in it.
        anchor = published or received or event_time
        event_key = f"{event_type}@{anchor.isoformat()}#{number}"
        base = {"schema": ARRIVAL_SCHEMA, "event_key": event_key, "event_time": event_time,
                "historical_availability": availability, "unit": unit or UNDECLARED, "period": period or UNDECLARED}
        if consensus is not None:
            consensus_seen = received or consensus_published or published or event_time
            if consensus_published is not None and consensus_published > consensus_seen:
                consensus_seen = consensus_published
            if assumed and received is not None and received < consensus_published:
                consensus_seen = consensus_published      # our receipt cannot precede an instant we declared
            arrival = {**base, "kind": "CONSENSUS", "consensus": consensus, "observed_at": consensus_seen}
            if consensus_published is not None:
                arrival["published_at"] = consensus_published
            arrivals.append(arrival)
        actual_seen = received or published or event_time
        if assumed and received is not None and received < published:
            actual_seen = published
        arrival = {**base, "kind": "ACTUAL", "actual": actual, "observed_at": actual_seen}
        if published is not None:
            arrival["published_at"] = published
        arrivals.append(arrival)
        releases.append({"row_number": number, "event_key": event_key, "event_type": event_type,
                         "event_time": event_time, "published_at": published, "as_of": actual_seen,
                         "actual": actual, "consensus": consensus, "previous": previous,
                         "unit": unit or UNDECLARED, "period": period or UNDECLARED, "arrivals": arrivals})
    # ordered by the SOURCE's clock where it exists, because that is the order the scale's history is read in; our
    # receipt orders only the rows whose source never declared one, and those can carry no release surprise anyway
    releases.sort(key=lambda r: (r["published_at"] or r["as_of"], r["as_of"], r["event_key"]))
    return {"path": str(path), "sha256": _file_digest(path), "columns": header, "rows_read": len(rows),
            "releases_with_an_actual": len(releases), "event_columns": event_columns,
            "event_time_columns": time_columns, "actual_column": actual_column,
            "consensus_column": consensus_column,
            "published_column": mapping.published, "consensus_published_column": mapping.consensus_published,
            "received_column": mapping.received, "previous_column": mapping.previous,
            "unit_column": mapping.unit, "period_column": mapping.period,
            "availability_column": mapping.availability, "historical_availability": historical_availability,
            "timezone": timezone_name, "timezone_declared": bool(timezone_declared),
            "publication_clock": publication_clock_block(publication_clock, assumed_tolerance_seconds),
            "units_declared": unit_declared and period_declared,
            "units_reading": ("the dataset declares unit and period, so the calendar module's INCOMPARABLE_SERIES "
                              "check is live" if unit_declared and period_declared else
                              f"the dataset declares no unit or period, so every arrival carries the constant "
                              f"{UNDECLARED!r}; the calendar module's INCOMPARABLE_SERIES check cannot fire, and "
                              f"nobody should read its silence as a comparability finding"),
            "releases": releases}


# ------------------------------------------------------------------------------------------- the paths and scales

class _Excluded:
    """Exact counts, capped examples. The count is the finding; the examples are there to be read."""

    def __init__(self):
        self.counts = {code: 0 for code in EXCLUSION_CODES}
        self.releases, self.event_horizons = [], []
        self.truncated = {code: 0 for code in EXCLUSION_CODES}

    def release(self, code, release, why):
        self.counts[code] += 1
        if len(self.releases) < MAX_EXCLUDED_DETAIL:
            self.releases.append({"event_key": release["event_key"], "event_type": release["event_type"],
                                  "as_of": release["as_of"].isoformat(), "code": code, "why": why})
        else:
            self.truncated[code] += 1

    def event_horizon(self, code, release, horizon, why):
        self.counts[code] += 1
        if len(self.event_horizons) < MAX_EXCLUDED_DETAIL:
            self.event_horizons.append({"event_key": release["event_key"], "event_type": release["event_type"],
                                        "as_of": release["as_of"].isoformat(), "horizon_minutes": horizon,
                                        "code": code, "why": why})
        else:
            self.truncated[code] += 1

    def document(self):
        return {"counts": self.counts,
                "counts_reading": ("MISSING_PUBLICATION_CLOCK, NO_CONSENSUS, INSUFFICIENT_HISTORY and "
                                   "NON_POSITIVE_RESIDUAL_SCALE count RELEASES; BARS_MISSING_AT_HORIZON counts "
                                   "(release, horizon) pairs, because a gap refuses one horizon and not the release"),
                "releases": self.releases, "event_horizons": self.event_horizons,
                "examples_truncated_at": MAX_EXCLUDED_DETAIL,
                "examples_not_listed": {code: n for code, n in self.truncated.items() if n}}


def _realized(log_price, start, end):
    """Sum of squared one-step log returns over bars `start..end`. The step is the bars' own and is declared."""
    if end <= start:
        return 0.0
    diffs = np.diff(log_price[start:end + 1])
    return float(np.dot(diffs, diffs))


def _contiguous(times, start, end, step):
    """True when every step between `start` and `end` is exactly the grid's. A gap here is an unobserved path."""
    if start < 0 or end >= times.size or end < start:
        return False
    if end == start:
        return True
    deltas = np.round(np.diff(times[start:end + 1])).astype(np.int64)
    return bool(np.all(deltas == step))


def build(bars_path, calendar_path, mapping, *, horizons_minutes=DEFAULT_HORIZONS_MINUTES,
          window_hours=DEFAULT_WINDOW_HOURS, pre_event_minutes=DEFAULT_PRE_EVENT_MINUTES,
          min_prior_releases=DEFAULT_MIN_PRIOR_RELEASES, bars_time_column=None, bars_price_column=None,
          bars_timezone="UTC", bars_timezone_declared=False, calendar_timezone="UTC",
          calendar_timezone_declared=False, calendar_columns=None, historical_availability=None, max_bars=None,
          publication_clock="observed", assumed_tolerance_seconds=DEFAULT_ASSUMED_TOLERANCE_SECONDS,
          max_neighbours_listed=None):
    """One row per (release, horizon), with the releases that did not become rows counted by name."""
    horizons = sorted({int(h) for h in horizons_minutes})
    if not horizons or horizons[0] <= 0:
        _refuse("BAD_HORIZONS", f"the horizons must be positive minutes, got {list(horizons_minutes)}")
    if window_hours <= 0:
        _refuse("BAD_WINDOW", f"the window must be positive hours, got {window_hours}")
    if pre_event_minutes <= 0:
        _refuse("BAD_PRE_EVENT_SPAN", f"the pre-event span must be positive minutes, got {pre_event_minutes}")
    if int(min_prior_releases) < 2:
        _refuse("BAD_MIN_HISTORY",
                f"a dispersion over fewer than two prior releases is not a dispersion, got {min_prior_releases}")
    min_prior_releases = int(min_prior_releases)

    bars = read_bars(bars_path, time_column=bars_time_column, price_column=bars_price_column,
                     timezone_name=bars_timezone, timezone_declared=bars_timezone_declared, max_rows=max_bars)
    step = bars["step_seconds"]
    off_grid_horizons = [h for h in horizons if (h * 60) % step]
    if off_grid_horizons:
        _refuse("HORIZON_OFF_THE_BAR_GRID",
                f"the bars sit on a {step}-second grid and the horizon(s) {off_grid_horizons} minute(s) are not a "
                f"whole number of bars; a path cannot end between two bars and nothing here interpolates one")
    if (pre_event_minutes * 60) % step:
        _refuse("PRE_EVENT_SPAN_OFF_THE_BAR_GRID",
                f"the bars sit on a {step}-second grid and the pre-event span of {pre_event_minutes} minute(s) is not "
                f"a whole number of bars")
    times, log_price = bars["times"], bars["log_price"]

    if max_neighbours_listed is not None and int(max_neighbours_listed) < 0:
        _refuse("BAD_NEIGHBOUR_CAP", f"the neighbour listing cap must not be negative, got {max_neighbours_listed}")
    calendar = read_calendar(calendar_path, mapping, timezone_name=calendar_timezone,
                             timezone_declared=calendar_timezone_declared, columns=calendar_columns,
                             historical_availability=historical_availability, publication_clock=publication_clock,
                             assumed_tolerance_seconds=assumed_tolerance_seconds)
    clock = calendar["publication_clock"]
    provenance = PROVENANCE[publication_clock]
    releases = calendar.pop("releases")

    excluded = _Excluded()
    # pass 1: the surprise of every release, from the calendar module's own release boundary, and the scale that the
    # information published before it -- and nothing else -- supports.
    history, computed = {}, []
    for release in releases:
        # one arrival store PER RELEASE. `event_key` is unique per release, and every view the module offers filters
        # by it, so a per-release store answers exactly what one shared store would -- while a shared store rescans
        # every arrival of the whole archive for every release, which on a hundred thousand releases is quadratic.
        store = PointInTimeCalendar()
        try:
            store.add_all(release.pop("arrivals"))
            answer = store.surprise(release["event_key"], release["as_of"])
        except CalendarRefusal as refusal:
            _refuse("CALENDAR_REFUSED", f"{release['event_key']}: {refusal}")
        raw = answer.get("release_surprise")
        record = {**release, "release_surprise": raw, "standardized": None, "scale": None, "scale_n": 0,
                  "status": "OK", "why": None,
                  "release_consensus": answer.get("release_consensus"),
                  "release_consensus_published_at": answer.get("release_consensus_published_at"),
                  "release_actual_published_at": answer.get("release_actual_published_at")}
        if answer.get("release_actual") is None:
            record["status"], record["why"] = "MISSING_PUBLICATION_CLOCK", answer.get("release_reason")
        elif raw is None:
            record["status"], record["why"] = "NO_CONSENSUS", answer.get("release_reason")
        else:
            published_epoch = datetime.fromisoformat(answer["release_actual_published_at"]).timestamp()
            # STRICTLY before: a release published at the same instant is not information anyone had beforehand
            prior = [value for moment, value in history.get(release["event_type"], []) if moment < published_epoch]
            record["scale_n"] = len(prior)
            if len(prior) < min_prior_releases:
                record["status"] = "INSUFFICIENT_HISTORY"
                record["why"] = (f"{len(prior)} release(s) of {release['event_type']!r} had been published before "
                                 f"this one, and {min_prior_releases} are declared as the fewest that make a "
                                 f"dispersion mean anything; standardizing by a scale estimated from fewer would put "
                                 f"an invented number where a missing one belongs")
            else:
                scale = float(np.std(np.asarray(prior, dtype=np.float64), ddof=1))
                if not math.isfinite(scale) or scale <= 0:
                    record["status"] = "NON_POSITIVE_RESIDUAL_SCALE"
                    record["why"] = (f"the {len(prior)} surprise(s) of {release['event_type']!r} published before "
                                     f"this one have dispersion {scale}; dividing by it would be infinite or "
                                     f"negative, and neither is a standardized surprise")
                else:
                    standardized = store.surprise(release["event_key"], release["as_of"],
                                                  scale=scale)["standardized"]
                    record["scale"], record["standardized"] = scale, standardized
            history.setdefault(release["event_type"], []).append((published_epoch, raw))
        computed.append(record)

    for record in computed:
        if record["status"] != "OK":
            excluded.release(record["status"], record, record["why"])

    # pass 2: the neighbours. Every release whose SOURCE published it has an instant a window can be measured from;
    # a release without one has no place on the offset axis and is not listed as a neighbour.
    placed = [r for r in computed if r["release_actual_published_at"] is not None]
    placed.sort(key=lambda r: r["release_actual_published_at"])
    placed_seconds = np.asarray([datetime.fromisoformat(r["release_actual_published_at"]).timestamp()
                                 for r in placed], dtype=np.float64)
    window_seconds = float(window_hours) * 3600.0

    rows = []
    for record in computed:
        if record["status"] != "OK":
            continue
        t_k = datetime.fromisoformat(record["release_actual_published_at"])
        t_epoch = t_k.timestamp()
        anchor = int(np.searchsorted(times, t_epoch, side="right")) - 1
        anchor_ok = anchor >= 0 and (t_epoch - float(times[anchor])) < step
        pre_start = anchor - (pre_event_minutes * 60) // step
        if anchor_ok and _contiguous(times, int(pre_start), anchor, step):
            pre_vol, pre_status = _realized(log_price, int(pre_start), anchor), "OK"
            pre_from = datetime.fromtimestamp(float(times[int(pre_start)]), _timezone.utc).isoformat()
        else:
            pre_vol, pre_from = None, None
            pre_status = ("BARS_MISSING_BEFORE_EVENT: the bars do not cover the declared pre-event span without a "
                          "gap, so the volatility that was already there was not observed and is not reported")
        left = int(np.searchsorted(placed_seconds, t_epoch - window_seconds, side="left"))
        right = int(np.searchsorted(placed_seconds, t_epoch + window_seconds, side="right"))
        neighbours = []
        for other in placed[left:right]:
            if other["event_key"] == record["event_key"]:
                continue
            other_epoch = datetime.fromisoformat(other["release_actual_published_at"]).timestamp()
            neighbours.append({"event_type": other["event_type"], "event_key": other["event_key"],
                               "published_at": other["release_actual_published_at"],
                               "offset_minutes": (other_epoch - t_epoch) / 60.0,
                               "surprise": other["standardized"], "surprise_raw": other["release_surprise"],
                               "surprise_status": other["status"]})
        neighbours.sort(key=lambda n: (n["offset_minutes"], n["event_key"]))
        neighbours_total = len(neighbours)
        if max_neighbours_listed is not None and neighbours_total > int(max_neighbours_listed):
            # the nearest by absolute offset survive, and the COUNT above is the exact one either way: a listing that
            # was cut short says so, and nobody reads its length as the number of releases in the window
            nearest = sorted(neighbours, key=lambda n: (abs(n["offset_minutes"]), n["event_key"]))
            keep = {n["event_key"] for n in nearest[:int(max_neighbours_listed)]}
            neighbours = [n for n in neighbours if n["event_key"] in keep]
        for horizon in horizons:
            if not anchor_ok:
                excluded.event_horizon("BARS_MISSING_AT_HORIZON", record, horizon,
                                       "no bar sits at or within one sampling step before the release instant; the "
                                       "release fell into a gap in the series and its path was not observed")
                continue
            end = anchor + (horizon * 60) // step
            if not _contiguous(times, anchor, int(end), step):
                excluded.event_horizon("BARS_MISSING_AT_HORIZON", record, horizon,
                                       f"the bars between the release instant and +{horizon} minute(s) are not "
                                       f"contiguous on the {step}-second grid; interpolating across the gap would "
                                       f"invent the movement this row is meant to measure")
                continue
            end = int(end)
            rows.append({
                "event_key": record["event_key"], "event_type": record["event_type"],
                "published_at": record["release_actual_published_at"],
                "event_time": record["event_time"].isoformat(),
                "horizon_minutes": horizon,
                "surprise": record["standardized"],
                "surprise_raw": record["release_surprise"],
                "surprise_scale": record["scale"], "surprise_scale_n": record["scale_n"],
                "surprise_boundary": "release",
                "consensus": record["release_consensus"],
                "consensus_published_at": record["release_consensus_published_at"],
                "actual": record["actual"], "previous": record["previous"],
                "unit": record["unit"], "period": record["period"],
                "anchor_time": datetime.fromtimestamp(float(times[anchor]), _timezone.utc).isoformat(),
                "anchor_log_price": float(log_price[anchor]),
                "end_time": datetime.fromtimestamp(float(times[end]), _timezone.utc).isoformat(),
                "log_return": float(log_price[end] - log_price[anchor]),
                "realized_vol": _realized(log_price, anchor, end),
                "pre_event_realized_vol": pre_vol, "pre_event_minutes": int(pre_event_minutes),
                "pre_event_from": pre_from, "pre_event_status": pre_status,
                "hour_of_day": t_k.astimezone(_timezone.utc).hour,
                "day_of_week": t_k.astimezone(_timezone.utc).weekday(),
                "other_releases_in_window": neighbours,
                "other_releases_in_window_count": neighbours_total,
                "other_releases_in_window_listed": len(neighbours),
                "publication_clock_mode": clock["mode"],
                "provenance": provenance,
            })
    rows.sort(key=lambda r: (r["published_at"], r["event_key"], r["horizon_minutes"]))

    by_type = {}
    for row in rows:
        entry = by_type.setdefault(row["event_type"], {"rows": 0, "releases": set(), "horizons": {}})
        entry["rows"] += 1
        entry["releases"].add(row["event_key"])
        entry["horizons"][str(row["horizon_minutes"])] = entry["horizons"].get(str(row["horizon_minutes"]), 0) + 1
    counts = {name: {"rows": entry["rows"], "releases": len(entry["releases"]),
                     "rows_by_horizon_minutes": dict(sorted(entry["horizons"].items(), key=lambda kv: int(kv[0])))}
              for name, entry in sorted(by_type.items())}

    return {
        "schema": SCHEMA,
        "provenance": provenance,
        "publication_clock": clock,
        "parameters": {
            "horizons_minutes": horizons,
            "window_hours": float(window_hours),
            "pre_event_minutes": int(pre_event_minutes),
            "min_prior_releases": min_prior_releases,
            "max_neighbours_listed": None if max_neighbours_listed is None else int(max_neighbours_listed),
            "realized_vol_step_seconds": step,
            "realized_vol": f"the sum of squared log returns over consecutive {step}-second bars inside the horizon",
            "surprise": ("app/economic_calendar.py release_surprise, standardized by the dispersion of that event "
                         "type's surprises over releases PUBLISHED STRICTLY BEFORE this one (sample standard "
                         "deviation, ddof=1); no release, and no part of any release, from at or after the instant "
                         "enters the scale"),
            "day_of_week": "Monday is 0, in UTC",
            "hour_of_day": "in UTC",
        },
        "bars": {key: value for key, value in bars.items() if key not in ("times", "log_price")},
        "calendar": calendar,
        "rows": rows,
        "counts": {"rows": len(rows), "releases_with_a_row": len({row["event_key"] for row in rows}),
                   "releases_read": len(computed), "by_event_type": counts},
        "excluded": excluded.document(),
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]), "numpy": np.__version__},
        "fitted": "NOTHING: this job writes the rows an event study is estimated from; no model is fitted here",
        "reading": (f"PROVENANCE {provenance}, publication clock {clock['mode']}. {clock['identification_caveat']}. "
                    "one row per (release, horizon). `surprise` is a standardized release surprise and `log_return` "
                    "and `realized_vol` are the paths that followed it -- an association these rows make measurable, "
                    "not an effect. `other_releases_in_window` with a POSITIVE offset landed AFTER this release and "
                    "is therefore not pre-release information: it is listed so a later stage can control for it or "
                    "exclude it, never as a control that was knowable at the instant"),
    }


# --------------------------------------------------------------------------------------------------------- the CLI

def _split(value):
    return [part.strip() for part in value.split(",") if part.strip()] if value else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.events",
        description="Build the event-window rows an event study is estimated from. No model is fitted.")
    parser.add_argument("--bars", required=True, help="the price bars (CSV, or parquet when pandas is installed)")
    parser.add_argument("--calendar", required=True, help="the economic calendar (CSV)")
    parser.add_argument("--out", help="where to write the rows document; stdout when absent")
    parser.add_argument("--bars-time-column")
    parser.add_argument("--bars-price-column")
    parser.add_argument("--bars-timezone", help="the IANA zone naive bar timestamps are read in; declared, never "
                                                "inferred")
    parser.add_argument("--max-bars", type=int, help="read at most this many bars")
    parser.add_argument("--calendar-columns", help="comma-separated column names for a calendar with no header row")
    parser.add_argument("--calendar-event-column", help="comma-separated column(s) forming the event-type label")
    parser.add_argument("--calendar-event-time-columns", help="comma-separated column(s) forming the event instant")
    parser.add_argument("--calendar-published-column", help="the instant the SOURCE published the actual")
    parser.add_argument("--calendar-consensus-published-column", help="the instant the SOURCE published the consensus")
    parser.add_argument("--calendar-received-column", help="the instant the row reached this system")
    parser.add_argument("--calendar-actual-column")
    parser.add_argument("--calendar-consensus-column")
    parser.add_argument("--calendar-previous-column")
    parser.add_argument("--calendar-unit-column")
    parser.add_argument("--calendar-period-column")
    parser.add_argument("--calendar-availability-column")
    parser.add_argument("--calendar-timezone", help="the IANA zone naive calendar timestamps are read in")
    parser.add_argument("--historical-availability", choices=list(AVAILABILITY),
                        help="declare it for every row when the dataset carries no column for it")
    parser.add_argument("--publication-clock", choices=list(PUBLICATION_CLOCK_MODES), default="observed",
                        help="'observed' reads the instants the dataset declares and excludes a release that "
                             "declares none; 'scheduled' DECLARES the assumption that each release was published at "
                             "the instant it was scheduled for, and stamps every artifact with it")
    parser.add_argument("--assume-publication-tolerance-seconds", type=int,
                        default=DEFAULT_ASSUMED_TOLERANCE_SECONDS,
                        help="how long before the scheduled instant the consensus is assumed to have stood; only "
                             "meaningful with --publication-clock scheduled")
    parser.add_argument("--max-neighbours-listed", type=int,
                        help="list at most this many of the other releases in the window, nearest first; the count "
                             "reported on every row stays exact")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS_MINUTES),
                        help="the horizons in minutes")
    parser.add_argument("--window-hours", type=float, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--pre-event-minutes", type=int, default=DEFAULT_PRE_EVENT_MINUTES)
    parser.add_argument("--min-prior-releases", type=int, default=DEFAULT_MIN_PRIOR_RELEASES)
    args = parser.parse_args(argv)
    mapping = CalendarMapping(event=_split(args.calendar_event_column),
                              event_time=_split(args.calendar_event_time_columns),
                              published=args.calendar_published_column,
                              consensus_published=args.calendar_consensus_published_column,
                              received=args.calendar_received_column,
                              actual=args.calendar_actual_column, consensus=args.calendar_consensus_column,
                              previous=args.calendar_previous_column, unit=args.calendar_unit_column,
                              period=args.calendar_period_column, availability=args.calendar_availability_column)
    try:
        document = build(args.bars, args.calendar, mapping, horizons_minutes=args.horizons,
                         window_hours=args.window_hours, pre_event_minutes=args.pre_event_minutes,
                         min_prior_releases=args.min_prior_releases, bars_time_column=args.bars_time_column,
                         bars_price_column=args.bars_price_column,
                         bars_timezone=args.bars_timezone or "UTC",
                         bars_timezone_declared=args.bars_timezone is not None,
                         calendar_timezone=args.calendar_timezone or "UTC",
                         calendar_timezone_declared=args.calendar_timezone is not None,
                         calendar_columns=_split(args.calendar_columns),
                         historical_availability=args.historical_availability, max_bars=args.max_bars,
                         publication_clock=args.publication_clock,
                         assumed_tolerance_seconds=args.assume_publication_tolerance_seconds,
                         max_neighbours_listed=args.max_neighbours_listed)
    except (EventsRefusal, CalendarRefusal) as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=False, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"{len(document['rows'])} row(s) written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
