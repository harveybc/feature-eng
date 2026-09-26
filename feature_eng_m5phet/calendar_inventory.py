"""Inventory the economic-calendar resources this machine holds: what each one carries, and what it does NOT.

RP150 asks for the governed economic dataset's inventory before any as-of transform is trusted: which resource, which
columns and units, which timezone, which vintages exist, which fields are absent. Every line this module prints is a
MEASUREMENT on named bytes, not a reading of a README. The READMEs are quoted separately, and where a README and the
bytes disagree the bytes are reported.

The reason it exists as code rather than as a document is CAL09 and the vintage question. A point-in-time calendar can
only be built from a resource that keeps MORE THAN ONE version of a field -- a first actual and then its revisions, a
consensus that moved -- and whether a file does that is not a matter of description. It is a count: group the rows by
(series, reference period) and see whether any key carries two rows with two different values. A file with one row per
release has no vintages, whatever its provenance says, and `app.economic_calendar` must then be fed a single arrival
per event with `historical_availability` declared by hand.

What is measured per resource
-----------------------------
* identity: path, size, sha256 of the exact bytes read;
* shape: rows, and every column with the kind its values actually are;
* the FIELD ROLES the CAL cases need, each PRESENT (with the column that carries it and how many rows are non-null) or
  ABSENT by name. The role vocabulary is `CAL_FIELD_ROLES`, and each role names the CAL cases that cannot be evaluated
  without it -- so a skipped acceptance test can say which field is missing and why;
* the clock: whether the instant column is timezone-aware, and for a naive one that it is naive (what its wall clock
  MEANS is `calendar_clock`'s measurement, not this module's guess);
* units: the distinct values of whatever column carries a unit, counted, because a "surprise" across two of them is
  arithmetic and not economics;
* vintages: keys, keys with more than one row, keys whose rows disagree about the actual, and the largest number of
  distinct values any one key carries -- measured over a LADDER of keys from coarse to as fine as the resource allows.
  Two rows disagreeing on a coarse key need not be a revision: in the 2011-2021 archive 10,451 keys disagree on
  (country, description, event_date) and 9 still disagree once every other column is in the key, so all but nine of
  them were one key failing to name a release, not a value that changed. So the verdict is deliberately three-valued.
  Zero disagreements at the finest key is `NO_VINTAGES`. Disagreements that
  survive it are `VINTAGE_UNDECIDABLE` whenever the resource carries neither an observation clock nor a version field,
  because nothing in the bytes can then tell a revision from two series sharing a key -- and calling that a vintage
  would be exactly the invention this whole contract exists to prevent.

Nothing here reads the network and nothing here is a model. Deterministic, CPU only, one pass per file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "m5phet.calendar_inventory.v1"

#: Every field role a CAL01-CAL12 case can need, and the cases that cannot be evaluated without it. A role absent from
#: a resource is not a defect of this module: it is the fact that decides whether an acceptance case is runnable
#: against that resource, and `tests/test_cal01_cal12_governed.py` skips BY NAME quoting the role it did not find.
CAL_FIELD_ROLES = {
    "schedule_instant": ("CAL01", "CAL11"),
    "consensus": ("CAL03", "CAL06"),
    "actual": ("CAL02", "CAL03", "CAL04", "CAL10"),
    "previous": (),
    "revision_marker": ("CAL04", "CAL10"),
    "publication_instant": ("CAL02", "CAL03", "CAL10"),
    "receipt_instant": ("CAL02", "CAL07", "CAL08", "CAL12"),
    "unit": ("CAL05",),
    "reference_period": ("CAL05",),
    "observed_sequence": ("CAL10",),
    "historical_availability": ("CAL09",),
    "vintage_version": ("CAL04", "CAL07", "CAL08"),
    "cancellation_state": ("CAL11",),
}

#: For each resource, which of its OWN columns carries each role. Declared here, per resource, from its data
#: dictionary and its bytes -- never inferred from a column name resembling a familiar one, which is the mistake that
#: turns a scheduled date into a publication instant.
_ARCHIVE_2011_2021_COLUMNS = ("event_date", "event_time", "country", "volatility", "description", "evaluation",
                              "data_format", "actual", "forecast", "previous")

RESOURCES = {
    "archive_2011_2021": {
        "path": "feature-eng/tests/data/economic_calendar_2011_2021.csv",
        "kind": "csv_no_header",
        "columns": _ARCHIVE_2011_2021_COLUMNS,
        "roles": {"consensus": "forecast", "actual": "actual", "previous": "previous",
                  "unit": "data_format", "schedule_instant": "event_date+event_time"},
        "series_key": ("country", "description"),
        "period_key": ("event_date",),
        "key_ladder": (("country", "description", "event_date"),
                       ("country", "description", "event_date", "event_time"),
                       ("country", "description", "event_date", "event_time", "data_format")),
        "value_column": "actual",
        "clock_column": "event_date+event_time",
        "clock_is_tz_aware": False,
    },
    "fxmacrodata_announcements": {
        "path": "financial-data/economic_calendar/release_actuals/fxmacrodata/announcements.parquet",
        "kind": "parquet",
        "roles": {"actual": "val", "publication_instant": "announcement_datetime_utc",
                  "reference_period": "date"},
        "series_key": ("currency", "indicator"),
        "period_key": ("date",),
        "key_ladder": (("currency", "indicator", "date"),
                       ("currency", "indicator", "date", "announcement_datetime_utc")),
        "value_column": "val",
        "clock_column": "announcement_datetime_utc",
    },
    "fxmacrodata_release_calendar": {
        "path": "financial-data/economic_calendar/scheduled_events/fxmacrodata/release_calendar.parquet",
        "kind": "parquet",
        "roles": {"schedule_instant": "announcement_datetime_utc"},
        "series_key": ("currency", "release"),
        "period_key": ("announcement_datetime_utc",),
        "value_column": None,
        "clock_column": "announcement_datetime_utc",
    },
    "fred_release_date_proxy": {
        "path": "financial-data/economic_calendar/scheduled_events/fred_release_date_proxy/scheduled_events.parquet",
        "kind": "parquet",
        "roles": {"actual": "actual", "consensus": "consensus_estimate",
                  "schedule_instant": "scheduled_date_proxy"},
        "series_key": ("event_slug",),
        "period_key": ("scheduled_date_proxy",),
        "value_column": "actual",
        "clock_column": "scheduled_date_proxy",
        "clock_is_tz_aware": False,
    },
    "fred_cpi_yoy_actuals": {
        "path": "financial-data/economic_calendar/release_actuals/cpi_yoy/actuals.parquet",
        "kind": "parquet",
        "roles": {"actual": "actual", "consensus": "consensus_estimate", "reference_period": "date",
                  "unit": "transform"},
        "series_key": ("fred_series",),
        "period_key": ("date",),
        "value_column": "actual",
        "clock_column": "date",
        "clock_is_tz_aware": False,
    },
}


class InventoryRefusal(RuntimeError):
    """A measurement that could not be made, named. No count is estimated to avoid one."""


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _provenance(path, measured_sha256):
    """The resource's own `provenance.json`, and whether its declared digest is the digest of these bytes.

    `acquired_at` is the ONLY receipt clock any of these resources carries, and it is at FILE grain: it says when the
    whole download reached this machine, not when any one release did. That distinction is the difference between an
    as-of view per release and a single as-of view for the entire file, so it is recorded as its own field and never
    mistaken for `receipt_instant`.
    """
    sidecar = Path(path).parent / "provenance.json"
    if not sidecar.is_file():
        return {"present": False, "reading": f"no provenance.json beside {Path(path).name}"}
    declared = json.loads(sidecar.read_text(encoding="utf-8"))
    entries = [f for f in declared.get("files", []) if Path(str(f.get("path", ""))).name == Path(path).name]
    digests = [f.get("sha256") for f in entries]
    return {"present": True, "source": declared.get("source"),
            "file_grain_receipt_instant": declared.get("acquired_at"),
            "declared_sha256": digests[0] if digests else None,
            "declared_digest_matches_these_bytes": bool(digests) and digests[0] == measured_sha256,
            "reading": ("`acquired_at` is a FILE-grain receipt clock: it dates the download, not any single release, so "
                        "it cannot order two releases inside the file and is never read as `receipt_instant`")}


def _kind_of(values):
    """What a column's values ARE, over the rows read: the set of Python type names, and how many are missing."""
    names, missing = Counter(), 0
    for value in values:
        if value is None or value != value or value == "":      # NaN != NaN; an empty cell is missing, not a string
            missing += 1
            continue
        names[type(value).__name__] += 1
    return {"types": dict(sorted(names.items())), "non_null": sum(names.values()), "null_or_empty": missing}


def _read_csv_no_header(path, columns):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(columns):
                raise InventoryRefusal(
                    f"RAGGED_ROW: {path} row has {len(row)} fields and {len(columns)} names are declared")
            rows.append({name: row[i].strip() for i, name in enumerate(columns)})
    return list(columns), rows


def _read_parquet(path):
    import pandas as pd                                        # imported here: the csv resource needs no pandas
    frame = pd.read_parquet(path)
    columns = [str(c) for c in frame.columns]
    dtypes = {str(c): str(frame[c].dtype) for c in frame.columns}
    return columns, frame, dtypes


def _combined(row, spec):
    """A composite key such as `event_date+event_time`, joined with a single space and nothing invented."""
    return " ".join(str(row[part]) for part in spec.split("+"))


def _collisions(keys_to_values):
    """One rung of the ladder: how many keys carry two rows, and how many carry two DIFFERENT values."""
    repeated = sum(1 for values in keys_to_values.values() if len(values) > 1)
    disagreeing = {key: sorted(set(values)) for key, values in keys_to_values.items() if len(set(values)) > 1}
    widest = max((len(set(v)) for v in keys_to_values.values()), default=0)
    return {"keys": len(keys_to_values),
            "keys_with_more_than_one_row": repeated,
            "keys_whose_rows_disagree_about_the_value": len(disagreeing),
            "most_distinct_values_on_one_key": widest,
            "examples": [{"key": list(k), "values": v} for k, v in list(sorted(disagreeing.items()))[:3]]}


def _vintage_verdict(finest, *, has_observation_clock, has_version_field, finest_key):
    """Three-valued on purpose. A disagreement is only a vintage when something in the bytes can date the two values."""
    if finest["keys_whose_rows_disagree_about_the_value"] == 0:
        return {"verdict": "NO_VINTAGES",
                "reading": (f"at the finest key this resource offers ({', '.join(finest_key)}) every key carries one "
                            "value, so no earlier version of any field survives here and a point-in-time view cannot be "
                            "reconstructed from these bytes")}
    if has_observation_clock or has_version_field:
        return {"verdict": "VINTAGES_PRESENT",
                "reading": ("values disagree on the finest key AND the resource carries something that dates them, so "
                            "the earlier version is recoverable")}
    return {"verdict": "VINTAGE_UNDECIDABLE",
            "reading": (f"{finest['keys_whose_rows_disagree_about_the_value']} keys still carry two different values at "
                        f"the finest key this resource offers ({', '.join(finest_key)}), and the resource carries neither "
                        "an observation clock nor a version field. Nothing in these bytes can tell a REVISION of one "
                        "release from two distinct releases sharing a key, so this is reported as undecidable rather "
                        "than counted as a vintage. How far the count fell between the coarsest and the finest rung is "
                        "the measure of how much of it was key collision")}


def _roles(spec, columns, column_stats):
    out = {}
    for role, cases in CAL_FIELD_ROLES.items():
        column = spec["roles"].get(role)
        if column is None:
            out[role] = {"present": False, "column": None, "non_null_rows": 0,
                         "cases_it_blocks": list(cases)}
            continue
        if "+" not in column and column not in columns:
            raise InventoryRefusal(f"DECLARED_COLUMN_ABSENT: {column!r} is declared for role {role} and is not in {columns}")
        stats = column_stats.get(column, {})
        out[role] = {"present": bool(stats.get("non_null", 0)), "column": column,
                     "non_null_rows": int(stats.get("non_null", 0)),
                     "cases_it_blocks": [] if stats.get("non_null", 0) else list(cases)}
        if column in columns and not stats.get("non_null", 0):
            out[role]["reading"] = ("COLUMN_PRESENT_BUT_EMPTY: the column exists and every row is null, which is the "
                                    "same absence as a missing column for any case that needs a value")
    return out


def measure_resource(name, root):
    spec = RESOURCES[name]
    path = Path(root) / spec["path"]
    if not path.is_file():
        return {"resource": name, "path": str(path), "status": "ABSENT",
                "reading": "the file named is not on this machine; nothing about it is asserted"}
    record = {"resource": name, "path": str(path), "status": "MEASURED",
              "bytes": path.stat().st_size, "sha256": _digest(path)}
    record["provenance"] = _provenance(path, record["sha256"])
    if spec["kind"] == "csv_no_header":
        columns, rows = _read_csv_no_header(path, spec["columns"])
        record["header_row"] = False
        record["column_names_declared_by"] = "feature_eng_m5phet.calendar_join.DEFAULT_ARCHIVE_COLUMNS"
        record["rows"] = len(rows)
        stats = {c: _kind_of([r[c] for r in rows]) for c in columns}
        for composite in {v for v in spec["roles"].values() if "+" in v}:
            stats[composite] = _kind_of([_combined(r, composite) for r in rows])
        getter = lambda row, column: (_combined(row, column) if "+" in column else row[column])
        iter_rows = rows
    else:
        columns, frame, dtypes = _read_parquet(path)
        record["rows"] = int(len(frame))
        record["pandas_dtypes"] = dtypes
        stats = {c: _kind_of(list(frame[c])) for c in columns}
        iter_rows = frame.to_dict("records")
        getter = lambda row, column: row[column]
    record["columns"] = {c: stats[c] for c in columns}
    record["field_roles"] = _roles(spec, columns, stats)

    clock = spec["clock_column"]
    record["clock"] = {"column": clock,
                       "declared_timezone_aware": bool(spec.get("clock_is_tz_aware", "+00:00" in str(
                           record.get("pandas_dtypes", {}).get(clock, "")) or "UTC" in str(
                           record.get("pandas_dtypes", {}).get(clock, ""))))}
    clock_values = [getter(row, clock) for row in iter_rows]
    present = [v for v in clock_values if v is not None and v == v and str(v) != ""]
    record["clock"]["first_and_last_values_in_file_order"] = [str(clock_values[0]), str(clock_values[-1])] if clock_values else []
    # the SPAN is min/max over the column, which is not the first and last rows: these files are not sorted by clock.
    record["clock"]["span"] = [str(min(present)), str(max(present))] if present else []
    # For a TEXT column the order is lexicographic, and the archive's `4:00:00` is not zero-padded, so its span's
    # time-of-day is the largest STRING and not the latest hour. Said here rather than left to be misread.
    record["clock"]["span_comparison"] = ("lexicographic_over_text" if present and isinstance(present[0], str)
                                          else "chronological")
    record["clock"]["values_missing"] = len(clock_values) - len(present)
    record["clock"]["measured_timezone_aware"] = all(
        getattr(v, "tzinfo", None) is not None for v in [getter(row, clock) for row in iter_rows[:200]]) if iter_rows else False
    if not record["clock"]["measured_timezone_aware"]:
        record["clock"]["reading"] = (
            "NAIVE_WALL_CLOCK: these values name no instant on their own. What this archive's wall clock means is "
            "measured by feature_eng_m5phet.calendar_clock, not declared here, and app.economic_calendar refuses a "
            "naive timestamp at ingestion (AMBIGUOUS_LOCAL_TIME)")

    unit_column = spec["roles"].get("unit")
    if unit_column and unit_column in columns:
        units = Counter(str(getter(row, unit_column)).strip() for row in iter_rows)
        record["units"] = {"column": unit_column, "distinct": len(units),
                           "most_common": [[u, n] for u, n in units.most_common(12)]}
    else:
        record["units"] = {"column": None, "distinct": 0,
                           "reading": "NO_UNIT_COLUMN: nothing in this resource says what its numbers are measured in"}

    if spec["value_column"]:
        value = spec["value_column"]
        ladder = list(spec.get("key_ladder") or (spec["series_key"] + spec["period_key"],))
        # the last rung is always every column except the value: two rows identical in every other field and
        # disagreeing about the number are the only disagreement no finer key can explain away.
        ladder.append(tuple(c for c in columns if c != value))
        rungs = []
        for key_columns in ladder:
            grouped = defaultdict(list)
            for row in iter_rows:
                grouped[tuple(str(getter(row, part)) for part in key_columns)].append(str(getter(row, value)))
            rungs.append({"key": list(key_columns), **_collisions(grouped)})
        roles = record["field_roles"]
        record["vintages"] = {
            "value_column": value,
            "key_ladder": rungs,
            "has_observation_clock": bool(roles["receipt_instant"]["present"]),
            "has_version_field": bool(roles["vintage_version"]["present"]),
            **_vintage_verdict(rungs[-1], has_observation_clock=roles["receipt_instant"]["present"],
                               has_version_field=roles["vintage_version"]["present"],
                               finest_key=[str(c) for c in rungs[-1]["key"]]),
        }
    else:
        record["vintages"] = {"key": None, "value_column": None, "verdict": "NOT_APPLICABLE_NO_VALUE_COLUMN",
                              "reading": "this resource carries schedules, not values, so it has no field to revise"}
    return record


def build_inventory(root):
    resources = [measure_resource(name, root) for name in RESOURCES]
    blocked = defaultdict(list)
    for record in resources:
        for role, found in (record.get("field_roles") or {}).items():
            for case in found.get("cases_it_blocks", []):
                blocked[case].append({"resource": record["resource"], "missing_role": role})
    return {"schema": SCHEMA, "root": str(root),
            "resources": resources,
            "cases_blocked_by_a_missing_field": {case: blocked[case] for case in sorted(blocked)},
            "reading": ("a case listed above cannot be evaluated against that resource until the named role exists; the "
                        "acceptance test for it is present and skipped by name, never quietly absent")}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the economic-calendar resources this machine holds.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[3]),
                        help="directory the resource paths are relative to (default: the sibling-repository root)")
    parser.add_argument("--out", default=None, help="write the inventory JSON here instead of stdout")
    args = parser.parse_args(argv)
    inventory = build_inventory(args.root)
    text = json.dumps(inventory, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":                                      # pragma: no cover - a CLI
    sys.exit(main())
