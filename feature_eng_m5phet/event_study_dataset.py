"""WP22 step 3(b), first half: the flat table an EconML study of a calendar surprise is fitted from.

`local_projections.py` answers rung 2 with one OLS per (event type, horizon, outcome) and reports `beta[k,h]` with a
HAC interval. That is the *homogeneous* answer: one number per event type and horizon for every release alike. WP22
asks for the heterogeneous one as well -- an effect that varies with the regime, with the volatility that was already
there and with the sign of the surprise -- and it asks for it through the causal provider's own `prepare-study --spec`,
which fits EconML. That estimator does not read an event-rows document: it reads a rectangular table of numbers with
one column per declared role. This module writes that table, and nothing else: no model is fitted here and no effect
is estimated here.

**One row per (release, horizon), the same rows the projection used.** The split into fitting and held-out events is
`local_projections._split_by_time` through `local_projections.prepare` -- not a second implementation of the same
rule, the same one -- so the events this table calls held out are the events the projections document calls held out,
and the two arms of the closure table are talking about the same releases. A row that the projection could not use
(no standardized surprise, a path with a hole in it, a pre-event span the bars did not cover) is dropped here for the
same reason and counted by name.

**The columns, and what each is a candidate for.** The treatment is `surprise`, the standardized release surprise --
continuous, as WP22 writes it, not a sign dummy. The outcomes are `log_return` and `realized_vol`, and a study
declares one of them and excludes the other. The candidate effect modifiers are the ones WP22 names: the surprise's
sign, the volatility that was already there (as a binary cut, because the engine reads an effect at a *declared
level* of a binary modifier), the hour bucket, and the regime -- the last only when the fitted reference the
unsupervised area serves can actually assign these rows. The candidate confounders are the other releases' pre-release
surprises and the day of week.

**A cut is declared, and it is measured on the fitting events only.** `pre_event_vol_high` is 1 above the median of
`pre_event_realized_vol` over the FIT rows of that table and 0 at or below it. The held-out rows are cut at the same
number, which was fixed before any of them was read; a median recomputed over the whole table would put the holdout's
own volatility into the definition of the modifier it is scored under.

**A column that does not vary is not offered.** A weekly release always lands on the same weekday and at the same
hour, so `day_of_week` and the hour buckets are constant in most of these tables. Handing a constant column to the
engine as a confounder makes its adjustment design rank deficient and the study is refused `NOT_IDENTIFIED` -- a
refusal about the design, not about the data. So every candidate column is checked here, and one that does not vary in
its own table is reported `CONSTANT_IN_THE_TABLE` and left out of the roles the manifest offers. The column itself is
still written into the CSV, with the role `exclude`, so the table is the same table whatever the spec asks of it.

**The regime is asked for, never assumed.** The unsupervised area serves a fitted reference; a reference declares the
features it was fitted on, and it can assign a row only when the row carries them. The event rows carry a price path
and a surprise, not a candle's body and range, so the demo reference cannot assign them -- and this module says that
by name (`REGIME_FEATURES_NOT_IN_THE_EVENT_ROWS`, naming the features it asked for) and leaves the column out. It
never fills the column with a label from somewhere else, and it never invents one.

Deterministic and CPU only: two runs over the same rows document with the same arguments write the same bytes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

from . import local_projections as lp

SCHEMA = "m5phet.event_study_table.v1"

#: the treatment column: the standardized release surprise, continuous
TREATMENT = "surprise"

#: the two outcomes an event row carries. A study declares one and excludes the other; both are written.
OUTCOMES = ("log_return", "realized_vol")

#: the UTC hour at which each declared bucket begins. The first bucket runs from the last edge round to the first.
DEFAULT_HOUR_BUCKET_EDGES = (0, 8, 14)

#: the fraction of each event type's releases, last by publication instant, that never enters a fit. The projections
#: document's own default, repeated from `local_projections` so the two splits cannot drift apart.
DEFAULT_HOLDOUT_FRACTION = lp.DEFAULT_HOLDOUT_FRACTION

#: the reference the unsupervised area serves by default; the regime column is attempted from it
DEFAULT_REGIME_REFERENCE = "~/.local/state/m5phet/examples/regimes/reference.joblib"

#: what a candidate column is offered as, when it varies
CANDIDATE_ROLES = {
    "surprise": "treatment",
    "log_return": "outcome",
    "realized_vol": "outcome",
    "pre_event_realized_vol": "confounder",
    "other_surprises_before": "confounder",
    "day_of_week": "confounder",
    "surprise_positive": "modifier",
    "pre_event_vol_high": "modifier",
    "hour_bucket": "modifier",
    "regime": "modifier",
}

#: the reasons a candidate column is not offered in a table's roles
CONSTANT_IN_THE_TABLE = "CONSTANT_IN_THE_TABLE"
REGIME_FEATURES_NOT_IN_THE_EVENT_ROWS = "REGIME_FEATURES_NOT_IN_THE_EVENT_ROWS"
REGIME_REFERENCE_NOT_READABLE = "REGIME_REFERENCE_NOT_READABLE"
REGIME_NOT_ASKED_FOR = "REGIME_NOT_ASKED_FOR"

#: the fewest rows the causal engine accepts; a table below it is written and flagged, never silently dropped
MIN_ENGINE_ROWS = 100

#: the most rows the causal engine accepts in one study
MAX_ENGINE_ROWS = 10_000

_SLUG = re.compile(r"[^a-z0-9]+")


class TableRefusal(ValueError):
    """A table this job will not write, carrying the code a caller matches on and what was refused."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise TableRefusal(code, why)


def slug(*parts):
    return "__".join(_SLUG.sub("_", str(part).lower()).strip("_") for part in parts)


def hour_bucket(hour, edges):
    """Which declared bucket a UTC hour falls in: the index of the last edge at or below it, wrapping at the first."""
    position = 0
    for index, edge in enumerate(edges):
        if hour >= edge:
            position = index
    return position


def _median(values):
    """The median of a sorted copy, by the ordinary rule. Written out so the cut in the document is reproducible."""
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return None
    middle = count // 2
    return float(ordered[middle]) if count % 2 else float((ordered[middle - 1] + ordered[middle]) / 2.0)


def regime_assignment(reference_path, rows):
    """The regime label of every row under the fitted reference the unsupervised area serves, or why there is none.

    Returns `(labels, None)` or `(None, {"refusal": ..., "why": ...})`. Nothing is fitted and nothing is cut here: the
    reference was fitted elsewhere, on features it declares, and a row that does not carry those features is not
    assigned to the nearest thing that happens to be available.
    """
    if reference_path is None:
        return None, {"refusal": REGIME_NOT_ASKED_FOR,
                      "why": "no fitted reference was named, so no regime was asked for and the column is absent"}
    path = Path(reference_path).expanduser()
    try:
        from .regimes import HierarchicalRegimes
        model = HierarchicalRegimes.load(path)
    except Exception as trouble:                                                              # noqa: BLE001
        return None, {"refusal": REGIME_REFERENCE_NOT_READABLE,
                      "why": f"the fitted reference at {path} could not be loaded in this environment "
                             f"({type(trouble).__name__}: {trouble}); a regime label is not invented to stand in for "
                             f"one a reference could not give"}
    features = list(model.metadata.get("features") or [])
    available = sorted(rows[0]) if rows else []
    missing = [name for name in features if name not in available]
    if missing:
        return None, {"refusal": REGIME_FEATURES_NOT_IN_THE_EVENT_ROWS,
                      "why": f"the fitted reference {model.metadata.get('task_id')!r} assigns rows that carry "
                             f"{features}, and an event row carries {available}; {missing} is not among them, so this "
                             f"reference cannot assign these rows. The column is left out rather than filled with a "
                             f"label from a feature vector assembled differently from the one the reference was "
                             f"fitted on",
                      "reference_features": features}
    assignment = model.assign([{"row_id": row["event_key"], **{name: row[name] for name in features}}
                               for row in rows])
    return {entry["row_id"]: entry["cluster_path"][-1] for entry in assignment["rows"]}, None


def build(rows_path, *, event_types=None, horizons=None, holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
          hour_bucket_edges=DEFAULT_HOUR_BUCKET_EDGES, window_hours=None, regime_reference=None,
          chunk_bytes=1 << 22):
    """One flat table per (event type, horizon): the rows, the declared cut, and the roles each column is offered for."""
    edges = tuple(sorted({int(edge) for edge in hour_bucket_edges}))
    if not edges or edges[0] != 0 or edges[-1] > 23:
        _refuse("BAD_HOUR_BUCKETS",
                f"the hour buckets are declared by the UTC hours they begin at, the first of which is 0 and the last "
                f"of which is at most 23; got {list(hour_bucket_edges)}")

    prepared = lp.prepare(rows_path, event_types=event_types, horizons=horizons,
                          holdout_fraction=holdout_fraction, window_hours=window_hours, chunk_bytes=chunk_bytes)
    rows, splits, header = prepared["rows"], prepared["splits"], prepared["header"]

    tables, dropped_total = [], {"surprise": 0, "log_return": 0, "realized_vol": 0, "pre_event_realized_vol": 0}
    for event_type in prepared["types"]:
        fit_keys, holdout_keys = splits.get(event_type, (set(), set()))
        for horizon in prepared["horizons"]:
            group = [row for row in rows if row["event_type"] == event_type and row["horizon_minutes"] == horizon]
            if not group:
                continue
            # a row enters the table only when EVERY modelled column of it is a number, for BOTH outcomes: one table
            # serves the two studies, and a table whose rows differ per outcome is two tables wearing one name
            usable = []
            counted = {key: 0 for key in dropped_total}
            for row in group:
                if row["surprise"] is None or not math.isfinite(row["surprise"]):
                    counted["surprise"] += 1
                    continue
                bad_outcome = next((name for name in OUTCOMES
                                    if row[name] is None or not math.isfinite(row[name])), None)
                if bad_outcome is not None:
                    counted[bad_outcome] += 1
                    continue
                pre = row["pre_event_realized_vol"]
                if row["pre_event_status"] != "OK" or pre is None or not math.isfinite(pre):
                    counted["pre_event_realized_vol"] += 1
                    continue
                usable.append(row)
            for key, value in counted.items():
                dropped_total[key] += value
            if not usable:
                continue

            fit_rows = [row for row in usable if row["event_key"] in fit_keys]
            holdout_rows = [row for row in usable if row["event_key"] in holdout_keys]
            # the cut is the median over the FITTING rows only, fixed before a held-out row is read
            cut = _median([row["pre_event_realized_vol"] for row in fit_rows])
            built = {"fit": [], "holdout": []}
            for split, source in (("fit", fit_rows), ("holdout", holdout_rows)):
                for row in source:
                    built[split].append({
                        "event_key": row["event_key"],
                        "published_at": row["published_at"],
                        "surprise": float(row["surprise"]),
                        "log_return": float(row["log_return"]),
                        "realized_vol": float(row["realized_vol"]),
                        "pre_event_realized_vol": float(row["pre_event_realized_vol"]),
                        "other_surprises_before": float(row["other_surprises"]),
                        "day_of_week": int(row["day_of_week"]),
                        "hour_of_day": int(row["hour_of_day"]),
                        "surprise_positive": 1 if row["surprise"] > 0 else 0,
                        "pre_event_vol_high": (0 if cut is None else
                                               (1 if row["pre_event_realized_vol"] > cut else 0)),
                        "hour_bucket": hour_bucket(int(row["hour_of_day"]), edges),
                    })
            tables.append({"event_type": event_type, "horizon_minutes": horizon, "cut": cut,
                           "dropped": counted, "rows": built})

    if not tables:
        _refuse("NO_TABLE",
                f"{rows_path} produced no (event type, horizon) table: every row was dropped for a named reason "
                f"{dropped_total}, and an empty table is not a table of zeros")

    # the regime, asked of the fitted reference over every row of every table at once, so one refusal covers them all
    every_row = [row for table in tables for split in table["rows"].values() for row in split]
    labels, regime_refusal = regime_assignment(regime_reference, every_row)
    if labels is not None:
        for row in every_row:
            row["regime"] = int(labels[row["event_key"]])

    columns = ["event_key", "published_at", TREATMENT, *OUTCOMES, "pre_event_realized_vol",
               "other_surprises_before", "day_of_week", "hour_of_day", "surprise_positive", "pre_event_vol_high",
               "hour_bucket"] + (["regime"] if labels is not None else [])

    documents = []
    for table in tables:
        fit, holdout = table["rows"]["fit"], table["rows"]["holdout"]
        offered, withheld = {}, {}
        for name, role in CANDIDATE_ROLES.items():
            if name not in columns:
                continue
            values = {row[name] for row in fit}
            if len(values) < 2:
                withheld[name] = {"role": role, "refusal": CONSTANT_IN_THE_TABLE,
                                  "why": f"{name} takes the single value "
                                         f"{(sorted(values)[0] if values else None)!r} "
                                         f"over the {len(fit)} fitting rows of this table; a constant column adjusts "
                                         f"for nothing, makes an adjustment design rank deficient and cannot carry a "
                                         f"level an effect is read at"}
                continue
            offered[name] = role
        documents.append({
            "event_type": table["event_type"],
            "horizon_minutes": table["horizon_minutes"],
            "slug": slug(table["event_type"], table["horizon_minutes"]),
            "pre_event_vol_cut": table["cut"],
            "counts": {"fit_rows": len(fit), "holdout_rows": len(holdout),
                       "fit_releases": len({row["event_key"] for row in fit}),
                       "holdout_releases": len({row["event_key"] for row in holdout}),
                       "dropped_by_reason": table["dropped"]},
            "engine_fit": ({"usable": True} if MIN_ENGINE_ROWS <= len(fit) <= MAX_ENGINE_ROWS else
                           {"usable": False,
                            "why": f"the causal engine fits between {MIN_ENGINE_ROWS} and {MAX_ENGINE_ROWS} rows and "
                                   f"this table's fitting split has {len(fit)}"}),
            "roles_offered": dict(sorted(offered.items())),
            "roles_withheld": dict(sorted(withheld.items())),
            "fit": fit,
            "holdout": holdout,
        })

    return {
        "schema": SCHEMA,
        "provenance": header.get("provenance"),
        "publication_clock": header.get("publication_clock"),
        "identification": "NOT_IDENTIFIED",
        "identification_reading": ("this document is a table of numbers, not a study. The verdict is carried from the "
                                   "rows document's own publication clock so that no study fitted from this table can "
                                   "be read without it; nothing here identifies anything"),
        "rows_document": str(rows_path),
        # where the LABELS came from: the price bars the outcome was read off, with their digest. The table's own
        # path identifies a design (which columns, which window); the bars identify the labels, and a later stage
        # that seals a corpus of these rows must name the second, not the first.
        "bars": header.get("bars"),
        "expectation": header.get("expectation"),
        "treatment": TREATMENT,
        "treatment_kind": "continuous",
        "outcomes": list(OUTCOMES),
        "columns": columns,
        "column_readings": {
            "surprise": (("the standardized release surprise (actual - MODEL_BASED_EXPECTATION) / scale, "
                          "continuous; the treatment. " + str((header.get("expectation") or {}).get("reading")))
                         if (header.get("expectation") or {}).get("kind") == "MODEL_BASED_EXPECTATION" else
                         "the standardized release surprise (actual - consensus) / scale, continuous; the treatment"),
            "log_return": "the log return from the release instant to the horizon",
            "realized_vol": "the realized variance over the horizon, the sum of squared bar log returns",
            "pre_event_realized_vol": "the realized variance over the declared pre-event span, before the release",
            "other_surprises_before": ("the sum of the standardized surprises of the other releases inside the "
                                       "declared window with NEGATIVE offsets only; a release at a positive offset "
                                       "landed after this one and is not pre-release information"),
            "day_of_week": "the UTC weekday of the release instant, Monday 0",
            "hour_of_day": "the UTC hour of the release instant",
            "surprise_positive": "1 when the standardized surprise is strictly positive, else 0",
            "pre_event_vol_high": ("1 above the median pre-event realized variance of this table's FITTING rows, else "
                                   "0; the cut is in `pre_event_vol_cut` and the held-out rows were cut at it"),
            "hour_bucket": "which declared UTC hour bucket the release fell in",
            "regime": "the leaf cluster of the fitted unsupervised reference, when it can assign these rows",
        },
        "hour_bucket_edges": list(edges),
        "window_hours": prepared["window_hours_used"],
        "window_hours_declared_by_the_rows": prepared["window_hours_declared_by_the_rows"],
        "holdout_fraction": float(holdout_fraction),
        "holdout_rule": ("local_projections._split_by_time: the last fraction of each event type's RELEASES by "
                         "publication instant, so the same release is held out at every horizon and these are the "
                         "same held-out events the projections document names"),
        "regime": ({"status": "ASSIGNED", "reference": str(Path(regime_reference).expanduser())}
                   if labels is not None else {"status": "ABSENT", **regime_refusal}),
        "tables": documents,
        "counts": {"tables": len(documents),
                   "fit_rows": sum(table["counts"]["fit_rows"] for table in documents),
                   "holdout_rows": sum(table["counts"]["holdout_rows"] for table in documents),
                   "dropped_by_reason": dropped_total},
        "fitted": "NOTHING: this job writes the table a study is fitted FROM; no effect is estimated here",
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3])},
        "execution_authorized": False,
    }


def write(document, out_dir):
    """One CSV per (event type, horizon) and split, plus the index that names every column and every withheld role."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = document["columns"]
    index = {key: value for key, value in document.items() if key != "tables"}
    index["tables"] = []
    for table in document["tables"]:
        files = {}
        for split in ("fit", "holdout"):
            name = f"{table['slug']}__{split}.csv"
            with (out_dir / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                for row in table[split]:
                    writer.writerow({column: row[column] for column in columns})
            files[split] = name
        index["tables"].append({**{key: value for key, value in table.items()
                                   if key not in ("fit", "holdout")}, "files": files})
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=False, allow_nan=False) + "\n",
                                        encoding="utf-8")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.event_study_dataset",
        description="Write the flat (release, horizon) table an EconML study of a calendar surprise is fitted from. "
                    "No model is fitted and no effect is estimated.")
    parser.add_argument("--rows", required=True, help="the m5phet.event_rows.v1 document")
    parser.add_argument("--out-dir", required=True, help="where the CSVs and their index are written")
    parser.add_argument("--event-type", action="append", dest="event_types",
                        help="restrict to this event type; repeatable")
    parser.add_argument("--horizons", type=int, nargs="+", help="restrict to these horizons in minutes")
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION,
                        help="the fraction of each event type's releases, last by publication instant, held out")
    parser.add_argument("--hour-bucket-edges", type=int, nargs="+", default=list(DEFAULT_HOUR_BUCKET_EDGES),
                        help="the UTC hours the declared buckets begin at; the first must be 0")
    parser.add_argument("--window-hours", type=float, default=None,
                        help="the window the other releases' pre-release surprises are summed over; the rows "
                             "document's own window when absent")
    parser.add_argument("--regime-reference", default=None,
                        help=f"the fitted unsupervised reference the regime column is asked of, e.g. "
                             f"{DEFAULT_REGIME_REFERENCE}; without it the column is absent and says so")
    parser.add_argument("--chunk-bytes", type=int, default=1 << 22)
    args = parser.parse_args(argv)
    try:
        document = build(args.rows, event_types=args.event_types, horizons=args.horizons,
                         holdout_fraction=args.holdout_fraction, hour_bucket_edges=args.hour_bucket_edges,
                         window_hours=args.window_hours, regime_reference=args.regime_reference,
                         chunk_bytes=args.chunk_bytes)
    except (TableRefusal, lp.ProjectionRefusal) as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    out = write(document, args.out_dir)
    print(f"{document['counts']['tables']} table(s), {document['counts']['fit_rows']} fitting and "
          f"{document['counts']['holdout_rows']} held-out row(s) written to {out}; regime "
          f"{document['regime']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
