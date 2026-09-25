"""Rung 1 of the ladder: what moves together with a surprise, said in the words of a correlation and nothing more.

WP22 step 2. Before any identification argument is made, somebody has to be able to see the naive picture: over the
event rows of step 1, how does the outcome line up with the standardized surprise, and what is the average outcome
when the surprise was negative rather than positive, small rather than large, or landed on a quiet rather than an
agitated market. That picture is the **reference** the closure table needs. It is also, on its own, worth nothing as
a causal statement, because every one of those cells is open to the hour of the day, the regime, the volatility that
was already there and the other releases inside the same window -- which is exactly what rungs 2 and 3 exist for.

So this module is deliberately small. It reports, per event type and horizon:

* **Pearson** on the values and **Spearman** on their average ranks, each with the `n` it was computed over, or a
  refusal by name (`TOO_FEW_EVENTS`, `ZERO_VARIANCE`) instead of a number that would be a division by zero dressed
  up as a finding;
* the **naive response table**: mean and median outcome by surprise sign and by surprise tercile, with `n` in every
  cell, so a cell holding three events cannot be read as if it held three hundred;
* the same table by **pre-event realized volatility tercile**, because "the market was already moving" is the first
  thing anyone asks of an event-study number.

The terciles are cut from the rows of that (event type, horizon) group alone, by a declared quantile rule, and the
cut points are reported: a table whose bins nobody can reproduce is a table nobody can check. Ties are assigned to
the lower bin, which is stated rather than left to the reader, and a degenerate split -- everything in one bin
because the values barely differ -- shows up as an `n` of zero in the others rather than as a rebalanced bin.

Deterministic and CPU only: numpy and the standard library, no fit, no seed, the same rows to the same bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .events import SCHEMA as ROWS_SCHEMA

SCHEMA = "m5phet.event_association.v1"

#: the outcome columns of an event row that this job reads. Adding one means adding it here, so the document always
#: says which columns it looked at rather than "whatever was numeric".
OUTCOMES = ("log_return", "realized_vol")

#: fewer events than this and a correlation is an artefact of the two or three points it was drawn through
MIN_CORRELATION_N = 3

#: how the terciles are cut. numpy's "linear" interpolation, stated so the cut points can be recomputed by hand.
QUANTILE_METHOD = "linear"

SIGN_BINS = ("negative", "zero", "positive")
TERCILE_BINS = ("T1", "T2", "T3")


class AssociationRefusal(ValueError):
    """A rows document this job will not summarise, carrying the code and naming what is wrong with it."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise AssociationRefusal(code, why)


def _rank(values):
    """Average ranks, ties shared. Written out rather than imported so this module needs nothing beyond numpy."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(x, y):
    """The correlation, or the name of the reason there is none. No number is reported where one does not exist."""
    if x.size < MIN_CORRELATION_N:
        return {"value": None, "n": int(x.size),
                "reason": f"TOO_FEW_EVENTS: {x.size} event(s) and {MIN_CORRELATION_N} are declared as the fewest a "
                          f"correlation is reported over"}
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx <= 0 or sy <= 0:
        return {"value": None, "n": int(x.size),
                "reason": "ZERO_VARIANCE: one of the two columns does not vary over these events, so their "
                          "correlation is a division by zero and not a number"}
    value = float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy))
    return {"value": max(-1.0, min(1.0, value)), "n": int(x.size), "reason": None}


def _spearman(x, y):
    if x.size < MIN_CORRELATION_N:
        return {"value": None, "n": int(x.size),
                "reason": f"TOO_FEW_EVENTS: {x.size} event(s) and {MIN_CORRELATION_N} are declared as the fewest a "
                          f"correlation is reported over"}
    return _pearson(_rank(x), _rank(y))


def _cell(values):
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None}
    return {"n": int(values.size), "mean": float(np.mean(values)), "median": float(np.median(values))}


#: the name of the sign bin a surprise falls into. One rule, written once, so the table that is BUILT here and the
#: prediction that is READ from it downstream can never disagree about where a surprise belongs.
SIGN_OF = ("negative", "zero", "positive")


def sign_bin(surprise):
    """Which bin of the naive response table a surprise belongs to. Zero is its own bin, never folded into either."""
    return "negative" if surprise < 0 else ("positive" if surprise > 0 else "zero")


def naive_response_by_sign(surprise, outcome):
    """The naive response table: mean and median outcome in each surprise-sign bin, with the `n` of every cell.

    This is rung 1's whole contribution to a closure table. It is public because step 3 must score its projection
    against THIS statistic and not against a second implementation of it that happens to look the same -- and
    because a naive reference computed over different rows than the model was scored on is not a reference at all.
    """
    surprise, outcome = np.asarray(surprise, dtype=np.float64), np.asarray(outcome, dtype=np.float64)
    return {"negative": _cell(outcome[surprise < 0]),
            "zero": _cell(outcome[surprise == 0]),
            "positive": _cell(outcome[surprise > 0])}


def naive_sign_prediction(table, surprise, *, fallback=None):
    """What the naive reference predicts for one surprise, or the name of the reason it predicts nothing.

    `table` is a :func:`naive_response_by_sign` table FITTED ON THE ROWS THE MODEL WAS FITTED ON -- a table that has
    seen the events it is about to be scored on is not a naive reference, it is a look-ahead. When the bin a
    surprise falls into holds no fitted event, the table has nothing to say about it: the caller's declared
    `fallback` is used and the reason is returned beside it, so the substitution is visible in the closure table
    rather than hidden inside a mean.
    """
    cell = (table or {}).get(sign_bin(float(surprise))) or {}
    value = cell.get("mean")
    if value is None:
        return fallback, (f"EMPTY_SIGN_BIN: no fitted event had a {sign_bin(float(surprise))} surprise, so the naive "
                          f"response table says nothing about this event and the declared fallback was used")
    return float(value), None


def _by_sign(surprise, outcome):
    return naive_response_by_sign(surprise, outcome)


def _terciles(values):
    """The two cut points, or the reason there are none. Ties go to the LOWER bin, which is a rule, not a rounding."""
    if values.size < 3:
        return None, (f"TOO_FEW_EVENTS: {values.size} event(s) cannot be split into three bins")
    low = float(np.quantile(values, 1.0 / 3.0, method=QUANTILE_METHOD))
    high = float(np.quantile(values, 2.0 / 3.0, method=QUANTILE_METHOD))
    if not (math.isfinite(low) and math.isfinite(high)):
        return None, "NON_FINITE_CUTS: the quantiles of this column are not finite"
    return (low, high), None


def _by_tercile(values, outcome):
    cuts, reason = _terciles(values)
    if cuts is None:
        return {"cuts": None, "reason": reason, "bins": {name: _cell(np.asarray([])) for name in TERCILE_BINS}}
    low, high = cuts
    bins = {"T1": _cell(outcome[values <= low]),
            "T2": _cell(outcome[(values > low) & (values <= high)]),
            "T3": _cell(outcome[values > high])}
    return {"cuts": {"lower": low, "upper": high}, "reason": None, "bins": bins,
            "rule": f"T1 is value <= {low!r}, T2 is value in ({low!r}, {high!r}], T3 is value > {high!r}; the "
                    f"quantiles use numpy's {QUANTILE_METHOD!r} method over these rows alone"}


def _group(rows, outcome_name):
    """The three aligned columns this job reads, keeping only the rows where all three are numbers."""
    surprise = np.asarray([row.get("surprise") if isinstance(row.get("surprise"), (int, float)) else np.nan
                           for row in rows], dtype=np.float64)
    outcome = np.asarray([row.get(outcome_name) if isinstance(row.get(outcome_name), (int, float)) else np.nan
                          for row in rows], dtype=np.float64)
    pre = np.asarray([row.get("pre_event_realized_vol")
                      if isinstance(row.get("pre_event_realized_vol"), (int, float)) else np.nan
                      for row in rows], dtype=np.float64)
    usable = np.isfinite(surprise) & np.isfinite(outcome)
    return surprise, outcome, pre, usable


def summarise(document):
    """Per event type and horizon, the naive picture. Nothing here is an effect and the document says so."""
    if not isinstance(document, dict):
        _refuse("BAD_TYPE", f"the rows document must be a JSON object, got {type(document).__name__}")
    if document.get("schema") != ROWS_SCHEMA:
        _refuse("WRONG_SCHEMA", f"schema is {document.get('schema')!r} and this reader only reads {ROWS_SCHEMA!r}")
    rows = document.get("rows")
    if not isinstance(rows, list):
        _refuse("BAD_TYPE", f"rows must be a list, got {type(rows).__name__}")

    grouped = {}
    for row in rows:
        grouped.setdefault((row["event_type"], int(row["horizon_minutes"])), []).append(row)

    groups = []
    for (event_type, horizon) in sorted(grouped):
        members = grouped[(event_type, horizon)]
        entry = {"event_type": event_type, "horizon_minutes": horizon, "rows": len(members), "outcomes": {}}
        for outcome_name in OUTCOMES:
            surprise, outcome, pre, usable = _group(members, outcome_name)
            s, y = surprise[usable], outcome[usable]
            pre_usable = usable & np.isfinite(pre)
            block = {
                "n": int(s.size),
                "rows_without_a_number": int(len(members) - s.size),
                "pearson": _pearson(s, y),
                "spearman": _spearman(s, y),
                "by_surprise_sign": _by_sign(s, y),
                "by_surprise_tercile": _by_tercile(s, y),
                "by_pre_event_volatility_tercile": _by_tercile(pre[pre_usable], outcome[pre_usable]),
                "pre_event_volatility_n": int(np.count_nonzero(pre_usable)),
            }
            entry["outcomes"][outcome_name] = block
        groups.append(entry)

    clock = document.get("publication_clock") or {}
    provenance = document.get("provenance")
    return {
        "schema": SCHEMA,
        "provenance": provenance,
        "publication_clock": clock,
        "rows_document": {"schema": document.get("schema"),
                          "bars": (document.get("bars") or {}).get("path"),
                          "bars_sha256": (document.get("bars") or {}).get("sha256"),
                          "calendar": (document.get("calendar") or {}).get("path"),
                          "calendar_sha256": (document.get("calendar") or {}).get("sha256"),
                          "parameters": document.get("parameters")},
        "outcomes_read": list(OUTCOMES),
        "minimum_events_for_a_correlation": MIN_CORRELATION_N,
        "tercile_rule": {"quantiles": [1 / 3, 2 / 3], "method": QUANTILE_METHOD,
                         "ties": "a value equal to a cut point goes to the LOWER bin",
                         "scope": "cut from the rows of each (event type, horizon) group alone"},
        "groups": groups,
        "counts": {"groups": len(groups), "rows": len(rows),
                   "event_types": sorted({row["event_type"] for row in rows})},
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]), "numpy": np.__version__},
        "fitted": "NOTHING: correlations and conditional means; no model is fitted here",
        "reading": (f"PROVENANCE {provenance}, publication clock {clock.get('mode')}. "
                    f"{clock.get('identification_caveat')}. "
                    "RUNG 1, ASSOCIATION ONLY. Every number here is a correlation or a conditional mean over observed "
                    "events. None of them is an effect, a response, a sensitivity or an impulse: each is open to the "
                    "hour of the day, the regime, the volatility that was already there and the other releases inside "
                    "the same window, none of which is held fixed anywhere in this document. It is the NAIVE "
                    "reference a later estimate is compared against, and it is not itself evidence that a surprise "
                    "moved anything"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.association",
        description="The naive association between event surprises and the paths that followed. Nothing causal.")
    parser.add_argument("--rows", required=True, help="the event rows document written by feature_eng_m5phet.events")
    parser.add_argument("--out", help="where to write the association document; stdout when absent")
    args = parser.parse_args(argv)
    try:
        path = Path(args.rows)
        if not path.is_file():
            _refuse("NO_SUCH_FILE", f"{path} is not a file this job can read")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _refuse("MALFORMED_JSON", f"{path} is not JSON: {exc}")
        summary = summarise(document)
    except AssociationRefusal as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    text = json.dumps(summary, indent=2, sort_keys=False, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"{len(summary['groups'])} group(s) written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
