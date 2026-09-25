"""Rung 3 of the ladder: the path the fitted model says would have followed had one release surprised nobody.

WP22 step 4. Step 3 fitted, per event type, horizon and outcome, a local projection

    y[t_k, t_k + h] = alpha + beta * s_k(t_k) + gamma' controls(t_k) + eps

and wrote every coefficient of it, with a HAC interval, into a `m5phet.event_projections.v1` document. This module
asks that fitted model the counterfactual question the owner posed: **over a past window, what would the outcome path
have been if one named event's surprise had been zero?** -- and it answers it the only way a fitted regression can,
by evaluating itself twice.

For every release inside the window and every horizon, three numbers travel together:

* `predicted_observed` -- the model's own prediction with the surprises as they were published;
* `predicted_counterfactual` -- the same model, the same row, with the named event type's surprise set to zero for
  every release of it INSIDE THE WINDOW, which moves two columns and no others: the anchor's own `surprise` when the
  anchor is one of the zeroed releases, and the neighbour control, the sum of the other releases' surprises at
  negative offsets, from which the zeroed ones drop out;
* `observed_outcome` -- what the market actually did, copied from the row, so a reader can see the residual the model
  never explained and is never shown the prediction alone.

Their difference, `attributed_transient = predicted_observed - predicted_counterfactual`, is what this file is for.
It is **a difference of two evaluations of one fitted model**, not a measurement: nothing here observed a world in
which the release surprised nobody. That is why every document carries `label: MODEL_BASED_COUNTERFACTUAL` at the
top, and why the phrase is repeated in the reading rather than left to be inferred from the schema name.

**The interval.** The attributed transient is a linear functional of the fitted coefficients,
`sum_j beta_j * (x_j_observed - x_j_counterfactual)`, so its uncertainty is propagated by the delta method: the
half-widths of the coefficients' own 95 % intervals are combined in quadrature over the columns that moved. The
projections document reports a standard error and an interval **per coefficient** and no covariance between them, so
the propagation treats the coefficients as uncorrelated. They are not. That is an approximation and it is declared in
`interval_caveat` on every document rather than hidden in a footnote; the interval it produces can be narrower or
wider than the exact one, and the direction is not known without the covariance the upstream document does not carry.

**What is refused, by name.**

* `SUPERPOSITION_FAILED` -- the projections' superposition test for that (horizon, outcome) is not `ADDITIVE_HOLDS`.
  Zeroing one pulse of a window and leaving the arithmetic of the others untouched is exactly the additivity that
  test examines; where it did not hold, the subtraction has no meaning and no number is produced for that cell.
* `EVENT_TYPE_NOT_FITTED` -- the named event type has no projection in the document.
* `EVENT_NOT_IN_WINDOW` -- no release of the named type was published inside the window, so there is nothing to zero.
* `NO_FITTED_PROJECTION`, `ROW_MISSING_A_NUMBER`, `UNKNOWN_DESIGN_COLUMN` -- per cell: the projection for that (event
  type, horizon, outcome) did not fit, the row lacks a column the design needs, or the design names a column this
  evaluator does not know how to rebuild. None of them is answered with a zero.

**What is flagged and still computed.** `NOT_IDENTIFIED_BY_CONSTRUCTION`, when the rows were built under an assumed
publication clock. The owner asked to SEE the path in that case, labelled -- so the flag travels on the document, on
every path row and in the reading, the identification block and the caveat are copied verbatim from the projections,
and not one number is withheld. A flag that travels is not a refusal, and this file does not pretend otherwise.

Deterministic, CPU only: numpy and the standard library, over two documents on disk.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

from .local_projections import (SCHEMA as PROJECTIONS_SCHEMA, ProjectionRefusal, _window_surprises, load)

SCHEMA = "m5phet.counterfactual_path.v1"

#: the label every document carries, at the top and in its reading. A path computed by evaluating a fitted model
#: twice is a model's statement about a world nobody observed, and it is named as such wherever it is shown.
LABEL = "MODEL_BASED_COUNTERFACTUAL"

#: the flag that travels when the rows' publication clock was assumed rather than observed. It is NOT a refusal: the
#: owner asked for the path, labelled, and withholding it would answer a different question than the one he asked.
ASSUMED_CLOCK_FLAG = "NOT_IDENTIFIED_BY_CONSTRUCTION"

#: the superposition verdict a (horizon, outcome) must carry before one pulse may be subtracted from a window
ADDITIVE = "ADDITIVE_HOLDS"

#: the two columns a zeroed event type can move, and the only ones this module rewrites
MOVED_COLUMNS = ("surprise", "other_surprises_in_window_negative_offsets")

INTERVAL_METHOD = (
    "delta method on the linear functional sum_j beta_j * (x_j_observed - x_j_counterfactual), with each "
    "coefficient's 95 % half-width taken from the HAC interval the projections document reports for it and the "
    "half-widths combined in quadrature over the columns that moved")

INTERVAL_CAVEAT = (
    "COVARIANCE_OFF_DIAGONAL_NOT_AVAILABLE: the projections document reports a standard error and an interval per "
    "coefficient and no covariance between coefficients, so this propagation treats them as uncorrelated. They are "
    "not uncorrelated -- an OLS design has a full covariance matrix -- so the interval below may be narrower or "
    "wider than the exact one, and which it is cannot be told from the document it was propagated from. It is "
    "reported this way rather than omitted because an interval with a declared approximation is still an honest "
    "statement about uncertainty, and a point estimate with no interval at all is not")


class CounterfactualRefusal(ValueError):
    """An input this job will not evaluate, carrying the code and naming what is wrong with it."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise CounterfactualRefusal(code, why)


def _digest(path):
    hasher = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _instant(text, *, where):
    try:
        parsed = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        _refuse("BAD_WINDOW", f"the window's {where} is not an ISO-8601 instant: {text!r}")
    if parsed.tzinfo is None:
        _refuse("BAD_WINDOW", f"the window's {where} carries no time zone; an instant without one names no moment")
    return parsed.astimezone(timezone.utc)


def read_projections(path):
    """The projections document, checked to be one. Nothing else in this file reads it."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as trouble:
        _refuse("PROJECTIONS_NOT_READABLE", f"the projections document at {path} could not be read: {trouble}")
    if not isinstance(document, dict) or document.get("schema") != PROJECTIONS_SCHEMA:
        _refuse("WRONG_SCHEMA", f"schema is {(document or {}).get('schema')!r} and this module only reads "
                                f"{PROJECTIONS_SCHEMA!r}")
    return document


def projection_index(document):
    """Every fitted projection of a document, keyed by (event type, horizon, outcome)."""
    return {(entry["event_type"], int(entry["horizon_minutes"]), entry["outcome"]): entry
            for entry in document.get("projections") or []}


def superposition_index(document):
    """The superposition verdict per (horizon, outcome), which is the grain the test was run at."""
    return {(int(test["horizon_minutes"]), test["outcome"]): test
            for test in ((document.get("superposition") or {}).get("tests") or [])}


def _column_value(name, row, surprise, other):
    """One design column rebuilt for one row, by the name the projection recorded. None when the name is unknown."""
    if name == "const":
        return 1.0
    if name == "surprise":
        return surprise
    if name == "pre_event_realized_vol":
        return row["pre_event_realized_vol"]
    if name == "other_surprises_in_window_negative_offsets":
        return other
    for field, prefix in (("hour_of_day", "hour_of_day="), ("day_of_week", "day_of_week=")):
        if name.startswith(prefix):
            try:
                level = int(name[len(prefix):])
            except ValueError:
                return None
            return 1.0 if int(row[field]) == level else 0.0
    return None


def _half_width(coefficient):
    """Half of a coefficient's reported 95 % interval -- the width this module propagates, not a standard error."""
    return abs(float(coefficient["ci_upper"]) - float(coefficient["ci_lower"])) / 2.0


def _evaluate(entry, row, surprise, other):
    """The fitted model's prediction for one row, and the propagated half-width of its interval.

    Returns (value, half_width, None) or (None, None, (code, why)). Every column the projection kept must be
    rebuildable and finite, or nothing is returned: a prediction missing one of its own terms is a different model's
    prediction."""
    coefficients = entry.get("coefficients") or {}
    value, variance = 0.0, 0.0
    for name in entry.get("columns") or []:
        column = _column_value(name, row, surprise, other)
        if column is None:
            return None, None, ("UNKNOWN_DESIGN_COLUMN",
                                f"the projection kept a column named {name!r} and this evaluator does not know how to "
                                f"rebuild it from an event row, so no prediction is formed from a design it cannot "
                                f"reproduce")
        if not math.isfinite(float(column)):
            return None, None, ("ROW_MISSING_A_NUMBER",
                                f"the row carries no finite value for the design column {name!r}, and a missing "
                                f"number is not imputed here")
        coefficient = coefficients.get(name)
        if coefficient is None:
            return None, None, ("UNKNOWN_DESIGN_COLUMN",
                                f"the projection lists the column {name!r} among the ones it kept but reports no "
                                f"coefficient for it")
        value += float(coefficient["value"]) * float(column)
        variance += (_half_width(coefficient) * float(column)) ** 2
    return float(value), float(math.sqrt(variance)), None


def _interval(value, half_width):
    return [float(value - half_width), float(value + half_width)]


def paths(projections_path, rows_path, *, window, zero_out, outcomes=None, horizons=None, event_types=None,
          chunk_bytes=1 << 22):
    """The observed and counterfactual paths of one window, per release, horizon and outcome. Refusals are by name."""
    document = read_projections(projections_path)
    start = _instant(window[0], where="start")
    end = _instant(window[1], where="end")
    if end <= start:
        _refuse("BAD_WINDOW", f"the window ends at {end.isoformat()}, at or before it starts at {start.isoformat()}")

    fitted = projection_index(document)
    verdicts = superposition_index(document)
    declared_types = list(document.get("event_types") or [])
    if zero_out not in declared_types:
        _refuse("EVENT_TYPE_NOT_FITTED",
                f"{zero_out!r} has no projection in this document; it fitted {declared_types}, and a counterfactual "
                f"that sets to zero a surprise no model was fitted on would be a subtraction from nothing")

    wanted_outcomes = tuple(outcomes) if outcomes else tuple(document.get("outcomes") or ())
    unknown = [name for name in wanted_outcomes if name not in (document.get("outcomes") or [])]
    if unknown:
        _refuse("UNKNOWN_OUTCOME", f"{unknown} is not among the outcomes this projections document carries, "
                                   f"{list(document.get('outcomes') or [])}")
    wanted_horizons = ([int(h) for h in horizons] if horizons
                       else [int(h) for h in document.get("horizons_minutes") or []])
    missing = [h for h in wanted_horizons if h not in (document.get("horizons_minutes") or [])]
    if missing:
        _refuse("UNKNOWN_HORIZON", f"{missing} is not among the horizons this projections document carries, "
                                   f"{list(document.get('horizons_minutes') or [])}")

    loaded = load(rows_path, event_types=event_types or declared_types, horizons=wanted_horizons,
                  chunk_bytes=chunk_bytes)
    header, rows, index = loaded["header"], loaded["rows"], loaded["index"]
    window_seconds = float((header.get("parameters") or {}).get("window_hours", 24.0)) * 3600.0

    # the releases of the named type published inside the window: exactly what is set to zero, and nothing else
    zeroed = sorted({row["event_key"] for row in rows
                     if row["event_type"] == zero_out
                     and start.timestamp() <= row["published_epoch"] <= end.timestamp()})
    if not zeroed:
        _refuse("EVENT_NOT_IN_WINDOW",
                f"no release of {zero_out!r} was published inside [{start.isoformat()}, {end.isoformat()}]; there is "
                f"no surprise of it in this window to set to zero")
    zeroed_set = set(zeroed)

    anchors = [row for row in rows if start.timestamp() <= row["published_epoch"] <= end.timestamp()]
    anchor_keys = sorted({row["event_key"] for row in anchors})

    # the neighbour control, observed and counterfactual, once per release rather than once per row
    control = {}
    for row in anchors:
        key = row["event_key"]
        if key in control:
            continue
        _, observed_other = _window_surprises(index, row["published_epoch"], window_seconds, exclude_key=key)
        removed = 0.0
        left = int(np.searchsorted(index["epochs"], row["published_epoch"] - window_seconds, side="left"))
        right = int(np.searchsorted(index["epochs"], row["published_epoch"], side="left"))
        for position in range(left, right):
            neighbour = index["keys"][position]
            if neighbour == key or neighbour not in zeroed_set:
                continue
            surprise = float(index["surprises"][position])
            if math.isfinite(surprise):
                removed += surprise
        control[key] = (float(observed_other), float(observed_other - removed))

    entries, counters = [], {"paths": 0, "refused": 0, "by_refusal": {}}
    for row in sorted(anchors, key=lambda r: (r["published_epoch"], r["event_key"], r["horizon_minutes"])):
        if row["horizon_minutes"] not in wanted_horizons:
            continue
        for outcome in wanted_outcomes:
            key = (row["event_type"], int(row["horizon_minutes"]), outcome)
            entry = {"event_type": row["event_type"], "event_key": row["event_key"],
                     "published_at": row["published_at"], "horizon_minutes": int(row["horizon_minutes"]),
                     "outcome": outcome, "label": LABEL,
                     "zeroed": row["event_key"] in zeroed_set,
                     "observed_outcome": row[outcome]}
            projection = fitted.get(key)
            verdict = verdicts.get((int(row["horizon_minutes"]), outcome))
            if verdict is None or verdict.get("verdict") != ADDITIVE:
                entry.update({"status": "REFUSED", "refusal": "SUPERPOSITION_FAILED",
                              "why": (f"the superposition test for horizon {row['horizon_minutes']} minutes and "
                                      f"outcome {outcome!r} reports "
                                      f"{(verdict or {}).get('verdict', 'NO_TEST')!r}"
                                      + (f" ({verdict.get('why')})" if verdict and verdict.get("why") else "")
                                      + "; setting one pulse of a window to zero and leaving the rest of the "
                                        "arithmetic alone is exactly the additivity that test examines, so no path "
                                        "is computed for this cell")})
            elif projection is None or projection.get("status") != "OK":
                entry.update({"status": "REFUSED", "refusal": "NO_FITTED_PROJECTION",
                              "why": (f"the projection for {row['event_type']!r} at horizon "
                                      f"{row['horizon_minutes']} minutes and outcome {outcome!r} reports status "
                                      f"{(projection or {}).get('status', 'ABSENT')!r}, so there is no fitted model "
                                      f"to evaluate twice")})
            else:
                observed_other, counterfactual_other = control[row["event_key"]]
                surprise = row["surprise"]
                if surprise is None or not math.isfinite(surprise):
                    entry.update({"status": "REFUSED", "refusal": "ROW_MISSING_A_NUMBER",
                                  "why": "this release carries no finite standardized surprise, so the model's own "
                                         "treatment column cannot be formed for it and no prediction is made"})
                else:
                    counterfactual_surprise = 0.0 if row["event_key"] in zeroed_set else float(surprise)
                    observed, observed_half, trouble = _evaluate(projection, row, float(surprise), observed_other)
                    if trouble is None:
                        counterfactual, counterfactual_half, trouble = _evaluate(
                            projection, row, counterfactual_surprise, counterfactual_other)
                    if trouble is not None:
                        code, why = trouble
                        entry.update({"status": "REFUSED", "refusal": code, "why": why})
                    else:
                        moved = {}
                        for name in MOVED_COLUMNS:
                            if name not in (projection.get("columns") or []):
                                continue
                            before = _column_value(name, row, float(surprise), observed_other)
                            after = _column_value(name, row, counterfactual_surprise, counterfactual_other)
                            if before != after:
                                moved[name] = float(before) - float(after)
                        difference = float(observed - counterfactual)
                        half = math.sqrt(sum((delta * _half_width(projection["coefficients"][name])) ** 2
                                             for name, delta in moved.items()))
                        entry.update({
                            "status": "OK",
                            "surprise_observed": float(surprise),
                            "surprise_counterfactual": counterfactual_surprise,
                            "other_surprises_observed": observed_other,
                            "other_surprises_counterfactual": counterfactual_other,
                            "predicted_observed": observed,
                            "predicted_observed_ci_95": _interval(observed, observed_half),
                            "predicted_counterfactual": counterfactual,
                            "predicted_counterfactual_ci_95": _interval(counterfactual, counterfactual_half),
                            "attributed_transient": difference,
                            "attributed_transient_ci_95": _interval(difference, half),
                            "columns_that_moved": moved,
                            "beta": projection.get("beta"), "beta_ci_95": list(projection.get("beta_ci_95") or []),
                            "n_fit_events": projection.get("n_fit_events"),
                            "in_fit_sample": row["event_key"] in set(projection.get("fit_event_keys") or []),
                            "held_out_of_the_fit": row["event_key"] in set(projection.get("holdout_event_keys") or []),
                        })
            if entry["status"] == "REFUSED":
                counters["refused"] += 1
                counters["by_refusal"][entry["refusal"]] = counters["by_refusal"].get(entry["refusal"], 0) + 1
            else:
                counters["paths"] += 1
            entries.append(entry)

    clock = document.get("publication_clock") or {}
    flags = ([ASSUMED_CLOCK_FLAG]
             if str(clock.get("mode") or "").startswith("ASSUMED_SCHEDULED_PUBLICATION") else [])
    for entry in entries:
        entry["flags"] = list(flags)

    return {
        "schema": SCHEMA,
        "label": LABEL,
        "label_reading": (
            "every path below is the fitted local-projection model evaluated twice on the same rows -- once with the "
            "surprises as published and once with the named event type's surprise set to zero -- and their "
            "difference. No world in which that release surprised nobody was ever observed. It is a model's "
            "statement, not a measurement, and it is worth exactly what the model and its identification are worth"),
        "provenance": document.get("provenance"),
        "publication_clock": clock,
        "identification": document.get("identification"),
        "identification_reasons": list(document.get("identification_reasons") or []),
        "identification_reading": document.get("identification_reading"),
        "identification_caveat": clock.get("identification_caveat"),
        "flags": flags,
        "flags_reading": (
            f"{ASSUMED_CLOCK_FLAG} travels on this document and on every path row when the rows were built under an "
            f"assumed publication clock. It is a flag and not a refusal: the path is computed and shown, labelled, "
            f"because a model's counterfactual under an assumed clock is still the model's counterfactual -- what it "
            f"is not is identified, and that is said here rather than left to be discovered"),
        "zero_out": zero_out,
        "window": {"start": start.isoformat(), "end": end.isoformat(),
                   "releases_in_window": len(anchor_keys),
                   "releases_of_the_zeroed_type_in_window": len(zeroed),
                   "zeroed_release_keys": zeroed,
                   "rule": "a release is inside the window when its publication instant lies in [start, end]; only "
                           "the zeroed type's releases inside it are set to zero, and a release of it outside the "
                           "window keeps the surprise it was published with"},
        "intervention": {
            "columns_rewritten": list(MOVED_COLUMNS),
            "reading": ("setting one event type's surprise to zero moves exactly two columns of the fitted design: "
                        "the anchor's own `surprise`, when the anchor is one of the zeroed releases, and the "
                        "neighbour control `other_surprises_in_window_negative_offsets`, from which the zeroed "
                        "releases at negative offsets drop out. Every other column -- the constant, the pre-event "
                        "volatility, the hour and weekday dummies -- is the row's own and is untouched, because "
                        "zeroing a surprise does not move the clock")},
        "interval_method": INTERVAL_METHOD,
        "interval_caveat": INTERVAL_CAVEAT,
        "estimator": document.get("estimator"),
        "superposition": {"verdict": (document.get("superposition") or {}).get("verdict"),
                          "by_horizon_and_outcome": {f"h={horizon} {outcome}": test.get("verdict")
                                                     for (horizon, outcome), test in sorted(verdicts.items())},
                          "required": ADDITIVE},
        "projections_document": {"path": str(projections_path), "sha256": _digest(projections_path),
                                 "schema": document.get("schema")},
        "rows_document": {"path": str(rows_path), "schema": header.get("schema"),
                          "parameters": header.get("parameters")},
        "event_types": declared_types,
        "horizons_minutes": wanted_horizons,
        "outcomes": list(wanted_outcomes),
        "paths": entries,
        "counters": dict(counters, rows_read=loaded["counters"]["rows_read"],
                         rows_kept=loaded["counters"]["rows_kept"]),
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]), "numpy": np.__version__},
        "execution_authorized": False,
        "reading": (
            f"{LABEL}. The paths below come from evaluating the fitted local projections of "
            f"{document.get('schema')} twice per row, with {zero_out!r} surprising nobody in the second evaluation. "
            f"The document's identification is {document.get('identification')!r}"
            + (f" and it carries {ASSUMED_CLOCK_FLAG}" if flags else "") +
            f". {clock.get('identification_caveat')}. NO_NEW_MEASUREMENT: no price was observed for this file, no "
            f"model was fitted by it, no market claim is asserted in it and nothing in it authorises an order"),
    }


# --------------------------------------------------------------------------------------------------------- the CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.counterfactual",
        description="The fitted local projections evaluated with one event type's surprise set to zero over a past "
                    "window. MODEL_BASED_COUNTERFACTUAL: a model's statement, never an observation.")
    parser.add_argument("--projections", required=True, help="the m5phet.event_projections.v1 document")
    parser.add_argument("--rows", required=True, help="the m5phet.event_rows.v1 document those projections were fitted "
                                                      "from")
    parser.add_argument("--window", nargs=2, metavar=("START", "END"), required=True,
                        help="the past window, two ISO-8601 instants with time zones")
    parser.add_argument("--zero-out", required=True, help="the event type whose surprise is set to zero inside it")
    parser.add_argument("--horizons", type=int, nargs="+", help="only these horizons in minutes")
    parser.add_argument("--outcomes", nargs="+", help="only these outcomes")
    parser.add_argument("--out", help="where to write the counterfactual document; stdout when absent")
    parser.add_argument("--chunk-bytes", type=int, default=1 << 22, help="the streaming read size for the rows")
    args = parser.parse_args(argv)
    try:
        document = paths(args.projections, args.rows, window=tuple(args.window), zero_out=args.zero_out,
                         outcomes=args.outcomes, horizons=args.horizons, chunk_bytes=args.chunk_bytes)
    except (CounterfactualRefusal, ProjectionRefusal) as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=False, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"{document['counters']['paths']} path(s) and {document['counters']['refused']} refusal(s) written to "
              f"{args.out}; {document['label']}, identification {document['identification']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
