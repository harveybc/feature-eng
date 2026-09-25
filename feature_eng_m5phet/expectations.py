"""A declared MODEL_BASED_EXPECTATION for every release, built from that release's own vintage history and nothing else.

WP28. The event study needs three things per release: what arrived, when it became public, and what somebody had
expected. This machine has an archive that observed the first two -- `announcements.parquet`, whose
`announcement_datetime_utc` is a publication instant nobody assumed -- and no archive at all that carries the third
over the same span. `docs/EVENT_STUDY_DATA_STATUS.md` measured that hole: the consensus archive stops in 2021, the
observed-clock archive starts in 2024-12, and the join rate between them is zero because the spans are disjoint.

This module fills the third slot **without pretending to have filled it**. For each release series it forecasts the
next value from the series' own prior vintages -- values PUBLISHED STRICTLY BEFORE that release -- and calls the
result what it is:

    MODEL_BASED_EXPECTATION -- a model's expectation is not the market's. No participant is known to have held this
    number, and a surprise measured against it is the residual of a declared time-series model, not the surprise the
    market traded on.

That sentence is `EXPECTATION_READING`, it is written into the document, onto every row, into the event rows built
from it, into the projections fitted on those and into the study manifest, and the word `consensus` is never used
for it anywhere in the chain.

**The declared family.** Two model kinds, both computed, neither defaulted:

* `SEASONAL_NAIVE(m)`: the value of the release `m` releases back in publication order. `m = 1` is "the same series'
  previous value"; `m = s` is "the same period last year", with `s` the number of releases a year implied by the
  series' own modal period spacing (12 monthly, 4 quarterly, ...), offered only when the series has `s + min_history`
  prior releases, and reported as UNAVAILABLE by name when it does not.
* `AR(p)`: an ordinary-least-squares autoregression of the series on its own prior values, with `p` chosen by **BIC**
  over `1..p_max` on the prior history alone. The criterion is declared, it is computed on the same slice the model
  is fitted on, and no information from at or after the release enters either.

**How the row's model is chosen, and what it costs.** Each candidate is scored by its own **expanding-window
one-step-ahead out-of-sample error** over the prior releases: for every earlier release `j`, the candidate's forecast
of `y_j` from values published strictly before `j`, against `y_j`. The candidate with the smallest prior out-of-sample
MAE wins the row. Both the chosen model's name and that error -- with the number of prior forecasts it averages --
travel on the row, because a stimulus built on an expectation that is itself wrong by more than the stimulus is a
number a reader must be able to discount.

**No look-ahead, and the arithmetic that enforces it.** Every forecast of release `i` is a function of
`y_1..y_{i-1}` only; every model selection at release `i` is a function of errors realized before `i` only; every
error `e_j` in that average is itself a forecast made from before `j`. `tests/test_model_expectations.py` plants a
different future and asserts, byte for byte, that no earlier row moves. The dispersion the stimulus is divided by is
NOT computed here: `events.py` computes it from the surprises published strictly before each release, under its own
no-look-ahead rule and its own test, and this module deliberately does not duplicate that.

Deterministic and CPU only: no seed is drawn, no randomness is used, and the same archive with the same arguments
produces the same bytes. numpy and the standard library, plus whatever `calendar_join.read_announcements` needs to
open the archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import timezone as _timezone
from pathlib import Path

import numpy as np

from .calendar_join import read_announcements
from .design import _time_parser

SCHEMA = "m5phet.model_expectations.v1"

#: what this number is, everywhere it appears. It is never called a consensus, in any artifact, in any answer.
EXPECTATION_KIND = "MODEL_BASED_EXPECTATION"

#: the one line that travels with the kind, verbatim
EXPECTATION_READING = ("a model's expectation is not the market's: this number is what a declared time-series model, "
                       "fitted on this series' own prior vintages only, would have forecast for this release. No "
                       "market participant is known to have held it, and the stimulus built from it is that model's "
                       "one-step out-of-sample residual, not the surprise the market traded on")

#: the largest autoregressive order considered. Declared, not tuned: a wider grid searched per release would be a
#: selection this document could not report in one line.
DEFAULT_MAX_AR_ORDER = 4

#: the fewest prior releases before any candidate is fitted at all
DEFAULT_MIN_HISTORY = 12

#: the fewest prior out-of-sample forecasts a candidate must already have made before it may be CHOSEN by them
DEFAULT_MIN_OOS = 8

#: the criterion the autoregressive order is chosen by, on the prior slice alone
ORDER_CRITERION = "BIC"

#: the criterion the model is chosen by at each release, from errors realized before it
SELECTION_CRITERION = "the smallest expanding-window one-step-ahead out-of-sample MAE over the prior releases"

#: releases a year, by the series' modal period spacing in days. A spacing outside this table offers no seasonal
#: candidate beyond the previous value, and the document says so rather than inventing a period for it.
SEASONAL_PERIODS = ((300.0, 450.0, 1), (80.0, 100.0, 4), (25.0, 35.0, 12), (6.0, 8.0, 52))

#: every reason a release does not get an expectation. Counted exactly; the count is the finding.
EXCLUSION_CODES = ("NO_OBSERVED_PUBLICATION_INSTANT", "NO_VALUE", "INSUFFICIENT_VINTAGE_HISTORY",
                   "NO_CANDIDATE_WITH_ENOUGH_OUT_OF_SAMPLE_HISTORY", "OUTSIDE_THE_DECLARED_EMISSION_WINDOW")

#: how many excluded releases are described one by one. The COUNTS are always exact.
MAX_EXCLUDED_DETAIL = 200

#: the columns of the calendar this module writes, in the names `events.py` is told to read
CALENDAR_COLUMNS = ("event_type", "currency", "indicator", "period", "event_time", "published_at", "actual",
                    "expectation", "previous", "historical_availability", "expectation_model",
                    "expectation_model_oos_mae", "expectation_model_oos_rmse", "expectation_model_oos_n",
                    "expectation_candidates", "series_key", "archive_row")

#: what the `event_time` column of that calendar means. This archive declares no SCHEDULED instant, so the column
#: repeats the observed publication instant; it is never read as a schedule by anything downstream, which reads
#: `published_at`, and saying it here is the only thing that keeps the repetition from looking like a measurement.
EVENT_TIME_READING = ("this archive declares no scheduled instant, only an observed publication instant, so the "
                      "`event_time` column repeats `published_at`. Nothing downstream reads it as a schedule: the "
                      "release boundary is taken from `published_at`, which was observed")


class ExpectationRefusal(ValueError):
    """An input this job will not build expectations from, carrying the code a caller matches on."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise ExpectationRefusal(code, why)


# ------------------------------------------------------------------------------------------------- the candidates

def _seasonal_period(period_days):
    """The releases-a-year the series' own modal period spacing implies, or None when the table declares none."""
    if period_days is None or not math.isfinite(period_days):
        return None
    for low, high, releases in SEASONAL_PERIODS:
        if low <= period_days <= high:
            return releases if releases >= 2 else None
    return None


def _ar_forecast(values, max_order, min_history):
    """One step ahead of `values`, from an AR(p) whose p is chosen by BIC on `values` alone.

    `values` is the whole prior history, in publication order, and nothing outside it is read. Returns
    (forecast, order) or (None, None) when no order is fittable on this much history.
    """
    n = len(values)
    if n < min_history:
        return None, None
    y = np.asarray(values, dtype=np.float64)
    best = (None, None, None)
    for order in range(1, int(max_order) + 1):
        rows = n - order
        if rows < order + 2:                       # an intercept, `order` slopes, and two degrees of freedom left
            break
        design = np.empty((rows, order + 1), dtype=np.float64)
        design[:, 0] = 1.0
        for lag in range(1, order + 1):
            design[:, lag] = y[order - lag:n - lag]
        target = y[order:]
        try:
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = target - design @ coefficients
        sigma2 = float(residual @ residual) / rows
        if not math.isfinite(sigma2):
            continue
        # BIC of a Gaussian likelihood at the OLS fit; a perfect fit gets -inf and is never preferred by accident
        bic = (math.inf if sigma2 <= 0 else
               rows * math.log(sigma2) + (order + 1) * math.log(rows))
        if best[0] is None or bic < best[0]:
            last = y[n - order:][::-1]             # y[n-1], y[n-2], ... in the order the coefficients expect
            forecast = float(coefficients[0] + float(np.dot(coefficients[1:], last)))
            if math.isfinite(forecast):
                best = (bic, forecast, order)
    if best[1] is None:
        return None, None
    return best[1], best[2]


def _candidate_names(seasonal, max_order):
    names = ["SEASONAL_NAIVE(m=1)"]
    if seasonal is not None:
        names.append(f"SEASONAL_NAIVE(m={seasonal})")
    names.append(f"AR(p<={int(max_order)},{ORDER_CRITERION})")
    return names


def _forecast_sequences(values, *, seasonal, max_order, min_history):
    """For every release `i`, each candidate's forecast of `y_i` from `y_0..y_{i-1}` and nothing else."""
    n = len(values)
    names = _candidate_names(seasonal, max_order)
    sequences = {name: [None] * n for name in names}
    orders = [None] * n
    for i in range(n):
        prior = values[:i]
        sequences["SEASONAL_NAIVE(m=1)"][i] = prior[-1] if prior else None
        if seasonal is not None:
            key = f"SEASONAL_NAIVE(m={seasonal})"
            # the same period a year back, offered only once the series has the declared history behind it
            sequences[key][i] = prior[-seasonal] if len(prior) >= seasonal + min_history else None
        forecast, order = _ar_forecast(prior, max_order, min_history)
        sequences[f"AR(p<={int(max_order)},{ORDER_CRITERION})"][i] = forecast
        orders[i] = order
    return names, sequences, orders


# ------------------------------------------------------------------------------------------------------ the series

class _Excluded:
    """Exact counts, capped examples."""

    def __init__(self):
        self.counts = {code: 0 for code in EXCLUSION_CODES}
        self.releases, self.truncated = [], {code: 0 for code in EXCLUSION_CODES}

    def drop(self, code, *, series, row_number, published_at, why):
        self.counts[code] += 1
        if len(self.releases) < MAX_EXCLUDED_DETAIL:
            self.releases.append({"series": series, "archive_row": row_number, "published_at": published_at,
                                  "code": code, "why": why})
        else:
            self.truncated[code] += 1

    def document(self):
        return {"counts": self.counts,
                "counts_reading": "every count is of RELEASES of the announcement archive",
                "releases": self.releases, "examples_truncated_at": MAX_EXCLUDED_DETAIL,
                "examples_not_listed": {code: n for code, n in self.truncated.items() if n}}


def _period_date(text):
    if not text:
        return None
    try:
        _, parser = _time_parser(text)
        return parser(text)
    except ValueError:
        return None


def _bounds(text, *, what):
    if text is None:
        return None
    try:
        _, parser = _time_parser(text)
        moment = parser(text)
    except ValueError:
        _refuse("BOUND_UNPARSEABLE", f"{what} reads {text!r}, which no declared format reads")
    if moment.tzinfo is None or moment.utcoffset() is None:
        _refuse("BOUND_WITHOUT_A_ZONE", f"{what} reads {text!r}, a wall clock with no offset; declare the instant")
    return moment.astimezone(_timezone.utc)


def build(announcements_path, *, max_ar_order=DEFAULT_MAX_AR_ORDER, min_history=DEFAULT_MIN_HISTORY,
          min_oos=DEFAULT_MIN_OOS, emit_from=None, emit_until=None, announcement_columns=None,
          currencies=None, indicators=None):
    """One expectation per release, from that release's own prior vintages. Nothing is fitted across series."""
    if int(max_ar_order) < 1:
        _refuse("BAD_AR_ORDER", f"the largest autoregressive order must be at least 1, got {max_ar_order}")
    if int(min_history) < 3:
        _refuse("BAD_MIN_HISTORY",
                f"an autoregression fitted on fewer than three prior values is not an estimate, got {min_history}")
    if int(min_oos) < 2:
        _refuse("BAD_MIN_OUT_OF_SAMPLE",
                f"a model chosen by fewer than two out-of-sample errors is chosen by noise, got {min_oos}")
    max_ar_order, min_history, min_oos = int(max_ar_order), int(min_history), int(min_oos)
    emit_from, emit_until = _bounds(emit_from, what="--emit-from"), _bounds(emit_until, what="--emit-until")

    meta, announcements = read_announcements(announcements_path, names=announcement_columns)
    excluded = _Excluded()
    kept = meta["rows_read"] - len(announcements)
    if kept:
        excluded.counts["NO_OBSERVED_PUBLICATION_INSTANT"] = kept

    series = {}
    for row in announcements:
        currency, indicator = row["currency"], row["indicator"]
        if currencies and currency not in currencies:
            continue
        if indicators and indicator not in indicators:
            continue
        key = f"{currency} | {indicator}"
        text = (row.get("value") or "").strip()
        try:
            value = float(text)
        except ValueError:
            excluded.drop("NO_VALUE", series=key, row_number=row["row_number"],
                          published_at=row["published_at"].isoformat(),
                          why=f"the archive's value cell reads {text!r}, which is not a number; a release with no "
                              f"number is not a release")
            continue
        if not math.isfinite(value):
            excluded.drop("NO_VALUE", series=key, row_number=row["row_number"],
                          published_at=row["published_at"].isoformat(),
                          why=f"the archive's value is {value}, which is not finite")
            continue
        series.setdefault(key, []).append({"currency": currency, "indicator": indicator,
                                           "published_at": row["published_at"], "value": value,
                                           "period": row.get("period") or "",
                                           "row_number": row["row_number"]})

    rows, by_series = [], {}
    for key in sorted(series):
        entries = sorted(series[key], key=lambda e: (e["published_at"], e["period"], e["row_number"]))
        values = [entry["value"] for entry in entries]
        periods = [_period_date(entry["period"]) for entry in entries]
        spacings = [abs((b - a).days) for a, b in zip(periods, periods[1:]) if a is not None and b is not None]
        period_days = float(statistics.median(spacings)) if spacings else None
        seasonal = _seasonal_period(period_days)
        names, sequences, orders = _forecast_sequences(values, seasonal=seasonal, max_order=max_ar_order,
                                                       min_history=min_history)
        # the errors realized BEFORE release i, per candidate: e_j = y_j - forecast_j, and forecast_j saw only
        # y_0..y_{j-1}. The average below therefore never contains an error from at or after i.
        errors = {name: [None if sequences[name][j] is None else values[j] - sequences[name][j] for j in range(len(values))]
                  for name in names}
        summary = {"releases": len(entries), "modal_period_days": period_days, "seasonal_releases_a_year": seasonal,
                   "candidates": names, "chosen": {}, "rows": 0,
                   "first_published_at": entries[0]["published_at"].isoformat(),
                   "last_published_at": entries[-1]["published_at"].isoformat()}
        if seasonal is None:
            summary["seasonal_candidate"] = ("UNAVAILABLE: the series' modal period spacing implies no declared "
                                             "seasonal period, so the only naive candidate is the previous value")
        for index, entry in enumerate(entries):
            moment = entry["published_at"]
            if (emit_from is not None and moment < emit_from) or (emit_until is not None and moment > emit_until):
                excluded.drop("OUTSIDE_THE_DECLARED_EMISSION_WINDOW", series=key, row_number=entry["row_number"],
                              published_at=moment.isoformat(),
                              why="this release is outside the declared emission window; its value still enters the "
                                  "vintage history of the releases that are inside it")
                continue
            scored = []
            for name in names:
                if sequences[name][index] is None:
                    continue
                realized = [abs(e) for e in errors[name][:index] if e is not None]
                if len(realized) < min_oos:
                    continue
                squared = [e * e for e in errors[name][:index] if e is not None]
                scored.append((float(sum(realized) / len(realized)), name,
                               float(math.sqrt(sum(squared) / len(squared))), len(realized)))
            if not scored:
                available = sum(1 for e in errors["SEASONAL_NAIVE(m=1)"][:index] if e is not None)
                code = ("INSUFFICIENT_VINTAGE_HISTORY" if index < min_history
                        else "NO_CANDIDATE_WITH_ENOUGH_OUT_OF_SAMPLE_HISTORY")
                excluded.drop(code, series=key, row_number=entry["row_number"], published_at=moment.isoformat(),
                              why=f"{index} release(s) of {key!r} had been published before this one and "
                                  f"{available} out-of-sample forecast(s) had been made; {min_oos} are declared as "
                                  f"the fewest a model may be CHOSEN by, and an expectation from a model nobody "
                                  f"could have preferred yet is an invented number")
                continue
            scored.sort(key=lambda item: (item[0], item[1]))
            mae, chosen, rmse, n_oos = scored[0]
            summary["chosen"][chosen] = summary["chosen"].get(chosen, 0) + 1
            summary["rows"] += 1
            rows.append({
                "event_type": key, "currency": entry["currency"], "indicator": entry["indicator"],
                "period": entry["period"],
                "event_time": moment.isoformat(), "published_at": moment.isoformat(),
                "actual": entry["value"],
                "expectation": sequences[chosen][index],
                "previous": values[index - 1] if index else None,
                "historical_availability": "KNOWN",
                "expectation_model": (chosen if not chosen.startswith("AR(") or orders[index] is None
                                      else f"{chosen}->p={orders[index]}"),
                "expectation_model_oos_mae": mae, "expectation_model_oos_rmse": rmse,
                "expectation_model_oos_n": n_oos,
                "expectation_candidates": ";".join(name for _, name, _, _ in sorted(scored, key=lambda i: i[1])),
                "series_key": key, "archive_row": entry["row_number"],
            })
        by_series[key] = summary

    rows.sort(key=lambda r: (r["published_at"], r["event_type"], r["archive_row"]))
    chosen_overall = {}
    for row in rows:
        name = row["expectation_model"].split("->")[0]
        chosen_overall[name] = chosen_overall.get(name, 0) + 1
    errors_all = [row["expectation_model_oos_mae"] for row in rows]
    return {
        "schema": SCHEMA,
        "expectation_kind": EXPECTATION_KIND,
        "expectation_reading": EXPECTATION_READING,
        "never_called": ("consensus. No field, count, answer or sentence produced from this document calls this "
                         "number a consensus, because nobody published one"),
        "archive": {key: value for key, value in meta.items() if key != "column_names"},
        "archive_columns_read": meta.get("column_names"),
        "parameters": {
            "max_ar_order": max_ar_order, "min_history": min_history, "min_out_of_sample": min_oos,
            "order_criterion": ORDER_CRITERION, "selection_criterion": SELECTION_CRITERION,
            "seasonal_periods_declared": [{"modal_period_days_from": low, "to": high, "releases_a_year": n}
                                          for low, high, n in SEASONAL_PERIODS],
            "emit_from": None if emit_from is None else emit_from.isoformat(),
            "emit_until": None if emit_until is None else emit_until.isoformat(),
            "emission_window_reading": ("a release outside the window is excluded from the CALENDAR and still enters "
                                        "the vintage history of the releases inside it; the window bounds what is "
                                        "emitted, never what a model was allowed to see"),
            "currencies": sorted(currencies) if currencies else None,
            "indicators": sorted(indicators) if indicators else None,
            "no_look_ahead": ("the forecast of release i is a function of y_0..y_{i-1} only; the model chosen at i is "
                              "chosen by errors realized strictly before i; each of those errors is itself a forecast "
                              "made from strictly before its own release. The DISPERSION the stimulus is divided by "
                              "is not computed here: events.py computes it from the surprises published strictly "
                              "before each release, under its own rule and its own test"),
        },
        "event_time_reading": EVENT_TIME_READING,
        "counts": {"rows": len(rows), "series": len(by_series),
                   "series_with_a_row": sum(1 for s in by_series.values() if s["rows"]),
                   "chosen_model": dict(sorted(chosen_overall.items())),
                   "out_of_sample_mae_median": (float(statistics.median(errors_all)) if errors_all else None)},
        "by_series": by_series,
        "excluded": excluded.document(),
        "rows": rows,
        "fitted": ("one autoregression and two naive rules PER RELEASE, on that release's own prior values only; "
                   "nothing is fitted across series, nothing is fitted on the price bars, and no causal quantity is "
                   "estimated anywhere in this file"),
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]), "numpy": np.__version__},
        "reading": (f"{EXPECTATION_KIND}. {EXPECTATION_READING}. Each row's `expectation_model` says which declared "
                    f"candidate produced it and `expectation_model_oos_mae` says how wrong that candidate had been, "
                    f"out of sample, on the releases before it -- read the stimulus against that error before "
                    f"reading anything into its size"),
    }


def write_csv(document, path):
    """The calendar, in the columns `events.py` is told to read. One row per release with an expectation."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CALENDAR_COLUMNS)
        for row in document["rows"]:
            writer.writerow(["" if row.get(name) is None else row.get(name) for name in CALENDAR_COLUMNS])
    return path


# --------------------------------------------------------------------------------------------------------- the CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.expectations",
        description="Build a MODEL_BASED_EXPECTATION for every release from its own vintage history. Never a "
                    "consensus: no market expectation is read, claimed or implied.")
    parser.add_argument("--announcements", required=True,
                        help="the archive of observed announcement instants (parquet or CSV)")
    parser.add_argument("--out", required=True, help="where to write the expectation calendar (CSV)")
    parser.add_argument("--report", help="where to write the expectation document (JSON); stdout when absent")
    parser.add_argument("--max-ar-order", type=int, default=DEFAULT_MAX_AR_ORDER)
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument("--min-out-of-sample", type=int, default=DEFAULT_MIN_OOS)
    parser.add_argument("--emit-from", help="emit only releases published at or after this instant")
    parser.add_argument("--emit-until", help="emit only releases published at or before this instant")
    parser.add_argument("--currency", action="append", dest="currencies", help="restrict to this currency; repeatable")
    parser.add_argument("--indicator", action="append", dest="indicators",
                        help="restrict to this indicator; repeatable")
    parser.add_argument("--announcement-currency-column")
    parser.add_argument("--announcement-indicator-column")
    parser.add_argument("--announcement-published-column")
    parser.add_argument("--announcement-value-column")
    parser.add_argument("--announcement-period-column")
    args = parser.parse_args(argv)
    names = {key: value for key, value in (("currency", args.announcement_currency_column),
                                           ("indicator", args.announcement_indicator_column),
                                           ("published", args.announcement_published_column),
                                           ("value", args.announcement_value_column),
                                           ("period", args.announcement_period_column)) if value}
    try:
        document = build(args.announcements, max_ar_order=args.max_ar_order, min_history=args.min_history,
                         min_oos=args.min_out_of_sample, emit_from=args.emit_from, emit_until=args.emit_until,
                         announcement_columns=names or None,
                         currencies=set(args.currencies) if args.currencies else None,
                         indicators=set(args.indicators) if args.indicators else None)
    except ExpectationRefusal as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    except Exception as refusal:                              # calendar_join refuses with its own type
        if type(refusal).__name__ != "JoinRefusal":
            raise
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    write_csv(document, args.out)
    text = json.dumps(document, indent=2, sort_keys=False, allow_nan=False)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    print(f"{document['counts']['rows']} expectation(s) over {document['counts']['series_with_a_row']} series "
          f"written to {args.out} ({EXPECTATION_KIND}, never a consensus)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
