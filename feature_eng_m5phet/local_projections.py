"""Rung 2 of the ladder: the transient response of EUR/USD to a calendar surprise, and the checks that refuse it.

WP22 step 3(a). Step 1 wrote, per release and horizon, the standardized surprise and the path that followed it; step 2
said plainly that those paths correlate with the surprise and that a correlation is not an effect. This module fits
the estimator the literature actually uses for the question -- a **local projection** (Jorda 2005): for horizon `h`,

    y[t_k, t_k + h] = alpha + beta[k, h] * s_k(t_k) + gamma' controls(t_k) + eps

with `y` either the log return or the realized volatility over the horizon, `s_k` the standardized release surprise,
and `beta[k, h]` read as the impulse response at that horizon. The identification argument is the standard
macro-announcement one (Andersen-Bollerslev-Diebold-Vega 2003; Gurkaynak-Sack-Swanson 2005): the surprise is
as-good-as-random **conditional on the pre-release information set**. So the controls are exactly that information
set, and nothing else:

* the **pre-event realized volatility** over the declared span before the release -- the agitation that was already
  there;
* **hour-of-day** and **day-of-week** dummies, in UTC, for the intraday and weekly seasonality of the series;
* the **sum of the other releases' standardized surprises inside the window `W`, with NEGATIVE offsets only**. A
  neighbour at a positive offset landed AFTER this release: it is not pre-release information, it is a consequence of
  the same news flow, and conditioning on it would be conditioning on a collider. It is therefore **excluded**, and
  this sentence is written into the document so no reader has to guess which half of the window entered.

Four things in this file exist to stop it from claiming more than it measured:

* **HAC standard errors.** Overlapping horizons make the residuals of neighbouring events serially correlated, so an
  OLS interval is too narrow. Newey-West through statsmodels is the declared covariance. If statsmodels cannot be
  imported the whole block is refused `STATSMODELS_NOT_AVAILABLE` by name: a plain OLS interval presented as a HAC
  interval would be a lie with a smaller number in it.
* **Held-out events by time.** The last fraction of each event type's releases never touch a fit. Every number that
  compares this estimator to something else is computed on those events, and the naive reference -- rung 1's mean
  outcome by surprise sign, imported from `association.py` rather than reimplemented -- is fitted on the SAME
  training events and scored on the SAME held-out events. A reference that saw the test set is not a reference.
* **The superposition test.** That several pulses inside one window add up is a hypothesis, not an arithmetic fact.
  The additive model and a model with the interaction of the two most co-occurring event types are scored on the
  held-out events, and the verdict is by held-out MSE with a declared margin.
* **The placebo.** Pseudo-events drawn from bar instants with no release inside the exclusion span, dressed with the
  event type's own empirical surprise distribution, must produce `beta` intervals that contain zero and do not touch
  the real ones. When they do not, the (event type, horizon) is `NOT_IDENTIFIED` and the document says so.

And the last word belongs to the clock. If the rows were built under `ASSUMED_SCHEDULED_PUBLICATION` -- an operator's
declared assumption that each release was published at the instant it was scheduled for -- then **nothing in this
document is identified**, however well every check inside it went, because the instant the surprise became public was
assumed rather than observed. The top-level `identification` is `NOT_IDENTIFIED` in that case by construction, the
caveat travels verbatim from the rows, and no sentence anywhere here may be read as a market claim.

Under `OBSERVED_ACTUAL_PUBLICATION` -- the joined calendar, whose release instants an archive actually observed --
that one reason is gone and the verdict is decided by the checks: the placebo, and whether it ran at all. What stays
assumed there is only that the consensus stood before the release, which is what the identification argument assumes
anyway; it is written into the same block so nobody reads `OBSERVED` as covering both instants.

Deterministic, seeded, CPU only: numpy, statsmodels, and the standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .association import naive_response_by_sign, naive_sign_prediction, sign_bin
from .events import SCHEMA as ROWS_SCHEMA, _contiguous, _realized, read_bars

SCHEMA = "m5phet.event_projections.v1"

#: the fraction of each event type's releases, LAST BY PUBLICATION INSTANT, that never enters a fit
DEFAULT_HOLDOUT_FRACTION = 0.20

#: how many pseudo-events are drawn per placebo
DEFAULT_PLACEBO_N = 200

#: the seed. Every draw in this module comes from it, so two runs of the same command write the same bytes.
DEFAULT_SEED = 1729

#: the relative held-out MSE the interaction model must beat the additive one by before it is called an improvement
DEFAULT_SUPERPOSITION_MARGIN = 0.05

#: the kind an event-rows document declares when the number every actual was read against came out of a model
MODEL_BASED_EXPECTATION = "MODEL_BASED_EXPECTATION"

#: the identification reason that kind carries, named so a reader can tell it from a missing clock
MODEL_BASED_EXPECTATION_REASON = "EXPECTATION_IS_MODEL_BASED"

#: fewer fitting events than this and a projection with a dozen seasonal dummies is fitted through its own noise
MIN_FIT_EVENTS = 20

#: fewer held-out events than this and an out-of-sample MSE is a statement about three numbers
MIN_HOLDOUT_EVENTS = 5

#: the outcomes read off an event row, by name. Adding one means adding it here.
OUTCOMES = ("log_return", "realized_vol")

#: which releases a placebo instant must be clear of. `all_releases` is the strict reading -- a pseudo-event is only
#: a pseudo-event if NOTHING was released near it; `tested_event_types` keeps only the tested pulses away and leaves
#: the rest of the calendar where it is. Declared on every document, never inferred.
PLACEBO_EXCLUSIONS = ("all_releases", "tested_event_types")

_STATSMODELS_REFUSAL = ("STATSMODELS_NOT_AVAILABLE: the Newey-West covariance this module reports comes from "
                        "statsmodels and statsmodels could not be imported; a plain OLS interval reported in its "
                        "place would be narrower than the truth and labelled HAC, so no interval is reported at all")


class ProjectionRefusal(ValueError):
    """An input this job will not fit, carrying the code and naming what is wrong with it."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise ProjectionRefusal(code, why)


def _statsmodels():
    try:
        import statsmodels.api as sm
    except Exception:                                                    # pragma: no cover - environment dependent
        return None
    return sm


# ------------------------------------------------------------------------------- reading 300 MB of rows exactly once

class _JsonStream:
    """A buffered JSON scanner. It exists so a rows document larger than memory is read once, value by value."""

    def __init__(self, handle, chunk_bytes=1 << 22):
        self._handle, self._chunk = handle, int(chunk_bytes)
        self._buffer, self._at, self._eof = "", 0, False
        self._decoder = json.JSONDecoder()

    def _fill(self):
        if self._eof:
            return False
        block = self._handle.read(self._chunk)
        if not block:
            self._eof = True
            return False
        if self._at:
            self._buffer = self._buffer[self._at:]
            self._at = 0
        self._buffer += block
        return True

    def peek(self):
        while True:
            while self._at < len(self._buffer) and self._buffer[self._at] in " \t\r\n":
                self._at += 1
            if self._at < len(self._buffer):
                return self._buffer[self._at]
            if not self._fill():
                return ""

    def take(self, expected):
        got = self.peek()
        if got != expected:
            where = repr(got) if got else "the end of the file"
            _refuse("MALFORMED_ROWS_DOCUMENT", f"expected {expected!r} and the document has {where} there")
        self._at += 1

    def value(self):
        while True:
            self.peek()
            try:
                obj, end = self._decoder.raw_decode(self._buffer, self._at)
            except ValueError as exc:
                if self._fill():
                    continue
                _refuse("MALFORMED_ROWS_DOCUMENT", f"the document does not parse as JSON: {exc}")
            # a number or a literal that runs to the end of the buffer may have been cut in half by the read
            if end >= len(self._buffer) and not self._eof and self._fill():
                continue
            self._at = end
            return obj


def read_rows_document(path, on_row, *, array_key="rows", chunk_bytes=1 << 22):
    """Read the rows document, calling `on_row` once per row, and return everything in it EXCEPT the rows array.

    The rows array is never materialised: on the real archive it is three hundred megabytes and a quarter of a
    million rows, and holding it is the difference between a job that runs inside its memory cap and one that does
    not. Every other key -- the parameters, the clock, the bars digest, the exclusion counts -- is small and is
    returned whole, whether it sits before or after the rows in the file.
    """
    with open(path, encoding="utf-8") as handle:
        stream = _JsonStream(handle, chunk_bytes)
        stream.take("{")
        document = {}
        if stream.peek() == "}":
            stream.take("}")
            return document
        while True:
            key = stream.value()
            if not isinstance(key, str):
                _refuse("MALFORMED_ROWS_DOCUMENT", f"an object key must be a string, got {type(key).__name__}")
            stream.take(":")
            if key == array_key:
                stream.take("[")
                if stream.peek() == "]":
                    stream.take("]")
                else:
                    while True:
                        on_row(stream.value())
                        nxt = stream.peek()
                        if nxt == ",":
                            stream.take(",")
                            continue
                        stream.take("]")
                        break
            else:
                document[key] = stream.value()
            nxt = stream.peek()
            if nxt == ",":
                stream.take(",")
                continue
            stream.take("}")
            break
        return document


def _epoch(text):
    return datetime.fromisoformat(text).timestamp()


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def load(rows_path, *, event_types=None, horizons=None, chunk_bytes=1 << 22):
    """One streaming pass over the rows: the kept rows, the release index, and the document's own header.

    The **release index** holds every release that produced a row -- its instant, its type and its standardized
    surprise -- and it is what the neighbour control is built from. The rows of the real archive were written with
    the neighbour listing capped at zero (the listing of forty neighbours on each of a quarter of a million rows is
    most of the file), so the control cannot always be read off the row; it is reconstructed from this index
    instead, which is the same sum over every neighbour the index knows. What the index does NOT know is a release
    that was excluded by name -- no consensus, not enough history, a path with a hole in it. Such a release had no
    standardized surprise to contribute, except in the one case where it had one and lost all its horizons to a gap
    in the bars, and the document says so rather than pretending the sum is over the whole calendar.
    """
    wanted = set(event_types) if event_types else None
    wanted_h = {int(h) for h in horizons} if horizons else None
    kept, releases = [], {}
    counters = {"rows_read": 0, "rows_kept": 0, "neighbour_listings_complete": 0, "neighbour_listings_capped": 0}

    def on_row(row):
        counters["rows_read"] += 1
        key = row.get("event_key")
        if key is not None and key not in releases:
            releases[key] = (_epoch(row["published_at"]), row["event_type"], _number(row.get("surprise")))
        if wanted is not None and row.get("event_type") not in wanted:
            return
        if wanted_h is not None and int(row.get("horizon_minutes", -1)) not in wanted_h:
            return
        listed, total = row.get("other_releases_in_window_listed"), row.get("other_releases_in_window_count")
        counters["neighbour_listings_complete" if listed == total else "neighbour_listings_capped"] += 1
        counters["rows_kept"] += 1
        kept.append({
            "event_key": key, "event_type": row["event_type"], "published_at": row["published_at"],
            "published_epoch": _epoch(row["published_at"]), "horizon_minutes": int(row["horizon_minutes"]),
            "surprise": _number(row.get("surprise")),
            "log_return": _number(row.get("log_return")), "realized_vol": _number(row.get("realized_vol")),
            "pre_event_realized_vol": _number(row.get("pre_event_realized_vol")),
            "pre_event_status": row.get("pre_event_status"),
            "hour_of_day": int(row["hour_of_day"]), "day_of_week": int(row["day_of_week"]),
        })

    header = read_rows_document(rows_path, on_row, chunk_bytes=chunk_bytes)
    if header.get("schema") != ROWS_SCHEMA:
        _refuse("WRONG_SCHEMA", f"schema is {header.get('schema')!r} and this reader only reads {ROWS_SCHEMA!r}")
    if wanted:
        missing = sorted(wanted - {row["event_type"] for row in kept})
        if missing:
            _refuse("EVENT_TYPE_NOT_IN_THE_ROWS",
                    f"the rows carry no row for {missing}; the event types present are named in the rows document's "
                    f"counts.by_event_type")
    kept.sort(key=lambda r: (r["published_epoch"], r["event_key"], r["horizon_minutes"]))
    order = sorted(releases.items(), key=lambda kv: (kv[1][0], kv[0]))
    index = {
        "keys": [key for key, _ in order],
        "epochs": np.asarray([value[0] for _, value in order], dtype=np.float64),
        "types": [value[1] for _, value in order],
        "surprises": np.asarray([np.nan if value[2] is None else value[2] for _, value in order], dtype=np.float64),
    }
    return {"header": header, "rows": kept, "index": index, "counters": counters}


# --------------------------------------------------------------------------------------------- the neighbour control

def _window_surprises(index, epoch, window_seconds, *, exclude_key=None, types=None):
    """The standardized surprises of the releases inside `[epoch - W, epoch)`, by type. Positive offsets never enter.

    A neighbour at a POSITIVE offset landed after the release this row is about. It is not information anybody had at
    the instant, it is part of what followed, and a control that is a consequence of the treatment opens a path
    rather than closing one. So the slice is strictly the past half of the window, and the document says it.
    """
    epochs = index["epochs"]
    left = int(np.searchsorted(epochs, epoch - window_seconds, side="left"))
    right = int(np.searchsorted(epochs, epoch, side="left"))
    totals = {} if types is None else {name: 0.0 for name in types}
    other = 0.0
    for position in range(left, right):
        if index["keys"][position] == exclude_key:
            continue
        surprise = float(index["surprises"][position])
        if not math.isfinite(surprise):
            continue
        name = index["types"][position]
        if types is not None and name in totals:
            totals[name] += surprise
        else:
            other += surprise
    return totals, other


# ----------------------------------------------------------------------------------------------- the design and fit

def _rank(matrix):
    return int(np.linalg.matrix_rank(matrix)) if matrix.size else 0


def _reduce(matrix, names, protected):
    """Keep the columns a design can carry, dropping the rest BY NAME. Deterministic: left to right, never reordered."""
    keep, dropped = [], []
    for position, name in enumerate(names):
        column = matrix[:, position]
        if name != "const" and position not in protected and float(np.std(column)) <= 0.0:
            dropped.append({"name": name, "why": "CONSTANT_IN_SAMPLE"})
            continue
        trial = matrix[:, keep + [position]]
        if _rank(trial) < len(keep) + 1:
            dropped.append({"name": name, "why": "COLLINEAR_WITH_THE_COLUMNS_BEFORE_IT"})
            continue
        keep.append(position)
    return keep, dropped


def _hac_lags(n):
    """Newey-West's rule of thumb, `4 * (n/100)^(2/9)`, floored, never below one. Declared so it can be recomputed."""
    return max(1, int(math.floor(4.0 * (max(int(n), 1) / 100.0) ** (2.0 / 9.0))))


def _fit(sm, matrix, names, y, *, protected, treatment):
    """One HAC-covariance OLS, or the name of the reason there is none. No number is returned where one does not exist."""
    if sm is None:
        return {"status": "STATSMODELS_NOT_AVAILABLE", "why": _STATSMODELS_REFUSAL}
    n = int(y.size)
    if n < MIN_FIT_EVENTS:
        return {"status": "TOO_FEW_FIT_EVENTS",
                "why": f"{n} fitting event(s) and {MIN_FIT_EVENTS} are declared as the fewest this projection is "
                       f"fitted over"}
    keep, dropped = _reduce(matrix, names, protected)
    kept_names = [names[position] for position in keep]
    for name in treatment:
        if name not in kept_names:
            return {"status": "TREATMENT_NOT_IN_THE_DESIGN",
                    "why": f"{name!r} does not survive the design: it is constant or collinear over the fitting "
                           f"events, so no coefficient on it is estimable and none is reported",
                    "controls_dropped": dropped}
    if n <= len(keep):
        return {"status": "MORE_COLUMNS_THAN_EVENTS",
                "why": f"{len(keep)} column(s) and {n} event(s); the residual degrees of freedom are not positive"}
    lags = _hac_lags(n)
    model = sm.OLS(y, matrix[:, keep])
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    interval = np.asarray(result.conf_int(alpha=0.05), dtype=np.float64)
    coefficients = {}
    for position, name in enumerate(kept_names):
        coefficients[name] = {"value": float(result.params[position]),
                              "std_error": float(result.bse[position]),
                              "ci_lower": float(interval[position, 0]), "ci_upper": float(interval[position, 1])}
    return {"status": "OK", "n": n, "r_squared": float(result.rsquared),
            "hac_maxlags": lags, "covariance": "HAC (Newey-West), statsmodels, small-sample correction on",
            "columns": kept_names, "controls_dropped": dropped,
            "params": np.asarray(result.params, dtype=np.float64), "kept": keep, "names": list(names),
            "coefficients": coefficients}


def _predict(fitted, matrix):
    return matrix[:, fitted["kept"]] @ fitted["params"]


def _public(fitted):
    """The part of a fit that goes into the document: no numpy arrays, no column indices, nothing unprintable."""
    return {key: value for key, value in fitted.items() if key not in ("params", "kept", "names")}


# ------------------------------------------------------------------------------------------- the per-event matrices

def _levels(rows, field):
    return sorted({int(row[field]) for row in rows})


def _matrix(rows, spec):
    """The design of the per-event-type projection, in the declared order: constant, treatment, then the controls."""
    n = len(rows)
    names, columns = ["const"], [np.ones(n, dtype=np.float64)]
    names.append("surprise")
    columns.append(np.asarray([row["surprise"] for row in rows], dtype=np.float64))
    names.append("pre_event_realized_vol")
    columns.append(np.asarray([row["pre_event_realized_vol"] for row in rows], dtype=np.float64))
    for level in spec["hours"][1:]:
        names.append(f"hour_of_day={level}")
        columns.append(np.asarray([1.0 if row["hour_of_day"] == level else 0.0 for row in rows]))
    for level in spec["days"][1:]:
        names.append(f"day_of_week={level}")
        columns.append(np.asarray([1.0 if row["day_of_week"] == level else 0.0 for row in rows]))
    if spec["with_other"]:
        names.append("other_surprises_in_window_negative_offsets")
        columns.append(np.asarray([row["other_surprises"] for row in rows], dtype=np.float64))
    return np.column_stack(columns), names


def _pooled_matrix(rows, spec):
    """The pooled design of the superposition test: one surprise column per tested type, plus the anchor's identity."""
    n = len(rows)
    names, columns = ["const"], [np.ones(n, dtype=np.float64)]
    for name in spec["types"]:
        names.append(f"surprise[{name}]")
        columns.append(np.asarray([row["window_surprise"][name] for row in rows], dtype=np.float64))
    if spec.get("interaction"):
        first, second = spec["interaction"]
        names.append(f"surprise[{first}]*surprise[{second}]")
        columns.append(np.asarray([row["window_surprise"][first] * row["window_surprise"][second] for row in rows],
                                  dtype=np.float64))
    names.append("pre_event_realized_vol")
    columns.append(np.asarray([row["pre_event_realized_vol"] for row in rows], dtype=np.float64))
    for level in spec["hours"][1:]:
        names.append(f"hour_of_day={level}")
        columns.append(np.asarray([1.0 if row["hour_of_day"] == level else 0.0 for row in rows]))
    for level in spec["days"][1:]:
        names.append(f"day_of_week={level}")
        columns.append(np.asarray([1.0 if row["day_of_week"] == level else 0.0 for row in rows]))
    for name in spec["types"][1:]:
        names.append(f"anchor_event_type={name}")
        columns.append(np.asarray([1.0 if row["event_type"] == name else 0.0 for row in rows]))
    names.append("other_surprises_in_window_negative_offsets_untested_types")
    columns.append(np.asarray([row["other_surprises_untested"] for row in rows], dtype=np.float64))
    return np.column_stack(columns), names


def _usable(rows, outcome):
    """A row enters a fit only when every column of it is a number. What is dropped is counted, never imputed."""
    good, dropped = [], {"surprise": 0, outcome: 0, "pre_event_realized_vol": 0}
    for row in rows:
        if row["surprise"] is None or not math.isfinite(row["surprise"]):
            dropped["surprise"] += 1
            continue
        value = row[outcome]
        if value is None or not math.isfinite(value):
            dropped[outcome] += 1
            continue
        pre = row["pre_event_realized_vol"]
        if row["pre_event_status"] != "OK" or pre is None or not math.isfinite(pre):
            dropped["pre_event_realized_vol"] += 1
            continue
        good.append(row)
    return good, dropped


# ------------------------------------------------------------------------------------------------------ the placebo

def _placebo_instants(bars, index, window_seconds, *, horizons, pre_event_minutes, exclude_types, rng, n):
    """Bar instants with nothing released within the exclusion span, whose whole path and pre-span were observed."""
    times, log_price, step = bars["times"], bars["log_price"], bars["step_seconds"]
    if exclude_types is None:
        forbidden = index["epochs"]
    else:
        keep = [position for position, name in enumerate(index["types"]) if name in exclude_types]
        forbidden = index["epochs"][keep] if keep else np.asarray([], dtype=np.float64)
    longest = max(int(h) for h in horizons)
    pre_bars = (int(pre_event_minutes) * 60) // step
    first = int(pre_bars)
    last = int(times.size - 1 - (longest * 60) // step)
    if last <= first:
        return [], 0
    candidates = np.arange(first, last + 1, dtype=np.int64)
    moments = times[candidates]
    if forbidden.size:
        left = np.searchsorted(forbidden, moments - window_seconds, side="left")
        right = np.searchsorted(forbidden, moments + window_seconds, side="right")
        candidates = candidates[(right - left) == 0]
    eligible = []
    for anchor in candidates.tolist():
        if not _contiguous(times, anchor - pre_bars, anchor, step):
            continue
        if not _contiguous(times, anchor, anchor + (longest * 60) // step, step):
            continue
        eligible.append(anchor)
    if not eligible:
        return [], 0
    total = len(eligible)
    chosen = np.asarray(eligible, dtype=np.int64)
    if total > n:
        chosen = np.sort(rng.choice(chosen, size=int(n), replace=False))
    return chosen.tolist(), total


def _placebo_rows(bars, anchors, *, horizon, pre_event_minutes):
    times, log_price, step = bars["times"], bars["log_price"], bars["step_seconds"]
    pre_bars = (int(pre_event_minutes) * 60) // step
    end_bars = (int(horizon) * 60) // step
    rows = []
    for anchor in anchors:
        moment = datetime.fromtimestamp(float(times[anchor]), tz=timezone.utc)
        rows.append({"event_key": f"placebo@{moment.isoformat()}", "event_type": "PLACEBO",
                     "published_at": moment.isoformat(), "published_epoch": float(times[anchor]),
                     "horizon_minutes": int(horizon), "surprise": None,
                     "log_return": float(log_price[anchor + end_bars] - log_price[anchor]),
                     "realized_vol": _realized(log_price, anchor, anchor + end_bars),
                     "pre_event_realized_vol": _realized(log_price, anchor - pre_bars, anchor),
                     "pre_event_status": "OK", "hour_of_day": moment.hour, "day_of_week": moment.weekday(),
                     "other_surprises": 0.0})
    return rows


def _overlap(one, two):
    return not (one[1] < two[0] or two[1] < one[0])


# ------------------------------------------------------------------------------------------------ the whole estimate

def _split_by_time(rows, fraction):
    """The last `fraction` of an event type's RELEASES, by publication instant, never enter a fit.

    The split is on releases and not on rows, so the same release is held out at every horizon: a model fitted on an
    event's five-minute row and scored on its four-hour row has seen the event, whatever the index says.
    """
    keys = sorted({(row["published_epoch"], row["event_key"]) for row in rows})
    if not keys:
        return set(), set()
    held = int(math.ceil(float(fraction) * len(keys)))
    held = min(max(held, 1), len(keys) - 1) if len(keys) > 1 else 0
    fit_keys = {key for _, key in keys[:len(keys) - held]}
    holdout_keys = {key for _, key in keys[len(keys) - held:]}
    return fit_keys, holdout_keys


def _score(fitted, matrix, y):
    residual = y - _predict(fitted, matrix)
    return float(np.mean(residual * residual))


def _naive(fit_rows, holdout_rows, outcome):
    """Rung 1's mean by surprise sign, fitted on the training events and scored on the held-out ones."""
    surprise = np.asarray([row["surprise"] for row in fit_rows], dtype=np.float64)
    values = np.asarray([row[outcome] for row in fit_rows], dtype=np.float64)
    table = naive_response_by_sign(surprise, values)
    fallback = float(np.mean(values)) if values.size else 0.0
    predictions, fallbacks = [], 0
    for row in holdout_rows:
        value, reason = naive_sign_prediction(table, row["surprise"], fallback=fallback)
        fallbacks += 1 if reason else 0
        predictions.append(value)
    return table, np.asarray(predictions, dtype=np.float64), fallbacks, fallback


def prepare(rows_path, *, event_types=None, horizons=None, holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
            window_hours=None, chunk_bytes=1 << 22):
    """The rows, their window surprises and the held-out split -- everything an estimate rests on, and nothing fitted.

    It is a function rather than four lines inside `estimate` because a second reader of the same artifacts exists:
    `evaluate_events.py` scores the fitted projections on the held-out events, and a split or a window sum computed
    a second time, even carefully, is a second definition. There is one here.
    """
    if not 0.0 < float(holdout_fraction) < 1.0:
        _refuse("BAD_HOLDOUT_FRACTION", f"the held-out fraction must lie in (0, 1), got {holdout_fraction}")
    loaded = load(rows_path, event_types=event_types, horizons=horizons, chunk_bytes=chunk_bytes)
    header, rows, index = loaded["header"], loaded["rows"], loaded["index"]
    parameters = header.get("parameters") or {}
    # the neighbour window is a CHOICE, not a property of the rows: the release index holds every release that
    # produced a row with its own instant, so the sum over the past half of a window of any length is computable from
    # the same document. The rows document's own W is the default and the one every artifact written before this
    # argument existed used; a caller that declares another gets it, and `window_hours_used` says which was applied.
    declared_window = float(parameters.get("window_hours", 24.0))
    if window_hours is not None and float(window_hours) <= 0:
        _refuse("BAD_WINDOW", f"the neighbour window must be positive hours, got {window_hours}")
    window_seconds = float(declared_window if window_hours is None else window_hours) * 3600.0
    pre_event_minutes = int(parameters.get("pre_event_minutes", 60))
    types = sorted({row["event_type"] for row in rows})
    used_horizons = sorted({row["horizon_minutes"] for row in rows})
    if not rows:
        # a rows document can be empty for an honest reason -- every release excluded by name, or a calendar that
        # joined nothing -- and the caller must be told that instead of failing somewhere inside the placebo, where
        # the traceback would say nothing about which input was missing
        excluded = ((header.get("excluded") or {}).get("counts")) or {}
        _refuse("NO_EVENT_ROWS",
                f"{rows_path} carries no (release, horizon) row that survived the row builder, so there is nothing "
                f"to project. The builder read {(header.get('counts') or {}).get('releases_read', 0)} release(s) and "
                f"excluded them by name: {excluded}. Nothing is estimated from an empty table, and an empty table is "
                f"not a result of zero")

    # the window surprises, once per release rather than once per row
    per_release = {}
    for row in rows:
        if row["event_key"] in per_release:
            continue
        totals, other = _window_surprises(index, row["published_epoch"], window_seconds,
                                          exclude_key=row["event_key"], types=types)
        per_release[row["event_key"]] = (totals, other)
    for row in rows:
        totals, other = per_release[row["event_key"]]
        own = row["surprise"] if row["surprise"] is not None and math.isfinite(row["surprise"]) else 0.0
        row["other_surprises"] = float(sum(totals.values()) + other)
        row["other_surprises_untested"] = float(other)
        row["window_surprise"] = {name: float(totals[name] + (own if name == row["event_type"] else 0.0))
                                  for name in types}

    splits = {name: _split_by_time([row for row in rows if row["event_type"] == name], holdout_fraction)
              for name in types}
    return {"header": header, "rows": rows, "index": index, "parameters": parameters,
            "window_seconds": window_seconds, "window_hours_used": window_seconds / 3600.0,
            "window_hours_declared_by_the_rows": declared_window,
            "pre_event_minutes": pre_event_minutes,
            "types": types, "horizons": used_horizons, "splits": splits,
            "counters": loaded["counters"], "holdout_fraction": float(holdout_fraction)}


def estimate(rows_path, *, event_types=None, horizons=None, outcomes=OUTCOMES,
             holdout_fraction=DEFAULT_HOLDOUT_FRACTION, placebo_n=DEFAULT_PLACEBO_N, seed=DEFAULT_SEED,
             superposition_margin=DEFAULT_SUPERPOSITION_MARGIN, bars_path=None,
             placebo_excludes="all_releases", placebo_exclusion_hours=None, chunk_bytes=1 << 22):
    """The local projections, the closure rows, the superposition verdict and the placebo -- or refusals by name."""
    if placebo_excludes not in PLACEBO_EXCLUSIONS:
        _refuse("BAD_PLACEBO_EXCLUSION", f"the placebo exclusion must be one of {list(PLACEBO_EXCLUSIONS)}")
    if not 0.0 < float(holdout_fraction) < 1.0:
        _refuse("BAD_HOLDOUT_FRACTION", f"the held-out fraction must lie in (0, 1), got {holdout_fraction}")
    if float(superposition_margin) < 0.0:
        _refuse("BAD_MARGIN", f"the superposition margin must not be negative, got {superposition_margin}")
    outcomes = tuple(outcomes)
    unknown = [name for name in outcomes if name not in OUTCOMES]
    if unknown:
        _refuse("UNKNOWN_OUTCOME", f"{unknown} is not among the outcomes an event row carries, {list(OUTCOMES)}")

    prepared = prepare(rows_path, event_types=event_types, horizons=horizons,
                       holdout_fraction=holdout_fraction, chunk_bytes=chunk_bytes)
    header, rows, index = prepared["header"], prepared["rows"], prepared["index"]
    parameters, window_seconds = prepared["parameters"], prepared["window_seconds"]
    pre_event_minutes = prepared["pre_event_minutes"]
    types, used_horizons, splits = prepared["types"], prepared["horizons"], prepared["splits"]
    sm = _statsmodels()

    projections, closure, fits = [], [], {}
    for name in types:
        fit_keys, holdout_keys = splits[name]
        for horizon in used_horizons:
            group = [row for row in rows if row["event_type"] == name and row["horizon_minutes"] == horizon]
            for outcome in outcomes:
                usable, dropped = _usable(group, outcome)
                fit_rows = [row for row in usable if row["event_key"] in fit_keys]
                holdout_rows = [row for row in usable if row["event_key"] in holdout_keys]
                spec = {"hours": _levels(fit_rows, "hour_of_day") or [0],
                        "days": _levels(fit_rows, "day_of_week") or [0], "with_other": True}
                entry = {"event_type": name, "horizon_minutes": horizon, "outcome": outcome,
                         "n_fit_events": len(fit_rows), "n_holdout_events": len(holdout_rows),
                         "rows_dropped_for_a_missing_number": dropped,
                         "seasonal_levels": {"hour_of_day": spec["hours"], "day_of_week": spec["days"],
                                             "base_level_absorbed_into_the_constant": {
                                                 "hour_of_day": spec["hours"][0], "day_of_week": spec["days"][0]}}}
                if not fit_rows:
                    entry.update({"status": "NO_FITTING_EVENTS",
                                  "why": "no event of this type and horizon carries a surprise, an outcome and a "
                                         "pre-event volatility at once"})
                    projections.append(entry)
                    continue
                matrix, names = _matrix(fit_rows, spec)
                y = np.asarray([row[outcome] for row in fit_rows], dtype=np.float64)
                fitted = _fit(sm, matrix, names, y, protected={1}, treatment=("surprise",))
                entry["fit_event_keys"] = [row["event_key"] for row in fit_rows]
                entry["holdout_event_keys"] = [row["event_key"] for row in holdout_rows]
                entry.update(_public(fitted))
                if fitted["status"] == "OK":
                    beta = fitted["coefficients"]["surprise"]
                    entry["beta"] = beta["value"]
                    entry["beta_ci_95"] = [beta["ci_lower"], beta["ci_upper"]]
                    entry["beta_std_error"] = beta["std_error"]
                    entry["controls"] = [column for column in fitted["columns"]
                                         if column not in ("const", "surprise")]
                    fits[(name, horizon, outcome)] = (fitted, spec)
                projections.append(entry)

                row_closure = {"event_type": name, "horizon_minutes": horizon, "outcome": outcome,
                               "n_fit_events": len(fit_rows), "n_holdout_events": len(holdout_rows)}
                if fitted["status"] != "OK":
                    row_closure.update({"comparability": "NOT_COMPARABLE", "why": fitted["status"],
                                        "local_projection_mse": None, "naive_error": None, "skill": None})
                    closure.append(row_closure)
                    continue
                if len(holdout_rows) < MIN_HOLDOUT_EVENTS:
                    row_closure.update({"comparability": "NOT_COMPARABLE",
                                        "why": f"TOO_FEW_HELD_OUT_EVENTS: {len(holdout_rows)} held-out event(s) and "
                                               f"{MIN_HOLDOUT_EVENTS} are declared as the fewest an out-of-sample "
                                               f"error is reported over",
                                        "local_projection_mse": None, "naive_error": None, "skill": None})
                    closure.append(row_closure)
                    continue
                holdout_matrix, _ = _matrix(holdout_rows, spec)
                truth = np.asarray([row[outcome] for row in holdout_rows], dtype=np.float64)
                model_mse = _score(fitted, holdout_matrix, truth)
                table, naive_predictions, fallbacks, fallback = _naive(fit_rows, holdout_rows, outcome)
                naive_residual = truth - naive_predictions
                naive_mse = float(np.mean(naive_residual * naive_residual))
                unseen = sum(1 for row in holdout_rows
                             if row["hour_of_day"] not in spec["hours"] or row["day_of_week"] not in spec["days"])
                row_closure.update({
                    "scale": "squared log return" if outcome == "log_return" else "squared realized variance",
                    "local_projection_mse": model_mse, "naive_error": naive_mse,
                    "naive_reference": "association.naive_response_by_sign, the mean outcome by surprise sign, "
                                       "FITTED ON THE SAME TRAINING EVENTS and scored on the same held-out events",
                    "naive_table_fitted_on_training_events": table,
                    "naive_predictions_falling_back_to_the_training_mean": fallbacks,
                    "naive_fallback_value": fallback,
                    "skill": (None if naive_mse <= 0 else float(1.0 - model_mse / naive_mse)),
                    "skill_reading": "1 - local_projection_mse / naive_error; positive means the projection beat the "
                                     "naive reference on events neither of them was fitted on",
                    "held_out_events_with_an_unseen_hour_or_weekday": unseen,
                    "unseen_level_rule": "a held-out event whose hour or weekday never occurred among the training "
                                         "events gets the base level's dummies (all zero), which folds it into the "
                                         "constant; it is counted here rather than hidden",
                    "comparability": "COMPARABLE",
                    "why": "both errors are computed on the same held-out events, with the same outcome and scale",
                })
                closure.append(row_closure)

    superposition = _superposition(sm, rows, types, used_horizons, outcomes, splits, superposition_margin)
    exclusion_seconds = (window_seconds if placebo_exclusion_hours is None
                         else float(placebo_exclusion_hours) * 3600.0)
    if exclusion_seconds <= 0:
        _refuse("BAD_PLACEBO_EXCLUSION_SPAN",
                f"the placebo exclusion span must be positive hours, got {placebo_exclusion_hours}")
    placebo = _placebo(sm, header, index, rows, types, used_horizons, outcomes, splits, fits,
                       exclusion_seconds, pre_event_minutes, placebo_n, seed, placebo_excludes, bars_path)

    clock = header.get("publication_clock") or {}
    reasons = []
    # every ASSUMED_SCHEDULED_PUBLICATION mode, localized or not. Measuring what the archive's wall clock MEANT
    # repairs the anchor; it does not turn a scheduled instant into an observed one, and a study that read the
    # localized mode as observed would claim identification from a correction.
    if str(clock.get("mode") or "").startswith("ASSUMED_SCHEDULED_PUBLICATION"):
        reasons.append("ASSUMED_PUBLICATION_CLOCK: " + str(clock.get("identification_caveat")))
    # WP28. The clock can be observed and the EXPECTATION still be a model's. The surprise is then
    # (actual - model forecast) = (actual - market consensus) + (market consensus - model forecast), and the second
    # term is pre-release information: it is a function of exactly the set the identification argument conditions on.
    # That is measurement error in the treatment correlated with the controls -- attenuation and bias, not noise --
    # so the reason is named, carried, and it disqualifies. It is also the reason a consensus feed would remove.
    expectation = header.get("expectation") or {}
    if expectation.get("kind") == MODEL_BASED_EXPECTATION:
        reasons.append(
            f"{MODEL_BASED_EXPECTATION_REASON}: {expectation.get('reading')}. The surprise regressed on here is "
            f"(actual - a model's forecast), which differs from (actual - the market's consensus) by a quantity that "
            f"was itself pre-release information; that is measurement error in the treatment correlated with the "
            f"conditioning set, and it biases every beta below toward zero by an amount nothing in this document "
            f"measures. A consensus feed covering this span removes this reason and no other")
    if placebo.get("status") != "OK":
        reasons.append(f"PLACEBO_NOT_RUN: {placebo.get('status')}")
    else:
        failed = [f"{verdict['event_type']} h={verdict['horizon_minutes']} {verdict['outcome']}"
                  for verdict in placebo["verdicts"] if verdict["verdict"] != "PLACEBO_PASSES"]
        if failed:
            reasons.append(f"PLACEBO_FAILED: {len(failed)} tested (event type, horizon, outcome) did not pass: "
                           f"{failed[:12]}")
        elif not placebo["verdicts"]:
            reasons.append("PLACEBO_TESTED_NOTHING: no (event type, horizon, outcome) reached a placebo verdict")

    return {
        "schema": SCHEMA,
        "provenance": header.get("provenance"),
        "publication_clock": clock,
        "identification": "NOT_IDENTIFIED" if reasons else "PLACEBO_CONSISTENT",
        "identification_reasons": reasons,
        "identification_reading": (
            "NOT_IDENTIFIED is the verdict whenever the release instant was ASSUMED "
            "(ASSUMED_SCHEDULED_PUBLICATION), whenever the placebo could not be run, or whenever any tested (event "
            "type, horizon, outcome) failed it. Under OBSERVED_ACTUAL_PUBLICATION that first reason is gone -- the "
            "instant the actual became public was observed -- and only the consensus's own instant stays assumed, "
            "which is the literature's pre-release information assumption rather than a missing clock. "
            "EXPECTATION_IS_MODEL_BASED is a THIRD reason, independent of both: the release instant may have been "
            "observed and the number the actual was read against still be a model's forecast rather than a market "
            "consensus, and a surprise measured against a model's expectation is not the surprise the market traded "
            "on. "
            "PLACEBO_CONSISTENT is NOT a claim of identification: it says only that the declared checks in this "
            "document did not refute it, and every coefficient below stays a conditional association"),
        "expectation": header.get("expectation"),
        "rows_document": {"path": str(rows_path), "schema": header.get("schema"),
                          "bars": (header.get("bars") or {}).get("path"),
                          "bars_sha256": (header.get("bars") or {}).get("sha256"),
                          "calendar": (header.get("calendar") or {}).get("path"),
                          "calendar_sha256": (header.get("calendar") or {}).get("sha256"),
                          "parameters": parameters, "counts": header.get("counts")},
        "estimator": {
            "name": "local projection (Jorda 2005), one OLS per event type, horizon and outcome",
            "equation": "y[t_k, t_k+h] = alpha + beta[k,h] * s_k(t_k) + gamma' controls(t_k) + eps",
            "covariance": ("HAC (Newey-West) through statsmodels, maxlags = floor(4 * (n/100)^(2/9)), events ordered "
                           "by publication instant before the covariance is formed"),
            "statsmodels": _version(sm),
            "controls_declared": ["pre_event_realized_vol", "hour_of_day dummies (UTC, first level absorbed)",
                                  "day_of_week dummies (UTC, first level absorbed)",
                                  "other_surprises_in_window_negative_offsets"],
            "controls_reading": (
                "the last control is the SUM of the standardized surprises of the other releases inside the declared "
                "window with NEGATIVE offsets only. Releases at POSITIVE offsets landed after this one, are not "
                "pre-release information, and are EXCLUDED: conditioning on them would condition on a consequence of "
                "the treatment. The sum is reconstructed from the releases that produced rows in this document, so a "
                "release the row builder excluded by name -- no consensus, not enough history -- contributes nothing "
                "to it, and that is a gap in the control rather than a zero"),
            "interval": "95 %, from the HAC covariance, on the t distribution statsmodels uses with use_t",
            "minimum_fit_events": MIN_FIT_EVENTS, "minimum_holdout_events": MIN_HOLDOUT_EVENTS,
        },
        "held_out": {"fraction": float(holdout_fraction),
                     "rule": ("the last ceil(fraction * R) RELEASES of each event type by publication instant, at "
                              "every horizon at once; they never enter any fit, any naive table or any superposition "
                              "model"),
                     "by_event_type": {name: {"fit_releases": len(splits[name][0]),
                                              "holdout_releases": len(splits[name][1]),
                                              "holdout_release_keys": sorted(splits[name][1])} for name in types}},
        "event_types": types, "horizons_minutes": used_horizons, "outcomes": list(outcomes),
        "projections": projections,
        "closure_table": closure,
        "superposition": superposition,
        "placebo": placebo,
        "counters": prepared["counters"],
        "seed": int(seed),
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]), "numpy": np.__version__},
        "execution_authorized": False,
        "reading": (
            f"PROVENANCE {header.get('provenance')}, publication clock {clock.get('mode')}. "
            f"{clock.get('identification_caveat')}. RUNG 2 ATTEMPTED, IDENTIFICATION "
            f"{'NOT_IDENTIFIED' if reasons else 'PLACEBO_CONSISTENT'}. Every beta here is the coefficient of a "
            "standardized surprise in a conditional regression over observed releases. It is a response only if the "
            "identification argument holds, and the document above says whether it does. NO_NEW_MEASUREMENT of any "
            "market claim is made by this file: no trading, no profit, no forecast of a price is asserted anywhere "
            "in it, and nothing in it authorises an order"),
    }


def _version(sm):
    if sm is None:
        return None
    try:
        import statsmodels
        return statsmodels.__version__
    except Exception:                                                    # pragma: no cover - environment dependent
        return None


# --------------------------------------------------------------------------------------------- the superposition test

def _superposition(sm, rows, types, horizons, outcomes, splits, margin):
    """Does one window of pulses add up? The additive model against the interaction of the two that co-occur most."""
    block = {"margin": float(margin),
             "rule": ("relative improvement r = (additive_mse - interaction_mse) / additive_mse on the HELD-OUT "
                      "events; r > margin is INTERACTIONS_IMPROVE, r <= margin is ADDITIVE_HOLDS, and a test that "
                      "could not be run at all -- fewer than two event types, no co-occurring pair, a model that did "
                      "not fit, too few held-out events -- is INCONCLUSIVE and names why"),
             "models": {"additive": "y = alpha + sum_j beta_j * s_j(window) + gamma' controls",
                        "interaction": "the additive model plus the product of the two most co-occurring types' "
                                       "window surprises"},
             "window_surprise": ("per tested event type j, the sum of the standardized surprises of every release of "
                                 "type j inside the window with a NEGATIVE offset, plus the anchor's own surprise "
                                 "when the anchor is of type j"),
             "tests": []}
    fit_keys = {name: splits[name][0] for name in types}
    holdout_keys = {name: splits[name][1] for name in types}
    for horizon in horizons:
        for outcome in outcomes:
            group = [row for row in rows if row["horizon_minutes"] == horizon]
            usable, _ = _usable(group, outcome)
            for row in usable:
                row["other_surprises_untested"] = row.get("other_surprises_untested", 0.0)
            fit_rows = [row for row in usable if row["event_key"] in fit_keys[row["event_type"]]]
            holdout_rows = [row for row in usable if row["event_key"] in holdout_keys[row["event_type"]]]
            test = {"horizon_minutes": horizon, "outcome": outcome,
                    "n_fit_events": len(fit_rows), "n_holdout_events": len(holdout_rows)}
            if len(types) < 2:
                test.update({"verdict": "INCONCLUSIVE",
                             "why": "ONLY_ONE_EVENT_TYPE: superposition is a statement about several pulses and only "
                                    "one event type was tested"})
                block["tests"].append(test)
                continue
            if len(holdout_rows) < MIN_HOLDOUT_EVENTS:
                test.update({"verdict": "INCONCLUSIVE",
                             "why": f"TOO_FEW_HELD_OUT_EVENTS: {len(holdout_rows)} and {MIN_HOLDOUT_EVENTS} are "
                                    f"declared as the fewest a held-out comparison is made over"})
                block["tests"].append(test)
                continue
            counts = {}
            for row in fit_rows:
                present = [name for name in types if row["window_surprise"].get(name)]
                for i, first in enumerate(present):
                    for second in present[i + 1:]:
                        counts[(first, second)] = counts.get((first, second), 0) + 1
            test["co_occurrence_counts"] = {f"{a} & {b}": n for (a, b), n in sorted(counts.items())}
            if not counts:
                test.update({"verdict": "INCONCLUSIVE",
                             "why": "NO_CO_OCCURRING_PAIR: no two tested event types have both surprises non-zero in "
                                    "any fitting window, so there is no product term to test"})
                block["tests"].append(test)
                continue
            pair = max(sorted(counts), key=lambda key: counts[key])
            test["pair"] = list(pair)
            test["pair_co_occurrences"] = counts[pair]
            base = {"types": types, "hours": _levels(fit_rows, "hour_of_day") or [0],
                    "days": _levels(fit_rows, "day_of_week") or [0]}
            y_fit = np.asarray([row[outcome] for row in fit_rows], dtype=np.float64)
            truth = np.asarray([row[outcome] for row in holdout_rows], dtype=np.float64)
            results = {}
            for label, spec in (("additive", dict(base)), ("interaction", dict(base, interaction=pair))):
                matrix, names = _pooled_matrix(fit_rows, spec)
                fitted = _fit(sm, matrix, names, y_fit, protected=set(), treatment=())
                if fitted["status"] != "OK":
                    results[label] = (fitted, None)
                    continue
                holdout_matrix, _ = _pooled_matrix(holdout_rows, spec)
                results[label] = (fitted, _score(fitted, holdout_matrix, truth))
            test["additive"] = dict(_public(results["additive"][0]), held_out_mse=results["additive"][1])
            test["interaction"] = dict(_public(results["interaction"][0]), held_out_mse=results["interaction"][1])
            interaction_name = f"surprise[{pair[0]}]*surprise[{pair[1]}]"
            fitted_interaction = results["interaction"][0]
            if results["additive"][1] is None or results["interaction"][1] is None:
                test.update({"verdict": "INCONCLUSIVE",
                             "why": f"A_MODEL_DID_NOT_FIT: additive {results['additive'][0]['status']}, interaction "
                                    f"{results['interaction'][0]['status']}"})
            elif interaction_name not in (fitted_interaction.get("columns") or []):
                test.update({"verdict": "INCONCLUSIVE",
                             "why": f"INTERACTION_NOT_IN_THE_DESIGN: {interaction_name!r} was dropped as constant or "
                                    f"collinear, so the two models are the same model"})
            else:
                additive_mse, interaction_mse = results["additive"][1], results["interaction"][1]
                improvement = (float("inf") if additive_mse <= 0
                               else float((additive_mse - interaction_mse) / additive_mse))
                test["relative_improvement"] = improvement
                test["verdict"] = "INTERACTIONS_IMPROVE" if improvement > float(margin) else "ADDITIVE_HOLDS"
                test["why"] = (f"the interaction model's held-out MSE is {interaction_mse!r} against the additive "
                               f"model's {additive_mse!r}, a relative improvement of {improvement!r} against a "
                               f"declared margin of {float(margin)!r}")
            block["tests"].append(test)
    verdicts = [test["verdict"] for test in block["tests"]]
    block["verdict"] = ("INCONCLUSIVE" if not verdicts or all(v == "INCONCLUSIVE" for v in verdicts)
                        else ("INTERACTIONS_IMPROVE" if "INTERACTIONS_IMPROVE" in verdicts else "ADDITIVE_HOLDS"))
    block["verdict_reading"] = ("the document-level verdict is INTERACTIONS_IMPROVE when the product term buys the "
                               "declared margin at ANY tested horizon and outcome, because superposition failing "
                               "anywhere is superposition failing; the per-test verdicts above are where to look")
    return block


# ------------------------------------------------------------------------------------------------------ the placebo

def _placebo(sm, header, index, rows, types, horizons, outcomes, splits, fits,
             exclusion_seconds, pre_event_minutes, placebo_n, seed, placebo_excludes, bars_path):
    """Pseudo-events with nothing released near them, dressed with a real surprise distribution. Beta must vanish."""
    block = {"n_requested": int(placebo_n), "seed": int(seed), "excludes": placebo_excludes,
             "exclusion_span_hours": exclusion_seconds / 3600.0,
             "rule": ("pseudo-event instants are drawn uniformly without replacement, from the seeded generator, "
                      "among the bar instants that have NO release inside +/- the exclusion span and whose pre-event "
                      "span and longest horizon are both covered by contiguous bars; each is given a surprise drawn "
                      "with replacement from the empirical distribution of the event type's TRAINING surprises, and "
                      "the same projection is fitted, minus the neighbour control, which is identically zero there "
                      "and is dropped by name"),
            "verdict_rule": ("PLACEBO_PASSES when the placebo's 95 % interval for beta contains zero AND does not "
                             "overlap the real interval; anything else is NOT_IDENTIFIED for that (event type, "
                             "horizon, outcome)"),
             "verdicts": []}
    source = bars_path or (header.get("bars") or {}).get("path")
    if not source:
        block["status"] = "NO_BARS_PATH: the rows document does not name the bars it was built from"
        return block
    if not Path(source).is_file():
        block["status"] = (f"BARS_NOT_READABLE: the bars this study needs for pseudo-event outcomes are not on disk "
                           f"at the path the rows document names, and nothing here downloads a price series")
        block["bars_path"] = str(source)
        return block
    declared = header.get("bars") or {}
    try:
        bars = read_bars(source, time_column=declared.get("time_column"), price_column=declared.get("price_column"),
                         timezone_name=declared.get("timezone") or "UTC",
                         timezone_declared=bool(declared.get("timezone_declared")))
    except Exception as exc:
        block["status"] = f"BARS_NOT_READABLE: {exc}"
        return block
    block["bars"] = {"path": bars["path"], "sha256": bars["sha256"], "rows": bars["rows"],
                     "step_seconds": bars["step_seconds"], "first": bars["first"], "last": bars["last"]}
    if declared.get("sha256") and declared["sha256"] != bars["sha256"]:
        block["status"] = (f"BARS_DIGEST_MISMATCH: the file at that path is not the one the rows were built from "
                           f"({declared['sha256']} was recorded, {bars['sha256']} is there now), and a placebo drawn "
                           f"from a different price series would not be a placebo for these rows")
        return block
    rng = np.random.default_rng(int(seed))
    exclude = None if placebo_excludes == "all_releases" else set(types)
    anchors, eligible = _placebo_instants(bars, index, exclusion_seconds, horizons=horizons,
                                          pre_event_minutes=pre_event_minutes, exclude_types=exclude,
                                          rng=rng, n=placebo_n)
    block["eligible_instants"] = int(eligible)
    block["n_drawn"] = len(anchors)
    if not anchors:
        block["status"] = ("NO_ELIGIBLE_PLACEBO_INSTANTS: no bar instant in this series has an empty "
                           f"+/-{exclusion_seconds / 3600.0:g} hour neighbourhood under the declared exclusion "
                           f"{placebo_excludes!r} with its whole path observed; on a calendar this dense there is no "
                           "quiet instant to compare against, so the placebo was NOT RUN and nothing here is "
                           "identified by it")
        return block
    if len(anchors) < MIN_FIT_EVENTS:
        block["status"] = (f"TOO_FEW_PLACEBO_INSTANTS: {len(anchors)} drawn and {MIN_FIT_EVENTS} are declared as the "
                           f"fewest a projection is fitted over")
        return block
    block["status"] = "OK"
    pseudo = {horizon: _placebo_rows(bars, anchors, horizon=horizon, pre_event_minutes=pre_event_minutes)
              for horizon in horizons}
    for order, name in enumerate(types):
        stream = np.random.default_rng([int(seed), order])
        fit_keys = splits[name][0]
        for horizon in horizons:
            for outcome in outcomes:
                key = (name, horizon, outcome)
                verdict = {"event_type": name, "horizon_minutes": horizon, "outcome": outcome}
                real = fits.get(key)
                if real is None:
                    verdict.update({"verdict": "NOT_IDENTIFIED",
                                    "why": "NO_REAL_PROJECTION: the real projection for this (event type, horizon, "
                                           "outcome) did not fit, so there is nothing for a placebo to be compared "
                                           "against"})
                    block["verdicts"].append(verdict)
                    continue
                pool = np.asarray([row["surprise"] for row in rows
                                   if row["event_type"] == name and row["horizon_minutes"] == horizon
                                   and row["event_key"] in fit_keys and row["surprise"] is not None
                                   and math.isfinite(row["surprise"])], dtype=np.float64)
                if pool.size == 0:
                    verdict.update({"verdict": "NOT_IDENTIFIED", "why": "NO_TRAINING_SURPRISES_TO_DRAW_FROM"})
                    block["verdicts"].append(verdict)
                    continue
                drawn = stream.choice(pool, size=len(anchors), replace=True)
                sample = [dict(row, surprise=float(value)) for row, value in zip(pseudo[horizon], drawn)]
                spec = {"hours": _levels(sample, "hour_of_day") or [0], "days": _levels(sample, "day_of_week") or [0],
                        "with_other": False}
                matrix, names = _matrix(sample, spec)
                y = np.asarray([row[outcome] for row in sample], dtype=np.float64)
                fitted = _fit(sm, matrix, names, y, protected={1}, treatment=("surprise",))
                verdict["placebo_fit"] = _public(fitted)
                if fitted["status"] != "OK":
                    verdict.update({"verdict": "NOT_IDENTIFIED",
                                    "why": f"PLACEBO_DID_NOT_FIT: {fitted['status']}"})
                    block["verdicts"].append(verdict)
                    continue
                placebo_beta = fitted["coefficients"]["surprise"]
                real_beta = real[0]["coefficients"]["surprise"]
                placebo_interval = (placebo_beta["ci_lower"], placebo_beta["ci_upper"])
                real_interval = (real_beta["ci_lower"], real_beta["ci_upper"])
                contains_zero = placebo_interval[0] <= 0.0 <= placebo_interval[1]
                overlaps = _overlap(real_interval, placebo_interval)
                verdict.update({"placebo_beta": placebo_beta["value"], "placebo_ci_95": list(placebo_interval),
                                "real_beta": real_beta["value"], "real_ci_95": list(real_interval),
                                "placebo_interval_contains_zero": bool(contains_zero),
                                "intervals_overlap": bool(overlaps)})
                if contains_zero and not overlaps:
                    verdict.update({"verdict": "PLACEBO_PASSES",
                                    "why": "the placebo interval contains zero and the real interval does not touch "
                                           "it"})
                else:
                    verdict.update({"verdict": "NOT_IDENTIFIED",
                                    "why": ("the placebo interval does not contain zero" if not contains_zero
                                            else "the real interval overlaps the placebo interval, so the estimate "
                                                 "is not distinguishable from what pure timing produces")})
                block["verdicts"].append(verdict)
    return block


# --------------------------------------------------------------------------------------------------------- the CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.local_projections",
        description="Local projections (Jorda) of the event rows, with HAC intervals, held-out events, the "
                    "superposition test and the placebo. Nothing here claims identification under an assumed clock.")
    parser.add_argument("--rows", required=True, help="the event rows document written by feature_eng_m5phet.events")
    parser.add_argument("--out", help="where to write the projections document; stdout when absent")
    parser.add_argument("--event-types", nargs="+", help="fit only these event types; all of them when absent")
    parser.add_argument("--horizons", type=int, nargs="+", help="fit only these horizons in minutes")
    parser.add_argument("--outcomes", nargs="+", default=list(OUTCOMES), choices=list(OUTCOMES))
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--placebo-n", type=int, default=DEFAULT_PLACEBO_N)
    parser.add_argument("--placebo-excludes", choices=list(PLACEBO_EXCLUSIONS), default="all_releases",
                        help="which releases a pseudo-event instant must be clear of")
    parser.add_argument("--placebo-exclusion-hours", type=float,
                        help="how far from a release a pseudo-event instant must be; the rows document's own window "
                             "when absent")
    parser.add_argument("--superposition-margin", type=float, default=DEFAULT_SUPERPOSITION_MARGIN)
    parser.add_argument("--bars", help="the price bars for the placebo; the path recorded in the rows document when "
                                       "absent")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-bytes", type=int, default=1 << 22, help="the streaming read size for the rows")
    args = parser.parse_args(argv)
    try:
        document = estimate(args.rows, event_types=args.event_types, horizons=args.horizons,
                            outcomes=tuple(args.outcomes), holdout_fraction=args.holdout_fraction,
                            placebo_n=args.placebo_n, seed=args.seed,
                            superposition_margin=args.superposition_margin, bars_path=args.bars,
                            placebo_excludes=args.placebo_excludes,
                            placebo_exclusion_hours=args.placebo_exclusion_hours, chunk_bytes=args.chunk_bytes)
    except ProjectionRefusal as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=False, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"{len(document['projections'])} projection(s) written to {args.out}; "
              f"identification {document['identification']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
