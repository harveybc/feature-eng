"""Design temporal representations from a dataset profile alone -- no model is fitted here.

WP06 stage 2. The job reads one CSV and nothing else: its columns, their types, the step its timestamps sit on, how
many rows there are, how many are missing. From that profile it runs the tests that already exist in this package's
installed dependencies -- ADF and KPSS from statsmodels when statsmodels is installed, autocorrelation from numpy,
sampling regularity and missingness from the timestamps themselves -- and emits a list of **candidate representation
specs**, each carrying the test result that motivated its windows and its lags.

Three commitments hold the job honest, and each of them is the opposite of something a "smart" version would do:

* **A candidate names its reason.** Every candidate carries a `why` block in which each choice is written as the
  measurement that produced it: `ACF peak at lag 1440 (rho 0.412, band 0.009) -> window 1440, lags [1, 1440]`. A
  window nobody can trace to a number in `tests` is a window somebody guessed.
* **What the data cannot decide is listed, not defaulted quietly.** A CSV cannot say whether its timestamps are a
  publication clock or a receipt clock, and it cannot say where a holdout should be cut. Both appear in every
  candidate's `not_decided` block, with the value the job put there so the spec would validate and the statement that
  a person must replace it. `--clock` removes the first entry by having somebody declare it.
* **An absent test says so.** Without statsmodels the ADF and KPSS blocks read `NOT_AVAILABLE` with the reason, and
  the stationarity verdict comes from a named autocorrelation heuristic that is reported as a heuristic and never as
  a test it is not.

Everything here is deterministic and CPU only: no seeds are drawn, no fit is performed, and the same file with the
same arguments produces the same bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import warnings
from datetime import datetime, timezone as _timezone
from pathlib import Path

import numpy as np

from . import representation
from .representation import SCHEMA as SPEC_SCHEMA, validate_spec

SCHEMA = "m5phet.representation_design.v1"

#: the column names this job recognises as a timestamp when `--time-column` is not given. Matching is
#: case-insensitive; `timestamp_label` is what the crispdm data foundation's public panels call theirs, `DATE_TIME`
#: what `app/column_roles.py` uses in its contract example.
TIME_COLUMN_NAMES = ("timestamp_label", "date_time", "datetime", "timestamp", "time", "date", "ds")

#: cells read as missing rather than as a number. `?` is the UCI household panel's missing marker and is declared in
#: that panel's parse receipt; the list is reported in the output so nobody has to guess what was counted as absent.
MISSING_TOKENS = ("", "?", "NA", "N/A", "NaN", "nan", "null", "NULL", "None")

#: timestamp formats tried, in order, after ISO-8601. The first that reads the first nonempty value is locked for the
#: whole column: a file whose rows need two different parsers is a file with two different clocks in one column.
TIME_FORMATS = ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M", "%d/%m/%Y", "%Y-%m-%d")

MIN_ROWS = 32                       #: below this no autocorrelation of a seasonal lag means anything
STATIONARITY_MAX_ROWS = 20000       #: ADF/KPSS read the HEAD of the series: never the end a holdout will be cut from
ACF_MAX_ROWS = 100000
ACF_MAX_LAG = 20160                 #: two weeks of one-minute steps; enough for a weekly period on the finest grid
MAX_PEAKS = 3                       #: how many autocorrelation peaks become seasonal candidates
#: two local maxima this close in lag are one bump of the estimator, not two periods; keeping both would emit two
#: candidates that differ by noise and read as two findings
PEAK_MIN_SEPARATION = 0.1
MAX_WINDOW_FRACTION = 0.25          #: a window longer than a quarter of the series leaves too few origins to fit on
DEFAULT_HOLDOUT_FRACTION = 0.2
DEFAULT_CLOCK = "receipt"

#: periods a calendar feature encodes, in seconds, with the feature `feature-eng` builds for each
CALENDAR_PERIODS = ((86400.0, "hour_of_day"), (604800.0, "day_of_week"), (2629746.0, "day_of_month"))


class DesignRefusal(ValueError):
    """A dataset this job will not design from, carrying the code and naming what is wrong with the file."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise DesignRefusal(code, why)


def _statsmodels_version(module):
    """The version the tests ran under, or the word that says they did not run."""
    if module is None:
        return "NOT_AVAILABLE"
    import statsmodels
    return statsmodels.__version__


def _statsmodels():
    """The stationarity tests, or None. Kept behind a function so a test can remove them without removing them."""
    try:
        from statsmodels.tsa import stattools
    except ImportError:
        return None
    return stattools


# --------------------------------------------------------------------------------------------- reading the dataset

def _delimiter(sample):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _time_parser(value):
    """The one parser that reads this column, chosen from the first nonempty cell and then never changed."""
    try:
        datetime.fromisoformat(value)
        return "iso8601", (lambda text: datetime.fromisoformat(text))
    except ValueError:
        pass
    for fmt in TIME_FORMATS:
        try:
            datetime.strptime(value, fmt)
            return fmt, (lambda text, fmt=fmt: datetime.strptime(text, fmt))
        except ValueError:
            continue
    if value.lstrip("-").isdigit():
        return "epoch_seconds", (lambda text: datetime.fromtimestamp(int(text), _timezone.utc))
    _refuse("TIMESTAMP_UNPARSEABLE",
            f"the time column's first value {value!r} is neither ISO-8601, nor epoch seconds, nor one of "
            f"{list(TIME_FORMATS)}")


def read_table(path, *, time_column=None, max_rows=None):
    """Columns, types, timestamps and row count. The only thing in this module that touches the caller's data."""
    path = Path(path)
    if not path.is_file():
        _refuse("NO_SUCH_FILE", f"{path} is not a file this job can read")
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        reader = csv.reader(handle, delimiter=_delimiter(sample))
        try:
            header = next(reader)
        except StopIteration:
            _refuse("EMPTY_FILE", f"{path} has no header row")
        header = [name.strip() for name in header]
        if len(set(header)) != len(header):
            _refuse("DUPLICATE_COLUMN", f"the header repeats a column name: {header}")
        if time_column is None:
            found = [name for name in header if name.lower() in TIME_COLUMN_NAMES]
            if not found:
                _refuse("NO_TIMESTAMP_COLUMN",
                        f"the file's columns are {header} and none of them is named as a timestamp "
                        f"({list(TIME_COLUMN_NAMES)}); a representation cannot declare a sampling step over rows "
                        f"whose instants are unknown -- name the column with --time-column")
            time_column = found[0]
        elif time_column not in header:
            _refuse("TIME_COLUMN_NOT_IN_DATASET",
                    f"--time-column {time_column!r} is not a column of this file; its columns are {header}")
        index_of = {name: i for i, name in enumerate(header)}
        time_index = index_of[time_column]
        cells = {name: [] for name in header if name != time_column}
        times, parser, fmt_name = [], None, None
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                _refuse("RAGGED_ROW", f"row {row_number} has {len(row)} fields and the header has {len(header)}")
            stamp = row[time_index].strip()
            if stamp in MISSING_TOKENS:
                _refuse("MISSING_TIMESTAMP", f"row {row_number} has no value in the time column {time_column!r}")
            if parser is None:
                fmt_name, parser = _time_parser(stamp)
            try:
                times.append(parser(stamp))
            except ValueError:
                _refuse("TIMESTAMP_UNPARSEABLE",
                        f"row {row_number} reads {stamp!r}, which the parser chosen from the first row "
                        f"({fmt_name}) cannot read")
            for name, i in index_of.items():
                if name != time_column:
                    cells[name].append(row[i].strip())
            if max_rows is not None and len(times) >= max_rows:
                break
    if len(times) < MIN_ROWS:
        _refuse("TOO_FEW_ROWS", f"{len(times)} rows were read and no autocorrelation over {MIN_ROWS} rows says "
                                f"anything about a period")
    return {"path": str(path), "sha256": _file_digest(path), "columns": header, "time_column": time_column,
            "time_format": fmt_name, "rows_read": len(times), "times": times,
            "truncated": max_rows is not None and len(times) >= max_rows, "cells": cells}


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric(values):
    """One column as float64 with missing cells as NaN, or None when a cell is neither a number nor a missing token."""
    out = np.empty(len(values), dtype=np.float64)
    for i, text in enumerate(values):
        if text in MISSING_TOKENS:
            out[i] = np.nan
            continue
        try:
            out[i] = float(text)
        except ValueError:
            return None, text
    return out, None


# ----------------------------------------------------------------------------------------------------- the tests

def sampling_test(times):
    """The grid the rows actually sit on: the modal step, and every row that does not honour it."""
    seconds = np.array([t.timestamp() for t in times], dtype=np.float64)
    deltas = np.diff(seconds)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        _refuse("IRREGULAR_SAMPLING", "no two consecutive rows advance in time; there is no sampling step to declare")
    values, counts = np.unique(np.rint(positive).astype(np.int64), return_counts=True)
    step = int(values[int(np.argmax(counts))])
    regular = int(np.count_nonzero(np.rint(deltas).astype(np.int64) == step))
    aware = times[0].tzinfo is not None
    return {"step_seconds": step, "step_rule": "the most frequent positive difference between consecutive timestamps",
            "regular_fraction": round(regular / deltas.size, 6), "intervals": int(deltas.size),
            "gaps_longer_than_step": int(np.count_nonzero(np.rint(deltas).astype(np.int64) > step)),
            "repeated_timestamps": int(np.count_nonzero(deltas == 0)),
            "backwards_steps": int(np.count_nonzero(deltas < 0)),
            "min_step_seconds": float(deltas.min()), "max_step_seconds": float(deltas.max()),
            "first": times[0].isoformat(), "last": times[-1].isoformat(),
            "timestamps_carry_an_offset": bool(aware),
            "note": ("the autocorrelation and stationarity tests below read the rows as EQUALLY spaced; where "
                     "regular_fraction is below 1.0 that reading is an approximation the file does not support")}


def missingness_test(table, numeric, non_numeric):
    """How much of each column is absent, counted against the declared missing tokens."""
    rows = table["rows_read"]
    per_column = {name: {"missing": int(np.count_nonzero(~np.isfinite(values))),
                         "missing_fraction": round(float(np.count_nonzero(~np.isfinite(values)) / rows), 6)}
                  for name, values in numeric.items()}
    return {"rows": rows, "missing_tokens": list(MISSING_TOKENS), "numeric_columns": sorted(numeric),
            "non_numeric_columns": {name: f"first unreadable value {value!r}" for name, value in non_numeric.items()},
            "per_column": per_column}


def longest_finite_run(values):
    """The longest stretch of the target with no missing cell: what the tests are computed on, and named as such."""
    finite = np.isfinite(values)
    best_start = best_length = start = length = 0
    for i, ok in enumerate(finite):
        if ok:
            if length == 0:
                start = i
            length += 1
            if length > best_length:
                best_start, best_length = start, length
        else:
            length = 0
    return best_start, best_length


def autocorrelation(series, max_lag):
    """The biased autocorrelation estimate, by FFT. Deterministic, and the only estimator used anywhere here."""
    centred = series - series.mean()
    size = 1 << int(math.ceil(math.log2(max(4, 2 * centred.size - 1))))
    spectrum = np.fft.rfft(centred, size)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: max_lag + 1]
    if acf[0] <= 0:
        return None
    return acf / acf[0]


def stationarity_test(series, statsmodels):
    """ADF and KPSS when statsmodels is installed; a named autocorrelation heuristic either way.

    The heuristic exists so that the job still reaches a verdict when the tests are absent, and so that the two can be
    read against each other when they are present. It is reported as what it is -- a rule over two autocorrelation
    numbers -- and it never borrows the name of a hypothesis test.
    """
    head = series[:STATIONARITY_MAX_ROWS]
    block = {"rows_used": int(head.size),
             "rows_rule": f"the first {STATIONARITY_MAX_ROWS} rows of the analysis segment at most; the head, never "
                          f"the end a holdout will be cut from"}
    lag_count = max(2, min(20, head.size // 10))
    acf = autocorrelation(head, lag_count)
    rho1 = float(acf[1]) if acf is not None else float("nan")
    persistence = float(np.mean(acf[1:lag_count + 1])) if acf is not None else float("nan")
    non_stationary = bool(acf is not None and rho1 >= 0.95 and persistence >= 0.5)
    block["heuristic"] = {
        "method": "autocorrelation persistence of the levels",
        "lag1_autocorrelation": None if acf is None else round(rho1, 6),
        "mean_autocorrelation_to_lag": lag_count,
        "mean_autocorrelation": None if acf is None else round(persistence, 6),
        "rule": "lag-1 autocorrelation >= 0.95 AND the mean autocorrelation over those lags >= 0.5 -> NON_STATIONARY",
        "verdict": "NON_STATIONARY" if non_stationary else "STATIONARY",
        "caveat": "this is a rule over two numbers, not a hypothesis test, and carries no p-value"}
    if statsmodels is None:
        reason = "statsmodels is not installed in the environment this job ran in"
        block["adf"] = {"status": "NOT_AVAILABLE", "reason": reason}
        block["kpss"] = {"status": "NOT_AVAILABLE", "reason": reason}
        block["verdict"] = block["heuristic"]["verdict"]
        block["verdict_source"] = "heuristic"
        return block
    try:                                           # statsmodels 0.15 announces a return-type change; ask for today's
        result = statsmodels.adfuller(head, autolag="AIC", result_object=False)
    except TypeError:                              # an older statsmodels has no such argument and already returns it
        result = statsmodels.adfuller(head, autolag="AIC")
    stat, p_value, used_lag, nobs = result[:4]
    block["adf"] = {"status": "OK", "statistic": round(float(stat), 6), "p_value": round(float(p_value), 6),
                    "used_lag": int(used_lag), "nobs": int(nobs), "autolag": "AIC",
                    "null_hypothesis": "a unit root is present",
                    "verdict": "STATIONARY" if p_value <= 0.05 else "NON_STATIONARY",
                    "rule": "p <= 0.05 rejects the unit root -> STATIONARY"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # statsmodels warns when the p-value falls off its lookup table
        k_stat, k_p, k_lags = statsmodels.kpss(head, regression="c", nlags="auto")[:3]
    block["kpss"] = {"status": "OK", "statistic": round(float(k_stat), 6), "p_value": round(float(k_p), 6),
                     "lags": int(k_lags), "regression": "c",
                     "null_hypothesis": "the series is stationary around a constant",
                     "p_value_is_bounded": bool(k_p <= 0.01 or k_p >= 0.1),
                     "verdict": "NON_STATIONARY" if k_p <= 0.05 else "STATIONARY",
                     "rule": "p <= 0.05 rejects stationarity -> NON_STATIONARY"}
    if block["adf"]["verdict"] == block["kpss"]["verdict"]:
        block["verdict"], block["verdict_source"] = block["adf"]["verdict"], "adf+kpss agree"
    else:
        block["verdict"], block["verdict_source"] = "INCONCLUSIVE", "adf and kpss disagree"
    return block


def seasonality_test(series, step_seconds, stationarity):
    """Autocorrelation peaks, on the levels when they are stationary and on the first difference when they are not.

    The autocorrelation of a non-stationary series decays with the sample, not with the process, and its "peaks" are
    an artefact of that decay. Differencing first is what makes a peak mean a period; the block says which series it
    read, so nobody has to infer it.
    """
    on_levels = stationarity["verdict"] == "STATIONARY"
    work = series[:ACF_MAX_ROWS] if on_levels else np.diff(series[:ACF_MAX_ROWS])
    max_lag = int(min(work.size // 2, ACF_MAX_LAG))
    block = {"computed_on": "levels" if on_levels else "first difference",
             "computed_on_reason": ("the stationarity verdict is STATIONARY, so the levels carry the periodicity"
                                    if on_levels else
                                    f"the stationarity verdict is {stationarity['verdict']}, and the "
                                    f"autocorrelation of such a series decays with the sample rather than with the "
                                    f"process; peaks are read on the first difference instead"),
             "rows_used": int(work.size), "max_lag": max_lag,
             "estimator": "biased autocorrelation by FFT, normalised at lag 0"}
    if max_lag < 3:
        block.update(status="NOT_AVAILABLE", reason="too few rows for any lag", peaks=[], significance_band=None)
        return block
    acf = autocorrelation(work, max_lag)
    if acf is None:
        block.update(status="NOT_AVAILABLE", reason="the series has zero variance", peaks=[], significance_band=None)
        return block
    band = 1.96 / math.sqrt(work.size)
    peaks = [{"lag": int(k), "autocorrelation": round(float(acf[k]), 6),
              "period_seconds": int(k) * step_seconds}
             for k in range(2, max_lag)
             if acf[k] > band and acf[k] > acf[k - 1] and acf[k] >= acf[k + 1]]
    peaks.sort(key=lambda peak: (-peak["autocorrelation"], peak["lag"]))
    selected = []
    for peak in peaks:
        if all(abs(peak["lag"] - kept["lag"]) > PEAK_MIN_SEPARATION * max(peak["lag"], kept["lag"])
               for kept in selected):
            selected.append(peak)
        if len(selected) == MAX_PEAKS:
            break
    decay = next((int(k) for k in range(1, max_lag + 1) if acf[k] < band), None)
    block.update(status="OK",
                 significance_band=round(band, 6),
                 band_rule="1.96 / sqrt(rows_used), the white-noise band of the estimator",
                 peaks=selected,
                 peaks_found=len(peaks),
                 peak_rule=f"the highest local maxima outside the band, one per bump: a peak within "
                           f"{PEAK_MIN_SEPARATION:.0%} of the lag of a higher one is the same bump of the estimator "
                           f"and not a second period",
                 decay_lag=decay,
                 decay_rule="the first lag whose autocorrelation falls inside the band")
    return block


# ------------------------------------------------------------------------------------------------ the candidates

def _calendar_features(period_seconds, step_seconds):
    """The calendar feature a period of this length asks for, when the grid is fine enough to carry it."""
    for length, feature in CALENDAR_PERIODS:
        if abs(period_seconds - length) <= max(float(step_seconds), 0.02 * length) and step_seconds < length:
            return [feature]
    return []


def _candidate(*, candidate_id, sampling, target, transform, windows, lags, features, why, not_decided,
               exogenous, provenance, clock, holdout):
    spec = {
        "schema": SPEC_SCHEMA,
        "sampling": sampling,
        "target": {"column": target, "transform": transform},
        "windows": sorted(set(windows)),
        "lags": sorted(set(lags)),
        "differencing": {"order": 0},
        "calendar": {"clock": clock, "columns": list(features)},
        "features": list(features),
        "holdout": holdout,
        "provenance": provenance,
        "candidate_id": candidate_id,
        "why": why,
        "not_decided": dict(not_decided),
    }
    if exogenous:
        spec["exogenous"] = list(exogenous)
    return validate_spec(spec)


def build_candidates(tests, *, target, exogenous, provenance, clock, clock_declared, holdout, holdout_declared,
                     timezone_name, timezone_declared, rows):
    """One candidate per motivated reading of the tests, each carrying the measurement that motivated it."""
    sampling = {"step_seconds": tests["sampling"]["step_seconds"], "timezone": timezone_name}
    step = sampling["step_seconds"]
    seasonality, stationarity = tests["seasonality"], tests["stationarity"]
    max_window = max(2, int(rows * MAX_WINDOW_FRACTION))
    not_decided = {}
    if not clock_declared:
        not_decided["calendar.clock"] = (
            f"a file of timestamps cannot say whether they are a publication clock or a receipt clock; {clock!r} is "
            f"written so the spec validates and a person must confirm it (--clock)")
    if not holdout_declared:
        not_decided["holdout"] = (
            f"the cut is a decision about what must stay unseen, which no test on the data can make; "
            f"{json.dumps(holdout)} is a convention, not a measurement")
    if not timezone_declared:
        not_decided["sampling.timezone"] = (
            f"the timestamps carry no offset, so {timezone_name!r} is what --timezone declared, not what was read")
    not_decided["exogenous"] = ("the other numeric columns are offered as names only; whether each is available at "
                               "prediction time is not a property of this file")
    common = dict(sampling=sampling, target=target, exogenous=exogenous, provenance=provenance,
                  clock=clock, holdout=holdout)
    candidates, skipped = [], []

    decay = seasonality.get("decay_lag")
    if decay:
        window = min(max(decay, 2), max_window)
        why_window = (f"the autocorrelation of the {seasonality['computed_on']} falls inside the "
                      f"+-{seasonality['significance_band']} band at lag {decay} -> window {window}")
        why_lags = f"lag 1 always, and the decay lag {window} as the last lag still outside the band"
    else:
        window = max_window
        why_window = (f"the autocorrelation of the {seasonality['computed_on']} never falls inside the band up to "
                      f"lag {seasonality.get('max_lag')} -> the window is the cap of {MAX_WINDOW_FRACTION} of the "
                      f"{rows} analysed rows, which is a cap and not a measurement")
        why_lags = "lag 1 always; no decay lag was measured, so no second lag is motivated"
    candidates.append(_candidate(
        candidate_id="short_memory", transform="level", windows=[window],
        lags=sorted({1, window}) if decay else [1], features=[],
        why={"windows": why_window, "lags": why_lags,
             "transform": f"level: the stationarity verdict is {stationarity['verdict']} "
                          f"({stationarity['verdict_source']})",
             "features": "none: no periodic peak motivates a calendar feature in this candidate"},
        not_decided=not_decided if decay else
        {**not_decided, "windows": "no autocorrelation decay was measured on this series, so this candidate's "
                                   "window is a cap rather than something the data chose"},
        **common))

    for peak in seasonality.get("peaks", []):
        lag, rho = peak["lag"], peak["autocorrelation"]
        if lag > max_window:
            skipped.append({"lag": lag, "reason": f"a window of {lag} exceeds {MAX_WINDOW_FRACTION} of the "
                                                  f"{rows} analysed rows"})
            continue
        features = _calendar_features(peak["period_seconds"], step)
        windows = sorted({lag} | ({decay} if decay and decay < lag else set()))
        candidates.append(_candidate(
            candidate_id=f"seasonal_lag_{lag}", transform="level", windows=windows, lags=sorted({1, lag}),
            features=features,
            why={"windows": f"ACF peak at lag {lag} (rho {rho}, band {seasonality['significance_band']}, computed on "
                            f"the {seasonality['computed_on']}) -> window {lag}"
                            + (f"; the decay lag {decay} is kept as the shorter window" if decay and decay < lag
                               else ""),
                 "lags": f"ACF peak at lag {lag} -> lags [1, {lag}]",
                 "transform": f"level: the stationarity verdict is {stationarity['verdict']} "
                              f"({stationarity['verdict_source']})",
                 "features": (f"the peak's period is {peak['period_seconds']} s, which is the period "
                              f"{features[0]} encodes -> features {features}") if features else
                             "none: the peak's period matches no calendar period feature-eng builds a feature for"},
            not_decided=not_decided, **common))

    if stationarity["verdict"] in ("NON_STATIONARY", "INCONCLUSIVE"):
        lags = sorted({1} | {peak["lag"] for peak in seasonality.get("peaks", []) if peak["lag"] <= max_window})
        windows = sorted({min(max(decay or 2, 2), max_window)} | {lag for lag in lags if lag > 1})
        features = sorted({feature for peak in seasonality.get("peaks", [])
                           for feature in _calendar_features(peak["period_seconds"], step)
                           if peak["lag"] <= max_window})
        candidates.append(_candidate(
            candidate_id="differenced", transform="diff", windows=windows, lags=lags, features=features,
            why={"transform": f"the stationarity verdict is {stationarity['verdict']} "
                              f"({stationarity['verdict_source']}); adf "
                              f"{stationarity['adf'].get('p_value', stationarity['adf'].get('status'))}, kpss "
                              f"{stationarity['kpss'].get('p_value', stationarity['kpss'].get('status'))}, lag-1 "
                              f"autocorrelation {stationarity['heuristic']['lag1_autocorrelation']} -> the target is "
                              f"modelled as its first difference",
                 "windows": f"windows {windows} from the same peaks and decay lag, read on the "
                            f"{seasonality['computed_on']}",
                 "lags": f"lag 1, and the peaks of the differenced series {lags[1:]}" if len(lags) > 1
                         else "lag 1 only: the differenced series has no peak outside the band",
                 "features": f"the peaks' periods motivate {features}" if features else
                             "none: no peak matches a calendar period"},
            not_decided=not_decided, **common))
    return candidates, skipped


# ---------------------------------------------------------------------------------------------------- the job

def design(data, target, *, time_column=None, timezone_name=None, provenance="DEVELOPMENT", clock=None,
           holdout=None, max_rows=None):
    """Read one dataset, run the tests, emit the candidates. Nothing is fitted and nothing is trained."""
    if provenance not in representation.PROVENANCE:
        _refuse("UNKNOWN_PROVENANCE", f"provenance {provenance!r} is not one of {list(representation.PROVENANCE)}")
    table = read_table(data, time_column=time_column, max_rows=max_rows)
    if target == table["time_column"]:
        _refuse("TARGET_IS_THE_TIME_COLUMN", f"{target!r} is the time column; a timestamp is not a target")
    if target not in table["cells"]:
        _refuse("TARGET_NOT_IN_DATASET",
                f"--target {target!r} is not a column of this file; its columns are {table['columns']}")
    numeric, non_numeric = {}, {}
    for name, values in table["cells"].items():
        column, bad = _numeric(values)
        if column is None:
            non_numeric[name] = bad
        else:
            numeric[name] = column
    if target in non_numeric:
        _refuse("TARGET_NOT_NUMERIC",
                f"--target {target!r} carries {non_numeric[target]!r}, which is neither a number nor one of the "
                f"declared missing tokens {list(MISSING_TOKENS)}")

    sampling = sampling_test(table["times"])
    missing = missingness_test(table, numeric, non_numeric)
    start, length = longest_finite_run(numeric[target])
    if length < MIN_ROWS:
        _refuse("TOO_FEW_FINITE_ROWS",
                f"the longest run of {target!r} with no missing cell is {length} rows, and no test over fewer than "
                f"{MIN_ROWS} rows says anything")
    series = numeric[target][start:start + length]
    segment = {"start_row": int(start), "rows": int(length),
               "first": table["times"][start].isoformat(), "last": table["times"][start + length - 1].isoformat(),
               "rule": f"the longest run of {target!r} with no missing cell; the tests below read only these rows"}
    statsmodels = _statsmodels()
    stationarity = stationarity_test(series, statsmodels)
    seasonality = seasonality_test(series, sampling["step_seconds"], stationarity)
    tests = {"sampling": sampling, "missingness": missing, "analysis_segment": segment,
             "stationarity": stationarity, "seasonality": seasonality}

    timezone_declared = timezone_name is not None
    timezone_name = timezone_name or "UTC"
    holdout_declared = holdout is not None
    holdout = holdout if holdout_declared else {"fraction": DEFAULT_HOLDOUT_FRACTION}
    clock_declared = clock is not None
    candidates, skipped = build_candidates(
        tests, target=target, exogenous=sorted(name for name in numeric if name != target),
        provenance=provenance, clock=clock or DEFAULT_CLOCK, clock_declared=clock_declared, holdout=holdout,
        holdout_declared=holdout_declared, timezone_name=timezone_name, timezone_declared=timezone_declared,
        rows=length)
    return {
        "schema": SCHEMA,
        "dataset": {key: table[key] for key in ("path", "sha256", "columns", "time_column", "time_format",
                                                "rows_read", "truncated")},
        "target": target,
        "tests": tests,
        "candidates": candidates,
        "candidates_skipped": skipped,
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]), "numpy": np.__version__,
                        "statsmodels": _statsmodels_version(statsmodels)},
        "fitted": "NOTHING: this job reads a dataset profile and emits candidate specs; no model is trained here",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.design",
        description="Design candidate temporal representations from a dataset profile. No model is fitted.")
    parser.add_argument("--data", required=True, help="the CSV to profile")
    parser.add_argument("--target", required=True, help="the column to be forecast")
    parser.add_argument("--out", help="where to write the candidates document; stdout when absent")
    parser.add_argument("--time-column", help="the timestamp column, when its name is not a usual one")
    parser.add_argument("--timezone", help="the IANA zone the timestamps are read in; declared, never inferred")
    parser.add_argument("--clock", choices=list(representation.CLOCKS),
                        help="declare the availability clock instead of leaving it in not_decided")
    parser.add_argument("--holdout-fraction", type=float, help="declare the holdout fraction")
    parser.add_argument("--holdout-cut", help="declare the holdout cut as an ISO-8601 instant")
    parser.add_argument("--provenance", default="DEVELOPMENT", choices=list(representation.PROVENANCE))
    parser.add_argument("--max-rows", type=int, help="read at most this many rows")
    args = parser.parse_args(argv)
    if args.holdout_fraction is not None and args.holdout_cut is not None:
        print("REFUSED TWO_HOLDOUTS: declare a fraction or a cut, not both", file=sys.stderr)
        return 2
    holdout = ({"fraction": args.holdout_fraction} if args.holdout_fraction is not None else
               {"cut": args.holdout_cut} if args.holdout_cut is not None else None)
    try:
        document = design(args.data, args.target, time_column=args.time_column, timezone_name=args.timezone,
                          provenance=args.provenance, clock=args.clock, holdout=holdout, max_rows=args.max_rows)
    except (DesignRefusal, representation.SpecError) as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=False, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"{len(document['candidates'])} candidate(s) written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
