"""The feature metric sheet: every column measured, every measurement named, no model fitted.

WP18 step 1, and a straight extension of WP06 stage 2. `design.py` profiles ONE column -- the target -- because a
candidate representation is a statement about the target's own memory. A pipeline that preprocesses each feature,
groups features by what they share and hands each group to its own extractor needs the same care spent on EVERY
column, plus the numbers that only exist BETWEEN columns. That is this module: a per-feature sheet and a per-pair
sheet, written so that the per-feature block can be handed to a chooser as a state text and nothing else.

The tests that already exist are called, never re-implemented. Stationarity is `design.stationarity_test` (ADF and
KPSS from statsmodels, or the named autocorrelation heuristic when statsmodels is absent); the autocorrelation peaks
and the decay lag are `design.seasonality_test`; the sampling grid is `design.sampling_test`; the file reader, the
missing tokens and the refusal type are `design`'s too. If a number here disagreed with a number there, one of the
two would be a second opinion nobody asked for.

Four commitments, each the opposite of a convenient shortcut:

* **A measurement that could not be taken says so by name.** Mutual information needs scikit-learn; without it the
  pair block reads `NOT_AVAILABLE` with the reason, and no correlation is quietly promoted to stand in for it. The
  same holds for a constant column (`ZERO_VARIANCE`) and for a pair with too few rows in common.
* **Every number carries its declared decimals.** `decimals` at the top of the document says how many, per family,
  and every number in the document was rounded with that rule. A number whose precision is not declared is a number
  whose stability across two runs is an accident.
* **The blocks are sorted and the document is byte-stable.** Two runs over the same file with the same arguments
  produce the same bytes. Keys inside `features[<name>]` and `pairs[<key>]` are sorted, so a state text rendered
  from a block is the same text on every machine -- which is what makes a decision digest mean anything.
* **A directed metric is not filed as a symmetric one.** `pairs` holds what is true of an unordered pair (Pearson,
  Spearman, mutual information). The lagged cross-correlation to the target -- feature at `t - lag` against the
  target at `t` -- is directed, so it lives on the feature, under `cross_correlation_to_target`, where the chooser
  that reads one feature's sheet will find it.

"Feature" in this module means a numeric column of the caller's file, not a name from
`representation.FEATURE_VOCABULARY`; those are the features `feature-eng` can BUILD, these are the ones the file
already carries.

Everything is deterministic and CPU only. Nothing is fitted, nothing is trained, no seed is drawn except the one
scikit-learn's estimator is explicitly given so its tie-breaking noise is the same on every run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import design, representation

SCHEMA = "m5phet.feature_metrics.v1"

#: the separator that builds a pair key; a column carrying it would make the key ambiguous, so such a file is refused
PAIR_SEPARATOR = "::"

#: the lags the cross-correlation to the target is measured at when no spec and no `--lags` declare them
DEFAULT_LAGS = (1, 24)

#: how many rows the mutual-information estimator reads: the HEAD of the common rows, never the end a holdout is cut
#: from, and capped so that a 2-million-row file does not turn a metric sheet into a compute job
MI_MAX_ROWS = 20000

#: the neighbour count handed to scikit-learn's estimator, declared here because the estimate depends on it
MI_NEIGHBOURS = 3

#: the seed handed to scikit-learn's estimator, which adds tie-breaking noise; declared so the number is repeatable
MI_RANDOM_STATE = 0

#: fewer rows in common than this and a pair correlation is an anecdote
MIN_PAIR_ROWS = 32

#: how many decimals each family of numbers carries. Every number in the document went through one of these.
DECIMALS = {
    "correlation": 6,
    "distribution": 6,
    "fraction": 6,
    "mutual_information": 6,
    "statistic": 6,
}


class MetricsRefusal(design.DesignRefusal):
    """A file or an argument this job will not measure from, carrying the code and naming what is wrong."""


def _refuse(code, why):
    raise MetricsRefusal(code, why)


def _round(value, family):
    """One number with its declared decimals, or None when the arithmetic did not produce a finite number."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, DECIMALS[family])


def _sorted_block(value):
    """The same mapping with its keys sorted, recursively: the shape a state text is rendered from."""
    if isinstance(value, dict):
        return {key: _sorted_block(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_block(item) for item in value]
    return value


def _sklearn_version():
    try:
        import sklearn
    except ImportError:
        return "NOT_AVAILABLE"
    return sklearn.__version__


def _mutual_info():
    """scikit-learn's mutual-information estimator, or None. Behind a function so a test can take it away."""
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        return None
    return mutual_info_regression


# ------------------------------------------------------------------------------------- the per-feature measurements

def distribution_summary(values):
    """Mean, dispersion, shape and tails of one column, over its finite cells only.

    Skewness and excess kurtosis are the moment estimators `g1 = m3 / m2**1.5` and `g2 = m4 / m2**2 - 3`, not their
    sample-size corrections: the correction would change the number by a factor of order `1/n` and the block would
    then have to say which convention it used anyway. It says it either way.
    """
    finite = values[np.isfinite(values)]
    block = {"rows_used": int(finite.size),
             "rows_rule": "the finite cells of the column; missing cells are excluded, never imputed"}
    if finite.size == 0:
        block.update(status="NO_FINITE_ROWS", reason="every cell of this column is missing or unreadable")
        return block
    mean = float(finite.mean())
    centred = finite - mean
    m2 = float(np.mean(centred ** 2))
    std = float(finite.std(ddof=1)) if finite.size > 1 else None
    if m2 > 0:
        skew = float(np.mean(centred ** 3)) / (m2 ** 1.5)
        kurtosis = float(np.mean(centred ** 4)) / (m2 ** 2) - 3.0
        shape_status = "OK"
    else:
        skew = kurtosis = None
        shape_status = "ZERO_VARIANCE"
    block.update(
        status="OK",
        mean=_round(mean, "distribution"),
        std=_round(std, "distribution"),
        std_rule="the sample standard deviation, ddof=1; None when a single row is finite",
        variance=_round(m2, "distribution"),
        variance_rule="the population second central moment, ddof=0, which the shape numbers below divide by",
        skew=_round(skew, "distribution"),
        skew_rule="g1 = m3 / m2**1.5, the moment estimator, uncorrected for sample size",
        excess_kurtosis=_round(kurtosis, "distribution"),
        excess_kurtosis_rule="g2 = m4 / m2**2 - 3; zero for a normal column",
        shape_status=shape_status,
        minimum=_round(finite.min(), "distribution"),
        maximum=_round(finite.max(), "distribution"),
        quantile_01=_round(np.quantile(finite, 0.01, method="linear"), "distribution"),
        quantile_99=_round(np.quantile(finite, 0.99, method="linear"), "distribution"),
        quantile_rule="numpy linear interpolation between order statistics",
    )
    return block


def scale_block(values):
    """How large this column's numbers are, as a decimal exponent: what tells one preprocessor from another."""
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0 or float(finite.max()) == 0.0:
        return {"magnitude": None, "max_absolute": _round(0.0 if finite.size else None, "distribution"),
                "median_absolute": None, "status": "NO_SCALE",
                "reason": "the column has no finite nonzero cell, so it has no order of magnitude",
                "rule": "floor(log10(max |x|)) over the finite cells"}
    return {"magnitude": int(math.floor(math.log10(float(finite.max())))),
            "max_absolute": _round(finite.max(), "distribution"),
            "median_absolute": _round(np.median(finite), "distribution"),
            "status": "OK",
            "rule": "floor(log10(max |x|)) over the finite cells"}


def dtype_block(values, texts):
    """What the column is once read, and what it was in the file: two different facts, both worth a chooser's time."""
    finite = values[np.isfinite(values)]
    integral = bool(finite.size and np.all(finite == np.rint(finite)))
    return {"storage": "float64",
            "storage_rule": "every numeric column is read as float64 with the declared missing tokens as NaN",
            "source": "numeric text",
            "values_are_integral": integral,
            "distinct_finite_values": int(np.unique(finite).size),
            "first_cell": texts[0] if texts else None}


def cross_correlation_to_target(values, target_values, lags):
    """Pearson correlation between this column at `t - lag` and the target at `t`, for each declared lag.

    The direction is the one a forecaster cares about: does knowing this column EARLIER help with the target NOW.
    The symmetric, contemporaneous number is in `pairs`; this one is not symmetric and is not filed there.
    """
    block = {"lags": list(lags),
             "rule": "Pearson correlation of feature[t - lag] against target[t], over rows where both are finite",
             "by_lag": []}
    for lag in lags:
        if lag >= values.size:
            block["by_lag"].append({"lag": int(lag), "status": "TOO_FEW_ROWS", "correlation": None, "rows_used": 0,
                                    "reason": f"lag {lag} is not shorter than the {values.size} rows read"})
            continue
        left, right = values[:values.size - lag], target_values[lag:]
        correlation, rows, status = _pearson(left, right)
        block["by_lag"].append({"lag": int(lag), "correlation": correlation, "rows_used": rows, "status": status})
    return block


# ------------------------------------------------------------------------------------------ the pair measurements

def _pearson(left, right):
    """The correlation over the rows both columns have, with the reason by name when there is no number."""
    both = np.isfinite(left) & np.isfinite(right)
    rows = int(np.count_nonzero(both))
    if rows < MIN_PAIR_ROWS:
        return None, rows, "TOO_FEW_ROWS"
    a, b = left[both], right[both]
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return None, rows, "ZERO_VARIANCE"
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denominator == 0.0:
        return None, rows, "ZERO_VARIANCE"
    return _round(float(np.dot(a, b)) / denominator, "correlation"), rows, "OK"


def _ranks(values):
    """Average ranks, ties shared -- the ranking Spearman is defined on, in numpy so scipy is not needed for it."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for i in range(1, sorted_values.size + 1):
        if i == sorted_values.size or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def _spearman(left, right):
    """The same correlation over the average ranks; a constant column has no ranks to correlate and says so."""
    both = np.isfinite(left) & np.isfinite(right)
    rows = int(np.count_nonzero(both))
    if rows < MIN_PAIR_ROWS:
        return None, rows, "TOO_FEW_ROWS"
    value, _, status = _pearson(_ranks(left[both]), _ranks(right[both]))
    return value, rows, status


def _mutual_information(left, right, estimator):
    """scikit-learn's k-nearest-neighbour estimate, or the word that says the environment cannot compute it."""
    if estimator is None:
        return {"value": None, "status": "NOT_AVAILABLE",
                "reason": "scikit-learn is not installed in the environment this job ran in, and no correlation "
                          "stands in for a mutual information"}
    both = np.isfinite(left) & np.isfinite(right)
    rows = int(np.count_nonzero(both))
    if rows < MIN_PAIR_ROWS:
        return {"value": None, "status": "TOO_FEW_ROWS", "rows_used": rows,
                "reason": f"{rows} rows in common and no estimate over fewer than {MIN_PAIR_ROWS} rows says anything"}
    a, b = left[both][:MI_MAX_ROWS], right[both][:MI_MAX_ROWS]
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return {"value": None, "status": "ZERO_VARIANCE", "rows_used": int(a.size),
                "reason": "one of the two columns is constant over the rows they share"}
    value = float(estimator(a.reshape(-1, 1), b, discrete_features=False, n_neighbors=MI_NEIGHBOURS,
                            random_state=MI_RANDOM_STATE)[0])
    return {"value": _round(value, "mutual_information"), "status": "OK", "rows_used": int(a.size),
            "estimator": "sklearn.feature_selection.mutual_info_regression",
            "n_neighbors": MI_NEIGHBOURS, "random_state": MI_RANDOM_STATE,
            "rows_rule": f"the first {MI_MAX_ROWS} rows the pair has in common; the head, never the end a holdout "
                         f"will be cut from",
            "units": "nats", "lower_bound": 0.0,
            "caveat": "an estimate, not a bound: it is not normalised and it is not comparable across row counts"}


def pair_key(left, right):
    a, b = sorted((left, right))
    return f"{a}{PAIR_SEPARATOR}{b}"


# -------------------------------------------------------------------------------------------------- the whole sheet

def _lags_from_spec(path):
    """The lags a design document or a representation spec already declared, so this sheet measures those."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        _refuse("SPEC_UNREADABLE", f"{path} does not contain a JSON object")
    if document.get("schema") == representation.SCHEMA:
        return sorted(set(int(lag) for lag in document.get("lags", []))), f"the representation spec {path}"
    if document.get("schema") == design.SCHEMA:
        lags = sorted({int(lag) for candidate in document.get("candidates", []) for lag in candidate.get("lags", [])})
        if not lags:
            _refuse("SPEC_DECLARES_NO_LAGS", f"the design document {path} carries no candidate lag")
        return lags, f"the union of the candidate lags of the design document {path}"
    _refuse("SPEC_SCHEMA_UNKNOWN",
            f"{path} carries schema {document.get('schema')!r}; this job reads {representation.SCHEMA!r} or "
            f"{design.SCHEMA!r}")


def feature_metrics(data, target, *, time_column=None, lags=None, spec=None, max_rows=None):
    """Read one dataset, measure every numeric column and every pair of them, emit the sheet. Nothing is fitted."""
    if lags is not None and spec is not None:
        _refuse("TWO_LAG_SOURCES", "declare --lags or --spec, not both; two sources would hide which one was read")
    if spec is not None:
        lags, lags_source = _lags_from_spec(spec)
    elif lags is not None:
        lags, lags_source = sorted(set(int(lag) for lag in lags)), "the --lags argument"
    else:
        lags, lags_source = list(DEFAULT_LAGS), "the declared default of this job"
    if any(lag < 1 for lag in lags):
        _refuse("LAG_NOT_POSITIVE", f"the lags {lags} contain a lag below 1; a cross-correlation at lag 0 is the "
                                    f"contemporaneous correlation and is already in `pairs`")

    table = design.read_table(data, time_column=time_column, max_rows=max_rows)
    for name in table["columns"]:
        if PAIR_SEPARATOR in name:
            _refuse("COLUMN_NAME_HAS_SEPARATOR",
                    f"the column {name!r} contains {PAIR_SEPARATOR!r}, which this job uses to build a pair key; "
                    f"rename the column or the pair keys would be ambiguous")
    if target == table["time_column"]:
        _refuse("TARGET_IS_THE_TIME_COLUMN", f"{target!r} is the time column; a timestamp is not a target")
    if target not in table["cells"]:
        _refuse("TARGET_NOT_IN_DATASET",
                f"--target {target!r} is not a column of this file; its columns are {table['columns']}")

    numeric, non_numeric = {}, {}
    for name, values in table["cells"].items():
        column, bad = design._numeric(values)
        if column is None:
            non_numeric[name] = bad
        else:
            numeric[name] = column
    if target in non_numeric:
        _refuse("TARGET_NOT_NUMERIC",
                f"--target {target!r} carries {non_numeric[target]!r}, which is neither a number nor one of the "
                f"declared missing tokens {list(design.MISSING_TOKENS)}")

    sampling = design.sampling_test(table["times"])
    statsmodels = design._statsmodels()
    rows = table["rows_read"]
    target_values = numeric[target]

    features = {}
    for name in sorted(numeric):
        values = numeric[name]
        missing = int(np.count_nonzero(~np.isfinite(values)))
        block = {
            "name": name,
            "is_target": name == target,
            "rows": rows,
            "missing": missing,
            "missing_fraction": _round(missing / rows, "fraction"),
            "missing_rule": f"cells equal to one of {list(design.MISSING_TOKENS)} or unreadable as a float, "
                            f"divided by the {rows} rows read",
            "dtype": dtype_block(values, table["cells"][name]),
            "distribution": distribution_summary(values),
            "scale": scale_block(values),
            "cross_correlation_to_target": cross_correlation_to_target(values, target_values, lags),
        }
        start, length = design.longest_finite_run(values)
        segment = {"start_row": int(start), "rows": int(length),
                   "rule": f"the longest run of {name!r} with no missing cell; the two tests below read only "
                           f"these rows"}
        if length < design.MIN_ROWS:
            reason = (f"the longest run of {name!r} with no missing cell is {length} rows, and no test over fewer "
                      f"than {design.MIN_ROWS} rows says anything")
            block["analysis_segment"] = dict(segment, status="TOO_FEW_FINITE_ROWS", reason=reason)
            block["stationarity"] = {"status": "NOT_AVAILABLE", "reason": reason, "verdict": "NOT_AVAILABLE"}
            block["acf"] = {"status": "NOT_AVAILABLE", "reason": reason, "peaks": [], "decay_lag": None}
        else:
            series = values[start:start + length]
            segment.update(status="OK",
                           first=table["times"][start].isoformat(),
                           last=table["times"][start + length - 1].isoformat())
            block["analysis_segment"] = segment
            stationarity = design.stationarity_test(series, statsmodels)
            block["stationarity"] = stationarity
            block["acf"] = design.seasonality_test(series, sampling["step_seconds"], stationarity)
        features[name] = _sorted_block(block)

    estimator = _mutual_info()
    names = sorted(numeric)
    pairs = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pearson, pearson_rows, pearson_status = _pearson(numeric[left], numeric[right])
            spearman, spearman_rows, spearman_status = _spearman(numeric[left], numeric[right])
            pairs[pair_key(left, right)] = _sorted_block({
                "features": [left, right],
                "rows_in_common": pearson_rows,
                "pearson": {"value": pearson, "status": pearson_status, "rows_used": pearson_rows,
                            "rule": "the linear correlation over the rows both columns have finite"},
                "spearman": {"value": spearman, "status": spearman_status, "rows_used": spearman_rows,
                             "rule": "the linear correlation of the average ranks, ties shared"},
                "mutual_information": _mutual_information(numeric[left], numeric[right], estimator),
            })

    return {
        "schema": SCHEMA,
        "dataset": {key: table[key] for key in ("path", "sha256", "columns", "time_column", "time_format",
                                                "rows_read", "truncated")},
        "target": target,
        "lags": list(lags),
        "lags_source": lags_source,
        "decimals": dict(DECIMALS),
        "sampling": sampling,
        "excluded_columns": {name: f"first unreadable value {value!r}; a metric sheet measures numbers"
                             for name, value in sorted(non_numeric.items())},
        "features": features,
        "pairs": pairs,
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]), "numpy": np.__version__,
                        "statsmodels": design._statsmodels_version(statsmodels), "sklearn": _sklearn_version()},
        "fitted": "NOTHING: this job measures a dataset's columns and the numbers between them; no model is trained",
    }


def decision_payload(document, feature):
    """One feature's sheet as the structured payload `m5phet.decide.decision_state` renders a state text from.

    The payload carries measurements and no rows, with the decimals the document declared, and its keys are sorted --
    the three properties that make the digest of the rendered text the same on two machines. Building the text is
    `decide`'s job, not this one's; this is the payload it is handed.
    """
    if feature not in document["features"]:
        _refuse("FEATURE_NOT_IN_SHEET",
                f"{feature!r} is not a measured feature of this sheet; it carries {sorted(document['features'])}")
    block = document["features"][feature]
    peaks = [peak["lag"] for peak in block["acf"].get("peaks", [])]
    payload = {
        "kind": "feature_profile",
        "feature": feature,
        "is_target": block["is_target"],
        "target": document["target"],
        "rows": block["rows"],
        "sampling_step_seconds": document["sampling"]["step_seconds"],
        "sampling_regular_fraction": document["sampling"]["regular_fraction"],
        "stationarity": block["stationarity"].get("verdict", "NOT_AVAILABLE"),
        "stationarity_source": block["stationarity"].get("verdict_source", block["stationarity"].get("status")),
        "acf_peaks": peaks,
        "acf_decay_lag": block["acf"].get("decay_lag"),
        "acf_computed_on": block["acf"].get("computed_on", "NOT_AVAILABLE"),
        "missing_fraction": block["missing_fraction"],
        "mean": block["distribution"].get("mean"),
        "std": block["distribution"].get("std"),
        "skew": block["distribution"].get("skew"),
        "excess_kurtosis": block["distribution"].get("excess_kurtosis"),
        "minimum": block["distribution"].get("minimum"),
        "maximum": block["distribution"].get("maximum"),
        "quantile_01": block["distribution"].get("quantile_01"),
        "quantile_99": block["distribution"].get("quantile_99"),
        "scale_magnitude": block["scale"]["magnitude"],
        "values_are_integral": block["dtype"]["values_are_integral"],
        "cross_correlation_to_target": {str(entry["lag"]): entry["correlation"]
                                        for entry in block["cross_correlation_to_target"]["by_lag"]},
        "decimals": dict(document["decimals"]),
    }
    return _sorted_block(payload)


def validate(document):
    """Read the sheet the way an envelope is read: a key that is not declared is refused, not ignored.

    The point is not type-checking for its own sake. Step 3 consumes this document and step 2 renders a state text
    from it; both have to be able to say "this file is not a metric sheet" instead of failing four functions later
    on a missing key.
    """
    if not isinstance(document, dict):
        _refuse("BAD_TYPE", "a metric sheet is a JSON object")
    required = ("schema", "dataset", "target", "lags", "lags_source", "decimals", "sampling", "excluded_columns",
                "features", "pairs", "environment", "fitted")
    missing = [key for key in required if key not in document]
    if missing:
        _refuse("MISSING_KEY", f"the metric sheet has no {missing}")
    unknown = [key for key in document if key not in required]
    if unknown:
        _refuse("UNKNOWN_KEY", f"the metric sheet carries undeclared keys {unknown}")
    if document["schema"] != SCHEMA:
        _refuse("WRONG_SCHEMA", f"this job reads {SCHEMA!r} and the document says {document['schema']!r}")
    if not isinstance(document["features"], dict) or not document["features"]:
        _refuse("NO_FEATURES", "the metric sheet measures no feature")
    if document["target"] not in document["features"]:
        _refuse("TARGET_NOT_MEASURED", f"the target {document['target']!r} is not among the measured features")
    for name, block in document["features"].items():
        if list(block) != sorted(block):
            _refuse("FEATURE_KEYS_NOT_SORTED", f"the block of {name!r} is not sorted, so its state text is not stable")
        for key in ("acf", "cross_correlation_to_target", "distribution", "dtype", "missing_fraction", "name",
                    "scale", "stationarity"):
            if key not in block:
                _refuse("FEATURE_MISSING_KEY", f"the block of {name!r} has no {key!r}")
        if block["name"] != name:
            _refuse("FEATURE_NAME_MISMATCH", f"the block filed under {name!r} names itself {block['name']!r}")
    for key, block in document["pairs"].items():
        if list(block) != sorted(block):
            _refuse("PAIR_KEYS_NOT_SORTED", f"the block of {key!r} is not sorted")
        members = block.get("features")
        if not isinstance(members, list) or len(members) != 2 or pair_key(*members) != key:
            _refuse("PAIR_KEY_MISMATCH", f"the pair {key!r} does not name the two features it is filed under")
        for member in members:
            if member not in document["features"]:
                _refuse("PAIR_MEMBER_NOT_MEASURED", f"the pair {key!r} names {member!r}, which has no feature block")
    return document


def dumps(document):
    """The one rendering of this document: the bytes two runs must agree on."""
    return json.dumps(document, indent=2, sort_keys=False, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.metrics",
        description="Measure every numeric column of a dataset and every pair of them. No model is fitted.")
    parser.add_argument("--data", required=True, help="the CSV to measure")
    parser.add_argument("--target", required=True, help="the column the cross-correlations are measured against")
    parser.add_argument("--out", help="where to write the metric sheet; stdout when absent")
    parser.add_argument("--time-column", help="the timestamp column, when its name is not a usual one")
    parser.add_argument("--lags", type=int, nargs="+", help=f"the lags of the cross-correlation to the target; "
                                                            f"the declared default is {list(DEFAULT_LAGS)}")
    parser.add_argument("--spec", help="take the lags from a representation spec or a design document instead")
    parser.add_argument("--max-rows", type=int, help="read at most this many rows")
    args = parser.parse_args(argv)
    try:
        document = feature_metrics(args.data, args.target, time_column=args.time_column, lags=args.lags,
                                   spec=args.spec, max_rows=args.max_rows)
        validate(document)
    except (design.DesignRefusal, representation.SpecError) as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    text = dumps(document)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{len(document['features'])} feature(s) and {len(document['pairs'])} pair(s) written to {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
