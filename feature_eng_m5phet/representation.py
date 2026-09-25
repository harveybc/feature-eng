"""The temporal representation, declared as one JSON object and validated like an envelope.

A forecasting bundle is not only a network. Before any weight exists somebody has decided what the model is allowed
to see: how often the series is sampled, which transform the target is modelled under, how far back a window reaches,
which individual lags are handed over, how many differences were taken, which calendar columns come along and -- the
decision that silently invalidates everything downstream when it is left implicit -- WHEN those calendar columns were
knowable, at publication or at receipt. Those decisions are the representation. Today each fitted state carries them
inside itself, in whatever shape its trainer happened to use, so two bundles cannot be compared by what they read.

This module gives those decisions one name and one shape, `m5phet.representation.v1`, so that a design job can emit
them (stage 2), a fit job can embed them in a manifest (stage 3), and an evaluation can rank them (stage 4) without
any stage restating the others' conventions. The object is validated the way the workbench validates an envelope:
every key is declared, an undeclared key is refused rather than ignored, and every refusal names the thing refused.

Four rules are worth stating because each of them was a way to produce a number nobody could defend:

* **`features` is a closed vocabulary.** A representation may only name features `feature-eng` actually knows how to
  build; the names and their origin are in `FEATURE_VOCABULARY`. A misspelled or foreign name is refused BY NAME here,
  where the spec is written, instead of becoming an all-zero column in a tensor three stages later.
* **`exogenous` is open, and says so.** Raw columns of the caller's own file cannot be checked against any vocabulary,
  because this object does not carry the file. They are validated as names only, and the fit job -- which does hold
  the data -- must check them against it. That asymmetry is declared, not hidden.
* **`holdout` must be declared.** Not defaulted. A representation with no stated holdout is a representation whose
  score cannot be trusted, and the cheapest place to refuse it is before training, not after.
* **A transform and a differencing order are not two ways of saying the same thing.** `target.transform` names the
  transform the target is modelled under; `differencing.order` names extra differences applied on top. Declaring both
  is refused as ambiguous rather than resolved on the author's behalf, since "diff with order 1" reads as one
  difference to one person and two to another.

`calendar.clock` has no default on purpose. `publication` is when the source put the number out; `receipt` is when it
reached this system. Choosing one from a dataset profile is impossible -- a CSV of timestamps cannot say which clock
stamped it -- so the spec insists that a person declare it, and the design job of stage 2 lists it as not decided.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA = "m5phet.representation.v1"

#: how the target is modelled. `level` is the column as it stands; `diff` its first difference; `log_return` the
#: difference of its logarithm, which only exists for a strictly positive column -- a fact this object cannot check
#: and the fit job must.
TRANSFORMS = ("level", "diff", "log_return")

#: the two availability clocks, spelled as `app/economic_calendar.py` spells the boundaries it reports side by side
CLOCKS = ("publication", "receipt")

#: what produced this representation, in the same words the fitted states under `~/.local/state/m5phet` use
PROVENANCE = ("DEVELOPMENT", "GOVERNED", "PRODUCTION")

#: The feature names `feature-eng` declares today, each with the source that declares it. This is a transcription of
#: what the code emits, not a wish list: the technical names are the keys `app/plugins/tech_indicator.py` writes into
#: `technical_indicators`, the seasonality names are the columns it adds when `seasonality_columns` is configured, and
#: the event names are the keys `app/economic_calendar.py` and the calendar processing of the same plugin report.
#: A name absent from this table is refused; adding a feature to `feature-eng` means adding it here too, which is the
#: point -- the vocabulary is a contract between the two, not a convention each side remembers separately.
FEATURE_VOCABULARY = {
    "RSI": "app/plugins/tech_indicator.py:technical_indicators['RSI']",
    "MACD": "app/plugins/tech_indicator.py:technical_indicators['MACD']",
    "MACD_Histogram": "app/plugins/tech_indicator.py:technical_indicators['MACD_Histogram']",
    "MACD_Signal": "app/plugins/tech_indicator.py:technical_indicators['MACD_Signal']",
    "EMA": "app/plugins/tech_indicator.py:technical_indicators['EMA']",
    "Stochastic_%K": "app/plugins/tech_indicator.py:technical_indicators['Stochastic_%K']",
    "Stochastic_%D": "app/plugins/tech_indicator.py:technical_indicators['Stochastic_%D']",
    "ADX": "app/plugins/tech_indicator.py:technical_indicators['ADX']",
    "DI+": "app/plugins/tech_indicator.py:technical_indicators['DI+']",
    "DI-": "app/plugins/tech_indicator.py:technical_indicators['DI-']",
    "ATR": "app/plugins/tech_indicator.py:technical_indicators['ATR']",
    "CCI": "app/plugins/tech_indicator.py:technical_indicators['CCI']",
    "BB_Upper": "app/plugins/tech_indicator.py:technical_indicators['BB_Upper']",
    "BB_Middle": "app/plugins/tech_indicator.py:technical_indicators['BB_Middle']",
    "BB_Lower": "app/plugins/tech_indicator.py:technical_indicators['BB_Lower']",
    "WilliamsR": "app/plugins/tech_indicator.py:technical_indicators['WilliamsR']",
    "Momentum": "app/plugins/tech_indicator.py:technical_indicators['Momentum']",
    "ROC": "app/plugins/tech_indicator.py:technical_indicators['ROC']",
    "day_of_month": "app/plugins/tech_indicator.py:additional_features_df['day_of_month']",
    "hour_of_day": "app/plugins/tech_indicator.py:additional_features_df['hour_of_day']",
    "day_of_week": "app/plugins/tech_indicator.py:additional_features_df['day_of_week']",
    "release_surprise": "app/economic_calendar.py:EventLedger.surprise()['release_surprise']",
    "available_surprise": "app/economic_calendar.py:EventLedger.surprise()['available_surprise']",
    "revision_surprise": "app/economic_calendar.py:EventLedger.surprise()['revision_surprise']",
    "standardized": "app/economic_calendar.py:EventLedger.surprise()['standardized']",
    "forecast_diff": "app/plugins/tech_indicator.py:econ_data['forecast_diff']",
    "volatility_weighted_diff": "app/plugins/tech_indicator.py:econ_data['volatility_weighted_diff']",
    "actual_minus_forecast": "app/plugins/tech_indicator.py:econ_data['actual_minus_forecast']",
    "actual_minus_previous": "app/plugins/tech_indicator.py:econ_data['actual_minus_previous']",
    "country_encoded": "app/plugins/tech_indicator.py:econ_data['country_encoded']",
    "description_encoded": "app/plugins/tech_indicator.py:econ_data['description_encoded']",
    "volatility": "app/plugins/tech_indicator.py:econ_data['volatility']",
    "trend_signal": "app/plugins/tech_indicator.py:df_signals['trend_signal']",
    "volatility_signal": "app/plugins/tech_indicator.py:df_signals['volatility_signal']",
}

#: the subset of the vocabulary that `calendar.columns` may name: what a clock applies to. A technical indicator is
#: computed from the series itself and has no publication instant, so naming one here is refused rather than ignored.
CALENDAR_VOCABULARY = frozenset({
    "day_of_month", "hour_of_day", "day_of_week",
    "release_surprise", "available_surprise", "revision_surprise", "standardized",
    "forecast_diff", "volatility_weighted_diff", "actual_minus_forecast", "actual_minus_previous",
    "country_encoded", "description_encoded", "volatility", "trend_signal", "volatility_signal",
})

#: every key the object may carry, and whether it must
REQUIRED_KEYS = ("schema", "sampling", "target", "windows", "lags", "differencing",
                 "calendar", "features", "holdout", "provenance")

#: optional keys. `why` and `not_decided` are ANNOTATIONS: stage 2 writes them to say which test result motivated a
#: candidate and what the data could not decide. They carry no modelling meaning and `spec_id` excludes them, so an
#: annotated candidate and the bare spec it proposes are the same representation.
OPTIONAL_KEYS = ("candidate_id", "exogenous", "fitted_state_ref", "why", "not_decided")

ANNOTATION_KEYS = ("candidate_id", "why", "not_decided")

MAX_ORDER = 2


class SpecError(ValueError):
    """A representation that cannot be read, carrying the code a caller matches on and the name of what was refused."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise SpecError(code, why)


def _mapping(value, where, code="BAD_TYPE"):
    if not isinstance(value, dict):
        _refuse(code, f"{where} must be a JSON object, got {type(value).__name__}")
    return value


def _keys(value, where, required, optional=()):
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        _refuse("UNKNOWN_KEY", f"{where} carries undeclared key(s) {unknown}; declared keys are "
                               f"{sorted(set(required) | set(optional))}")
    missing = sorted(set(required) - set(value))
    if missing:
        _refuse("MISSING_KEY", f"{where} is missing required key(s) {missing}")


def _int(value, where, *, minimum=None, maximum=None):
    if type(value) is not int:       # a bool is an int in Python and is not a length, a lag or an order
        _refuse("BAD_TYPE", f"{where} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        _refuse("BAD_VALUE", f"{where} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        _refuse("BAD_VALUE", f"{where} must be <= {maximum}, got {value}")
    return value


def _name(value, where):
    if not isinstance(value, str) or not value.strip():
        _refuse("BAD_TYPE", f"{where} must be a nonempty string, got {value!r}")
    return value


def _increasing(values, where, *, code, minimum=1, allow_empty=True):
    if not isinstance(values, list):
        _refuse("BAD_TYPE", f"{where} must be a list of integers, got {type(values).__name__}")
    if not values and not allow_empty:
        _refuse(code, f"{where} must name at least one value; an empty list is a representation that reads nothing")
    for value in values:
        _int(value, f"{where} entry", minimum=minimum)
    if values != sorted(set(values)):
        _refuse("BAD_VALUE", f"{where} must be strictly increasing and free of duplicates, got {values}")
    return values


def _validate_sampling(sampling):
    _mapping(sampling, "sampling")
    _keys(sampling, "sampling", ("step_seconds", "timezone"))
    _int(sampling["step_seconds"], "sampling.step_seconds", minimum=1)
    timezone = _name(sampling["timezone"], "sampling.timezone")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        _refuse("UNKNOWN_TIMEZONE", f"sampling.timezone {timezone!r} is not a zone this machine knows ({exc})")


def _validate_target(target):
    _mapping(target, "target")
    _keys(target, "target", ("column", "transform"))
    _name(target["column"], "target.column")
    if target["transform"] not in TRANSFORMS:
        _refuse("UNKNOWN_TRANSFORM", f"target.transform {target['transform']!r} is not one of {list(TRANSFORMS)}")


def _validate_differencing(differencing, target):
    _mapping(differencing, "differencing")
    _keys(differencing, "differencing", ("order",))
    order = _int(differencing["order"], "differencing.order", minimum=0, maximum=MAX_ORDER)
    if order and target.get("transform") != "level":
        _refuse("AMBIGUOUS_DIFFERENCING",
                f"target.transform {target['transform']!r} already differences the target and differencing.order is "
                f"{order}; declare the transform with order 0, or the level with an order, but not both")


def _validate_calendar(calendar):
    _mapping(calendar, "calendar")
    _keys(calendar, "calendar", ("clock", "columns"))
    if calendar["clock"] not in CLOCKS:
        _refuse("UNKNOWN_CLOCK", f"calendar.clock {calendar['clock']!r} is not one of {list(CLOCKS)}; it has no "
                                 f"default because no dataset can say which clock stamped it")
    columns = calendar["columns"]
    if not isinstance(columns, list):
        _refuse("BAD_TYPE", f"calendar.columns must be a list of names, got {type(columns).__name__}")
    for column in columns:
        _name(column, "calendar.columns entry")
        if column not in CALENDAR_VOCABULARY:
            _refuse("FOREIGN_CALENDAR_COLUMN",
                    f"calendar.columns names {column!r}, which is not a calendar feature feature-eng declares; "
                    f"declared calendar features are {sorted(CALENDAR_VOCABULARY)}")
    if len(set(columns)) != len(columns):
        _refuse("BAD_VALUE", f"calendar.columns repeats a name: {columns}")


def _validate_features(features, vocabulary):
    if not isinstance(features, list):
        _refuse("BAD_TYPE", f"features must be a list of names, got {type(features).__name__}")
    for feature in features:
        _name(feature, "features entry")
        if feature not in vocabulary:
            _refuse("FOREIGN_FEATURE",
                    f"features names {feature!r}, which feature-eng does not declare; it builds "
                    f"{len(vocabulary)} features and this is not one of them")
    if len(set(features)) != len(features):
        _refuse("BAD_VALUE", f"features repeats a name: {features}")


def _validate_exogenous(exogenous, target_column):
    if not isinstance(exogenous, list):
        _refuse("BAD_TYPE", f"exogenous must be a list of column names, got {type(exogenous).__name__}")
    for column in exogenous:
        _name(column, "exogenous entry")
        if column == target_column:
            _refuse("BAD_VALUE", f"exogenous names the target column {column!r}; the target's own past is declared "
                                 f"by lags, not as an exogenous column")
    if len(set(exogenous)) != len(exogenous):
        _refuse("BAD_VALUE", f"exogenous repeats a name: {exogenous}")


def _validate_holdout(holdout):
    _mapping(holdout, "holdout", code="HOLDOUT_NOT_DECLARED")
    if set(holdout) == set():
        _refuse("HOLDOUT_NOT_DECLARED", "holdout declares neither a fraction nor a cut; a representation whose "
                                        "holdout is left to the fit job has no score anyone can read")
    _keys(holdout, "holdout", (), ("fraction", "cut"))
    if len(holdout) != 1:
        _refuse("HOLDOUT_NOT_DECLARED",
                f"holdout must declare exactly one of 'fraction' or 'cut', got {sorted(holdout)}; two ways of cutting "
                f"the same series is two holdouts")
    if "fraction" in holdout:
        fraction = holdout["fraction"]
        if type(fraction) not in (int, float) or type(fraction) is bool:
            _refuse("BAD_TYPE", f"holdout.fraction must be a number, got {fraction!r}")
        if not 0 < float(fraction) < 1:
            _refuse("BAD_VALUE", f"holdout.fraction must lie strictly between 0 and 1, got {fraction}")
    else:
        cut = _name(holdout["cut"], "holdout.cut")
        try:
            datetime.fromisoformat(cut)
        except ValueError:
            _refuse("BAD_VALUE", f"holdout.cut {cut!r} is not an ISO-8601 instant; a cut nobody can parse cuts nothing")


def validate_spec(obj, *, vocabulary=None):
    """Read one representation, or refuse it by name. Returns the object itself when it is whole.

    `vocabulary` exists so that a caller holding a DIFFERENT declared feature list -- another installation of
    feature-eng, a later release with more indicators -- can validate against that list instead. It does not exist so
    that a caller can widen the vocabulary to admit a name: passing `{}` refuses every feature, which is the honest
    behaviour of an empty declaration.
    """
    vocabulary = FEATURE_VOCABULARY if vocabulary is None else vocabulary
    _mapping(obj, "the representation")
    _keys(obj, "the representation", REQUIRED_KEYS, OPTIONAL_KEYS)
    if obj["schema"] != SCHEMA:
        _refuse("WRONG_SCHEMA", f"schema is {obj['schema']!r} and this reader only reads {SCHEMA!r}")
    _validate_sampling(obj["sampling"])
    _validate_target(obj["target"])
    _increasing(obj["windows"], "windows", code="EMPTY_WINDOWS", minimum=1, allow_empty=False)
    _increasing(obj["lags"], "lags", code="EMPTY_LAGS", minimum=1, allow_empty=True)
    _validate_differencing(obj["differencing"], obj["target"])
    _validate_calendar(obj["calendar"])
    _validate_features(obj["features"], vocabulary)
    _validate_exogenous(obj.get("exogenous", []), obj["target"]["column"])
    _validate_holdout(obj["holdout"])
    if obj["provenance"] not in PROVENANCE:
        _refuse("UNKNOWN_PROVENANCE", f"provenance {obj['provenance']!r} is not one of {list(PROVENANCE)}")
    if "fitted_state_ref" in obj and obj["fitted_state_ref"] is not None:
        _name(obj["fitted_state_ref"], "fitted_state_ref")
    for key in ("why", "not_decided"):
        if key in obj:
            _mapping(obj[key], key)
    if "candidate_id" in obj:
        _name(obj["candidate_id"], "candidate_id")
    return obj


def modelling_keys(spec):
    """The spec without its annotations: what actually decides the tensors a fit job builds."""
    return {key: value for key, value in spec.items() if key not in ANNOTATION_KEYS}


def canonical_json(spec):
    """One byte string per representation, so two stages that hold the same spec hold the same identity."""
    return json.dumps(modelling_keys(spec), sort_keys=True, separators=(",", ":"), allow_nan=False)


def spec_id(spec):
    """The identity a fit job embeds in its manifest and an evaluation joins on."""
    return hashlib.sha256(canonical_json(spec).encode()).hexdigest()


def dumps(spec, *, indent=2):
    """Validate, then write. Nothing leaves this module without having been read by `validate_spec`."""
    return json.dumps(validate_spec(spec), indent=indent, sort_keys=True, allow_nan=False)


def loads(text):
    """Read, then validate. A document that is not an object is refused as one, not as a JSON error."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError("MALFORMED_JSON", f"the representation is not JSON: {exc}") from exc
    return validate_spec(obj)
