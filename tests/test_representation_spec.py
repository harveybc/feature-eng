"""The representation spec: what round-trips, and what is refused by name.

Every refusal below is a defect that used to be discoverable only after a training run: a feature name nobody builds
becomes a column of zeros, an undeclared holdout becomes a score measured on rows the model had seen, a window list
that is empty becomes a model reading nothing. They are asserted here by their codes, because a code is what stage 3
and stage 4 will match on when they refuse to fit or to score a representation.
"""

import json

import pytest

from feature_eng_m5phet.representation import (
    CALENDAR_VOCABULARY, FEATURE_VOCABULARY, SCHEMA, SpecError,
    canonical_json, dumps, loads, spec_id, validate_spec,
)


def spec(**overrides):
    """A whole representation of the household forecast, which every test below bends in exactly one place."""
    body = {
        "schema": SCHEMA,
        "sampling": {"step_seconds": 60, "timezone": "UTC"},
        "target": {"column": "Global_active_power", "transform": "level"},
        "windows": [60, 1440],
        "lags": [1, 60, 1440],
        "differencing": {"order": 0},
        "calendar": {"clock": "receipt", "columns": ["hour_of_day", "day_of_week"]},
        "features": ["hour_of_day", "day_of_week"],
        "holdout": {"fraction": 0.2},
        "provenance": "DEVELOPMENT",
    }
    body.update(overrides)
    return body


def refusal(obj, **kwargs):
    with pytest.raises(SpecError) as raised:
        validate_spec(obj, **kwargs)
    return raised.value


def test_a_whole_representation_round_trips_through_json_unchanged():
    original = spec()
    assert loads(dumps(original)) == original
    assert validate_spec(original) is original


def test_identity_ignores_annotations_and_follows_the_modelling_keys():
    bare = spec()
    annotated = spec(candidate_id="seasonal", why={"windows": "ACF peak at lag 1440"},
                     not_decided={"calendar.clock": "the file cannot say"})
    validate_spec(annotated)
    assert spec_id(annotated) == spec_id(bare)
    assert "why" not in json.loads(canonical_json(annotated))
    assert spec_id(spec(windows=[30])) != spec_id(bare)


def test_an_undeclared_key_is_refused_and_named():
    error = refusal(spec(seasonality="daily"))
    assert error.code == "UNKNOWN_KEY" and "seasonality" in error.why


def test_a_missing_required_key_is_refused_and_named():
    body = spec()
    del body["holdout"]
    error = refusal(body)
    assert error.code == "MISSING_KEY" and "holdout" in error.why


def test_another_schema_is_not_read_by_this_reader():
    assert refusal(spec(schema="m5phet.representation.v2")).code == "WRONG_SCHEMA"


def test_a_foreign_feature_name_is_refused_by_that_name():
    error = refusal(spec(features=["hour_of_day", "Global_active_power"]))
    assert error.code == "FOREIGN_FEATURE" and "Global_active_power" in error.why


def test_a_misspelled_declared_feature_is_refused_rather_than_corrected():
    assert refusal(spec(features=["hour_of_the_day"])).code == "FOREIGN_FEATURE"


def test_every_declared_feature_validates_and_the_vocabulary_names_its_source():
    validate_spec(spec(features=sorted(FEATURE_VOCABULARY),
                       calendar={"clock": "publication", "columns": sorted(CALENDAR_VOCABULARY)}))
    assert CALENDAR_VOCABULARY <= set(FEATURE_VOCABULARY)
    assert all(origin.startswith("app/") for origin in FEATURE_VOCABULARY.values())


def test_a_caller_may_supply_another_declaration_but_not_widen_this_one():
    validate_spec(spec(features=["EXTERNAL_FEATURE"]), vocabulary={"EXTERNAL_FEATURE": "another installation"})
    assert refusal(spec(features=["hour_of_day"]), vocabulary={}).code == "FOREIGN_FEATURE"


def test_a_technical_indicator_is_not_a_calendar_column():
    error = refusal(spec(calendar={"clock": "receipt", "columns": ["RSI"]}))
    assert error.code == "FOREIGN_CALENDAR_COLUMN" and "RSI" in error.why


def test_an_empty_window_list_is_refused():
    assert refusal(spec(windows=[])).code == "EMPTY_WINDOWS"


def test_windows_and_lags_are_strictly_increasing_positive_integers():
    assert refusal(spec(windows=[1440, 60])).code == "BAD_VALUE"
    assert refusal(spec(lags=[1, 1, 2])).code == "BAD_VALUE"
    assert refusal(spec(windows=[0])).code == "BAD_VALUE"
    assert refusal(spec(lags=[True])).code == "BAD_TYPE"
    validate_spec(spec(lags=[]))                      # a representation may read only windows, and says so explicitly


def test_a_holdout_that_is_not_declared_is_refused():
    assert refusal(spec(holdout={})).code == "HOLDOUT_NOT_DECLARED"
    assert refusal(spec(holdout=None)).code == "HOLDOUT_NOT_DECLARED"
    assert refusal(spec(holdout={"fraction": 0.2, "cut": "2010-01-01T00:00:00+00:00"})).code == "HOLDOUT_NOT_DECLARED"


def test_a_holdout_is_a_fraction_inside_the_series_or_an_instant_that_parses():
    validate_spec(spec(holdout={"cut": "2010-11-01T00:00:00+00:00"}))
    assert refusal(spec(holdout={"fraction": 1})).code == "BAD_VALUE"
    assert refusal(spec(holdout={"fraction": 0})).code == "BAD_VALUE"
    assert refusal(spec(holdout={"cut": "last november"})).code == "BAD_VALUE"


def test_a_transform_and_a_differencing_order_may_not_both_difference():
    assert refusal(spec(target={"column": "x", "transform": "diff"},
                        differencing={"order": 1})).code == "AMBIGUOUS_DIFFERENCING"
    validate_spec(spec(target={"column": "x", "transform": "diff"}, differencing={"order": 0}))
    validate_spec(spec(differencing={"order": 1}))
    assert refusal(spec(differencing={"order": 3})).code == "BAD_VALUE"


def test_the_unreadable_vocabularies_are_refused_by_their_own_codes():
    assert refusal(spec(target={"column": "x", "transform": "returns"})).code == "UNKNOWN_TRANSFORM"
    assert refusal(spec(calendar={"clock": "wall", "columns": []})).code == "UNKNOWN_CLOCK"
    assert refusal(spec(provenance="DEV")).code == "UNKNOWN_PROVENANCE"
    assert refusal(spec(sampling={"step_seconds": 60, "timezone": "Mars/Olympus"})).code == "UNKNOWN_TIMEZONE"
    assert refusal(spec(sampling={"step_seconds": 0, "timezone": "UTC"})).code == "BAD_VALUE"


def test_exogenous_columns_are_names_only_and_never_the_target():
    validate_spec(spec(exogenous=["Voltage", "Global_intensity"]))
    error = refusal(spec(exogenous=["Global_active_power"]))
    assert error.code == "BAD_VALUE" and "Global_active_power" in error.why
    assert refusal(spec(exogenous=["Voltage", "Voltage"])).code == "BAD_VALUE"


def test_a_document_that_is_not_json_is_refused_as_a_representation():
    assert refusal(spec(target="Global_active_power")).code == "BAD_TYPE"
    with pytest.raises(SpecError) as raised:
        loads("{not json")
    assert raised.value.code == "MALFORMED_JSON"
    with pytest.raises(SpecError) as raised:
        loads("[]")
    assert raised.value.code == "BAD_TYPE"
