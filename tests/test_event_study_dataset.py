"""The flat table an EconML study is fitted from: the same rows, the same split, and every omission by name.

The table itself carries no result, so nothing here checks a number for being right. What it checks is the three
things a later stage would silently get wrong: that the held-out events are the SAME events the projections document
holds out (otherwise the closure table compares two holdouts), that the modifier cut was measured on the fitting rows
only (otherwise the holdout defines the subgroup it is scored in), and that a column which cannot be built -- a
constant weekday, a regime no fitted reference can assign -- is named rather than filled.

The world is the planted one from `test_event_projections.py`, for the same reason `test_evaluate_events.py` uses it:
a market where a response was planted is a market where the table's own columns can be checked against what was
planted into them.
"""

import csv
import json

import pytest

from feature_eng_m5phet import event_study_dataset as esd
from feature_eng_m5phet import local_projections as lp

from tests.test_event_projections import additive_planter, world


@pytest.fixture(scope="module")
def rows_document(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("table")
    document, _bars, _calendar = world(tmp_path, planter=additive_planter)
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return tmp_path, path, document


def test_one_row_per_release_and_horizon_with_the_declared_columns(rows_document):
    _tmp, path, _document = rows_document
    table = esd.build(str(path))
    assert table["schema"] == esd.SCHEMA
    assert table["treatment"] == "surprise" and table["treatment_kind"] == "continuous"
    assert table["outcomes"] == ["log_return", "realized_vol"]
    assert table["tables"]
    for entry in table["tables"]:
        keys = [row["event_key"] for row in entry["fit"]] + [row["event_key"] for row in entry["holdout"]]
        assert len(set(keys)) == len(keys), "a release appears twice at one horizon"
        for row in entry["fit"]:
            assert set(table["columns"]) <= set(row)


def test_the_held_out_events_are_the_ones_the_projection_holds_out(rows_document):
    """One split rule, read through `local_projections.prepare`: the two arms must name the same releases."""
    _tmp, path, _document = rows_document
    table = esd.build(str(path))
    prepared = lp.prepare(str(path))
    for entry in table["tables"]:
        _fit_keys, holdout_keys = prepared["splits"][entry["event_type"]]
        assert {row["event_key"] for row in entry["holdout"]} <= set(holdout_keys)
        assert not {row["event_key"] for row in entry["fit"]} & set(holdout_keys)


def test_the_volatility_cut_is_the_median_of_the_fitting_rows_and_the_holdout_is_cut_at_it(rows_document):
    _tmp, path, _document = rows_document
    table = esd.build(str(path))
    entry = next(e for e in table["tables"] if e["counts"]["fit_rows"] > 4 and e["counts"]["holdout_rows"] > 0)
    cut = entry["pre_event_vol_cut"]
    assert cut == esd._median([row["pre_event_realized_vol"] for row in entry["fit"]])
    for row in entry["fit"] + entry["holdout"]:
        assert row["pre_event_vol_high"] == (1 if row["pre_event_realized_vol"] > cut else 0)


def test_the_sign_column_is_the_sign_of_the_treatment_and_nothing_else(rows_document):
    _tmp, path, _document = rows_document
    table = esd.build(str(path))
    for entry in table["tables"]:
        for row in entry["fit"] + entry["holdout"]:
            assert row["surprise_positive"] == (1 if row["surprise"] > 0 else 0)


def test_a_column_that_does_not_vary_is_withheld_by_name_and_still_written(rows_document):
    """A weekly release has one weekday; the manifest must say so rather than offer a constant confounder."""
    _tmp, path, _document = rows_document
    table = esd.build(str(path), hour_bucket_edges=(0,))
    withheld = [entry for entry in table["tables"] if entry["roles_withheld"]]
    assert withheld, "the single-bucket hour column is constant and must be withheld somewhere"
    for entry in withheld:
        for name, reason in entry["roles_withheld"].items():
            assert reason["refusal"] == esd.CONSTANT_IN_THE_TABLE
            assert name in table["columns"], "a withheld column is still written into the table"
            assert name not in entry["roles_offered"]


def test_the_regime_column_is_absent_and_says_why_when_no_reference_was_named(rows_document):
    _tmp, path, _document = rows_document
    table = esd.build(str(path))
    assert table["regime"]["status"] == "ABSENT"
    assert table["regime"]["refusal"] == esd.REGIME_NOT_ASKED_FOR
    assert "regime" not in table["columns"]


def test_a_reference_that_cannot_assign_these_rows_is_refused_by_name_not_approximated(rows_document, tmp_path):
    """The fitted reference declares the features it assigns on; an event row does not carry them."""
    pytest.importorskip("sklearn")
    from feature_eng_m5phet.regimes import HierarchicalRegimes

    reference = HierarchicalRegimes.fit_reference(
        [{"row_id": f"r{index}", "body_pipettes": float(index % 7), "range_pipettes": float(index % 5) + 1.0}
         for index in range(40)],
        features=["body_pipettes", "range_pipettes"], levels=[2, 4], task_id="not-an-event-reference")
    path_to_reference = tmp_path / "reference.joblib"
    reference.save(path_to_reference)

    _tmp, path, _document = rows_document
    table = esd.build(str(path), regime_reference=str(path_to_reference))
    assert table["regime"]["status"] == "ABSENT"
    assert table["regime"]["refusal"] == esd.REGIME_FEATURES_NOT_IN_THE_EVENT_ROWS
    assert table["regime"]["reference_features"] == ["body_pipettes", "range_pipettes"]
    assert "regime" not in table["columns"]


def test_an_unreadable_reference_is_refused_by_its_own_name(rows_document, tmp_path):
    _tmp, path, _document = rows_document
    missing = tmp_path / "nothing.joblib"
    table = esd.build(str(path), regime_reference=str(missing))
    assert table["regime"]["refusal"] == esd.REGIME_REFERENCE_NOT_READABLE


def test_the_written_csvs_carry_every_declared_column_and_the_index_names_the_files(rows_document, tmp_path):
    _tmp, path, _document = rows_document
    table = esd.build(str(path))
    out = esd.write(table, tmp_path / "out")
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["schema"] == esd.SCHEMA and "fit" not in index["tables"][0]
    for entry in index["tables"]:
        for split, name in entry["files"].items():
            with (out / name).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            assert list(rows[0]) == index["columns"] if rows else True
            assert len(rows) == entry["counts"][f"{split}_rows"]


def test_two_runs_over_the_same_rows_write_the_same_bytes(rows_document, tmp_path):
    _tmp, path, _document = rows_document
    first = esd.write(esd.build(str(path)), tmp_path / "a")
    second = esd.write(esd.build(str(path)), tmp_path / "b")
    for name in sorted(item.name for item in first.iterdir()):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_bad_hour_buckets_are_refused_by_name(rows_document):
    _tmp, path, _document = rows_document
    with pytest.raises(esd.TableRefusal) as refused:
        esd.build(str(path), hour_bucket_edges=(3, 9))
    assert refused.value.code == "BAD_HOUR_BUCKETS"


def test_an_empty_rows_document_is_refused_rather_than_producing_an_empty_table(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"schema": "m5phet.event_rows.v1", "parameters": {}, "counts": {"releases_read": 0},
                                "excluded": {"counts": {}}, "rows": []}), encoding="utf-8")
    with pytest.raises(lp.ProjectionRefusal) as refused:
        esd.build(str(path))
    assert refused.value.code == "NO_EVENT_ROWS"


def test_the_neighbour_window_is_a_choice_and_the_document_says_which_one_was_applied(rows_document):
    """WP22 step 7 lets the chooser pick W; a shorter window sums fewer neighbours, and both numbers are declared."""
    _tmp, path, _document = rows_document
    wide = esd.build(str(path))
    narrow = esd.build(str(path), window_hours=2.0)
    assert wide["window_hours"] == wide["window_hours_declared_by_the_rows"]
    assert narrow["window_hours"] == 2.0
    assert narrow["window_hours_declared_by_the_rows"] == wide["window_hours"]
    by_key = {row["event_key"]: row for row in narrow["tables"][0]["fit"]}
    changed = [row for row in wide["tables"][0]["fit"]
               if row["event_key"] in by_key
               and abs(row["other_surprises_before"] - by_key[row["event_key"]]["other_surprises_before"]) > 0]
    assert changed, "a shorter window must sum fewer neighbours somewhere in this world"


def test_a_window_that_is_not_a_duration_is_refused_by_name(rows_document):
    _tmp, path, _document = rows_document
    with pytest.raises(lp.ProjectionRefusal) as refused:
        esd.build(str(path), window_hours=0)
    assert refused.value.code == "BAD_WINDOW"


def test_the_table_names_the_bars_the_labels_were_read_off(rows_document):
    """The table path identifies a design; the bars identify the labels, and a seal over these rows names the bars."""
    _tmp, path, document = rows_document
    table = esd.build(str(path))
    assert table["bars"]["path"] == document["bars"]["path"]
    assert table["bars"]["sha256"] == document["bars"]["sha256"]
