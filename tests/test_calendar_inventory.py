"""The inventory's own arithmetic, on synthetic rows: the ladder, the three-valued verdict, and the refusals.

`docs/CALENDAR_DATASET_INVENTORY.md` is only worth reading if the job that produced it counts correctly, and the one
count it would be easiest to get wrong is the vintage count -- because the wrong answer (every coarse-key collision is a
revision) is the one that makes a dataset look richer than it is. So the verdict logic is pinned here on rows built in
the test, where the right answer is known by construction and no file is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feature_eng_m5phet.calendar_inventory import (CAL_FIELD_ROLES, RESOURCES, InventoryRefusal, _collisions,
                                                   _kind_of, _provenance, _vintage_verdict, build_inventory,
                                                   measure_resource)


def test_a_key_that_does_not_identify_a_release_is_not_a_vintage():
    """Two rows, one key, two values -- and nothing to date them. Undecidable, never VINTAGES_PRESENT."""
    finest = _collisions({("US", "CPI", "2026-02"): ["2.4", "0.4"]})
    assert finest["keys_whose_rows_disagree_about_the_value"] == 1
    verdict = _vintage_verdict(finest, has_observation_clock=False, has_version_field=False,
                              finest_key=["country", "release", "period"])
    assert verdict["verdict"] == "VINTAGE_UNDECIDABLE"
    assert "REVISION of one release from two distinct releases" in verdict["reading"]


@pytest.mark.parametrize("clock,version", [(True, False), (False, True), (True, True)])
def test_the_same_disagreement_is_a_vintage_once_something_can_date_it(clock, version):
    finest = _collisions({("US", "CPI", "2026-02"): ["2.4", "0.4"]})
    verdict = _vintage_verdict(finest, has_observation_clock=clock, has_version_field=version, finest_key=["k"])
    assert verdict["verdict"] == "VINTAGES_PRESENT"


def test_one_value_per_key_is_the_finding_that_no_earlier_version_survives():
    finest = _collisions({("US", "CPI", "2026-02"): ["2.4"], ("US", "CPI", "2026-03"): ["2.5"]})
    assert finest["keys_whose_rows_disagree_about_the_value"] == 0
    verdict = _vintage_verdict(finest, has_observation_clock=True, has_version_field=True, finest_key=["k"])
    assert verdict["verdict"] == "NO_VINTAGES"
    assert "cannot be reconstructed" in verdict["reading"]


def test_two_rows_with_the_same_value_are_one_value_and_not_a_disagreement():
    finest = _collisions({("US", "CPI", "2026-02"): ["2.4", "2.4"]})
    assert finest["keys_with_more_than_one_row"] == 1
    assert finest["keys_whose_rows_disagree_about_the_value"] == 0


def test_an_empty_cell_and_a_nan_are_the_same_absence():
    stats = _kind_of(["2.4", "", None, float("nan"), "0.4"])
    assert stats["non_null"] == 2 and stats["null_or_empty"] == 3
    assert stats["types"] == {"str": 2}


def test_a_column_declared_for_a_role_and_absent_from_the_file_is_a_refusal_not_a_silent_skip(tmp_path, monkeypatch):
    path = tmp_path / "financial-data" / "toy.csv"
    path.parent.mkdir(parents=True)
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setitem(RESOURCES, "toy", {
        "path": "financial-data/toy.csv", "kind": "csv_no_header", "columns": ("a", "b"),
        "roles": {"actual": "no_such_column"}, "series_key": ("a",), "period_key": ("b",),
        "value_column": "a", "clock_column": "b", "clock_is_tz_aware": False})
    with pytest.raises(InventoryRefusal, match="DECLARED_COLUMN_ABSENT"):
        measure_resource("toy", tmp_path)


def test_a_resource_that_is_not_on_this_machine_asserts_nothing_about_itself(tmp_path):
    record = measure_resource("fred_cpi_yoy_actuals", tmp_path / "nowhere")
    assert record["status"] == "ABSENT"
    assert "nothing about it is asserted" in record["reading"]
    assert "rows" not in record and "vintages" not in record


def test_a_provenance_digest_that_does_not_match_the_bytes_is_reported_as_not_matching(tmp_path):
    data = tmp_path / "actuals.parquet"
    data.write_bytes(b"not really a parquet file")
    (tmp_path / "provenance.json").write_text(json.dumps({
        "source": "toy", "acquired_at": "2026-05-01T00:00:00+00:00",
        "files": [{"path": "somewhere/actuals.parquet", "sha256": "0" * 64}]}), encoding="utf-8")
    result = _provenance(data, "deadbeef")
    assert result["present"] is True
    assert result["declared_digest_matches_these_bytes"] is False
    assert result["file_grain_receipt_instant"] == "2026-05-01T00:00:00+00:00"
    assert "FILE-grain" in result["reading"]


def test_a_file_grain_receipt_clock_is_never_reported_as_the_receipt_role():
    """The distinction the whole inventory turns on: `acquired_at` dates a download, not a release."""
    for name in RESOURCES:
        spec = RESOURCES[name]
        assert "receipt_instant" not in spec["roles"], f"{name} must not claim a per-release receipt clock"


def test_every_cal_case_is_named_by_at_least_one_field_role():
    """CAL01-CAL12 each depend on something, so each must be reachable from the role table the skips read."""
    named = {case for cases in CAL_FIELD_ROLES.values() for case in cases}
    assert named >= {f"CAL{n:02d}" for n in range(1, 13)} - {"CAL06"}, sorted(named)
    assert "CAL06" in {c for cases in CAL_FIELD_ROLES.values() for c in cases}


def test_the_inventory_lists_every_case_blocked_by_a_missing_field(tmp_path):
    """With no resource present at all, the blocked-case map is empty rather than wrong: nothing was measured."""
    inventory = build_inventory(tmp_path / "nowhere")
    assert inventory["schema"] == "m5phet.calendar_inventory.v1"
    assert all(r["status"] == "ABSENT" for r in inventory["resources"])
    assert inventory["cases_blocked_by_a_missing_field"] == {}


def test_the_module_reads_no_network():
    import ast

    source = Path(__file__).resolve().parents[1] / "feature_eng_m5phet" / "calendar_inventory.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported == {"__future__", "argparse", "csv", "hashlib", "json", "sys", "collections", "pathlib", "pandas"}
