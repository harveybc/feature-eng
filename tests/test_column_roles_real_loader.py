"""The contract, exercised through the loader and the processor this application runs.

R1 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:
the previous round tested a copied helper. These rules drive `app.data_handler.load_csv` with
a file on disk and `app.data_processor.process_data` with a plugin that refuses to be called,
so a refusal is proven to happen *before* any transformation runs.

Declared scope: the primary input path. `load_additional_csv`, `load_sp500_csv` and the
hourly/high-frequency loaders are not covered here and still coerce their own columns.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from app.column_roles import ColumnRoleError
from app.data_handler import load_csv
from app.data_processor import process_data

CONTRACT = {"time": "DATE_TIME", "features": ["OPEN", "HIGH", "LOW", "CLOSE"],
            "targets": ["CLOSE"], "metadata": ["available_time"],
            "allow_target_as_feature": True}


class PluginThatMustNotRun:
    """Its methods are the assertion: a refused contract never reaches a transformation."""

    def process(self, data):  # pragma: no cover - reaching it is the failure
        raise AssertionError("plugin.process ran on a contract that should have been refused")

    def process_additional_datasets(self, data, config):  # pragma: no cover
        raise AssertionError("process_additional_datasets ran on a refused contract")


def csv_at(tmp_path, columns, rows=3):
    stamps = [f"2024-01-0{i + 1} 00:00:00" for i in range(rows)]
    body = {name: (stamps if name in ("DATE_TIME", "available_time")
                   else [float(i + 1) for i in range(rows)]) for name in columns}
    path = tmp_path / "series.csv"
    pd.DataFrame(body).to_csv(path, index=False)
    return str(path)


def frame(columns, rows=3):
    stamps = pd.to_datetime([f"2024-01-0{i + 1} 00:00:00" for i in range(rows)])
    return pd.DataFrame({name: (stamps if name in ("DATE_TIME", "available_time")
                                else [float(i + 1) for i in range(rows)])
                         for name in columns})


def test_the_declared_features_survive_in_the_declared_order(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "CLOSE", "OPEN", "available_time", "HIGH", "LOW"])
    config = dict(column_roles=CONTRACT)
    loaded = load_csv(path, config)
    assert list(loaded.columns) == ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"]
    assert len(config["column_roles_applied"]["input_file"]["contract_sha256"]) == 64


def test_an_undeclared_column_stops_the_load(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time",
                             "SURPRISE"])
    with pytest.raises(ColumnRoleError, match="SURPRISE"):
        load_csv(path, dict(column_roles=CONTRACT))


def test_the_target_overlap_needs_the_declaration_at_the_real_loader(tmp_path):
    """Musashi's first counterexample, through the loader rather than through the helper."""
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    without = {key: value for key, value in CONTRACT.items()
               if key != "allow_target_as_feature"}
    with pytest.raises(ColumnRoleError, match="CLOSE"):
        load_csv(path, dict(column_roles=without))


def test_a_timestamp_declared_as_a_feature_is_refused_not_coerced(tmp_path):
    """Coercion ran before the contract, so the timestamp became NaN and the run continued."""
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN", "available_time"])
    contract = {"time": "DATE_TIME", "features": ["OPEN", "available_time"], "metadata": []}
    with pytest.raises(ColumnRoleError, match="available_time"):
        load_csv(path, dict(column_roles=contract))


def test_the_processor_refuses_before_any_transformation_runs():
    """`process_data` coerces the OHLC block with fillna(0): a timestamp would become 0."""
    data = frame(["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time"])
    contract = {"time": "DATE_TIME",
                "features": ["OPEN", "HIGH", "LOW", "CLOSE", "available_time"],
                "targets": ["CLOSE"], "metadata": [], "allow_target_as_feature": True}
    with pytest.raises(ColumnRoleError, match="available_time"):
        process_data(data, PluginThatMustNotRun(), dict(column_roles=contract))


def test_the_processor_refuses_an_undeclared_column_before_the_plugin():
    data = frame(["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "available_time", "SURPRISE"])
    with pytest.raises(ColumnRoleError, match="SURPRISE"):
        process_data(data, PluginThatMustNotRun(), dict(column_roles=CONTRACT))


def test_a_run_without_a_contract_is_refused_at_the_loader(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN"])
    with pytest.raises(ColumnRoleError, match="column_roles"):
        load_csv(path, {})


def test_the_declared_legacy_migration_still_loads_everything(tmp_path):
    path = csv_at(tmp_path, ["DATE_TIME", "OPEN"])
    loaded = load_csv(path, {"column_roles_migration": "LEGACY_ALL_COLUMNS_ARE_FEATURES"})
    assert list(loaded.columns) == ["DATE_TIME", "OPEN"]
