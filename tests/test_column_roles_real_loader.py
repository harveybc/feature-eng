"""The contract, exercised through the loader and the processor this application runs.

R1 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:
the previous round tested a copied helper. These rules drive `app.data_handler.load_csv` with
a file on disk and `app.data_processor.process_data` with a plugin that refuses to be called,
so a refusal is proven to happen *before* any transformation runs.

Declared scope: every loader `app/plugins/tech_indicator.py` actually calls — the primary
`load_csv`, `load_and_fix_hourly_data` (which re-reads the MAIN input) and
`load_additional_csv` (vix, forex_15m). `load_high_frequency_data` is covered too, under the
key `high_frequency`. `load_sp500_csv` is imported but never called, and is left alone.
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


# --- R1: the other loaders this application really uses ----------------------------------
#
# `app/plugins/tech_indicator.py` calls load_and_fix_hourly_data (on the MAIN input file),
# load_additional_csv (vix, forex_15m) and load_high_frequency_data. Each was outside the
# contract, so a column could reach the run around the primary loader.

from app.data_handler import load_additional_csv, load_and_fix_hourly_data


def test_the_hourly_loader_reads_the_main_input_under_the_main_contract(tmp_path):
    """`load_and_fix_hourly_data(config['input_file'], config)` re-reads the same file."""
    path = tmp_path / "hourly.csv"
    pd.DataFrame({"datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
                  "OPEN": [1.0, 2.0], "HIGH": [1.0, 2.0], "LOW": [1.0, 2.0],
                  "CLOSE": [1.0, 2.0], "available_time": ["2024-01-01 00:00:00",
                                                          "2024-01-01 01:00:00"],
                  "LEAK": [9.0, 9.0]}).to_csv(path, index=False)
    contract = dict(CONTRACT, time="datetime")
    with pytest.raises(ColumnRoleError, match="LEAK"):
        load_and_fix_hourly_data(str(path), dict(column_roles=contract))


def test_an_auxiliary_dataset_carries_its_own_declaration(tmp_path):
    path = tmp_path / "vix.csv"
    pd.DataFrame({"date": ["2024-01-01", "2024-01-02"],
                  "vix_close": [13.0, 14.0]}).to_csv(path, index=False)
    config = {"column_roles": CONTRACT,
              "column_roles_by_file": {"vix": {"time": "date", "features": ["vix_close"]}}}
    data = load_additional_csv(str(path), dataset_type="vix", config=config)
    assert list(data.columns) == ["vix_close"]
    assert config["column_roles_applied"]["vix"]["features"] == ["vix_close"]


def test_an_auxiliary_dataset_with_no_declaration_is_refused_by_its_key(tmp_path):
    """The main contract does not cover a second dataset, and silence is not permission."""
    path = tmp_path / "vix.csv"
    pd.DataFrame({"date": ["2024-01-01"], "vix_close": [13.0]}).to_csv(path, index=False)
    with pytest.raises(ColumnRoleError, match="vix"):
        load_additional_csv(str(path), dataset_type="vix",
                            config={"column_roles": CONTRACT})


def test_an_undeclared_column_in_an_auxiliary_dataset_is_refused_by_name(tmp_path):
    path = tmp_path / "vix.csv"
    pd.DataFrame({"date": ["2024-01-01"], "vix_close": [13.0],
                  "SURPRISE": [1.0]}).to_csv(path, index=False)
    config = {"column_roles_by_file": {"vix": {"time": "date", "features": ["vix_close"]}}}
    with pytest.raises(ColumnRoleError, match="SURPRISE"):
        load_additional_csv(str(path), dataset_type="vix", config=config)


def test_an_auxiliary_loader_without_a_configuration_keeps_working(tmp_path):
    """These loaders are also called with no configuration; that path is left as it was."""
    path = tmp_path / "vix.csv"
    pd.DataFrame({"date": ["2024-01-01"], "vix_close": [13.0]}).to_csv(path, index=False)
    data = load_additional_csv(str(path), dataset_type="vix", config=None)
    assert "vix_close" in data.columns


def test_the_high_frequency_loader_is_declared_too(tmp_path):
    from app.data_handler import load_high_frequency_data

    path = tmp_path / "hf.csv"
    pd.DataFrame({"DATE_TIME": ["2024.01.01 00:00:00", "2024.01.01 00:15:00"],
                  "OPEN": [1.0, 2.0], "SURPRISE": [9.0, 9.0]}).to_csv(path, index=False)
    config = {"column_roles_by_file": {"high_frequency": {"time": "DATE_TIME",
                                                          "features": ["OPEN"]}}}
    with pytest.raises(ColumnRoleError, match="SURPRISE"):
        load_high_frequency_data(str(path), config)
