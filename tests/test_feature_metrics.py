"""The metric sheet, checked against columns whose answer is known before the job runs.

A column built as `2 * a + noise` has a correlation close to one and nothing to argue about; a column with four
missing cells out of a thousand has a missing fraction of exactly 0.004; a sinusoid of period P has its
autocorrelation peak at P. Those are the assertions. The two that are not about a number are just as important:
the sheet must be the same bytes on two runs, and a measurement the environment cannot take must be refused BY
NAME rather than replaced.

Nothing here fits a model and nothing needs a GPU.
"""

import builtins
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from feature_eng_m5phet import metrics
from feature_eng_m5phet.design import DesignRefusal

PERIOD = 24
ROWS = 1000


def write_csv(path, columns, step_seconds=60, start="2020-01-01T00:00:00"):
    """A CSV on an exact one-minute grid; `columns` maps a name to a sequence, `None` being a missing cell."""
    first = datetime.fromisoformat(start)
    names = list(columns)
    rows = len(columns[names[0]])
    lines = [",".join(["timestamp"] + names)]
    for i in range(rows):
        stamp = (first + timedelta(seconds=i * step_seconds)).isoformat()
        cells = ["" if columns[name][i] is None else repr(float(columns[name][i])) for name in names]
        lines.append(",".join([stamp] + cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def blocks():
    """Two correlated blocks and one independent column: the structure step 3 has to recover, built once here.

    `a1, a2, a3` are one latent series plus small independent wobbles; `b1, b2` are a second, unrelated latent
    series; `c` shares a latent with nobody. Everything is deterministic: one seeded generator, drawn once.
    """
    rng = np.random.default_rng(20260925)
    index = np.arange(ROWS)
    first = np.sin(2 * np.pi * index / PERIOD)
    second = np.cos(2 * np.pi * index / 137.0) * 5.0
    return {
        "a1": first + rng.normal(0, 0.02, ROWS),
        "a2": 2.0 * first + rng.normal(0, 0.02, ROWS),
        "a3": -first + rng.normal(0, 0.02, ROWS),
        "b1": second + rng.normal(0, 0.05, ROWS),
        "b2": 0.5 * second + rng.normal(0, 0.05, ROWS),
        "c": rng.normal(0, 1.0, ROWS),
    }


@pytest.fixture()
def sheet(tmp_path):
    path = write_csv(tmp_path / "blocks.csv", {name: list(values) for name, values in blocks().items()})
    return metrics.feature_metrics(path, "a1")


def test_the_sheet_validates_against_its_own_reader(sheet):
    assert sheet["schema"] == metrics.SCHEMA
    assert metrics.validate(sheet) is sheet
    assert sorted(sheet["features"]) == ["a1", "a2", "a3", "b1", "b2", "c"]
    assert len(sheet["pairs"]) == 15                      # 6 features choose 2


def test_every_feature_block_has_sorted_keys_so_a_state_text_is_stable(sheet):
    for name, block in sheet["features"].items():
        assert list(block) == sorted(block), name


def test_a_known_period_shows_as_an_autocorrelation_peak(sheet):
    peaks = [peak["lag"] for peak in sheet["features"]["a1"]["acf"]["peaks"]]
    assert peaks and peaks[0] == PERIOD, peaks
    assert sheet["features"]["a1"]["acf"]["decay_lag"] is not None
    assert sheet["features"]["a1"]["stationarity"]["verdict"] in ("STATIONARY", "NON_STATIONARY", "INCONCLUSIVE")


def test_a_correlated_pair_and_an_independent_column_are_told_apart(sheet):
    assert sheet["pairs"]["a1::a2"]["pearson"]["value"] > 0.99
    assert sheet["pairs"]["a1::a3"]["pearson"]["value"] < -0.99
    assert abs(sheet["pairs"]["a1::c"]["pearson"]["value"]) < 0.2
    assert abs(sheet["pairs"]["a1::b1"]["pearson"]["value"]) < 0.5
    assert sheet["pairs"]["a1::a2"]["spearman"]["value"] > 0.99


def test_the_missing_fraction_is_exact(tmp_path):
    values = list(np.linspace(0.0, 1.0, ROWS))
    for index in (3, 17, 250, 999):
        values[index] = None
    path = write_csv(tmp_path / "gappy.csv", {"y": values, "x": list(np.linspace(1.0, 2.0, ROWS))})
    sheet = metrics.feature_metrics(path, "x")
    assert sheet["features"]["y"]["missing"] == 4
    assert sheet["features"]["y"]["missing_fraction"] == 0.004
    assert sheet["features"]["x"]["missing_fraction"] == 0.0


def test_the_distribution_summary_reproduces_numpy_on_the_finite_cells(tmp_path):
    column = blocks()["c"]
    path = write_csv(tmp_path / "one.csv", {"c": list(column), "x": list(np.arange(ROWS, dtype=float))})
    block = metrics.feature_metrics(path, "x")["features"]["c"]["distribution"]
    assert block["mean"] == pytest.approx(float(column.mean()), abs=1e-6)
    assert block["std"] == pytest.approx(float(column.std(ddof=1)), abs=1e-6)
    assert block["minimum"] == pytest.approx(float(column.min()), abs=1e-6)
    assert block["quantile_99"] == pytest.approx(float(np.quantile(column, 0.99)), abs=1e-6)
    assert abs(block["skew"]) < 0.5 and abs(block["excess_kurtosis"]) < 1.0


def test_the_lagged_cross_correlation_is_measured_at_the_declared_lags(tmp_path):
    index = np.arange(ROWS)
    wave = np.sin(2 * np.pi * index / PERIOD)
    lead = np.concatenate([wave[6:], wave[:6]])          # a column whose value at t-6 is the target at t
    path = write_csv(tmp_path / "lead.csv", {"y": list(wave), "lead": list(lead)})
    sheet = metrics.feature_metrics(path, "y", lags=[6, 12])
    by_lag = {entry["lag"]: entry["correlation"]
              for entry in sheet["features"]["lead"]["cross_correlation_to_target"]["by_lag"]}
    assert sorted(by_lag) == [6, 12]
    assert by_lag[6] > 0.99, by_lag
    assert abs(by_lag[12]) < 0.2, by_lag
    assert sheet["lags"] == [6, 12] and sheet["lags_source"] == "the --lags argument"


def test_the_default_lags_are_the_declared_ones(sheet):
    assert sheet["lags"] == list(metrics.DEFAULT_LAGS) == [1, 24]
    assert sheet["lags_source"] == "the declared default of this job"


def test_mutual_information_degrades_by_name_when_sklearn_is_absent(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn in this environment (monkeypatched)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    assert metrics._mutual_info() is None
    path = write_csv(tmp_path / "blocks.csv", {name: list(values) for name, values in blocks().items()})
    sheet = metrics.feature_metrics(path, "a1")
    block = sheet["pairs"]["a1::a2"]["mutual_information"]
    assert block["status"] == "NOT_AVAILABLE" and block["value"] is None
    assert "scikit-learn is not installed" in block["reason"]
    assert sheet["environment"]["sklearn"] == "NOT_AVAILABLE"
    assert sheet["pairs"]["a1::a2"]["pearson"]["value"] > 0.99      # the correlations are still there


@pytest.mark.skipif(metrics._mutual_info() is None, reason="scikit-learn is not installed")
def test_mutual_information_is_larger_for_the_dependent_pair(sheet):
    dependent = sheet["pairs"]["a1::a2"]["mutual_information"]
    independent = sheet["pairs"]["a1::c"]["mutual_information"]
    assert dependent["status"] == independent["status"] == "OK"
    assert dependent["value"] > independent["value"]
    assert dependent["random_state"] == metrics.MI_RANDOM_STATE


def test_two_runs_produce_the_same_bytes(tmp_path):
    path = write_csv(tmp_path / "blocks.csv", {name: list(values) for name, values in blocks().items()})
    first = metrics.dumps(metrics.feature_metrics(path, "a1"))
    second = metrics.dumps(metrics.feature_metrics(path, "a1"))
    assert first == second
    assert json.loads(first)["dataset"]["sha256"] == json.loads(second)["dataset"]["sha256"]


def test_the_decision_payload_carries_measurements_and_no_rows(sheet):
    payload = metrics.decision_payload(sheet, "a2")
    assert payload["kind"] == "feature_profile" and payload["feature"] == "a2"
    assert list(payload) == sorted(payload)
    assert payload["stationarity"] == sheet["features"]["a2"]["stationarity"]["verdict"]
    assert payload["missing_fraction"] == 0.0
    assert set(payload["cross_correlation_to_target"]) == {"1", "24"}
    assert json.dumps(payload, allow_nan=False)                     # renderable, and no NaN ever reaches a state text
    with pytest.raises(DesignRefusal) as refusal:
        metrics.decision_payload(sheet, "not_a_column")
    assert refusal.value.code == "FEATURE_NOT_IN_SHEET"


def test_a_sheet_missing_a_declared_key_is_refused_by_name(sheet):
    broken = dict(sheet)
    broken.pop("pairs")
    with pytest.raises(DesignRefusal) as refusal:
        metrics.validate(broken)
    assert refusal.value.code == "MISSING_KEY"
    with pytest.raises(DesignRefusal) as second:
        metrics.validate(dict(sheet, surprise=1))
    assert second.value.code == "UNKNOWN_KEY"


def test_two_lag_sources_are_refused(tmp_path):
    path = write_csv(tmp_path / "blocks.csv", {name: list(values) for name, values in blocks().items()})
    with pytest.raises(DesignRefusal) as refusal:
        metrics.feature_metrics(path, "a1", lags=[1], spec=str(tmp_path / "spec.json"))
    assert refusal.value.code == "TWO_LAG_SOURCES"


def test_the_lags_can_come_from_a_design_document(tmp_path):
    from feature_eng_m5phet import design
    path = write_csv(tmp_path / "blocks.csv", {name: list(values) for name, values in blocks().items()})
    document = design.design(path, "a1")
    spec_path = tmp_path / "candidates.json"
    spec_path.write_text(json.dumps(document), encoding="utf-8")
    sheet = metrics.feature_metrics(path, "a1", spec=str(spec_path))
    expected = sorted({lag for candidate in document["candidates"] for lag in candidate["lags"]})
    assert sheet["lags"] == expected and "design document" in sheet["lags_source"]


def test_the_cli_writes_the_sheet(tmp_path, capsys):
    path = write_csv(tmp_path / "blocks.csv", {name: list(values) for name, values in blocks().items()})
    out = tmp_path / "feature_metrics.json"
    assert metrics.main(["--data", path, "--target", "a1", "--out", str(out)]) == 0
    metrics.validate(json.loads(out.read_text(encoding="utf-8")))
    assert metrics.main(["--data", path, "--target", "nope", "--out", str(out)]) == 2
    assert "REFUSED TARGET_NOT_IN_DATASET" in capsys.readouterr().err
