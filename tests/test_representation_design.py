"""The design job, checked against series whose answer is known before the job runs.

A synthetic series with period P has one right answer about P, a random walk has one right answer about stationarity,
and a file with no timestamp column has no answer at all. Those three are asserted here, together with the invariant
that binds stage 2 to stage 1: whatever the job emits, `validate_spec` reads it. A candidate the spec would refuse is
a candidate the fit job could never consume.

Nothing here fits a model, and nothing here needs a GPU: the job is arithmetic over one column.
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from feature_eng_m5phet import design
from feature_eng_m5phet.representation import SCHEMA as SPEC_SCHEMA, validate_spec

PERIOD = 24


def write_csv(path, rows, header=("timestamp", "value"), step_seconds=60, start="2020-01-01T00:00:00"):
    """A CSV with a timestamp column on an exact grid, which is what the job's other tests assume."""
    first = datetime.fromisoformat(start)
    lines = [",".join(header)]
    for i, row in enumerate(rows):
        values = row if isinstance(row, (list, tuple)) else [row]
        stamp = (first + timedelta(seconds=i * step_seconds)).isoformat()
        lines.append(",".join([stamp] + ["" if v is None else repr(float(v)) for v in values]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def periodic(n=2000, period=PERIOD, noise=0.05):
    """A sinusoid of known period with a little reproducible noise; the period is the only structure it has."""
    index = np.arange(n)
    wobble = np.sin(index * 7.0) * noise            # deterministic, seedless, and not periodic at `period`
    return np.sin(2 * np.pi * index / period) + wobble


def random_walk(n=2000):
    """A random walk built from a fixed seed, so the verdict below is the same on every machine."""
    return np.cumsum(np.random.default_rng(20260924).standard_normal(n))


def run(tmp_path, values, target="value", **kwargs):
    csv_path = write_csv(tmp_path / "series.csv", values)
    return design.design(str(csv_path), target, **kwargs)


def test_a_known_period_appears_as_a_window_and_as_a_lag(tmp_path):
    document = run(tmp_path, periodic())
    windows = {window for candidate in document["candidates"] for window in candidate["windows"]}
    lags = {lag for candidate in document["candidates"] for lag in candidate["lags"]}
    assert PERIOD in windows, f"the candidate windows {sorted(windows)} do not contain the period {PERIOD}"
    assert PERIOD in lags, f"the candidate lags {sorted(lags)} do not contain the period {PERIOD}"
    peaks = [peak["lag"] for peak in document["tests"]["seasonality"]["peaks"]]
    assert peaks[0] == PERIOD and document["tests"]["seasonality"]["status"] == "OK"


def test_the_window_and_the_lag_name_the_measurement_that_motivated_them(tmp_path):
    document = run(tmp_path, periodic())
    seasonal = [c for c in document["candidates"] if c["candidate_id"] == f"seasonal_lag_{PERIOD}"]
    assert seasonal, [c["candidate_id"] for c in document["candidates"]]
    why = seasonal[0]["why"]
    assert f"ACF peak at lag {PERIOD}" in why["windows"] and f"window {PERIOD}" in why["windows"]
    assert f"lags [1, {PERIOD}]" in why["lags"]
    assert str(document["tests"]["seasonality"]["significance_band"]) in why["windows"]


def test_a_random_walk_is_read_as_non_stationary_and_one_candidate_differences_it(tmp_path):
    document = run(tmp_path, random_walk())
    stationarity = document["tests"]["stationarity"]
    assert stationarity["verdict"] == "NON_STATIONARY", stationarity
    assert stationarity["heuristic"]["verdict"] == "NON_STATIONARY"
    differenced = [c for c in document["candidates"] if c["target"]["transform"] == "diff"]
    assert differenced, [c["target"]["transform"] for c in document["candidates"]]
    assert differenced[0]["differencing"]["order"] == 0            # the transform differences; the order does not
    assert "NON_STATIONARY" in differenced[0]["why"]["transform"]


def test_a_stationary_series_is_not_differenced_by_any_candidate(tmp_path):
    document = run(tmp_path, periodic())
    assert document["tests"]["stationarity"]["verdict"] == "STATIONARY"
    assert {c["target"]["transform"] for c in document["candidates"]} == {"level"}


def test_a_dataset_without_a_timestamp_column_is_refused_by_name(tmp_path):
    path = tmp_path / "no_time.csv"
    path.write_text("alpha,value\n" + "\n".join(f"{i},{i * 0.5}" for i in range(200)) + "\n", encoding="utf-8")
    with pytest.raises(design.DesignRefusal) as raised:
        design.design(str(path), "value")
    assert raised.value.code == "NO_TIMESTAMP_COLUMN"
    assert "alpha" in raised.value.why and "--time-column" in raised.value.why


def test_an_unusual_timestamp_column_is_accepted_only_when_it_is_named(tmp_path):
    path = write_csv(tmp_path / "odd.csv", periodic(200), header=("instant", "value"))
    with pytest.raises(design.DesignRefusal) as raised:
        design.design(str(path), "value")
    assert raised.value.code == "NO_TIMESTAMP_COLUMN"
    document = design.design(str(path), "value", time_column="instant")
    assert document["dataset"]["time_column"] == "instant"


def test_the_target_is_refused_by_name_when_the_file_does_not_carry_it(tmp_path):
    path = write_csv(tmp_path / "series.csv", periodic(200))
    with pytest.raises(design.DesignRefusal) as raised:
        design.design(str(path), "Global_active_power")
    assert raised.value.code == "TARGET_NOT_IN_DATASET" and "Global_active_power" in raised.value.why
    with pytest.raises(design.DesignRefusal) as raised:
        design.design(str(path), "timestamp")
    assert raised.value.code == "TARGET_IS_THE_TIME_COLUMN"


def test_a_target_that_is_not_a_number_is_refused_before_any_test_runs(tmp_path):
    path = tmp_path / "text.csv"
    rows = "\n".join(f"2020-01-01T00:{i // 60:02d}:{i % 60:02d},up" for i in range(200))
    path.write_text("timestamp,value\n" + rows + "\n", encoding="utf-8")
    with pytest.raises(design.DesignRefusal) as raised:
        design.design(str(path), "value")
    assert raised.value.code == "TARGET_NOT_NUMERIC" and "'up'" in raised.value.why


def test_every_emitted_candidate_is_a_representation_stage_one_reads(tmp_path):
    for values in (periodic(), random_walk()):
        document = run(tmp_path, values)
        assert document["candidates"]
        for candidate in document["candidates"]:
            assert validate_spec(json.loads(json.dumps(candidate))) is not None
            assert candidate["schema"] == SPEC_SCHEMA
            assert candidate["provenance"] == "DEVELOPMENT"


def test_what_the_data_cannot_decide_is_listed_in_every_candidate(tmp_path):
    document = run(tmp_path, periodic())
    for candidate in document["candidates"]:
        assert candidate["calendar"]["clock"] == "receipt"
        assert "calendar.clock" in candidate["not_decided"]
        assert "holdout" in candidate["not_decided"] and candidate["holdout"] == {"fraction": 0.2}
        assert "sampling.timezone" in candidate["not_decided"]


def test_a_declared_clock_holdout_and_zone_leave_the_not_decided_block(tmp_path):
    document = run(tmp_path, periodic(), clock="publication", holdout={"cut": "2020-01-02T00:00:00+00:00"},
                   timezone_name="Europe/Madrid")
    for candidate in document["candidates"]:
        assert candidate["calendar"]["clock"] == "publication"
        assert candidate["holdout"] == {"cut": "2020-01-02T00:00:00+00:00"}
        assert candidate["sampling"]["timezone"] == "Europe/Madrid"
        assert set(candidate["not_decided"]) == {"exogenous"}


def test_the_sampling_step_is_read_from_the_timestamps_and_gaps_are_counted(tmp_path):
    path = write_csv(tmp_path / "step.csv", periodic(400), step_seconds=300)
    document = design.design(str(path), "value")
    sampling = document["tests"]["sampling"]
    assert sampling["step_seconds"] == 300 and sampling["regular_fraction"] == 1.0
    assert sampling["gaps_longer_than_step"] == 0 and sampling["backwards_steps"] == 0
    assert all(c["sampling"]["step_seconds"] == 300 for c in document["candidates"])


def test_missing_cells_are_counted_and_the_tests_read_the_longest_whole_run(tmp_path):
    values = list(periodic(600))
    values[100] = None                                # an empty cell, which is a declared missing token
    path = write_csv(tmp_path / "gappy.csv", values)
    document = design.design(str(path), "value")
    assert document["tests"]["missingness"]["per_column"]["value"]["missing"] == 1
    segment = document["tests"]["analysis_segment"]
    assert segment["start_row"] == 101 and segment["rows"] == 499


def test_a_daily_period_motivates_the_calendar_feature_feature_eng_builds(tmp_path):
    minutes_per_day = 1440
    values = np.sin(2 * np.pi * np.arange(6 * minutes_per_day) / minutes_per_day)
    path = write_csv(tmp_path / "daily.csv", values, step_seconds=60)
    document = design.design(str(path), "value")
    daily = [c for c in document["candidates"] if minutes_per_day in c["windows"]]
    assert daily, [c["windows"] for c in document["candidates"]]
    assert daily[0]["features"] == ["hour_of_day"] == daily[0]["calendar"]["columns"]
    assert "86400 s" in daily[0]["why"]["features"]


def test_without_statsmodels_the_tests_say_not_available_and_the_job_still_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(design, "_statsmodels", lambda: None)
    document = run(tmp_path, random_walk())
    stationarity = document["tests"]["stationarity"]
    assert stationarity["adf"]["status"] == "NOT_AVAILABLE" and "statsmodels" in stationarity["adf"]["reason"]
    assert stationarity["kpss"]["status"] == "NOT_AVAILABLE"
    assert stationarity["verdict_source"] == "heuristic" and stationarity["verdict"] == "NON_STATIONARY"
    assert document["environment"]["statsmodels"] == "NOT_AVAILABLE"
    assert [c for c in document["candidates"] if c["target"]["transform"] == "diff"]


def test_the_same_file_designs_to_the_same_bytes(tmp_path):
    path = write_csv(tmp_path / "series.csv", periodic())
    first = design.design(str(path), "value")
    second = design.design(str(path), "value")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["dataset"]["sha256"] == second["dataset"]["sha256"]
    assert first["fitted"].startswith("NOTHING")


def test_the_cli_writes_the_candidates_and_refuses_with_a_code(tmp_path, capsys):
    path = write_csv(tmp_path / "series.csv", periodic())
    out = tmp_path / "candidates.json"
    assert design.main(["--data", str(path), "--target", "value", "--out", str(out)]) == 0
    document = json.loads(out.read_text())
    assert document["schema"] == "m5phet.representation_design.v1" and document["candidates"]
    assert design.main(["--data", str(path), "--target", "nope", "--out", str(out)]) == 2
    assert "REFUSED TARGET_NOT_IN_DATASET" in capsys.readouterr().err
