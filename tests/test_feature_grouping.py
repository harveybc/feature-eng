"""The grouping job, checked against features whose blocks are known before the job runs.

Six synthetic columns: `a1, a2, a3` share one latent series, `b1, b2` share another, `c` shares none. The k = 3 cut
has exactly one right answer and it is asserted here. So are the properties that make the document usable by the
next step: the cut is a partition, the groups' summaries quote the numbers the metric sheet measured, the two
clustering backends agree, and two runs produce the same bytes.

Nothing here fits a model and nothing needs a GPU.
"""

import json

import numpy as np
import pytest

from feature_eng_m5phet import grouping, metrics
from feature_eng_m5phet.design import DesignRefusal
from tests.test_feature_metrics import PERIOD, blocks, write_csv


@pytest.fixture(scope="module")
def sheet(tmp_path_factory):
    path = write_csv(tmp_path_factory.mktemp("grouping") / "blocks.csv",
                     {name: list(values) for name, values in blocks().items()})
    return metrics.feature_metrics(path, "a1")


@pytest.fixture(scope="module")
def document(sheet):
    return grouping.group_features(sheet)


def members_of(document, k):
    return sorted(tuple(group["members"]) for group in document["cuts"][str(k)]["groups"])


def test_the_document_validates_against_its_own_reader(document):
    assert document["schema"] == grouping.SCHEMA
    assert grouping.validate(document) is document
    assert sorted(document["cuts"], key=int) == ["2", "3", "4", "5"]      # min(6, 6 - 1) = 5
    assert document["max_k"] == 5 and "min(6, features - 1)" in document["max_k_rule"]


def test_the_k3_cut_recovers_the_two_blocks_and_the_independent_feature(document):
    assert members_of(document, 3) == [("a1", "a2", "a3"), ("b1", "b2"), ("c",)]


def test_every_cut_is_a_partition_of_the_features(document):
    for k in range(2, document["max_k"] + 1):
        members = [name for group in document["cuts"][str(k)]["groups"] for name in group["members"]]
        assert sorted(members) == document["features"]
        assert len(document["cuts"][str(k)]["groups"]) == k


def test_a_group_is_tighter_inside_than_between(document):
    cut = document["cuts"]["3"]
    assert cut["within_group_mean_abs_correlation"] > cut["between_group_mean_abs_correlation"]
    tight = next(group for group in cut["groups"] if group["members"] == ["a1", "a2", "a3"])
    assert tight["within_mean_abs_correlation"] > 0.99
    assert next(group for group in cut["groups"] if group["members"] == ["c"])["within_mean_abs_correlation"] is None


def test_a_group_summary_names_its_members_its_verdict_and_its_shared_peaks(document):
    tight = next(group for group in document["cuts"]["3"]["groups"] if group["members"] == ["a1", "a2", "a3"])
    assert "a1, a2, a3" in tight["summary"]
    assert tight["dominant_stationarity"]["verdict"] in tight["summary"]
    assert tight["dominant_stationarity"]["members"] == 3
    assert PERIOD in tight["shared_acf_peaks"], tight["shared_acf_peaks"]
    assert f"lags {tight['shared_acf_peaks']}" in tight["summary"]
    alone = next(group for group in document["cuts"]["3"]["groups"] if group["members"] == ["c"])
    assert "single feature" in alone["summary"]


def test_the_recommended_k_is_the_largest_silhouette(document):
    scored = {int(k): value for k, value in document["silhouette_by_k"].items() if value is not None}
    assert document["recommended_k"] == min(scored, key=lambda k: (-scored[k], k))
    assert document["cuts"][str(document["recommended_k"])]["silhouette"] == max(scored.values())
    assert "DETERMINISTIC_RECOMMENDATION" in document["recommendation_status"]
    assert "not a confirmation" in document["recommendation_status"]


def test_the_distance_is_one_minus_the_absolute_correlation(document, sheet):
    order = document["distance"]["order"]
    matrix = document["distance"]["matrix"]
    i, j = order.index("a1"), order.index("a2")
    pearson = sheet["pairs"]["a1::a2"]["pearson"]["value"]
    assert matrix[i][j] == pytest.approx(1.0 - abs(pearson), abs=1e-6)
    assert matrix[i][i] == 0.0 and matrix[i][j] == matrix[j][i]
    assert document["distance"]["rule"].startswith("1 - |pearson|")


def test_the_numpy_fallback_produces_the_same_groups_as_scipy(sheet, monkeypatch):
    with_scipy = grouping.group_features(sheet)
    assert with_scipy["linkage"]["backend"].startswith("scipy"), "this test compares against scipy; it is installed"
    monkeypatch.setattr(grouping, "_scipy", lambda: None)
    without = grouping.group_features(sheet)
    assert without["linkage"]["backend"].startswith("numpy Lance-Williams")
    assert without["environment"]["scipy"] == "NOT_AVAILABLE"
    for k in range(2, with_scipy["max_k"] + 1):
        assert members_of(without, k) == members_of(with_scipy, k), k
    assert without["linkage"]["merges"] == with_scipy["linkage"]["merges"]
    assert without["silhouette_by_k"] == with_scipy["silhouette_by_k"]


@pytest.mark.parametrize("method", ["single", "complete", "average"])
def test_every_declared_linkage_still_recovers_the_blocks(sheet, method):
    document = grouping.group_features(sheet, linkage=method)
    assert members_of(document, 3) == [("a1", "a2", "a3"), ("b1", "b2"), ("c",)], method


def test_the_mutual_information_distance_is_available_and_declared(sheet):
    if metrics._mutual_info() is None:
        pytest.skip("scikit-learn is not installed")
    document = grouping.group_features(sheet, distance="mutual_information")
    assert document["distance"]["kind"] == "mutual_information"
    assert document["distance"]["rule"].startswith("1 - mi / max(mi)")
    assert members_of(document, 3) == [("a1", "a2", "a3"), ("b1", "b2"), ("c",)]


def test_the_mutual_information_distance_is_refused_when_the_sheet_has_none(sheet):
    crippled = json.loads(json.dumps(sheet))
    for block in crippled["pairs"].values():
        block["mutual_information"] = {"status": "NOT_AVAILABLE", "value": None, "reason": "no scikit-learn"}
    with pytest.raises(DesignRefusal) as refusal:
        grouping.group_features(crippled, distance="mutual_information")
    assert refusal.value.code == "MUTUAL_INFORMATION_NOT_AVAILABLE"


def test_a_pair_without_a_correlation_stops_the_job_by_name(sheet):
    crippled = json.loads(json.dumps(sheet))
    crippled["pairs"]["a1::c"]["pearson"] = {"status": "ZERO_VARIANCE", "value": None, "rows_used": 1000,
                                             "rule": "the linear correlation over the rows both columns have finite"}
    with pytest.raises(DesignRefusal) as refusal:
        grouping.group_features(crippled)
    assert refusal.value.code == "CORRELATION_NOT_AVAILABLE" and "'a1','c'" in refusal.value.why
    kept = grouping.group_features(crippled, exclude=["c"])
    assert kept["features"] == ["a1", "a2", "a3", "b1", "b2"]
    assert kept["excluded_features"] == {"c": "excluded by --exclude"}


def test_two_runs_produce_the_same_bytes(sheet):
    assert grouping.dumps(grouping.group_features(sheet)) == grouping.dumps(grouping.group_features(sheet))


def test_the_decision_payload_carries_the_cut_and_no_rows(document):
    payload = grouping.decision_payload(document, 3)
    assert payload["kind"] == "feature_grouping" and payload["k"] == 3
    assert list(payload) == sorted(payload)
    assert [group["members"] for group in payload["groups"]] == [list(m) for m in members_of(document, 3)]
    assert payload["recommended_k"] == document["recommended_k"]
    assert json.dumps(payload, allow_nan=False)
    with pytest.raises(DesignRefusal) as refusal:
        grouping.decision_payload(document, 99)
    assert refusal.value.code == "CUT_NOT_IN_DOCUMENT"


def test_a_broken_document_is_refused_by_name(document):
    with pytest.raises(DesignRefusal) as missing:
        grouping.validate({key: value for key, value in document.items() if key != "cuts"})
    assert missing.value.code == "MISSING_KEY"
    broken = json.loads(json.dumps(document))
    broken["cuts"]["3"]["groups"][0]["members"] = ["a1"]
    with pytest.raises(DesignRefusal) as partition:
        grouping.validate(broken)
    assert partition.value.code == "CUT_IS_NOT_A_PARTITION"


def test_a_sheet_that_is_not_a_metric_sheet_is_refused(document):
    with pytest.raises(DesignRefusal) as refusal:
        grouping.group_features({"schema": "something.else.v1"})
    assert refusal.value.code in ("MISSING_KEY", "WRONG_SCHEMA")


def test_the_cli_writes_the_groups(tmp_path, sheet, capsys):
    sheet_path = tmp_path / "feature_metrics.json"
    sheet_path.write_text(metrics.dumps(sheet), encoding="utf-8")
    out = tmp_path / "groups.json"
    assert grouping.main(["--metrics", str(sheet_path), "--out", str(out)]) == 0
    document = grouping.validate(json.loads(out.read_text(encoding="utf-8")))
    assert members_of(document, 3) == [("a1", "a2", "a3"), ("b1", "b2"), ("c",)]
    assert "the deterministic recommendation is k =" in capsys.readouterr().out
    assert grouping.main(["--metrics", str(tmp_path / "absent.json"), "--out", str(out)]) == 2
    assert "REFUSED METRIC_SHEET_UNREADABLE" in capsys.readouterr().err
