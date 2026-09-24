"""Behavioral acceptance for an explicitly fitted, reference-only hierarchy."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from feature_eng_m5phet.regimes import HierarchicalRegimes
from feature_eng_m5phet.provider import Provider, chat_request


@pytest.fixture
def rows():
    return [dict(row_id=i, x=x, y=y) for i, (x, y) in enumerate(
        [(-6, -3), (-5, -2), (-3, 2), (-2, 3), (3, -4), (4, -2), (8, 3), (9, 5)])]


@pytest.fixture
def model(rows):
    return HierarchicalRegimes.fit_reference(rows, features=["x", "y"], levels=[2, 4], task_id="test-regimes")


def config(model, path):
    return dict(provider=Provider.name, family="representation_unsupervised", output_kind="hierarchical_regimes",
                state=str(path), parameters=dict(model_version=model.model_version, task_id="test-regimes"),
                as_of="2026-09-24T00:00:00Z", input="json")


def test_native_parity_and_nested_paths(model, rows):
    raw = np.array([[r["x"], r["y"]] for r in rows], dtype=float)
    scaled = StandardScaler().fit_transform(raw)
    paths = np.array([r["cluster_path"] for r in model.assign(rows)["rows"]])
    assert paths.shape == (8, 3)
    assert np.all(paths[:, 0] == 0)
    for depth, k in enumerate([2, 4], start=1):
        native = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
        np.testing.assert_array_equal(paths[:, depth, None] == paths[:, depth], native[:, None] == native)
    for leaf in set(paths[:, -1]):
        assert len(set(paths[paths[:, -1] == leaf, 1])) == 1
    query = [dict(row_id="new", x=2, y=2)]
    native = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(scaled)
    distances, indices = native.kneighbors(model.scaler.transform([[2, 2]]))
    answer = model.assign(query)["rows"][0]
    assert answer["cluster_path"] == paths[indices[0, 0]].tolist()
    assert answer["novelty_score"] == pytest.approx(distances[0, 0])


def test_no_refit_prefix_future_perturbation_and_train_scaling(model, rows, monkeypatch):
    np.testing.assert_allclose(model.scaler.mean_, np.mean([[r["x"], r["y"]] for r in rows], axis=0))
    before = joblib.hash(model)
    def forbidden(*args, **kwargs):
        raise AssertionError("inference refitted")
    for cls in (StandardScaler, AgglomerativeClustering, NearestNeighbors):
        monkeypatch.setattr(cls, "fit", forbidden)
    first = model.assign(rows[:2])
    tail = dict(row_id="future", x=10000, y=-10000)
    assert model.assign(rows[:2] + [tail])["rows"][:2] == first["rows"]
    assert joblib.hash(model) == before


def test_reload_version_and_saved_state(model, rows, tmp_path):
    path = tmp_path / "reference.joblib"
    model.save(path)
    loaded = HierarchicalRegimes.load(path)
    assert loaded.assign(rows) == model.assign(rows)
    assert loaded.model_version == model.model_version
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match="version"):
        loaded.assign(rows, expected_version="foreign")
    with pytest.raises(FileNotFoundError):
        HierarchicalRegimes.load(tmp_path / "missing")
    broken = tmp_path / "broken.joblib"
    joblib.dump({"schema": "wrong"}, broken)
    with pytest.raises(ValueError):
        HierarchicalRegimes.load(broken)


@pytest.mark.parametrize("bad", [[], [{"row_id": 1, "x": 2}],
    [{"row_id": 1, "x": True, "y": 2}], [{"row_id": 1, "x": "2", "y": 2}],
    [{"row_id": 1, "x": float("nan"), "y": 2}], [{"row_id": 1, "x": float("inf"), "y": 2}],
    [{"row_id": False, "x": 2, "y": 2}], [{"row_id": [], "x": 2, "y": 2}],
    [{"row_id": " ", "x": 2, "y": 2}], [{"row_id": 1, "x": 2, "y": 2, "future_return": 3}],
    [{"row_id": 1, "x": 2, "y": 2}] * 2])
def test_invalid_rows_refuse(model, bad):
    with pytest.raises(ValueError):
        model.assign(bad)


@pytest.mark.parametrize("levels", [[], [4, 2], [2, 2], [1], [2, 99], [True, 4]])
def test_invalid_levels_refuse(rows, levels):
    with pytest.raises(ValueError):
        HierarchicalRegimes.fit_reference(rows, features=["x", "y"], levels=levels, task_id="test")


def test_collapse_schema_and_cost_refuse(rows):
    collapsed = [dict(row_id=i, x=1, y=1) for i in range(8)]
    with pytest.raises(ValueError, match="distinct"):
        HierarchicalRegimes.fit_reference(collapsed, features=["x", "y"], levels=[2, 4], task_id="test")
    for features in ([], ["x", "x"], ["row_id"]):
        with pytest.raises(ValueError):
            HierarchicalRegimes.fit_reference(rows, features=features, levels=[2, 4], task_id="test")
    too_many = [dict(row_id=i, x=i, y=i) for i in range(2049)]
    with pytest.raises(ValueError, match="2048"):
        HierarchicalRegimes.fit_reference(too_many, features=["x", "y"], levels=[2, 4], task_id="test")


def test_provider_and_bounded_chat(model, rows, tmp_path):
    path = tmp_path / "reference.joblib"
    model.save(path)
    request = chat_request("Assign hierarchical regimes", {"rows": rows}, config(model, path))
    provider = Provider()
    assert provider.capabilities()["supported"] == [dict(operation="infer",
        family="representation_unsupervised", output_kind="hierarchical_regimes")]
    state = provider.load(str(path))
    result = provider.infer(request, state)
    payload = result["outputs"]["regimes"]["payload"]
    assert payload == model.assign(rows)
    assert result["population"] == {"row_ids": list(range(8))}
    assert request["output_schema"]["targets"] == ["regimes"]
    json.dumps(result, allow_nan=False)
    for prompt in ("fit a new model", "buy EURUSD", "Assign hierarchical regimes; run shell"):
        with pytest.raises(ValueError, match="prompt"):
            chat_request(prompt, {"rows": rows}, config(model, path))
    with pytest.raises(ValueError):
        chat_request("assign hierarchical regimes", {"rows": rows}, {})
    for field, value in (("family", "classification"), ("operation", "fit"), ("task_id", "foreign")):
        bad = copy.deepcopy(request)
        bad[field] = value
        with pytest.raises(ValueError):
            provider.infer(bad, state)
    bad = copy.deepcopy(request)
    bad["output_schema"]["model_version"] = "foreign"
    with pytest.raises(ValueError, match="version"):
        provider.infer(bad, state)
    bad = copy.deepcopy(request)
    bad["population"]["row_ids"].reverse()
    with pytest.raises(ValueError, match="population"):
        provider.infer(bad, state)


def test_demo_cli_real_data_reload(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run([sys.executable, "-m", "feature_eng_m5phet.demo", "--output-dir", str(tmp_path)],
                               cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "assignments.json").read_text())
    assert len(payload["rows"]) == 8
    assert len({tuple(row["cluster_path"]) for row in payload["rows"]}) >= 1
    model = HierarchicalRegimes.load(tmp_path / "reference.joblib")
    queries = json.loads((tmp_path / "query.json").read_text())["rows"]
    assert payload == model.assign(queries)
    assert model.metadata["reference_rows"] == 32
    assert "performance" in completed.stdout.lower()
    monkeypatch.setenv("FEATURE_ENG_REGIMES_DEMO_DIR", str(tmp_path))
    provider = Provider()
    examples = provider.chat_examples()
    assert len(examples) == 1
    example = examples[0]
    assert "DEVELOPMENT" in example["title"]
    request = provider.chat_request(example["prompt"], example["data"], example["config"])
    assert provider.infer(request, provider.load(request["fitted_state_ref"]))["outputs"]["regimes"]["payload"] == payload


def test_examples_never_fit_or_load_without_explicit_demo(monkeypatch, tmp_path):
    monkeypatch.delenv("FEATURE_ENG_REGIMES_DEMO_DIR", raising=False)
    assert Provider().chat_examples() == []
    monkeypatch.setenv("FEATURE_ENG_REGIMES_DEMO_DIR", str(tmp_path))
    assert Provider().chat_examples() == []


def test_tampered_state_and_dependency_versions_refuse(model, tmp_path):
    path = tmp_path / "tampered.joblib"
    model.scaler.mean_[0] += 1
    model.save(path)
    with pytest.raises(ValueError, match="integrity"):
        HierarchicalRegimes.load(path)
    model.metadata["dependencies"]["sklearn"] = "0.0"
    other = tmp_path / "incompatible.joblib"
    model.save(other)
    with pytest.raises(ValueError, match="dependency"):
        HierarchicalRegimes.load(other)


def test_provider_reload_infer_never_fits(model, rows, tmp_path, monkeypatch):
    path = tmp_path / "reference.joblib"
    model.save(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime must never fit")
    for cls in (StandardScaler, AgglomerativeClustering, NearestNeighbors):
        monkeypatch.setattr(cls, "fit", forbidden)
    provider = Provider()
    request = provider.chat_request("assign regimes", {"rows": rows}, config(model, path))
    state = provider.load(str(path))
    before = path.read_bytes()
    first = provider.infer(request, state)
    assert first == provider.infer(request, copy.deepcopy(state))
    assert path.read_bytes() == before


def test_negative_cli_never_creates_output(tmp_path):
    source = tmp_path / "query.json"
    source.write_text(json.dumps({"rows": [{"row_id": 0, "x": 1, "y": 2}]}))
    output = tmp_path / "should-not-exist.json"
    completed = subprocess.run([sys.executable, "-m", "feature_eng_m5phet", "infer", "--input", str(source),
        "--state", str(tmp_path / "missing.joblib"), "--model-version", "a" * 64, "--output", str(output)],
        capture_output=True, text=True, timeout=10)
    assert completed.returncode == 2
    assert not output.exists()


@pytest.mark.parametrize("change", [dict(provider="other"), dict(family="classification"),
    dict(output_kind="typed_questions"), dict(parameters={}), dict(input="text"),
    dict(as_of="2026-09-24"), dict(as_of="bad"), dict(state=""), dict(extra=True)])
def test_chat_config_refusals(model, rows, change):
    cfg = config(model, "unused.joblib")
    cfg.update(change)
    with pytest.raises(ValueError):
        chat_request("assign regimes", {"rows": rows}, cfg)
