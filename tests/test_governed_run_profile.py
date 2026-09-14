"""The governed wrapper declares this repository's inputs, outputs and metrics."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_GOV = Path(os.environ.get("DATA_GOV_CHECKOUT") or ROOT.parent / "data-gov")

pytestmark = pytest.mark.skipif(
    not (DATA_GOV / "tools" / "governed_exec.py").is_file(), reason="data-gov checkout not available"
)


def _load():
    spec = importlib.util.spec_from_file_location("feature_eng_governed_run", ROOT / "tools" / "governed_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["feature_eng_governed_run"] = module
    spec.loader.exec_module(module)
    return module


def test_profile_builds_a_valid_spec(tmp_path, capsys):
    lake = tmp_path / "lake"
    (lake / "fx").mkdir(parents=True)
    (lake / "fx" / "eurusd_4h.csv").write_text("Gmt time,open,high,low,close,volume\n")
    (lake / "vix.csv").write_text("date,close\n")
    config = tmp_path / "phase_y" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({
        "input_file": str(lake / "fx" / "eurusd_4h.csv"), "vix_dataset": str(lake / "vix.csv"),
        "sp500_dataset": None, "output_file": "./indicators_output.csv", "save_log": "./debug_log.json",
        "save_config": "./output_config.json", "plugin": "tech_indicator", "headers": True,
    }))
    module = _load()
    assert module.main([
        "--load_config", str(config), "--experiment-key", "feng-001", "--lake", "financial_files",
        "--lake-root", str(lake), "--out-dir", str(tmp_path / "out"), "--print-spec",
    ]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["project"] == "feature-eng" and spec["cwd"] == "{out_dir}"
    assert [(d["resource"], d["role"]) for d in spec["datasets"]] == [("fx/eurusd_4h.csv", "input_file"), ("vix.csv", "vix_dataset")]
    assert spec["command"][1:] == ["{repo_root}/app/main.py", "--load_config", "{config}"]
    assert spec["metrics"] == {"kind": "row_counts", "files": [{"path": "indicators_output.csv", "split": "output", "header": True}]}
    assert spec["output_keys"] == ["output_file", "save_log", "save_config"]
    # an input outside the lake root is refused before anything is registered
    config.write_text(json.dumps({"input_file": str(tmp_path / "elsewhere.csv"), "output_file": "o.csv"}))
    assert module.main([
        "--load_config", str(config), "--experiment-key", "feng-002", "--lake", "financial_files",
        "--lake-root", str(lake), "--out-dir", str(tmp_path / "out2"), "--print-spec",
    ]) == 1
    assert "not under the lake root" in capsys.readouterr().err
