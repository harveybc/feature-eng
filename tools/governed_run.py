#!/usr/bin/env python3
"""Governed feature-eng run (data-gov Flow v3).

Declares what this repository consumes and produces; the protocol itself lives
in data-gov (`tools/governed_exec.py`): campaign before data, governed download
with hash confirmation, fresh output namespace, command on CPU, terminal
COMPLETED | FAILED | INCONCLUSIVE | REFUSED through a durable outbox, then
reconciliation. A result that is not reconciled is not governing.

Every dataset key of the config that names a file (`input_file`,
`high_freq_dataset`, `sp500_dataset`, `vix_dataset`, `economic_calendar`) is a
governed input; each must be a resource of the given lake. The command runs
with the output directory as working directory because the default plugin
also writes fixed-name CSVs into the current directory. Metrics: row count of
`output_file`.

The data-gov checkout is found through DATA_GOV_CHECKOUT or the sibling
directory `../data-gov`.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _governed_exec():
    checkout = Path(os.environ.get("DATA_GOV_CHECKOUT") or REPO_ROOT.parent / "data-gov").expanduser()
    path = checkout / "tools" / "governed_exec.py"
    if not path.is_file():
        raise SystemExit(f"governed_run: data-gov checkout not found at {checkout} (set DATA_GOV_CHECKOUT)")
    spec = importlib.util.spec_from_file_location("data_gov_governed_exec", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_gov_governed_exec"] = module
    spec.loader.exec_module(module)
    return module


def _metrics(config: dict) -> dict:
    output = Path(str(config.get("output_file") or "indicators_output.csv")).name
    return {"kind": "row_counts", "files": [{"path": output, "split": "output", "header": bool(config.get("headers", True))}]}


PROFILE = {
    "project": "feature-eng",
    "input_keys": ["input_file", "high_freq_dataset", "sp500_dataset", "vix_dataset", "economic_calendar"],
    "output_keys": ["output_file", "save_log", "save_config"],
    "command": [sys.executable, "{repo_root}/app/main.py", "--load_config", "{config}"],
    "cwd": "{out_dir}",
    "metrics": _metrics,
    "artifacts": {"output": "output_file", "debug_log": "save_log", "effective_config": "save_config"},
    "tags": {"plugin": "tech_indicator"},
}


def main(argv=None) -> int:
    return _governed_exec().consumer_main(PROFILE, argv, repo_root=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
