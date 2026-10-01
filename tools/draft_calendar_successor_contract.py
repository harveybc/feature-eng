#!/usr/bin/env python3
"""Draft a successor dataset contract that carries a task's CALENDAR split (prepared, never activated).

The C127 contract stays untouched. Boundaries are computed from the timestamp column only
(no other value is read) as the first row at or after each calendar cut, then sealed with the
same predictor tools/df_contract.py the C127 contracts used, so every structural rule applies.
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--contracts", required=True)
ap.add_argument("--dataset-id", required=True)
ap.add_argument("--lake-root", required=True)
ap.add_argument("--df-contract", required=True, help="predictor tools/df_contract.py at the C127 module version")
ap.add_argument("--cuts", required=True, help='JSON {"calibration":"2024-01-01T00:00:00Z","confirmation":"2025-01-01T00:00:00Z"}')
ap.add_argument("--task-rule", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()
raw = Path(a.contracts).read_bytes()
base = next(c for c in json.loads(raw)["contracts"] if c["dataset_id"] == a.dataset_id)
mod_bytes = Path(a.df_contract).read_bytes()
spec = importlib.util.spec_from_file_location("dfc", a.df_contract); dfc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dfc)
f = base["files"][0]
path = Path(a.lake_root) / f["name"]
h = hashlib.sha256(path.read_bytes()).hexdigest()
if h != f["sha256"]:
    raise SystemExit(f"bytes differ from the contract: {h} != {f['sha256']}")
import pyarrow.parquet as pq
import pandas as pd
ts = pd.to_datetime(pq.read_table(path, columns=["timestamp"]).column("timestamp").to_pandas(), utc=True)
n = len(ts)
cuts = json.loads(a.cuts)
b1 = int((ts < pd.Timestamp(cuts["calibration"])).sum())
b2 = int((ts < pd.Timestamp(cuts["confirmation"])).sum())
if not (ts.is_monotonic_increasing and 0 < b1 < b2 < n):
    raise SystemExit("timestamps not monotonic or a block would be empty")
doc = json.loads(json.dumps(base))
doc["version"] = f"{base.get('version', '1')}+calendar_successor_draft"
doc["partitions"] = {"scheme": "calendar_cut_on_timestamp_column",
                     "fractions": {"train": b1 / n, "calibration": (b2 - b1) / n, "confirmation": (n - b2) / n},
                     "boundaries": {"train": [0, b1], "calibration": [b1, b2], "confirmation": [b2, n]},
                     "frozen_before_profile": True, "sealed_periods_excluded": []}
sealed = dfc.seal(doc)
out = {"schema": "lane_b_successor_contract_draft.v1", "status": "PREPARED_NOT_ACTIVATED",
       "supersedes_for_task_only": {"c127_contract_sha256": base["contract_sha256"], "c127_contracts_file_sha256": hashlib.sha256(raw).hexdigest(),
                                    "kept": "C127 is unchanged; this draft is a separate object and the two are never mixed"},
       "task_rule": a.task_rule, "cuts": cuts,
       "timestamps": {"rows": n, "first": str(ts.iloc[0]), "last": str(ts.iloc[-1]), "train_last": str(ts.iloc[b1 - 1]),
                      "calibration_first": str(ts.iloc[b1]), "confirmation_first": str(ts.iloc[b2])},
       "df_contract_module_sha256": hashlib.sha256(mod_bytes).hexdigest(), "file_bytes_verified_sha256": h,
       "contract": sealed,
       "not_done": ["not written to financial-data census or any lake/data-gov configuration",
                    "availability stays UNKNOWN: a split contract is not availability (owner question 19)"],
       "activation_path": "owner: review, then add as a NEW contract file beside C127 and register; no edit of C127"}
Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"contract_sha256": sealed["contract_sha256"], **out["timestamps"], "boundaries": sealed["partitions"]["boundaries"]}))
