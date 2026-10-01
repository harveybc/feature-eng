#!/usr/bin/env python3
"""Coordinator-approved bounded read (2026-10-01): EURUSD 5m lake appearance -> 1h OHLC derivative view.

Reads only the declared 5m file (columns timestamp/open/high/low/close), verifies its sha256 against the
census record, resamples to 1h with closed='left', label='right' (a bar is stamped when it is COMPLETE),
and writes a CSV plus a provenance record. The lake is not modified. Development class only."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--lake-root", required=True); ap.add_argument("--rel", required=True); ap.add_argument("--expect-sha", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
src = Path(a.lake_root) / a.rel
h = hashlib.sha256()
with open(src, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
if h.hexdigest() != a.expect_sha:
    raise SystemExit(f"sha mismatch {h.hexdigest()}")
t = pq.read_table(src, columns=["timestamp", "open", "high", "low", "close"])
df = t.to_pandas(); del t
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp").sort_index()
r = df.resample("1h", closed="left", label="right")
out = pd.DataFrame({"OPEN": r["open"].first(), "HIGH": r["high"].max(), "LOW": r["low"].min(), "CLOSE": r["close"].last(),
                    "N_5M_BARS": r["close"].count()})
out = out[out["N_5M_BARS"] > 0]
out.index.name = "DATE_TIME"
od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=False)
csvp = od / "eurusd_1h_from_lake_5m.csv"
out.reset_index().assign(DATE_TIME=lambda d: d["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(csvp, index=False)
csv_sha = hashlib.sha256(csvp.read_bytes()).hexdigest()
prov = {"schema": "lane_b_derivative_view.v1", "availability_class": "DEVELOPMENT", "source_state": "BOUNDED_AT_FILE_GRAIN",
        "source": {"lake_relative_path": a.rel, "sha256": a.expect_sha, "census_appearance": "app_ae9142c201e6694a3b1e1fde",
                   "raw_parent": "market_data/forex/g10/eurusd/5m.parquet (HistData; raw provenance sha d527b46a…)", "rows_5m": int(len(df))},
        "transform": {"resample": "1h, closed=left, label=right (stamp = bar completion)", "ohlc": "first/max/min/last of 5m bars", "n_5m_bars_column": "N_5M_BARS",
                      "timezone": "as stored (UTC-parsed; source timezone undocumented)"},
        "output": {"file": csvp.name, "sha256": csv_sha, "rows": int(len(out)), "first": str(out.index[0]), "last": str(out.index[-1])},
        "approval": "coordinator, 2026-10-01: one bounded read, <=256M, ionice -c3, <=10 min; lake not modified"}
(od / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps(prov["output"]))
