#!/usr/bin/env python3
"""Coordinator-approved bounded read of one source family -> a daily panel (one column per series), written to a
state directory and moved to worker_b. Only daily series are kept (median date step <= 3 days); others are named
as excluded. Each file's sha256 and row count are recorded; nothing in the lake is modified."""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
ap = argparse.ArgumentParser()
ap.add_argument("--lake-root", required=True); ap.add_argument("--family", required=True, choices=["fred_daily", "yahoo_daily"])
ap.add_argument("--files", nargs="+", required=True); ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
root = Path(a.lake_root); cols, prov, excluded = {}, [], []
for rel in a.files:
    p = root / rel; b = p.read_bytes(); sha = hashlib.sha256(b).hexdigest()
    if a.family == "fred_daily":
        t = pq.read_table(p, columns=["date", "value", "realtime_start", "realtime_end"]).to_pandas()
        s = pd.to_numeric(t["value"], errors="coerce"); s.index = pd.to_datetime(t["date"]); rs = sorted(t["realtime_start"].astype(str).unique())[:3]
        name = rel.split("/")[-2]; meta = {"realtime_start_values": rs, "realtime_start_distinct": int(t["realtime_start"].nunique())}
    else:
        t = pq.read_table(p, columns=["Date", "Close"]).to_pandas()
        s = pd.to_numeric(t["Close"], errors="coerce"); s.index = pd.to_datetime(t["Date"], utc=True).dt.tz_localize(None).dt.normalize()
        name = rel.split("/")[-2]; meta = {}
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    step = s.index.to_series().diff().dt.days.median()
    rec = {"path": rel, "sha256": sha, "rows": int(len(s)), "first": str(s.index[0].date()) if len(s) else None, "last": str(s.index[-1].date()) if len(s) else None,
           "median_step_days": float(step) if step == step else None, **meta}
    if len(s) and step <= 3:
        cols[f"{a.family.split('_')[0]}__{name}"] = s; rec["kept"] = True
    else:
        rec["kept"] = False; excluded.append({"series": name, "reason": f"NOT_DAILY (median step {step} days): publication and revision timing not established for this slice"})
    prov.append(rec)
panel = pd.DataFrame(cols).sort_index(); panel.index.name = "DATE"
od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=False)
f = od / f"{a.family}_panel.csv"; panel.reset_index().assign(DATE=lambda d: d["DATE"].dt.strftime("%Y-%m-%d")).to_csv(f, index=False)
doc = {"schema": "lane_b_daily_panel.v1", "family": a.family, "availability_class": "DEVELOPMENT", "source_state": "BOUNDED_AT_FILE_GRAIN",
       "files": prov, "excluded": excluded, "output": {"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "columns": list(panel.columns), "rows": int(len(panel))},
       "availability_rule": ("FRED daily: value dated d treated as usable only from day d+1 (next-business-day publication); realtime_start recorded" if a.family == "fred_daily"
                             else "Yahoo daily close dated d treated as usable only from day d+1 (market close timezone undocumented)"),
       "approval": "coordinator standing approval: one bounded read per source family, <=256M, ionice -c3, <=10 min; lake not modified"}
(od / "PROVENANCE.json").write_text(json.dumps(doc, indent=1) + "\n")
print(json.dumps({"family": a.family, "kept": len(cols), "excluded": len(excluded), "rows": len(panel), "sha": doc["output"]["sha256"]}))
