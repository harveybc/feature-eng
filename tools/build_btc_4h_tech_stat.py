#!/usr/bin/env python3
"""Coordinator-approved bounded read: BTCUSDT 4h bars + Stage 2.2 technical/statistical files -> one CSV with
the ETH view's 83-feature recipe (technical columns + statistical columns, statistical log_return_1 renamed
statistical__log_return_1), joined on the bar timestamp. Shas verified against census / census extension.
No warm-up row is dropped (the profile reports it). The lake is not modified. Development class."""
import argparse, hashlib, json
from pathlib import Path
import pyarrow.parquet as pq, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--lake-root", required=True); ap.add_argument("--feature-order-manifest", required=True); ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
root = Path(a.lake_root)
SRC = {"bars": ("features/trading_asset_data/btcusdt/4h.parquet", "83eaaefccabd08fcce968891a9391cc9fdda8e8404d63816119ca21aa5358758"),
       "technical": ("features/trading_asset_features/btcusdt/4h/technical.parquet", "37a569c668f32f623e46bd6818a188d4a46ffff1e7a8880934a015ab13faee66"),
       "statistical": ("features/trading_asset_features/btcusdt/4h/statistical.parquet", "136ba338bcd1560eeced086ba383505c48e3549d67ef95c8fb95e45a512c3322")}
for k, (rel, sha) in SRC.items():
    h = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if h != sha:
        raise SystemExit(f"sha mismatch for {rel}: {h}")
order = json.load(open(a.feature_order_manifest))["features"]
bars = pq.read_table(root / SRC["bars"][0], columns=["timestamp", "open", "high", "low", "close", "volume"]).to_pandas()
tech = pq.read_table(root / SRC["technical"][0]).to_pandas()
stat = pq.read_table(root / SRC["statistical"][0]).to_pandas().rename(columns={"log_return_1": "statistical__log_return_1"})
for d in (bars, tech, stat):
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
df = bars.merge(tech, on="timestamp", how="left", validate="one_to_one").merge(stat, on="timestamp", how="left", validate="one_to_one")
missing = [c for c in order if c not in df.columns]
if missing:
    raise SystemExit(f"recipe columns absent: {missing}")
out = df[["timestamp", "open", "high", "low", "close", "volume"] + order].rename(columns={"timestamp": "DATE_TIME", "open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "volume": "VOLUME"})
od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=False)
p = od / "btcusdt_4h_tech_stat.csv"
out.assign(DATE_TIME=out["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(p, index=False)
prov = {"schema": "lane_b_derivative_view.v1", "availability_class": "DEVELOPMENT", "source_state": "BOUNDED_AT_FILE_GRAIN",
        "sources": {k: {"lake_relative_path": v[0], "sha256": v[1]} for k, v in SRC.items()},
        "raw_parent": "market_data/crypto/spot_top50/btcusdt/4h.parquet (Binance Spot)",
        "producer_of_features": "financial-data stage22_trading_features_worker.py (technical/statistical; NOT temporally verified)",
        "transform": {"join": "left join of technical and statistical on the bar timestamp; statistical log_return_1 renamed statistical__log_return_1",
                      "feature_order": "the ETH view's 83-feature order (eth_4h v1 manifest)", "warm_up": "kept; NaN rows reported by the profile"},
        "output": {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": int(len(out)), "first": str(out["DATE_TIME"].iloc[0]), "last": str(out["DATE_TIME"].iloc[-1])},
        "approval": "coordinator, 2026-10-01: one bounded read per source, <=256M, ionice -c3, <=10 min; lake not modified"}
(od / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n")
print(json.dumps(prov["output"]))
