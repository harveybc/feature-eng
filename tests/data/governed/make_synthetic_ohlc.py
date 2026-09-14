#!/usr/bin/env python3
"""Synthetic hourly OHLC bars for governed integration tests of feature-eng.

Provenance is this file: a seeded random walk (numpy PCG64, seed 20260914),
hourly bars over ISO wall-clock timestamps labelled by the bar's OPEN time
(`WINDOW_START`), complete one hour later. The series is synthetic: it proves
mechanics (governed download, fresh outputs, terminal, lineage), never public
or financial utility, and its timestamps are not publication times of any
venue. Columns follow the `forex_15m` header mapping of `app/config.py`
(`datetime,open,high,low,close`) so the default pipeline reads them unchanged.

    python tests/data/governed/make_synthetic_ohlc.py   # rewrites the two CSVs deterministically
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20260914
START = "2013-01-01 00:00:00"
FILES = {"synthetic_ohlc_1h_a.csv": (SEED, 24 * 90), "synthetic_ohlc_1h_b.csv": (SEED + 1, 24 * 90)}


def bars(seed: int, n: int):
    rng = np.random.Generator(np.random.PCG64(seed))
    close = 1.30 + np.cumsum(rng.normal(0.0, 0.0008, size=n))
    open_ = np.concatenate([[1.30], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.0006, size=n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return open_, high, low, close


def write(name: str, seed: int, n: int) -> dict:
    import datetime as dt

    t0 = dt.datetime.strptime(START, "%Y-%m-%d %H:%M:%S")
    o, h, l, c = bars(seed, n)
    lines = ["datetime,open,high,low,close"]
    for i in range(n):
        stamp = (t0 + dt.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{stamp},{o[i]:.5f},{h[i]:.5f},{l[i]:.5f},{c[i]:.5f}")
    raw = ("\n".join(lines) + "\n").encode("ascii")
    (HERE / name).write_bytes(raw)
    return {"file": name, "seed": seed, "rows": n, "sha256": hashlib.sha256(raw).hexdigest(),
            "label": "WINDOW_START", "completion_lag_max": "1h", "frequency": "1h",
            "timezone": "NAIVE_WALL_CLOCK (synthetic)", "generator": HERE.name + "/" + Path(__file__).name}


def main() -> int:
    manifest = {"schema": "feature_eng_synthetic_ohlc_fixtures.v1", "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": [write(name, seed, n) for name, (seed, n) in FILES.items()]}
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
