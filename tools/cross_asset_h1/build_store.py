"""Build the EURUSD H1 cross-asset CANDIDATE store (outside the lake and Git) on the admitted UTC hourly grid.

    PYTHONPATH=. python tools/cross_asset_h1/build_store.py --config examples/config/cross_asset_eurusd_h1.json

Every source is pinned by sha256 and read with the TEST boundary enforced at read (no row at or after 2025-01-01
survives the read; the grid ends 2024-12-31). Checks written to MANIFEST.json: the HistData clock gate (EURUSD
re-localized from the lake clock vs the admitted EURUSD, weekly: lag 0 must win in >= 97 % of weeks, median corr >= 0.98), and a real-data FUTURE PERTURBATION check:
every source row available after a cut is replaced by noise and the store is rebuilt; every grid value whose
decision time is <= the cut must be bit-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from app.plugins import cross_asset_h1 as CA  # noqa: E402

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pinned(root: Path, spec: dict) -> Path:
    p = root / spec["file"]
    got = sha256(p)
    if got != spec["sha256"]:
        raise SystemExit(f"SOURCE_PIN: {spec['file']} {got[:12]} != {spec['sha256'][:12]}")
    return p


def _before_test(df: pd.DataFrame, t: pd.DatetimeIndex) -> pd.DataFrame:
    keep = np.asarray(t < TEST_START)
    out = df.loc[keep]
    return out


def load_sources(cfg: dict) -> dict:
    root = Path(cfg["financial_data_root"])
    src = {"fx": {}, "daily": {}, "pins": {}}
    for pair, spec in cfg["fx"].items():
        p = pinned(root, spec)
        raw = pd.read_parquet(p)
        t = pd.DatetimeIndex(pd.to_datetime(raw["datetime"]))
        raw = _before_test(raw, t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC"))
        src["fx"][pair] = CA.relocalize_lake_clock(raw, "datetime")
        src["fx"][pair] = src["fx"][pair][src["fx"][pair].index < TEST_START]
        src["pins"][spec["file"]] = spec["sha256"]
    b = cfg["btc"]
    p = pinned(root, b)
    raw = pd.read_parquet(p)
    t = pd.DatetimeIndex(pd.to_datetime(raw["open_time"]))
    t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
    raw = _before_test(raw, t)
    raw.index = t[t < TEST_START]
    src["btc"] = raw.sort_index()
    src["pins"][b["file"]] = b["sha256"]
    for name, spec in cfg["daily"].items():
        p = pinned(root, spec)
        raw = pd.read_parquet(p)
        dcol, vcol = spec["date_column"], spec["value_column"]
        d = pd.DatetimeIndex(pd.to_datetime(raw[dcol]))
        d = d.tz_localize(None) if d.tz is not None else d                       # local trading date
        v = pd.to_numeric(raw[vcol], errors="coerce")
        s = pd.Series(v.to_numpy(float), index=d).sort_index()
        s = s[s.index < TEST_START.tz_localize(None)]
        src["daily"][name] = (s[~s.index.duplicated(keep="last")], spec["kind"])
        src["pins"][spec["file"]] = spec["sha256"]
    c = cfg["cot"]
    p = pinned(root, c)
    raw = pd.read_csv(p)
    raw = raw[pd.to_datetime(raw["Date"]) < TEST_START.tz_localize(None)]
    src["cot"] = raw
    src["pins"][c["file"]] = c["sha256"]
    return src


def admitted_grid(cfg: dict):
    a = cfg["admitted"]
    if sha256(a["file"]) != a["sha256"]:
        raise SystemExit("ADMITTED_PIN")
    raw = pd.read_parquet(a["file"], columns=[a["time_column"], a["close_column"]])
    t = pd.DatetimeIndex(pd.to_datetime(raw[a["time_column"]]))
    t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
    keep = (t >= pd.Timestamp(cfg["grid_start"], tz="UTC")) & (t < TEST_START)
    close = pd.Series(raw[a["close_column"]].to_numpy(float)[keep], index=t[keep]).sort_index()
    return close.index, close


def build(cfg: dict, src: dict, grid) -> pd.DataFrame:
    pl = CA.Plugin()
    pl.set_params(**cfg.get("params", {}))
    return pl.build(grid, fx=src["fx"], btc=src["btc"], daily=src["daily"], cot=src["cot"])


def perturb_after(src: dict, cut: pd.Timestamp, seed: int = 7) -> dict:
    """Replace every source row AVAILABLE after ``cut`` by noise (same shape); availability rules as the plugin."""
    rng = np.random.default_rng(seed)
    out = {"fx": {}, "daily": {}}
    for k, df in src["fx"].items():
        d = df.copy()
        m = (d.index + CA.HOUR) > cut
        d.loc[m, "close"] = d["close"][m] * np.exp(rng.normal(0, 0.01, int(m.sum())))
        out["fx"][k] = d
    b = src["btc"].copy()
    m = (b.index + CA.HOUR) > cut
    for c in ("close", "volume", "taker_buy_base_volume"):
        b.loc[m, c] = b[c][m] * np.exp(rng.normal(0, 0.05, int(m.sum())))
    out["btc"] = b
    for name, (s, kind) in src["daily"].items():
        s2 = s.copy()
        m = np.asarray(CA.daily_available(s2.index) > cut)
        s2[m] = s2[m] * (1 + rng.normal(0, 0.1, int(m.sum())))
        out["daily"][name] = (s2, kind)
    c = src["cot"].copy()
    m = np.asarray(CA.cot_available(c["Date"]) > cut)
    c.loc[m, "EUR_Net_Long"] = c["EUR_Net_Long"][m] + rng.integers(-50000, 50000, int(m.sum()))
    out["cot"] = c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    grid, adm_close = admitted_grid(cfg)
    src = load_sources(cfg)
    # clock gate on the HistData EURUSD (re-localized exactly as the other pairs)
    root = Path(cfg["financial_data_root"])
    g = cfg["clock_gate"]
    raw = pd.read_parquet(pinned(root, g))
    t = pd.DatetimeIndex(pd.to_datetime(raw["datetime"]))
    raw = _before_test(raw, t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC"))
    eu = CA.relocalize_lake_clock(raw, "datetime")["close"]
    win = (eu.index >= pd.Timestamp(g["window"][0], tz="UTC")) & (eu.index < pd.Timestamp(g["window"][1], tz="UTC"))
    aw = (adm_close.index >= pd.Timestamp(g["window"][0], tz="UTC")) & (adm_close.index < pd.Timestamp(g["window"][1], tz="UTC"))
    pp = cfg.get("params", {})
    gate = CA.clock_gate(eu[win], adm_close[aw], min_share_lag0=float(pp.get("min_share_lag0", 0.97)),
                         min_median_corr=float(pp.get("min_median_corr", 0.98)))
    X = build(cfg, src, grid)
    # future perturbation on real data
    checks = []
    for cut_s in cfg["perturbation_cuts"]:
        cut = pd.Timestamp(cut_s, tz="UTC")
        Xp = build(cfg, perturb_after(src, cut), grid)
        dec = grid + CA.HOUR
        before = np.asarray(dec <= cut)
        A, B = X.to_numpy(float)[before], Xp.to_numpy(float)[before]
        same = (A == B) | (np.isnan(A) & np.isnan(B))
        after = ~before
        changed_after = int((~((X.to_numpy(float)[after] == Xp.to_numpy(float)[after]) |
                               (np.isnan(X.to_numpy(float)[after]) & np.isnan(Xp.to_numpy(float)[after])))).sum())
        checks.append({"cut": str(cut), "rows_at_or_before": int(before.sum()), "cells_changed_at_or_before": int((~same).sum()),
                       "cells_changed_after": changed_after})
    ok = all(c["cells_changed_at_or_before"] == 0 for c in checks) and all(c["cells_changed_after"] > 0 for c in checks)
    if not ok:
        raise SystemExit(f"PERTURBATION_FAILED: {checks}")
    store = X.copy()
    store.insert(0, "timestamp", grid)
    store = store.reset_index(drop=True)
    f = out_dir / "cross_asset_eurusd_h1_2020_2024.parquet"
    store.to_parquet(f, index=False)
    cov = {str(y): {c: float(X.loc[X.index.year == y, c].notna().mean()) for c in X.columns} for y in sorted(set(X.index.year))}
    manifest = {"schema": "feature_eng.cross_asset_h1.store.v1", "plugin": "cross_asset_h1", "instrument": "EURUSD", "bar": "1h",
                "grid": {"source": "admitted", "source_sha256": cfg["admitted"]["sha256"], "time": "UTC bar start",
                         "decision": "bar start + 1 h", "span": [str(grid[0]), str(grid[-1])], "rows": int(len(grid))},
                "source_sha256": cfg["admitted"]["sha256"], "test_read": False, "test_start_closed": str(TEST_START),
                "file": f.name, "file_sha256": sha256(f), "columns": list(X.columns), "features": CA.feature_meta(X.columns),
                "source_pins": src["pins"], "params": CA.Plugin().plugin_params | cfg.get("params", {}),
                "clock_gate": gate, "future_perturbation": checks, "coverage_by_year": cov,
                "excluded": {"STLFSI4": "revised series; point-in-time leak already audited (I6N)",
                             "FRED DTWEX* / macro composites": "computed indices with revisions; not point-in-time",
                             "economic calendar 2020-2024 with consensus": "not available point-in-time (Investing.com export forbidden by its terms; FXMacroData only from 2024-12)"}}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    if cfg.get("manifest_copy"):
        mc = REPO / cfg["manifest_copy"]
        mc.parent.mkdir(parents=True, exist_ok=True)
        mc.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"rows": len(store), "columns": len(X.columns), "clock_gate": {k: gate[k] for k in ("weeks", "share_best_lag0", "median_corr_lag0")}, "perturbation": checks,
                      "file_sha256": manifest["file_sha256"][:16]}))


if __name__ == "__main__":
    main()
