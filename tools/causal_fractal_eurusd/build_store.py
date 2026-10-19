"""Build the EURUSD H1 causal-fractal CANDIDATE feature store (outside the lake, outside Git).

    PYTHONPATH=. python tools/causal_fractal_eurusd/build_store.py chunk    --config examples/config/causal_fractal_eurusd_h1_w200.json --year 2020
    PYTHONPATH=. python tools/causal_fractal_eurusd/build_store.py assemble --config examples/config/causal_fractal_eurusd_h1_w200.json [--config ...]

``chunk`` computes one calendar year with warm-up history from the previous
December (each row uses only its trailing ``window_bars`` closes, so chunking
by year gives bit-identical values to a single pass; see the test). ``assemble``
concatenates the chunks of every configured window into one parquet per split
and writes MANIFEST.json (store copy + committed copy).

The lake is read-only and the TEST boundary is enforced at read time: no
request may reach 2025-01-01 and the loaded frame is re-checked after the
filter. There is no TEST split file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.plugins.causal_fractal import (  # noqa: E402
    CANDIDATE_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    FEATURE_COLUMNS,
    INTERMEDIATE_COLUMNS,
    Plugin,
    rs_scales_for_window,
)

SCHEMA = "causal_fractal_features.eurusd_h1.v1"
RETSU_SOURCE = {
    "path": "agent-multi/tools/causal_fractal_features.py (untracked in that checkout when ported)",
    "sha256": "703ffb5afb1d76073fdafeeb5f1e7f6ea27916f3b7777ee5e6f12ba72e09cef4",
}
DEFAULT_LAKE_ROOT = Path.home() / "Documents/GitHub/financial-data"


class TestRowsError(ValueError):
    """A request or a frame would touch TEST rows."""


def utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("plugin") != "causal_fractal":
        raise ValueError("config must select plugin 'causal_fractal'")
    return cfg


def resolve_lake(cfg: dict, lake_arg: str | None) -> Path:
    if lake_arg:
        path = Path(lake_arg)
    else:
        root = Path(os.environ.get(cfg.get("input_root_env", "FINANCIAL_DATA_ROOT"), DEFAULT_LAKE_ROOT))
        path = root / cfg["input_file"]
    if not path.exists():
        raise FileNotFoundError(f"lake file not found: {path}")
    digest = file_sha256(path)
    if digest != cfg["input_sha256"]:
        raise SystemExit(f"lake file digest {digest} != pinned {cfg['input_sha256']}")
    return path


def load_close(path: Path, start, end_exclusive, test_start) -> pd.DataFrame:
    """[start, end_exclusive) of the lake, refusing any window reaching TEST (mirrors heuristic-strategy)."""
    start, end, test_start = utc(start), utc(end_exclusive), utc(test_start)
    if end > test_start:
        raise TestRowsError(f"window end {end} reaches TEST rows (>= {test_start})")
    raw = pd.read_parquet(path, filters=[("datetime", ">=", start), ("datetime", "<", end)])
    idx = pd.DatetimeIndex(raw["datetime"])
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    frame = pd.DataFrame({"close": raw["close"].astype(float).to_numpy()}, index=idx.rename("date_time"))
    if len(frame) and frame.index.max() >= test_start:
        raise TestRowsError("TEST rows present after filtering")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("bar labels must be unique and increasing")
    if not np.isfinite(frame["close"].to_numpy()).all() or (frame["close"] <= 0).any():
        raise ValueError("non-finite or non-positive close")
    return frame


def plugin_from_config(cfg: dict) -> Plugin:
    plugin = Plugin()
    plugin.set_params(**{k: cfg[k] for k in Plugin.plugin_params if k in cfg})
    return plugin


def chunk_path(store: Path, window: int, year: int) -> Path:
    return store / "chunks" / f"w{window}" / f"{year}.parquet"


def cmd_chunk(args) -> None:
    cfg = load_config(args.config)
    lake = resolve_lake(cfg, args.lake)
    window = int(cfg["window_bars"])
    year = int(args.year)
    year_start, year_end = utc(f"{year}-01-01"), utc(f"{year + 1}-01-01")
    warm_start = utc(f"{year - 1}-12-01")
    if year_start < utc(cfg["warmup_start"]) + pd.DateOffset(months=1):
        raise ValueError("year lies before the configured warm-up span")
    frame = load_close(lake, warm_start, year_end, cfg["test_start_closed"])
    warm_rows = int((frame.index < year_start).sum())
    if warm_rows < window:
        raise ValueError(f"warm-up has {warm_rows} bars < window {window}")
    plugin = plugin_from_config(cfg)
    data = frame.rename(columns={"close": cfg.get("close_column", "CLOSE")})
    feats = plugin.process(data)
    out = pd.concat([frame, feats], axis=1)
    out = out.loc[out.index >= year_start].copy()
    out.insert(1, "window_bars", window)
    out = out.reset_index()
    for col in CANDIDATE_COLUMNS:
        if out[f"{col}_w{window}"].isna().any():
            raise ValueError(f"{col} has NaN inside year {year} despite warm-up")
    store = Path(cfg["output_dir"])
    path = chunk_path(store, window, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"chunk w{window} {year}: rows={len(out)} warm_up_bars={warm_rows} -> {path}")


def _split_of(ts: pd.Series, cfg: dict) -> pd.Series:
    labels = pd.Series("outside_configured_span", index=ts.index, dtype=object)
    for name, b in cfg["splits"].items():
        mask = (ts >= utc(b["start_inclusive"])) & (ts < utc(b["end_exclusive"]))
        labels.loc[mask] = name
    return labels


def cmd_assemble(args) -> None:
    cfgs = [load_config(p) for p in args.config]
    cfg0 = cfgs[0]
    store = Path(cfg0["output_dir"])
    lake = resolve_lake(cfg0, args.lake)
    test_start = utc(cfg0["test_start_closed"])
    manifest = {
        "schema": SCHEMA,
        "instrument": cfg0["instrument"],
        "bar": cfg0["bar"],
        "source_name": cfg0["input_file"],
        "source_sha256": cfg0["input_sha256"],
        "source_read_rule": "read-only; filtered at read to [warmup_start, test_start_closed); any request reaching TEST is refused",
        "ported_from": RETSU_SOURCE,
        "plugin": "feature-eng app/plugins/causal_fractal.py (entry point feature_eng.plugins:causal_fractal)",
        "window_is_observed_bars_not_calendar_time": True,
        "row_i_uses_closes": "[i-window+1, i] inclusive, and no later close",
        "parameters_fit_on_validation": False,
        "parameters_fit_on_test": False,
        "global_scaler_fit": False,
        "warmup": {
            "start": cfg0["warmup_start"],
            "rule": "each calendar year is computed with history from the previous 1 December; warm-up rows are dropped, never emitted",
        },
        "splits_utc": cfg0["splits"],
        "test_split_file": None,
        "test_start_closed": cfg0["test_start_closed"],
        "store_dir": str(store),
        "store_is_outside_lake_and_git": True,
        "columns_per_window": {
            "candidates": list(CANDIDATE_COLUMNS),
            "diagnostic_only": list(DIAGNOSTIC_COLUMNS),
            "intermediate": list(INTERMEDIATE_COLUMNS),
            "naming": "<feature>_w<window_bars>",
        },
        "diagnostic_reason": "correlation_dimension and spectral_slope are past-only but fragile on one short window; correlation_dimension is NaN for window_bars < 60",
        "windows": {},
        "does_not_select_a_feature": True,
        "does_not_claim_predictive_value": True,
        "does_not_authorize_a_strategy": True,
    }
    for cfg in cfgs:
        if Path(cfg["output_dir"]) != store or cfg["input_sha256"] != cfg0["input_sha256"]:
            raise ValueError("all configs must share output_dir and input_sha256")
        window = int(cfg["window_bars"])
        years = sorted(int(p.stem) for p in (store / "chunks" / f"w{window}").glob("*.parquet"))
        parts = [pd.read_parquet(chunk_path(store, window, y)) for y in years]
        frame = pd.concat(parts, ignore_index=True)
        ts = pd.DatetimeIndex(frame["date_time"])
        if not ts.is_unique or not ts.is_monotonic_increasing:
            raise ValueError("assembled rows are not unique and increasing")
        if ts.max() >= test_start:
            raise TestRowsError("assembled store reaches TEST")
        frame["split"] = _split_of(frame["date_time"], cfg).to_numpy()
        frame = frame.loc[frame["split"] != "outside_configured_span"].reset_index(drop=True)
        # continuity against the lake: every lake bar in the span is present exactly once
        lake_bars = load_close(lake, min(utc(b["start_inclusive"]) for b in cfg["splits"].values()),
                               max(utc(b["end_exclusive"]) for b in cfg["splits"].values()), test_start)
        if len(lake_bars) != len(frame) or not (lake_bars.index == pd.DatetimeIndex(frame["date_time"])).all():
            raise ValueError(f"w{window}: assembled rows differ from lake bars in span ({len(frame)} vs {len(lake_bars)})")
        entry = {"files": {}, "sha256": {}, "row_counts": {}, "finite_candidate_rows": {}, "medians": {},
                 "years_chunked": years, "rs_scales": [int(s) for s in rs_scales_for_window(window)],
                 "correlation_dimension_defined": window >= 60}
        feat_cols = [f"{c}_w{window}" for c in FEATURE_COLUMNS]
        for name in cfg["splits"]:
            part = frame.loc[frame["split"] == name, ["date_time", "close", "split", "window_bars"] + feat_cols]
            fname = f"eurusd_h1_causal_fractal_w{window}_{name}.parquet"
            part.to_parquet(store / fname, index=False)
            entry["files"][name] = fname
            entry["sha256"][name] = file_sha256(store / fname)
            entry["row_counts"][name] = int(len(part))
            entry["finite_candidate_rows"][name] = int(part[[f"{c}_w{window}" for c in CANDIDATE_COLUMNS]].notna().all(axis=1).sum())
            entry["medians"][name] = {c: (None if np.isnan(v) else round(float(v), 6))
                                      for c, v in part[feat_cols].median().items()}
        manifest["windows"][str(window)] = entry
        print(f"w{window}: " + ", ".join(f"{k}={v}" for k, v in entry["row_counts"].items()))
    text = json.dumps(manifest, indent=2) + "\n"
    (store / "MANIFEST.json").write_text(text, encoding="utf-8")
    copy = REPO / cfg0["manifest_copy"]
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text, encoding="utf-8")
    print("manifest sha256", hashlib.sha256(text.encode()).hexdigest(), "->", store / "MANIFEST.json", "and", copy)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunk"); c.add_argument("--config", required=True); c.add_argument("--year", required=True, type=int)
    c.add_argument("--lake", default=None); c.set_defaults(func=cmd_chunk)
    a = sub.add_parser("assemble"); a.add_argument("--config", required=True, action="append")
    a.add_argument("--lake", default=None); a.set_defaults(func=cmd_assemble)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
