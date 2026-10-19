"""Redundancy / diagnostic report of the EURUSD H1 causal-fractal candidates on TRAIN only.

    PYTHONPATH=. python tools/causal_fractal_eurusd/report_redundancy.py --config examples/config/causal_fractal_eurusd_h1_w200.json [--config ...]

Aggregates only (no row-level data is written): per window, Pearson and
Spearman correlation of each fractal column with trailing realized volatility
(rv5, rv24: std of hourly log returns over the trailing 5 / 24 observed bars),
with the ATR ratio (Wilder ATR(14) / close), with each other; medians; and the
within-day variance share. Writes docs/causal_fractal_eurusd/REDUNDANCY.{json,md}.
This report describes redundancy among candidates; it measures no predictive value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.plugins.causal_fractal import CANDIDATE_COLUMNS, DIAGNOSTIC_COLUMNS  # noqa: E402
from tools.causal_fractal_eurusd.build_store import load_config, load_close, resolve_lake, utc  # noqa: E402

REPORT_COLUMNS = list(CANDIDATE_COLUMNS) + list(DIAGNOSTIC_COLUMNS)
OUT_DIR = REPO / "docs/causal_fractal_eurusd"


def wilder_atr(high, low, close, period=14):
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def reference_columns(lake: Path, cfg: dict) -> pd.DataFrame:
    tr = cfg["splits"]["train"]
    start = utc(tr["start_inclusive"]) - pd.Timedelta(days=10)   # trailing warm-up only
    raw = pd.read_parquet(lake, filters=[("datetime", ">=", start), ("datetime", "<", utc(tr["end_exclusive"]))])
    idx = pd.DatetimeIndex(raw["datetime"]); idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if idx.max() >= utc(cfg["test_start_closed"]):
        raise ValueError("reference columns touched TEST")
    f = pd.DataFrame({"close": raw["close"].astype(float).to_numpy(), "high": raw["high"].astype(float).to_numpy(),
                      "low": raw["low"].astype(float).to_numpy()}, index=idx.rename("date_time"))
    r = np.log(f["close"]).diff()
    f["rv5"] = r.rolling(5).std(ddof=1)
    f["rv24"] = r.rolling(24).std(ddof=1)
    f["atr_ratio"] = wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy()) / f["close"]
    return f.loc[f.index >= utc(tr["start_inclusive"]), ["rv5", "rv24", "atr_ratio"]]


def within_day(series: pd.Series) -> dict:
    s = series.dropna()
    days = s.index.tz_convert("UTC").floor("D")
    total_var = float(s.var(ddof=1))
    day_means = s.groupby(days).transform("mean")
    within_var = float((s - day_means).var(ddof=1))
    within_std = s.groupby(days).std(ddof=1)
    return {"within_day_variance_share": round(within_var / total_var, 4) if total_var > 0 else None,
            "mean_within_day_std": round(float(within_std.mean()), 6),
            "overall_std": round(float(s.std(ddof=1)), 6),
            "days": int(days.nunique())}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, action="append")
    ap.add_argument("--lake", default=None)
    args = ap.parse_args(argv)
    cfgs = [load_config(p) for p in args.config]
    cfg0 = cfgs[0]
    store = Path(cfg0["output_dir"])
    manifest = json.loads((store / "MANIFEST.json").read_text())
    lake = resolve_lake(cfg0, args.lake)
    ref = reference_columns(lake, cfg0)
    report = {"schema": "causal_fractal_redundancy.eurusd_h1.v1", "split": "train", "source_sha256": cfg0["input_sha256"],
              "store_manifest_schema": manifest["schema"], "reference_columns": {
                  "rv5": "std(ddof=1) of hourly log returns over the trailing 5 observed bars",
                  "rv24": "same over the trailing 24 observed bars",
                  "atr_ratio": "Wilder ATR(14) / close"},
              "measures_predictive_value": False, "windows": {}}
    md = ["# Redundancia de los fractales causales EURUSD H1 (solo TRAIN 2020–2023)", "",
          "Agregados únicamente. Correlaciones de Pearson / Spearman con la volatilidad realizada (rv5, rv24), con la razón ATR(14)/close y entre sí; medianas; y la parte de la varianza que ocurre dentro del día. Nada aquí mide valor predictivo.", ""]
    for cfg in cfgs:
        w = int(cfg["window_bars"])
        train = pd.read_parquet(store / manifest["windows"][str(w)]["files"]["train"])
        train = train.set_index(pd.DatetimeIndex(train["date_time"]))
        cols = [f"{c}_w{w}" for c in REPORT_COLUMNS]
        joined = train[cols].join(ref, how="inner")
        assert len(joined) == len(train), "reference columns must align on every TRAIN bar"
        entry = {"rows": int(len(joined)), "medians": {}, "vs_reference": {}, "within_day": {}, "pearson_among": {}, "spearman_among": {}}
        pear = joined.corr(method="pearson"); spear = joined.corr(method="spearman")
        for c in cols:
            short = c[: -len(f"_w{w}")]
            entry["medians"][short] = None if joined[c].isna().all() else round(float(joined[c].median()), 6)
            entry["vs_reference"][short] = {r: {"pearson": None if np.isnan(pear.loc[c, r]) else round(float(pear.loc[c, r]), 4),
                                                "spearman": None if np.isnan(spear.loc[c, r]) else round(float(spear.loc[c, r]), 4)}
                                            for r in ("rv5", "rv24", "atr_ratio")}
            entry["within_day"][short] = within_day(joined[c]) if not joined[c].isna().all() else None
            entry["pearson_among"][short] = {d[: -len(f"_w{w}")]: (None if np.isnan(pear.loc[c, d]) else round(float(pear.loc[c, d]), 4)) for d in cols}
            entry["spearman_among"][short] = {d[: -len(f"_w{w}")]: (None if np.isnan(spear.loc[c, d]) else round(float(spear.loc[c, d]), 4)) for d in cols}
        report["windows"][str(w)] = entry
        md += [f"## Ventana {w} barras (filas TRAIN: {len(joined)})", "",
               "| columna | mediana | r rv5 | ρ rv5 | r rv24 | ρ rv24 | r atr_ratio | ρ atr_ratio | varianza intradía | std intradía media | std total |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for short in REPORT_COLUMNS:
            v = entry["vs_reference"][short]; wd = entry["within_day"][short]
            fmt = lambda x: "—" if x is None else f"{x:.3f}"
            md.append(f"| {short} | {fmt(entry['medians'][short])} | {fmt(v['rv5']['pearson'])} | {fmt(v['rv5']['spearman'])} | {fmt(v['rv24']['pearson'])} | {fmt(v['rv24']['spearman'])} | {fmt(v['atr_ratio']['pearson'])} | {fmt(v['atr_ratio']['spearman'])} | "
                      + (f"{wd['within_day_variance_share']:.3f} | {wd['mean_within_day_std']:.4f} | {wd['overall_std']:.4f} |" if wd else "— | — | — |"))
        md += ["", f"Pearson entre columnas (ventana {w}):", "", "| | " + " | ".join(REPORT_COLUMNS) + " |", "|---|" + "---:|" * len(REPORT_COLUMNS)]
        for a in REPORT_COLUMNS:
            md.append(f"| {a} | " + " | ".join("—" if entry["pearson_among"][a][b] is None else f"{entry['pearson_among'][a][b]:.3f}" for b in REPORT_COLUMNS) + " |")
        md.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REDUNDANCY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "REDUNDANCY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", OUT_DIR / "REDUNDANCY.json", OUT_DIR / "REDUNDANCY.md")


if __name__ == "__main__":
    main()
