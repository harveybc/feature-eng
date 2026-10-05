"""Lane A batch_002 runner (covariates on the batch_001 grid, targets and folds)."""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np
import pandas as pd

from . import covariates as CV
from . import inventory as INV
from . import profile as P
from .run_batch import jdump, sha


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--base", required=True, help="batch_001 directory (grid, folds, targets)")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-commit", default="UNCOMMITTED")
    ap.add_argument("--split", default="train", choices=["train", "validation_2024"])
    a = ap.parse_args(argv)
    t_all = time.time()
    from . import contract as C
    split_doc = C.configure_split(a.split)
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "READY")):
        raise SystemExit("REFUSED: batch already READY")
    if a.split != "train" and "validation_2024" not in os.path.abspath(a.out):
        raise SystemExit("REFUSED: validation output must live under a validation_2024 directory")
    base_ready = json.load(open(os.path.join(a.base, "READY")))
    folds_doc = json.load(open(os.path.join(a.base, "folds.json")))
    grid = pd.read_parquet(os.path.join(a.base, "targets_train.parquet"), columns=["row_id", "t_decision_utc"])
    decision = pd.DatetimeIndex(grid["t_decision_utc"])
    C.guard_rows(decision)
    srcs = pd.read_csv(os.path.join(a.base, "inventory_sources.csv"))
    b2 = srcs[srcs["batch"] == "batch_002"]
    fd = os.path.join(a.inputs, "fd")
    meta_root = os.path.join(a.inputs, "fd_meta")
    feats, meta, custody, clocks, col_rows, skipped = [], [], [], [], [], []
    seen = {}
    t0 = time.time()
    for _, r in b2.iterrows():
        rel = r["path"]
        if r["family"] in ("yahoo_daily", "fred"):
            prov = json.load(open(os.path.join(meta_root, rel, "provenance.json")))
            for fl in prov.get("files", []):
                p = os.path.join(fd, fl["path"])
                if not fl["path"].endswith(".parquet"):
                    continue
                if not os.path.exists(p):
                    skipped.append({"path": fl["path"], "reason": "NOT_TRANSFERRED"}); continue
                got = sha(p)
                custody.append({"path": fl["path"], "sha256": got, "provenance_sha256": fl.get("sha256"),
                                "custody": "DIGEST_MATCHES_PROVENANCE" if got == fl.get("sha256") else "DIGEST_MISMATCH"})
                parts = fl["path"].split("/")
                name = ("yh." + parts[-2]) if r["family"] == "yahoo_daily" else ("fred." + parts[-3] + "." + parts[-2])
                if got != fl.get("sha256"):
                    skipped.append({"path": fl["path"], "reason": "DIGEST_MISMATCH: not profiled"}); continue
                if got in seen:
                    skipped.append({"path": fl["path"], "reason": f"DUPLICATE_BYTES_OF {seen[got]}: profiled once"}); continue
                seen[got] = fl["path"]
                if r["family"] == "yahoo_daily":
                    s = CV.yahoo_series(p)
                    lev = name in ("yh.vix",)
                    f, m = CV.daily_features(name, s, "close", decision, "yahoo", INV.YAHOO_LIC, "lake:" + fl["path"], levels=lev)
                    span = [str(s["date"].min()), str(s["date"].max())] if len(s) else None
                else:
                    s = CV.fred_series(p)
                    f, m = CV.daily_features(name, s, "value", decision, "fred", INV.FRED_LIC, "lake:" + fl["path"], levels=True)
                    span = [str(s["date"].min()), str(s["date"].max())] if len(s) else None
                col_rows.append({"source": "lake:" + fl["path"], "column": "Close" if r["family"] == "yahoo_daily" else "value",
                                 "rows_read_until_read_end": int(len(s)), "span_read": span, "frequency": "1d",
                                 "licence": r["licence"], "event_time": r["event_time"], "availability_time": r["availability_time"]})
                feats.append(f); meta += m
        elif r["family"] == "fx_cross_pairs" and rel.startswith("features/trading_asset_data/"):
            pair = os.path.basename(rel)
            p = os.path.join(fd, rel, "1h.parquet")
            custody.append({"path": rel + "/1h.parquet", "sha256": sha(p), "provenance_sha256": None,
                            "custody": "DIGEST_RECORDED (provenance lists no per-file digest for derived timeframes)"})
            f, m, clk = CV.fx_pair_features(pair, p, decision)
            clocks.append({k: clk.get(k) for k in ("file", "status", "clock", "weeks")} | {"by_dst": clk.get("by_dst")})
            if not m:
                skipped.append({"path": rel, "reason": "FX_CLOCK_UNDETERMINED: not converted, not profiled"})
            feats.append(f); meta += m
        else:
            skipped.append({"path": rel, "reason": "DUPLICATE_LOCATION (market_data/forex copy of features/trading_asset_data)"})
    build_s = time.time() - t0
    X = pd.concat(feats, axis=1)
    cells, cov = [], []
    t0 = time.time()
    per = build_s / max(1, len(meta))
    for m in meta:
        cells += P.profile_feature(m["feature_id"], X[m["feature_id"]], build_cost_s=per)
        cov += P.coverage_rows(m["feature_id"], X[m["feature_id"]], folds_doc["folds"])
        x = X[m["feature_id"]].to_numpy(float)
        m["admissibility"] = "ADMISSIBLE"
        m["train_rows"] = int(len(x)); m["train_finite"] = int(np.isfinite(x).sum()); m["train_coverage"] = m["train_finite"] / len(x)
    prof_s = time.time() - t0
    out = a.out
    pd.DataFrame(meta).to_csv(os.path.join(out, "admissible_features.csv"), index=False)
    jdump({"schema": "laneA_admissible_features.v1", "batch": a.batch, "features": meta}, os.path.join(out, "admissible_features.json"))
    cdf = pd.DataFrame(cells)
    cdf["split"] = a.split
    cdf["value"] = cdf["value"].map(lambda v: json.dumps(v, default=str) if v is not None else "")
    cdf.to_csv(os.path.join(out, "profile_cells.csv"), index=False)
    cdf.pivot(index="feature_id", columns="metric", values="state")[P.METRICS].to_csv(os.path.join(out, "metric_state_matrix.csv"))
    pd.DataFrame(cov).to_csv(os.path.join(out, "coverage_matrix.csv"), index=False)
    pd.DataFrame(col_rows).to_csv(os.path.join(out, "inventory_columns.csv"), index=False)
    pd.DataFrame(custody).to_csv(os.path.join(out, "custody.csv"), index=False)
    Xo = X.copy(); Xo.index.name = "t_decision_utc"; Xo.insert(0, "row_id", grid["row_id"].to_numpy())
    Xo.reset_index().to_parquet(os.path.join(out, "features_train.parquet"), index=False)
    # role overlay for earlier READY batches (their files are immutable): plan update 02434903
    from . import features as F
    b1 = pd.read_csv(os.path.join(a.base, "admissible_features.csv"))
    sel = b1["source"].eq(F.ARCHIVE_SRC)
    overlay = {"schema": "laneA_role_overlay.v1", "applies_to_batch": "batch_001", "authority": "plan update 02434903 (2026-10-03)",
               "rule": "economic-calendar columns are SELECTOR_EPISODE_SOURCE (PS3-C episodes only); excluded from the PS1 "
                       "model-input candidate denominator; calendar as model input = I11 DEFERRED_FINAL_OPTIONAL. Known time "
                       "encodings (cal.*) are unaffected.",
               "selector_episode_source_features": b1.loc[sel, "feature_id"].tolist(),
               "source_rows_reroled": ["feature-eng:tests/data/economic_calendar_2011_2021.csv",
                                       "economic_calendar/release_actuals/fxmacrodata", "economic_calendar/scheduled_events/fxmacrodata",
                                       "coordinator:~/.local/state/financial-data/point_in_time", "economic_calendar/* FRED release proxies"]}
    jdump(overlay, os.path.join(out, "role_overlay_batch_001.json"))
    b1_model = b1[(~sel) & b1["role"].eq("feature")]
    cum = {"schema": "laneA_cumulative_denominators.v1", "through_batch": a.batch,
           "model_input_candidates": {"batch_001": int(len(b1_model)), a.batch: int(len(meta)), "total": int(len(b1_model) + len(meta))},
           "selector_episode_source_columns": {"batch_001": int(sel.sum()), a.batch: 0, "total": int(sel.sum())},
           "excluded_role_quality": int((b1["role"] == "quality_excluded").sum()),
           "metric_cells_model_input": {"batch_001": int(len(b1_model)) * len(P.METRICS), a.batch: len(cdf)},
           "metric_cells_selector_source": {"batch_001": int(sel.sum()) * len(P.METRICS)}}
    jdump(cum, os.path.join(out, "cumulative_denominators.json"))
    report = {"schema": "laneA_batch_report.v1", "batch": a.batch, "code_commit": a.code_commit, "split": split_doc,
              "base_batch": {"path_role": "worker_a ps1/batch_001", "ready": base_ready},
              "grid_targets_folds": "identical to batch_001 (row_id aligned); targets not rewritten",
              "fx_cross_clocks": clocks, "skipped": skipped,
              "denominators": {"sources_in_batch": int(len(b2)), "files_transferred": len(custody),
                               "files_profiled": len(custody) - sum(1 for s in skipped if s["reason"].startswith(("DIGEST", "DUPLICATE_BYTES", "FX_CLOCK"))),
                               "skipped_by_reason": pd.Series([s["reason"].split(" ")[0].split(":")[0] for s in skipped]).value_counts().to_dict(),
                               "features": len(meta), "metric_cells": len(cdf), "metric_cells_by_state": cdf["state"].value_counts().to_dict(),
                               "folds": len(folds_doc["folds"]), "decision_rows": int(len(decision)), "decision_rows_train": int(len(decision)) if a.split == "train" else 0,
                               "custody": pd.DataFrame(custody)["custody"].value_counts().to_dict()},
              "cost": {"build_s": build_s, "profile_s": prof_s, "wall_s": time.time() - t_all,
                       "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
              "host_role": "worker_a", "hardware": "CPU only"}
    jdump(report, os.path.join(out, "batch_report.json"))
    arts = {n: sha(os.path.join(out, n)) for n in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, n)) and n not in ("digests.json", "READY")}
    jdump({"inputs_sha256": {c["path"]: c["sha256"] for c in custody}, "artifacts_sha256": arts, "code_commit": a.code_commit,
           "base_batch_digests_sha256": base_ready["digests_sha256"]}, os.path.join(out, "digests.json"))
    with open(os.path.join(out, "READY"), "w") as f:
        f.write(json.dumps({"batch": a.batch, "digests_sha256": sha(os.path.join(out, "digests.json")),
                            "written_utc": pd.Timestamp.now(tz="UTC").isoformat()}) + "\n")
    print(json.dumps(report["denominators"] | {"cost": report["cost"], "skipped": len(skipped)}, default=str))


if __name__ == "__main__":
    main()
