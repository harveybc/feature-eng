"""Lane A batch runner: python -m tools.eurusd_ps.run_batch --batch batch_001 --inputs DIR --out DIR"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import contract as C
from . import features as F
from . import inventory as INV
from . import profile as P
from . import sources as S
from . import targets as T
from . import variants as V

NY = "America/New_York"
# measured clock eras of the 2011-2021 archive (data-gov CALENDAR_REGISTRATION_2026_09_26, registrations.v1.json);
# from 2018-03 the registered reading is America/New_York local time (month-grained offsets are the estimation grain).
ARCHIVE_ERAS = [
    {"from": "2011-01-01", "to": "2012-04-30", "status": "UNDETERMINED"},
    {"from": "2012-05-01", "to": "2018-01-31", "status": "DETERMINED", "utc_offset_seconds": -18000},
    {"from": "2018-02-01", "to": "2018-02-28", "status": "UNDETERMINED"},
    {"from": "2018-03-01", "to": "2018-08-31", "status": "DETERMINED", "tz": NY},
    {"from": "2018-09-01", "to": "2018-09-30", "status": "UNDETERMINED"},
    {"from": "2018-10-01", "to": "2021-04-30", "status": "DETERMINED", "tz": NY},
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def reconcile_pinned(hourly: pd.DataFrame, inputs: str) -> dict:
    out = {"rule": "pinned stamp S (UTC bar start) <-> lake-derived UTC bar with end S+1h", "files": {}}
    for nm in ("base_d2.csv", "base_d3.csv", "base_d5.csv", "base_d6.csv"):
        p = os.path.join(inputs, nm)
        d = pd.read_csv(p, usecols=["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"])
        st = pd.to_datetime(d["DATE_TIME"]).dt.tz_localize("UTC")
        d = d[st + pd.Timedelta(hours=1) <= C.READ_END]
        res = {"sha256": sha(p), "rows": int(len(d))}
        for lab, off in (("as_UTC_bar_start", 1), ("as_UTC_bar_end", 0), ("as_UTC_start_minus1h", 2)):
            key = pd.to_datetime(d["DATE_TIME"]).dt.tz_localize("UTC") + pd.Timedelta(hours=off)
            j = hourly.reindex(pd.DatetimeIndex(key))
            m = j["close"].notna().to_numpy()
            eq = np.isclose(j["close"].to_numpy()[m], d["CLOSE"].to_numpy()[m], atol=1e-5)
            res[lab] = {"matched_rows": int(m.sum()), "close_equal_share": float(eq.mean()) if m.sum() else None}
        out["files"][nm] = res
    return out


def runtime_future_perturbation(b5: pd.DataFrame, decision: pd.DatetimeIndex) -> dict:
    """FS01 on real bytes: perturb every 5m bar ending after a cut; features and
    rows at or before the cut must be bit-identical."""
    cut = decision[len(decision) // 2]
    b5p = b5.copy()
    m = b5p["end_utc"] > cut
    rng = np.random.default_rng(7)
    for c in ("open", "high", "low", "close"):
        b5p.loc[m, c] = b5p.loc[m, c].to_numpy() * np.exp(rng.normal(0, 0.01, int(m.sum())))
    h0, h1 = S.hourly_from_5m(b5), S.hourly_from_5m(b5p)
    d0 = decision[decision <= cut]
    f0, _ = F.price_features(h0, d0)
    f1, _ = F.price_features(h1, d0)
    diff = (f0.fillna(-9e99) != f1.fillna(-9e99)).to_numpy().sum()
    return {"cut": str(cut), "rows_checked": int(len(d0)), "columns": int(f0.shape[1]), "cells_changed": int(diff),
            "status": "PASS" if diff == 0 else "FAIL"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-commit", default="UNCOMMITTED")
    a = ap.parse_args(argv)
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "READY")):
        raise SystemExit("REFUSED: batch already READY; a new batch number is required")
    cost = {}
    t0 = time.time()
    b5, clock = S.load_lake_5m(os.path.join(a.inputs, "eurusd_5m.parquet"))
    hourly = S.hourly_from_5m(b5)
    cost["load_5m_and_hourly_s"] = time.time() - t0
    lake1h = pd.read_parquet(os.path.join(a.inputs, "eurusd_1h_lake.parquet"),
                             filters=[("timestamp", "<", (C.READ_END - pd.Timedelta(days=1)))])
    clock_1h = S.infer_fx_clock(lake1h["timestamp"])
    recon = reconcile_pinned(hourly, a.inputs)
    decision = hourly.index[(hourly.index >= C.TRAIN_START) & (hourly.index < C.TRAIN_END)]
    assert decision.max() < C.TRAIN_END and b5["end_utc"].max() <= C.READ_END

    t0 = time.time()
    tg = T.build_targets(hourly, b5, decision)
    cost["targets_s"] = time.time() - t0

    feats, meta, build_s = [], [], {}
    t0 = time.time(); pf, pm = F.price_features(hourly, decision); build_s["price"] = time.time() - t0
    feats.append(pf); meta += pm
    t0 = time.time(); cf, cm = F.calendar_features(decision); build_s["calendar"] = time.time() - t0
    feats.append(cf); meta += cm
    t0 = time.time()
    arch, arch_stats = S.load_archive_calendar(os.path.join(a.inputs, "economic_calendar_2011_2021.csv"), ARCHIVE_ERAS)
    ev = F.archive_event_table(arch)
    ef, em, _ = F.event_features(ev, decision)
    build_s["events"] = time.time() - t0
    feats.append(ef); meta += em
    X = pd.concat(feats, axis=1)
    cost["features_s"] = build_s

    sup = {m["feature_id"]: m["support_h"] for m in meta}
    folds = C.inner_folds(decision)
    contract = C.contract_document(sup)

    fam_n = pd.Series([m["family"] for m in meta]).value_counts().to_dict()
    cells, cov = [], []
    t0 = time.time()
    for m in meta:
        fid = m["feature_id"]
        fam_cost = build_s["price"] if fid.startswith(("px.", "ta.", "q.")) else build_s["calendar"] if fid.startswith("cal.") else build_s["events"]
        cells += P.profile_feature(fid, X[fid], build_cost_s=fam_cost / max(1, sum(1 for k in meta if k["feature_id"][:3] == fid[:3])))
        cov += P.coverage_rows(fid, X[fid], folds)
    cost["profile_s"] = time.time() - t0

    # admissibility
    for m in meta:
        if m["role"] != "feature":
            m["admissibility"] = "EXCLUDED_ROLE:" + m["role"]
        elif m["source"] == F.ARCHIVE_SRC:
            m["admissibility"] = "ADMISSIBLE_WITH_DECLARED_ASSUMPTION (archive clock measured; availability = scheduled+1min; provenance unknown)"
        else:
            m["admissibility"] = "ADMISSIBLE"
        x = X[m["feature_id"]]
        m["train_rows"] = int(len(x)); m["train_finite"] = int(np.isfinite(x.to_numpy(float)).sum())
        m["train_coverage"] = m["train_finite"] / m["train_rows"]

    # inventory
    t0 = time.time()
    lake_rows = [INV.classify_lake_source(r) for r in INV.scan_lake_metadata(os.path.join(a.inputs, "fd_meta"))]
    src_rows = lake_rows + INV.non_lake_sources()
    fxa = pd.read_parquet(os.path.join(a.inputs, "fxmacrodata_announcements.parquet"))
    fxc = pd.read_parquet(os.path.join(a.inputs, "fxmacrodata_release_calendar.parquet"))
    fxa_tr = (fxa["announcement_datetime_utc"] >= C.TRAIN_START) & (fxa["announcement_datetime_utc"] < C.TRAIN_END)
    fxc_tr = (fxc["announcement_datetime_utc"] >= C.TRAIN_START) & (fxc["announcement_datetime_utc"] < C.TRAIN_END)
    col_rows = []
    col_rows += INV.column_rows_from_frame("fxmacrodata_announcements", fxa, "announcement_datetime_utc", fxa_tr, INV.FXM_LIC, "event",
                                           "date (reference period, string)", "announcement_datetime_utc (observed publication); receipt file-grain",
                                           {"val": {"unit": "ABSENT_IN_SOURCE"}})
    col_rows += INV.column_rows_from_frame("fxmacrodata_release_calendar", fxc, "announcement_datetime_utc", fxc_tr, INV.FXM_LIC, "event",
                                           "scheduled instant", "file-grain acquired_at 2026-05-01")
    arch_raw = pd.read_csv(os.path.join(a.inputs, "economic_calendar_2011_2021.csv"), header=None, dtype=str, keep_default_na=False,
                           names=["event_date", "event_time", "country", "volatility", "description", "evaluation", "data_format", "actual", "forecast", "previous"])
    arch_raw["_t"] = pd.to_datetime(arch_raw["event_date"].str.strip(), format="%Y/%m/%d", errors="coerce")
    col_rows += INV.column_rows_from_frame("calendar_archive_2011_2021", arch_raw.drop(columns=["_t"]), None,
                                           (arch_raw["_t"] >= "2012-05-01") & (arch_raw["_t"] < "2024-01-01"), "UNKNOWN_PROVENANCE", "event",
                                           "event_date+event_time (naive, measured eras)", "scheduled+1min (assumed)",
                                           {"actual": {"unit": "data_format marker (%/K/M/B/T), not a series unit"},
                                            "forecast": {"note": "consensus/forecast"}, "previous": {"note": "as published with the release"},
                                            "volatility": {"note": "importance tier"}})
    col_rows += [{"source": "lake_eurusd_5m", "column": c, "dtype": str(b5[c].dtype) if c in b5 else "ABSENT", "rows_total": int(len(b5)),
                  "non_null_total": int(b5[c].notna().sum()) if c in b5 else 0, "rows_in_train": int(((b5["end_utc"] > C.TRAIN_START)).sum()),
                  "non_null_in_train": int((b5[c].notna() & (b5["end_utc"] > C.TRAIN_START)).sum()) if c in b5 else 0,
                  "unit": "EUR in USD" if c in ("open", "high", "low", "close") else ("UTC instant" if "utc" in c else "ABSENT"),
                  "frequency": "5m", "licence": INV.HISTDATA_LIC, "event_time": "bar interval", "availability_time": "bar end",
                  "span": [str(b5["start_utc"].min()), str(b5["end_utc"].max())],
                  "note": "" if c in b5 else "ABSENT_IN_LAKE: no volume or spread in the HistData-derived bytes"}
                 for c in ("start_utc", "end_utc", "open", "high", "low", "close", "volume", "spread")]
    # transform variants with measured prefix tests on TRAIN log prices
    xs = np.log(hourly["close"].reindex(decision).dropna().to_numpy()[:3000])
    probes = [700, 1200, 1800, 2400, 2900]
    tv_rows = []
    for v in INV.TRANSFORM_VARIANTS:
        t1 = time.time()
        try:
            r = V.prefix_test(V.IMPLS[v["variant_id"]], xs, probes)
        except Exception as e:
            r = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
        r["cost_s"] = time.time() - t1
        expected = "PREFIX_INVARIANT_MEASURED" if v["causal_by_construction"] else "PREFIX_VIOLATION_MEASURED"
        tv_rows.append(v | r | {"agrees_with_declaration": r.get("status") == expected,
                                "admissible_as_feature": r.get("status") == "PREFIX_INVARIANT_MEASURED",
                                "ps1_profile": "PENDING (PS4: profiled after PS2 priority; identity fixed here)"})
    cost["inventory_and_variants_s"] = time.time() - t0

    t0 = time.time()
    fp = runtime_future_perturbation(b5, decision)
    cost["runtime_fs01_s"] = time.time() - t0

    # ---- write artifacts
    out = a.out
    adm = pd.DataFrame(meta)
    adm.to_csv(os.path.join(out, "admissible_features.csv"), index=False)
    jdump({"schema": "laneA_admissible_features.v1", "batch": a.batch, "features": meta}, os.path.join(out, "admissible_features.json"))
    cdf = pd.DataFrame(cells)
    cdf["value"] = cdf["value"].map(lambda v: json.dumps(v, default=str) if v is not None else "")
    cdf.to_csv(os.path.join(out, "profile_cells.csv"), index=False)
    piv = cdf.pivot(index="feature_id", columns="metric", values="state")[P.METRICS]
    piv.to_csv(os.path.join(out, "metric_state_matrix.csv"))
    pd.DataFrame(cov).to_csv(os.path.join(out, "coverage_matrix.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(os.path.join(out, "inventory_sources.csv"), index=False)
    pd.DataFrame(col_rows).to_csv(os.path.join(out, "inventory_columns.csv"), index=False)
    pd.DataFrame(tv_rows).to_csv(os.path.join(out, "transform_variants.csv"), index=False)
    jdump({"fxmacrodata_field_requirements": [{"field": f, "column": c, "state": s} for f, c, s in INV.FXM_FIELD_REQUIREMENTS],
           "fxmacrodata_train_rows": int(fxa_tr.sum()), "fxmacrodata_span": [str(fxa["announcement_datetime_utc"].min()), str(fxa["announcement_datetime_utc"].max())],
           "archive_load": arch_stats, "archive_eras": ARCHIVE_ERAS,
           "archive_train_releases_with_consensus": int((ev["forecast_v"].notna()).sum()),
           "archive_last_release_utc": str(ev["scheduled_utc"].max())},
          os.path.join(out, "event_sources.json"))
    tgt = tg.copy()
    tgt.insert(0, "row_id", np.arange(len(tgt)))
    tgt.index.name = "t_decision_utc"
    for c in tgt.columns:
        if tgt[c].dtype == object:
            tgt[c] = tgt[c].astype(str)
    tgt.reset_index().to_parquet(os.path.join(out, "targets_train.parquet"), index=False)
    Xo = X.copy(); Xo.index.name = "t_decision_utc"
    Xo.insert(0, "row_id", np.arange(len(Xo)))
    Xo.reset_index().to_parquet(os.path.join(out, "features_train.parquet"), index=False)
    jdump({"schema": "laneA_folds.v1", "decision_rows": int(len(decision)), "first": str(decision[0]), "last": str(decision[-1]),
           "folds": folds, "purge": contract["purge"]}, os.path.join(out, "folds.json"))
    jdump(contract, os.path.join(out, "contract.json"))
    tstate = {}
    for spec in C.Y_B_SPECS:
        tstate[spec["name"]] = tg[spec["name"] + "_state"].value_counts().to_dict()
    tstats = {c: {"n_finite": int(np.isfinite(tg[c]).sum()), "mean": float(np.nanmean(tg[c])), "std": float(np.nanstd(tg[c]))}
              for c in tg.columns if c.startswith(("Y_s_", "Y_l_")) and not c.endswith("staleness_h")}
    state_counts = cdf["state"].value_counts().to_dict()
    report = {
        "schema": "laneA_batch_report.v1", "batch": a.batch, "code_commit": a.code_commit,
        "clock_5m": clock, "clock_lake_1h": clock_1h, "reconciliation_pinned": recon,
        "denominators": {"sources_inventoried": len(src_rows), "source_columns_inventoried": len(col_rows),
                         "features": len(meta), "features_admissible": int(sum(m["admissibility"].startswith("ADMISSIBLE") for m in meta)),
                         "feature_families": fam_n, "metrics_per_feature": len(P.METRICS), "metric_cells": len(cdf),
                         "metric_cells_by_state": state_counts, "folds": len(folds), "decision_rows_train": int(len(decision)),
                         "transform_variants": len(tv_rows)},
        "targets": {"stats": tstats, "Y_b_states": tstate},
        "runtime_checks": {"FS01_future_perturbation_real_bytes": fp,
                           "read_end": str(C.READ_END), "max_5m_end_read": str(b5["end_utc"].max()),
                           "max_decision_t": str(decision.max())},
        "cost": cost | {"wall_s": time.time() - t_all, "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "host_role": "worker_a", "hardware": "CPU only (CUDA_VISIBLE_DEVICES empty)",
    }
    jdump(report, os.path.join(out, "batch_report.json"))
    inputs = {n: sha(os.path.join(a.inputs, n)) for n in sorted(os.listdir(a.inputs)) if os.path.isfile(os.path.join(a.inputs, n))}
    arts = {n: sha(os.path.join(out, n)) for n in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, n)) and n not in ("digests.json", "READY")}
    jdump({"inputs_sha256": inputs, "artifacts_sha256": arts, "code_commit": a.code_commit}, os.path.join(out, "digests.json"))
    with open(os.path.join(out, "READY"), "w") as f:
        f.write(json.dumps({"batch": a.batch, "digests_sha256": sha(os.path.join(out, "digests.json")),
                            "written_utc": pd.Timestamp.now(tz="UTC").isoformat()}) + "\n")
    print(json.dumps(report["denominators"] | {"cost": report["cost"], "fs01": fp["status"]}, default=str))


if __name__ == "__main__":
    main()
