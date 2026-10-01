#!/usr/bin/env python3
"""Per-iteration selection ledger: every probed candidate is ELIGIBLE, REJECTED or PENDING, with reason, target,
horizon, rows and cost. ELIGIBLE requires MAE and MSE below the zero-return naive in every fold AND a passing
intercept-only (drift) control; a pass without that control is PENDING, never eligible. Deferred catalogue families
are PENDING with their named missing action. Cost is a compute proxy (features x train rows per fold) unless a
measured wall time is present in the source file."""
import argparse, json, hashlib
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--iteration", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="additional stage-pipeline outcome files (lane_b_stage_outcome.v1); a later file's rows for a "
                    "(dataset, candidate) supersede an earlier file's horizon='all' PENDING row for the same pair")
    ap.add_argument("--closure", nargs="*", default=[], help="lane_b_intercept_closure.v1 files (paths under --base); a verdict keyed by "
                    "(dataset, candidate, horizon, split) decides that row: FEATURE_SIGNAL -> ELIGIBLE, otherwise REJECTED")
    ap.add_argument("--hscoped-v2", nargs="*", default=[], help="DATASET=path pairs: horizon-scoped probe v2 outputs carrying the intercept-only control")
    a = ap.parse_args(); b = Path(a.base); L = lambda p: json.load(open(b / p))
    summ = L("LANE_B_PROBE_SUMMARY.v1.json"); drift = L("macro_daily/DRIFT_CHECK.v1.json")
    drift_ok = {(r["case"].split("_S")[0] if "_S" in r["case"] else r["case"], r["case"], r["horizon_h"]) for r in drift["results"] if r["verdict"] == "FEATURE_SIGNAL"}
    case_map = {("EURUSD_1h_gitpinned", "inner_TRAIN"): "EURUSD_gitpinned", ("EURUSD_1h_lake", "S1_70_15_15"): "EURUSD_lake_S1", ("EURUSD_1h_lake", "S2_prospective_reserve"): "EURUSD_lake_S2",
                ("GBPUSD_1h_lake", "S1_70_15_15"): "GBPUSD_lake_S1", ("GBPUSD_1h_lake", "S2_prospective_reserve"): "GBPUSD_lake_S2"}
    hs2 = {}
    for spec in a.hscoped_v2:
        ds, pth = spec.split("=", 1); d = json.load(open(b / pth)); hz = d["target"].split("@")[1].split()[0]
        for cand, v in d["candidates"].items():
            hs2[(ds, cand, hz)] = (v["verdict"], pth)
    clo = {}
    for pth in a.closure:
        for v in json.load(open(b / pth))["verdicts"]:
            assert v["reproduces_v1"], (pth, v["candidate"], v["horizon"])
            clo[(v["dataset"], v["candidate"], v["horizon"], v["split"])] = (v["verdict"], pth)
    rows = []
    def add(**k): k["id"] = len(rows); rows.append(k)
    for r in summ["rows"]:
        h = int(r["horizon"].rstrip("hd")) if r["horizon"][-1] in "hd" else r["horizon"]
        folds = [f for f in r["folds"] if "mae" in f]
        n_rows = {"val_rows": sum(f.get("val_rows", 0) for f in folds), "folds": len(folds)}
        cost = {"proxy": "features x train_rows not stored per row; see source", "measured_wall_seconds": None}
        cv = clo.get((r["asset"], r["candidate"], r["horizon"], r["split"]))
        if cv:
            st = "ELIGIBLE" if cv[0] == "FEATURE_SIGNAL" else "REJECTED"
            why = (f"MAE and MSE below the zero-return naive AND the intercept-only control in every fold; v1 rows reproduced exactly ({cv[1]})" if st == "ELIGIBLE"
                   else f"closure verdict {cv[0]}: not below both the zero-return naive and the intercept-only control in every fold, MAE and MSE; v1 rows reproduced exactly ({cv[1]})")
        elif r["gate"] == "PASS":
            ck = case_map.get((r["asset"], r["split"]))
            v2 = hs2.get((r["asset"], r["candidate"], r["horizon"]))
            if v2 and v2[0] == "FEATURE_SIGNAL":
                st, why = "ELIGIBLE", f"MAE and MSE below the zero-return naive AND the intercept-only control in every fold, identical rows ({v2[1]})"
            elif v2:
                st, why = "REJECTED", f"intercept-only control verdict {v2[0]} ({v2[1]})"
            elif r["candidate"] == "range" and ck and any(x[1] == ck and x[2] == h for x in drift_ok):
                st, why = "ELIGIBLE", "MAE and MSE below the zero-return naive in every fold; intercept-only control passed (DRIFT_CHECK.v1)"
            else:
                st, why = "PENDING", "passes the zero-return naive but the intercept-only control was not run for this candidate"
        elif r["gate"] == "MSE_NOT_STORED":
            st, why = "PENDING", "MSE not stored by this early probe; rerun with MAE and MSE"
        else:
            st, why = "REJECTED", "not below the zero-return naive in every fold (MAE and MSE)"
        add(dataset=r["asset"], candidate=r["candidate"], target="cumulative log return (elapsed seconds)", horizon=r["horizon"], split=r["split"],
            status=st, reason=why, rows=n_rows, cost=cost, source=r["source"])
    for a_ in ("eth", "btc", "eurusd", "gbpusd"):
        d = L(f"seasonal/SEASONAL_REFERENCE_PROBE.{a_}.v1.json")
        for ds, v in d["datasets"].items():
            for cand, hz in v["results"].items():
                for h, x in hz.items():
                    add(dataset=ds, candidate=f"{cand}+seasonal_reference", target="y - seasonal naive (added back)", horizon=h, split="inner_TRAIN",
                        status="ELIGIBLE" if x["gate_model_vs_zero"] == "PASS" else "REJECTED",
                        reason="seasonal reference never beats the zero-return naive; model+seasonal not below zero naive in every fold" if x["gate_model_vs_zero"] != "PASS" else "passes",
                        rows={"val_rows": sum(rr["val_rows"] for rr in x["rows"]), "folds": len(x["rows"])}, cost={"proxy": "see source", "measured_wall_seconds": None},
                        source=f"seasonal/SEASONAL_REFERENCE_PROBE.{a_}.v1.json")
    m = L("macro_daily/MACRO_DAILY_PROBE.v1.json")
    for asset, v in m["assets"].items():
        for fam, r in v["results"].items():
            if r.get("status"):
                add(dataset=asset, candidate=fam, target="daily cumulative log return", horizon="1..6d", split="inner_TRAIN_daily", status="PENDING",
                    reason=r["status"], rows={}, cost={}, source="macro_daily/MACRO_DAILY_PROBE.v1.json"); continue
            for h, x in r.items():
                if not isinstance(x, dict): continue
                add(dataset=asset, candidate=fam, target="daily cumulative log return", horizon=h, split="inner_TRAIN_daily",
                    status="ELIGIBLE" if x["gate"] == "PASS" else "REJECTED",
                    reason="not below BOTH the zero-return naive and the intercept-only control in every fold" if x["gate"] != "PASS" else "passes both controls",
                    rows={"val_rows": sum(rr.get("val_rows", 0) for rr in x["rows"]), "folds": len(x["rows"]), "channels": len(r["channels_used"])},
                    cost={"proxy": "see source", "measured_wall_seconds": None}, source="macro_daily/MACRO_DAILY_PROBE.v1.json")
    for e in a.extra:
        new_rows = json.load(open(e))["ledger_rows"]; pairs = {(x["dataset"], x["candidate"]) for x in new_rows}
        kept = [r for r in rows if not (r["horizon"] == "all" and r["status"] == "PENDING" and (r["dataset"], r["candidate"]) in pairs)]
        rows[:] = kept
        for i, r in enumerate(rows): r["id"] = i
        for x in new_rows:
            add(**x)
    den = L("SELECTION_DENOMINATORS.v1.json")
    for t in den["rows"]:
        if t["deferred"]:
            add(dataset="catalogue:" + t["source_family"], candidate="deferred channels", target="n/a", horizon="n/a", split="n/a", status="PENDING",
                reason="DEFERRED: " + (list(t["top_reasons"])[0] if t["top_reasons"] else "not yet staged"), rows={"channels": t["deferred"]}, cost={}, source="SELECTION_DENOMINATORS.v1.json")
    cnt = {s: sum(r["status"] == s for r in rows) for s in ("ELIGIBLE", "REJECTED", "PENDING")}
    doc = {"schema": "lane_b_selection_ledger.v1", "iteration": a.iteration, "rows": rows, "counts": cnt,
           "eligible": [r for r in rows if r["status"] == "ELIGIBLE"],
           "inputs": {**{p: hashlib.sha256((b / p).read_bytes()).hexdigest() for p in ("LANE_B_PROBE_SUMMARY.v1.json", "macro_daily/DRIFT_CHECK.v1.json", "macro_daily/MACRO_DAILY_PROBE.v1.json")},
                      **{spec.split("=", 1)[1]: hashlib.sha256((b / spec.split("=", 1)[1]).read_bytes()).hexdigest() for spec in a.hscoped_v2},
                      **{pth: hashlib.sha256((b / pth).read_bytes()).hexdigest() for pth in a.closure},
                      **{Path(e).name: hashlib.sha256(Path(e).read_bytes()).hexdigest() for e in a.extra}},
           "rule": __doc__}
    json.dump(doc, open(a.out, "w"), indent=1)
    print(json.dumps({"counts": cnt, "eligible": [(r["dataset"], r["candidate"], r["horizon"], r["split"]) for r in doc["eligible"]]}))

if __name__ == "__main__":
    main()
