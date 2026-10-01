#!/usr/bin/env python3
"""LANE_B_PROBE_SUMMARY: one table across lane B's linear-probe files, generated, never hand-edited.

Rows: asset x split x horizon x candidate family, with per-fold MAE/MSE beside the zero-return naive and
pass/fail per fold. A fold passes only if MAE AND MSE are strictly below naive; where a probe file stored
MAE only, the row says MSE_NOT_STORED and cannot pass. S1/S2 agreement is reported where both exist."""
import argparse, hashlib, json, re
from pathlib import Path


def fold_pass(r):
    if "mae" not in r:
        return None
    if "mse" not in r:
        return False
    return r["mae"] < r["naive_mae"] and r["mse"] < r["naive_mse"]


def rows_from_split_probe(asset, path, data):
    out = []
    for split, s in data["splits"].items():
        for cand, hz in s["results"].items():
            for h, v in hz.items():
                out.append(dict(asset=asset, source=path, split=split, horizon=h, candidate=cand, folds=v["rows"]))
    return out


def rows_from_candidates(asset, path, data, horizon, split="inner_TRAIN"):
    return [dict(asset=asset, source=path, split=split, horizon=horizon, candidate=c, folds=v["rows"]) for c, v in data["candidates"].items()]


def rows_from_variant_a(asset, path, data):
    by = {}
    for r in data["rows"]:
        if r.get("status") == "MEASURED":
            by.setdefault(r["target"].split("@")[1], []).append(r)
    return [dict(asset=asset, source=path, split="inner_TRAIN", horizon=h, candidate="A", folds=v) for h, v in by.items()]


def rows_from_daily(asset, path, data):
    return [dict(asset=asset, source=path, split="inner_TRAIN_daily", horizon=h, candidate=c, folds=v["rows"])
            for c, hz in data["results"].items() for h, v in hz.items()]


def rows_from_multi(asset, path, data, subset_key="subset"):
    by = {}
    for r in data["rows"]:
        by.setdefault((r[subset_key], r["target"].split("@")[1]), []).append(r)
    return [dict(asset=asset, source=path, split="inner_TRAIN", horizon=h, candidate=c, folds=v) for (c, h), v in by.items()]


def summarize(rows):
    for r in rows:
        fp = [fold_pass(f) for f in r["folds"]]
        r["fold_pass"] = fp
        r["mse_stored"] = all("mse" in f for f in r["folds"] if "mae" in f)
        r["gate"] = "PASS" if fp and all(x is True for x in fp) else ("MSE_NOT_STORED" if not r["mse_stored"] else "FAIL")
        m = [f for f in r["folds"] if "mae" in f]
        r["mean_mae"] = sum(f["mae"] for f in m) / len(m) if m else None
        r["mean_naive_mae"] = sum(f["naive_mae"] for f in m) / len(m) if m else None
        r["mean_mse"] = sum(f["mse"] for f in m) / len(m) if m and r["mse_stored"] else None
        r["mean_naive_mse"] = sum(f["naive_mse"] for f in m) / len(m) if m and r["mse_stored"] else None
    return rows


def agreement(rows):
    idx = {(r["asset"], r["candidate"], r["horizon"], r["split"]): r["gate"] for r in rows}
    out = []
    for (a, c, h, s), g in idx.items():
        if s == "S1_70_15_15" and (a, c, h, "S2_prospective_reserve") in idx:
            g2 = idx[(a, c, h, "S2_prospective_reserve")]
            out.append({"asset": a, "candidate": c, "horizon": h, "S1": g, "S2": g2, "agree": g == g2})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--manifests", nargs="*", default=[])
    a = ap.parse_args()
    b = Path(a.base)
    L = lambda p: json.load(open(b / p))
    rows = []
    rows += rows_from_multi("ETH_4h", "f1/eth_f1_probe.json", L("f1/eth_f1_probe.json"))
    rows += rows_from_multi("ETH_4h", "f1/eth_variant_c_probe.json", L("f1/eth_variant_c_probe.json"))
    for fam, v in L("f1/ETH_VARIANT_D_FAMILIES_PROBE.v1.json")["families"].items():
        for h in {r["target"].split("@")[1] for r in v["rows"]}:
            rows.append(dict(asset="ETH_4h", source="f1/ETH_VARIANT_D_FAMILIES_PROBE.v1.json", split="inner_TRAIN", horizon=h, candidate=fam,
                             folds=[r for r in v["rows"] if r["target"].endswith("@" + h)]))
    rows += rows_from_candidates("ETH_4h", "f1/ETH_4H_HORIZON_SCOPED_PROBE.v1.json", L("f1/ETH_4H_HORIZON_SCOPED_PROBE.v1.json"), "4h")
    rows += rows_from_variant_a("EURUSD_1h_gitpinned", "f1/EURUSD_1H_VARIANT_A_PROBE.v1.json", L("f1/EURUSD_1H_VARIANT_A_PROBE.v1.json"))
    rows += rows_from_candidates("EURUSD_1h_gitpinned", "f1/EURUSD_1H_HORIZON_SCOPED_PROBE.v1.json", L("f1/EURUSD_1H_HORIZON_SCOPED_PROBE.v1.json"), "1h")
    rows += rows_from_daily("EURUSD_1d_from_gitpinned_1h", "f1/EURUSD_1D_HORIZON_GATE.v1.json", L("f1/EURUSD_1D_HORIZON_GATE.v1.json"))
    rows += rows_from_split_probe("EURUSD_1h_lake", "eurusd_lake_5m_to_1h/lake_probe.json", L("eurusd_lake_5m_to_1h/lake_probe.json"))
    rows += rows_from_split_probe("BTCUSDT_4h_lake", "btcusdt_4h_lake/probe.json", L("btcusdt_4h_lake/probe.json"))
    rows += rows_from_split_probe("GBPUSD_1h_lake", "gbpusd_lake_5m_to_1h/probe.json", L("gbpusd_lake_5m_to_1h/probe.json"))
    rows = summarize(rows)
    variant_a = [r for r in rows if r["candidate"] in ("A", "C_ALL")]
    range_pass = sorted({(r["asset"], r["split"], r["horizon"]) for r in rows if r["candidate"] == "range" and r["gate"] == "PASS"})
    published = {}
    for m in a.manifests:
        d = json.load(open(b / m))
        published[Path(m).name] = {"variant": d["variant"], "valid_horizons": d.get("valid_horizons", d.get("valid_horizons_hours", "all (control)")),
                                    "canonical": d.get("manifest_sha256_canonical")}
    statement = {"variant_A_passes_anywhere": any(r["gate"] == "PASS" for r in variant_a),
                 "variant_A_rows": len(variant_a), "range_family_passing": [list(x) for x in range_pass],
                 "text": ("Under the declared linear (ridge) probe, variant A (all admissible inputs) never beats the zero-return naive on any asset, split or horizon; "
                          "the range family passes only at short or specific horizons listed in range_family_passing. This is a probe diagnostic, not the temporal model.")}
    if statement["variant_A_passes_anywhere"]:
        statement["text"] = "variant A passes somewhere: see rows (the plain statement does not hold)"
    doc = {"schema": "lane_b_probe_summary.v1", "rows": rows, "s1_s2_agreement": agreement(rows), "published_manifests": published, "statement": statement,
           "counts": {"rows": len(rows), "pass": sum(r["gate"] == "PASS" for r in rows), "fail": sum(r["gate"] == "FAIL" for r in rows),
                      "mse_not_stored": sum(r["gate"] == "MSE_NOT_STORED" for r in rows)},
           "inputs": {r["source"]: hashlib.sha256(open(b / r["source"], "rb").read()).hexdigest() for r in rows}}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"counts": doc["counts"], "variant_A_passes_anywhere": statement["variant_A_passes_anywhere"], "range_pass": len(range_pass)}))


if __name__ == "__main__":
    main()
