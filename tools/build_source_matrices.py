#!/usr/bin/env python3
"""Build the source entitlement-to-use matrix, the file catalogue and the transform-family ledger.

Inputs are metadata only: the discovery snapshot (paths, sizes, footers), the census
artifact, the C127 contracts, the lane B inventory v3 index, the data-gov calendar
registrations and the curated provider evidence. Outputs keep old and new denominators
side by side. No data value is read.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

LADDER = ("OWNER_REPORTED", "DOCUMENTED_ENTITLEMENT", "FUNCTIONING_CONNECTOR", "RETAINED_BYTES",
          "POINT_IN_TIME_ADMISSIBLE", "PROFILED", "EVALUATED")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--contracts", required=True)
    ap.add_argument("--inventory-index", required=True)
    ap.add_argument("--registrations", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True, help="financial-data checkout, for provenance.json metadata only")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    disc = json.load(open(a.discovery))["files"]
    census = json.load(open(a.census))
    contracts = json.load(open(a.contracts))["contracts"]
    ev = json.load(open(a.evidence))["providers"]
    regs = json.load(open(a.registrations))

    # census joins: entity -> providers, path -> appearance
    ent_prov = collections.defaultdict(set)
    dir_prov = {}
    for v in census["variables"]:
        decl = v.get("lineage", {}).get("source_declarations") or ["UNRESOLVED"]
        ent_prov[v["entity"]].update(decl)
        for d in v.get("lineage", {}).get("upstream_source_dirs", [])[:1]:
            dir_prov.setdefault(d, set()).update(decl)
    apps = {x["relative_path"]: x for x in census["appearances"]}
    contracted = {c["original_fields"]["census_appearance"]["relative_path"]: c for c in contracts}
    covered_ds = collections.Counter()
    with open(a.inventory_index) as fh:
        for r in csv.DictReader(fh):
            if r["covered"] == "True":
                covered_ds[r["dataset_id"]] += 1

    root = Path(a.root)
    prov_cache = {}

    def provenance_source(dir_rel):
        """Walk up from a directory to the nearest provenance.json; follow a Stage 2.1 source_dir."""
        parts = dir_rel.split("/")
        for k in range(len(parts), 0, -1):
            d = "/".join(parts[:k])
            if d in prov_cache:
                return prov_cache[d]
            f = root / d / "provenance.json"
            if f.is_file():
                try:
                    doc = json.loads(f.read_text())
                except Exception:
                    doc = {}
                src = doc.get("source")
                if not src and doc.get("source_dir"):
                    src = provenance_source(doc["source_dir"])
                prov_cache[d] = src if src and src != "UNKNOWN" else None
                return prov_cache[d]
        return None

    def provider_of(row):
        p = row["path"]
        parts0 = p.split("/")
        if p.startswith(("features/trading_asset_data/", "features/trading_asset_features/", "features/learned_inputs/")):
            src = provenance_source(f"features/trading_asset_data/{parts0[2]}")
            if src:
                return src
        if not p.startswith("features/"):
            src = provenance_source("/".join(parts0[:-1]))
            if src:
                return src
        if p in apps:
            return "|".join(sorted(ent_prov.get(apps[p]["entity"], {"UNRESOLVED"})))
        parts = p.split("/")
        if p.startswith("features/trading_asset_features/") or p.startswith("features/learned_inputs/"):
            asset, freq = parts[2], parts[3] if len(parts) > 4 else ""
            parent = f"features/trading_asset_data/{asset}/{freq}.parquet"
            if parent in apps:
                return "|".join(sorted(ent_prov.get(apps[parent]["entity"], {"UNRESOLVED"})))
            return "|".join(sorted(ent_prov.get(asset, {"UNRESOLVED"})))
        if p.startswith("features/cross_source_statistical/"):
            ent = Path(parts[-1]).stem
            return "|".join(sorted(ent_prov.get(ent, {"UNRESOLVED"})))
        best = None
        for d, prov in dir_prov.items():
            if p.startswith(d + "/") and (best is None or len(d) > len(best[0])):
                best = (d, prov)
        if best:
            return "|".join(sorted(best[1]))
        top = "/".join(parts[:2])
        guess = {"alternative_data/cryptoquant": "CryptoQuant", "alternative_data/cot_reports": "CFTC",
                 "alternative_data/short_interest": "FINRA / short interest"}.get(top)
        return guess or "UNRESOLVED"

    cat = []
    for d in disc:
        prov = provider_of(d)
        app = apps.get(d["path"])
        c = contracted.get(d["path"])
        ds = f"financial_data.census_appearance.{app['appearance_id']}" if app else ""
        status = ("PROFILED" if covered_ds.get(ds) else "RETAINED_BYTES")
        cov = "COVERED" if covered_ds.get(ds) else "UNCOVERED"
        if covered_ds.get(ds):
            missing = "evaluate on the business targets (PS2/PS5); availability contract still required for PIT use"
        elif c:
            missing = "TRAIN contract exists; profile TRAIN (c162 profile missing for this variable set) and an availability contract"
        elif app:
            missing = "in census, no TRAIN contract: seal one (see EURUSD request pattern) and an availability contract"
        elif d["kind"] == "DERIVED":
            missing = "outside the census: link producer+recipe+parent bytes (FEATURE_DAG.v3), method id, temporal tests (lane C), TRAIN contract"
        else:
            missing = "raw file outside the census: census entry, provenance, TRAIN contract, availability contract"
        cat.append({"path": d["path"], "kind": d["kind"], "family": d.get("family"), "asset": d.get("asset", ""),
                    "frequency": d.get("frequency", ""), "bytes": d["bytes"], "columns": len(d.get("columns", [])),
                    "num_rows": d.get("num_rows", ""), "provider": prov, "in_census": bool(app),
                    "census_appearance_id": app["appearance_id"] if app else "", "train_contract": bool(c),
                    "lane_b_covered_columns": covered_ds.get(ds, 0), "status": status, "coverage": cov,
                    "missing_action": missing})
    # declared sources with no bytes become rows too
    seen = {p for r in cat for p in r["provider"].split("|")}
    for e in ev:
        names = e["census_names"] or [e["provider"]]
        if not any(n in seen for n in names):
            cat.append({"path": "", "kind": "SOURCE_WITHOUT_BYTES", "family": "", "asset": "", "frequency": "",
                        "bytes": 0, "columns": 0, "num_rows": "", "provider": e["provider"], "in_census": False,
                        "census_appearance_id": "", "train_contract": False, "lane_b_covered_columns": 0,
                        "status": "OWNER_REPORTED" if e["owner_reported"] else "NOT_PRESENT", "coverage": "UNCOVERED",
                        "missing_action": e["missing_action"]})
    write_csv(out / "CATALOGUE.v1.csv", cat, list(cat[0].keys()))

    # source table per provider
    table = []
    for e in ev:
        names = e["census_names"] or [e["provider"]]
        rows = [r for r in cat if any(n in r["provider"].split("|") for n in names) and r["kind"] != "SOURCE_WITHOUT_BYTES"]
        if e["provider"] == "CryptoQuant":
            rows = [r for r in cat if r["path"].startswith(("alternative_data/cryptoquant", "features/cross_source_features"))
                    and "cryptoquant" in r["path"]]
        appsx = [apps[r["path"]] for r in rows if r["path"] in apps]
        freqs = sorted({r["frequency"] for r in rows if r["frequency"]})
        start = min((x["period_start"] for x in appsx), default="")
        end = max((x["period_end"] for x in appsx), default="")
        prof = sum(r["lane_b_covered_columns"] for r in rows)
        status = "PROFILED" if prof else ("RETAINED_BYTES" if rows else
                                          ("FUNCTIONING_CONNECTOR" if "EXECUTION ONLY" in e["connector"] else
                                           "OWNER_REPORTED" if e["owner_reported"] else "NOT_PRESENT"))
        pit = "NOT_ADMISSIBLE: no availability contract instance (census availability UNAVAILABLE; split contracts are not availability)"
        fams = collections.Counter()
        for r in rows:
            fams[r["family"]] += 1
        assets = sorted({(r["asset"] or Path(r["path"]).stem) for r in rows})[:60]
        rungs = {"OWNER_REPORTED": bool(e["owner_reported"]),
                 "DOCUMENTED_ENTITLEMENT": not e["entitlement_evidence"].startswith(("NONE", "UNKNOWN")),
                 "FUNCTIONING_CONNECTOR": bool(rows) or "EXECUTION ONLY" in e["connector"],
                 "RETAINED_BYTES": bool(rows), "POINT_IN_TIME_ADMISSIBLE": False, "PROFILED": bool(prof), "EVALUATED": False}
        table.append({"provider": e["provider"], "product": e["product"], "rungs": rungs,
                      "rung_gaps": [k for k, v in rungs.items() if not v], "asset_universe_count": len({r['asset'] for r in rows if r['asset']}) or len(assets),
                      "asset_universe_sample": assets[:25], "field_families": dict(fams.most_common(12)), "frequencies": freqs,
                      "coverage_dates": [start, end] if start else "not in census",
                      "entitlement_evidence": e["entitlement_evidence"], "owner_reported": e["owner_reported"],
                      "connector": e["connector"], "lake_files": len(rows), "lake_bytes": sum(r["bytes"] for r in rows),
                      "raw_files": sum(r["kind"] == "RAW" for r in rows), "derived_files": sum(r["kind"] == "DERIVED" for r in rows),
                      "census_appearances": len(appsx), "train_contracts": sum(r["train_contract"] for r in rows),
                      "clocks": e["clocks"], "revision_policy": e["revision_policy"], "license": e["license"],
                      "point_in_time": pit, "lane_b_profiled_columns": prof, "status": status,
                      "evaluated": "NO", "missing_action": e["missing_action"]})
    unres = [r for r in cat if r["provider"] == "UNRESOLVED"]
    table.append({"provider": "UNRESOLVED", "product": "files whose provider is not declared in census lineage",
                  "lake_files": len(unres), "lake_bytes": sum(r["bytes"] for r in unres),
                  "field_families": dict(collections.Counter(r["family"] for r in unres).most_common(12)),
                  "status": "RETAINED_BYTES", "missing_action": "locate provenance.json or producer; until then no use",
                  "evaluated": "NO"})
    (out / "SOURCE_TABLE.v1.json").write_text(json.dumps({"schema": "lane_b_source_table.v1", "ladder": LADDER,
                                                         "rows": table}, indent=1) + "\n")
    write_csv(out / "SOURCE_TABLE.v1.csv", table, ["provider", "product", "status", "rung_gaps", "lake_files", "lake_bytes", "raw_files",
                                                   "derived_files", "census_appearances", "train_contracts",
                                                   "lane_b_profiled_columns", "evaluated", "frequencies", "coverage_dates",
                                                   "asset_universe_count", "field_families", "entitlement_evidence",
                                                   "owner_reported", "connector", "clocks", "revision_policy", "license",
                                                   "point_in_time", "missing_action"])

    # FXMacroData: the eight census appearances against current contracts and the registry
    reg_in_force = regs.get("in_force") or regs.get("rows_in_force") or []
    fxm = []
    for c in contracts:
        ca = c["original_fields"]["census_appearance"]
        if "fxmacrodata" not in ca["entity"]:
            continue
        raw = ("economic_calendar/release_actuals/fxmacrodata/announcements.parquet" if "announcements" in ca["entity"]
               else "economic_calendar/scheduled_events/fxmacrodata/release_calendar.parquet")
        fxm.append({"appearance_id": ca["appearance_id"], "entity": ca["entity"], "frequency": ca["frequency"],
                    "relative_path": ca["relative_path"], "physical_sha256": ca["physical_sha256"],
                    "census_availability": "UNAVAILABLE (census 2026-09-10)",
                    "train_contract": c["contract_sha256"], "train_boundaries": c["partitions"]["boundaries"],
                    "raw_parent": raw,
                    "raw_parent_registry": ("OBSERVED_ACTUAL_PUBLICATION; absences incl. NO_CONSENSUS_AT_ALL, NO_AVAILABILITY_CONTRACT, NO_REVISION_HISTORY"
                                            if "announcements" in raw else
                                            "ASSUMED_SCHEDULED_PUBLICATION; forward schedule with 0 values; NO_AVAILABILITY_CONTRACT"),
                    "resolution": ("STALE_STRING_SUPERSEDED_FOR_THE_RAW_PARENT_BUT_DERIVED_STILL_UNAVAILABLE: the parent's "
                                   "publication clock is now a registered fact (data-gov registry 2026-09-26), yet this "
                                   "resampled appearance's producer is not located, so no derived earliest-available time "
                                   "exists; the TRAIN contract is a split, not availability"),
                    "not_invented": "no consensus and no first-release vintage exist for this source; none is assumed"})
    (out / "FXMACRODATA_AVAILABILITY_RESOLUTION.v1.json").write_text(json.dumps(
        {"schema": "lane_b_fxmacrodata_resolution.v1", "registrations_sha256": sha_file(a.registrations),
         "records": fxm, "count": len(fxm)}, indent=1) + "\n")

    # denominators side by side
    den = {"old": {"grain": "dataset x column rows (inventory v3)", "distinct_rows": 15228, "covered": 3538,
                   "source": "feature-eng inventory_v3 coverage_summary.json sha 884b595e"},
           "new": {"grain": "discovered files in raw and derived directories + declared sources without bytes",
                   "files": sum(r["kind"] in ("RAW", "DERIVED") for r in cat),
                   "raw_files": sum(r["kind"] == "RAW" for r in cat), "derived_files": sum(r["kind"] == "DERIVED" for r in cat),
                   "sources_without_bytes": sum(r["kind"] == "SOURCE_WITHOUT_BYTES" for r in cat),
                   "in_census": sum(r["in_census"] for r in cat), "outside_census": sum((not r["in_census"]) and r["kind"] != "SOURCE_WITHOUT_BYTES" for r in cat),
                   "with_train_contract": sum(r["train_contract"] for r in cat),
                   "with_lane_b_profiled_columns": sum(bool(r["lane_b_covered_columns"]) for r in cat),
                   "column_slots": sum(r["columns"] for r in cat),
                   "covered_files": sum(r["coverage"] == "COVERED" for r in cat)},
           "note": "the two grains are not commensurable: a file is not a column and a column appearance is not an economic signal; both are shown, neither replaces the other"}
    (out / "DENOMINATORS.v1.json").write_text(json.dumps(den, indent=1) + "\n")
    print(json.dumps(den["new"]))


if __name__ == "__main__":
    main()
