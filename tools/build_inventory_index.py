#!/usr/bin/env python3
"""M03 inventory coverage index: one row per known dataset x column, with an explicit status.

Metadata and receipts only. It computes no statistic of any data value. Reused profiles
count as measured only when all three identities verify:
  bytes           the profile artifact SHA256 equals the receipt and the terminal
  split           the receipt's contract SHA256 equals the sealed contract, whose own file
                  digest is bound by the host-sync manifest (financial, public) or recomputed
                  from the unit directory (synthetic)
  implementation  the receipt's composite code SHA256 recomputes from its per-file digests
Only TRAIN-partition rows of a reused artifact are ever read; the calibration and confirmation
rows those artifacts also hold are skipped by an exact partition filter and are never summarised.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

FIELDS = ["row_id", "source", "bank", "authority", "dataset_id", "column", "column_identity",
          "dataset_identity_sha256", "split_identity", "train_rows", "role", "status", "covered",
          "profile_path", "profile_sha256", "implementation_identity", "families_present",
          "families_absent", "missing_reason", "superseded_by", "family_states"]
FAMILIES = ("missingness", "distribution", "volatility", "trend", "acf", "spectral", "stationarity", "seasonality")
# c162 metric name -> family (anything not listed is recorded under "other")
C162_FAMILY = [("missing_count", "missingness"), ("non_finite_count", "missingness"), ("constant_flag", "missingness"),
               ("coverage", "missingness"), ("cardinality", "missingness"),
               ("quantile_", "distribution"), ("mad", "distribution"), ("iqr", "distribution"),
               ("range", "distribution"), ("tail_ratio", "distribution"), ("kurtosis", "distribution"),
               ("skewness", "distribution"), ("robust_z", "distribution"),
               ("acf_lag_", "acf"), ("correlation_time", "acf"),
               ("spectral_entropy", "spectral"), ("near_nyquist", "spectral"),
               ("adf_", "stationarity"), ("kpss_", "stationarity")]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def home_relative(p) -> str:
    """Never write an account's home path into a committed index."""
    return str(p).replace(str(Path.home()), "~", 1)


def c162_family(metric: str):
    for prefix, fam in C162_FAMILY:
        if metric.startswith(prefix) or prefix in metric:
            return fam
    return None


def load_sync_manifest(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split(None, 1)
        out[rel.strip()] = digest
    return out


def read_c162(state: Path, sync: dict, fin_contracts: dict, predictor: Path):
    """-> {dataset_id: {...verification..., 'vars': {variable_id: {completed, not_completed}}}}"""
    synth = None
    spec = importlib.util.spec_from_file_location("dsc", predictor / "tools/df_synthetic_contract.py")
    try:
        synth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(synth)
    except Exception as exc:                                   # recorded, never silent
        synth = exc
    out = {}
    for rdir in sorted(state.glob("profiles_c162_v1_*")):
        receipt = json.loads((rdir / "PROFILE_RUN_RECEIPT.json").read_text())
        composite = hashlib.sha256(json.dumps(receipt["code_file_sha256"], sort_keys=True).encode()).hexdigest()
        impl_ok = composite == receipt["code_sha256"]
        for d in receipt["datasets"]:
            art = rdir / d["file"]
            v = {"receipt": str(rdir.name), "artifact": art, "impl_ok": impl_ok,
                 "impl": f"c162 composite {receipt['code_sha256'][:16]} ({'recomputed' if impl_ok else 'MISMATCH'})",
                 "bank": d["bank"], "contract_sha256": d["contract_sha256"], "vars": {}, "names": {}}
            term = json.loads((rdir / d["terminal"]).read_text()) if (rdir / d["terminal"]).is_file() else {}
            v["bytes_ok"] = art.is_file() and sha_file(art) == d["sha256"] == term.get("output_sha256")
            v["artifact_sha256"] = d["sha256"]
            # split identity
            if d["bank"] == "FINANCIAL":
                c = fin_contracts["by_id"].get(d["dataset_id"])
                v["split_ok"] = bool(c and c["contract_sha256"] == d["contract_sha256"] and fin_contracts["file_ok"])
                if c:
                    v["names"] = {x["variable_id"]: (x["name"], x["original_fields"].get("census_variable_id"))
                                  for x in c["variables"]}
                    v["train_rows"] = c["partitions"]["boundaries"]["train"] if "partitions" in c else None
            elif d["bank"] == "PUBLIC":
                cdir = Path(d["job"].split(":", 1)[1].replace("~", str(Path.home())))
                cfile = cdir / "CONTRACT.json"
                rel = f"public_panels_c126_v2/{cdir.name}/CONTRACT.json"
                c = json.loads(cfile.read_text()) if cfile.is_file() else {}
                v["split_ok"] = bool(c) and c.get("contract_sha256") == d["contract_sha256"] and sync.get(rel) == sha_file(cfile)
                v["names"] = {x.get("variable_id"): (x["name"], None) for x in c.get("variables", [])}
                v["train_rows"] = c.get("partitions", {}).get("boundaries", {}).get("train")
                v["dataset_bytes"] = next((f.get("sha256") for f in c.get("files", []) if f.get("role") == "DERIVED_CANONICAL_PANEL"), "")
            else:
                udir = Path(d["job"].split(":", 1)[1].replace("~", str(Path.home())))
                try:
                    c = synth.unit_contract(udir)
                    v["split_ok"] = c.get("contract_sha256") == d["contract_sha256"]
                    v["names"] = {x.get("variable_id"): (x["name"], None) for x in c.get("variables", [])}
                    v["train_rows"] = c.get("partitions", {}).get("boundaries", {}).get("train")
                except Exception as exc:
                    v["split_ok"] = False
                    v["split_error"] = f"{type(exc).__name__}: {exc}"[:160]
            if v["bytes_ok"]:
                with art.open() as fh:
                    for line in fh:
                        if '"partition": "train"' not in line:
                            continue
                        row = json.loads(line)["row"]
                        if row.get("partition") != "train" or "variable_id" not in row:
                            continue
                        s = v["vars"].setdefault(row["variable_id"], {"completed": set(), "other": set()})
                        (s["completed"] if row["status"] == "COMPLETED" else s["other"]).add(row["metric"])
            v["skipped"] = d.get("skipped_variables", [])
            out[d["dataset_id"]] = v
    return out


def families(completed):
    present = sorted({c162_family(m) for m in completed} - {None})
    return present, [f for f in FAMILIES if f not in present]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictor", type=Path, required=True)
    ap.add_argument("--financial-contracts", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True, help="crispdm-data-foundation state directory")
    ap.add_argument("--new-profiles", type=Path, required=True, help="directory of M03 profile_train_wide outputs")
    ap.add_argument("--inherited-profile", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output must be a new directory")
    idx_bytes = (a.predictor / "examples/research/crispdm_bank_index.v1.json").read_bytes()
    idx = json.loads(idx_bytes)
    inv = json.loads((a.predictor / "examples/research/crispdm_dataset_inventory.v1.json").read_text())
    sync = load_sync_manifest(a.state / "host_sync_c161_v1/MANIFEST.sha256")
    fc_sha = sha_file(a.financial_contracts)
    fc = json.loads(a.financial_contracts.read_text())
    fin = {"file_ok": sync.get("financial_first_batch_c127_v1/FINANCIAL_FIRST_BATCH_CONTRACTS.v1.json") == fc_sha,
           "by_id": {c["dataset_id"]: c for c in fc["contracts"]}}
    c162 = read_c162(a.state, sync, fin, a.predictor)
    rows = []

    def add(**kw):
        r = {k: "" for k in FIELDS}
        r.update(kw)
        r["row_id"] = len(rows)
        r["covered"] = bool(kw.get("covered", False))
        rows.append(r)

    def reuse_row(v, var_id, name, base):
        s = v["vars"].get(var_id)
        ok = v["bytes_ok"] and v["split_ok"] and v["impl_ok"]
        if not ok:
            why = [k for k in ("bytes_ok", "split_ok", "impl_ok") if not v[k]]
            add(**base, status="REUSE_REFUSED_IDENTITY_UNVERIFIED", missing_reason="failed: " + ",".join(why),
                profile_path=home_relative(v["artifact"]), profile_sha256=v["artifact_sha256"])
            return
        if s is None:
            reason = "SKIPPED_BY_PRODUCER_NONNUMERIC" if name in v["skipped"] else "NO_TRAIN_ROWS_FOR_VARIABLE_IN_ARTIFACT"
            add(**base, status="NOT_MEASURED", missing_reason=reason, profile_path=home_relative(v["artifact"]),
                profile_sha256=v["artifact_sha256"], implementation_identity=v["impl"])
            return
        present, absent = families(s["completed"])
        add(**base, status="MEASURED_TRAIN_VERIFIED_REUSED", covered=True, profile_path=home_relative(v["artifact"]),
            profile_sha256=v["artifact_sha256"], implementation_identity=v["impl"],
            families_present=";".join(present), families_absent=";".join(absent),
            missing_reason=("absent families were not produced by the c162 implementation; "
                            f"{len(s['other'])} train metrics recorded non-COMPLETED status") if absent or s["other"] else "")

    # 1. financial census appearances x variables
    fb = idx["banks"]["financial_domain"]
    by_entity = collections.defaultdict(list)
    for var in fb["variables"]:
        by_entity[var["entity"]].append(var)
    for app in fb["appearances"]:
        ds = f"financial_data.census_appearance.{app['appearance_id']}"
        v = c162.get(ds)
        contract = fin["by_id"].get(ds)
        census_to_c162 = {}
        if v:
            census_to_c162 = {cv: vid for vid, (nm, cv) in v["names"].items()}
        for var in sorted(by_entity[app["entity"]], key=lambda x: x["concept_name"]):
            base = dict(source="bank_index.financial_appearance_x_variable", bank="FINANCIAL",
                        authority="FINANCIAL_DOMAIN_DEVELOPMENT_ONLY", dataset_id=ds, column=var["concept_name"],
                        column_identity=var["variable_id"], dataset_identity_sha256=app["physical_sha256"],
                        role="timestamp" if var["concept_name"].lower() in ("timestamp", "date", "datetime", "date_time") else "UNDECLARED")
            if contract is None:
                add(**base, split_identity="NONE", status="NOT_MEASURED",
                    missing_reason="NO_TRAIN_SPLIT_CONTRACT: appearance outside the 198 sealed C127 contracts; discovery is not permission to profile")
                continue
            base.update(split_identity=f"c127 contract {contract['contract_sha256'][:16]}",
                        train_rows=json.dumps(contract.get("partitions", {}).get("boundaries", {}).get("train")))
            if base["role"] == "timestamp":
                add(**base, status="EXCLUDED_ROLE", missing_reason="TIMESTAMP: time index, not a branch input")
                continue
            if v is None:
                add(**base, status="NOT_MEASURED", missing_reason="CONTRACTED_BUT_NO_C162_RECEIPT")
                continue
            vid = census_to_c162.get(var["variable_id"])
            if vid is None:
                add(**base, status="NOT_MEASURED", missing_reason="CENSUS_VARIABLE_NOT_IN_CONTRACT_VARIABLES")
                continue
            base["column_identity"] = f"{var['variable_id']}|c162:{vid[:16]}"
            reuse_row(v, vid, var["concept_name"], base)

    # 2. financial model-ready views (legacy inventory): summaries only, not TRAIN-certified
    profiled_ids = {json.loads(pj.read_text())["dataset_id"] for pj in a.new_profiles.glob("*/profile.json")}
    m03_shas = {}
    for mp in sorted((a.new_profiles.parent / "manifests").glob("*.v2.json")):
        mm = json.loads(mp.read_text())
        if mm["dataset_id"] in profiled_ids:              # supersede only when a full-TRAIN profile exists
            m03_shas[mm["resource_sha256"]] = mm["dataset_id"]
    for ds in inv["datasets"]:
        holdout = "phase1_test" in ds["dataset_id"]
        sup = m03_shas.get(ds["physical_sha256"]) if not holdout else None
        for var in ds["variables"]:
            if sup:
                add(source="dataset_inventory.model_ready_view", bank="FINANCIAL", authority="FINANCIAL_DOMAIN_DEVELOPMENT_ONLY",
                    dataset_id=ds["dataset_id"], column=var["name"], dataset_identity_sha256=ds["physical_sha256"],
                    split_identity="CORRECTION: TRAIN contract exists (predictor 14a1077f)", role=var.get("role", ""),
                    status="SUPERSEDED", superseded_by=sup,
                    missing_reason=("CORRECTION 2026-10-01: v2 said 'no TRAIN boundary is declared for this view'; WRONG - the "
                                    "immutable manifest 14a1077f declares train 2017-09-28..2023-12-31; this row is replaced by "
                                    f"the full-TRAIN row of {sup}"))
                continue
            add(source="dataset_inventory.model_ready_view", bank="FINANCIAL", authority="FINANCIAL_DOMAIN_DEVELOPMENT_ONLY",
                dataset_id=ds["dataset_id"], column=var["name"], dataset_identity_sha256=ds["physical_sha256"],
                split_identity="NONE_DECLARED", role=var.get("role", ""),
                status="EXCLUDED_HOLDOUT_FILE" if holdout else "NOT_MEASURED",
                profile_path="predictor examples/research/crispdm_dataset_inventory.v1.json",
                missing_reason=("legacy TEST file: never profiled for selection" if holdout else
                                "only a whole-file basic summary exists; no TRAIN boundary is declared for this view"))

    # 3. public T2 forecasting bank series
    pb = idx["banks"]["public_forecasting"]
    dsets = {d["dataset_id"]: d for d in pb["datasets"]}
    for s in pb["series"]:
        d = dsets.get(s["dataset_id"], {})
        add(source="bank_index.public_series", bank="PUBLIC", authority="PUBLIC_FORECASTING_EVIDENCE",
            dataset_id=s["dataset_id"], column=s["series_id"], dataset_identity_sha256=d.get("bytes_sha256", s.get("digest", "")),
            split_identity="NONE_IN_INDEX", status="NOT_MEASURED", role="series",
            missing_reason="NO_TRAIN_BOUNDARY_DECLARED: T2 bank index carries no per-series TRAIN contract; bank adjudicated by T2 screen, profiling it needs a sealed split first")

    # 4. synthetic T1 generators in the index
    for g in idx["banks"]["synthetic_known_mechanism"]["generators"]:
        add(source="bank_index.synthetic_generator", bank="SYNTHETIC", authority="SYNTHETIC_KNOWN_MECHANISM_CALIBRATION_ONLY",
            dataset_id=g["generator_id"], column="observed_signal",
            dataset_identity_sha256=g.get("reconstruction", {}).get("observed_signal_sha256", ""),
            split_identity="NONE", status="NOT_MEASURED", role="calibration_series",
            missing_reason="CALIBRATION_ONLY_T1_GENERATOR: no realization contract; no equivalence to the c128 units is asserted")

    # 5/6. c162-profiled public panels (c126) and synthetic units (c128) outside the index
    for ds, v in sorted(c162.items()):
        if v["bank"] == "FINANCIAL":
            continue
        source = "public_panels_c126_v2" if v["bank"] == "PUBLIC" else "synthetic_bank_c128_v1"
        for vid, (name, _) in sorted(v["names"].items(), key=lambda kv: kv[1][0]):
            base = dict(source=source, bank=v["bank"],
                        authority="PUBLIC_DEVELOPMENT_PANEL" if v["bank"] == "PUBLIC" else "SYNTHETIC_CALIBRATION_ONLY",
                        dataset_id=ds, column=name, column_identity=f"c162:{(vid or '')[:16]}",
                        dataset_identity_sha256=v.get("dataset_bytes", ""),
                        split_identity=f"contract {v['contract_sha256'][:16]}" + ("" if v["split_ok"] else " UNVERIFIED"),
                        train_rows=json.dumps(v.get("train_rows")), role="variable")
            reuse_row(v, vid, name, base)

    # 7/8. M03 new wide profiles (lake + local) and the inherited 512-row prefix
    new = {}
    for pj in sorted(a.new_profiles.glob("*/profile.json")):
        p = json.loads(pj.read_text())
        new[p["dataset_id"]] = (pj, p)
    manifests = sorted((a.new_profiles.parent / "manifests").glob("*.v2.json"))
    for mp in manifests:
        m = json.loads(mp.read_text())
        if m["dataset_id"] in new:
            pj, p = new[m["dataset_id"]]
            psha = sha_file(pj)
            fstate = {}
            ml = pj.parent / "metrics_long.csv"
            if ml.is_file():
                acc = collections.defaultdict(list)
                with ml.open() as fh:
                    for r in csv.DictReader(fh):
                        acc[(r["column"], r["family"])].append(r["status"])
                for (col, fam), st in acc.items():
                    ok = [x in ("OK", "OK_WITH_WARNING") for x in st]
                    fstate.setdefault(col, {})[fam] = ("COMPLETE" if all(ok) else "FAILED" if "FAILED" in st
                                                       else "PARTIAL" if any(ok) else "NOT_RUN")
            for c in p["columns"]:
                fam_ok = FAMILIES if c["status"] == "PROFILED_ADMISSIBLE" else ()
                add(source=f"m03.{m['governance'].lower()}", bank="PUBLIC_BENCHMARK" if m["governance"] != "LOCAL_FILE" else "FINANCIAL",
                    authority=m.get("lake") or "LOCAL_DEVELOPMENT", dataset_id=m["dataset_id"], column=c["column"],
                    dataset_identity_sha256=p["resource_identity"]["sha256"],
                    split_identity=f"TRAIN {p['train_prefix']['rows']} prefix {p['train_prefix']['prefix_sha256'][:16]}",
                    train_rows=json.dumps(p["train_prefix"]["rows"]), role=c["role"],
                    status={"PROFILED_ADMISSIBLE": "MEASURED_TRAIN_NEW", "PROFILED_EXCLUDED": "MEASURED_TRAIN_NEW_EXCLUDED",
                            "EXCLUDED": "EXCLUDED_ROLE"}[c["status"]],
                    covered=c["status"] in ("PROFILED_ADMISSIBLE", "PROFILED_EXCLUDED"),
                    profile_path=os.path.relpath(pj, Path(__file__).resolve().parents[1]), profile_sha256=psha,
                    implementation_identity=f"profile_train_wide {p['implementation']['sha256'][:16]} "
                                            + json.dumps(p["implementation"]["versions"], sort_keys=True),
                    families_present=";".join(fam_ok) if fam_ok else "",
                    families_absent="" if fam_ok else ";".join(FAMILIES),
                    missing_reason=c.get("exclusion_reason") or ("see metrics_long.csv for per-metric NOT_RUN/FAILED rows"
                                                                  if fam_ok else ""),
                    family_states=json.dumps(fstate.get(c["column"], {}), sort_keys=True))
        else:
            header = m.get("columns_total")
            add(source=f"m03.{m['governance'].lower()}", bank="PUBLIC_BENCHMARK" if m["governance"] != "LOCAL_FILE" else "FINANCIAL",
                authority=m.get("lake") or "LOCAL_DEVELOPMENT", dataset_id=m["dataset_id"], column=f"<all {header} columns>",
                dataset_identity_sha256=m["resource_sha256"], split_identity=f"TRAIN {m['boundaries']['train']} (declared)",
                status="NOT_MEASURED_QUEUED", missing_reason="profile run pending admission; row expands per column when it lands")
    gp = json.loads(a.inherited_profile.read_text())
    gsha = sha_file(a.inherited_profile)
    for f in gp["features"]:
        add(source="inherited.gibbs_dce037b.train_512", bank="FINANCIAL", authority="LOCAL_DEVELOPMENT",
            dataset_id=gp["dataset_id"] + ".prefix512", column=f["column"], dataset_identity_sha256=gp["consumed_input_sha256"],
            split_identity="TRAIN rows [0,512) of [0,7656)", train_rows="[0, 512]", role=f["role"],
            status="PARTIAL_PREFIX_VERIFIED_REUSED" if f["role"] == "feature" else "EXCLUDED_ROLE", covered=False,
            superseded_by="predictor.legacy.eurusd.phase1b.normalized_d4.train (full TRAIN)",
            profile_path="docs/feature_metrics/evidence/train_512/profile.json", profile_sha256=gsha,
            implementation_identity=f"profile_train_features {gp['source_code_sha256'][:16]}",
            missing_reason="512 of 7656 TRAIN rows: verified replay, but not full-TRAIN coverage; superseded by the M03 full-TRAIN row when present"
                           if f["role"] == "feature" else (f.get("exclusion_reason") or f["role"]))

    a.output.mkdir(parents=True)
    with (a.output / "coverage_index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    by_status = collections.Counter(r["status"] for r in rows)
    by_source = collections.defaultdict(collections.Counter)
    for r in rows:
        by_source[r["source"]][r["status"]] += 1
    reuse = collections.Counter()
    for v in c162.values():
        reuse[("bytes_ok" if v["bytes_ok"] else "bytes_FAIL", "split_ok" if v["split_ok"] else "split_FAIL",
               "impl_ok" if v["impl_ok"] else "impl_FAIL")] += 1
    distinct = [r for r in rows if not r["superseded_by"]]
    fam_cells = collections.Counter()
    for r in distinct:
        if r["family_states"]:
            st = json.loads(r["family_states"])
            for f in FAMILIES:
                fam_cells[(f, st.get(f, "ABSENT"))] += 1
        elif r["status"] == "MEASURED_TRAIN_VERIFIED_REUSED":
            pres = set(filter(None, r["families_present"].split(";")))
            for f in FAMILIES:
                fam_cells[(f, "PRESENT_REUSED" if f in pres else "ABSENT_IN_REUSED_PROFILE")] += 1
        else:
            for f in FAMILIES:
                fam_cells[(f, "NOT_MEASURED:" + r["status"])] += 1
    by_family = {}
    for (f, st), n in fam_cells.items():
        by_family.setdefault(f, {})[st] = n
    summary = {"schema": "m03_inventory_coverage.v2",
               "denominator_distinct_rows": len(distinct),
               "covered_distinct_rows": sum(r["covered"] for r in distinct),
               "superseded_rows": len(rows) - len(distinct),
               "family_denominator_cells": len(distinct) * len(FAMILIES),
               "families": list(FAMILIES), "by_family_state": by_family,
               "acceptance_denominator_note": ("MS14 acceptance uses distinct dataset x column rows; FS15 acceptance uses "
                                               "rows x families (a covered row with ACF but without stationarity is not "
                                               "complete in that family); superseded rows count in neither"),
               "denominator_rows": len(rows), "covered_rows": sum(r["covered"] for r in rows),
               "covered_definition": "a feature column with a TRAIN-only profile over its full declared TRAIN population whose bytes, split and implementation identities verify (new or reused); excluded-role and partial-prefix rows are accounted but not covered",
               "by_status": dict(by_status), "by_source": {k: dict(v) for k, v in by_source.items()},
               "c162_artifact_verification": {"|".join(k): n for k, n in reuse.items()},
               "inputs": {"bank_index_sha256": hashlib.sha256(idx_bytes).hexdigest(),
                          "financial_contracts_sha256": fc_sha, "financial_contracts_bound_by_c161_sync": fin["file_ok"],
                          "inherited_profile_sha256": gsha},
               "rule": "metadata only; no data value read here; reused artifacts filtered to partition == train"}
    (a.output / "coverage_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("denominator_rows", "denominator_distinct_rows", "covered_distinct_rows",
                                              "superseded_rows", "family_denominator_cells", "by_status")}))


if __name__ == "__main__":
    main()
