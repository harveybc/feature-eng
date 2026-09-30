#!/usr/bin/env python3
"""Audit existing JSON metadata and bounded profile artifacts, never raw datasets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), sha(raw)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matching_id(row):
    # Only this documented naming relationship is accepted, never fuzzy aliases.
    return ("financial_data.census_appearance." + row["id"]
            if row["kind"] == "physical_appearance" else row["id"])


def audit(index_path, inventory_path, receipt_paths, output, profile_path=None,
          artifact_byte_budget=128 << 20, per_artifact_cap=8 << 20):
    repo = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to(repo) or output.exists():
        raise ValueError("Output must be a new directory inside this worktree")
    index, index_hash = load(index_path)
    inventory, inventory_hash = load(inventory_path)
    entries, variable_rows, sources = [], [], []
    consumed = 0
    for receipt_path in receipt_paths:
        receipt, receipt_hash = load(receipt_path)
        sources.append({"receipt": receipt_path.parent.name + "/" + receipt_path.name,
                        "sha256": receipt_hash, "claimed_rows_total": receipt.get("rows_total")})
        for ds in receipt["datasets"]:
            rel = ds.get("file")
            path = (receipt_path.parent / rel).resolve() if rel else None
            if path and not path.is_relative_to(receipt_path.parent.resolve()):
                raise ValueError("Artifact path escapes receipt directory")
            exists = bool(path and path.is_file())
            row = {"dataset_id": ds["dataset_id"], "bank": ds["bank"],
                   "receipt": receipt_path.parent.name, "artifact": rel,
                   "receipt_status": ds["status"], "numeric_variables_claimed": ds.get("numeric_variables"),
                   "artifact_present": exists, "artifact_bytes": path.stat().st_size if exists else 0,
                   "verification": "MISSING_ARTIFACT" if not exists else "NOT_INSPECTED_RESOURCE_CAP"}
            if exists and row["artifact_bytes"] <= min(per_artifact_cap, artifact_byte_budget - consumed):
                raw = path.read_bytes()
                consumed += len(raw)
                row["verification"] = "HASH_VERIFIED" if sha(raw) == ds.get("sha256") else "HASH_MISMATCH"
                statuses, partitions, metrics = Counter(), Counter(), set()
                variables = defaultdict(lambda: {"metrics": set(), "statuses": Counter(), "ok": set()})
                for line in raw.splitlines():
                    r = json.loads(line)["row"]
                    if r.get("dataset_id") != ds["dataset_id"]:
                        row["verification"] = "DATASET_ID_MISMATCH"
                    statuses[r.get("status", "UNDECLARED")] += 1
                    partitions[r.get("partition", "UNDECLARED")] += 1
                    metric = r.get("metric", "UNDECLARED")
                    metrics.add(metric)
                    if r.get("partition") == "train" and r.get("variable_id"):
                        v = variables[r["variable_id"]]
                        v["metrics"].add(metric)
                        v["statuses"][r.get("status", "UNDECLARED")] += 1
                        if r.get("status") == "COMPLETED":
                            v["ok"].add(metric)
                row.update(metric_rows=sum(statuses.values()), metric_statuses=json.dumps(dict(statuses), sort_keys=True),
                           partitions=json.dumps(dict(partitions), sort_keys=True), unique_metrics=len(metrics),
                           train_variables_observed=len(variables))
                for vid, v in variables.items():
                    variable_rows.append({"dataset_id": ds["dataset_id"], "variable_id": vid,
                                          "artifact_verification": row["verification"],
                                          "train_metric_names": ";".join(sorted(v["metrics"])),
                                          "train_completed_metric_count": len(v["ok"]),
                                          "train_adf_pvalue_completed": "adf_pvalue" in v["ok"],
                                          "train_kpss_pvalue_completed": "kpss_pvalue" in v["ok"],
                                          "train_status_counts": json.dumps(dict(v["statuses"]), sort_keys=True)})
            entries.append(row)
    by_id = defaultdict(list)
    for e in entries:
        by_id[e["dataset_id"]].append(e)
    basic = {d["dataset_id"]: d for d in inventory["datasets"]}
    coverage = []
    for item in index["common_rows"]:
        matches = by_id.get(matching_id(item), [])
        status = "NO_MATCH_IN_AUDITED_RECEIPTS"
        if any(e["verification"] == "HASH_VERIFIED" for e in matches):
            status = "PROFILE_ARTIFACT_HASH_VERIFIED"
        elif any(e["artifact_present"] for e in matches):
            status = "PROFILE_PRESENT_NOT_VERIFIED"
        elif matches:
            status = "RECEIPT_ONLY_ARTIFACT_MISSING"
        coverage.append({"bank": item["bank"], "kind": item["kind"], "id": item["id"],
                         "profile_status": status,
                         "basic_summary_columns": len(basic.get(item["id"], {}).get("variables", [])),
                         "feature_completeness": "UNKNOWN: inventory-to-column identity/metric contract not established"})
    counts = defaultdict(Counter)
    for row in coverage:
        counts[row["bank"] + ":" + row["kind"]][row["profile_status"]] += 1
    flat = []
    for ds in inventory["datasets"]:
        for v in ds.get("variables", []):
            flat.append(dict(dataset_id=ds["dataset_id"], profile_scope="legacy full-dataset summary; NOT TRAIN-certified", **v))
    summary = {"schema": "feature_inventory_audit.v1", "index_sha256": index_hash,
               "inventory_sha256": inventory_hash, "receipt_sources": sources,
               "inventory_basic_datasets": len(basic), "inventory_basic_columns": len(flat),
               "inventory_detailed_train_certified_datasets": 0,
               "index_entries": len(coverage), "coverage_by_grain": dict(counts),
               "profile_receipt_entries": len(entries), "profile_unique_dataset_ids": len(by_id),
               "receipt_statuses": dict(Counter(e["receipt_status"] for e in entries)),
               "artifact_verification": dict(Counter(e["verification"] for e in entries)),
               "artifact_bytes_inspected": consumed, "artifact_byte_budget": artifact_byte_budget,
               "artifact_per_file_cap": per_artifact_cap,
               "observed_train_variables_in_inspected_artifacts": len(variable_rows),
               "profile_entries_by_bank": dict(Counter(e["bank"] for e in entries)),
               "unmatched_profile_ids": sorted(set(by_id) - {matching_id(r) for r in index["common_rows"]}),
               "caveats": ["No raw dataset input read by this audit",
                           "Completed job != completed metrics; see per-variable statuses",
                           "Missing means no match in these sources, not absent everywhere",
                           "Index grains are incompatible: do not sum as datasets or features",
                           "Generator-to-realization and public alias mappings are not inferred",
                           "Legacy profiles include non-TRAIN metric partitions; not selection-ready certificates",
                           "Full all-inventory per-feature completeness remains UNKNOWN"]}
    if profile_path:
        new, h = load(profile_path)
        summary["new_bounded_train_profile"] = {
            "sha256": h, "dataset_id": new["dataset_id"], "profile_range": new["profile_range"],
            "in_bank_index": any(r["id"] == new["dataset_id"] for r in index["common_rows"]),
            "columns": len(new["features"]), "profiled": sum(f["status"] == "PROFILED" for f in new["features"]),
            "branches": len(new["branches"]), "full_train_coverage": new["full_train_coverage"]}
    output.mkdir(parents=True)
    (output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output / "inventory_coverage.csv", coverage)
    write_csv(output / "existing_profile_artifacts.csv", entries)
    write_csv(output / "existing_train_variable_coverage.csv", variable_rows)
    write_csv(output / "legacy_basic_column_metrics.csv", flat)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--receipt", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    a = p.parse_args()
    summary = audit(a.index, a.inventory, a.receipt, a.output, a.profile)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("unmatched_profile_ids", "receipt_sources")}, indent=2))


if __name__ == "__main__":
    main()
