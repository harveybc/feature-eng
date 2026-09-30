#!/usr/bin/env python3
"""Turn one M03 TRAIN profile into an admissible-input declaration for M02/M04.

The declaration is mechanical: it copies the profile's column statuses, keeps every
exclusion with its reason, assigns one admissible feature per branch in file order, and
binds the resource, TRAIN prefix, manifest and profile digests. It never ranks, drops or
merges admissible features; that is the selection protocol's job, fitted inside training.
declaration_sha256 = SHA256 of the canonical JSON (sorted keys, no whitespace) without that field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "declaration_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def declare(profile_path: Path) -> dict:
    raw = profile_path.read_bytes()
    p = json.loads(raw)
    cols = p["columns"]
    admissible = [c["column"] for c in cols if c["status"] == "PROFILED_ADMISSIBLE"]
    doc = {
        "schema": "m03_admissible_inputs.v1",
        "dataset_id": p["dataset_id"], "governance": p["governance"], "lake": p.get("lake"),
        "resource": p.get("resource"), "resource_sha256": p["resource_identity"]["sha256"],
        "train_rows": p["train_prefix"]["rows"], "train_prefix_sha256": p["train_prefix"]["prefix_sha256"],
        "split_rule": p["train_prefix"]["split_rule"], "manifest_sha256": p["manifest_sha256"],
        "profile_sha256": hashlib.sha256(raw).hexdigest(), "profiler_sha256": p["implementation"]["sha256"],
        "sampling": {k: p["sampling"].get(k) for k in ("status", "verified_step_seconds", "median_step_seconds")},
        "target_channels": p["manifest"].get("target_channels", []),
        "admissible_count": len(admissible), "columns_total": len(cols),
        "branches_one_feature_each": [{"branch": i, "feature": f} for i, f in enumerate(admissible)],
        "all_admissible_control": admissible,
        "exclusions": [{"column": c["column"], "status": c["status"], "reason": c["exclusion_reason"]}
                       for c in cols if c["status"] != "PROFILED_ADMISSIBLE"],
        "admissibility_rule": ("role=feature AND every TRAIN cell numeric AND at least one finite value AND "
                               "not constant on TRAIN AND nonfinite fraction <= manifest max_missing_fraction"),
        "not_established": ["point-in-time availability", "upstream causality of derived columns",
                            "which features or groups help a model (inner validation decides)"],
        "authority": ("LAKE resource identity verified by digest; local profile, not a governed warehouse metric"
                      if p["governance"] == "LAKE_RESOURCE_IDENTITY" else "LOCAL development file; not governed"),
    }
    flags = []
    mpath = profile_path.parent / "metrics_long.csv"
    if mpath.is_file():
        import csv
        for r in csv.DictReader(mpath.open()):
            if r["metric"] == "min" and r["value"] and float(r["value"]) <= -9999:
                flags.append({"column": r["column"], "flag": "SENTINEL_LIKE_MINIMUM", "value": float(r["value"]),
                              "consequence": "kept admissible; a sentinel policy must be declared before fitting, not inferred here"})
    doc["quality_flags"] = flags
    doc["sampling_note"] = p["sampling"].get("consequence", "")
    doc["declaration_sha256"] = canonical_sha(doc)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("output exists; declarations are immutable, write a new file")
    doc = declare(a.profile)
    a.output.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"dataset_id": doc["dataset_id"], "admissible": doc["admissible_count"],
                      "excluded": len(doc["exclusions"]), "declaration_sha256": doc["declaration_sha256"]}))


if __name__ == "__main__":
    main()
