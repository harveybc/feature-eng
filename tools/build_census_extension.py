#!/usr/bin/env python3
"""Census successor entries for files the census never listed, one provider batch per run.

Uses financial-data's own census module for identity (appearance_id over logical coordinates)
and digests (sha256_file), so a successor entry is shaped like a census appearance. Columns,
row counts and the first timestamp come from parquet footers (row-group statistics); no data
value is read. Provenance is the nearest provenance.json `source` (following a Stage 2.1
`source_dir`), else UNRESOLVED, never inferred from a name. The census artifact itself is not
modified: entries are written as an additive extension beside it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq


def load_census_module(root: Path):
    spec = importlib.util.spec_from_file_location("incremental_census", root / "_scripts/lib/incremental_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_timestamp(path: Path):
    """Minimum of the timestamp column from row-group statistics, or UNKNOWN."""
    try:
        md = pq.ParquetFile(path).metadata
        names = [md.schema.column(i).name for i in range(md.num_columns)]
        for cand in ("timestamp", "date", "DATE_TIME", "open_time"):
            if cand in names:
                j = names.index(cand)
                mins = []
                for g in range(md.num_row_groups):
                    st = md.row_group(g).column(j).statistics
                    if st is not None and st.has_min_max:
                        mins.append(st.min)
                if mins:
                    v = min(mins)
                    return v.isoformat() if hasattr(v, "isoformat") else str(v)
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def coordinates(rel: str):
    parts = rel.split("/")
    stem = Path(rel).stem
    if rel.startswith("features/trading_asset_features/") and len(parts) >= 5:
        return "trading_asset_derived", f"{parts[2]}__{stem}", parts[3]
    if rel.startswith("features/learned_inputs/") and len(parts) >= 5:
        return "learned_input", f"{parts[2]}__{stem}", parts[3]
    if rel.startswith("features/cross_source_statistical/") and len(parts) >= 4:
        return "cross_source_statistical", stem, parts[2]
    if rel.startswith("features/cross_source_features/") and len(parts) >= 4:
        return "cross_source", stem, parts[2]
    return "raw_source", "__".join(parts[:-1]), stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="financial-data checkout (read-only)")
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--provider", required=True, help="provider batch to build ('UNRESOLVED' allowed)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    ic = load_census_module(root)
    rows = [r for r in csv.DictReader(open(a.catalogue))
            if r["path"] and r["in_census"] == "False" and r["provider"] == a.provider]
    entries = []
    for r in rows:
        p = root / r["path"]
        sc, entity, freq = coordinates(r["path"])
        start = first_timestamp(p) if p.suffix == ".parquet" else "UNKNOWN"
        st = p.stat()
        cols, nrows = [], None
        if p.suffix == ".parquet":
            md = pq.ParquetFile(p).metadata
            cols = [md.schema.column(i).name for i in range(md.num_columns)]
            nrows = md.num_rows
        entries.append({
            "appearance_id": ic.appearance_id(sc, entity, freq, str(start)),
            "source_class": sc, "entity": entity, "frequency": freq, "relative_path": r["path"],
            "presence": "PRESENT", "period_start": start, "period_end": "UNKNOWN",
            "declared_rows": nrows, "declared_columns": cols, "size_bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
            "physical_sha256": ic.sha256_file(p), "digest_state": "PHYSICALLY_DIGESTED",
            "bytes_read_for_digest": st.st_size, "kind": r["kind"],
            "provenance": {"source": a.provider} if a.provider != "UNRESOLVED" else ic.UNAVAILABLE,
            "provenance_state": "DECLARED_IN_PROVENANCE_JSON" if a.provider != "UNRESOLVED" else "UNRESOLVED",
            "profile_depth": ic.PROFILE_DECLARED if hasattr(ic, "PROFILE_DECLARED") else "DECLARED_SCHEMA_ONLY",
            "availability": "UNAVAILABLE (no availability contract instance)", "train_contract": "NONE"})
    doc = {"schema": "financial_data.census_extension_batch.v1", "provider": a.provider,
           "derived_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "census_module_sha256": hashlib.sha256((root / "_scripts/lib/incremental_census.py").read_bytes()).hexdigest(),
           "extends": "features/census/artifacts/census-49a8813d… (unchanged)",
           "rule": "identity from logical coordinates via incremental_census.appearance_id; digests computed now; footers only; provenance from provenance.json or UNRESOLVED",
           "entries": entries, "count": len(entries), "bytes_digested": sum(e["size_bytes"] for e in entries)}
    doc["batch_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    safe = "".join(ch if ch.isalnum() else "_" for ch in a.provider)
    out = Path(a.out_dir) / f"CENSUS_EXTENSION.batch_{safe}.json"
    out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps({"provider": a.provider, "count": len(entries), "bytes": doc["bytes_digested"], "batch_sha256": doc["batch_sha256"]}))


if __name__ == "__main__":
    main()
