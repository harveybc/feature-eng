"""Resolve source and transform provenance for UNRESOLVED catalogue files from git-tracked metadata only
(provenance.json `source`, walking up the directory the entity name encodes). Run inside a financial-data
checkout; no data file is opened. A file that cannot be resolved keeps UNRESOLVED with the exact missing fact."""
import csv, json, subprocess, sys, collections, hashlib
CAT, OUT = sys.argv[1], sys.argv[2]
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n"))
prov_dirs = {p.rsplit("/", 1)[0] for p in tracked if p.endswith("provenance.json")}
cache = {}
def source_of(d):
    if d not in cache:
        try:
            doc = json.loads(subprocess.run(["git", "show", f"HEAD:{d}/provenance.json"], capture_output=True, text=True).stdout)
            cache[d] = doc.get("source")
        except Exception:
            cache[d] = None
    return cache[d]
TRANSFORM = {"features/cross_source_features": "stage21_cross_source_worker.py (resample/align of the raw source onto the trading grid)",
             "features/cross_source_statistical": "stage22_cross_source_stats_worker.py (rolling statistics of the cross-source series)"}
rows = []
for r in csv.DictReader(open(CAT)):
    if r["provider"] != "UNRESOLVED" or not r["path"]:
        continue
    parts = r["path"].split("/")
    fam = "/".join(parts[:2]); entity = parts[-1].rsplit(".", 1)[0]
    segs = entity.split("__")
    found, why = None, None
    for k in range(len(segs), 0, -1):
        d = "/".join(segs[:k])
        if d in prov_dirs:
            s = source_of(d)
            if s and s != "UNKNOWN":
                found = (d, s)
            else:
                why = f"provenance.json at {d} carries no source (value: {s!r})"
            break
    if not found and not why:
        why = f"no provenance.json in any ancestor of {'/'.join(segs)} (entity name encodes the source path by convention)"
    ch = max(0, int(r["columns"] or 0) - 1)
    rows.append({"path": r["path"], "family": fam, "entity": entity, "channels": ch,
                 "source_provider": found[1] if found else "UNRESOLVED", "source_provenance_dir": found[0] if found else "",
                 "transform_producer": "financial-data _scripts/workers/" + TRANSFORM.get(fam, "UNLOCATED"),
                 "state": "RESOLVED_FROM_PROVENANCE" if found else "DEFERRED_MISSING_FACT", "missing_fact": "" if found else why})
c = collections.Counter((x["state"], x["source_provider"]) for x in rows)
summ = {"files": len(rows), "channels": sum(x["channels"] for x in rows),
        "resolved_files": sum(x["state"] == "RESOLVED_FROM_PROVENANCE" for x in rows),
        "resolved_channels": sum(x["channels"] for x in rows if x["state"] == "RESOLVED_FROM_PROVENANCE"),
        "deferred_files": sum(x["state"] != "RESOLVED_FROM_PROVENANCE" for x in rows),
        "deferred_channels": sum(x["channels"] for x in rows if x["state"] != "RESOLVED_FROM_PROVENANCE"),
        "by_provider": {f"{k[0]}|{k[1]}": v for k, v in c.most_common()},
        "missing_facts": dict(collections.Counter(x["missing_fact"].split(" at ")[0].split(" in ")[0] for x in rows if x["missing_fact"]).most_common())}
doc = {"schema": "lane_b_unresolved_resolution.v1", "financial_data_commit_used": head, "rule": "git-tracked provenance.json only; no data opened; no inference from names beyond the repository's declared entity-path convention",
       "summary": summ, "rows": rows}
json.dump(doc, open(OUT, "w"), indent=1)
print(json.dumps(summ)[:900])
