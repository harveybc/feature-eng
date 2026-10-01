#!/usr/bin/env python3
"""Evaluation/selection denominators per source family: candidates, profiled, screened, evaluated,
selected, deferred, excluded, frozen controls. Profiles computed are never counted as selection.

Candidate grain: one channel = one non-timestamp column of a discovered file (catalogue), plus the
lane B lake/local channel sets that are not in the financial-data catalogue (TSL, legacy views).
"""
import collections, csv, json, sys
cat_path, out_path = sys.argv[1], sys.argv[2]
EXCLUDED_FAMILIES = {"hilbert": "timing VIOLATED <1000 rows/restart", "multitaper": "timing VIOLATED <1000 rows/restart",
                     "emd": "backend undeclared", "sota_hmm_regime": "fit on full series + smoothing (non-causal)"}
rows = list(csv.DictReader(open(cat_path)))
agg = collections.defaultdict(lambda: collections.Counter())
reasons = collections.defaultdict(collections.Counter)
for r in rows:
    if not r["path"]:
        prov = r["provider"]
        agg[prov]["sources_without_files"] += 1
        continue
    prov = r["provider"]
    ch = max(0, int(r["columns"] or 0) - 1)
    prof = int(r["lane_b_covered_columns"] or 0)
    a = agg[prov]
    a["files"] += 1; a["candidates"] += ch; a["profiled"] += min(prof, ch) if ch else prof
    fam = r["family"]
    if fam in EXCLUDED_FAMILIES:
        a["excluded"] += ch; reasons[prov][f"excluded:{fam}: {EXCLUDED_FAMILIES[fam]}"] += ch
    else:
        d = max(0, ch - min(prof, ch))
        a["deferred"] += d
        if d:
            reasons[prov]["deferred: " + r["missing_action"][:90]] += d
extra = {"sota_benchmarks (TSL, lake identity)": {"files": 3, "candidates": 1204, "profiled": 1204, "excluded": 0, "deferred": 0},
         "legacy phase-1b d4 (local)": {"files": 1, "candidates": 23, "profiled": 23, "excluded": 0, "deferred": 0},
         "heuristic-strategy EURUSD 1h (git-pinned, provider undocumented)": {"files": 1, "candidates": 4, "profiled": 4, "excluded": 0, "deferred": 0},
         "predictor ETH 4h model-ready view (git-pinned; parents Binance Spot)": {"files": 1, "candidates": 83, "profiled": 83, "excluded": 0, "deferred": 0}}
table = []
for prov, a in sorted(agg.items(), key=lambda kv: -kv[1]["candidates"]):
    table.append({"source_family": prov, "files": a["files"], "candidates": a["candidates"], "profiled": a["profiled"],
                  "screened": 0, "evaluated": 0, "selected": 0, "frozen_controls": 0, "deferred": a["deferred"],
                  "excluded": a["excluded"], "sources_without_files": a["sources_without_files"],
                  "top_reasons": dict(reasons[prov].most_common(3))})
for k, v in extra.items():
    scr = 83 if "ETH" in k else 0
    fz = 83 if "ETH" in k else (4 if "EURUSD" in k else 0)
    table.append({"source_family": k, **v, "screened": scr, "evaluated": 0, "selected": 0, "frozen_controls": fz,
                  "sources_without_files": 0, "top_reasons": {}})
tot = collections.Counter()
for t in table:
    for k in ("files", "candidates", "profiled", "screened", "evaluated", "selected", "frozen_controls", "deferred", "excluded"):
        tot[k] += t[k]
doc = {"schema": "lane_b_selection_denominators.v1", "date": "2026-10-01",
       "definitions": {"candidate": "one non-timestamp column of a discovered file, or one channel of a lane B dataset",
                       "profiled": "has a TRAIN-only profile (lane B or verified c162); a profile is NOT a selection",
                       "screened": "ranked by the PS2 Spearman screen against business targets (not an evaluation)",
                       "evaluated": "measured utility on the declared target under the selection protocol (inner validation); none yet",
                       "selected": "chosen by the protocol's F1 freeze; none yet",
                       "frozen_controls": "frozen as the all-admissible control (variant A) for pilots; not a selection decision",
                       "deferred": "not profiled yet, each with its named missing action", "excluded": "producer family excluded with a reason"},
       "rows": table, "totals": dict(tot),
       "note": "candidates of derived files overlap economically with their parents; this is a channel count, not a count of distinct economic signals"}
json.dump(doc, open(out_path, "w"), indent=1)
w = csv.DictWriter(open(out_path.replace(".json", ".csv"), "w", newline=""), fieldnames=list(table[0].keys()), lineterminator="\n")
w.writeheader()
for t in table:
    w.writerow({**t, "top_reasons": json.dumps(t["top_reasons"])})
print(json.dumps(doc["totals"]))
