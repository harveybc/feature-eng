"""Selection denominators v4: v3 plus the staged family screen (STAGE_OUTCOME v1 and the evidence-clock FX calendar
v2), the git-pinned EURUSD 1h intercept-only closure (horizon-scoped probe v2) and ledger it2 counts. A channel counts
as evaluated when it entered the ridge utility stage (kept after the redundancy stage)."""
import json, hashlib, sys
from pathlib import Path
v3 = json.load(open(sys.argv[1])); ledger = json.load(open(sys.argv[2])); hs2 = json.load(open(sys.argv[3])); stages = sys.argv[4:-1]; OUT = sys.argv[-1]
staged, fam_outcomes = {}, []
for p in stages:
    d = json.load(open(p))
    for fam, rec in d["families"].items():
        if "stage_3_redundancy" not in rec:
            continue
        staged.setdefault(d["dataset"], {})[fam] = len(rec["stage_3_redundancy"]["kept"])   # a later file (v2 calendar) replaces the same key
        fam_outcomes.append((d["dataset"], fam, any(u["gate"] == "PASS" for u in rec["stage_4_utility"].values())))
last = {}
for ds, fam, ok in fam_outcomes: last[(ds, fam)] = ok
staged_ev = sum(sum(v.values()) for v in staged.values())
ev3 = v3["evaluated_by_linear_probe"]["channels"]
doc = {"schema": "lane_b_selection_denominators.v4", "date": "2026-10-01", "supersedes": "SELECTION_DENOMINATORS.v3.json (kept)",
       "catalogue_channels_v1": v3["catalogue_channels_v1"],
       "evaluated_by_linear_probe": {"channels": ev3 + staged_ev, "from_v3": ev3, "staged_screen": {"channels": staged_ev, "by_dataset": staged},
                                     "by_dataset_v3": v3["evaluated_by_linear_probe"]["by_dataset"],
                                     "definition": v3["evaluated_by_linear_probe"]["definition"] + "; staged-screen channels counted after the |r|>=0.95 redundancy stage"},
       "selected_by_linear_probe": {**v3["selected_by_linear_probe"],
                                    "gitpinned_eurusd_1h_intercept_closure": {k: v["verdict"] for k, v in hs2["candidates"].items()},
                                    "ledger_it2_eligible_rows": ledger["counts"]["ELIGIBLE"]},
       "frozen_controls": v3["frozen_controls"],
       "skipped_not_better_than_naive": {"outcomes": v3["skipped_not_better_than_naive"]["outcomes"] + sum(1 for ok in last.values() if not ok),
                                         "added": [f"staged family screen: {sum(1 for ok in last.values() if not ok)} dataset x family outcomes with no pass (incl. FX calendar under the evidence clock)"]},
       "ledger_it2_counts": ledger["counts"],
       "deferred_remaining_channels": v3["deferred_remaining_channels"],
       "evaluated_by_temporal_model": 0, "selected_by_temporal_model": 0,
       "inputs": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [sys.argv[1], sys.argv[2], sys.argv[3], *stages]}}
json.dump(doc, open(OUT, "w"), indent=1)
print(json.dumps({"evaluated": doc["evaluated_by_linear_probe"]["channels"], "staged": staged_ev, "skipped": doc["skipped_not_better_than_naive"]["outcomes"], "ledger": ledger["counts"]}))
