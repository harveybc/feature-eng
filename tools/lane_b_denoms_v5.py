"""Selection denominators v5: v4 plus the intercept-only closures (older lake range passes; ETH early probes rerun
with MSE) and ledger it3 counts. Evaluated-channel counts are unchanged (the closures re-score channels already
counted); what changes is how many outcomes are settled."""
import json, hashlib, sys
from collections import Counter
from pathlib import Path
v4 = json.load(open(sys.argv[1])); ledger = json.load(open(sys.argv[2])); closures = sys.argv[3:-1]; OUT = sys.argv[-1]
verd = [v for p in closures for v in json.load(open(p))["verdicts"]]
doc = {**v4, "schema": "lane_b_selection_denominators.v5", "supersedes": "SELECTION_DENOMINATORS.v4.json (kept)",
       "intercept_closures": {"rows": len(verd), "by_verdict": dict(Counter(v["verdict"] for v in verd)),
                              "all_reproduce_v1": all(v["reproduces_v1"] for v in verd),
                              "feature_signal": sorted(f'{v["dataset"]}|{v["split"]}|{v["candidate"]}|{v["horizon"]}' for v in verd if v["verdict"] == "FEATURE_SIGNAL")},
       "selected_by_linear_probe": {**v4["selected_by_linear_probe"], "ledger_it3_eligible_rows": ledger["counts"]["ELIGIBLE"]},
       "skipped_not_better_than_naive": {"outcomes": v4["skipped_not_better_than_naive"]["outcomes"] + sum(v["verdict"] != "FEATURE_SIGNAL" for v in verd),
                                         "added": v4["skipped_not_better_than_naive"]["added"] + [f"{sum(v['verdict'] != 'FEATURE_SIGNAL' for v in verd)} closure rows (DRIFT_ONLY or FAIL)"]},
       "ledger_it3_counts": ledger["counts"],
       "inputs": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [sys.argv[1], sys.argv[2], *closures]}}
doc.pop("ledger_it2_counts", None)
json.dump(doc, open(OUT, "w"), indent=1)
print(json.dumps({"closures": doc["intercept_closures"]["by_verdict"], "skipped": doc["skipped_not_better_than_naive"]["outcomes"], "ledger": ledger["counts"]}))
