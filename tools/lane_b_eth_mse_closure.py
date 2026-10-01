"""Close the ETH 4h MSE_NOT_STORED ledger rows from the v2 reruns of the early probes (F1, variant C, variant D),
which store MSE and the intercept-only (fold-TRAIN mean) control beside the published MAE. Each v2 row's MAE,
naive MAE and val rows are checked against v1 (exact). Verdict per candidate x target, over the 3 inner folds:
FEATURE_SIGNAL / DRIFT_ONLY / FAIL as in lane_b_intercept_closure.v1."""
import json, sys
from collections import defaultdict
f1v1, f1v2, cv1, cv2, dv1, dv2, OUT = sys.argv[1:8]
L = lambda p: json.load(open(p))
groups = []
for v1, v2, src in ((f1v1, f1v2, "f1/eth_f1_probe.json"), (cv1, cv2, "f1/eth_variant_c_probe.json")):
    a, b = L(v1)["rows"], L(v2)["rows"]
    g1, g2 = defaultdict(list), defaultdict(list)
    for r in a: g1[(r["subset"], r["target"])].append(r)
    for r in b: g2[(r["subset"], r["target"])].append(r)
    groups += [(k[0], k[1], g1[k], g2[k], src) for k in g2]
A, B = L(dv1)["families"], L(dv2)["families"]
for fam in B:
    g1, g2 = defaultdict(list), defaultdict(list)
    for r in A[fam]["rows"]: g1[r["target"]].append(r)
    for r in B[fam]["rows"]: g2[r["target"]].append(r)
    groups += [(fam, t, g1[t], g2[t], "f1/ETH_VARIANT_D_FAMILIES_PROBE.v1.json") for t in g2]
out = []
for cand, target, r1, r2, src in groups:
    repro = len(r1) == len(r2) and all(x["mae"] == y["mae"] and x["naive_mae"] == y["naive_mae"] and x["val_rows"] == y["val_rows"] for x, y in zip(r1, r2))
    z = all(r["mae"] < r["naive_mae"] and r["mse"] < r["naive_mse"] for r in r2)
    m = all(r["mae"] < r["mean_only_mae"] and r["mse"] < r["mean_only_mse"] for r in r2)
    out.append({"dataset": "ETH_4h", "candidate": cand, "horizon": target.split("@")[1], "split": "inner_TRAIN", "reproduces_v1": repro, "supersedes_source": src,
                "verdict": "FEATURE_SIGNAL" if z and m else ("DRIFT_ONLY" if z else "FAIL"),
                "mae_below_zero_all_folds": all(r["mae"] < r["naive_mae"] for r in r2), "mse_below_zero_all_folds": all(r["mse"] < r["naive_mse"] for r in r2), "rows": r2})
json.dump({"schema": "lane_b_intercept_closure.v1", "rule": __doc__, "verdicts": out}, open(OUT, "w"), indent=1)
print(json.dumps([(o["candidate"], o["horizon"], o["verdict"], o["reproduces_v1"]) for o in out]))
