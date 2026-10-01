"""Declared default ridge probe (alpha 1.0, fold-TRAIN standardization) vs zero-return naive on inner folds.
Usage: probe_generic.py PS_PATH CSV MANIFEST N_TRAIN STEP_SECONDS PRICE_COL TS_COL PURGE HOURS_JSON OUT [EXTRA_FEATURES_PY]"""
import hashlib, importlib.util, json, sys
import numpy as np, pandas as pd
PS_PATH, CSV, MANIFEST, NT, STEP, PRICE, TSC, PURGE, HOURS, OUT = sys.argv[1:11]
NT, STEP, PURGE = int(NT), int(STEP), int(PURGE); hours = json.loads(HOURS)
spec = importlib.util.spec_from_file_location("ps", PS_PATH); ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
man = json.load(open(MANIFEST)); feats = man["features"]
df = pd.read_csv(CSV, nrows=NT)
ts = ps.timestamps_to_seconds(df[TSC].tolist(), "%Y-%m-%d %H:%M:%S")
X = df[feats].to_numpy(float)
tg = ps.build_targets(df[PRICE].to_numpy(float), ts, PRICE, PRICE, {"Y_s": [h for h in hours if h <= 6], "Y_l": [h for h in hours if h > 6]}, STEP)
folds = ps.inner_folds(NT, k=3, val_frac=0.15, purge=PURGE)
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ w + ym
rows = []
for f in folds:
    a, b = f["train"]; va0, va1 = f["val"]
    for (tn, h), t in tg.items():
        if t.status != "CONSTRUCTED":
            rows.append({"fold": f["name"], "target": f"{tn}@{h}h", "status": t.status}); continue
        tr = ps.label_rows(t, b, a)
        va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]
        trm = tr[np.isfinite(X[tr]).all(1)]; vam = va[np.isfinite(X[va]).all(1)]
        if len(trm) < 50 or len(vam) < 30:
            rows.append({"fold": f["name"], "target": f"{tn}@{h}h", "status": "NOT_RUN_TOO_FEW_ROWS"}); continue
        pred = ridge(X[trm], t.values[trm], X[vam]); y = t.values[vam]
        rows.append({"fold": f["name"], "target": f"{tn}@{h}h", "status": "MEASURED", "val_rows": int(len(vam)),
                     "mae": float(np.mean(np.abs(pred - y))), "naive_mae": float(np.mean(np.abs(y))),
                     "mse": float(np.mean((pred - y) ** 2)), "naive_mse": float(np.mean(y ** 2))})
m = [r for r in rows if r["status"] == "MEASURED"]
per_target = {}
for r in m:
    per_target.setdefault(r["target"], []).append(r)
pt = {k: {"mae": float(np.mean([x["mae"] for x in v])), "naive_mae": float(np.mean([x["naive_mae"] for x in v])),
          "beats_naive_all_folds": all(x["mae"] < x["naive_mae"] for x in v)} for k, v in per_target.items()}
out = {"schema": "lane_b_ridge_probe.v1", "probe": "ridge alpha=1.0, fold-TRAIN standardization; naive = zero return; elapsed-second log-return targets",
       "manifest_canonical": man.get("manifest_sha256_canonical"), "features": feats, "rows": rows, "per_target": pt,
       "targets_beating_naive": sorted(k for k, v in pt.items() if v["beats_naive_all_folds"]),
       "all_consumed_targets_beat_naive": all(v["beats_naive_all_folds"] for v in pt.values()) and len(pt) == len(tg),
       "status": None}
out["status"] = "PASSES_NAIVE_ON_EVERY_TARGET" if out["all_consumed_targets_beat_naive"] else "SKIPPED_NOT_BETTER_THAN_NAIVE"
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({"status": out["status"], "beating": out["targets_beating_naive"], "n_targets": len(pt)}))
