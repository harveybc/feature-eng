"""ETH 4h F1 inner-validation comparison with a declared default linear probe (ridge, alpha=1.0).

Candidates compared on identical inner-validation rows: C_ALL (variant A, 83), B_SCREEN_UNION (58,
worklist 397b67d6, fitted on inner-train only), plus per-fold screen tiers. Standardization and ridge
are fitted on each fold's TRAIN rows only; targets are elapsed-second log returns; naive = zero return
(price persistence). Development evidence; a linear probe is a diagnostic, not the temporal model.
"""
import csv, hashlib, importlib.util, json, sys
import numpy as np, pandas as pd
PS_PATH, VIEW, MANIFEST, WORKLIST, OUT = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("ps", PS_PATH); ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
man = json.load(open(MANIFEST))
feats = man["features"]
df = pd.read_csv(VIEW, nrows=13699)
assert len(df) == 13699
ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
X = df[feats].to_numpy(float)
tg = ps.build_targets(df["CLOSE"].to_numpy(float), ts, "CLOSE", "CLOSE", {"Y_s": [4], "Y_l": [24, 144]}, 14400)
folds = ps.inner_folds(13699, k=3, val_frac=0.15, purge=60)
wl = list(csv.DictReader(open(WORKLIST)))
union = sorted({r["feature"] for r in wl if r["tier"] in ("PRIORITY", "SYNERGY", "REPRESENTATIVE", "EXPLORATORY")})
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A = (Xtr - mu) / sd; B = (Xva - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ w + ym
rows = []
for f in folds:
    a, b = f["train"]; va0, va1 = f["val"]
    per_fold_tiers = sorted({r["feature"] for r in wl if r["fold"] == f["name"] and r["tier"] in ("PRIORITY", "SYNERGY", "REPRESENTATIVE", "EXPLORATORY")})
    subsets = {"C_ALL": feats, "B_SCREEN_UNION": union, "B_FOLD_TIERS": per_fold_tiers}
    for (tn, h), t in tg.items():
        tr = ps.label_rows(t, b, a)
        va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]
        for name, cols in subsets.items():
            idx = [feats.index(c) for c in cols]
            trm = tr[np.isfinite(X[tr][:, idx]).all(1)]; vam = va[np.isfinite(X[va][:, idx]).all(1)]
            common = np.intersect1d(vam, va)
            pred = ridge(X[trm][:, idx], t.values[trm], X[vam][:, idx])
            y = t.values[vam]
            mae, naive = float(np.mean(np.abs(pred - y))), float(np.mean(np.abs(y)))
            rows.append({"fold": f["name"], "target": f"{tn}@{h}h", "subset": name, "n_features": len(cols), "train_rows": int(len(trm)),
                         "val_rows": int(len(vam)), "mae": mae, "naive_mae": naive, "mse": float(np.mean((pred - y) ** 2)), "naive_mse": float(np.mean(y ** 2)), "mean_only_mae": float(np.mean(np.abs(float(t.values[trm].mean()) - y))), "mean_only_mse": float(np.mean((float(t.values[trm].mean()) - y) ** 2)), "skill": 1 - mae / naive if naive > 0 else None})
agg = {}
for r in rows:
    agg.setdefault((r["subset"], r["target"]), []).append(r["mae"])
summary = {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in agg.items()}
by_subset = {}
for s in ("C_ALL", "B_SCREEN_UNION", "B_FOLD_TIERS"):
    by_subset[s] = float(np.mean([v for k, v in summary.items() if k.startswith(s + "|")]))
naive_mean = float(np.mean([r["naive_mae"] for r in rows if r["subset"] == "C_ALL"]))
winner = min(by_subset, key=by_subset.get)
out = {"schema": "lane_b_eth_f1_linear_probe.v2", "probe": "ridge alpha=1.0, fold-train standardization, default (declared before measuring)",
       "targets": ["Y_s@4h", "Y_l@24h", "Y_l@144h"], "rows": rows, "mean_mae_by_subset_target": summary,
       "mean_mae_by_subset": by_subset, "naive_mean_mae": naive_mean, "rule": "strict minimum mean inner-validation MAE across folds and targets",
       "winner": winner, "beats_naive_every_target": {s: all(r["mae"] < r["naive_mae"] for r in rows if r["subset"] == s) for s in by_subset},
       "inputs": {"manifest_canonical": man.get("manifest_sha256_canonical"), "worklist_sha256": hashlib.sha256(open(WORKLIST, "rb").read()).hexdigest(),
                  "view_sha256": hashlib.sha256(open(VIEW, "rb").read()).hexdigest()}}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({"by_subset": by_subset, "naive": naive_mean, "winner": winner, "beats_naive": out["beats_naive_every_target"]}))
