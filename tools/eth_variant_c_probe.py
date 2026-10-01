"""ETH 4h variant C probe: persistent level inputs (outer-TRAIN acf_lag_1 >= 0.99, from the frozen v2 profile)
replaced by causal first differences x_t - x_{t-1}; everything else as in the F1 linear probe."""
import csv, hashlib, importlib.util, json, sys
import numpy as np, pandas as pd
PS_PATH, VIEW, MANIFEST, WORKLIST, METRICS, OUT = sys.argv[1:7]
spec = importlib.util.spec_from_file_location("ps", PS_PATH); ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
man = json.load(open(MANIFEST)); feats = man["features"]
acf1 = {r["column"]: float(r["value"]) for r in csv.DictReader(open(METRICS)) if r["metric"] == "acf_lag_1" and r["status"] in ("OK", "OK_WITH_WARNING")}
persistent = sorted(f for f in feats if acf1.get(f, 0) >= 0.99)
df = pd.read_csv(VIEW, nrows=13699)
ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
Xdf = df[feats].copy()
for f in persistent:
    Xdf[f] = Xdf[f].diff()
X = Xdf.to_numpy(float)
tg = ps.build_targets(df["CLOSE"].to_numpy(float), ts, "CLOSE", "CLOSE", {"Y_s": [4], "Y_l": [24, 144]}, 14400)
folds = ps.inner_folds(13699, k=3, val_frac=0.15, purge=60)
wl = list(csv.DictReader(open(WORKLIST)))
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ w + ym
rows = []
for f in folds:
    a, b = f["train"]; va0, va1 = f["val"]
    tiers = sorted({r["feature"] for r in wl if r["fold"] == f["name"] and r["tier"] in ("PRIORITY", "SYNERGY", "REPRESENTATIVE", "EXPLORATORY")})
    for (tn, h), t in tg.items():
        tr = ps.label_rows(t, b, a); tr = tr[tr >= 1]
        va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]
        for name, cols in {"C_DIFF_ALL": feats, "C_DIFF_FOLD_TIERS": tiers}.items():
            idx = [feats.index(c) for c in cols]
            trm = tr[np.isfinite(X[tr][:, idx]).all(1)]; vam = va[np.isfinite(X[va][:, idx]).all(1)]
            pred = ridge(X[trm][:, idx], t.values[trm], X[vam][:, idx]); y = t.values[vam]
            rows.append({"fold": f["name"], "target": f"{tn}@{h}h", "subset": name, "n_features": len(cols), "val_rows": int(len(vam)),
                         "mae": float(np.mean(np.abs(pred - y))), "naive_mae": float(np.mean(np.abs(y))), "mse": float(np.mean((pred - y) ** 2)), "naive_mse": float(np.mean(y ** 2)), "mean_only_mae": float(np.mean(np.abs(float(t.values[trm].mean()) - y))), "mean_only_mse": float(np.mean((float(t.values[trm].mean()) - y) ** 2))})
by = {s: float(np.mean([r["mae"] for r in rows if r["subset"] == s])) for s in ("C_DIFF_ALL", "C_DIFF_FOLD_TIERS")}
beats = {s: all(r["mae"] < r["naive_mae"] for r in rows if r["subset"] == s) for s in by}
out = {"schema": "lane_b_eth_variant_c_probe.v2", "transform": "first difference of inputs with outer-TRAIN acf_lag_1 >= 0.99 (from the v2 TRAIN profile); fitted nothing",
       "persistent_inputs_differenced": persistent, "rows": rows, "mean_mae_by_subset": by,
       "naive_mean_mae": float(np.mean([r["naive_mae"] for r in rows if r["subset"] == "C_DIFF_ALL"])), "beats_naive_every_target": beats}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({"n_diff": len(persistent), "by": by, "naive": out["naive_mean_mae"], "beats": beats}))
