"""Horizon-scoped probe gate: declared ridge (alpha 1.0, fold-TRAIN standardization) vs zero-return naive,
ONE target, 3 inner folds inside TRAIN; PASS only if MAE and MSE are strictly below naive in every fold.
Range and realized-volatility families are causal (rows <= t, bar-close convention)."""
import hashlib, importlib.util, json, sys
import numpy as np, pandas as pd
PS_PATH, CSV, MANIFEST, NT, STEP, H, PURGE, OUT = sys.argv[1:9]
NT, STEP, H, PURGE = int(NT), int(STEP), int(H), int(PURGE)
s = importlib.util.spec_from_file_location("ps", PS_PATH); ps = importlib.util.module_from_spec(s); s.loader.exec_module(ps)
man = json.load(open(MANIFEST)); A = man["features"]
df = pd.read_csv(CSV, nrows=NT)
ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
C, Hh, L, O = (df[c].astype(float) for c in ("CLOSE", "HIGH", "LOW", "OPEN"))
lc = np.log(C); r1 = lc.diff(); hl = np.log(Hh / L)
rng = pd.DataFrame({"log_high_low": hl, "close_location": (C - L) / (Hh - L).replace(0, np.nan), "log_close_open": np.log(C / O)})
rv = pd.DataFrame({f"rv_std_ret_{n}": r1.rolling(n, min_periods=n).std() for n in (6, 24, 42)})
for n in (6, 24):
    rv[f"parkinson_{n}"] = np.sqrt((hl ** 2).rolling(n, min_periods=n).mean() / (4 * np.log(2)))
base = df[A].astype(float)
cands = {"A": base, "A+range": pd.concat([base, rng], axis=1), "A+range+rv": pd.concat([base, rng, rv], axis=1),
         "range": rng, "range+rv": pd.concat([rng, rv], axis=1)}
name = "Y_s" if H <= 6 else "Y_l"
t = ps.build_targets(C.to_numpy(float), ts, "CLOSE", "CLOSE", {name: [H]}, STEP)[(name, H)]
folds = ps.inner_folds(NT, k=3, val_frac=0.15, purge=PURGE)
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Am = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(Am.T @ Am + alpha * np.eye(Am.shape[1]), Am.T @ (ytr - ym)); return B @ w + ym
res = {}
for cn, F in cands.items():
    X = F.to_numpy(float); rows = []
    for f in folds:
        a, b = f["train"]; va0, va1 = f["val"]
        tr = ps.label_rows(t, b, a)
        va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]
        # identical scored rows for model and naive, and across candidates: rows finite for the widest candidate
        Xw = cands["A+range+rv"].to_numpy(float)
        va = va[np.isfinite(Xw[va]).all(1)]; trm = tr[np.isfinite(X[tr]).all(1)]
        pred = ridge(X[trm], t.values[trm], X[va]); y = t.values[va]
        rows.append({"fold": f["name"], "val_rows": int(len(va)), "train_rows": int(len(trm)),
                     "mae": float(np.mean(np.abs(pred - y))), "naive_mae": float(np.mean(np.abs(y))),
                     "mse": float(np.mean((pred - y) ** 2)), "naive_mse": float(np.mean(y ** 2))})
    ok = all(r["mae"] < r["naive_mae"] and r["mse"] < r["naive_mse"] for r in rows)
    res[cn] = {"features": list(F.columns), "rows": rows, "gate": "PASS" if ok else "SKIPPED_NOT_BETTER_THAN_NAIVE",
               "mean_mae": float(np.mean([r["mae"] for r in rows])), "mean_naive_mae": float(np.mean([r["naive_mae"] for r in rows])),
               "mean_mse": float(np.mean([r["mse"] for r in rows])), "mean_naive_mse": float(np.mean([r["naive_mse"] for r in rows]))}
out = {"schema": "lane_b_horizon_scoped_probe.v1", "target": f"{name}@{H * STEP // 3600}h (elapsed seconds)", "horizon_bars": 1 if H * 3600 == STEP else None,
       "gate_rule": "MAE and MSE strictly below the zero-return naive in every one of 3 inner folds; same scored rows for every candidate",
       "manifest_canonical": man.get("manifest_sha256_canonical"), "candidates": res}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: (v["gate"], round(v["mean_mae"], 7), round(v["mean_naive_mae"], 7), round(v["mean_mse"], 10), round(v["mean_naive_mse"], 10)) for k, v in res.items()}))
