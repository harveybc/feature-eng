"""Re-check every published horizon-scoped PASS against an intercept-only (TRAIN-mean) control on identical rows."""
import importlib.util, json, sys
import numpy as np, pandas as pd
PS, OUT = sys.argv[1], sys.argv[2]; cases = json.loads(sys.argv[3])
s = importlib.util.spec_from_file_location("ps", PS); ps = importlib.util.module_from_spec(s); s.loader.exec_module(ps)
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Am = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(Am.T @ Am + alpha * np.eye(Am.shape[1]), Am.T @ (ytr - ym)); return B @ w + ym
out = []
for c in cases:
    df = pd.read_csv(c["csv"], nrows=c["n_train"]); ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
    C, H, L, O = (df[k].astype(float) for k in ("CLOSE", "HIGH", "LOW", "OPEN"))
    X = pd.DataFrame({"a": np.log(H / L), "b": (C - L) / (H - L).replace(0, np.nan), "c": np.log(C / O)}).to_numpy()
    Xw = np.column_stack([X, df[["OPEN", "LOW", "HIGH", "CLOSE"]].astype(float).to_numpy()])
    folds = ps.inner_folds(c["n_train"], k=3, val_frac=0.15, purge=c["purge"])
    for h in c["hours"]:
        nm = "Y_s" if h <= 6 else "Y_l"
        t = ps.build_targets(C.to_numpy(float), ts, "CLOSE", "CLOSE", {nm: [h]}, 3600)[(nm, h)]
        rows = []
        for f in folds:
            a, b = f["train"]; va0, va1 = f["val"]
            tr = ps.label_rows(t, b, a); tr = tr[np.isfinite(X[tr]).all(1)]
            va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]; va = va[np.isfinite(Xw[va]).all(1)]
            pred = ridge(X[tr], t.values[tr], X[va]); y = t.values[va]; m0 = float(t.values[tr].mean())
            rows.append({"fold": f["name"], "model_mae": float(np.mean(np.abs(pred - y))), "model_mse": float(np.mean((pred - y) ** 2)),
                         "zero_mae": float(np.mean(np.abs(y))), "zero_mse": float(np.mean(y ** 2)),
                         "mean_only_mae": float(np.mean(np.abs(m0 - y))), "mean_only_mse": float(np.mean((m0 - y) ** 2))})
        beats_zero = all(r["model_mae"] < r["zero_mae"] and r["model_mse"] < r["zero_mse"] for r in rows)
        beats_mean = all(r["model_mae"] < r["mean_only_mae"] and r["model_mse"] < r["mean_only_mse"] for r in rows)
        mean_beats_zero = all(r["mean_only_mae"] < r["zero_mae"] and r["mean_only_mse"] < r["zero_mse"] for r in rows)
        out.append({"case": c["name"], "horizon_h": h, "range_beats_zero": beats_zero, "range_beats_mean_only": beats_mean,
                    "mean_only_beats_zero": mean_beats_zero, "verdict": "FEATURE_SIGNAL" if beats_zero and beats_mean else ("DRIFT_ONLY" if beats_zero else "FAIL"), "rows": rows})
json.dump({"schema": "lane_b_drift_check.v1", "results": out}, open(OUT, "w"), indent=1)
print(json.dumps([(r["case"], r["horizon_h"], r["verdict"]) for r in out]))
