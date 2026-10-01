"""Intercept-only closure for published lake range passes. Recomputes the range-family ridge exactly as the lake
probes did (same split rows, purge, scored rows filtered on A+range finiteness) and adds the intercept-only
(fold-TRAIN mean) control on identical rows. Reproduction of the published v1 fold MAE/naive MAE is checked and
recorded. Verdict: FEATURE_SIGNAL (below zero naive AND intercept control, MAE and MSE, every fold), DRIFT_ONLY
(below zero naive only), FAIL."""
import importlib.util, json, sys
import numpy as np, pandas as pd
PS, CSV, DECL, NT, PURGE, STEP, SPLIT, HOURS, V1, DATASET, OUT = sys.argv[1:12]
NT, PURGE, STEP, HOURS = int(NT), int(PURGE), int(STEP), json.loads(HOURS)
s = importlib.util.spec_from_file_location("ps", PS); ps = importlib.util.module_from_spec(s); s.loader.exec_module(ps)
decl = json.load(open(DECL)); A = decl["all_admissible_control"]; v1 = json.load(open(V1))["splits"][SPLIT]["results"]["range"]
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Am = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    return B @ np.linalg.solve(Am.T @ Am + alpha * np.eye(Am.shape[1]), Am.T @ (ytr - ym)) + ym
df = pd.read_csv(CSV, nrows=NT); ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
C, H, L, O = (df[c].astype(float) for c in ("CLOSE", "HIGH", "LOW", "OPEN"))
rng = pd.DataFrame({"log_high_low": np.log(H / L), "close_location": (C - L) / (H - L).replace(0, np.nan), "log_close_open": np.log(C / O)})
X = rng.to_numpy(float); Xw = pd.concat([df[A].astype(float), rng], axis=1).to_numpy(float)
folds = ps.inner_folds(NT, k=3, val_frac=0.15, purge=PURGE); out = []
for h in HOURS:
    nm = "Y_s" if h <= 6 else "Y_l"
    t = ps.build_targets(C.to_numpy(float), ts, "CLOSE", "CLOSE", {nm: [h]}, STEP)[(nm, h)]
    rows = []
    for f in folds:
        a, b = f["train"]; va0, va1 = f["val"]
        tr = ps.label_rows(t, b, a); trm = tr[np.isfinite(X[tr]).all(1)]
        va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]; va = va[np.isfinite(Xw[va]).all(1)]
        pred = ridge(X[trm], t.values[trm], X[va]); y = t.values[va]; m0 = float(t.values[trm].mean())
        rows.append({"fold": f["name"], "val_rows": int(len(va)), "mae": float(np.mean(np.abs(pred - y))), "naive_mae": float(np.mean(np.abs(y))),
                     "mse": float(np.mean((pred - y) ** 2)), "naive_mse": float(np.mean(y ** 2)),
                     "mean_only_mae": float(np.mean(np.abs(m0 - y))), "mean_only_mse": float(np.mean((m0 - y) ** 2))})
    ref = v1[f"{h}h"]["rows"]
    repro = all(r["mae"] == q["mae"] and r["naive_mae"] == q["naive_mae"] and r["val_rows"] == q["val_rows"] for r, q in zip(rows, ref))
    z = all(r["mae"] < r["naive_mae"] and r["mse"] < r["naive_mse"] for r in rows)
    m = all(r["mae"] < r["mean_only_mae"] and r["mse"] < r["mean_only_mse"] for r in rows)
    out.append({"dataset": DATASET, "candidate": "range", "horizon": f"{h}h", "split": SPLIT, "reproduces_v1": repro,
                "verdict": "FEATURE_SIGNAL" if z and m else ("DRIFT_ONLY" if z else "FAIL"),
                "intercept_only_beats_zero": all(r["mean_only_mae"] < r["naive_mae"] and r["mean_only_mse"] < r["naive_mse"] for r in rows), "rows": rows})
json.dump({"schema": "lane_b_intercept_closure.v1", "rule": __doc__, "purge": PURGE, "n_train": NT, "verdicts": out}, open(OUT, "w"), indent=1)
print(json.dumps([(o["dataset"], o["split"], o["horizon"], o["verdict"], o["reproduces_v1"]) for o in out]))
