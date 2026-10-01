"""ETH 4h PS4 variant-D candidate families, each causal (row t uses rows <= t only) and probe-gated
against the zero-return naive with the declared default ridge probe on the same inner folds.
Families: native wavelet (financial-data 028f4284 compute_wavelet_native on log close, db4, level 3,
window 128 declared), rolling z-scores, realized volatility, range. Nothing is fitted outside fold TRAIN."""
import hashlib, importlib.util, json, sys
import numpy as np, pandas as pd
PS_PATH, NW_PATH, VIEW, MANIFEST, OUT = sys.argv[1:6]
def load(name, p):
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ps, nw = load("ps", PS_PATH), load("nw", NW_PATH)
man = json.load(open(MANIFEST)); feats = man["features"]
df = pd.read_csv(VIEW, nrows=13699)
ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
lc = np.log(df["CLOSE"].astype(float)); r1 = lc.diff()
fam = {}
w, wmeta = nw.compute_wavelet_native(pd.DataFrame({"timestamp": df["DATE_TIME"], "close": lc}), window=128, wavelet="db4", level=3,
                                     frozen_from={"fold_id": "declared_default", "train_end": "2023-12-31T20:00:00Z"}, sample_interval_seconds=14400)
fam["native_wavelet"] = w.drop(columns=["timestamp"])
z = pd.DataFrame(index=df.index)
for n in (20, 60):
    z[f"z_logclose_{n}"] = (lc - lc.rolling(n, min_periods=n).mean()) / lc.rolling(n, min_periods=n).std()
z["z_ret_60"] = (r1 - r1.rolling(60, min_periods=60).mean()) / r1.rolling(60, min_periods=60).std()
fam["rolling_zscore"] = z
rv = pd.DataFrame(index=df.index)
hl = np.log(df["HIGH"].astype(float) / df["LOW"].astype(float))
for n in (6, 24, 42):
    rv[f"rv_std_ret_{n}"] = r1.rolling(n, min_periods=n).std()
for n in (6, 24):
    rv[f"parkinson_{n}"] = np.sqrt((hl ** 2).rolling(n, min_periods=n).mean() / (4 * np.log(2)))
fam["realized_volatility"] = rv
rg = pd.DataFrame(index=df.index)
rg["log_high_low"] = hl
rg["close_location"] = (df["CLOSE"] - df["LOW"]) / (df["HIGH"] - df["LOW"]).replace(0, np.nan)
rg["log_close_open"] = np.log(df["CLOSE"] / df["OPEN"])
fam["range"] = rg
tg = ps.build_targets(df["CLOSE"].to_numpy(float), ts, "CLOSE", "CLOSE", {"Y_s": [4], "Y_l": [24, 144]}, 14400)
folds = ps.inner_folds(13699, k=3, val_frac=0.15, purge=60)
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    wv = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym)); return B @ wv + ym
def probe(X):
    rows = []
    for f in folds:
        a, b = f["train"]; va0, va1 = f["val"]
        for (tn, h), t in tg.items():
            tr = ps.label_rows(t, b, a)
            va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]
            trm = tr[np.isfinite(X[tr]).all(1)]; vam = va[np.isfinite(X[va]).all(1)]
            pred = ridge(X[trm], t.values[trm], X[vam]); y = t.values[vam]
            rows.append({"fold": f["name"], "target": f"{tn}@{h}h", "val_rows": int(len(vam)), "train_rows": int(len(trm)),
                         "mae": float(np.mean(np.abs(pred - y))), "naive_mae": float(np.mean(np.abs(y)))})
    return rows
res = {}
for name, F in fam.items():
    rows = probe(F.to_numpy(float))
    res[name] = {"columns": list(F.columns), "rows": rows, "mean_mae": float(np.mean([r["mae"] for r in rows])),
                 "naive_mean_mae": float(np.mean([r["naive_mae"] for r in rows])),
                 "beats_naive_every_fold_target": all(r["mae"] < r["naive_mae"] for r in rows),
                 "targets_beating_naive_all_folds": sorted({r["target"] for r in rows} - {r["target"] for r in rows if r["mae"] >= r["naive_mae"]})}
    res[name]["gate"] = "PASS" if res[name]["beats_naive_every_fold_target"] else "SKIPPED_NOT_BETTER_THAN_NAIVE"
out = {"schema": "lane_b_eth_variant_d_families.v1", "causality": "every family uses rows <= t only (trailing windows, bar-close convention); wavelet via the native trailing-window producer",
       "wavelet_meta": {k: wmeta[k] for k in ("method_id", "method_name", "library", "version", "window", "level", "mode")},
       "probe": "ridge alpha=1.0, fold-TRAIN standardization, naive zero return", "families": res}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: (round(v["mean_mae"], 5), v["gate"], v["targets_beating_naive_all_folds"]) for k, v in res.items()}), round(res["range"]["naive_mean_mae"], 5))
