"""Per-horizon naive gate on the lake-derived EURUSD 1h resource for both split variants (TRAIN rows only):
declared ridge (alpha 1.0, fold-TRAIN standardization); PASS = MAE and MSE strictly below the zero-return
naive in all 3 inner folds, identical scored rows across candidates. Candidates: A (OHLC), range family."""
import importlib.util, json, sys
import numpy as np, pandas as pd
PS_PATH, CSV, DECL, OUT, S1T, S2T = sys.argv[1:7]
s = importlib.util.spec_from_file_location("ps", PS_PATH); ps = importlib.util.module_from_spec(s); s.loader.exec_module(ps)
decl = json.load(open(DECL)); A = decl["all_admissible_control"]
HOURS = list(range(1, 25)) + [48, 72, 96, 120, 144]
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Am = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(Am.T @ Am + alpha * np.eye(Am.shape[1]), Am.T @ (ytr - ym)); return B @ w + ym
out = {"schema": "lane_b_lake_eurusd_probe.v1", "declaration_sha256": decl["declaration_sha256"], "splits": {}}
for split, NT in (("S1_70_15_15", int(S1T)), ("S2_prospective_reserve", int(S2T))):
    df = pd.read_csv(CSV, nrows=NT)
    ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
    C, H, L, O = (df[c].astype(float) for c in ("CLOSE", "HIGH", "LOW", "OPEN"))
    rng = pd.DataFrame({"log_high_low": np.log(H / L), "close_location": (C - L) / (H - L).replace(0, np.nan), "log_close_open": np.log(C / O)})
    cands = {"A": df[A].astype(float), "range": rng}
    Xw = pd.concat([cands["A"], rng], axis=1).to_numpy(float)
    folds = ps.inner_folds(NT, k=3, val_frac=0.15, purge=168)
    res = {}
    for h in HOURS:
        nm = "Y_s" if h <= 6 else "Y_l"
        t = ps.build_targets(C.to_numpy(float), ts, "CLOSE", "CLOSE", {nm: [h]}, 3600)[(nm, h)]
        for cn, F in cands.items():
            X = F.to_numpy(float); rows = []
            for f in folds:
                a, b = f["train"]; va0, va1 = f["val"]
                tr = ps.label_rows(t, b, a); trm = tr[np.isfinite(X[tr]).all(1)]
                va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]
                va = va[np.isfinite(Xw[va]).all(1)]
                pred = ridge(X[trm], t.values[trm], X[va]); y = t.values[va]
                rows.append({"fold": f["name"], "val_rows": int(len(va)), "mae": float(np.mean(np.abs(pred - y))), "naive_mae": float(np.mean(np.abs(y))),
                             "mse": float(np.mean((pred - y) ** 2)), "naive_mse": float(np.mean(y ** 2))})
            ok = all(r["mae"] < r["naive_mae"] and r["mse"] < r["naive_mse"] for r in rows)
            res.setdefault(cn, {})[f"{h}h"] = {"gate": "PASS" if ok else "FAIL", "mean_mae": float(np.mean([r["mae"] for r in rows])),
                                               "mean_naive_mae": float(np.mean([r["naive_mae"] for r in rows])), "rows": rows}
    out["splits"][split] = {"train_rows": NT, "train_last": str(df["DATE_TIME"].iloc[-1]),
                            "passing_horizons": {cn: [k for k, v in r.items() if v["gate"] == "PASS"] for cn, r in res.items()}, "results": res}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v["passing_horizons"] for k, v in out["splits"].items()}))
