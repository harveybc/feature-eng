"""EURUSD daily horizon-scoped gate. Daily bars are aggregated from the git-pinned 1h view (calendar date of
DATE_TIME as stored; OHLC first/max/min/last; nothing fitted). Only TRAIN rows (< row 65158) are read.
Per horizon h = 1..6 days (elapsed-second labels: a day stamp plus h*86400 must exist), the declared ridge
gate: MAE and MSE strictly below the zero-return naive in all 3 inner folds."""
import importlib.util, json, sys
import numpy as np, pandas as pd
PS_PATH, CSV, OUT = sys.argv[1:4]
s = importlib.util.spec_from_file_location("ps", PS_PATH); ps = importlib.util.module_from_spec(s); s.loader.exec_module(ps)
h1 = pd.read_csv(CSV, nrows=65158)
h1["DATE_TIME"] = pd.to_datetime(h1["DATE_TIME"])
last_day = h1["DATE_TIME"].iloc[-1].normalize()
h1 = h1[h1["DATE_TIME"] < last_day]                      # drop the partial last TRAIN day
g = h1.groupby(h1["DATE_TIME"].dt.normalize())
d = pd.DataFrame({"OPEN": g["OPEN"].first(), "HIGH": g["HIGH"].max(), "LOW": g["LOW"].min(), "CLOSE": g["CLOSE"].last(), "N_1H": g["CLOSE"].count()})
d = d[d["N_1H"] >= 12]                                    # declared: a day needs at least 12 hourly bars
n = len(d)
ts = (pd.Series(d.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
ts = ts.to_numpy(np.int64)
C, H, L, O = d["CLOSE"], d["HIGH"], d["LOW"], d["OPEN"]
lc = np.log(C); r1 = lc.diff(); hl = np.log(H / L)
rng = pd.DataFrame({"log_high_low": hl, "close_location": (C - L) / (H - L).replace(0, np.nan), "log_close_open": np.log(C / O)})
rv = pd.DataFrame({f"rv_std_ret_{k}": r1.rolling(k, min_periods=k).std() for k in (5, 20)})
for k in (5, 20):
    rv[f"parkinson_{k}"] = np.sqrt((hl ** 2).rolling(k, min_periods=k).mean() / (4 * np.log(2)))
A = d[["OPEN", "LOW", "HIGH", "CLOSE"]]
cands = {"A": A, "A+range": pd.concat([A, rng], axis=1), "range": rng, "range+rv": pd.concat([rng, rv], axis=1)}
folds = ps.inner_folds(n, k=3, val_frac=0.15, purge=26)   # declared: 20-day context + 6-day max horizon
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Am = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(Am.T @ Am + alpha * np.eye(Am.shape[1]), Am.T @ (ytr - ym)); return B @ w + ym
Xw = cands["range+rv"].join(A).to_numpy(float)
res = {}
for hd in range(1, 7):
    t = ps.build_targets(C.to_numpy(float), ts, "CLOSE", "CLOSE", {"Y_l" if hd * 24 > 6 else "Y_s": [hd * 24]}, 86400)
    t = list(t.values())[0]
    for cn, F in cands.items():
        X = F.to_numpy(float); rows = []
        for f in folds:
            a, b = f["train"]; va0, va1 = f["val"]
            tr = ps.label_rows(t, b, a)
            va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]
            va = va[np.isfinite(Xw[va]).all(1)]; trm = tr[np.isfinite(X[tr]).all(1)]
            pred = ridge(X[trm], t.values[trm], X[va]); y = t.values[va]
            rows.append({"fold": f["name"], "val_rows": int(len(va)), "mae": float(np.mean(np.abs(pred - y))), "naive_mae": float(np.mean(np.abs(y))),
                         "mse": float(np.mean((pred - y) ** 2)), "naive_mse": float(np.mean(y ** 2))})
        ok = all(r["mae"] < r["naive_mae"] and r["mse"] < r["naive_mse"] for r in rows)
        res.setdefault(cn, {})[f"{hd}d"] = {"gate": "PASS" if ok else "FAIL", "mean_mae": float(np.mean([r["mae"] for r in rows])),
                                            "mean_naive_mae": float(np.mean([r["naive_mae"] for r in rows])), "rows": rows}
summary = {cn: [h for h, v in r.items() if v["gate"] == "PASS"] for cn, r in res.items()}
out = {"schema": "lane_b_daily_horizon_gate.v1", "daily_bars": n, "first_day": str(d.index[0].date()), "last_day": str(d.index[-1].date()),
       "aggregation": "calendar date of DATE_TIME as stored; OHLC first/max/min/last of hourly bars; days with >= 12 hourly bars; TRAIN rows only",
       "gate": "per horizon: MAE and MSE strictly below the zero-return naive in all 3 inner folds (declared ridge alpha 1.0)",
       "passing_horizons_by_candidate": summary, "results": res}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({"days": n, "passing": summary}))
