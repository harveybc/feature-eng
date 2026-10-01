"""Seasonal-reference probe (declared): for each horizon h, the seasonal naive of the cumulative log return over (t, t+h]
repeats the last period's bar returns (period P bars), looked up by elapsed seconds and causal (every lookup <= t).
Reported per fold on identical rows: zero-return naive, seasonal naive, and ridge on the residual (y - seasonal) added
back to the seasonal naive. Gate: ridge+seasonal MAE and MSE strictly below the zero-return naive in all 3 folds."""
import importlib.util, json, sys
import numpy as np, pandas as pd
cfg = json.load(open(sys.argv[1])); PS = sys.argv[2]; OUT = sys.argv[3]
s = importlib.util.spec_from_file_location("ps", PS); ps = importlib.util.module_from_spec(s); s.loader.exec_module(ps)
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Am = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(Am.T @ Am + alpha * np.eye(Am.shape[1]), Am.T @ (ytr - ym)); return B @ w + ym
out = {"schema": "lane_b_seasonal_reference_probe.v1", "definition": __doc__, "datasets": {}}
for d in cfg["datasets"]:
    df = pd.read_csv(d["csv"], nrows=d["n_train"])
    ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
    step, P = d["step"], d["period_bars"]
    lc = np.log(df["CLOSE"].astype(float).to_numpy())
    pos = {int(t): i for i, t in enumerate(ts)}
    def at(t):
        i = pos.get(int(t)); return lc[i] if i is not None else np.nan
    C, H, L, O = (df[c].astype(float) for c in ("CLOSE", "HIGH", "LOW", "OPEN"))
    feats = json.load(open(d["manifest"]))["features"]
    cands = {"A": df[feats].astype(float).to_numpy(), "range": pd.DataFrame({"a": np.log(H / L), "b": (C - L) / (H - L).replace(0, np.nan), "c": np.log(C / O)}).to_numpy()}
    folds = ps.inner_folds(d["n_train"], k=3, val_frac=0.15, purge=d["purge"])
    res = {}
    for hh in d["hours"]:
        hb = hh * 3600 // step
        t = ps.build_targets(C.to_numpy(float), ts, "CLOSE", "CLOSE", {("Y_s" if hh <= 6 else "Y_l"): [hh]}, step)
        t = list(t.values())[0]
        # vectorized: term k reads the one-bar return at offset (k - ceil(k/P)*P) bars, one of P distinct offsets
        sn = np.zeros(len(ts))
        for o in range(0, P):
            cnt = sum(1 for k in range(1, hb + 1) if (k - (-(-k // P)) * P) == -o)
            if not cnt:
                continue
            ta = ts - o * step; tb = ta - step
            ia = np.searchsorted(ts, ta); ib = np.searchsorted(ts, tb)
            ia_ok = (ia < len(ts)) & (ts[np.minimum(ia, len(ts) - 1)] == ta); ib_ok = (ib < len(ts)) & (ts[np.minimum(ib, len(ts) - 1)] == tb)
            r1 = np.where(ia_ok & ib_ok, lc[np.minimum(ia, len(ts) - 1)] - lc[np.minimum(ib, len(ts) - 1)], np.nan)
            sn = sn + cnt * r1
        for cn, X in cands.items():
            rows = []
            for f in folds:
                a, b = f["train"]; va0, va1 = f["val"]
                tr = ps.label_rows(t, b, a); tr = tr[np.isfinite(sn[tr]) & np.isfinite(X[tr]).all(1)]
                va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va]) & np.isfinite(sn[va])]
                va = va[np.isfinite(cands["A"][va]).all(1) & np.isfinite(cands["range"][va]).all(1)]
                y = t.values[va]; s_ = sn[va]
                pred = ridge(X[tr], t.values[tr] - sn[tr], X[va]) + s_
                rows.append({"fold": f["name"], "val_rows": int(len(va)),
                             "zero_mae": float(np.mean(np.abs(y))), "zero_mse": float(np.mean(y ** 2)),
                             "seasonal_mae": float(np.mean(np.abs(y - s_))), "seasonal_mse": float(np.mean((y - s_) ** 2)),
                             "model_mae": float(np.mean(np.abs(y - pred))), "model_mse": float(np.mean((y - pred) ** 2))})
            ok = all(r["model_mae"] < r["zero_mae"] and r["model_mse"] < r["zero_mse"] for r in rows)
            sea_ok = all(r["seasonal_mae"] < r["zero_mae"] and r["seasonal_mse"] < r["zero_mse"] for r in rows)
            res.setdefault(cn, {})[f"{hh}h"] = {"gate_model_vs_zero": "PASS" if ok else "FAIL", "seasonal_naive_beats_zero": sea_ok, "rows": rows,
                                                 "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("fold", "val_rows")}}
    out["datasets"][d["name"]] = {"period_bars": P, "train_rows": d["n_train"], "results": res,
                                  "passing": {cn: [h for h, v in r.items() if v["gate_model_vs_zero"] == "PASS"] for cn, r in res.items()},
                                  "seasonal_naive_beats_zero_at": [h for h, v in res["A"].items() if v["seasonal_naive_beats_zero"]]}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: (v["passing"], v["seasonal_naive_beats_zero_at"]) for k, v in out["datasets"].items()}))
