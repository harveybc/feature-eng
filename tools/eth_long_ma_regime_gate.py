"""Lane B probe gate for the PS5 (M01) long-horizon finding on ETH 4h: long moving average x vol_regime_low.

Declared before measuring (lane B rules):
- data: eth_4h_long v1 manifest resource (git-pinned ETH 4h view), TRAIN rows [0, 13699) only; the declared calendar
  split (validation 2024, test 2025) is never read.
- targets: Y_l cumulative close log return at 72/96/120/144 h, located by elapsed seconds (4 h bars).
- folds: 3 expanding chronological inner folds inside TRAIN, validation 15 percent each, purge 60 rows.
- probe: ridge alpha 1.0, fold-TRAIN standardization; columns are the dataset's own (sma_100, sma_200, ema_200,
  vol_regime_low, close_sma_ratio_*), products formed row-wise (causal: both factors use rows <= t).
- scored rows: identical for every candidate and control (finite for the widest candidate).
- gate (PASS): MAE AND MSE strictly below the zero-return naive AND the intercept-only (fold-TRAIN mean) control in
  every one of the 3 folds.
- stability: each fold's scored rows split into first and second halves in time; STABLE when the model's MAE and MSE
  are below both controls in BOTH halves of EVERY fold.
- ELIGIBLE = PASS and STABLE. Candidates: VRL alone; VRL + MA + MA x VRL for each MA; all three; A83 + the three
  products; scale-free secondary (close_sma_ratio_{100,200} x VRL)."""
import hashlib, importlib.util, json, sys
import numpy as np, pandas as pd
PS, VIEW, MAN, OUT = sys.argv[1:5]
s = importlib.util.spec_from_file_location("ps", PS); ps = importlib.util.module_from_spec(s); s.loader.exec_module(ps)
man = json.load(open(MAN)); A = man["features"]; NT, PURGE, HOURS = 13699, 60, [72, 96, 120, 144]
df = pd.read_csv(VIEW, nrows=NT); assert len(df) == NT
ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
v = df["vol_regime_low"].astype(float)
P = {f"{m}_x_vrl": df[m].astype(float) * v for m in ("sma_100", "sma_200", "ema_200", "close_sma_ratio_100", "close_sma_ratio_200")}
F = pd.concat([df[A].astype(float), pd.DataFrame(P)], axis=1)
F = F.loc[:, ~F.columns.duplicated()]
cands = {"VRL": ["vol_regime_low"]}
for m in ("sma_100", "sma_200", "ema_200"):
    cands[f"VRL+{m}+{m}_x_vrl"] = ["vol_regime_low", m, f"{m}_x_vrl"]
cands["VRL+3MA+3products"] = ["vol_regime_low", "sma_100", "sma_200", "ema_200", "sma_100_x_vrl", "sma_200_x_vrl", "ema_200_x_vrl"]
cands["A83+3products"] = A + ["sma_100_x_vrl", "sma_200_x_vrl", "ema_200_x_vrl"]
cands["secondary:VRL+ratio100+ratio200+products"] = ["vol_regime_low", "close_sma_ratio_100", "close_sma_ratio_200", "close_sma_ratio_100_x_vrl", "close_sma_ratio_200_x_vrl"]
allcols = sorted({c for v_ in cands.values() for c in v_}); Xall = F[allcols].to_numpy(float)
folds = ps.inner_folds(NT, k=3, val_frac=0.15, purge=PURGE)
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Am = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    return B @ np.linalg.solve(Am.T @ Am + alpha * np.eye(Am.shape[1]), Am.T @ (ytr - ym)) + ym
def m(pred, y): return float(np.mean(np.abs(pred - y))), float(np.mean((pred - y) ** 2))
res = {}
for h in HOURS:
    t = ps.build_targets(df["CLOSE"].astype(float).to_numpy(), ts, "CLOSE", "CLOSE", {"Y_l": [h]}, 14400)[("Y_l", h)]
    for cn, cols in cands.items():
        X = F[cols].to_numpy(float); rows = []
        for f in folds:
            a, b = f["train"]; va0, va1 = f["val"]
            tr = ps.label_rows(t, b, a); tr = tr[np.isfinite(X[tr]).all(1)]
            va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]; va = va[np.isfinite(Xall[va]).all(1)]
            pred = ridge(X[tr], t.values[tr], X[va]); y = t.values[va]; m0 = np.full(len(y), float(t.values[tr].mean()))
            r = {"fold": f["name"], "train_rows": int(len(tr)), "val_rows": int(len(va))}
            r["mae"], r["mse"] = m(pred, y); r["zero_mae"], r["zero_mse"] = m(np.zeros(len(y)), y); r["mean_only_mae"], r["mean_only_mse"] = m(m0, y)
            hv = len(y) // 2; halves = []
            for sl in (slice(0, hv), slice(hv, None)):
                hh = {}; hh["mae"], hh["mse"] = m(pred[sl], y[sl]); hh["zero_mae"], hh["zero_mse"] = m(np.zeros(len(y[sl])), y[sl]); hh["mean_only_mae"], hh["mean_only_mse"] = m(m0[sl], y[sl])
                hh["below_both"] = hh["mae"] < hh["zero_mae"] and hh["mse"] < hh["zero_mse"] and hh["mae"] < hh["mean_only_mae"] and hh["mse"] < hh["mean_only_mse"]
                halves.append(hh)
            r["halves"] = halves
            r["below_zero"] = r["mae"] < r["zero_mae"] and r["mse"] < r["zero_mse"]; r["below_mean_only"] = r["mae"] < r["mean_only_mae"] and r["mse"] < r["mean_only_mse"]
            r["rel_mae_vs_zero"] = r["mae"] / r["zero_mae"] - 1
            rows.append(r)
        gate = all(r["below_zero"] and r["below_mean_only"] for r in rows); stable = all(hh["below_both"] for r in rows for hh in r["halves"])
        res.setdefault(cn, {})[f"{h}h"] = {"gate": "PASS" if gate else "FAIL", "stable": stable, "status": "ELIGIBLE" if gate and stable else "REJECTED",
                                           "verdict": "FEATURE_SIGNAL" if gate else ("DRIFT_ONLY" if all(r["below_zero"] for r in rows) else "FAIL"),
                                           "folds_below_both": sum(r["below_zero"] and r["below_mean_only"] for r in rows), "rows": rows}
out = {"schema": "lane_b_eth_long_ma_regime_gate.v1", "rule": __doc__, "manifest_canonical": man.get("manifest_sha256_canonical"),
       "view_sha256": hashlib.sha256(open(VIEW, "rb").read()).hexdigest(), "candidates": {k: {"columns": v_, "horizons": res[k]} for k, v_ in cands.items()}}
json.dump(out, open(OUT, "w"), indent=1)
for k in cands:
    print(k, {h: (x["status"], x["folds_below_both"], [round(r["rel_mae_vs_zero"] * 100, 2) for r in x["rows"]], [round(r["mae"] / r["mean_only_mae"] * 100 - 100, 2) for r in x["rows"]]) for h, x in res[k].items()})
