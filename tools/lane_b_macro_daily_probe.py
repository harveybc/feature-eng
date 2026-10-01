"""Daily macro/index families for EURUSD and GBPUSD (lake 1h derivatives, S1 TRAIN only -> daily bars).
Feature for day D = one-day change of each series, taken AS OF D-1 (value dated <= D-1, backward as-of join):
log change for Yahoo closes, first difference for FRED levels. Gate per horizon 1..6 d: declared ridge, MAE and MSE
strictly below BOTH the zero-return naive AND the intercept-only (TRAIN-mean drift) control in all 3 inner folds."""
import importlib.util, json, sys
import numpy as np, pandas as pd
PS, OUT = sys.argv[1], sys.argv[2]; assets = json.loads(sys.argv[3]); panels = json.loads(sys.argv[4])
s = importlib.util.spec_from_file_location("ps", PS); ps = importlib.util.module_from_spec(s); s.loader.exec_module(ps)
FAM = {"fred_rates": ["dff", "dfii10", "dfii5", "dgs1", "dgs10", "dgs2", "dgs20", "dgs30", "dgs3mo", "dgs5", "dgs7", "dprime", "dtb3", "t10y2y", "t10y3m", "t10yie", "t5yie", "t5yifr"],
       "fred_credit": ["aaa10y", "baa10y", "bamlc0a0cm", "bamlc0a0cmey", "bamlh0a0hym2", "bamlh0a0hym2ey", "aaa", "baa"],
       "fred_stress": ["tedrate", "vixcls", "stlfsi4", "nfci", "anfci"], "fred_fx_indices": ["dtwexafegs", "dtwexb", "dtwexbgs", "dtwexemegs", "dtwexm"]}
fr = pd.read_csv(panels["fred"], parse_dates=["DATE"]).set_index("DATE"); ya = pd.read_csv(panels["yahoo"], parse_dates=["DATE"]).set_index("DATE")
def per_series_change(df, log):
    """change computed on each series' OWN observations (no NaN-bridging across the union calendar)"""
    out = {}
    for c in df.columns:
        v = df[c].dropna()
        out[c] = (np.log(v.where(v > 0)).diff() if log else v.diff())
    return pd.DataFrame(out)
fams = {}
for k, names in FAM.items():
    cols = [f"fred__{n}" for n in names if f"fred__{n}" in fr.columns]
    if cols: fams[k] = per_series_change(fr[cols], log=False)
yi = [c for c in ya.columns if any(x in c for x in ("000001_ss", "axjo", "bsesn", "bvsp", "fchi", "ftse", "gdaxi", "gsptse", "hsi", "ks11", "mxx", "n225", "stoxx50e", "twii"))]
fams["yahoo_indices"] = per_series_change(ya[yi], log=True); fams["yahoo_commodities"] = per_series_change(ya[[c for c in ya.columns if c not in yi]], log=True)
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Am = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    w = np.linalg.solve(Am.T @ Am + alpha * np.eye(Am.shape[1]), Am.T @ (ytr - ym)); return B @ w + ym
out = {"schema": "lane_b_macro_daily_probe.v1", "definition": __doc__, "families": {k: list(v.columns) for k, v in fams.items()}, "assets": {}}
for an, ad in assets.items():
    h1 = pd.read_csv(ad["csv"], nrows=ad["n_train"]); h1["DATE_TIME"] = pd.to_datetime(h1["DATE_TIME"])
    h1 = h1[h1["DATE_TIME"] < h1["DATE_TIME"].iloc[-1].normalize()]
    g = h1.groupby(h1["DATE_TIME"].dt.normalize()); d = pd.DataFrame({"CLOSE": g["CLOSE"].last(), "N": g["CLOSE"].count()}); d = d[d["N"] >= 12]
    ts = ((pd.Series(d.index) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    folds = ps.inner_folds(len(d), k=3, val_frac=0.15, purge=26); res = {}
    for fk, F in fams.items():
        Fl = F.copy(); Fl.index = Fl.index + pd.Timedelta(days=1)     # value dated D-1 becomes usable at D
        X = pd.merge_asof(pd.DataFrame(index=d.index).reset_index().rename(columns={"DATE_TIME": "D"}).sort_values("D"),
                          Fl.reset_index().rename(columns={"DATE": "D"}).sort_values("D"), on="D", direction="backward").drop(columns=["D"])
        keep = [c for c in X.columns if X[c].notna().mean() > 0.8]
        if not keep:
            res[fk] = {"channels_used": [], "channels_dropped_low_coverage": list(X.columns), "status": "NOT_RUN_NO_CHANNEL_WITH_COVERAGE"}
            continue
        Xn = X[keep].to_numpy(float)
        for hd in range(1, 7):
            t = list(ps.build_targets(d["CLOSE"].to_numpy(float), ts, "CLOSE", "CLOSE", {"Y_l": [hd * 24]}, 86400).values())[0]
            rows = []
            for f in folds:
                a, b = f["train"]; va0, va1 = f["val"]
                tr = ps.label_rows(t, b, a); tr = tr[np.isfinite(Xn[tr]).all(1)]
                va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]; va = va[np.isfinite(Xn[va]).all(1)]
                if len(tr) < 50 or len(va) < 30:
                    rows.append({"fold": f["name"], "status": "NOT_RUN_TOO_FEW_ROWS"}); continue
                pred = ridge(Xn[tr], t.values[tr], Xn[va]); y = t.values[va]; m0 = float(t.values[tr].mean())
                rows.append({"fold": f["name"], "val_rows": int(len(va)), "mae": float(np.mean(np.abs(pred - y))), "naive_mae": float(np.mean(np.abs(y))),
                             "mse": float(np.mean((pred - y) ** 2)), "naive_mse": float(np.mean(y ** 2)),
                             "mean_only_mae": float(np.mean(np.abs(m0 - y))), "mean_only_mse": float(np.mean((m0 - y) ** 2))})
            ok = all("mae" in r and r["mae"] < r["naive_mae"] and r["mse"] < r["naive_mse"]
                     and r["mae"] < r["mean_only_mae"] and r["mse"] < r["mean_only_mse"] for r in rows)
            res.setdefault(fk, {"channels_used": keep, "channels_dropped_low_coverage": [c for c in X.columns if c not in keep]})[f"{hd}d"] = {"gate": "PASS" if ok else "FAIL", "rows": rows}
    out["assets"][an] = {"daily_train_bars": len(d), "results": res, "passing": {fk: [h for h, v in r.items() if isinstance(v, dict) and v.get("gate") == "PASS"] for fk, r in res.items()}}
ev = sorted({c for a in out["assets"].values() for r in a["results"].values() for c in r["channels_used"]})
out["evaluated_channels"] = ev; out["evaluated_count"] = len(ev)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({"evaluated": len(ev), "passing": {k: v["passing"] for k, v in out["assets"].items()}}))
