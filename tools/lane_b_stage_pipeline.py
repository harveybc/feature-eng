"""Lane B staged family screen (order e9d689e6 §4.B): availability filter -> cheap metrics -> redundancy -> predictive
utility (ridge vs zero-return naive AND intercept-only control, per fold, MAE and MSE) -> hand-offs (causal ladder C2,
extraction capacity M01). Families are computed causally (rows <= t); anything fitted (regime thresholds, Kalman noise
ratio) is fitted on each inner fold's TRAIN rows only. No cartesian product: each family is screened alone."""
import importlib.util, json, sys, time
import numpy as np, pandas as pd
cfg = json.load(open(sys.argv[1])); PS = sys.argv[2]; NW = sys.argv[3]; OUT = sys.argv[4]
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ps, nw = load("ps", PS), load("nw", NW)
df = pd.read_csv(cfg["csv"], nrows=cfg["n_train"]); n = len(df)
ts = ps.timestamps_to_seconds(df["DATE_TIME"].tolist(), "%Y-%m-%d %H:%M:%S")
C, H, L, O = (df[k].astype(float) for k in ("CLOSE", "HIGH", "LOW", "OPEN"))
lc = np.log(C); r1 = lc.diff(); bars_day = 86400 // cfg["step"]
folds = ps.inner_folds(n, k=3, val_frac=0.15, purge=cfg["purge"])

def ema(x, span): return x.ewm(span=span, adjust=False).mean()
def fam_returns(_end): return pd.DataFrame({f"ret_{k}": lc - lc.shift(k) for k in (1, 2, 4, 8, bars_day)})
def fam_technical(_end):
    d = C.diff(); up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    macd = ema(C, 12) - ema(C, 26); tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
    m20, s20 = C.rolling(20).mean(), C.rolling(20).std()
    return pd.DataFrame({"rsi14": 100 - 100 / (1 + up / dn), "macd_hist": macd - ema(macd, 9), "atr14_over_close": tr.ewm(alpha=1 / 14, adjust=False).mean() / C,
                         "bb_position": (C - (m20 - 2 * s20)) / (4 * s20), "roc10": C / C.shift(10) - 1})
def fam_vol_regime(end):
    rv = r1.rolling(bars_day).std(); lo, hi = np.nanquantile(rv.iloc[:end], [1 / 3, 2 / 3])   # thresholds from fold TRAIN only
    return pd.DataFrame({"rv_day": rv, "regime_high": (rv > hi).astype(float).where(rv.notna()), "regime_low": (rv < lo).astype(float).where(rv.notna())})
def fam_wavelet(_end):
    w, _ = nw.compute_wavelet_native(pd.DataFrame({"timestamp": df["DATE_TIME"], "close": lc}), window=128, wavelet="db4", level=3,
                                     frozen_from={"fold_id": "declared", "train_end": "fold"}, sample_interval_seconds=cfg["step"])
    return w.drop(columns=["timestamp"])
def fam_kalman(end):
    d = r1.iloc[1:end].dropna().to_numpy(); v = d.var(); rho = np.corrcoef(d[1:], d[:-1])[0, 1]
    r = max(0.0, -rho * v); q = max(v - 2 * r, 1e-12 * v)        # local level: var(dx)=q+2r, cov lag1=-r (fold TRAIN moments)
    x = lc.to_numpy(); m = np.empty(n); p = 1.0; mu = x[0]; innov = np.empty(n)
    for i in range(n):
        p = p + q; k = p / (p + r) if (p + r) > 0 else 1.0; e = x[i] - mu; mu = mu + k * e; p = (1 - k) * p; m[i] = mu; innov[i] = e
    return pd.DataFrame({"kf_deviation": x - m, "kf_innovation": innov, "kf_level_change": pd.Series(m).diff().to_numpy()}), {"q": q, "r": r}
def fam_calendar(_end):
    raw = pd.to_datetime(df["DATE_TIME"])
    if cfg.get("stamp_tz"):   # clock inferred from evidence (tools/infer_fx_clock.py); features describe the bar MIDPOINT in UTC
        loc = raw.dt.tz_localize(cfg["stamp_tz"], ambiguous="NaT", nonexistent="NaT")   # the 1-2 DST-transition stamps per year become NaN rows
        t = (loc + pd.Timedelta(minutes=-30 if cfg.get("stamp_is", "END") == "END" else 30) * (cfg["step"] / 3600)).dt.tz_convert("UTC")
    else:
        t = raw.dt.tz_localize("UTC")
    h = t.dt.hour + t.dt.minute / 60; dw = t.dt.dayofweek
    out = {"hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24), "dow_sin": np.sin(2 * np.pi * dw / 7), "dow_cos": np.cos(2 * np.pi * dw / 7)}
    for name, tz, a_, b_ in (("london", "Europe/London", 8, 16), ("newyork", "America/New_York", 8, 17), ("tokyo", "Asia/Tokyo", 9, 18)):
        lt = t.dt.tz_convert(tz); out[f"session_{name}"] = ((lt.dt.hour >= a_) & (lt.dt.hour < b_)).astype(float).where(t.notna())   # DST-safe via zoneinfo
    out["overlap_london_newyork"] = out["session_london"] * out["session_newyork"]
    return pd.DataFrame(out)
FAMS = {"returns": fam_returns, "technical": fam_technical, "vol_regime": fam_vol_regime, "native_wavelet": fam_wavelet, "kalman_local_level": fam_kalman, "calendar_session": fam_calendar}
def ridge(Xtr, ytr, Xva, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A = (Xtr - mu) / sd; B = (Xva - mu) / sd; ym = ytr.mean()
    return B @ np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym)) + ym
targets = {h: list(ps.build_targets(C.to_numpy(float), ts, "CLOSE", "CLOSE", {("Y_s" if h <= 6 else "Y_l"): [h]}, cfg["step"]).values())[0] for h in cfg["hours"]}
out = {"schema": "lane_b_stage_outcome.v1", "dataset": cfg["name"], "families": {}, "ledger_rows": []}
for fam, fn in FAMS.items():
    if cfg.get("only_families") and fam not in cfg["only_families"]:
        continue
    rec = {"stage_1_availability": None}
    if fam == "calendar_session" and not cfg.get("timezone_utc") and not cfg.get("stamp_tz"):
        rec["stage_1_availability"] = "PENDING: stored timestamp timezone undocumented; hour/session features are not DST-safe without it"
        out["families"][fam] = rec
        out["ledger_rows"].append(dict(dataset=cfg["name"], candidate=fam, target="cumulative log return", horizon="all", split="inner_TRAIN", status="PENDING",
                                       reason=rec["stage_1_availability"], rows={}, cost={}, source="stage pipeline")); continue
    rec["stage_1_availability"] = "PASS: computed from bar t and earlier (bar-close convention); fitted parts use fold TRAIN only"
    if fam == "calendar_session" and cfg.get("stamp_tz"):
        rec["stage_1_availability"] += f"; clock {cfg['stamp_tz']} stamped at bar {cfg.get('stamp_is', 'END')} per {cfg.get('clock_evidence')}"
    t0 = time.process_time()
    built = {}
    for f in folds:
        r = fn(f["train"][1]); built[f["name"]] = r[0] if isinstance(r, tuple) else r
        if isinstance(r, tuple): rec.setdefault("fitted_params", {})[f["name"]] = r[1]
    X0 = built[folds[-1]["name"]]; a_, b_ = folds[-1]["train"]
    cheap = {c: {"missing_fraction": float(X0[c].iloc[a_:b_].isna().mean()), "constant": bool(np.nanstd(X0[c].iloc[a_:b_].to_numpy()) == 0),
                 "acf1": float(X0[c].iloc[a_:b_].autocorr(1)) if X0[c].iloc[a_:b_].notna().sum() > 50 else None} for c in X0.columns}
    rec["stage_2_cheap_metrics"] = cheap
    keep = [c for c in X0.columns if cheap[c]["missing_fraction"] < 0.2 and not cheap[c]["constant"]]
    corr = X0[keep].iloc[a_:b_].corr().abs().to_numpy() if len(keep) > 1 else np.zeros((1, 1)); drop = set()
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            if keep[i] not in drop and keep[j] not in drop and corr[i, j] >= 0.95: drop.add(keep[j])
    rep = [c for c in keep if c not in drop]
    rec["stage_3_redundancy"] = {"kept": rep, "dropped_as_redundant_abs_corr_ge_0.95": sorted(drop), "excluded_cheap": [c for c in X0.columns if c not in keep]}
    util = {}
    for h, t in targets.items():
        rows = []
        for f in folds:
            X = built[f["name"]][rep].to_numpy(float); a, b = f["train"]; va0, va1 = f["val"]
            tr = ps.label_rows(t, b, a); tr = tr[np.isfinite(X[tr]).all(1)]
            va = np.arange(va0, va1); va = va[(t.label_index[va] >= 0) & (t.label_index[va] < va1) & np.isfinite(t.values[va])]; va = va[np.isfinite(X[va]).all(1)]
            if len(tr) < 50 or len(va) < 30 or not rep:
                rows.append({"fold": f["name"], "status": "NOT_RUN"}); continue
            y = t.values[va]; pred = ridge(X[tr], t.values[tr], X[va]); m0 = float(t.values[tr].mean())
            rows.append({"fold": f["name"], "train_rows": int(len(tr)), "val_rows": int(len(va)), "mae": float(np.mean(np.abs(pred - y))), "mse": float(np.mean((pred - y) ** 2)),
                         "zero_mae": float(np.mean(np.abs(y))), "zero_mse": float(np.mean(y ** 2)), "mean_only_mae": float(np.mean(np.abs(m0 - y))), "mean_only_mse": float(np.mean((m0 - y) ** 2))})
        ok = all("mae" in r and r["mae"] < r["zero_mae"] and r["mse"] < r["zero_mse"] and r["mae"] < r["mean_only_mae"] and r["mse"] < r["mean_only_mse"] for r in rows)
        util[f"{h}h"] = {"gate": "PASS" if ok else "FAIL", "rows": rows}
        out["ledger_rows"].append(dict(dataset=cfg["name"], candidate=fam, target="cumulative log return (elapsed seconds)", horizon=f"{h}h", split="inner_TRAIN",
                                       status="ELIGIBLE" if ok else "REJECTED",
                                       reason=("below BOTH the zero-return naive and the intercept-only control in every fold (MAE and MSE); next: causal ladder (C2), extraction capacity (M01)"
                                               if ok else "not below both controls in every fold"),
                                       rows={"val_rows": sum(r.get("val_rows", 0) for r in rows), "train_rows": sum(r.get("train_rows", 0) for r in rows), "channels": len(rep)},
                                       cost={"cpu_seconds_family_total": None}, source=f"stage pipeline {cfg['name']}"))
    rec["stage_4_utility"] = util; rec["cpu_seconds"] = time.process_time() - t0
    for lr in out["ledger_rows"]:
        if lr["candidate"] == fam and lr["cost"] == {"cpu_seconds_family_total": None}:
            lr["cost"] = {"cpu_seconds_family_total": rec["cpu_seconds"], "measured": True}
    rec["stage_5_causal_ladder"] = "hand eligible rows to C2" ; rec["stage_6_extraction_capacity"] = "hand eligible rows to M01 (branch width/cost)"
    out["families"][fam] = rec
for fam, why in () if cfg.get("only_families") else (("multitaper", "no timing-verified producer (Stage 2.3 multitaper VIOLATED for restart/prefix; lane C)"),
                 ("hilbert", "no timing-verified producer (Stage 2.3 Hilbert VIOLATED for restart/prefix; lane C)"),
                 ("stl", "no causal rolling STL producer implemented; a global STL is non-causal")):
    out["ledger_rows"].append(dict(dataset=cfg["name"], candidate=fam, target="cumulative log return", horizon="all", split="inner_TRAIN", status="PENDING", reason=why, rows={}, cost={}, source="stage pipeline"))
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({f: [h for h, v in r.get("stage_4_utility", {}).items() if v["gate"] == "PASS"] for f, r in out["families"].items()}))
