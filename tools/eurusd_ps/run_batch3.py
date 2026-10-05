"""Lane A batch_003: causal regime features (feature-eng regime_detector),
prefix-tested transform variants of EURUSD log close, CFTC commodity
positioning (gold, WTI). Same grid, folds and targets as batch_001."""
from __future__ import annotations

import argparse
import functools
import importlib.util
import io
import json
import os
import resource
import time
import zipfile

import numpy as np
import pandas as pd
from scipy import signal

from . import contract as C
from . import profile as P
from . import sources as S
from . import variants as V
from .asof import asof_last
from .features import LAKE_LICENCE, PRICE_SRC, _meta
from .run_batch import jdump, sha

REGIME_KEEP = ["atr_pct", "atr_ratio", "bb_width_pct", "price_vs_ema50", "ema_alignment", "di_spread", "macd_hist", "roc_12"]
REGIME_NOT_ADMITTED = {
    "classify_regime_v3": "NOT_ADMITTED: centroids fitted on the full 15-year EURUSD history (includes validation/test) and clusters "
                          "mapped to regimes by forward returns -> test-period and target leakage",
    "classify_regime / classify_regime_v2": "PENDING_PROVENANCE: fixed thresholds of unknown origin (described as hand-tuned or GA-optimized); "
                                            "a TRAIN-fold-fitted regime is a PS2/PS4 variant",
    "duplicates_of_batch_001": "rsi, stoch_k, adx, plus_di, minus_di, bb_position, atr_raw, bands and EMAs duplicate ta.* or are price levels",
}
COT_MARKETS = {"088691": "gold_comex", "067651": "wti_nymex"}
COT_LIC = "CFTC (US government work, public domain)"


@functools.lru_cache(maxsize=4)
def _tapers(n):
    return signal.windows.dpss(n, NW=3, Kmax=5)


def variant_features(lc: pd.Series, pos: np.ndarray, q: float) -> tuple[pd.DataFrame, list[dict]]:
    x = lc.to_numpy(float)
    W = V.W
    out = {k: np.full(len(pos), np.nan) for k in
           ["tv.wav_d1", "tv.wav_d2", "tv.wav_d3", "tv.wav_d4", "tv.wav_d5", "tv.mt_band_6_48h", "tv.hilbert_amp",
            "tv.stl_dev", "tv.stl_seasonal", "tv.kalman_dev"]}
    from statsmodels.tsa.seasonal import STL
    m, _ = V._kalman(x, q, q)
    f = np.fft.rfftfreq(W - 1, d=1.0); band = (f >= 1 / 48) & (f <= 1 / 6)
    for j, i in enumerate(pos):
        if i < 24 * 14:
            continue
        seg = x[i - W + 1: i + 1]
        wv = V._haar_modwt_last(seg - seg.mean())
        for k in range(5):
            out[f"tv.wav_d{k + 1}"][j] = wv[k]
        d = np.diff(seg)
        p = np.mean([np.abs(np.fft.rfft(d * tp)) ** 2 for tp in _tapers(len(d))], axis=0)
        out["tv.mt_band_6_48h"][j] = p[band].sum() / p[1:].sum()
        out["tv.hilbert_amp"][j] = np.abs(signal.hilbert(signal.detrend(seg)))[-1]
        s2 = x[i - 24 * 14 + 1: i + 1]
        r = STL(s2, period=24, robust=False).fit()
        out["tv.stl_dev"][j] = s2[-1] - r.trend[-1]
        out["tv.stl_seasonal"][j] = r.seasonal[-1]
        out["tv.kalman_dev"][j] = x[i] - m[i]
    ident = {"tv.wav_d": "tv.wavelet_modwt_haar_causal", "tv.mt_": "tv.multitaper_trailing", "tv.hilbert": "tv.hilbert_trailing_lastsample",
             "tv.stl": "tv.stl_trailing_lastsample", "tv.kalman": "tv.kalman_local_level_filter"}
    meta = []
    for k in out:
        vid = next(v for p_, v in ident.items() if k.startswith(p_))
        meta.append(_meta(k, "transform_variant", PRICE_SRC, "log-price units", 24 * 14 if "stl" in k else W, "UTC hourly bar", "bar end",
                          f"{vid} applied to ln(close); output {k}", note="variant identity and prefix test in batch_001 transform_variants.csv"))
    return pd.DataFrame(out, index=lc.index[pos]), meta


def cot_features(zips: list[str], decision: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    frames = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                d = pd.read_csv(io.BytesIO(zf.read(n)), dtype={"CFTC_Contract_Market_Code": str}, low_memory=False)
                frames.append(d[d["CFTC_Contract_Market_Code"].str.strip().isin(COT_MARKETS)])
    d = pd.concat(frames)
    d["asof"] = pd.to_datetime(d["Report_Date_as_YYYY-MM-DD"])
    d["avail_utc"] = (d["asof"] + pd.Timedelta(days=7)).dt.tz_localize("UTC")
    d = d[d["avail_utc"] < C.READ_END]
    f = pd.DataFrame(index=decision); meta = []; cols = []
    for code, nm in COT_MARKETS.items():
        s = d[d["CFTC_Contract_Market_Code"].str.strip() == code].drop_duplicates("asof").sort_values("asof")
        net = (s["M_Money_Positions_Long_All"].astype(float) - s["M_Money_Positions_Short_All"].astype(float)) / s["Open_Interest_All"].astype(float)
        for fid, val, tr in ((f"cot.{nm}.mm_net_oi", net, "managed-money (long-short)/open interest"),
                             (f"cot.{nm}.mm_net_oi_chg_1w", net.diff(), "1-report change of the above")):
            v, _ = asof_last(decision, s["avail_utc"], val, max_age_h=24 * 14)
            f[fid] = v
            meta.append(_meta(fid, "cftc_positioning", "lake:alternative_data/cot_reports/cftc_disaggregated", "fraction", 24 * 14,
                              "report as-of date (Tuesday)", "as-of + 7 days 00:00 UTC (conservative; nominal release Friday 15:30 ET)",
                              tr, COT_LIC, frequency="weekly"))
        cols.append({"source": "cftc_disaggregated", "market_code": code, "market": nm, "reports_read_until_read_end": int(len(s)),
                     "span": [str(s["asof"].min()), str(s["asof"].max())]})
    return f, meta, cols


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--regime-module", required=True); ap.add_argument("--code-commit", default="UNCOMMITTED")
    ap.add_argument("--split", default="train", choices=["train", "validation_2024"])
    a = ap.parse_args(argv)
    t_all = time.time()
    split_doc = C.configure_split(a.split)
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "READY")):
        raise SystemExit("REFUSED: batch already READY")
    if a.split != "train" and "validation_2024" not in os.path.abspath(a.out):
        raise SystemExit("REFUSED: validation output must live under a validation_2024 directory")
    base_ready = json.load(open(os.path.join(a.base, "READY")))
    folds = json.load(open(os.path.join(a.base, "folds.json")))["folds"]
    grid = pd.read_parquet(os.path.join(a.base, "targets_train.parquet"), columns=["row_id", "t_decision_utc"])
    decision = pd.DatetimeIndex(grid["t_decision_utc"])
    C.guard_rows(decision)
    cost = {}
    t0 = time.time()
    b5, _ = S.load_lake_5m(os.path.join(a.inputs, "eurusd_5m.parquet"))
    hourly = S.hourly_from_5m(b5)
    base_lc = pd.read_parquet(os.path.join(a.base, "features_train.parquet"), columns=["px.log_close"])["px.log_close"].to_numpy()
    assert np.array_equal(np.log(hourly["close"].reindex(decision).to_numpy()), base_lc), "grid drift vs batch_001"
    cost["load_s"] = time.time() - t0
    feats, meta = [], []
    # regime (feature-eng app/regime_detector.py, imported from a pinned copy)
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("fe_regime", a.regime_module); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
    rf = rd.compute_regime_features(hourly[["open", "high", "low", "close"]])
    rfx = rf[REGIME_KEEP].add_prefix("rg.").reindex(decision)
    feats.append(rfx)
    for c in rfx.columns:
        meta.append(_meta(c, "regime_causal", PRICE_SRC, "indicator", 200 if "ema" in c else 180, "UTC hourly bar", "bar end",
                          f"feature-eng compute_regime_features['{c[3:]}'] (trailing windows)",
                          note="regime LABELS not admitted (see report)"))
    cost["regime_s"] = time.time() - t0
    # transform variants
    t0 = time.time()
    lc = np.log(hourly["close"])
    tr_mask = (hourly.index >= C.TRAIN_START) & (hourly.index < C.TRAIN_END)
    q = float(np.nanvar(lc.diff()[tr_mask]))
    pos = np.searchsorted(hourly.index.values, decision.values)
    vf, vm = variant_features(lc, pos, q)
    feats.append(vf); meta += vm
    cost["variants_s"] = time.time() - t0
    # runtime FS01 for the variants: truncate the series after a cut; rows at/before the cut must not change
    t0 = time.time()
    cut_j = len(pos) // 2
    probe = np.arange(cut_j - 40, cut_j + 1)
    vt, _ = variant_features(lc.iloc[: pos[cut_j] + 1], pos[probe], q)
    vref = vf.iloc[probe]
    kal_note = "kalman q fitted on full TRAIN (declared): identical q used in both runs"
    fs01 = {"rows_checked": int(len(probe)), "max_abs_change": float(np.nanmax(np.abs(vt.to_numpy() - vref.to_numpy()))),
            "note": kal_note}
    fs01["status"] = "PASS" if fs01["max_abs_change"] == 0 else "FAIL"
    cost["fs01_s"] = time.time() - t0
    # COT
    t0 = time.time()
    zdir = os.path.join(a.inputs, "fd", "alternative_data/cot_reports/cftc_disaggregated/raw")
    zips = sorted(os.path.join(zdir, z) for z in os.listdir(zdir) if z.endswith(".zip") and int(z[-8:-4]) <= 2023)
    prov = json.load(open(os.path.join(a.inputs, "fd_meta", "alternative_data/cot_reports/cftc_disaggregated/provenance.json")))
    psha = {os.path.basename(f["path"]): f["sha256"] for f in prov["files"]}
    custody = [{"path": os.path.basename(z), "sha256": sha(z), "custody": "DIGEST_MATCHES_PROVENANCE" if sha(z) == psha.get(os.path.basename(z)) else "DIGEST_MISMATCH"} for z in zips]
    assert all(c["custody"] == "DIGEST_MATCHES_PROVENANCE" for c in custody)
    cf, cm, ccols = cot_features(zips, decision)
    feats.append(cf); meta += cm
    cost["cot_s"] = time.time() - t0
    X = pd.concat(feats, axis=1)
    t0 = time.time()
    cells, cov = [], []
    for m in meta:
        cells += P.profile_feature(m["feature_id"], X[m["feature_id"]])
        cov += P.coverage_rows(m["feature_id"], X[m["feature_id"]], folds)
        x = X[m["feature_id"]].to_numpy(float)
        m["admissibility"] = "ADMISSIBLE"; m["train_rows"] = int(len(x)); m["train_finite"] = int(np.isfinite(x).sum())
        m["train_coverage"] = m["train_finite"] / len(x)
    cost["profile_s"] = time.time() - t0
    out = a.out
    pd.DataFrame(meta).to_csv(os.path.join(out, "admissible_features.csv"), index=False)
    jdump({"schema": "laneA_admissible_features.v1", "batch": a.batch, "features": meta}, os.path.join(out, "admissible_features.json"))
    cdf = pd.DataFrame(cells); cdf["split"] = a.split; cdf["value"] = cdf["value"].map(lambda v: json.dumps(v, default=str) if v is not None else "")
    cdf.to_csv(os.path.join(out, "profile_cells.csv"), index=False)
    cdf.pivot(index="feature_id", columns="metric", values="state")[P.METRICS].to_csv(os.path.join(out, "metric_state_matrix.csv"))
    pd.DataFrame(cov).to_csv(os.path.join(out, "coverage_matrix.csv"), index=False)
    pd.DataFrame(ccols).to_csv(os.path.join(out, "inventory_columns.csv"), index=False)
    pd.DataFrame(custody).to_csv(os.path.join(out, "custody.csv"), index=False)
    Xo = X.copy(); Xo.index.name = "t_decision_utc"; Xo.insert(0, "row_id", grid["row_id"].to_numpy())
    Xo.reset_index().to_parquet(os.path.join(out, "features_train.parquet"), index=False)
    gaps = [
        {"item": "CFTC Traders in Financial Futures (EURO FX positioning)", "state": "ACQUISITION_GAP",
         "reason": "lake holds only the disaggregated (commodity) COT report; TFF is public (no credential) but not acquired through the lake"},
        {"item": "EURUSD volume and spread", "state": "ABSENT_IN_LAKE", "reason": "HistData-derived bytes carry OHLC only; J_policy costs need a spread source"},
        {"item": "FXMacroData history before 2024-12-12", "state": "CREDENTIAL_BLOCKER_IF_ORDERED",
         "reason": "the governed snapshot starts 2024-12-12; a deeper pull needs the FXMacroData API key, which is held outside the data-gov path "
                   "(field: FXMACRODATA_API_KEY in financial-data _metadata/.env). Calendar is SELECTOR_EPISODE_SOURCE only, so not blocking selection"},
        {"item": "holiday flags", "state": "PENDING_EVIDENCE", "reason": "python-holidays is retrospective code; publication-in-advance not evidenced"},
    ]
    report = {"schema": "laneA_batch_report.v1", "batch": a.batch, "code_commit": a.code_commit, "split": split_doc,
              "kalman_q_fit_window": [str(C.TRAIN_START), str(C.TRAIN_END)],
              "base_batch": {"ready": base_ready}, "regime_module_sha256": sha(a.regime_module),
              "regime_labels_not_admitted": REGIME_NOT_ADMITTED, "kalman_q": q,
              "runtime_checks": {"FS01_variants_truncation": fs01, "grid_equal_to_batch_001": True},
              "gaps": gaps,
              "denominators": {"features": len(meta), "families": pd.Series([m["family"] for m in meta]).value_counts().to_dict(),
                               "metric_cells": len(cdf), "metric_cells_by_state": cdf["state"].value_counts().to_dict(),
                               "folds": len(folds), "decision_rows": int(len(decision)), "decision_rows_train": int(len(decision)) if a.split == "train" else 0},
              "cost": cost | {"wall_s": time.time() - t_all, "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
              "host_role": "worker_a", "hardware": "CPU only"}
    jdump(report, os.path.join(out, "batch_report.json"))
    arts = {n: sha(os.path.join(out, n)) for n in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, n)) and n not in ("digests.json", "READY")}
    jdump({"inputs_sha256": {c["path"]: c["sha256"] for c in custody} | {"eurusd_5m.parquet": sha(os.path.join(a.inputs, "eurusd_5m.parquet")),
                                                                         "regime_detector.py": report["regime_module_sha256"]},
           "artifacts_sha256": arts, "code_commit": a.code_commit, "base_batch_digests_sha256": base_ready["digests_sha256"]},
          os.path.join(out, "digests.json"))
    with open(os.path.join(out, "READY"), "w") as fh:
        fh.write(json.dumps({"batch": a.batch, "digests_sha256": sha(os.path.join(out, "digests.json")),
                             "written_utc": pd.Timestamp.now(tz="UTC").isoformat()}) + "\n")
    print(json.dumps(report["denominators"] | {"cost": report["cost"], "fs01": fs01}, default=str))


if __name__ == "__main__":
    main()
