#!/usr/bin/env python3
"""Transform-family ledger per eligible input type (addendum 256c61a6, order b).

Rows are written through source_transform_ledger.transform_row so NOT_APPLICABLE always carries a
domain justification and proxy method ids never certify a native family. Method ids, units and
timing follow lane C's dossier (financial-data satoshi/c-method-semantics-20261001 @ 99766205,
METHOD_SEMANTICS_DOSSIER_2026_10_01.md sections 2 and 5); lane C owns those semantics and tests.
Materialized counts come from the discovery snapshot (metadata only).
"""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stl", HERE / "source_transform_ledger.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

C = "financial-data satoshi/c-method-semantics-20261001 99766205 METHOD_SEMANTICS_DOSSIER_2026_10_01.md §2 (per-producer table) and §5"
REG = "feature-eng satoshi/c-method-semantics-20261001 5f25cc1 (regime fold-boundary tests)"
CARD = "predictor satoshi/c-contracts-20261001 f509955f (representation_candidate_card.v1 method block)"
NA_OHLC = "an OHLC/volume indicator needs open, high, low, close (and volume); this input is a single value per observation"
NA_EVENT = "irregular event records, not a regularly sampled level: spectral/OHLC transforms are undefined on them"
LIT = "literature reproduction keeps the author protocol exactly (orders); alternative transforms are separate variants"


def build(disc_path, out_dir):
    files = json.load(open(disc_path))["files"]
    fam = collections.Counter(f["family"] for f in files if f["path"].startswith("features/trading_asset_features/"))
    xs = sum(f["path"].startswith("features/cross_source_statistical/") for f in files)
    xf = sum(f["path"].startswith("features/cross_source_features/") for f in files)
    ta = sum(f["path"].startswith("features/trading_asset_data/") for f in files)
    li = sum(f["path"].startswith("features/learned_inputs/") for f in files)
    m = lambda k: fam.get(k, 0)
    R = []

    def row(it, family, method, states, evidence, reason="", owner="lane B", next_step="", na=None, files_=0):
        r = L.transform_row(it, family, method, states, not_applicable=na, owner=owner, reason=reason, next_step=next_step)
        r.update(evidence=evidence, materialized_files=files_)
        R.append(r)

    it = "ohlc_price_bar_intraday (FX G10 HistData, crypto spot/perp Binance; 50 assets x 5m/15m/1h/4h)"
    row(it, "raw", "RAW_OHLCV_STAGE21", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
        "features/trading_asset_data 200 files; c162 TRAIN profiles for 198 contracted appearances", files_=ta,
        next_step="PS2/PS5 evaluation on Y_s/Y_l after availability contract")
    row(it, "returns_logreturns", "STAGE22_TECHNICAL_RETURNS", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
        "technical.parquet return_*/log_return_*; profiled only inside the ETH 4h model-ready view", files_=m("technical"),
        next_step="temporal test of the stage22 producer (lane C); profile per asset")
    row(it, "differences", "DIFF_1", {"applicable": 1}, "not materialized as a channel; used internally by PS1 volatility",
        reason="no recipe emitted", next_step="add to the compact grid (lane B)")
    row(it, "normalization", "TRAIN_FIT_SCALER_PREDICTOR", {"applicable": 1, "implemented": 1, "temporally_verified": 1},
        "predictor P3 rules 2026-09-14: scaler fitted on TRAIN does not move; materialized at model time", owner="M01/M02")
    row(it, "fracdiff", "NATIVE_FIXED_WIDTH_FRACDIFF_d0.4_0.6_0.8", {"applicable": 1, "implemented": 1, "materialized": 1},
        f"fracdiff.parquet; past-only every bar per {C}", files_=m("fracdiff"), owner="lane C (temporal test) / lane B (profile)",
        next_step="lane C prefix/restart test, then TRAIN profile")
    for fam_, cols in (("technical_trend", "sma/ema/close_sma_ratio/macd/trend_slope"), ("technical_momentum", "rsi/stoch/williams/cci/roc/mom"),
                       ("technical_volatility", "bb/atr/natr/hist_vol"), ("technical_volume", "obv/volume_sma/vwap/mfi")):
        row(it, fam_, "STAGE22_TECHNICAL (financial-data) ; FEATURE_ENG_PANDAS_TA (separate producer)",
            {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
            f"technical.parquet ({cols}); feature-eng's own pandas_ta set is lookahead-tested (tests/test_real_indicator_causality.py) "
            "but that is NOT the producer of the materialized files; profiled only in the ETH 4h view",
            reason="temporally_verified false for the materialized producer", files_=m("technical"),
            owner="lane C (temporal) / lane B (profile)", next_step="temporal test of stage22 producer; per-asset profile; compact parameter grid")
    row(it, "statistical_rolling_moments", "STAGE22_STATISTICAL", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
        "statistical.parquet; profiled only in the ETH 4h view", files_=m("statistical"), next_step="temporal test; per-asset profile")
    row(it, "market_state_rule", "RULE_VOL_REGIME_HIGH_LOW ; FEATURE_ENG_REGIME_V1_V2_RULES", {"applicable": 1, "implemented": 1, "profiled": 1},
        "ETH 4h view vol_regime_high/low profiled; no materialized hierarchical state provider", reason="not materialized per asset",
        next_step="emit with available history only; durations/transitions not produced")
    row(it, "market_state_learned_hmm", "LEARNED_GAUSSIAN_HMM_3STATE_FULLSERIES", {"applicable": 1, "implemented": 1, "materialized": 1, "excluded": 1},
        "stage25 regime_labels(); lane B SOTA_PRODUCERS.v1.json", reason="fit and scaler on the whole series incl. holdout; Viterbi/forward-backward smoothing uses observations after t: NON_CAUSAL",
        files_=m("sota_hmm_regime"), owner="lane C", next_step="refit per TRAIN fold, forward filtering only")
    row(it, "market_state_learned_gmm", "FEATURE_ENG_REGIME_V3_GMM (fixed centroids, 15y EURUSD, forward-return label map)",
        {"applicable": 1, "implemented": 1, "excluded": 1}, f"feature-eng d081d0f app/regime_detector.py:247-302; LEARNED, fit period UNKNOWN; {REG}; {C}",
        reason="fitted outside any fold and labels chosen from forward returns: refused until refit on fold TRAIN", owner="lane C")
    row(it, "wavelet", "NATIVE_DWT_DB4_TRAILING_WINDOW", {"applicable": 1, "implemented": 1},
        "financial-data satoshi/b-native-wavelet-20261001 028f42847 _scripts/lib/native_wavelet.py (tests 8/8 on coordinator); "
        "lane C repointed test green on that tree (financial-data satoshi/c-method-semantics-20261001 1c2ccc88f); the Stage 2.3 worker still emits only the proxy",
        reason="implemented only: NOT materialized on lake data, NOT temporally verified on lake data, NOT profiled, NOT evaluated",
        owner="lane B (materialize/profile) / lane C (verdict)", next_step=f"causal trailing-window pywt DWT producer with frozen TRAIN windows, card method block per {CARD}; the successor must not keep the worker's hard-coded user-home PROJECT3_ROOT default (line 22), and the existing worker is not edited in place")
    row(it, "wavelet", "PROXY_ROLLING_MEAN_MULTISCALE_16_32_64_128", {"applicable": 1, "implemented": 1, "materialized": 1},
        f"wavelet.parquet; {C}: MULTISCALE_ROLLING_MEAN_PROXY, proxy_of DWT_DB4", files_=m("wavelet"), owner="lane C (temporal) / lane B",
        next_step="keep with honest method id; never certifies native wavelet")
    row(it, "hilbert", "NATIVE_SCIPY_HILBERT_TRAILING_WINDOW (ht_inst_freq == ht_phase_difference, rad/bar)",
        {"applicable": 1, "implemented": 1, "materialized": 1, "excluded": 1}, f"hilbert.parquet; {C}",
        reason="temporal verification VIOLATED for n<1,000 rows and for restart/chunked replay (grid anchored at input start); radians per bar, no physical interval; feature age up to step-1 bars (forward-filled endpoints)", files_=m("hilbert"),
        owner="lane C", next_step="freeze window from TRAIN, prefix/restart invariance test, declare units and feature age")
    row(it, "multitaper", "NATIVE_DPSS (cycles per bar)", {"applicable": 1, "implemented": 1, "materialized": 1, "excluded": 1},
        f"multitaper.parquet; {C}", reason="temporal verification VIOLATED for n<1,000 rows and restart/chunked replay; cycles per bar, no physical interval; endpoints every 512/256/96/32 bars, feature age up to step-1 bars",
        files_=m("multitaper"), owner="lane C", next_step="freeze windows from TRAIN; restart test")
    row(it, "stl", "PREDICTOR_STL_PREPROCESSOR", {"applicable": 1, "implemented": 1, "deferred": 1},
        "predictor stl preprocessor exists; M03 profiles compute STL strength descriptively only", reason="not materialized as causal channels; a global STL is non-causal",
        owner="lane C", next_step="rolling causal STL recipe or explicit exclusion")
    row(it, "emd", "EMD_BACKEND_UNDECLARED (PyEMD or rolling proxy)", {"applicable": 1, "implemented": 1, "materialized": 1, "excluded": 1},
        f"emd.parquet; {C}: silent proxy fallback when PyEMD is absent (absent on the coordinator) or n>80,000; no backend recorded in sidecars or logs",
        reason="method id of the materialized bytes is unknown", files_=m("emd"), owner="lane C", next_step="record backend per file or regenerate with a declared method")
    row(it, "calendar_time", "HOD_DOW_SIN_COS", {"applicable": 1, "implemented": 1, "profiled": 1}, "d4 hod/dow columns profiled",
        reason="not materialized per asset in financial-data", next_step="emit from the bar timestamp with declared timezone")
    row(it, "event_overlap", "CALENDAR_JOIN", {"applicable": 1, "deferred": 1}, "needs the FXMacroData availability contract",
        reason="no admissible calendar availability yet", owner="lane C (study) / lane B (feature)")
    row(it, "learned_cnn_lstm", "AUTOENCODER_FIT_TRAIN_LT_2024_EARLYSTOP_ON_2024", {"applicable": 1, "implemented": 1, "materialized": 1, "deferred": 1},
        "stage24 cnn/lstm autoencoder workers; lane B SOTA_PRODUCERS.v1.json", reason="donor early-stopped on 2024, the task's validation year",
        files_=li, owner="M02", next_step="refit inside inner TRAIN folds; donor manifest")
    row(it, "learned_branch_core_ae", "PREDICTOR_MODULAR_PRETRAIN (M01/M02)", {"applicable": 1, "implemented": 1, "deferred": 1},
        "tools/modular_pretrain.py; donors trained on TSL/synthetic so far", reason="no financial donor", owner="M02")
    SP = "financial-data stage25_sota_feature_enrichment_worker.py@ef0ba661; lane B SOTA_PRODUCERS.v1.json"
    row(it, "sota_intrabar_realized", "REALIZED_MOMENTS_FROM_LOWER_FREQUENCY_LABEL_LEFT", {"applicable": 1, "implemented": 1, "materialized": 1, "deferred": 1},
        SP, reason="label-left: value stamped at bar open summarizes the whole bar (available one bar later)", files_=m("sota_intrabar_realized"),
        owner="lane C (test) / lane B", next_step="shift one bar before any join; temporal test")
    row(it, "sota_pair_spreads", "OLS_HEDGE_RATIO_FIXED_CUT_2024_01_01", {"applicable": 1, "implemented": 1, "materialized": 1, "deferred": 1},
        SP, reason="hedge ratio fitted before a hard-coded cut (full-series fallback below 500 rows), not fold-bound", files_=m("sota_pair_spreads"),
        next_step="fit inside each TRAIN fold in a successor")
    row(it, "sota_funding_term_structure", "TRAILING_FUNDING_EVENT_MEANS_ASOF_BACKWARD", {"applicable": 1, "implemented": 1, "materialized": 1, "deferred": 1},
        SP, reason="past-only only if fundingTime is the publication instant (unconfirmed)", files_=m("sota_funding_term_structure"),
        next_step="confirm fundingTime semantics; temporal test")
    for k, why in (("surprise", "a price bar has no consensus/actual; surprise belongs to the event source and reaches bars via event_overlap"),
                   ("revision", "a traded price is not revised; vendor corrections are a data-quality matter, not a revision feature")):
        row(it, k, "NA", {}, "", na=why)

    it = "daily_market_series (Yahoo indices/ETFs/commodity futures/EM FX/VIX, resampled into intraday cross-source files)"
    row(it, "raw", "CROSS_SOURCE_RESAMPLED_FFILL", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
        "cross_source_features; c162 profiles (474 Yahoo columns)", files_=xf,
        reason="daily values propagated to intraday rows: propagation is not observation (subplan 4.1)")
    row(it, "returns_logreturns", "DAILY_RETURN", {"applicable": 1}, "not materialized", next_step="compact grid")
    row(it, "statistical_rolling_moments", "CROSS_SOURCE_STATISTICAL", {"applicable": 1, "implemented": 1, "materialized": 1},
        "features/cross_source_statistical", files_=xs, next_step="producer location, temporal test, profile")
    row(it, "technical_trend", "STAGE22_TECHNICAL", {"applicable": 1, "deferred": 1}, "OHLCV exists for indices; not materialized",
        reason="not in the first compact grid")
    for k in ("wavelet", "hilbert", "multitaper", "emd"):
        row(it, k, "NOT_MATERIALIZED", {"applicable": 1, "deferred": 1}, "", reason="daily series; spectral producers read trading assets only")
    row(it, "surprise", "NA", {}, "", na="index levels are market prices, not scheduled releases with consensus")

    it = "single_macro_value (FRED, OECD, BEA)"
    row(it, "raw", "CROSS_SOURCE_RESAMPLED_FFILL", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
        "c162 profiles (FRED 31, OECD 4 columns)", reason="latest-vintage values only")
    row(it, "differences", "PCT_CHANGE", {"applicable": 1}, "not materialized")
    row(it, "statistical_rolling_moments", "CROSS_SOURCE_STATISTICAL", {"applicable": 1, "implemented": 1, "materialized": 1}, "cross_source_statistical")
    row(it, "revision", "ALFRED_VINTAGES", {"applicable": 1, "excluded": 1}, "no vintages captured", reason="REVISED_SERIES_WITHOUT_VINTAGES; first release unknown",
        owner="lane B (source)", next_step="ALFRED vintage capture for series used point-in-time")
    row(it, "surprise", "CONSENSUS_MINUS_ACTUAL", {"applicable": 1, "excluded": 1}, "no consensus source overlaps (data-gov registry)", reason="NO_CONSENSUS_AT_ALL")
    for k in ("technical_trend", "technical_momentum", "technical_volatility", "technical_volume"):
        row(it, k, "NA", {}, "", na=NA_OHLC)
    row(it, "multitaper", "NOT_MATERIALIZED", {"applicable": 1, "deferred": 1}, "", reason="monthly/quarterly series give too few TRAIN points for a stable spectrum")

    it = "event_calendar (FXMacroData announcements and release calendar)"
    row(it, "raw", "FXMACRODATA_RAW", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
        "8 contracted appearances; c162 profiled 24 columns; data-gov registry 2026-09-26", reason="availability contract absent")
    row(it, "release_timing", "FEATURE_ENG_ECONOMIC_CALENDAR (CL16: release vs available clocks)", {"applicable": 1, "implemented": 1, "temporally_verified": 1},
        "feature-eng app/economic_calendar.py with 37 tests (publication and receipt boundaries)", reason="not materialized on lake data",
        next_step="materialize after the availability contract")
    row(it, "surprise", "RELEASE_AND_AVAILABLE_SURPRISE", {"applicable": 1, "implemented": 1, "excluded": 1}, "feature-eng economic_calendar.py",
        reason="NO_CONSENSUS_AT_ALL for FXMacroData; no vintage may be invented")
    row(it, "revision", "REVISION_SURPRISE", {"applicable": 1, "implemented": 1, "excluded": 1}, "", reason="NO_REVISION_HISTORY")
    row(it, "event_overlap", "EVENT_WINDOW_FLAGS", {"applicable": 1, "deferred": 1}, "lane C PS3-C episodes", owner="lane C")
    for k in ("technical_trend", "wavelet", "hilbert", "multitaper"):
        row(it, k, "NA", {}, "", na=NA_EVENT)

    it = "onchain_count (CoinMetrics, Etherscan, Blockchain.com, mempool.space, CryptoQuant, DeFiLlama)"
    row(it, "raw", "CROSS_SOURCE_RESAMPLED_FFILL", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
        "CoinMetrics 36 columns profiled; CryptoQuant 388 appearances unprofiled", reason="CryptoQuant producer UNRESOLVED in census")
    row(it, "returns_logreturns", "LOG_DIFF", {"applicable": 1}, "not materialized")
    row(it, "statistical_rolling_moments", "CROSS_SOURCE_STATISTICAL", {"applicable": 1, "implemented": 1, "materialized": 1}, "cross_source_statistical")
    row(it, "technical_trend", "NA", {}, "", na=NA_OHLC)
    row(it, "multitaper", "NOT_MATERIALIZED", {"applicable": 1, "deferred": 1}, "", reason="not in the first grid")

    it = "positioning (FINRA short volume / short interest; CFTC COT)"
    row(it, "raw", "FINRA_FILES (outside census)", {"applicable": 1, "implemented": 1, "materialized": 1}, "4 files", reason="not in census, no TRAIN contract",
        next_step="census entry and publication clock")
    row(it, "raw", "CFTC_COT", {"applicable": 1, "deferred": 1}, "no parquet/csv retained", reason="NOT_PRESENT as data files", owner="owner decides scope")
    row(it, "technical_trend", "NA", {}, "", na=NA_OHLC)

    it = "public_benchmark_channel (TSL Electricity, Traffic, Weather)"
    row(it, "raw", "RAW_CHANNELS", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1}, "lane B full-TRAIN profiles, 1 204 channels")
    row(it, "normalization", "TSL_TRAIN_STANDARD_SCALER (author)", {"applicable": 1, "implemented": 1, "temporally_verified": 1}, "author protocol", owner="M04/M06")
    for k in ("fracdiff", "technical_trend", "wavelet", "hilbert", "multitaper", "stl", "emd", "market_state_learned_hmm"):
        row(it, k, "AUTHOR_PROTOCOL", {"applicable": 1, "excluded": 1}, "", reason=LIT, owner="M04")
    row(it, "sentinel_cleaning", "WEATHER_MINUS_9999_POLICY", {"applicable": 1, "deferred": 1}, "3 Weather channels hit -9999", reason="separate variant; M02 proposes, coordinator rules", owner="M02")

    it = "legacy_derived_views (phase-1b d4 indicators; ETH 4h tech/stat model-ready)"
    row(it, "raw", "ALREADY_DERIVED_COLUMNS", {"applicable": 1, "implemented": 1, "materialized": 1, "profiled": 1},
        "lane B full-TRAIN profiles (d4 23, ETH 83) and the ETH PS2 screen 397b67d6", reason="a screen is not an evaluation")

    it = "filings_text (SEC EDGAR)"
    row(it, "raw", "SEC_EDGAR_FILES", {"applicable": 1, "materialized": 1, "deferred": 1}, "2 files", reason="text features not in scope of this batch", owner="owner decides scope")

    out = Path(out_dir)
    fields = ["input_type", "family", "method_id", "state"] + list(L.LEDGER_STATES) + ["materialized_files", "evidence", "reason", "owner", "next_step"]
    with open(out / "TRANSFORM_LEDGER.v1.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in R:
            w.writerow({**{k: r.get(k, "") for k in fields if k not in L.LEDGER_STATES}, **r["states"]})
    fc = L.family_coverage([r for r in R if r["input_type"].startswith("ohlc_price_bar")])
    summary = {"rows": len(R), "by_state": dict(collections.Counter(r["state"] for r in R)),
               "evaluated": sum(r["states"].get("evaluated", False) for r in R),
               "selected": sum(r["states"].get("selected", False) for r in R),
               "temporally_verified": sum(r["states"].get("temporally_verified", False) for r in R),
               "native_vs_proxy_on_price_bars": fc,
               "materialized_derived_files_by_family": dict(fam)}
    (out / "TRANSFORM_LEDGER.v1.json").write_text(json.dumps({"schema": "lane_b_transform_ledger.v1", "summary": summary, "rows": R}, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
