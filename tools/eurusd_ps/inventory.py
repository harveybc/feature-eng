"""PS0 inventory: one row per source and one row per source column, with unit,
frequency, licence, event_time, availability_time, coverage and the state of
that row in this lane (which batch, or why it is not admitted). Nothing is
omitted silently: a source that is not profiled carries its reason."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from . import contract as C

YAHOO_LIC = "Yahoo Finance terms (personal/non-commercial; no redistribution) -- research use; production use needs licence review"
FRED_LIC = "FRED (St. Louis Fed) terms; series-level third-party copyrights may apply"
HISTDATA_LIC = "HistData.com free data (research use; redistribution of raw bytes not granted)"
FXM_LIC = "FXMacroData paid subscription (credentialed API; owner's plan)"

FXM_FIELD_REQUIREMENTS = [
    # field, present?, column, state
    ("event", "indicator", "PRESENT"),
    ("country_currency", "currency", "PRESENT"),
    ("consensus", None, "ABSENT: provider payload carries no consensus/forecast (consensus_or_forecast_columns_present=[])"),
    ("actual", "val", "PRESENT (186 rows null)"),
    ("previous", None, "NOT_PROVIDED: derivable only as the prior row of the same indicator (same vintage), not the as-published previous"),
    ("revision", None, "ABSENT: single snapshot (acquired 2026-05-01); RP150 found 45 VINTAGE_UNDECIDABLE keys, no revision field"),
    ("importance", None, "ABSENT: no importance/impact field"),
    ("published_time", "announcement_datetime_utc", "PRESENT: OBSERVED_ACTUAL_PUBLICATION, tz-aware UTC (132 rows null)"),
    ("received_time", None, "FILE_GRAIN_ONLY: acquired_at 2026-05-01T23:38:43Z for the whole file; per-release receipt clock exists only in the "
                            "point-in-time capture store from 2026-09-25 (45 OBSERVED + 700 AWAITED rows at 2026-10-02, all releases >= 2026-02-05)"),
    ("unit", None, "ABSENT: nothing in the resource states the unit of val"),
]


def column_rows_from_frame(source: str, df: pd.DataFrame, time_col: str | None, train_mask: pd.Series | None,
                           licence: str, frequency: str, event_time: str, availability_time: str, notes: dict | None = None) -> list[dict]:
    rows = []
    notes = notes or {}
    for col in df.columns:
        s = df[col]
        tr = s[train_mask] if train_mask is not None else s.iloc[0:0]
        rows.append({
            "source": source, "column": col, "dtype": str(s.dtype), "rows_total": int(len(s)),
            "non_null_total": int(s.notna().sum() if s.dtype != object else (s.astype(str).str.strip() != "").sum()),
            "rows_in_train": int(len(tr)), "non_null_in_train": int(tr.notna().sum() if tr.dtype != object else (tr.astype(str).str.strip() != "").sum()),
            "unit": notes.get(col, {}).get("unit", "UNDECLARED_BY_SOURCE"),
            "frequency": frequency, "licence": licence, "event_time": event_time, "availability_time": availability_time,
            "span": [str(df[time_col].min()), str(df[time_col].max())] if time_col and time_col in df else None,
            "note": notes.get(col, {}).get("note", ""),
        })
    return rows


def scan_lake_metadata(meta_root: str) -> list[dict]:
    """Source rows from provenance.json files of the financial lake (text, git-tracked)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(meta_root, "**", "provenance.json"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(p), meta_root)
        if rel.startswith("features/cross_source"):
            continue
        try:
            d = json.load(open(p))
        except Exception as e:
            rows.append({"path": rel, "provider": "UNPARSEABLE", "error": str(e)}); continue
        prov = str(d.get("source") or d.get("provider") or "UNDECLARED")
        files = [f.get("path") for f in d.get("files", [])] if isinstance(d.get("files"), list) else []
        rows.append({"path": rel, "provider": prov, "acquired_at": d.get("acquired_at") or d.get("generated_at"),
                     "files": files, "timeframes": list(d.get("timeframes", {}).keys()) if isinstance(d.get("timeframes"), dict) else []})
    return rows


SELECTOR_FAMILIES = {"calendar_archive", "fxmacrodata_announcements", "fxmacrodata_calendar", "pit_capture", "fred_release_proxy"}


def source_role(family: str) -> str:
    if family in SELECTOR_FAMILIES:
        return "SELECTOR_EPISODE_SOURCE"
    if family in ("eurusd_price_pinned", "eurusd_price_raw"):
        return "RECONCILIATION_ONLY"
    if family in ("crypto", "us_equity_alt"):
        return "NOT_APPLICABLE_TO_EURUSD"
    return "MODEL_INPUT_CANDIDATE_SOURCE"


def classify_lake_source(r: dict) -> dict:
    """Decide the lane-A disposition of a lake source for the EURUSD manifest."""
    p, prov = r["path"], r["provider"]
    out = dict(r)
    if p.startswith("features/trading_asset_data/eurusd"):
        out.update(family="eurusd_price", frequency="5m/15m/1h/4h", licence=HISTDATA_LIC, batch="batch_001",
                   state="IN_BATCH_001 (5m used; 1h/15m/4h reconciled, not used)",
                   event_time="bar interval (NY local wall clock, bar START for 5m; measured)", availability_time="bar end")
    elif p.startswith("market_data/forex/g10/eurusd"):
        out.update(family="eurusd_price_raw", frequency="5m/15m/1h/4h", licence=HISTDATA_LIC, batch="none",
                   state="SUPERSEDED_BY features/trading_asset_data/eurusd (same bytes lineage)", event_time="bar", availability_time="bar end")
    elif p.startswith("market_data/forex/g10/") or p.startswith("features/trading_asset_data/") and any(k in p for k in ("gbpusd", "usdjpy", "usdchf", "usdcad", "audusd", "nzdusd", "eurgbp", "eurjpy", "gbpjpy")):
        out.update(family="fx_cross_pairs", frequency="5m..4h", licence=HISTDATA_LIC, batch="batch_002",
                   state="QUEUED_BATCH_002 (same clock inference per file required)", event_time="bar", availability_time="bar end")
    elif prov == "Yahoo Finance":
        out.update(family="yahoo_daily", frequency="1d", licence=YAHOO_LIC, batch="batch_002",
                   state="QUEUED_BATCH_002 (Close only; Adj Close is revised -> not point-in-time)",
                   event_time="trading date D (exchange session)", availability_time="conservative: (D+1) 00:00 UTC")
    elif p.startswith("macro_economic/fred"):
        daily = any(k in p for k in ("/rates/", "/stress/", "/fx_indices/", "/credit/", "inflation_expectations/t"))
        out.update(family="fred", frequency="1d" if daily else "weekly/monthly/quarterly", licence=FRED_LIC,
                   batch="batch_002" if daily else "none",
                   state=("QUEUED_BATCH_002 (market-observed daily; current vintage ~ first print)" if daily else
                          "NOT_ADMITTED: current-vintage revised macro with no publication instant (NO_VINTAGE/NO_PUBLICATION_CLOCK); "
                          "its release information enters through the calendar archive instead"),
                   event_time="observation date", availability_time="conservative: (D+1) 00:00 UTC for daily; undefined for revised macro")
    elif p.startswith("economic_calendar/release_actuals/fxmacrodata"):
        out.update(family="fxmacrodata_announcements", frequency="event", licence=FXM_LIC, batch="batch_001",
                   state="INVENTORIED; NOT_ADMITTED_FOR_SELECTION: 0 TRAIN rows (span 2024-12-12..2026-05-01 lies in validation/test)",
                   event_time="announcement_datetime_utc (observed)", availability_time="file-grain acquired_at; per-release receipt only from 2026-09-25")
    elif p.startswith("economic_calendar/scheduled_events/fxmacrodata"):
        out.update(family="fxmacrodata_calendar", frequency="event", licence=FXM_LIC, batch="batch_001",
                   state="INVENTORIED; NOT_ADMITTED: forward schedule 2026-02-05..2027-07-14, 0 TRAIN rows, no values",
                   event_time="scheduled instant", availability_time="file-grain acquired_at 2026-05-01")
    elif p.startswith("economic_calendar/"):
        out.update(family="fred_release_proxy", frequency="monthly", licence=FRED_LIC, batch="none",
                   state="NOT_ADMITTED: NO_PUBLICATION_INSTANT (proxy dates / naive dates; registered absences)",
                   event_time="reference date", availability_time="UNDEFINED")
    elif p.startswith("market_data/crypto") or p.startswith("alternative_data/onchain") or p.startswith("alternative_data/cryptoquant") \
            or p.startswith("alternative_data/defi") or "usdt" in p:
        out.update(family="crypto", frequency="various", licence="per provider", batch="none",
                   state="NOT_APPLICABLE_TO_EURUSD_MANIFEST (ETH/crypto manifest is separate; not mixed into the EURUSD denominator)",
                   event_time="per file", availability_time="per file")
    elif p.startswith("reference_data/holidays") or p.startswith("reference_data/trading_calendars"):
        out.update(family="calendar_reference", frequency="daily", licence="python-holidays (MIT) / generated", batch="batch_002",
                   state="QUEUED_BATCH_002: holiday flags need publication-in-advance evidence (python-holidays is retrospective code)",
                   event_time="date", availability_time="UNDEFINED_UNTIL_EVIDENCED")
    elif p.startswith("alternative_data/cot_reports"):
        out.update(family="cftc_cot", frequency="weekly", licence="CFTC public domain", batch="batch_003",
                   state="QUEUED_BATCH_003: positions as of Tuesday, released Friday 15:30 ET; release-time rule to bind",
                   event_time="Tuesday as-of date", availability_time="Friday 15:30 America/New_York of the same week")
    elif p.startswith("macro_economic/"):
        out.update(family="macro_other", frequency="monthly/quarterly", licence="per provider (public)", batch="none",
                   state="NOT_ADMITTED: revised macro without vintage/publication instant", event_time="reference period", availability_time="UNDEFINED")
    elif p.startswith("alternative_data/short_interest") or p.startswith("alternative_data/sec_filings"):
        out.update(family="us_equity_alt", frequency="daily/event", licence="FINRA/SEC public", batch="none",
                   state="NOT_APPLICABLE_TO_EURUSD_MANIFEST (single-name US equity data)", event_time="per file", availability_time="per file")
    else:
        out.update(family="other", frequency="UNDECLARED", licence="UNDECLARED", batch="none",
                   state="INVENTORIED_UNCLASSIFIED (reason: no EURUSD role identified; kept for review)", event_time="UNDECLARED", availability_time="UNDECLARED")
    return out


def non_lake_sources() -> list[dict]:
    return [
        {"path": "external:alpaca_market_data", "provider": "Alpaca", "family": "alpaca", "frequency": "1m..1d (US equities/ETFs, crypto)",
         "licence": "Alpaca market-data agreement (account-bound; IEX feed on free plan)", "batch": "none",
         "state": "NOT_INGESTED: no Alpaca bytes in the lake; Alpaca has no FX/EURUSD data; equity covariates already present via Yahoo daily. "
                  "Intraday SPY/QQQ via Alpaca would need the LTS-held credential outside the data-gov path -> CREDENTIAL_BLOCKER if ordered",
         "event_time": "bar", "availability_time": "bar end (+feed delay on free plan)"},
        {"path": "feature-eng:tests/data/economic_calendar_2011_2021.csv", "provider": "UNKNOWN (no provenance sidecar)", "family": "calendar_archive",
         "frequency": "event", "licence": "UNKNOWN_PROVENANCE", "batch": "batch_001",
         "state": "IN_BATCH_001: the only consensus+actual+previous+importance source overlapping TRAIN (2012-05..2021-04 with measured clock)",
         "event_time": "scheduled instant (measured clock eras)", "availability_time": "scheduled + 1 min (ASSUMED_SCHEDULED_PUBLICATION_LOCALIZED)"},
        {"path": "predictor:examples/data/phase_1/base_d{2,3,5,6}.csv", "provider": "git-pinned EURUSD 1h (UTC bar start)", "family": "eurusd_price_pinned",
         "frequency": "1h", "licence": "derived from HistData (git-pinned)", "batch": "batch_001",
         "state": "RECONCILED_ONLY (fragmented 2010-2020 segments; used to verify the lake clock conversion, not as a feature source)",
         "event_time": "UTC bar start", "availability_time": "bar end"},
        {"path": "coordinator:~/.local/state/financial-data/point_in_time", "provider": "own collector (receipt clock)", "family": "pit_capture",
         "frequency": "3h passes", "licence": "own", "batch": "batch_001",
         "state": "INVENTORIED; NOT_ADMITTED_FOR_SELECTION: 745 rows (45 OBSERVED, 700 AWAITED), releases 2026-02-05..2026-04-30 received from 2026-09-25 -- post-test era",
         "event_time": "scheduled/source publication", "availability_time": "received_at (own clock)"},
        {"path": "feature-eng:regime_detector / m5phet regimes", "provider": "feature-eng", "family": "regime", "frequency": "1h (derived)",
         "licence": "own", "batch": "batch_003", "state": "QUEUED_BATCH_003: causal (filtering) regime probabilities only; smoothed/Viterbi labels are non-causal",
         "event_time": "t", "availability_time": "t if filtered; NOT_ADMISSIBLE if smoothed"},
    ]


TRANSFORM_VARIANTS = [
    {"variant_id": "tv.wavelet_modwt_haar_causal", "family": "wavelet", "identity": "MODWT/a-trous Haar, J=5 levels, trailing window 256 bars, value at last sample",
     "causal_by_construction": True},
    {"variant_id": "tv.wavelet_dwt_db4_global", "family": "wavelet", "identity": "pywt.wavedec db4 over the full series (global)", "causal_by_construction": False},
    {"variant_id": "tv.multitaper_trailing", "family": "multitaper", "identity": "DPSS NW=3, K=5 tapers on trailing 256 bars; band power 6-48 h",
     "causal_by_construction": True},
    {"variant_id": "tv.hilbert_trailing_lastsample", "family": "hilbert", "identity": "scipy.signal.hilbert on trailing 256 bars of detrended log price, amplitude at last sample",
     "causal_by_construction": True, "note": "prefix-invariant but end-effect biased"},
    {"variant_id": "tv.hilbert_global", "family": "hilbert", "identity": "scipy.signal.hilbert over the full series", "causal_by_construction": False},
    {"variant_id": "tv.stl_trailing_lastsample", "family": "stl", "identity": "statsmodels STL period=24 on trailing 24*14 bars, components at last sample",
     "causal_by_construction": True},
    {"variant_id": "tv.stl_global", "family": "stl", "identity": "statsmodels STL period=24 over the full series (centred LOESS)", "causal_by_construction": False},
    {"variant_id": "tv.kalman_local_level_filter", "family": "kalman", "identity": "local-level Kalman FILTER (q/r fitted on fold-train), filtered state",
     "causal_by_construction": True},
    {"variant_id": "tv.kalman_local_level_smoother", "family": "kalman", "identity": "RTS smoother of the same model", "causal_by_construction": False},
]
