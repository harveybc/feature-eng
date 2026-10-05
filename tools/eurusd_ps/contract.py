"""EURUSD business contract: periods, targets, supports, purge and inner folds.

Every instant is UTC. A decision row is indexed by t = the END of an hourly
UTC bar; everything with availability_time <= t may be used at t.

Decisions recorded here (lane A, 2026-10-03) and their reasons:

* TRAIN starts 2012-05-01. The only consensus/forecast source covering the
  history (the 2011-2021 calendar archive) has a measured clock only from
  2012-05-01 (earlier months are UNDETERMINED); a 4-year TRAIN (2020-2023)
  would leave consensus inside 16 of 48 months and inside at most one inner
  fold. The 4-year recipe window is kept as a declared sub-window
  (``RECIPE_WINDOW_4Y``) for downstream fitting comparisons.
* External validation = calendar 2024; external test = calendar 2025 (the
  last full year in the lake). Test is SEALED: no loader returns a value with
  event or availability time at or after VALIDATION start, and PS0/PS1 do not
  read validation either (READ_END = TRAIN end).
* Purge is derived from supports, never a fixed embargo: a fold-train row t
  is kept only if t + max(target support) < next block start. Inner folds are
  forward-chaining (train strictly before validation), so the feature-window
  embargo is zero and is recorded as such together with the largest window.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

CONTRACT_SCHEMA = "eurusd_business_contract.v1"

TRAIN_START = pd.Timestamp("2012-05-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2024-01-01T00:00:00Z")
VALIDATION_START = TRAIN_END
VALIDATION_END = pd.Timestamp("2025-01-01T00:00:00Z")
TEST_START = VALIDATION_END
TEST_END = pd.Timestamp("2026-01-01T00:00:00Z")
READ_END = TRAIN_END          # PS0/PS1 read nothing at or after this instant
SPLIT = "train"
DECISION_START, DECISION_END = TRAIN_START, TRAIN_END


def configure_split(split: str) -> dict:
    """Select the decision window and the read bound. 'train' is the default;
    'validation_2024' is the one-time external-validation materialisation for
    FS-CLOSE: decisions 2024-01-01 <= t < 2025-01-01, READ_END = 2025-01-01,
    so TEST (2025) is never read. Nothing is fitted on 2024: every window,
    sigma and Kalman parameter keeps its TRAIN definition."""
    global SPLIT, READ_END, DECISION_START, DECISION_END
    if split == "train":
        SPLIT, READ_END, DECISION_START, DECISION_END = "train", TRAIN_END, TRAIN_START, TRAIN_END
    elif split == "validation_2024":
        SPLIT, READ_END, DECISION_START, DECISION_END = "validation_2024", VALIDATION_END, VALIDATION_START, VALIDATION_END
    else:
        raise ValueError(f"unknown split {split!r}; TEST is never materialised by this producer")
    assert READ_END <= TEST_START, "the producer never reads TEST"
    return {"split": SPLIT, "read_end": str(READ_END), "decision_window": [str(DECISION_START), str(DECISION_END)]}


def guard_rows(index) -> None:
    """Refuse any decision row at or after READ_END (and before the split's start)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        raise ValueError("no decision rows")
    if idx.max() >= READ_END or idx.max() >= TEST_START:
        raise ValueError(f"REFUSED: decision row {idx.max()} at or after READ_END {READ_END}")
    if idx.min() < DECISION_START:
        raise ValueError(f"REFUSED: decision row {idx.min()} before split start {DECISION_START}")
WARMUP_START = pd.Timestamp("2012-01-01T00:00:00Z")  # past-only warm-up for windows
RECIPE_WINDOW_4Y = (pd.Timestamp("2020-01-01T00:00:00Z"), TRAIN_END)

Y_S_HOURS = [1, 2, 3, 4, 5, 6]
Y_L_HOURS = [24, 48, 72, 96, 120, 144]
SIGMA_SPEC = {"kind": "ewma_std_of_1h_log_returns", "halflife_bars": 24, "min_bars": 168,
              "available_at": "t (includes the return of the bar ending at t)"}
Y_B_SPECS = [
    {"name": "Y_b_s6", "timeout_h": 6, "width_sigma_mult": 1.0},
    {"name": "Y_b_l144", "timeout_h": 144, "width_sigma_mult": 1.0},
]
Y_B_RULES = {
    "entry": "close of the bar ending at t",
    "barrier_width": "w = width_sigma_mult * sigma_t * sqrt(timeout_h); TP = entry*exp(+w), SL = entry*exp(-w)",
    "path": "5-minute bars whose start >= t and whose end <= t + timeout_h (elapsed time, weekends included)",
    "label": "+1 TP touched first, -1 SL touched first, 0 neither by timeout",
    "intrabar_ambiguity": "both touched inside one 5-minute bar -> AMBIGUOUS, label NaN, flag kept (version v1: no tick data)",
    "censoring": "t + timeout_h >= READ_END -> CENSORED, label NaN",
    "costs": "none inside the label; costs belong to J_policy",
    "tp_sl_source": "fixed rule above; never an oracle forecast",
}
J_POLICY = {
    "status": "PLACEHOLDER_NOT_CONSTRUCTED",
    "definition": "episode objective of the RL/heuristic environment: net PnL after spread, commission and swap, "
                  "subject to declared risk limits; specified by the environment contract (gym-fx / heuristic-strategy)",
    "blocked_on": "no historical EURUSD spread series in the lake (OHLC only); environment cost contract not bound here",
}


def target_supports() -> dict:
    """Elapsed hours of future information each target consumes after t."""
    sup = {f"Y_s_{h}h": h for h in Y_S_HOURS}
    sup.update({f"Y_l_{h}h": h for h in Y_L_HOURS})
    sup.update({s["name"]: s["timeout_h"] for s in Y_B_SPECS})
    return sup


def derive_purge(feature_support_h: dict | None = None, forward_chaining: bool = True) -> dict:
    sup = target_supports()
    label_purge = max(sup.values())
    max_feat = max(feature_support_h.values()) if feature_support_h else 0
    return {
        "label_purge_h": int(label_purge),
        "label_purge_rule": "fold-train row t kept iff t + label_purge_h < validation block start",
        "derived_from": "max target support (Y_s, Y_l, Y_b timeouts) in elapsed hours",
        "feature_embargo_h": 0 if forward_chaining else int(max_feat),
        "feature_embargo_rule": ("forward-chaining folds: no training row lies after a validation row, so "
                                 "window support needs no embargo" if forward_chaining else
                                 "embargo = largest feature window support"),
        "max_feature_support_h": int(max_feat),
        "target_supports_h": sup,
    }


def inner_folds(decision_times: pd.DatetimeIndex, val_years=(2019, 2020, 2021, 2022, 2023),
                label_purge_h: int | None = None) -> list[dict]:
    """Forward-chaining inner folds inside TRAIN; validation blocks are calendar years."""
    if label_purge_h is None:
        label_purge_h = derive_purge()["label_purge_h"]
    t = pd.DatetimeIndex(decision_times)
    if len(t) and (t.max() >= TRAIN_END or t.min() < TRAIN_START):
        raise ValueError("inner folds take TRAIN decision rows only")
    purge = pd.Timedelta(hours=label_purge_h)
    folds = []
    for y in val_years:
        vs = pd.Timestamp(f"{y}-01-01T00:00:00Z")
        ve = min(pd.Timestamp(f"{y + 1}-01-01T00:00:00Z"), TRAIN_END)
        tr = np.where((t >= TRAIN_START) & (t + purge < vs))[0]
        va = np.where((t >= vs) & (t < ve))[0]
        folds.append({
            "name": f"inner_{y}", "train_rows": [int(tr.min()), int(tr.max()) + 1] if len(tr) else None,
            "val_rows": [int(va.min()), int(va.max()) + 1] if len(va) else None,
            "train_n": int(len(tr)), "val_n": int(len(va)),
            "train_time": [str(t[tr.min()]), str(t[tr.max()])] if len(tr) else None,
            "val_time": [str(t[va.min()]), str(t[va.max()])] if len(va) else None,
            "label_purge_h": label_purge_h,
            "val_label_rule": "validation labels whose support crosses READ_END are NaN (censored), never filled",
        })
    return folds


def contract_document(feature_support_h: dict | None = None) -> dict:
    doc = {
        "schema": CONTRACT_SCHEMA,
        "asset": "EURUSD",
        "decision_grid": {"frequency": "1h", "timezone": "UTC", "row_time": "end of the hourly UTC bar",
                          "rows": "hours in which at least one 5-minute bar exists (market open)"},
        "periods": {
            "warmup_past_only": [str(WARMUP_START), str(TRAIN_START)],
            "train": [str(TRAIN_START), str(TRAIN_END)],
            "external_validation": [str(VALIDATION_START), str(VALIDATION_END)],
            "external_test_SEALED": [str(TEST_START), str(TEST_END)],
            "read_end_for_ps0_ps1": str(READ_END),
            "recipe_window_4y": [str(RECIPE_WINDOW_4Y[0]), str(RECIPE_WINDOW_4Y[1])],
        },
        "targets": {
            "Y_s": {"hours": Y_S_HOURS, "definition": "ln(C_asof(t+h) / C(t)), C_asof = close of last bar ending <= t+h (elapsed hours, not rows)"},
            "Y_l": {"hours": Y_L_HOURS, "definition": "same as Y_s at long horizons; staleness (weekend) recorded per row"},
            "Y_b": {"specs": Y_B_SPECS, "rules": Y_B_RULES, "sigma": SIGMA_SPEC},
            "J_policy": J_POLICY,
        },
        "purge": derive_purge(feature_support_h),
        "asof_join": {"rule": "value usable at t iff availability_time <= t; ties at equality admitted; "
                              "age = t - availability_time in elapsed hours; no backfill, no forward fill past the declared max age",
                      "time_axis": "elapsed time, never row offsets"},
        "selection_scope": "selection and threshold fitting only on inner TRAIN folds; external validation does not select; test is never read",
    }
    doc["contract_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()
    return doc
