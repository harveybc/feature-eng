#!/usr/bin/env python3
"""Lane B, PS0-PS2: per-cell TRAIN-fold profiles, business-target probes, reversible priority.

PS1  profiles admissible inputs in batches, one batch per (fold, metric family). Every metric
     cell carries a status: MEASURED, MEASURED_REUSED, FAILED or NOT_RUN. FAILED and NOT_RUN
     carry a reason and no value, so a failure is never read as zero. STL, stationarity and
     spectral metrics are deferred to PS4 on inner folds.
PS2  ranks inputs against the declared business targets Y_s, Y_l and Y_b, using inner-fold
     TRAIN rows whose label support also ends inside the fold. It never discards anything: it
     outputs tiers (priority, synergy pairs, redundancy representatives, a declared exploratory
     sample outside the ranking, and reversible deferral).
Nothing here fits on validation, outer validation or test rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import zlib

import numpy as np
from scipy import stats

PS1_FAMILIES = ("quality", "distribution", "volatility", "acf_selected", "seasonality", "cost")
COMPUTED_FAMILIES = PS1_FAMILIES[:-1]
DEFERRED_FAMILIES = ("stationarity", "spectral", "stl")
DEFERRED_METRICS = {"stationarity": ("adf_c_aic_pvalue", "kpss_c_auto_pvalue"),
                    "spectral": ("peak1_period_rows", "entropy_normalized"),
                    "stl": ("stl_seasonal_strength_primary",)}
CELL_STATUSES = {"MEASURED", "MEASURED_REUSED", "FAILED", "NOT_RUN"}
TARGET_NAMES = ("Y_s", "Y_l", "Y_b")
TIERS = ("PRIORITY", "SYNERGY", "REPRESENTATIVE", "EXPLORATORY", "DEFERRED", "UNRANKED", "INELIGIBLE_IN_FOLD")
REINCORPORATION = ("re-enters on: PS5 inner-validation gain as a member of a group or pair, a later exploratory "
                   "draw, a new vintage, or a change of declared target")
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# ----------------------------------------------------------------- folds
def inner_folds(n: int, k: int = 3, val_frac: float = 0.15, purge: int = 0):
    """Expanding chronological folds inside TRAIN [0, n): train [0, e), purge gap, then validation."""
    v = max(1, int(round(n * val_frac)))
    folds = []
    for j in range(1, k + 1):
        val_end = n - (k - j) * v
        val_start = val_end - v
        train_end = val_start - purge
        if train_end <= 1:
            raise ValueError("fold has no training rows; reduce k, val_frac or purge")
        folds.append({"name": f"inner_{j}", "train": (0, train_end), "val": (val_start, val_end), "purge": purge})
    return folds


def outer_fold(n: int):
    return {"name": "outer_train", "train": (0, n), "val": None, "purge": 0}


def emit_inputs(X, fold):
    """Standardize with statistics of the fold's TRAIN rows only."""
    a, b = fold["train"]
    mu = np.nanmean(X[a:b], axis=0)
    sd = np.nanstd(X[a:b], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


# ----------------------------------------------------------------- PS1 metric families
def family_metrics(family, params):
    periods = params.get("declared_periods_rows", {})
    if family == "quality":
        return ("missing_fraction", "nonfinite_count", "constant_flag", "unique_ratio", "sentinel_le_m9999_count")
    if family == "distribution":
        return ("mean", "std", "median", "iqr", "mad", "skewness", "excess_kurtosis", "q01", "q99",
                "upper_tail_ratio", "lower_tail_ratio")
    if family == "volatility":
        return ("diff_std", "diff_std_over_std", "rolling_std_cv")
    if family == "acf_selected":
        return tuple(f"acf_lag_{l}" for l in params.get("acf_lags", [1]))
    if family == "seasonality":
        out = []
        for name, s in periods.items():
            out += [f"seasonal_acf_{name}_{s}", f"seasonal_diff_var_ratio_{name}_{s}"]
        return tuple(out) or ("no_declared_period",)
    if family == "cost":
        return tuple(f"seconds_{f}" for f in COMPUTED_FAMILIES)
    if family in DEFERRED_METRICS:
        return DEFERRED_METRICS[family]
    raise KeyError(family)


def _longest_run(x):
    finite = np.isfinite(x)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return x[:0]
    i = int(np.argmax(ends - starts))
    return x[starts[i]:ends[i]]


def _quality(x, params):
    fin = x[np.isfinite(x)]
    return {"missing_fraction": float(np.mean(~np.isfinite(x))) if len(x) else (None, "EMPTY_FOLD"),
            "nonfinite_count": int((~np.isfinite(x)).sum()),
            "constant_flag": bool(len(fin) > 0 and np.ptp(fin) == 0),
            "unique_ratio": float(len(np.unique(fin)) / len(fin)) if len(fin) else (None, "NO_FINITE_VALUES"),
            "sentinel_le_m9999_count": int((fin <= -9999).sum())}


def _distribution(fin, params):
    q = np.quantile(fin, [0.005, 0.01, 0.25, 0.5, 0.75, 0.99, 0.995])
    q005, q01, q25, med, q75, q99, q995 = q
    up, lo = q75 - med, med - q25
    n = len(fin)
    return {"mean": float(fin.mean()), "std": float(fin.std(ddof=1)) if n > 1 else (None, "INSUFFICIENT_SAMPLE"),
            "median": float(med), "iqr": float(q75 - q25), "mad": float(np.median(np.abs(fin - med))),
            "skewness": float(stats.skew(fin, bias=False)) if n > 3 else (None, "INSUFFICIENT_SAMPLE"),
            "excess_kurtosis": float(stats.kurtosis(fin, bias=False)) if n > 3 else (None, "INSUFFICIENT_SAMPLE"),
            "q01": float(q01), "q99": float(q99),
            "upper_tail_ratio": float((q995 - med) / up) if up > 0 else (None, "ZERO_UPPER_QUARTILE_SPREAD"),
            "lower_tail_ratio": float((med - q005) / lo) if lo > 0 else (None, "ZERO_LOWER_QUARTILE_SPREAD")}


def _volatility(y, params):
    d = np.diff(y)
    sd = float(y.std(ddof=1)) if len(y) > 1 else 0.0
    w = params.get("declared_periods_rows", {}).get(params.get("primary_period"))
    out = {"diff_std": float(d.std(ddof=1)) if len(d) > 1 else (None, "INSUFFICIENT_SAMPLE"),
           "diff_std_over_std": float(d.std(ddof=1) / sd) if len(d) > 1 and sd > 0 else (None, "INSUFFICIENT_SAMPLE")}
    if not w:
        out["rolling_std_cv"] = (None, "NO_DECLARED_PRIMARY_PERIOD")
    elif len(y) < 4 * w:
        out["rolling_std_cv"] = (None, "SEGMENT_SHORTER_THAN_4_PERIODS")
    else:
        blocks = y[: len(y) // w * w].reshape(-1, w).std(axis=1, ddof=1)
        out["rolling_std_cv"] = float(blocks.std(ddof=1) / blocks.mean()) if blocks.mean() > 0 else (None, "ZERO_BLOCK_STD")
    return out


def _acf(y, params):
    c = y - y.mean()
    e = float(c @ c)
    out = {}
    for l in params.get("acf_lags", [1]):
        if l >= len(y):
            out[f"acf_lag_{l}"] = (None, "LAG_EXCEEDS_SEGMENT")
        elif e <= 0:
            out[f"acf_lag_{l}"] = (None, "ZERO_VARIANCE")
        else:
            out[f"acf_lag_{l}"] = float((c[l:] @ c[:-l]) / e)
    return out


def _seasonality(y, params):
    periods = params.get("declared_periods_rows", {})
    if not periods:
        return {"no_declared_period": (None, "NO_DECLARED_PERIOD")}
    out, var = {}, float(y.var())
    c = y - y.mean()
    for name, s in periods.items():
        k1, k2 = f"seasonal_acf_{name}_{s}", f"seasonal_diff_var_ratio_{name}_{s}"
        if len(y) <= 2 * s or var <= 0:
            reason = "SEGMENT_SHORTER_THAN_2_PERIODS" if var > 0 else "ZERO_VARIANCE"
            out[k1] = out[k2] = (None, reason)
        else:
            out[k1] = float((c[s:] @ c[:-s]) / (c @ c))
            out[k2] = float(np.var(y[s:] - y[:-s]) / var)
    return out


FAMILY_FN = {"quality": "_quality", "distribution": "_distribution", "volatility": "_volatility",
             "acf_selected": "_acf", "seasonality": "_seasonality"}


def _family_cells(family, x, feature, fold, params):
    metrics = family_metrics(family, params)
    cell = lambda m, status, value=None, reason="": {"feature": feature, "fold": fold["name"], "family": family,
                                                     "metric": m, "status": status, "value": value, "reason": reason}
    fin = x[np.isfinite(x)]
    if family != "quality":
        if not len(fin):
            return [cell(m, "NOT_RUN", reason="NO_FINITE_VALUES") for m in metrics]
        if np.ptp(fin) == 0:
            return [cell(m, "NOT_RUN", reason="CONSTANT_IN_FOLD") for m in metrics]
    arg = {"quality": x, "distribution": fin}.get(family)
    if arg is None:
        arg = _longest_run(x)
    try:
        values = globals()[FAMILY_FN[family]](arg, params)
    except Exception as exc:                                    # a failure is a status, never a zero
        return [cell(m, "FAILED", reason=f"{type(exc).__name__}: {exc}"[:200]) for m in metrics]
    out = []
    for m in metrics:
        v = values.get(m, (None, "NOT_PRODUCED"))
        if isinstance(v, tuple):
            out.append(cell(m, "NOT_RUN", reason=v[1]))
        elif isinstance(v, float) and not math.isfinite(v):
            out.append(cell(m, "FAILED", reason="NONFINITE_RESULT"))
        else:
            out.append(cell(m, "MEASURED", value=v))
    return out


class ProfileCache:
    """Batch cache keyed by bytes, vintage, fold rows, family, parameters, feature list and code."""

    def __init__(self, root: Path):
        self.root = Path(root) / "ps1_cache"
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(identity, fold, family, params, names):
        return _sha({"identity": identity, "fold": [fold["name"], list(fold["train"])], "family": family,
                     "params": params, "names": list(names), "code": CODE_SHA})

    def get(self, key):
        p = self.root / f"{key}.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def put(self, key, payload):
        p = self.root / f"{key}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, p)


def run_ps1(X, names, folds, params, cache: ProfileCache | None = None, identity: dict | None = None):
    if cache is not None and not identity:
        raise ValueError("caching requires a resource identity (bytes sha256 and vintage)")
    cells = []
    for fold in folds:
        a, b = fold["train"]
        seconds, reused = {}, {}
        fam_cells = {}
        for family in COMPUTED_FAMILIES:
            key = ProfileCache.key(identity, fold, family, params, names) if cache else None
            hit = cache.get(key) if cache else None
            if hit and not any(c["status"] == "FAILED" for c in hit["cells"]):
                fam_cells[family] = [dict(c, status="MEASURED_REUSED", reason=f"cache {key[:16]}")
                                     if c["status"] == "MEASURED" else c for c in hit["cells"]]
                seconds[family], reused[family] = hit["seconds"], True
                continue
            t0 = time.process_time()
            batch = []
            for i, f in enumerate(names):
                batch += _family_cells(family, X[a:b, i], f, fold, params)
            seconds[family], reused[family] = time.process_time() - t0, False
            fam_cells[family] = batch
            if cache:
                cache.put(key, {"cells": batch, "seconds": seconds[family], "identity": identity})
        for i, f in enumerate(names):
            for family in COMPUTED_FAMILIES:
                cells += [c for c in fam_cells[family] if c["feature"] == f]
            for family in COMPUTED_FAMILIES:
                cells.append({"feature": f, "fold": fold["name"], "family": "cost", "metric": f"seconds_{family}",
                              "status": "MEASURED_REUSED" if reused[family] else "MEASURED",
                              "value": seconds[family] / max(1, len(names)),
                              "reason": "process CPU seconds of the batch divided by its features"})
            for family in DEFERRED_FAMILIES:
                for m in DEFERRED_METRICS[family]:
                    cells.append({"feature": f, "fold": fold["name"], "family": family, "metric": m,
                                  "status": "NOT_RUN", "value": None, "reason": "DEFERRED_TO_PS4"})
    return cells


def coverage(cells, names, folds, required=PS1_FAMILIES):
    acc = {}
    for c in cells:
        acc.setdefault((c["feature"], c["fold"]), {}).setdefault(c["family"], []).append(c["status"])
    out = {}
    for f in names:
        for fold in folds:
            fams = acc.get((f, fold["name"]), {})
            state = {}
            for fam in PS1_FAMILIES + DEFERRED_FAMILIES:
                st = fams.get(fam, [])
                if not st:
                    state[fam] = "ABSENT"
                elif "FAILED" in st:
                    state[fam] = "FAILED"
                elif all(s in ("MEASURED", "MEASURED_REUSED") for s in st):
                    state[fam] = "COMPLETE"
                elif any(s in ("MEASURED", "MEASURED_REUSED") for s in st):
                    state[fam] = "PARTIAL"
                else:
                    state[fam] = "NOT_RUN"
            out[(f, fold["name"])] = {"families": state,
                                      "basic_complete": all(state[x] == "COMPLETE" for x in required),
                                      "full_complete": all(state[x] == "COMPLETE" for x in PS1_FAMILIES + DEFERRED_FAMILIES)}
    return out


def metric_coverage(cells):
    acc = {}
    for c in cells:
        r = acc.setdefault((c["fold"], c["family"], c["metric"]),
                           {"fold": c["fold"], "family": c["family"], "metric": c["metric"],
                            "denominator": 0, "measured": 0, "reused": 0, "failed": 0, "not_run": 0, "not_run_reasons": {}})
        r["denominator"] += 1
        if c["status"] in ("MEASURED", "MEASURED_REUSED"):
            r["measured"] += 1
            r["reused"] += c["status"] == "MEASURED_REUSED"
        elif c["status"] == "FAILED":
            r["failed"] += 1
        else:
            r["not_run"] += 1
            r["not_run_reasons"][c["reason"]] = r["not_run_reasons"].get(c["reason"], 0) + 1
    return list(acc.values())


class DecisionStore:
    """Selection decisions keyed by resource bytes, vintage and fold; nothing crosses a vintage."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, identity, fold):
        return self.root / f"{_sha({'identity': identity, 'fold': fold})}.json"

    def save(self, decision, identity, fold):
        self._path(identity, fold).write_text(json.dumps({"identity": identity, "fold": fold, "decision": decision}))

    def load(self, identity, fold):
        p = self._path(identity, fold)
        if not p.is_file():
            return None
        doc = json.loads(p.read_text())
        return doc["decision"] if doc["identity"] == identity and doc["fold"] == fold else None


# ----------------------------------------------------------------- targets (FS02)
class TargetSeries:
    """A business target: name in TARGET_NAMES, horizon in hours, values and the row each label reads."""

    def __init__(self, name, horizon_hours, values, label_index, status, source_column, note=""):
        self.name, self.horizon_hours, self.values, self.label_index = name, horizon_hours, values, label_index
        self.status, self.source_column, self.note = status, source_column, note


def build_targets(price, ts_seconds, asset_column, source_column, horizons, step_seconds, barrier_rule=None):
    """Business targets from the declared asset price only. Y(t,h) = log P(t+h hours) / P(t).

    The label row is located by elapsed time, not by row count: a horizon that is not a
    multiple of the sampling step is NOT_CONSTRUCTIBLE_AT_SAMPLING, and a missing bar at
    exactly t+h leaves that label empty instead of borrowing a neighbour."""
    if source_column != asset_column:
        raise ValueError(f"SELF_FORECAST_REFUSED: targets come from the declared asset price '{asset_column}', "
                         f"not from '{source_column}'")
    price = np.asarray(price, dtype=float)
    ts = np.asarray(ts_seconds, dtype=np.int64)
    n = len(price)
    out = {}
    for name, hs in horizons.items():
        if name not in TARGET_NAMES:
            raise ValueError(f"unknown target {name}; declared targets are {TARGET_NAMES}")
        for h in hs:
            empty = np.full(n, np.nan), np.full(n, -1, dtype=np.int64)
            if name == "Y_b":
                note = ("candidate rule: " + barrier_rule) if barrier_rule else "no versioned TP/SL rule supplied"
                out[(name, h)] = TargetSeries(name, h, *empty, "NOT_EVALUATED_NO_VERSIONED_RULE", source_column, note)
                continue
            if (h * 3600) % step_seconds:
                out[(name, h)] = TargetSeries(name, h, *empty, "NOT_CONSTRUCTIBLE_AT_SAMPLING", source_column,
                                              f"{h} h is not a multiple of the {step_seconds} s step")
                continue
            j = np.searchsorted(ts, ts + h * 3600)
            ok = (j < n)
            ok[ok] = ts[j[ok]] == ts[ok] + h * 3600
            vals, lab = empty
            vals = vals.copy()
            lab = lab.copy()
            lab[ok] = j[ok]
            with np.errstate(divide="ignore", invalid="ignore"):
                vals[ok] = np.log(price[j[ok]] / price[ok])
            out[(name, h)] = TargetSeries(name, h, vals, lab, "CONSTRUCTED", source_column,
                                          f"elapsed-time label; {int((~ok).sum())} rows without a bar at t+h")
    return out


def label_rows(target: TargetSeries, end: int, start: int = 0):
    idx = np.arange(len(target.values))
    m = (idx >= start) & (idx < end) & (target.label_index >= 0) & (target.label_index < end) & np.isfinite(target.values)
    return idx[m]


def _spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def relevance(x, target, start, end):
    if not isinstance(target, TargetSeries):
        raise TypeError("a probe target must be a TargetSeries built from the declared asset price")
    if target.name not in TARGET_NAMES:
        raise ValueError(f"probe target must be one of {TARGET_NAMES}, got {target.name}")
    base = {"target": target.name, "horizon_hours": target.horizon_hours}
    if target.status != "CONSTRUCTED":
        return dict(base, status="NOT_EVALUATED", reason=target.status, rho=None, n=0)
    rows = label_rows(target, end, start)
    rows = rows[np.isfinite(x[rows])]
    if len(rows) < 30:
        return dict(base, status="NOT_RUN", reason="FEWER_THAN_30_LABELLED_ROWS", rho=None, n=int(len(rows)))
    rho = _spearman(x[rows], target.values[rows])
    if rho is None:
        return dict(base, status="NOT_RUN", reason="CONSTANT_INPUT_OR_TARGET", rho=None, n=int(len(rows)))
    return dict(base, status="MEASURED", reason="", rho=rho, n=int(len(rows)),
                estimator="Spearman, fold TRAIN rows whose label support ends inside the fold")


# ----------------------------------------------------------------- PS2 (FS16)
def _components(C, names, thr):
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    iu = np.triu_indices(len(names), 1)
    for i, j in zip(*iu):
        if abs(C[i, j]) >= thr:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(len(names)):
        groups.setdefault(find(i), []).append(names[i])
    return [g for g in groups.values() if len(g) > 1]


def prioritize(X, names, targets, fold, params, no_target_reason=None):
    a, b = fold["train"]
    Xf = X[a:b]
    rows = {f: {"feature": f, "fold": fold["name"], "tier": None, "reasons": [], "group": None,
                "score_by_target": {}, "explore_probability": None, "explore_rule": None, "reincorporation": None}
            for f in names}
    eligible = []
    for i, f in enumerate(names):
        fin = Xf[:, i][np.isfinite(Xf[:, i])]
        if not len(fin) or np.ptp(fin) == 0:
            rows[f].update(tier="INELIGIBLE_IN_FOLD", reincorporation=REINCORPORATION)
            rows[f]["reasons"].append("CONSTANT_IN_FOLD" if len(fin) else "NO_FINITE_VALUES")
        else:
            eligible.append(i)
    # redundancy on fold TRAIN rows
    groups = []
    if len(eligible) > 1:
        sub = Xf[:, eligible]
        ok = np.isfinite(sub).all(axis=1)
        if ok.sum() > 2:
            C = np.corrcoef(sub[ok], rowvar=False)
            groups = _components(C, [names[i] for i in eligible], params.get("redundancy_threshold", 0.95))
    for g_id, g in enumerate(groups):
        for f in g:
            rows[f]["group"] = f"R{g_id}"
            rows[f]["reasons"].append(f"REDUNDANCY_GROUP R{g_id} (|r|>={params.get('redundancy_threshold', 0.95)}): "
                                      + ",".join(g))
    built = {k: t for k, t in targets.items() if t.status == "CONSTRUCTED"}
    if not built:
        why = "NO_DECLARED_TARGET_CONSTRUCTED: " + (
            "; ".join(f"{k[0]}@{k[1]}h={t.status}" for k, t in targets.items()) or no_target_reason
            or "resource declares no business target")
        for i in eligible:
            rows[names[i]].update(tier="UNRANKED", reincorporation=REINCORPORATION)
            rows[names[i]]["reasons"].append(why)
        return [rows[f] for f in names]
    # individual relevance
    score = {}
    for i in eligible:
        f = names[i]
        best = 0.0
        for (tn, h), t in built.items():
            r = relevance(X[:, i], t, a, b)          # rows and their labels both inside [a, b)
            rows[f]["score_by_target"][f"{tn}@{h}h"] = r["rho"] if r["status"] == "MEASURED" else r["status"]
            if r["status"] == "MEASURED":
                best = max(best, abs(r["rho"]))
        score[f] = best
    ranked = sorted((names[i] for i in eligible), key=lambda f: (-score[f], f))
    top_q = params.get("top_q", 10)
    for f in ranked[:top_q]:
        rows[f]["tier"] = "PRIORITY"
        rows[f]["reasons"].append(f"TOP_{top_q}_INDIVIDUAL max|rho|={score[f]:.4g}")
    # pair synergy screen
    elig_names = [names[i] for i in eligible]
    if len(elig_names) <= params.get("pair_feature_cap", 64):
        pairs = [(p, q) for k, p in enumerate(elig_names) for q in elig_names[k + 1:]]
        pair_rule = "all pairs"
    else:
        head = ranked[:40]
        pairs = [(p, q) for k, p in enumerate(head) for q in head[k + 1:]]
        rng = np.random.default_rng([params.get("seed", 0), zlib.crc32(fold["name"].encode())])
        extra = rng.choice(len(elig_names), size=(2000, 2))
        pairs += [(elig_names[u], elig_names[v]) for u, v in extra if u < v]
        pairs = list(dict.fromkeys(pairs))
        pair_rule = "pairs among the top 40 plus 2000 seeded random pairs"
    syn = []
    col = {f: i for i, f in enumerate(names)}
    for (tn, h), t in built.items():
        lr = label_rows(t, b, a)
        if len(lr) < 30:
            continue                                  # no synergy evidence without labelled fold rows
        y = t.values[lr]
        floor = 3 / math.sqrt(len(lr))
        Z = {}
        for f in sorted(set(sum(pairs, ()))):
            v = X[lr, col[f]]
            Z[f] = (v - np.nanmean(v)) / (np.nanstd(v) or 1.0)
        indiv = {}
        for f in Z:
            fm = np.isfinite(Z[f])
            indiv[f] = abs(_spearman(Z[f][fm], y[fm]) or 0.0)
        for p, q in pairs:
            prod = Z[p] * Z[q]
            m = np.isfinite(prod)
            if m.sum() < 30:
                continue
            rp = _spearman(prod[m], y[m])
            if rp is None:
                continue
            gain = abs(rp) - max(indiv[p], indiv[q])
            if gain > floor:
                syn.append((gain, p, q, f"{tn}@{h}h", abs(rp), max(indiv[p], indiv[q])))
    syn.sort(key=lambda s: (-s[0], s[1], s[2]))
    for gain, p, q, tk, jr, ir in syn[:params.get("max_synergy_pairs", 10)]:
        for f in (p, q):
            if rows[f]["tier"] is None:
                rows[f]["tier"] = "SYNERGY"
            rows[f]["reasons"].append(f"SYNERGY_PAIR {p}*{q} on {tk}: joint |rho|={jr:.4g} vs individual max {ir:.4g} "
                                      f"({pair_rule}; noise floor 3/sqrt(n))")
    # redundancy representatives
    for g in groups:
        best = max(g, key=lambda f: (score.get(f, 0.0), f))
        if rows[best]["tier"] is None:
            rows[best]["tier"] = "REPRESENTATIVE"
            rows[best]["reasons"].append(f"BEST_SCORED_MEMBER_OF {rows[best]['group']}")
    # exploratory sample outside the ranking
    rest = [f for f in ranked if rows[f]["tier"] is None]
    if rest:
        k = min(len(rest), max(params.get("explore_min", 1), int(round(params.get("explore_fraction", 0.1) * len(rest)))))
        rng = np.random.default_rng([params.get("seed", 0), zlib.crc32(("explore:" + fold["name"]).encode())])
        pick = set(rng.choice(sorted(rest), size=k, replace=False).tolist())
        rule = (f"uniform without replacement over the {len(rest)} unranked eligible inputs, seed={params.get('seed', 0)}, "
                f"fold={fold['name']}; drawn independently of relevance")
        for f in rest:
            if f in pick:
                rows[f].update(tier="EXPLORATORY", explore_probability=k / len(rest), explore_rule=rule)
                rows[f]["reasons"].append("EXPLORATORY_DRAW (outside the ranking)")
            else:
                rows[f].update(tier="DEFERRED", reincorporation=REINCORPORATION)
                rows[f]["reasons"].append(f"BELOW_TOP_{top_q}_REVERSIBLE max|rho|={score[f]:.4g}")
    return [rows[f] for f in names]


# ----------------------------------------------------------------- dataset runner
def _load_wide():
    spec = importlib.util.spec_from_file_location("wide", Path(__file__).resolve().parent / "profile_train_wide.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _overlay_outer(cells, v2_metrics: Path, v2_profile_sha: str):
    """Attach verified full-TRAIN v2 values to the deferred outer-train cells."""
    want = {(c["feature"], c["metric"]): c for c in cells if c["fold"] == "outer_train" and c["family"] in DEFERRED_FAMILIES}
    with v2_metrics.open() as fh:
        for r in csv.DictReader(fh):
            c = want.get((r["column"], r["metric"]))
            if not c:
                continue
            if r["status"] in ("OK", "OK_WITH_WARNING"):
                c.update(status="MEASURED_REUSED", value=float(r["value"]) if r["value"] not in ("", "None") else r["value"],
                         reason=f"v2 full-TRAIN profile {v2_profile_sha[:16]} ({r['status']})")
            else:
                c.update(status="NOT_RUN" if r["status"] != "FAILED" else "FAILED",
                         reason=f"v2 profile {r['status']}: {r['reason']}")


def run_dataset(manifest_path, declaration_path, source_root, out_dir, cache_root, purge, v2_dir=None,
                targets_spec=None, target_note=None, fold_rule=None):
    W = _load_wide()
    m = json.loads(Path(manifest_path).read_text())
    decl_bytes = Path(declaration_path).read_bytes()
    decl = json.loads(decl_bytes)
    root = Path(source_root).resolve()
    path = (root / m["path"]).resolve()
    n = m["boundaries"]["train"][1]
    digest, _, records, end_offset = W.identity_pass(path, 1 << 30, n)
    if digest != m["resource_sha256"] or digest != decl["resource_sha256"]:
        raise ValueError("resource identity differs from manifest or declaration")
    header, Xall, stamps, nonnum, _, prefix_sha = W.read_train(path, end_offset, m)
    if prefix_sha != decl["train_prefix_sha256"]:
        raise ValueError("TRAIN prefix differs from the admissible declaration")
    names = decl["all_admissible_control"]
    X = Xall[:, [header.index(f) for f in names]]
    params = {"declared_periods_rows": m.get("declared_periods_rows", {}), "primary_period": m.get("primary_period"),
              "acf_lags": sorted({1} | set(m.get("declared_periods_rows", {}).values())
                                 | {2 * m["declared_periods_rows"][m["primary_period"]]} if m.get("primary_period") else {1}),
              "redundancy_threshold": 0.95, "top_q": 10, "explore_fraction": 0.1, "explore_min": 2,
              "max_synergy_pairs": 10, "pair_feature_cap": 64, "seed": 20261001}
    folds = [outer_fold(n)] + inner_folds(n, k=3, val_frac=0.15, purge=purge)
    identity = {"dataset_id": m["dataset_id"], "resource_sha256": digest, "train_prefix_sha256": prefix_sha,
                "vintage": m.get("vintage", m["resource_sha256"][:16])}
    cache = ProfileCache(Path(cache_root))
    t0 = time.time()
    cells = run_ps1(X, names, folds, params, cache=cache, identity=identity)
    if v2_dir:
        v2p = json.loads((Path(v2_dir) / "profile.json").read_text())
        if v2p["train_prefix"]["prefix_sha256"] != prefix_sha:
            raise ValueError("v2 profile was computed on different TRAIN bytes; not reused")
        _overlay_outer(cells, Path(v2_dir) / "metrics_long.csv",
                       hashlib.sha256((Path(v2_dir) / "profile.json").read_bytes()).hexdigest())
    ps1_seconds = time.time() - t0
    # PS2
    targets, target_status = {}, []
    if targets_spec:
        import pandas as pd
        ts = pd.to_datetime(pd.Series(stamps), format=m.get("timestamp_format")).astype("int64").to_numpy() // 10**9
        price = Xall[:, header.index(targets_spec["asset_column"])]
        targets = build_targets(price, ts, targets_spec["asset_column"], targets_spec["asset_column"],
                                targets_spec["horizons"], m["step_seconds"], targets_spec.get("barrier_rule"))
    for k, t in targets.items():
        target_status.append({"target": k[0], "horizon_hours": k[1], "status": t.status, "note": t.note})
    if not targets:
        applicable = not (target_note or "").startswith("NOT_APPLICABLE")
        target_status.append({"target": "Y_s/Y_l/Y_b", "horizon_hours": None,
                              "status": "NOT_EVALUATED" if applicable else "NOT_APPLICABLE",
                              "note": target_note or "no business target declared for this resource"})
    worklists = []
    acf1 = {(c["feature"], c["fold"]): c["value"] for c in cells
            if c["metric"] == "acf_lag_1" and c["status"] in ("MEASURED", "MEASURED_REUSED")}
    for fold in folds[1:]:
        for r in prioritize(X, names, targets, fold, params, no_target_reason=target_note):
            a1 = acf1.get((r["feature"], fold["name"]))
            if r["tier"] in ("PRIORITY", "SYNERGY", "REPRESENTATIVE") and a1 is not None and a1 >= 0.99:
                r["reasons"].append(f"CAUTION_PERSISTENT_INPUT acf1={a1:.4f}: a Spearman screen against overlapping "
                                    "multi-bar return labels on a near-unit-root input can reflect regime/trend "
                                    "confounding; PS4 must test a differenced or ratio variant before any claim")
            worklists.append(r)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "ps1_cells.csv", cells, ["feature", "fold", "family", "metric", "status", "value", "reason"])
    rule = fold_rule or "inner expanding chronological folds inside the declared TRAIN prefix"
    _write_csv(out / "metric_coverage.csv", [dict(r, not_run_reasons=json.dumps(r["not_run_reasons"]), dataset=m["dataset_id"],
                                                  fold_rule=rule) for r in metric_coverage(cells)],
               ["dataset", "fold_rule", "fold", "family", "metric", "denominator", "measured", "reused", "failed", "not_run",
                "not_run_reasons"])
    cov = coverage(cells, names, folds)
    _write_csv(out / "coverage_feature_fold.csv",
               [dict(feature=f, fold=fo, basic_complete=v["basic_complete"], full_complete=v["full_complete"],
                     **{f"fam_{k}": s for k, s in v["families"].items()}) for (f, fo), v in cov.items()],
               ["feature", "fold", "basic_complete", "full_complete"] + [f"fam_{x}" for x in PS1_FAMILIES + DEFERRED_FAMILIES])
    _write_csv(out / "worklist.csv", [dict(r, reasons=" | ".join(r["reasons"]), score_by_target=json.dumps(r["score_by_target"]))
                                      for r in worklists],
               ["feature", "fold", "tier", "group", "score_by_target", "explore_probability", "explore_rule",
                "reincorporation", "reasons"])
    _write_csv(out / "target_status.csv", target_status, ["target", "horizon_hours", "status", "note"])
    tiers = {}
    for r in worklists:
        tiers.setdefault(r["fold"], {}).setdefault(r["tier"], 0)
        tiers[r["fold"]][r["tier"]] += 1
    stab = {}
    for r in worklists:
        if r["tier"] in ("PRIORITY", "SYNERGY"):
            stab[r["feature"]] = stab.get(r["feature"], 0) + 1
    summary = {"schema": "lane_b_ps0_ps2_run.v1", "dataset_id": m["dataset_id"], "identity": identity,
               "fold_rule": fold_rule or "inner expanding chronological folds inside the declared TRAIN prefix",
               "declaration_sha256": decl["declaration_sha256"], "declaration_file_sha256": hashlib.sha256(decl_bytes).hexdigest(),
               "code_sha256": CODE_SHA, "params": params, "folds": folds, "features": len(names),
               "cells": len(cells), "cells_by_status": _count(cells, "status"),
               "basic_complete": sum(v["basic_complete"] for v in cov.values()),
               "full_complete": sum(v["full_complete"] for v in cov.values()), "feature_fold_pairs": len(cov),
               "targets": target_status, "tiers_by_fold": tiers,
               "priority_or_synergy_in_k_of_3_folds": stab, "ps1_wall_seconds": ps1_seconds}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def _count(rows, key):
    out = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def _write_csv(path, rows, fields):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--purge", type=int, required=True, help="rows: lookback + maximum horizon support")
    ap.add_argument("--v2-profile-dir")
    ap.add_argument("--fold-rule", help="declared fold rule of this dataset family, recorded in the coverage table")
    ap.add_argument("--targets-json", help='{"asset_column":..., "horizons":{...}, "barrier_rule":...} or {"why": ...}')
    a = ap.parse_args()
    spec = json.loads(a.targets_json) if a.targets_json else {}
    s = run_dataset(a.manifest, a.declaration, a.source_root, a.output, a.cache_root, a.purge, a.v2_profile_dir,
                    spec if "asset_column" in spec else None, spec.get("why"), a.fold_rule)
    print(json.dumps({k: s[k] for k in ("dataset_id", "features", "cells", "cells_by_status", "basic_complete",
                                         "full_complete", "feature_fold_pairs", "tiers_by_fold")}, default=str))


if __name__ == "__main__":
    main()
