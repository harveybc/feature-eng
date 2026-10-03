"""PS2 reversible priority for the EURUSD selection manifest (inner chronological TRAIN folds only).

Per target (Y_s h=1..6 h, Y_l h=24..144 h, Y_b barrier), horizon and inner fold this computes:

* robust associations: Spearman and binned mutual information, each calibrated against a temporal
  null (evenly spaced circular shifts of the feature against the target, minimum shift declared),
  with Benjamini-Hochberg q-values per (target, horizon, fold, statistic) family and the number of
  tests of every family recorded (multiplicity bookkeeping);
* out-of-fold incremental utility of a univariate binned-mean model (fit on the fold's fit rows,
  scored on the fold's evaluation rows, both inside TRAIN) against the intercept / fit-row mean
  baseline; a linear model and the zero-return persistence naive are recorded on the same rows;
* conditional redundancy inside dependence clusters (|Spearman| complete linkage on fit rows) and
  clusters of standardized TRAIN-only profiles. Clusters PROPOSE groups; they never select;
* domain groups (joint utility and drop-one contribution) and pairwise synergy checks
  (linear model with the product term), so a pair useful only jointly resurfaces (FS16);
* an exploration sample drawn by a declared hash rule that never reads the ranking.

Statuses per (feature, target, horizon):
  PROVISIONAL_SURVIVOR, PROVISIONAL_LOW_PRIORITY, EXPLORATION,
  TECHNICAL_REJECT  (only unavailable / leak / invalid; never association or utility).
Every status carries reason codes. Nothing is a final discard: the all-admissible control list is
always emitted, and every LOW_PRIORITY row lists how it can re-enter.

Isolation: the caller passes the TRAIN boundary; every row at or after it is removed before any
computation. Inner fold k only reads rows whose time is <= its evaluation end, labels are kept only
when their full support ends inside the segment they are used in, and every seed / shift depends
only on names, fold index and declared parameters, never on data values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

SCHEMA = "feature_eng.ps2_selection.v1"
STATUSES = ("PROVISIONAL_SURVIVOR", "PROVISIONAL_LOW_PRIORITY", "EXPLORATION", "TECHNICAL_REJECT")
TECHNICAL_CODES = ("T_UNAVAILABLE", "T_UNAVAILABLE_CONSTANT_IN_TRAIN", "T_LEAK_DECLARED",
                   "T_LEAK_NAME_TOKEN", "T_INVALID_DECLARED")
LOOKAHEAD_TOKENS = ("future", "fwd_", "forward", "lead_", "next_", "target", "label", "centered")
REINCORPORATION = ("re-enters on: PS4/PS5 inner-validation gain alone or as a member of a group "
                   "or pair, a later exploration draw, a new vintage, a new declared target, or "
                   "causal/extractibility evidence (PS3-C/PS3-R)")
HOUR = 3600
DEFAULT_PARAMS = {
    "targets": {"Y_s": [1, 2, 3, 4, 5, 6], "Y_l": [24, 48, 72, 96, 120, 144], "Y_b": [24]},
    "barrier": {"horizon_hours": 24, "k_sigma": 1.0, "vol_window_hours": 168,
                "min_vol_returns": 48, "price_rule": "close-only first touch, ties impossible",
                "classes": {"-1": "lower first", "0": "timeout", "1": "upper first"}},
    "label_rule": "log(price_asof(t+h)/price(t)); price_asof = last TRAIN bar at or before t+h "
                  "elapsed hours; label support ends at t+h",
    "n_inner_folds": 3,
    "inner_eval_frac": 0.15,
    "min_fit_rows": 300,
    "min_eval_rows": 100,
    "n_null": 99,
    "null_min_shift_rows": 336,
    "mi_bins": 8,
    "util_bins": 10,
    "bin_shrink": 20.0,
    "winsor_q": 0.005,
    "fdr_q": 0.10,
    "fold_majority": 2,
    "redundancy_threshold": 0.90,
    "redundancy_min_pairs": 50,
    "profile_acf_lags": [1, 24, 168],
    "synergy_max_pairs": 3000,
    "synergy_top_k": 20,
    "synergy_min_rel_gain": 0.001,
    "n_util_null": 19,
    "util_null_p": 0.10,
    "synergy_null_p": 0.05,
    "synergy_fold_min": 3,
    "group_max_members": 200,
    "group_drop_one_max": 30,
    "exploration_fraction": 0.2,
    "exploration_min": 2,
    "exploration_seed": 20261003,
    "leak_alarm_abs_spearman": 0.98,
}


class PS2Error(ValueError):
    pass


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_digest() -> str:
    with open(__file__, "rb") as fh:
        return _sha(fh.read())


def _f(x):
    """Float for output: None for non-finite; full repr precision (1e-6 differences survive)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------------------- data

@dataclass
class Batch:
    ts: np.ndarray            # int64 epoch seconds, strictly increasing, TRAIN rows only
    X: np.ndarray             # float64 (n, p), NaN = missing
    names: list
    price: np.ndarray         # float64 (n,), close used for the labels
    domains: dict = field(default_factory=dict)      # feature -> domain group
    declared: dict = field(default_factory=dict)     # feature -> (code, detail) technical
    batch_id: str = "batch"


def restrict_to_train(ts, X, price, train_end: int):
    """Drop every row at or after the TRAIN boundary. Nothing after it is ever read again."""
    ts = np.asarray(ts, dtype="int64")
    keep = ts < int(train_end)
    if not keep.any():
        raise PS2Error("no TRAIN rows before the boundary")
    ts, X, price = ts[keep], np.asarray(X, dtype="float64")[keep], np.asarray(price, "float64")[keep]
    if np.any(np.diff(ts) <= 0):
        raise PS2Error("timestamps must be strictly increasing")
    return ts, X, price


def train_digest(b: Batch) -> str:
    h = hashlib.sha256()
    h.update(_canon(list(b.names)).encode())
    h.update(np.ascontiguousarray(b.ts, dtype="int64").tobytes())
    h.update(np.ascontiguousarray(b.X, dtype="float64").tobytes())
    h.update(np.ascontiguousarray(b.price, dtype="float64").tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- targets

def build_targets(ts, price, p):
    """Return {(target, h): (y, support_end)} on TRAIN rows; y NaN when its support leaves TRAIN."""
    ts = np.asarray(ts, "int64")
    logp = np.log(np.asarray(price, "float64"))
    last = ts[-1]
    out = {}
    for tname in ("Y_s", "Y_l"):
        for h in p["targets"].get(tname, []):
            end = ts + int(h) * HOUR
            j = np.searchsorted(ts, end, side="right") - 1
            y = logp[j] - logp
            y[end > last] = np.nan
            out[(tname, int(h))] = (y, end)
    for H in p["targets"].get("Y_b", []):
        y, end = _barrier(ts, logp, int(H), p["barrier"])
        out[("Y_b", int(H))] = (y, end)
    return out


def _barrier(ts, logp, H, bp):
    n = len(ts)
    r = np.diff(logp, prepend=np.nan)
    r[0] = np.nan
    ok = np.isfinite(r)
    c1 = np.cumsum(np.where(ok, r, 0.0))
    c2 = np.cumsum(np.where(ok, r * r, 0.0))
    cn = np.cumsum(ok.astype("int64"))
    start = np.searchsorted(ts, ts - bp["vol_window_hours"] * HOUR, side="right")
    idx = np.arange(n)
    prev = start - 1
    s1 = c1 - np.where(prev >= 0, c1[np.maximum(prev, 0)], 0.0)
    s2 = c2 - np.where(prev >= 0, c2[np.maximum(prev, 0)], 0.0)
    cnt = cn - np.where(prev >= 0, cn[np.maximum(prev, 0)], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.maximum(cnt, 1)) / np.maximum(cnt - 1, 1)
    sig = np.sqrt(np.maximum(var, 0.0)) * math.sqrt(H) * bp["k_sigma"]
    end = ts + H * HOUR
    y = np.full(n, np.nan)
    stop = np.searchsorted(ts, end, side="right")
    last = ts[-1]
    for i in idx:
        if end[i] > last or cnt[i] < bp["min_vol_returns"] or not sig[i] > 0:
            continue
        path = logp[i + 1:stop[i]] - logp[i]
        up = np.nonzero(path >= sig[i])[0]
        dn = np.nonzero(path <= -sig[i])[0]
        fu = up[0] if up.size else np.inf
        fd = dn[0] if dn.size else np.inf
        y[i] = 1.0 if fu < fd else (-1.0 if fd < fu else 0.0)
    return y, end


# ----------------------------------------------------------------------------- folds

def inner_folds(ts, p):
    """Expanding chronological folds inside TRAIN. Boundaries depend on the row count only."""
    n = len(ts)
    K = int(p["n_inner_folds"])
    v = max(1, int(round(n * p["inner_eval_frac"])))
    folds = []
    for k in range(1, K + 1):
        e_end = n - (K - k) * v
        e_start = e_end - v
        if e_start <= 1:
            raise PS2Error("inner fold has no fit rows; reduce n_inner_folds or inner_eval_frac")
        folds.append({"fold": f"inner_{k}", "eval_start_row": int(e_start),
                      "eval_end_row": int(e_end), "eval_start_time": int(ts[e_start]),
                      "eval_end_time": int(ts[e_end - 1])})
    return folds


def fold_rows(fold, ts, y, end):
    """Fit rows: label support ends before the eval segment starts. Eval rows: support ends by eval end."""
    i = np.arange(len(ts))
    good = np.isfinite(y)
    fit = good & (i < fold["eval_start_row"]) & (end < fold["eval_start_time"])
    ev = good & (i >= fold["eval_start_row"]) & (i < fold["eval_end_row"]) & \
        (end <= fold["eval_end_time"])
    return np.nonzero(fit)[0], np.nonzero(ev)[0]


# ----------------------------------------------------------------------------- statistics

def _ranks(a):
    from scipy.stats import rankdata
    return rankdata(a, method="average")


def _null_shifts(m, p):
    lo = int(p["null_min_shift_rows"])
    if m < 2 * lo + 10:
        lo = max(1, m // 4)
    hi = m - lo
    if hi <= lo:
        return np.array([], dtype="int64")
    return np.unique(np.linspace(lo, hi, int(p["n_null"])).round().astype("int64"))


def spearman_null(x, y, shifts):
    """Observed Spearman and the circular-shift null for all shifts at once (FFT)."""
    rx = _ranks(x); ry = _ranks(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if den == 0:
        return None, None
    obs = float(rx @ ry) / den
    m = len(rx)
    cc = np.fft.irfft(np.conj(np.fft.rfft(ry)) * np.fft.rfft(rx), n=m)  # cc[s]=sum rx[i+s]ry[i]
    null = cc[shifts] / den if len(shifts) else np.array([])
    return obs, null


def _qbins(v, B):
    r = _ranks(v)
    return np.minimum((r - 1) * B // len(v), B - 1).astype("int64")


def mi_null(x, y, shifts, B, y_classes=False):
    bx = _qbins(x, B)
    if y_classes:
        by = (np.round(y).astype("int64") + 1)
        By = 3
    else:
        by = _qbins(y, B); By = B

    def mi(a):
        j = np.bincount(a * By + by, minlength=B * By).reshape(B, By).astype("float64")
        j /= j.sum()
        px = j.sum(1, keepdims=True); py = j.sum(0, keepdims=True)
        nz = j > 0
        return float((j[nz] * np.log(j[nz] / (px @ py)[nz])).sum())
    obs = mi(bx)
    null = np.array([mi(np.roll(bx, int(s))) for s in shifts])
    return obs, null


def emp_p(obs, null):
    if obs is None or null is None or len(null) == 0:
        return None
    return (1.0 + float(np.sum(np.abs(null) >= abs(obs) - 1e-15))) / (1.0 + len(null))


def emp_p_upper(obs, null):
    """One-sided: share of null utilities at least as large as the observed utility."""
    if obs is None or null is None or len(null) == 0:
        return None
    return (1.0 + float(np.sum(null >= obs - 1e-18))) / (1.0 + len(null))


def _roll_shifts(m, k, p):
    lo = min(int(p["null_min_shift_rows"]), max(1, m // 4))
    hi = m - lo
    if hi <= lo or k <= 0:
        return []
    return [int(v) for v in np.unique(np.linspace(lo, hi, k).round().astype("int64"))]


def _util_null(xf, yf, xe, ye, p, classes):
    """Temporal null for OOF utility: circular shifts of x inside the fit and eval segments."""
    k = int(p["n_util_null"])
    sf, se = _roll_shifts(len(xf), k, p), _roll_shifts(len(xe), k, p)
    out = []
    for a, c in zip(sf, se):
        lb, lm = binned_utility(np.roll(xf, a), yf, np.roll(xe, c), ye, p, classes)
        out.append(lb - lm)
    return np.array(out)


def bh(pvals):
    """Benjamini-Hochberg q-values; None stays None and is not counted as a test."""
    idx = [i for i, v in enumerate(pvals) if v is not None]
    q = [None] * len(pvals)
    m = len(idx)
    if not m:
        return q, 0
    order = sorted(idx, key=lambda i: pvals[i])
    run = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, pvals[i] * m / rank)
        q[i] = run
    return q, m


# ----------------------------------------------------------------------------- utility models

class _Std:
    """Winsorized standardization with NaN -> 0 (the mean), fit on fit rows only."""

    def __init__(self, x, wq):
        f = x[np.isfinite(x)]
        if f.size < 2:
            self.lo = self.hi = self.mu = 0.0; self.sd = 1.0
            return
        self.lo, self.hi = np.quantile(f, [wq, 1 - wq])
        c = np.clip(f, self.lo, self.hi)
        self.mu = float(c.mean()); self.sd = float(c.std()) or 1.0

    def __call__(self, x):
        z = (np.clip(x, self.lo, self.hi) - self.mu) / self.sd
        return np.where(np.isfinite(z), z, 0.0)


def _lin_loss(Zf, yf, Ze, ye):
    A = np.column_stack([np.ones(len(yf)), Zf])
    coef, *_ = np.linalg.lstsq(A, yf, rcond=None)
    pe = np.column_stack([np.ones(len(ye)), Ze]) @ coef
    return float(np.mean((ye - pe) ** 2))


def binned_utility(xf, yf, xe, ye, p, classes=False):
    """Univariate binned-mean (regression) or binned class frequency (Y_b) model vs fit-row mean.

    Returns (baseline_loss, model_loss). Loss is MSE, or log-loss for classes. NaN is its own bin.
    """
    B, s = int(p["util_bins"]), float(p["bin_shrink"])
    fin = np.isfinite(xf)
    edges = np.unique(np.quantile(xf[fin], np.linspace(0, 1, B + 1)[1:-1])) if fin.any() else []

    def bins(x):
        b = np.searchsorted(edges, x, side="right") + 1
        return np.where(np.isfinite(x), b, 0)
    nb = len(edges) + 2
    bf, be = bins(xf), bins(xe)
    if not classes:
        mu = float(yf.mean())
        sm = np.bincount(bf, weights=yf, minlength=nb)
        ct = np.bincount(bf, minlength=nb)
        pred = ((sm + s * mu) / (ct + s))[be]
        return float(np.mean((ye - mu) ** 2)), float(np.mean((ye - pred) ** 2))
    cf = np.round(yf).astype(int) + 1; ce = np.round(ye).astype(int) + 1
    base = (np.bincount(cf, minlength=3) + 1.0) / (len(cf) + 3.0)
    tab = np.zeros((nb, 3))
    np.add.at(tab, (bf, cf), 1.0)
    prob = (tab + s * base) / (tab.sum(1, keepdims=True) + s)
    ll_b = float(-np.mean(np.log(base[ce])))
    ll_m = float(-np.mean(np.log(prob[be, ce])))
    return ll_b, ll_m


# ----------------------------------------------------------------------------- profiles / clusters

def profile_vector(x, lags):
    f = x[np.isfinite(x)]
    n = len(x)
    out = [1.0 - len(f) / max(n, 1)]
    if f.size < 3 or f.std() == 0:
        return out + [0.0] * (4 + len(lags))
    z = (f - f.mean()) / f.std()
    out += [float(np.mean(z ** 3)), float(np.log1p(abs(np.mean(z ** 4) - 3))),
            float(np.mean(f == 0)), float(len(np.unique(f)) / f.size)]
    for L in lags:
        out.append(float(np.mean(z[L:] * z[:-L])) if f.size > L + 2 else 0.0)
    return out


def complete_link_clusters(names, dist, cut):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(names) < 2:
        return [list(names)] if names else []
    d = np.clip((dist + dist.T) / 2.0, 0.0, None)
    np.fill_diagonal(d, 0.0)
    lab = fcluster(linkage(squareform(d, checks=False), method="complete"), t=cut,
                   criterion="distance")
    g: dict = {}
    for nm, l in zip(names, lab):
        g.setdefault(int(l), []).append(nm)
    return sorted((sorted(v) for v in g.values()), key=lambda v: v[0])


def dependence_clusters(X, names, rows, p, cols):
    """|Spearman| (pairwise complete, fit rows only) complete linkage; returns clusters and matrix."""
    import pandas as pd
    if len(names) == 0:
        return [], None
    df = pd.DataFrame(X[np.ix_(rows, cols)], columns=names)
    rho = df.corr(method="spearman", min_periods=int(p["redundancy_min_pairs"])).abs()
    d = 1.0 - rho.fillna(0.0).to_numpy()
    return complete_link_clusters(list(names), d, 1.0 - p["redundancy_threshold"] + 1e-12), rho


def profile_clusters(X, names, rows, p, cols):
    if len(names) == 0:
        return []
    P = np.array([profile_vector(X[rows, j], p["profile_acf_lags"]) for j in cols])
    sd = P.std(0); sd[sd == 0] = 1.0
    Z = (P - P.mean(0)) / sd
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(names) < 2:
        return [list(names)]
    k = max(1, int(math.ceil(math.sqrt(len(names)))))
    lab = fcluster(linkage(Z, method="average"), t=k, criterion="maxclust")
    g: dict = {}
    for nm, l in zip(names, lab):
        g.setdefault(int(l), []).append(nm)
    return sorted((sorted(v) for v in g.values()), key=lambda v: v[0])


def domain_of(name, domains):
    if name in domains and domains[name]:
        return str(domains[name])
    for sep in ("__", ":", "."):
        if sep in name:
            return name.split(sep)[0]
    return name.split("_")[0] if "_" in name else "ungrouped"


# ----------------------------------------------------------------------------- exploration

def _u(seed, name):
    return int(_sha(f"{seed}|{name}".encode())[:16], 16) / float(1 << 64)


def exploration_draw(admissible, low_pool, p):
    """Hash draw that never reads the ranking; top-up from the low-priority pool if it missed it."""
    seed, frac, mn = p["exploration_seed"], float(p["exploration_fraction"]), int(p["exploration_min"])
    u = {f: _u(seed, f) for f in admissible}
    drawn = sorted(f for f in admissible if u[f] < frac)
    topup = []
    covered = [f for f in drawn if f in low_pool]
    if len(covered) < min(mn, len(low_pool)):
        for f in sorted(low_pool, key=lambda f: (u[f], f)):
            if f not in drawn and len(covered) + len(topup) < min(mn, len(low_pool)):
                topup.append(f)
    rule = {"rule": "u(f)=int(sha256(f'{seed}|{feature}')[:16],16)/2**64; draw if u<fraction over "
                    "all admissible features (ranking never read); if fewer than exploration_min "
                    "drawn features are low priority in some cell, add low-priority features by "
                    "ascending u until exploration_min",
            "seed": seed, "fraction": frac, "min": mn, "inclusion_probability_stage1": frac,
            "u": {f: u[f] for f in sorted(u)}}
    return sorted(drawn + topup), drawn, topup, rule


# ----------------------------------------------------------------------------- main build

def technical_screen(b: Batch):
    tech = {}
    for j, f in enumerate(b.names):
        codes = []
        if f in b.declared and b.declared[f]:
            code, detail = b.declared[f]
            codes.append((code if code in TECHNICAL_CODES else "T_INVALID_DECLARED", detail))
        low = f.lower()
        if any(t in low for t in LOOKAHEAD_TOKENS):
            codes.append(("T_LEAK_NAME_TOKEN", "forward-looking token in name"))
        x = b.X[:, j]
        fin = x[np.isfinite(x)]
        if fin.size == 0:
            codes.append(("T_UNAVAILABLE", "no finite TRAIN value"))
        elif np.unique(fin).size <= 1:
            codes.append(("T_UNAVAILABLE_CONSTANT_IN_TRAIN", "one distinct finite TRAIN value"))
        if codes:
            tech[f] = codes
    return tech


def build(b: Batch, params: dict | None = None, log=None) -> dict:
    t0 = time.time()
    p = json.loads(json.dumps(DEFAULT_PARAMS))
    for k, v in (params or {}).items():
        p[k] = v
    names = list(b.names)
    tech = technical_screen(b)
    adm = [f for f in names if f not in tech]
    col = {f: j for j, f in enumerate(names)}
    targets = build_targets(b.ts, b.price, p)
    folds = inner_folds(b.ts, p)
    cells = []          # per feature x target x horizon x fold
    groups_rows, syn_rows = [], []
    fold_clusters = {}
    tests_total = 0
    for fd in folds:
        # target-free structure on the fold's fit segment (rows before eval start)
        seg = np.arange(fd["eval_start_row"])
        acols = [col[f] for f in adm]
        dcl, rho = dependence_clusters(b.X, adm, seg, p, acols)
        pcl = profile_clusters(b.X, adm, seg, p, acols)
        fold_clusters[fd["fold"]] = {"dependence": dcl, "profile": pcl}
        for (tn, h), (y, end) in sorted(targets.items()):
            classes = tn == "Y_b"
            fr, er = fold_rows(fd, b.ts, y, end)
            fam = []
            for f in adm:
                x = b.X[:, col[f]]
                c = {"feature": f, "target": tn, "horizon": h, "fold": fd["fold"],
                     "n_fit": int(len(fr)), "n_eval": int(len(er))}
                cf = fr[np.isfinite(x[fr])]
                if len(fr) < p["min_fit_rows"] or len(er) < p["min_eval_rows"] or \
                        len(cf) < p["min_fit_rows"] or np.unique(x[cf]).size < 2:
                    c["cell_status"] = "NOT_EVALUATED"
                    c["cell_reason"] = "insufficient fit/eval support or constant in fold"
                    cells.append(c); fam.append(c)
                    continue
                xc, yc = x[cf], y[cf]
                sh = _null_shifts(len(cf), p)
                rs, rnull = spearman_null(xc, yc, sh)
                mi, minull = mi_null(xc, yc, sh, int(p["mi_bins"]), classes)
                lb, lm = binned_utility(x[fr], y[fr], x[er], y[er], p, classes)
                unull = _util_null(x[fr], y[fr], x[er], y[er], p, classes)
                st = _Std(x[fr], p["winsor_q"])
                ylin_f = y[fr]; ylin_e = y[er]
                base_mse = float(np.mean((ylin_e - ylin_f.mean()) ** 2))
                lin = _lin_loss(st(x[fr])[:, None], ylin_f, st(x[er])[:, None], ylin_e)
                c.update(cell_status="MEASURED", spearman=rs, spearman_p=emp_p(rs, rnull),
                         spearman_null_abs_q95=_f(np.quantile(np.abs(rnull), 0.95))
                         if rnull is not None and len(rnull) else None,
                         mi=mi, mi_null_mean=_f(minull.mean()) if len(minull) else None,
                         mi_p=emp_p(mi, minull), n_null=int(len(sh)),
                         null_min_shift_rows=int(sh.min()) if len(sh) else None,
                         loss_name="logloss" if classes else "mse",
                         loss_base=lb, loss_model=lm, oof_delta=lb - lm,
                         oof_skill=(lb - lm) / lb if lb > 0 else None,
                         oof_null_p=emp_p_upper(lb - lm, unull), n_util_null=int(len(unull)),
                         lin_mse_base=base_mse, lin_mse=lin, lin_delta=base_mse - lin,
                         zero_naive_mse=None if classes else float(np.mean(ylin_e ** 2)))
                cells.append(c); fam.append(c)
            for stat in ("spearman", "mi"):
                q, m = bh([c.get(stat + "_p") for c in fam])
                tests_total += m
                for c, qq in zip(fam, q):
                    c[stat + "_q"] = qq
                    c[stat + "_family_m"] = m
            if log:
                log(f"{fd['fold']} {tn} h{h}: {len(fam)} features fit={len(fr)} eval={len(er)}")
            groups_rows += _groups(b, adm, col, fd, tn, h, y, fr, er, dcl, p, classes)
            syn_rows += _synergy(b, adm, col, fd, tn, h, y, fr, er, p, cells, classes)
    status_rows = _decide(b, names, adm, tech, cells, groups_rows, syn_rows, folds, targets, p)
    low_pool = sorted({r["feature"] for r in status_rows
                       if r["status"] == "PROVISIONAL_LOW_PRIORITY"})
    sample, drawn, topup, rule = exploration_draw(adm, low_pool, p)
    for r in status_rows:
        r["exploration_sample"] = r["feature"] in sample
        if r["feature"] in sample and r["status"] == "PROVISIONAL_LOW_PRIORITY":
            r["status"] = "EXPLORATION"
            r["reasons"] = r["reasons"] + ["X_EXPLORATION_SAMPLE"]
    counts = {}
    for r in status_rows:
        k = f"{r['target']}|h{r['horizon']}"
        counts.setdefault(k, {s: 0 for s in STATUSES})[r["status"]] += 1
    pdig = _sha(_canon(p).encode())
    res = {"schema": SCHEMA, "batch_id": b.batch_id, "fit_scope": "inner chronological TRAIN folds",
           "train_rows": int(len(b.ts)), "train_first_time": int(b.ts[0]),
           "train_last_time": int(b.ts[-1]), "train_data_digest": train_digest(b),
           "parameters": p, "parameter_digest": pdig, "code_digest": code_digest(),
           "folds": folds, "n_features": len(names), "n_admissible": len(adm),
           "control_all_admissible": adm, "technical_rejects": {f: tech[f] for f in sorted(tech)},
           "fold_clusters": fold_clusters, "multiplicity": {
               "tests_total": tests_total, "families": "target x horizon x fold x statistic",
               "correction": "Benjamini-Hochberg within family", "fdr_q": p["fdr_q"]},
           "exploration": {"sample": sample, "stage1_drawn": drawn, "topup": topup,
                           "low_priority_pool": low_pool, **rule},
           "counts": counts, "status_policy": STATUS_POLICY,
           "reincorporation": REINCORPORATION, "status": status_rows, "cells": cells,
           "groups": groups_rows, "synergy": syn_rows,
           "elapsed_seconds_unhashed": round(time.time() - t0, 3)}
    return res


STATUS_POLICY = {
    "TECHNICAL_REJECT": "declared unavailable/leak/invalid by PS0/PS1, forward-looking name "
                        "token, no finite TRAIN value, or constant in TRAIN. Never association.",
    "PROVISIONAL_SURVIVOR": "any of S_OOF_UTILITY (binned OOF delta>0 in >=fold_majority folds, "
                            "median>0, and temporal-null p<=util_null_p in >=fold_majority folds), "
                            "S_ASSOC_ROBUST (Spearman or MI q<=fdr_q in "
                            ">=fold_majority folds), S_SYNERGY_PAIR, S_GROUP_CONTRIBUTION",
    "PROVISIONAL_LOW_PRIORITY": "otherwise, or LP_CONDITIONALLY_REDUNDANT (survivor whose "
                                "conditional delta within its dependence cluster is <=0 in "
                                ">=fold_majority folds while a cluster member has a larger "
                                "median OOF delta). Reversible; kept in the control list.",
    "EXPLORATION": "a low-priority cell of a feature in the exploration sample",
}


def _zmat(b, col, feats, fr, er, p):
    Zf, Ze = [], []
    for f in feats:
        x = b.X[:, col[f]]
        s = _Std(x[fr], p["winsor_q"])
        Zf.append(s(x[fr])); Ze.append(s(x[er]))
    return np.column_stack(Zf), np.column_stack(Ze)


def _yb_reg(y):
    return y  # class codes -1/0/1 as signed value for linear checks (declared)


def _groups(b, adm, col, fd, tn, h, y, fr, er, dcl, p, classes):
    if len(fr) < p["min_fit_rows"] or len(er) < p["min_eval_rows"]:
        return []
    yf, ye = _yb_reg(y[fr]), _yb_reg(y[er])
    base = float(np.mean((ye - yf.mean()) ** 2))
    out = []
    dom: dict = {}
    for f in adm:
        dom.setdefault(domain_of(f, b.domains), []).append(f)
    for kind, glist in (("domain", sorted(dom.items())),
                        ("dependence_cluster", [(f"dc{i}", m) for i, m in enumerate(dcl)])):
        for gid, mem in glist:
            if len(mem) < 2 or len(mem) > p["group_max_members"]:
                continue
            Zf, Ze = _zmat(b, col, mem, fr, er, p)
            full = _lin_loss(Zf, yf, Ze, ye)
            row = {"kind": kind, "group": gid, "members": list(mem), "target": tn, "horizon": h,
                   "fold": fd["fold"], "mse_base": base, "mse_group": full,
                   "group_delta": base - full, "drop_one": {}}
            if len(mem) > p["group_drop_one_max"]:
                row["drop_one_status"] = "NOT_EVALUATED_GROUP_LARGER_THAN_group_drop_one_max"
                out.append(row)
                continue
            for i, f in enumerate(mem):
                keep = [k for k in range(len(mem)) if k != i]
                row["drop_one"][f] = _lin_loss(Zf[:, keep], yf, Ze[:, keep], ye) - full
            out.append(row)
    return out


def _synergy(b, adm, col, fd, tn, h, y, fr, er, p, cells, classes):
    if len(fr) < p["min_fit_rows"] or len(er) < p["min_eval_rows"] or len(adm) < 2:
        return []
    pairs = [(adm[i], adm[j]) for i in range(len(adm)) for j in range(i + 1, len(adm))]
    rule = "all pairs"
    if len(pairs) > p["synergy_max_pairs"]:
        mine = {c["feature"]: c.get("oof_delta") or -np.inf for c in cells
                if c["target"] == tn and c["horizon"] == h and c["fold"] == fd["fold"]}
        top = sorted(adm, key=lambda f: (-mine.get(f, -np.inf), f))[: p["synergy_top_k"]]
        ts_ = set(top)
        dom = {f: domain_of(f, b.domains) for f in adm}
        keep = [pr for pr in pairs if (pr[0] in ts_ and pr[1] in ts_) or dom[pr[0]] == dom[pr[1]]]
        ks = set(keep)
        rest = sorted((pr for pr in pairs if pr not in ks),
                      key=lambda pr: _u(p["exploration_seed"], pr[0] + "|" + pr[1]))
        keep += rest[: max(0, p["synergy_max_pairs"] - len(keep))]
        pairs = keep
        rule = "top_k by fold OOF delta + same-domain pairs + hash-ordered fill to synergy_max_pairs"
    yf, ye = _yb_reg(y[fr]), _yb_reg(y[er])
    base = float(np.mean((ye - yf.mean()) ** 2))
    Z = {}
    for f in {f for pr in pairs for f in pr}:
        s = _Std(b.X[fr, col[f]], p["winsor_q"])
        Z[f] = (s(b.X[fr, col[f]]), s(b.X[er, col[f]]))
    single = {f: _lin_loss(Z[f][0][:, None], yf, Z[f][1][:, None], ye) for f in Z}
    out = []
    n_eval = 0
    shf = list(zip(_roll_shifts(len(fr), int(p["n_util_null"]), p),
                   _roll_shifts(len(er), int(p["n_util_null"]), p)))

    def inter_gain(af, cf, ae, ce):
        add = _lin_loss(np.column_stack([af, cf]), yf, np.column_stack([ae, ce]), ye)
        joint = _lin_loss(np.column_stack([af, cf, af * cf]), yf,
                          np.column_stack([ae, ce, ae * ce]), ye)
        return add, joint
    for a, c in pairs:
        n_eval += 1
        (af, ae), (cf, ce) = Z[a], Z[c]
        add, joint = inter_gain(af, cf, ae, ce)
        best_single = min(single[a], single[c])
        gain = best_single - joint
        igain = add - joint
        if not (gain > p["synergy_min_rel_gain"] * base and igain > p["synergy_min_rel_gain"] * base):
            continue
        null = []
        for s1, s2 in shf:  # break the joint alignment, keep each marginal and its autocorrelation
            a2, j2 = inter_gain(af, np.roll(cf, s1), ae, np.roll(ce, s2))
            null.append(a2 - j2)
        pn = emp_p_upper(igain, np.array(null))
        if pn is not None and pn <= p["synergy_null_p"]:
            out.append({"a": a, "b": c, "target": tn, "horizon": h, "fold": fd["fold"],
                        "mse_base": base, "mse_best_single": best_single, "mse_additive": add,
                        "mse_joint": joint, "joint_gain_over_best_single": gain,
                        "interaction_gain": igain, "interaction_null_p": pn,
                        "n_null": len(null), "pairs_evaluated_in_cell": len(pairs),
                        "pair_rule": rule})
    return out


def _decide(b, names, adm, tech, cells, groups_rows, syn_rows, folds, targets, p):
    maj = int(p["fold_majority"])
    by = {}
    for c in cells:
        by.setdefault((c["feature"], c["target"], c["horizon"]), []).append(c)
    syn = {}
    for s in syn_rows:
        for f, o in ((s["a"], s["b"]), (s["b"], s["a"])):
            syn.setdefault((f, s["target"], s["horizon"]), {}).setdefault(o, []).append(s["fold"])
    grp = {}
    for g in groups_rows:
        if g["group_delta"] <= 0:
            continue
        for f, d in g["drop_one"].items():
            if d > 0:
                grp.setdefault((f, g["target"], g["horizon"], g["kind"], g["group"]), set()).add(
                    g["fold"])
    last = folds[-1]["fold"]
    out = []
    for (tn, h) in sorted(targets):
        prelim = {}
        for f in names:
            r = {"feature": f, "target": tn, "horizon": h, "domain": domain_of(f, b.domains),
                 "evidence_scope": "inner_train_folds", "causal_evidence_level": "NOT_EVALUATED",
                 "reversible": True}
            if f in tech:
                r.update(status="TECHNICAL_REJECT", reasons=[c for c, _ in tech[f]],
                         reason_detail=[d for _, d in tech[f]])
                out.append(r); continue
            cs = sorted(by.get((f, tn, h), []), key=lambda c: c["fold"])
            meas = [c for c in cs if c.get("cell_status") == "MEASURED"]
            deltas = [c["oof_delta"] for c in meas]
            r.update(folds_measured=len(meas), folds_total=len(folds),
                     oof_delta_by_fold={c["fold"]: c["oof_delta"] for c in meas},
                     oof_delta_median=_f(np.median(deltas)) if deltas else None,
                     oof_delta_pos_folds=int(sum(d > 0 for d in deltas)),
                     spearman_by_fold={c["fold"]: c["spearman"] for c in meas},
                     spearman_q_by_fold={c["fold"]: c["spearman_q"] for c in meas},
                     mi_q_by_fold={c["fold"]: c["mi_q"] for c in meas},
                     flags=[])
            if any(c.get("spearman") is not None and abs(c["spearman"]) >= p["leak_alarm_abs_spearman"]
                   for c in meas):
                r["flags"].append("F_LEAK_ALARM_ASSOCIATION_NOT_A_CERTIFICATE")
            reasons = []
            if not meas:
                reasons.append("LP_NOT_EVALUATED_INSUFFICIENT_SUPPORT")
            calib = [c for c in meas if c["oof_delta"] > 0 and c.get("oof_null_p") is not None
                     and c["oof_null_p"] <= p["util_null_p"]]
            r["oof_null_p_by_fold"] = {c["fold"]: c.get("oof_null_p") for c in meas}
            if deltas and r["oof_delta_pos_folds"] >= min(maj, len(meas)) and \
                    r["oof_delta_median"] > 0:
                if len(calib) >= min(maj, len(meas)):
                    reasons.append("S_OOF_UTILITY")
                else:
                    reasons.append("LP_OOF_GAIN_NOT_CALIBRATED")
            qs = [c for c in meas if (c["spearman_q"] is not None and c["spearman_q"] <= p["fdr_q"])
                  or (c["mi_q"] is not None and c["mi_q"] <= p["fdr_q"])]
            if meas and len(qs) >= min(maj, len(meas)):
                reasons.append("S_ASSOC_ROBUST")
            partners = {o: fl for o, fl in syn.get((f, tn, h), {}).items()
                        if len(fl) >= min(int(p["synergy_fold_min"]), len(folds))}
            if partners:
                reasons.append("S_SYNERGY_PAIR")
                r["synergy_partners"] = sorted(partners)
            gs = [k for k, fl in grp.items() if k[0] == f and k[1] == tn and k[2] == h
                  and len(fl) >= maj]
            if gs:
                reasons.append("S_GROUP_CONTRIBUTION")
                r["contributing_groups"] = sorted(f"{k[3]}:{k[4]}" for k in gs)
            surv = any(x.startswith("S_") for x in reasons)
            if not surv:
                if meas and "S_OOF_UTILITY" not in reasons and \
                        "LP_OOF_GAIN_NOT_CALIBRATED" not in reasons:
                    reasons.append("LP_NO_ROBUST_OOF_GAIN")
                if meas and "S_ASSOC_ROBUST" not in reasons:
                    reasons.append("LP_ASSOC_NOT_ROBUST")
            r["status"] = "PROVISIONAL_SURVIVOR" if surv else "PROVISIONAL_LOW_PRIORITY"
            r["reasons"] = reasons
            prelim[f] = r
            out.append(r)
        # conditional redundancy inside dependence clusters (last fold's clusters, all folds' deltas)
        cl_of = {}
        for gi, mem in enumerate(_last_clusters(groups_rows, last, tn, h)):
            for m in mem:
                cl_of[m] = (gi, mem)
        cond = {}
        for g in groups_rows:
            if g["kind"] != "dependence_cluster" or g["target"] != tn or g["horizon"] != h:
                continue
            for f, d in g["drop_one"].items():
                cond.setdefault(f, []).append(d)
        for f, r in prelim.items():
            if f not in cl_of:
                continue
            gi, mem = cl_of[f]
            r["dependence_cluster"] = mem
            r["conditional_delta_by_fold"] = cond.get(f, [])
            if r["status"] != "PROVISIONAL_SURVIVOR" or "S_SYNERGY_PAIR" in r["reasons"]:
                continue
            cd = cond.get(f, [])
            if len(cd) and sum(d <= 0 for d in cd) >= maj:
                better = [m for m in mem if m != f and m in prelim
                          and prelim[m]["status"] == "PROVISIONAL_SURVIVOR"
                          and (prelim[m]["oof_delta_median"] or -np.inf) >
                          (r["oof_delta_median"] or -np.inf)]
                if better:
                    r["status"] = "PROVISIONAL_LOW_PRIORITY"
                    r["reasons"] = r["reasons"] + ["LP_CONDITIONALLY_REDUNDANT"]
                    r["redundant_with"] = sorted(better)
        for r in prelim.values():
            if r["status"] == "PROVISIONAL_LOW_PRIORITY":
                r["reincorporation"] = REINCORPORATION
    return out


def _last_clusters(groups_rows, last, tn, h):
    return [g["members"] for g in groups_rows if g["kind"] == "dependence_cluster"
            and g["fold"] == last and g["target"] == tn and g["horizon"] == h]


# ----------------------------------------------------------------------------- writing

STATUS_COLS = ["feature", "target", "horizon", "status", "reasons", "domain", "exploration_sample",
               "folds_measured", "folds_total", "oof_delta_median", "oof_delta_pos_folds",
               "oof_delta_by_fold", "oof_null_p_by_fold", "spearman_by_fold", "spearman_q_by_fold", "mi_q_by_fold",
               "synergy_partners", "contributing_groups", "dependence_cluster", "redundant_with",
               "conditional_delta_by_fold", "flags", "evidence_scope", "causal_evidence_level",
               "reversible", "reason_detail"]
CELL_COLS = ["feature", "target", "horizon", "fold", "cell_status", "cell_reason", "n_fit",
             "n_eval", "spearman", "spearman_p", "spearman_q", "spearman_family_m",
             "spearman_null_abs_q95", "mi", "mi_null_mean", "mi_p", "mi_q", "mi_family_m",
             "n_null", "null_min_shift_rows", "loss_name", "loss_base", "loss_model", "oof_delta",
             "oof_skill", "oof_null_p", "n_util_null", "lin_mse_base", "lin_mse", "lin_delta", "zero_naive_mse"]


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, dict)):
        return _canon(v)
    return str(v)


def _write_csv(path, rows, cols):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r.get(c)) if c != "reasons" else ";".join(r.get(c, []))
                        for c in cols])


def write(res: dict, out_dir: str, ready: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    srt = sorted(res["status"], key=lambda r: (r["target"], r["horizon"], r["feature"]))
    _write_csv(os.path.join(out_dir, "ps2_status.csv"), srt, STATUS_COLS)
    cel = sorted(res["cells"], key=lambda r: (r["target"], r["horizon"], r["fold"], r["feature"]))
    _write_csv(os.path.join(out_dir, "ps2_fold_cells.csv"), cel, CELL_COLS)
    grp = sorted(res["groups"], key=lambda r: (r["kind"], r["group"], r["target"], r["horizon"],
                                                r["fold"]))
    _write_csv(os.path.join(out_dir, "ps2_groups.csv"), grp,
               ["kind", "group", "target", "horizon", "fold", "members", "mse_base",
                "mse_group", "group_delta", "drop_one"])
    syn = sorted(res["synergy"], key=lambda r: (r["target"], r["horizon"], r["fold"], r["a"], r["b"]))
    _write_csv(os.path.join(out_dir, "ps2_synergy.csv"), syn,
               ["a", "b", "target", "horizon", "fold", "mse_base", "mse_best_single",
                "mse_additive", "mse_joint", "joint_gain_over_best_single", "interaction_gain",
                "interaction_null_p", "n_null", "pairs_evaluated_in_cell", "pair_rule"])
    with open(os.path.join(out_dir, "ps2_exploration.json"), "w", encoding="utf-8") as fh:
        json.dump(res["exploration"], fh, indent=1, sort_keys=True)
    for fn in ("ps2_status.csv", "ps2_fold_cells.csv", "ps2_groups.csv", "ps2_synergy.csv",
               "ps2_exploration.json"):
        with open(os.path.join(out_dir, fn), "rb") as fh:
            files[fn] = _sha(fh.read())
    surv = {}
    for r in res["status"]:
        if r["status"] in ("PROVISIONAL_SURVIVOR", "EXPLORATION"):
            surv.setdefault(r["feature"], []).append(f"{r['target']}_h{r['horizon']}:{r['status']}")
    man = {k: res[k] for k in ("schema", "batch_id", "fit_scope", "train_rows", "train_first_time",
                               "train_last_time", "train_data_digest", "parameters",
                               "parameter_digest", "code_digest", "folds", "n_features",
                               "n_admissible", "control_all_admissible", "technical_rejects",
                               "fold_clusters", "multiplicity", "counts", "status_policy",
                               "reincorporation")}
    man["exploration_sample"] = res["exploration"]["sample"]
    man["extractor_worklist"] = {f: surv[f] for f in sorted(surv)}
    man["output_sha256"] = files
    body = _canon(man).encode()
    man_digest = _sha(body)
    with open(os.path.join(out_dir, "ps2_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "ps2_cost.json"), "w", encoding="utf-8") as fh:
        json.dump({"elapsed_seconds": res["elapsed_seconds_unhashed"]}, fh)
    if ready:
        tmp = os.path.join(out_dir, ".READY.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"schema": SCHEMA, "manifest_canonical_sha256": man_digest,
                       "batch_id": res["batch_id"]}, fh, sort_keys=True)
        os.replace(tmp, os.path.join(out_dir, "READY"))
    return {"manifest_canonical_sha256": man_digest, "files": files}


# ----------------------------------------------------------------------------- loading

def epoch_seconds(values) -> np.ndarray:
    """Resolution-independent epoch seconds (pandas 2/3 safe; no astype(int64)//1e9)."""
    import pandas as pd
    t = pd.to_datetime(pd.Series(values), utc=True)
    return ((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(
        dtype="int64")


def batch_from_frame(df, ts_col, price_col, features, train_end, domains=None, declared=None,
                     batch_id="batch") -> Batch:
    ts = epoch_seconds(df[ts_col])
    te = int(epoch_seconds([train_end])[0]) if not isinstance(train_end, (int, np.integer)) \
        else int(train_end)
    X = df[features].to_numpy(dtype="float64")
    X = np.where(np.isfinite(X), X, np.nan)
    ts, X, price = restrict_to_train(ts, X, df[price_col].to_numpy(dtype="float64"), te)
    return Batch(ts=ts, X=X, names=list(features), price=price, domains=dict(domains or {}),
                 declared=dict(declared or {}), batch_id=batch_id)


def main(argv=None) -> int:
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="CSV or parquet with timestamp, price, features")
    ap.add_argument("--ts-col", required=True)
    ap.add_argument("--price-col", required=True)
    ap.add_argument("--train-end", required=True, help="exclusive TRAIN boundary (timestamp)")
    ap.add_argument("--features-json", required=True,
                    help="JSON: {features:[...], domains:{f:g}, declared:{f:[code,detail]}}")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--batch-id", default="batch")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-ready", action="store_true")
    a = ap.parse_args(argv)
    with open(a.features_json) as fh:
        spec = json.load(fh)
    feats = spec["features"]
    cols = [a.ts_col, a.price_col] + feats
    df = pd.read_parquet(a.data, columns=cols) if a.data.endswith(".parquet") else \
        pd.read_csv(a.data, usecols=cols, float_precision="round_trip")
    b = batch_from_frame(df, a.ts_col, a.price_col, feats, a.train_end, spec.get("domains"),
                         {k: tuple(v) for k, v in (spec.get("declared") or {}).items()},
                         a.batch_id)
    params = json.load(open(a.params_json)) if a.params_json else None
    res = build(b, params, log=lambda s: print(s, flush=True))
    w = write(res, a.out_dir, ready=not a.no_ready)
    print(json.dumps({"counts": res["counts"], "manifest": w["manifest_canonical_sha256"],
                      "elapsed_seconds": res["elapsed_seconds_unhashed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
