"""B2 first versioned selection table for the 83 ETH 4h input features (TRAIN rows only).

Every feature receives exactly one status, SURVIVOR / REJECT / PENDING, with reason codes, in
three declared stages:

1. availability / semantic screen (may REJECT or hold PENDING)
   - R_UNAVAILABLE: no finite train value.
   - R_CONSTANT: one distinct finite train value (or zero variance).
   - R_MISSING: missing_fraction > ``max_missing_fraction``.
   - R_LEADING_MISSING: leading_missing > ``max_leading_missing_fraction`` of the train rows.
   - R_LOOKAHEAD_NAME: the name carries a forward-looking token (``LOOKAHEAD_TOKENS``).
   - R_ORIGIN_DEPENDENT_CUMSUM: the value is a cumulative sum from the first row of the file, so
     its level depends on where the history starts (train/serve skew by construction).
   - P_WARMUP_ENCODED_AS_ZERO: undefined warm-up rows are encoded as a real 0.0 (repairable by
     masking the warm-up; held PENDING, not rejected).
   - flags that never change the status: F_WARMUP_<n> (leading NaN within the allowance),
     F_PRICE_SCALE_LEVEL (value is in price or volume units, non-stationary level).
2. cheap-metric redundancy among the features that pass stage 1:
   train-only pairwise |Spearman| (pairwise-complete rows), complete-linkage clustering cut at
   distance ``1 - redundancy_threshold``, so every member of a cluster has |rho| >= threshold with
   every other member. One representative per cluster by an association-free rule: most finite
   train rows, then fewest leading missing, then the earliest position in the manifest order.
   Others become PENDING with P_REDUNDANT and a pointer to the representative; nothing is dropped.
3. target / horizon association: the train-only Pearson/Spearman with log(close[t+h]/close[t]) at
   h = 6..36 is copied from the metric table as COLUMNS ONLY. It is never read by any status rule;
   ``status_inputs`` lists every field the rules read and the tests prove association alone never
   rejects.

The all-83 set is kept as the control list. Isolation: the train frame is loaded by
``train_feature_metrics.load_train_frame`` (dates read until the train boundary, then ``nrows``),
and the recomputed train data digest must equal the metric table's, or the run is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from app import train_feature_metrics as tfm

SCHEMA = "feature_eng.b2_selection_table.v1"
TABLE_VERSION = 1
DEFAULT_PARAMS = {
    "max_missing_fraction": 0.5,
    "max_leading_missing_fraction": 0.25,
    "redundancy_threshold": 0.95,
    "redundancy_method": "spearman_abs_pairwise_complete",
    "linkage": "complete",
    "representative_rule": "max n_finite, then min leading_missing, then manifest order",
    "min_pairs": 30,
    "association_horizons": [6, 12, 18, 24, 30, 36],
}
LOOKAHEAD_TOKENS = ("future", "fwd", "forward", "lead_", "next_", "target", "label", "centered")
# Semantic facts from the producer definitions (financial-data stage22 trading-features worker and
# its semantic census): every remaining feature is a trailing window/difference of past rows.
ORIGIN_DEPENDENT_CUMSUM = ("obv",)
WARMUP_ENCODED_AS_ZERO = ("vol_regime_high", "vol_regime_low")
PRICE_SCALE_LEVEL = ("sma_10", "ema_10", "sma_20", "ema_20", "sma_50", "ema_50", "sma_100",
                     "ema_100", "sma_200", "ema_200", "bb_upper", "bb_middle", "bb_lower",
                     "vwap_60", "atr_14", "macd", "macd_signal", "macd_hist", "mom_10", "mom_20",
                     "obv_delta_20", "volume_sma_10", "volume_sma_20")
STATUS_INPUTS = ["n_finite", "n_rows", "n_unique", "variance", "is_constant", "missing_fraction",
                 "leading_missing", "feature name (semantic lists)",
                 "train-only feature-feature |Spearman|"]


class B2Error(ValueError):
    pass


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha(path: str) -> str:
    with open(path, "rb") as fh:
        return _sha(fh.read())


# ----------------------------------------------------------------------------- stage 1

def screen(row: dict, p: dict) -> tuple[str | None, list, list]:
    """Return (status or None if it passes, reasons, flags) from availability/semantics only."""
    name, n = row["feature"], row["n_rows"]
    reasons, flags = [], []
    if not row["n_finite"]:
        reasons.append("R_UNAVAILABLE")
    if row["is_constant"] or row["n_unique"] <= 1 or (row["variance"] is not None
                                                      and row["variance"] == 0):
        reasons.append("R_CONSTANT")
    mf = row["missing_fraction"]
    if mf is not None and mf > p["max_missing_fraction"]:
        reasons.append("R_MISSING")
    if n and row["leading_missing"] > p["max_leading_missing_fraction"] * n:
        reasons.append("R_LEADING_MISSING")
    elif row["leading_missing"] > 0:
        flags.append(f"F_WARMUP_{row['leading_missing']}")
    low = name.lower()
    if any(t in low for t in LOOKAHEAD_TOKENS):
        reasons.append("R_LOOKAHEAD_NAME")
    if name in ORIGIN_DEPENDENT_CUMSUM:
        reasons.append("R_ORIGIN_DEPENDENT_CUMSUM")
    if name in PRICE_SCALE_LEVEL:
        flags.append("F_PRICE_SCALE_LEVEL")
    if reasons:
        return "REJECT", reasons, flags
    if name in WARMUP_ENCODED_AS_ZERO:
        return "PENDING", ["P_WARMUP_ENCODED_AS_ZERO"], flags
    return None, [], flags


# ----------------------------------------------------------------------------- stage 2

def abs_spearman(train_df: pd.DataFrame, feats: list, min_pairs: int) -> pd.DataFrame:
    if not feats:
        return pd.DataFrame()
    c = train_df[feats].astype("float64").replace([np.inf, -np.inf], np.nan)
    return c.corr(method="spearman", min_periods=min_pairs).abs()


def clusters(rho: pd.DataFrame, threshold: float) -> list:
    """Complete-linkage clusters on distance 1-|rho|; unknown pairs count as distance 1."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    names = list(rho.columns)
    if len(names) < 2:
        return [names] if names else []
    d = 1.0 - rho.fillna(0.0).to_numpy()
    d = np.clip((d + d.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(d, 0.0)
    lab = fcluster(linkage(squareform(d, checks=False), method="complete"),
                   t=1.0 - threshold + 1e-12, criterion="distance")
    groups: dict = {}
    for name, l in zip(names, lab):
        groups.setdefault(int(l), []).append(name)
    return list(groups.values())


def representative(members: list, rows: dict, order: dict) -> str:
    return sorted(members, key=lambda f: (-rows[f]["n_finite"], rows[f]["leading_missing"],
                                          order[f]))[0]


# ----------------------------------------------------------------------------- table

def build(metric_table: dict, train_df: pd.DataFrame, features: list,
          params: dict | None = None, input_digests: dict | None = None) -> dict:
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    rows = {r["feature"]: r for r in metric_table["rows"]}
    if sorted(rows) != sorted(features):
        raise B2Error("metric table features differ from the manifest features")
    target = metric_table["parameters"]["target_column"]
    if len(train_df) != metric_table["fit_rows"]["stop"] - metric_table["fit_rows"]["start"]:
        raise B2Error("train frame row count differs from the metric table fit rows")
    dd = tfm.data_digest(train_df[features].to_numpy(dtype="float64"),
                         train_df[target].to_numpy(dtype="float64"), features, target)
    if dd != metric_table["data_digest"]:
        raise B2Error("train data digest does not match the metric table; refusing")
    order = {f: i for i, f in enumerate(features)}
    out = {}
    passing = []
    for f in features:
        st, reasons, flags = screen(rows[f], p)
        out[f] = {"status": st, "reasons": reasons, "flags": flags,
                  "cluster_id": None, "cluster_size": None, "representative": None,
                  "rho_to_representative": None}
        if st is None:
            passing.append(f)
    rho = abs_spearman(train_df, passing, p["min_pairs"])
    cl = sorted(clusters(rho, p["redundancy_threshold"]), key=lambda m: min(order[x] for x in m))
    for cid, members in enumerate(cl):
        rep = representative(members, rows, order)
        for f in members:
            o = out[f]
            o.update(cluster_id=cid, cluster_size=len(members), representative=rep)
            if f == rep:
                o["status"] = "SURVIVOR"
                o["reasons"] = ["S_PASSED_SCREEN", "S_CLUSTER_REPRESENTATIVE" if len(members) > 1
                                else "S_SINGLETON"]
                o["rho_to_representative"] = 1.0
            else:
                r = rho.loc[f, rep]
                o["status"] = "PENDING"
                o["reasons"] = ["P_REDUNDANT"]
                o["rho_to_representative"] = None if pd.isna(r) else round(float(r), 12)
    table = []
    for f in features:
        r, o = rows[f], out[f]
        rec = {"feature": f, "manifest_index": order[f], "status": o["status"],
               "reasons": o["reasons"], "flags": o["flags"], "cluster_id": o["cluster_id"],
               "cluster_size": o["cluster_size"], "representative": o["representative"],
               "rho_to_representative": o["rho_to_representative"],
               "n_finite": r["n_finite"], "missing_fraction": r["missing_fraction"],
               "leading_missing": r["leading_missing"], "n_unique": r["n_unique"]}
        best = None
        for h in p["association_horizons"]:  # columns only; no rule reads them
            sp = r.get(f"target_spearman_h{h}")
            rec[f"assoc_spearman_h{h}"] = sp
            rec[f"assoc_pearson_h{h}"] = r.get(f"target_pearson_h{h}")
            if sp is not None and (best is None or abs(sp) > best):
                best = abs(sp)
        rec["assoc_max_abs_spearman"] = best
        table.append(rec)
    counts = {s: sum(1 for t in table if t["status"] == s) for s in ("SURVIVOR", "REJECT", "PENDING")}
    reason_counts: dict = {}
    for t in table:
        for c in t["reasons"]:
            reason_counts[c] = reason_counts.get(c, 0) + 1
    p_digest = _sha(_canon(p).encode())
    return {"schema": SCHEMA, "table_version": TABLE_VERSION, "fit_scope": "train_only",
            "fit_rows": metric_table["fit_rows"],
            "protected_test_rows": list(tfm.PROTECTED_TEST_ROWS),
            "parameters": p, "parameter_digest": p_digest,
            "lookahead_tokens": list(LOOKAHEAD_TOKENS),
            "semantic_lists": {"origin_dependent_cumsum": list(ORIGIN_DEPENDENT_CUMSUM),
                               "warmup_encoded_as_zero": list(WARMUP_ENCODED_AS_ZERO),
                               "price_scale_level": list(PRICE_SCALE_LEVEL)},
            "status_inputs": STATUS_INPUTS,
            "association_policy": "reported as columns only; never read by a status rule",
            "inputs": dict(input_digests or {}, train_data_digest=dd,
                           metric_table_parameter_digest=metric_table["parameter_digest"],
                           metric_table_schema=metric_table["schema"]),
            "n_features": len(features), "counts": counts,
            "reason_counts": dict(sorted(reason_counts.items())),
            "control_list_all": list(features),
            "survivors": [t["feature"] for t in table if t["status"] == "SURVIVOR"],
            "clusters": [{"cluster_id": i, "representative": representative(m, rows, order),
                          "members": sorted(m, key=order.get)} for i, m in enumerate(cl)],
            "rows": table}


def run(metric_json: str, csv_path: str, manifest_path: str, out_dir: str,
        params: dict | None = None, stem: str = "B2_SELECTION_TABLE_v1") -> dict:
    with open(metric_json, "r", encoding="utf-8") as fh:
        mt = json.load(fh)
    contract = tfm.load_contract(manifest_path)
    mp = dict(tfm.DEFAULT_PARAMS)
    mp.update(mt["parameters"])
    df = tfm.load_train_frame(csv_path, contract, mp)
    digests = {"metric_table_sha256": _file_sha(metric_json),
               "manifest_file_sha256": _file_sha(manifest_path),
               "manifest_declared_data_sha256": contract["manifest_sha256"],
               "train_end": contract["train_end"]}
    res = build(mt, df, contract["features"], params, digests)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    flat = []
    for t in res["rows"]:
        r = dict(t)
        r["reasons"] = ";".join(t["reasons"])
        r["flags"] = ";".join(t["flags"])
        r["schema"] = SCHEMA
        r["parameter_digest"] = res["parameter_digest"]
        r["train_data_digest"] = res["inputs"]["train_data_digest"]
        flat.append(r)
    pd.DataFrame(flat).to_csv(os.path.join(out_dir, stem + ".csv"), index=False)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric-json", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    res = run(a.metric_json, a.csv, a.manifest, a.out_dir)
    print(json.dumps({"counts": res["counts"], "reason_counts": res["reason_counts"],
                      "parameter_digest": res["parameter_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
