#!/usr/bin/env python3
"""KALMAN_CANDIDATE_FEATURES: per input, persistence diagnostics on TRAIN rows only and a declared state-space form.

Rule (declared before measuring, persistence-based, not smoothness-based):
  acf1(x) = lag-1 autocorrelation of the level; acf1(dx) = lag-1 autocorrelation of its first difference.
  - STATIONARY_NOT_A_LEVEL_STATE   if acf1(x) < 0.99         (use as an observed input / AR noise, not a latent level)
  - LEVEL_PLUS_SLOPE               if acf1(x) >= 0.99 and acf1(dx) >= 0.5   (differences are themselves persistent: I(2)-like drift)
  - LOCAL_LEVEL                    if acf1(x) >= 0.99 and acf1(dx) < 0.5    (random-walk-like level plus noise)
  - INSUFFICIENT                   if fewer than 50 finite contiguous rows
No row at or after the TRAIN boundary is read."""
import argparse, json, hashlib
import numpy as np, pandas as pd


def acf1(v):
    v = v[np.isfinite(v)]
    if len(v) < 50 or np.ptp(v) == 0:
        return None
    c = v - v.mean()
    return float((c[1:] @ c[:-1]) / (c @ c))


def classify(x):
    x = np.asarray(x, dtype=float)
    a, ad = acf1(x), acf1(np.diff(x))
    if a is None or ad is None:
        return {"form": "INSUFFICIENT", "acf1_level": a, "acf1_diff": ad}
    form = "STATIONARY_NOT_A_LEVEL_STATE" if a < 0.99 else ("LEVEL_PLUS_SLOPE" if ad >= 0.5 else "LOCAL_LEVEL")
    return {"form": form, "acf1_level": a, "acf1_diff": ad}


def build(csv_path, n_train, features, step_seconds, dataset, manifest_sha, availability):
    df = pd.read_csv(csv_path, nrows=n_train)
    rows = []
    for f in features:
        x = pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        c = classify(x)
        fin = x[np.isfinite(x)]
        rows.append({"feature": f, **c, "scale": {"median": float(np.median(fin)), "iqr": float(np.subtract(*np.percentile(fin, [75, 25]))),
                                                  "std": float(fin.std(ddof=1)), "diff_std": float(np.nanstd(np.diff(x), ddof=1))},
                     "finite_rows": int(len(fin)), "sampling_step_seconds": step_seconds, "availability": availability})
    return {"dataset": dataset, "manifest_canonical": manifest_sha, "train_rows": n_train, "features": rows,
            "counts": {k: sum(r["form"] == k for r in rows) for k in ("LOCAL_LEVEL", "LEVEL_PLUS_SLOPE", "STATIONARY_NOT_A_LEVEL_STATE", "INSUFFICIENT")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    doc = {"schema": "lane_b_kalman_candidate_features.v1", "rule": classify.__module__ and __doc__.split("Rule")[1].split("No row")[0].strip(),
           "datasets": [build(**d) for d in spec]}
    json.dump(doc, open(a.out, "w"), indent=1)
    print(json.dumps([d["counts"] for d in doc["datasets"]]))


if __name__ == "__main__":
    main()
