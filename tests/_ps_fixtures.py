"""Shared synthetic fixtures for the FS behavioural tests (local mechanics only)."""
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("ps", Path(__file__).resolve().parents[1] / "tools/progressive_selection.py")
PS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PS)

HOUR = 3600


def synthetic_market(n=3000, seed=0, decoys=6):
    """Price driven by x1*x2 (useful only jointly) plus weakly informative decoys.

    Returns (X, names, price, ts_seconds). r[t+1] depends on features at t, so every
    feature is causally available before the return it helps predict.
    """
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    noise = rng.normal(size=n)
    r = np.zeros(n)
    r[1:] = 0.01 * x1[:-1] * x2[:-1] + 0.002 * noise[1:]
    cols = {"x1": x1, "x2": x2}
    for k in range(decoys):
        d = np.zeros(n)
        d[:-1] = 0.15 * r[1:] / r.std() + rng.normal(size=n - 1)   # weak individual signal
        cols[f"decoy{k}"] = d
    cols["const"] = np.full(n, 2.0)
    price = 100 * np.exp(np.cumsum(r))
    ts = np.arange(n, dtype=np.int64) * HOUR
    names = list(cols)
    return np.column_stack([cols[c] for c in names]), names, price, ts


def params(**over):
    p = {"declared_periods_rows": {"daily": 24, "weekly": 168}, "primary_period": "daily",
         "acf_lags": [1, 24, 48], "redundancy_threshold": 0.95, "top_q": 3, "explore_fraction": 0.1,
         "explore_min": 1, "max_synergy_pairs": 5, "pair_feature_cap": 64, "seed": 7}
    p.update(over)
    return p
