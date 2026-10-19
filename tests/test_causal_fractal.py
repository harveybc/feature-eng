"""Causality, warm-up, window-in-bars and replay tests for the causal_fractal plugin.

Run from the repository root: ``PYTHONPATH=. python -m pytest tests/test_causal_fractal.py -q``.
No lake file is needed; the replay test uses a synthetic series unless the
EURUSD candidate store built by tools/causal_fractal_eurusd/build_store.py is
present (then one real validation row is replayed as well).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.plugins.causal_fractal import (  # noqa: E402
    CANDIDATE_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    FEATURE_COLUMNS,
    MIN_WINDOW,
    Plugin,
    causal_fractal_frame,
    causal_fractal_row,
    rs_scales_for_window,
)

WINDOWS = (200, 96, 48)


def _prices(seed: int, n: int = 420, drift: float = 0.0, reversion: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, 0.01, n)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = drift + reversion * returns[i - 1] + shock[i]
    return 100.0 * np.exp(np.cumsum(returns))


# ----------------------------------------------------------------------------
# 1. Future perturbation: altering bars after i changes nothing at i.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("window", WINDOWS)
def test_future_perturbation_leaves_closed_rows_unchanged(window):
    prices = _prices(7, n=window + 220)
    original = causal_fractal_frame(prices, window)
    cut = prices.size - 25
    for kind in ("jump", "noise", "reverse"):
        mutated = prices.copy()
        if kind == "jump":
            mutated[cut:] *= 3.0
        elif kind == "noise":
            mutated[cut:] *= np.exp(np.random.default_rng(1).normal(0, 0.05, prices.size - cut))
        else:
            mutated[cut:] = mutated[cut:][::-1]
        changed = causal_fractal_frame(mutated, window)
        pd.testing.assert_frame_equal(original.iloc[:cut], changed.iloc[:cut])
        # and the rows AFTER the cut do move (the test has teeth)
        assert not original.iloc[cut:]["hurst_rs"].equals(changed.iloc[cut:]["hurst_rs"])


@pytest.mark.parametrize("window", WINDOWS)
def test_row_i_equals_the_window_computed_alone(window):
    """Row i of the full series == the last row of close[i-w+1 : i+1] alone."""
    prices = _prices(5, n=window + 150)
    full = causal_fractal_frame(prices, window)
    for i in (window - 1, window + 17, prices.size - 1):
        alone = causal_fractal_frame(prices[i - window + 1 : i + 1], window)
        np.testing.assert_allclose(full.iloc[i].to_numpy(), alone.iloc[-1].to_numpy(), rtol=0, atol=0, equal_nan=True)
        # one bar fewer on the left changes the value: the window is exactly w bars
        narrower = causal_fractal_frame(prices[i - window + 2 : i + 1], window - 1) if window - 1 >= MIN_WINDOW else None
        if narrower is not None:
            assert not np.allclose(full.iloc[i]["box_counting_dimension"], narrower.iloc[-1]["box_counting_dimension"]) or \
                   not np.allclose(full.iloc[i]["variogram_hurst"], narrower.iloc[-1]["variogram_hurst"])


# ----------------------------------------------------------------------------
# 2. Warm-up emptiness.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("window", WINDOWS)
def test_first_incomplete_window_is_empty_and_the_rest_is_finite(window):
    frame = causal_fractal_frame(_prices(3, n=window + 100), window)
    assert frame.iloc[: window - 1].isna().all().all()
    for col in CANDIDATE_COLUMNS:
        assert frame[col].iloc[window - 1 :].notna().all(), col


def test_window_below_minimum_is_refused():
    with pytest.raises(ValueError):
        causal_fractal_frame(_prices(1, n=100), MIN_WINDOW - 1)


# ----------------------------------------------------------------------------
# 3. Window-in-bars convention: gaps in the calendar do not change the value.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("window", WINDOWS)
def test_window_counts_observed_bars_not_calendar_time(window):
    prices = _prices(9, n=window + 60)
    dense = pd.date_range("2024-03-04", periods=prices.size, freq="h")
    # the same bars with a two-day hole in the middle of the index
    with_gap = dense.to_series().copy()
    with_gap.iloc[window // 2 :] += pd.Timedelta(days=2)
    data_dense = pd.DataFrame({"CLOSE": prices}, index=dense)
    data_gap = pd.DataFrame({"CLOSE": prices}, index=pd.DatetimeIndex(with_gap.values))
    plugin = Plugin()
    plugin.set_params(window_bars=window)
    out_dense = plugin.process(data_dense)
    out_gap = plugin.process(data_gap)
    np.testing.assert_array_equal(out_dense.to_numpy(), out_gap.to_numpy())
    assert list(out_dense.columns)[0] == f"hurst_rs_w{window}"
    assert (out_gap.index == data_gap.index).all()


def test_rs_scales_follow_the_window():
    assert list(rs_scales_for_window(200)) == [8, 16, 32, 64]
    assert list(rs_scales_for_window(96)) == [8, 16, 32, 64]
    assert list(rs_scales_for_window(48)) == [8, 16, 32]


def test_correlation_dimension_is_nan_below_sixty_bars():
    frame = causal_fractal_frame(_prices(2, n=120), 48)
    assert frame["correlation_dimension"].iloc[47:].isna().all()
    frame96 = causal_fractal_frame(_prices(2, n=160), 96)
    assert frame96["correlation_dimension"].iloc[95:].notna().all()


def test_persistent_returns_score_above_reverting_returns():
    persistent = causal_fractal_frame(_prices(11, n=500, reversion=0.6), 200)
    reverting = causal_fractal_frame(_prices(11, n=500, reversion=-0.6), 200)
    assert persistent["hurst_rs"].iloc[-1] > reverting["hurst_rs"].iloc[-1]


def test_plugin_column_selection():
    plugin = Plugin()
    plugin.set_params(window_bars=48, include_diagnostic=False, include_intermediate=False)
    out = plugin.process(pd.DataFrame({"CLOSE": _prices(4, n=100)}))
    assert list(out.columns) == [f"{c}_w48" for c in CANDIDATE_COLUMNS]
    assert not any(c.startswith(d) for c in out.columns for d in DIAGNOSTIC_COLUMNS)


# ----------------------------------------------------------------------------
# 4. Replay of Retsu's verification on one validation row: recompute the row
#    from its 200 closes by an independent, plain-python R/S and variogram.
# ----------------------------------------------------------------------------
def _independent_hurst_rs(closes: list[float]) -> float:
    import math
    r = [math.log(closes[k + 1]) - math.log(closes[k]) for k in range(len(closes) - 1)]
    xs, ys = [], []
    for scale in (8, 16, 32, 64):
        if scale > len(r):
            continue
        blocks = len(r) // scale
        ratios = []
        for b in range(blocks):
            seg = r[b * scale : (b + 1) * scale]
            mean = sum(seg) / scale
            cum, lo, hi = 0.0, 0.0, 0.0
            for v in seg:
                cum += v - mean
                lo, hi = min(lo, cum), max(hi, cum)
            # numpy profile min/max is over the cumulative sum points only (not 0)
            prof = []
            cum = 0.0
            for v in seg:
                cum += v - mean
                prof.append(cum)
            span = max(prof) - min(prof)
            var = sum((v - mean) ** 2 for v in seg) / (scale - 1)
            sd = math.sqrt(var)
            if sd > 0:
                ratios.append(span / sd)
        if ratios:
            xs.append(math.log(scale))
            ys.append(math.log(sum(ratios) / len(ratios)))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def _independent_variogram_hurst(closes: list[float]) -> float:
    import math
    xs, ys = [], []
    for tau in (1, 2, 4, 8, 16, 32):
        if tau >= len(closes):
            continue
        inc = [closes[k + tau] - closes[k] for k in range(len(closes) - tau)]
        gamma = 0.5 * sum(v * v for v in inc) / len(inc)
        if gamma > 0:
            xs.append(math.log(tau))
            ys.append(math.log(gamma))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return 0.5 * sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def _replay(closes: np.ndarray, window: int, stored: pd.Series | None = None):
    row = causal_fractal_row(closes[-window:])
    got = dict(zip(FEATURE_COLUMNS, row))
    assert abs(got["hurst_rs"] - _independent_hurst_rs(list(closes[-window:]))) < 1e-9
    assert abs(got["variogram_hurst"] - _independent_variogram_hurst(list(closes[-window:]))) < 1e-9
    assert abs(got["fractal_dimension_graph"] - (2.0 - got["hurst_rs"])) < 1e-12
    if stored is not None:
        for col in FEATURE_COLUMNS:
            s = float(stored[f"{col}_w{window}"])
            if np.isnan(s):
                assert np.isnan(got[col]), col
            else:
                assert abs(s - got[col]) < 1e-12, (col, s, got[col])


def test_replay_one_row_by_independent_arithmetic():
    prices = _prices(21, n=400)
    _replay(prices[: 333], 200)
    _replay(prices[: 333], 96)
    _replay(prices[: 333], 48)


STORE = Path(os.environ.get("CAUSAL_FRACTAL_STORE", "/var/tmp/features-candidates/eurusd_h1_causal_fractal"))


@pytest.mark.skipif(not (STORE / "MANIFEST.json").exists(), reason="candidate store not built on this host")
@pytest.mark.parametrize("window", WINDOWS)
def test_replay_one_real_validation_row_from_the_store(window):
    """Retsu-style replay: one 2024 row recomputed from its own trailing closes only."""
    manifest = json.loads((STORE / "MANIFEST.json").read_text())
    val = pd.read_parquet(STORE / manifest["windows"][str(window)]["files"]["validation"])
    train = pd.read_parquet(STORE / manifest["windows"][str(window)]["files"]["train"])
    closes = pd.concat([train, val])[["date_time", "close"]].sort_values("date_time")
    assert closes["date_time"].max() < pd.Timestamp("2025-01-01", tz="UTC")
    pick = val.iloc[len(val) // 2]
    pos = int(np.flatnonzero(closes["date_time"].to_numpy() == pick["date_time"])[0])
    history = closes["close"].to_numpy()[: pos + 1]
    _replay(history, window, stored=pick)
