"""The eleven indicators this application really computes, against a changed future.

R2 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "run actual feature-eng indicators... Record lookahead, warm-up, output availability and
     response characteristics separately."

`tech_indicator` is the plugin the smoke configuration selects, and its declared set is
rsi, macd, ema, stoch, adx, atr, cci, bbands, williams, momentum and roc — pandas_ta, not a
reimplementation. `Plugin.process` is called here exactly as `process_data` calls it.

Two things are measured and reported separately, because they are different questions:

* **lookahead** — does a row's value change when rows AFTER it change? (it must not);
* **warm-up** — how many leading rows have no value at all, per indicator (a fact to record,
  not a defect).

Measured on 400 rows, all fifteen produced columns, none of which reads the future:

    MACD_Histogram 33   MACD_Signal 33   ADX 27   MACD 25   Stochastic_%D 17
    Stochastic_%K  15   RSI 14   DI+ 14   DI- 14   ATR 14   CCI 13
    WilliamsR      13   Momentum 10   ROC 10   EMA 9

So a run that trims fewer than 33 rows feeds a model values that do not exist yet for some of
its inputs. That is the number this file exists to make visible.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("pandas_ta")

from app.plugins.tech_indicator import Plugin

ROWS = 400
FUTURE_FROM = 300


def ohlc(rows=ROWS, tail_shift=0.0, hole=None):
    """A deterministic OHLC trajectory; the perturbation only ever touches row 300 onwards."""
    state = 20260915
    close, level = [], 100.0
    for index in range(rows):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        level += ((state % 1000) / 1000.0 - 0.5)
        value = level + (tail_shift if index >= FUTURE_FROM else 0.0)
        if hole is not None and index >= hole:
            value = float("nan")
        close.append(value)
    close = pd.Series(close)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=rows, freq="h"),
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
    })


def indicators(frame):
    """The real plugin, with its real defaults."""
    plugin = Plugin()
    return plugin.process(frame.copy())


@pytest.fixture(scope="module")
def base():
    return indicators(ohlc())


@pytest.fixture(scope="module")
def moved():
    return indicators(ohlc(tail_shift=40.0))


def test_the_declared_indicators_are_the_ones_actually_produced(base):
    """What the configuration asks for and what comes back, compared by name."""
    assert not base.empty
    produced = set(base.columns)
    for expected in ("RSI", "MACD", "ATR", "CCI", "ADX"):
        assert any(column.upper().startswith(expected) for column in produced), (
            f"{expected} is declared in plugin_params but no column carries it: {sorted(produced)}")


def test_no_indicator_reads_the_future(base, moved):
    """Every column, row by row, over the admissible prefix. This is the whole point.

    A single two-sided indicator — a centred average, a backward fill — would move values
    before row 300 when only rows from 300 onwards changed.
    """
    assert list(base.columns) == list(moved.columns)
    offenders = []
    for column in base.columns:
        if not np.issubdtype(base[column].dtype, np.number):
            continue
        left = base[column].iloc[:FUTURE_FROM]
        right = moved[column].iloc[:FUTURE_FROM]
        if not left.equals(right):
            first = int(np.argmax((left.values != right.values) & ~(
                pd.isna(left.values) & pd.isna(right.values))))
            offenders.append(f"{column} (first divergence at row {first})")
    assert not offenders, (
        "these indicators changed in the past when only the future changed: "
        + ", ".join(offenders))


def test_missingness_in_the_future_does_not_reach_back_either(base):
    """The other perturbation: stop the series dead at row 320."""
    gapped = indicators(ohlc(hole=FUTURE_FROM + 20))
    offenders = [column for column in base.columns
                 if np.issubdtype(base[column].dtype, np.number)
                 and not base[column].iloc[:FUTURE_FROM].equals(
                     gapped[column].iloc[:FUTURE_FROM])]
    assert not offenders, f"missing future rows moved past values of: {offenders}"


def test_each_indicator_declares_its_warm_up(base):
    """Recorded per indicator, because they differ and downstream code must know.

    This is a measurement, not a pass/fail on quality: an indicator with a 200-row warm-up is
    not broken, but a run that trims 20 rows and feeds it to a model is.
    """
    warm_up = {}
    for column in base.columns:
        if not np.issubdtype(base[column].dtype, np.number):
            continue
        values = base[column]
        first_valid = values.first_valid_index()
        warm_up[column] = 0 if first_valid is None else int(first_valid)
    assert warm_up, "no numeric indicator column was produced"
    assert max(warm_up.values()) < ROWS, (
        f"an indicator never produces a value on {ROWS} rows: {warm_up}")
    # the longest warm-up belongs to a long-period indicator, and it is worth naming
    longest = max(warm_up, key=warm_up.get)
    assert warm_up[longest] >= 30, (
        f"expected the MACD family to carry the longest warm-up; longest is {longest} at "
        f"{warm_up[longest]} rows")
    assert len(warm_up) == 15, f"fifteen indicator columns are expected, got {len(warm_up)}"


def test_an_indicator_computed_two_sided_would_be_caught(base, monkeypatch):
    """Check on the check: force pandas_ta's RSI to be centred and re-run the real plugin."""
    import pandas_ta as ta

    original = ta.rsi

    def centred(close, *args, **kwargs):
        series = original(close, *args, **kwargs)
        return series.rolling(5, center=True, min_periods=1).mean()

    monkeypatch.setattr(ta, "rsi", centred)
    mutated_base = indicators(ohlc())
    mutated_moved = indicators(ohlc(tail_shift=40.0))
    column = "RSI"
    assert column in mutated_base.columns
    assert not mutated_base[column].iloc[:FUTURE_FROM].equals(
        mutated_moved[column].iloc[:FUTURE_FROM]), (
        "a centred RSI must be detected by the very comparison the passing rule makes")
