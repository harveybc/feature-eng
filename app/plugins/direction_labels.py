"""
Direction Label Generator Plugin for Feature Engineering.

Produces binary directional labels using ATR-based TP/SL path scanning,
matching the direction_ideal_oracle in prediction_provider:
  - direction_long_label:  1 if buy TP hit before buy SL within horizon
  - direction_short_label: 1 if sell TP hit before sell SL within horizon

TP/SL levels:
  buy_tp  = close + ATR * tp_mult + cost_dist
  buy_sl  = close - ATR * sl_mult
  sell_tp = close - ATR * tp_mult - cost_dist
  sell_sl = close + ATR * sl_mult

TP detected on CLOSE, SL detected on LOW (buy) / HIGH (sell).

Usage (as feature-eng plugin):
    python app/main.py --plugin direction_labels \\
        --input_file data/ohlc_hourly.csv \\
        --output_file data/direction_labeled_ohlc.csv

The output CSV has the original columns plus the two label columns.
"""

import numpy as np
import pandas as pd


class Plugin:
    """Generates directional binary labels via ATR-based TP/SL path scanning."""

    plugin_params = {
        # ATR parameters
        'atr_period': 14,
        'tp_mult': 2.0,        # TP distance = ATR * tp_mult
        'sl_mult': 1.0,        # SL distance = ATR * sl_mult
        # Cost parameters (in pipettes)
        'spread_pips': 15.0,
        'commission_per_lot': 7.0,
        'slippage_pips': 5.0,
        'pip_cost': 0.00001,
        # Scan horizon
        'prediction_horizon': 120,   # max bars to scan for TP/SL
        'friday_close_hour': 20,     # Friday close hour (UTC)
        # Column names
        'close_column': 'CLOSE',
        'high_column': 'HIGH',
        'low_column': 'LOW',
        'datetime_column': 'DATE_TIME',
        'ohlc_order': 'ohlc',
    }

    plugin_debug_vars = [
        'atr_period', 'tp_mult', 'sl_mult',
        'spread_pips', 'commission_per_lot', 'slippage_pips',
        'prediction_horizon',
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {v: self.params.get(v) for v in self.plugin_debug_vars}

    # ------------------------------------------------------------------
    # ATR computation (Wilder smoothing, matching oracle)
    # ------------------------------------------------------------------

    def _compute_atr(self, high, low, close):
        """Compute ATR using Wilder smoothing (EMA-based)."""
        n = len(close)
        period = int(self.params['atr_period'])
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr = np.full(n, np.nan)
        if period <= n:
            atr[period - 1] = np.mean(tr[:period])
            alpha = 1.0 / period
            for i in range(period, n):
                atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
        return atr

    # ------------------------------------------------------------------
    # TP/SL path scanning (matching oracle._scan_tp_sl)
    # ------------------------------------------------------------------

    def _scan_tp_sl(self, idx, tp_price, sl_price, horizon,
                    direction, high, low, close):
        """Scan future bars: does TP get hit before SL?

        Buy:  SL hit when LOW  <= sl_price,  TP hit when CLOSE >= tp_price
        Sell: SL hit when HIGH >= sl_price,  TP hit when CLOSE <= tp_price

        Returns 1.0 if TP hit first, 0.0 otherwise.
        """
        n = len(close)
        for i in range(idx + 1, min(idx + 1 + horizon, n)):
            if direction == "buy":
                if low[i] <= sl_price:
                    return 0.0
                if close[i] >= tp_price:
                    return 1.0
            else:  # sell
                if high[i] >= sl_price:
                    return 0.0
                if close[i] <= tp_price:
                    return 1.0
        return 0.0

    def _bars_to_friday_close(self, idx, dt_index):
        """Compute bars until next Friday close (matching oracle)."""
        friday_hour = self.params['friday_close_hour']
        n = len(dt_index)
        count = 0
        for i in range(idx + 1, n):
            dt = dt_index[i]
            count += 1
            if dt.weekday() == 4 and dt.hour >= friday_hour:
                return count
            if dt.weekday() == 0 and i > idx + 1:
                return count
        return max(count, self.params['prediction_horizon'])

    # ------------------------------------------------------------------
    # Main feature-eng plugin entry point
    # ------------------------------------------------------------------

    def process(self, data):
        """Generate direction labels using ATR-based TP/SL path scanning.

        Parameters
        ----------
        data : pd.DataFrame
            Must have OPEN, HIGH, LOW, CLOSE columns.

        Returns
        -------
        pd.DataFrame
            Label columns: direction_long_label, direction_short_label.
        """
        p = self.params
        pip = p['pip_cost']

        # Resolve column names case-insensitively
        col_map = {}
        for key in ['close_column', 'high_column', 'low_column']:
            target = p[key]
            for col in data.columns:
                if col.upper() == target.upper():
                    col_map[key] = col
                    break
            else:
                col_map[key] = target

        close = data[col_map['close_column']].values.astype(float)
        high = data[col_map['high_column']].values.astype(float)
        low = data[col_map['low_column']].values.astype(float)
        n = len(data)

        # Compute ATR
        atr = self._compute_atr(high, low, close)

        # Cost adjustment (matching oracle)
        cost_pips = p['spread_pips'] + p['slippage_pips']
        if p['commission_per_lot'] > 0:
            cost_pips += (p['commission_per_lot'] / (100000.0 * pip)) * 2
        cost_dist = cost_pips * pip

        tp_mult = p['tp_mult']
        sl_mult = p['sl_mult']

        # Get datetime index for friday logic
        if isinstance(data.index, pd.DatetimeIndex):
            dt_index = data.index
        elif p['datetime_column'] in data.columns:
            dt_index = pd.to_datetime(data[p['datetime_column']])
        else:
            dt_index = None

        direction_long = np.full(n, np.nan, dtype=np.float64)
        direction_short = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            atr_val = atr[i]
            tp_dist = atr_val * tp_mult
            sl_dist = atr_val * sl_mult
            current = close[i]

            # Compute horizon (bars to friday or fixed)
            if dt_index is not None:
                horizon = self._bars_to_friday_close(i, dt_index)
            else:
                horizon = p['prediction_horizon']

            # Ensure enough future data
            if i + 1 >= n:
                continue

            # Buy: TP = close + tp_dist + costs, SL = close - sl_dist
            buy_tp = current + tp_dist + cost_dist
            buy_sl = current - sl_dist
            direction_long[i] = self._scan_tp_sl(
                i, buy_tp, buy_sl, horizon, "buy", high, low, close)

            # Sell: TP = close - tp_dist - costs, SL = close + sl_dist
            sell_tp = current - tp_dist - cost_dist
            sell_sl = current + sl_dist
            direction_short[i] = self._scan_tp_sl(
                i, sell_tp, sell_sl, horizon, "sell", high, low, close)

        labels = pd.DataFrame({
            'direction_long_label': direction_long,
            'direction_short_label': direction_short,
        }, index=data.index)

        return labels

    # ------------------------------------------------------------------
    # Process additional datasets (for compatibility with FE pipeline)
    # ------------------------------------------------------------------

    def process_additional_datasets(self, data, config):
        """Process auxiliary datasets — same logic as process()."""
        labels = self.process(data)
        return labels, None, None
