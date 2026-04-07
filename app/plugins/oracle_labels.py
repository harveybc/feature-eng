"""
Oracle Label Generator Plugin for Feature Engineering.

Scans future OHLC bars to produce binary labels for supervised learning:
  - buy_entry_label:  1 if buy TP hit before buy SL within weekly horizon, else 0
  - sell_entry_label: 1 if sell TP hit before sell SL within weekly horizon, else 0
  - buy_exit_label:   1 if buy TP still reachable from bar i, else 0
  - sell_exit_label:  1 if sell TP still reachable from bar i, else 0
  - bars_to_friday:   number of bars until Friday 20:00 (horizon proxy)

These labels match the binary_ideal_oracle logic in prediction_provider,
so ML models trained on them are directly compatible with the HS↔PP API.

Usage (as feature-eng plugin):
    python app/main.py --plugin oracle_labels \\
        --input_file data/ohlc_hourly.csv \\
        --output_file data/labeled_ohlc.csv \\
        --tp_pips 131.3 --sl_pips 93.3 \\
        --spread_pips 30 --commission_per_lot 10 --slippage_pips 10

The output CSV has the original OHLC columns plus the label columns.
"""

import numpy as np
import pandas as pd


class Plugin:
    """Generates oracle binary labels by scanning future OHLC data."""

    plugin_params = {
        # TP / SL distances (in pipettes, matching strategy defaults)
        'tp_pips': 131.325,           # tp_multiplier(5.15) * profit_threshold(25.50)
        'sl_pips': 93.33,             # sl_multiplier(3.66) * profit_threshold(25.50)
        # Trading cost parameters (worst-case, matching strategy)
        'spread_pips': 30.0,          # 3.0 real pips
        'commission_per_lot': 10.0,   # $10 per standard lot
        'slippage_pips': 10.0,        # 1.0 real pip
        # Price units
        'pip_cost': 0.00001,
        # Weekly-scope scanning
        'friday_close_hour': 20,
        'prediction_horizon': 120,    # fallback if Friday calc fails
        # Column names
        'close_column': 'CLOSE',
        'high_column': 'HIGH',
        'low_column': 'LOW',
        'open_column': 'OPEN',
        'datetime_column': 'DATE_TIME',
        # OHLC order for compatibility with FE pipeline
        'ohlc_order': 'ohlc',
    }

    plugin_debug_vars = [
        'tp_pips', 'sl_pips', 'spread_pips', 'commission_per_lot',
        'slippage_pips', 'pip_cost', 'friday_close_hour',
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
    # Main feature-eng plugin entry point
    # ------------------------------------------------------------------

    def process(self, data):
        """Generate oracle labels for the full dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Must have OPEN, HIGH, LOW, CLOSE columns (case-insensitive).

        Returns
        -------
        pd.DataFrame
            Label columns: buy_entry_label, sell_entry_label,
            buy_exit_label, sell_exit_label, bars_to_friday.
        """
        p = self.params
        pip = p['pip_cost']
        tp_pips = float(p['tp_pips'])
        sl_pips = float(p['sl_pips'])

        # Compute cost buffer (same formula as oracle)
        spread = float(p['spread_pips'])
        commission = float(p['commission_per_lot'])
        slippage = float(p['slippage_pips'])
        commission_pips = commission / (100_000 * pip) if pip > 0 else 0
        cost_pips = spread + slippage + commission_pips * 2

        close_col = p['close_column']
        high_col = p['high_column']
        low_col = p['low_column']

        # Remap columns case-insensitively
        col_map = {}
        for col in data.columns:
            cu = col.upper()
            if cu == close_col.upper():
                col_map[close_col] = col
            elif cu == high_col.upper():
                col_map[high_col] = col
            elif cu == low_col.upper():
                col_map[low_col] = col
        close_col = col_map.get(close_col, close_col)
        high_col = col_map.get(high_col, high_col)
        low_col = col_map.get(low_col, low_col)

        closes = data[close_col].values.astype(float)
        highs = data[high_col].values.astype(float)
        lows = data[low_col].values.astype(float)
        n = len(data)

        # Pre-compute Friday-close horizons
        if hasattr(data.index, 'weekday'):
            dts = data.index
        elif p['datetime_column'] in data.columns:
            dts = pd.to_datetime(data[p['datetime_column']])
        else:
            dts = data.index

        horizons = self._compute_horizons(dts, n)

        # Allocate label arrays
        buy_entry = np.zeros(n, dtype=np.int8)
        sell_entry = np.zeros(n, dtype=np.int8)
        buy_exit = np.zeros(n, dtype=np.int8)
        sell_exit = np.zeros(n, dtype=np.int8)
        bars_to_fri = np.zeros(n, dtype=np.int32)

        for i in range(n):
            h = horizons[i]
            bars_to_fri[i] = h
            if h <= 0:
                continue

            price = closes[i]

            # -- Entry labels (TP widened by cost) --
            buy_tp = price + (tp_pips + cost_pips) * pip
            buy_sl = price - sl_pips * pip
            buy_entry[i] = self._scan(closes, highs, lows, i, buy_tp, buy_sl, h, 'buy')

            sell_tp = price - (tp_pips + cost_pips) * pip
            sell_sl = price + sl_pips * pip
            sell_entry[i] = self._scan(closes, highs, lows, i, sell_tp, sell_sl, h, 'sell')

            # -- Exit labels (NO extra cost widening — position already open) --
            buy_tp_exit = price + tp_pips * pip
            buy_sl_exit = price - sl_pips * pip
            buy_exit[i] = self._scan(closes, highs, lows, i, buy_tp_exit, buy_sl_exit, h, 'buy')

            sell_tp_exit = price - tp_pips * pip
            sell_sl_exit = price + sl_pips * pip
            sell_exit[i] = self._scan(closes, highs, lows, i, sell_tp_exit, sell_sl_exit, h, 'sell')

        labels = pd.DataFrame({
            'buy_entry_label': buy_entry,
            'sell_entry_label': sell_entry,
            'buy_exit_label': buy_exit,
            'sell_exit_label': sell_exit,
            'bars_to_friday': bars_to_fri,
        }, index=data.index)

        return labels

    # ------------------------------------------------------------------
    # Scanning logic (matches binary_ideal_oracle._scan_tp_sl)
    # ------------------------------------------------------------------

    @staticmethod
    def _scan(closes, highs, lows, idx, tp, sl, horizon, direction):
        """Scan future bars.  Return 1 if TP hit before SL, else 0."""
        n = len(closes)
        for step in range(1, horizon + 1):
            fi = idx + step
            if fi >= n:
                break
            bar_close = closes[fi]
            bar_high = highs[fi]
            bar_low = lows[fi]

            if direction == 'buy':
                if bar_low <= sl:
                    return 0
                if bar_close >= tp:
                    return 1
            else:  # sell
                if bar_high >= sl:
                    return 0
                if bar_close <= tp:
                    return 1
        return 0  # horizon exhausted without TP hit

    # ------------------------------------------------------------------
    # Horizon computation (matches oracle._bars_to_friday_close)
    # ------------------------------------------------------------------

    def _compute_horizons(self, dts, n):
        """Compute bars-to-Friday-close for each row."""
        friday_hour = int(self.params['friday_close_hour'])
        fallback = int(self.params['prediction_horizon'])

        horizons = np.full(n, fallback, dtype=np.int32)

        try:
            weekdays = np.array([d.weekday() for d in dts])
            hours = np.array([d.hour for d in dts])
        except Exception:
            return horizons

        for i in range(n):
            wd = weekdays[i]
            hr = hours[i]
            if wd == 4 and hr >= friday_hour:
                horizons[i] = 0
                continue
            # Count bars until next Friday at friday_hour
            bars = 0
            for j in range(i + 1, n):
                bars += 1
                if weekdays[j] == 4 and hours[j] >= friday_hour:
                    break
            horizons[i] = bars if bars > 0 else fallback

        return horizons

    # ------------------------------------------------------------------
    # Stub for additional datasets (not needed for labels)
    # ------------------------------------------------------------------

    def process_additional_datasets(self, data, config):
        """No additional datasets needed for oracle labels."""
        return pd.DataFrame(), None, None
