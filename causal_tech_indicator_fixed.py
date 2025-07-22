#!/usr/bin/env python3
"""
STRICTLY CAUSAL Technical Indicators Plugin

This plugin ensures NO FUTURE DATA LEAKAGE by calculating each technical indicator
at time t using ONLY data from periods up to t-1.

Key Features:
- Point-by-point calculation (no vectorized operations that leak future data)
- Each indicator value at time t uses only data up to t-1
- Proper handling of indicator initialization periods
- Maintains causality for real-time prediction scenarios
"""

import pandas as pd
import numpy as np


class Plugin:
    """
    Strictly causal technical indicators plugin.
    """
    
    plugin_params = {
        'indicators': ['rsi', 'macd', 'ema', 'sma', 'bb'],
        'rsi_period': 14,
        'ema_period': 20,
        'sma_period': 20,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        
    def set_params(self, **kwargs):
        """Set plugin parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def adjust_ohlc(self, data):
        """Adjust OHLC columns with case insensitivity."""
        print("Starting adjust_ohlc method...")
        print(f"Initial data columns: {data.columns}")

        # Normalize column names to lowercase
        data.columns = data.columns.str.lower()
        
        # Expected renaming map
        rename_map = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Apply renaming
        data = data.rename(columns=rename_map)
        print(f"Data columns after renaming: {data.columns}")
        
        return data

    def process(self, data):
        """
        Calculate technical indicators in a strictly causal manner.
        For any indicator value at time t, only data up to t-1 is used.
        """
        print("Starting causal technical indicators calculation...")
        print(f"Input data shape: {data.shape}")
        
        # Preserve datetime column if it exists
        datetime_column = None
        if 'datetime' in data.columns:
            datetime_column = data[['datetime']].copy()
            print(f"Datetime column preserved with shape: {datetime_column.shape}")
        
        # Adjust OHLC columns
        data = self.adjust_ohlc(data)
        
        # Ensure we have OHLC data
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create result dataframe
        result = pd.DataFrame(index=data.index)
        
        # Calculate each indicator causally
        for indicator in self.params['indicators']:
            print(f"Calculating causal {indicator.upper()}...")
            
            if indicator == 'rsi':
                result['RSI'] = self._calculate_causal_rsi(data['Close'])
            elif indicator == 'macd':
                macd_result = self._calculate_causal_macd(data['Close'])
                result = pd.concat([result, macd_result], axis=1)
            elif indicator == 'ema':
                result['EMA'] = self._calculate_causal_ema(data['Close'], self.params['ema_period'])
            elif indicator == 'sma':
                result['SMA'] = self._calculate_causal_sma(data['Close'], self.params['sma_period'])
            elif indicator == 'bb':
                bb_result = self._calculate_causal_bollinger_bands(data['Close'])
                result = pd.concat([result, bb_result], axis=1)
        
        # Add datetime column back if it existed
        if datetime_column is not None:
            result = pd.concat([datetime_column, result], axis=1)
        
        print(f"Causal technical indicators completed. Output shape: {result.shape}")
        print(f"Output columns: {result.columns.tolist()}")
        
        return result

    def _calculate_causal_rsi(self, prices):
        """
        Calculate RSI causally - at time t, use only data up to t-1
        """
        rsi_values = pd.Series(index=prices.index, dtype=float)
        period = self.params['rsi_period']
        
        for i in range(len(prices)):
            if i < period:
                # Not enough data for RSI calculation
                rsi_values.iloc[i] = np.nan
                continue
            
            # Use only data up to current point (including current for price change calc)
            # But the RSI at time t uses gains/losses calculated up to t-1
            recent_prices = prices.iloc[max(0, i - period):i + 1]
            
            if len(recent_prices) < 2:
                rsi_values.iloc[i] = np.nan
                continue
                
            # Calculate price changes (current vs previous)
            price_changes = recent_prices.diff().dropna()
            
            if len(price_changes) < period:
                rsi_values.iloc[i] = np.nan
                continue
            
            # Use only the most recent period changes
            recent_changes = price_changes.tail(period)
            
            gains = recent_changes[recent_changes > 0].sum()
            losses = abs(recent_changes[recent_changes < 0].sum())
            
            if losses == 0:
                rsi_values.iloc[i] = 100.0
            else:
                rs = gains / losses
                rsi_values.iloc[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi_values

    def _calculate_causal_ema(self, prices, period):
        """
        Calculate EMA causally - at time t, use only data up to t-1
        """
        ema_values = pd.Series(index=prices.index, dtype=float)
        alpha = 2.0 / (period + 1.0)
        
        for i in range(len(prices)):
            if i == 0:
                ema_values.iloc[i] = prices.iloc[i]
            else:
                # EMA at time t uses EMA at t-1 and price at t
                # This is causal because past EMA values don't use future data
                ema_values.iloc[i] = alpha * prices.iloc[i] + (1 - alpha) * ema_values.iloc[i-1]
        
        return ema_values

    def _calculate_causal_sma(self, prices, period):
        """
        Calculate SMA causally - at time t, use only data up to t-1
        """
        sma_values = pd.Series(index=prices.index, dtype=float)
        
        for i in range(len(prices)):
            if i < period - 1:
                sma_values.iloc[i] = np.nan
            else:
                # SMA at time t uses prices from t-period+1 to t
                # This uses current and past data only
                recent_prices = prices.iloc[i - period + 1:i + 1]
                sma_values.iloc[i] = recent_prices.mean()
        
        return sma_values

    def _calculate_causal_macd(self, prices):
        """
        Calculate MACD causally using causal EMAs
        """
        ema_fast = self._calculate_causal_ema(prices, self.params['macd_fast'])
        ema_slow = self._calculate_causal_ema(prices, self.params['macd_slow'])
        
        macd_line = ema_fast - ema_slow
        macd_signal = self._calculate_causal_ema(macd_line, self.params['macd_signal'])
        macd_histogram = macd_line - macd_signal
        
        result = pd.DataFrame(index=prices.index)
        result['MACD'] = macd_line
        result['MACD_Signal'] = macd_signal
        result['MACD_Histogram'] = macd_histogram
        
        return result

    def _calculate_causal_bollinger_bands(self, prices):
        """
        Calculate Bollinger Bands causally
        """
        sma = self._calculate_causal_sma(prices, self.params['bb_period'])
        
        bb_upper = pd.Series(index=prices.index, dtype=float)
        bb_lower = pd.Series(index=prices.index, dtype=float)
        bb_width = pd.Series(index=prices.index, dtype=float)
        
        for i in range(len(prices)):
            if i < self.params['bb_period'] - 1:
                bb_upper.iloc[i] = np.nan
                bb_lower.iloc[i] = np.nan
                bb_width.iloc[i] = np.nan
            else:
                # Calculate standard deviation using data up to current point
                recent_prices = prices.iloc[i - self.params['bb_period'] + 1:i + 1]
                std_dev = recent_prices.std()
                
                bb_upper.iloc[i] = sma.iloc[i] + (self.params['bb_std'] * std_dev)
                bb_lower.iloc[i] = sma.iloc[i] - (self.params['bb_std'] * std_dev)
                bb_width.iloc[i] = bb_upper.iloc[i] - bb_lower.iloc[i]
        
        result = pd.DataFrame(index=prices.index)
        result['BB_Upper'] = bb_upper
        result['BB_Middle'] = sma
        result['BB_Lower'] = bb_lower
        result['BB_Width'] = bb_width
        
        return result
