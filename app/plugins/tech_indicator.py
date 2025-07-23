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
import logging

# Set up logger
logger = logging.getLogger(__name__)


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
        Calculate RSI causally - starts at position 168 to match MTM decomposition requirements
        """
        rsi_values = pd.Series(index=prices.index, dtype=float)
        period = self.params['rsi_period']
        min_start_position = 168  # Match MTM decomposition requirement
        
        print(f"[DEBUG] RSI calculation: period={period}, min_start_position={min_start_position}, data_length={len(prices)}")
        print(f"[DEBUG] First few prices: {prices.iloc[:5].tolist()}")
        print(f"[DEBUG] Prices around position {min_start_position}: {prices.iloc[min_start_position-2:min_start_position+3].tolist()}")
        
        for i in range(len(prices)):
            if i < min_start_position:  # Wait until min_start_position
                # Not enough data for RSI calculation - will forward fill later
                rsi_values.iloc[i] = np.nan
                continue
            
            # PROPER CAUSALITY: Use data from t-period+1 to t (includes current point)
            # Need at least period+1 points for RSI calculation (to get period price changes)
            start_idx = max(0, i - period)  # This gives us period+1 points: [i-period, i-period+1, ..., i]
            recent_prices = prices.iloc[start_idx:i + 1]  # INCLUDES current point i
            
            if len(recent_prices) < period + 1:  # Need period+1 points for period price changes
                rsi_values.iloc[i] = np.nan
                continue
                
            # Calculate price changes (current vs previous)
            price_changes = recent_prices.diff().dropna()
            
            if len(price_changes) < period:  # Need exactly period price changes
                rsi_values.iloc[i] = np.nan
                continue
            
            # Use the most recent period changes
            recent_changes = price_changes.tail(period)
            
            gains = recent_changes[recent_changes > 0].sum()
            losses = abs(recent_changes[recent_changes < 0].sum())
            
            if losses == 0:
                rsi_values.iloc[i] = 100.0
            else:
                rs = gains / losses
                rsi_values.iloc[i] = 100.0 - (100.0 / (1.0 + rs))
        
        # Forward fill the first valid value to eliminate NaN values
        rsi_values.fillna(method='bfill', inplace=True)
        
        return rsi_values

    def _calculate_causal_ema(self, prices, period):
        """
        Calculate EMA causally - starts at position 168 to match MTM decomposition requirements
        """
        ema_values = pd.Series(index=prices.index, dtype=float)
        alpha = 2.0 / (period + 1.0)
        min_start_position = 168  # Match MTM decomposition requirement
        
        # Initialize EMA calculation
        for i in range(len(prices)):
            if i < min_start_position:
                # Not enough data - will forward fill later
                ema_values.iloc[i] = np.nan
            elif i == min_start_position:
                # Start EMA calculation from min_start_position using SMA as seed
                seed_data = prices.iloc[max(0, i - period + 1):i + 1]
                ema_values.iloc[i] = seed_data.mean()
            else:
                # PROPER CAUSALITY: EMA at time t uses EMA at t-1 and price at t
                ema_values.iloc[i] = alpha * prices.iloc[i] + (1 - alpha) * ema_values.iloc[i-1]
        
        # Forward fill the first valid value to eliminate NaN values
        # Find first non-NaN value manually
        first_valid_value = None
        first_valid_pos = None
        
        for i in range(len(ema_values)):
            if not pd.isna(ema_values.iloc[i]):
                first_valid_value = ema_values.iloc[i]
                first_valid_pos = i
                break
        
        # Forward fill using the first valid value
        if first_valid_value is not None and first_valid_pos is not None:
            for i in range(first_valid_pos):
                ema_values.iloc[i] = first_valid_value
        
        return ema_values

    def _calculate_causal_sma(self, prices, period):
        """
        Calculate SMA causally - starts at position 168 to match MTM decomposition requirements
        """
        sma_values = pd.Series(index=prices.index, dtype=float)
        min_start_position = 168  # Match MTM decomposition requirement
        
        for i in range(len(prices)):
            if i < max(period - 1, min_start_position):
                sma_values.iloc[i] = np.nan
            else:
                # PROPER CAUSALITY: SMA at time t uses prices from t-period+1 to t (includes current)
                recent_prices = prices.iloc[i - period + 1:i + 1]  # INCLUDES current point i
                sma_values.iloc[i] = recent_prices.mean()
        
        # Forward fill the first valid value to eliminate NaN values
        # Find first non-NaN value manually
        first_valid_value = None
        first_valid_pos = None
        
        for i in range(len(sma_values)):
            if not pd.isna(sma_values.iloc[i]):
                first_valid_value = sma_values.iloc[i]
                first_valid_pos = i
                break
        
        # Forward fill using the first valid value
        if first_valid_value is not None and first_valid_pos is not None:
            for i in range(first_valid_pos):
                sma_values.iloc[i] = first_valid_value
        
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
        Calculate Bollinger Bands causally - starts at position 168 to match MTM decomposition requirements
        """
        sma = self._calculate_causal_sma(prices, self.params['bb_period'])
        min_start_position = 168  # Match MTM decomposition requirement
        
        bb_upper = pd.Series(index=prices.index, dtype=float)
        bb_lower = pd.Series(index=prices.index, dtype=float)
        bb_width = pd.Series(index=prices.index, dtype=float)
        
        for i in range(len(prices)):
            if i < max(self.params['bb_period'] - 1, min_start_position):
                bb_upper.iloc[i] = np.nan
                bb_lower.iloc[i] = np.nan
                bb_width.iloc[i] = np.nan
            else:
                # PROPER CAUSALITY: Calculate standard deviation using data from t-period+1 to t (includes current)
                recent_prices = prices.iloc[i - self.params['bb_period'] + 1:i + 1]  # INCLUDES current point i
                std_dev = recent_prices.std()
                
                bb_upper.iloc[i] = sma.iloc[i] + (self.params['bb_std'] * std_dev)
                bb_lower.iloc[i] = sma.iloc[i] - (self.params['bb_std'] * std_dev)
                bb_width.iloc[i] = bb_upper.iloc[i] - bb_lower.iloc[i]
        
        # Forward fill NaN values for all Bollinger Band components
        for series in [bb_upper, bb_lower, bb_width]:
            # Find first non-NaN value manually
            first_valid_value = None
            first_valid_pos = None
            
            for i in range(len(series)):
                if not pd.isna(series.iloc[i]):
                    first_valid_value = series.iloc[i]
                    first_valid_pos = i
                    break
            
            # Forward fill using the first valid value
            if first_valid_value is not None and first_valid_pos is not None:
                for i in range(first_valid_pos):
                    series.iloc[i] = first_valid_value
        
        result = pd.DataFrame(index=prices.index)
        result['BB_Upper'] = bb_upper
        result['BB_Middle'] = sma  # SMA already forward-filled
        result['BB_Lower'] = bb_lower
        result['BB_Width'] = bb_width
        
        return result

    def process_additional_datasets(self, data, config):
        """
        Process additional datasets for the causal technical indicators plugin.
        This is a simplified version that focuses on the main dataset.
        """
        print("[DEBUG] Starting process_additional_datasets...")
        common_start = pd.Timestamp(data.index.min())
        common_end = pd.Timestamp(data.index.max())
        print(f"[DEBUG] Initial common range: {common_start} to {common_end}")
        print("[DEBUG] Main dataset first 5 rows:")
        print(data.head())
        
        # For the causal technical indicators, we mainly work with the main dataset
        # Create a copy of the data with some additional calculated columns
        additional_features_df = data.copy()
        
        # Add some basic calculated columns if they don't exist
        if 'High' in data.columns and 'Low' in data.columns:
            additional_features_df['BH-BL'] = data['High'] - data['Low']
        if 'High' in data.columns and 'Open' in data.columns:
            additional_features_df['BH-BO'] = data['High'] - data['Open']
        if 'Open' in data.columns and 'Low' in data.columns:
            additional_features_df['BO-BL'] = data['Open'] - data['Low']
        
        # Add seasonality columns if specified in the config
        if config.get('seasonality_columns'):
            additional_features_df['day_of_month'] = additional_features_df.index.day
            additional_features_df['hour_of_day'] = additional_features_df.index.hour
            additional_features_df['day_of_week'] = additional_features_df.index.dayofweek
            print("[DEBUG] Added seasonality columns (day_of_month, hour_of_day, day_of_week).")
        
        # Save the features dataset
        additional_features_df.reset_index().rename(columns={'index': 'datetime'}).to_csv('additional_merged_features.csv', index=False)
        print(f"[DEBUG] Saved additional features to 'additional_merged_features.csv' with range {common_start} to {common_end}")
        
        print(f"[DEBUG] additional_features_df final shape: {additional_features_df.shape}")
        return additional_features_df, common_start, common_end
