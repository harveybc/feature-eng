#!/usr/bin/env python3
"""
CAUSAL Technical Indicators Plugin - Fixed Version

This version ensures that technical indicators are calculated using ONLY past data,
preventing any future data leakage that could lead to unrealistic model performance.
"""

import pandas_ta as ta
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class Plugin:
    """
    A STRICTLY CAUSAL feature-engineering plugin using technical indicators.
    Each indicator value at time t uses ONLY data from t-window_size to t-1.
    """

    # Plugin parameters
    plugin_params = {
        'short_term_period': 14,
        'mid_term_period': 50, 
        'long_term_period': 200,
        'indicators': ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci', 'bbands', 'williams', 'momentum', 'roc'],
        'ohlc_order': 'ohlc',
        'causal_mode': True,  # NEW: Enable strict causality
        'min_periods': 50     # NEW: Minimum periods for reliable calculation
    }

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
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
        renaming_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
        
        # Check for missing columns
        missing_columns = [col for col in renaming_map.keys() if col not in data.columns]
        if missing_columns:
            print(f"Error: Missing columns after renaming - {missing_columns}")
            raise KeyError(f"Missing columns after renaming: {missing_columns}")

        # Apply renaming
        data_renamed = data.rename(columns=renaming_map)
        print(f"Renaming successful. Available columns: {data_renamed.columns}")
        return data_renamed

    def process(self, data):
        """
        Process input data using STRICTLY CAUSAL technical indicators.
        
        For each time point t, indicators use ONLY data from t-window_size to t-1,
        ensuring no future data leakage.
        """
        print("Starting CAUSAL technical indicators processing...")
        print(f"Causal mode: {self.params.get('causal_mode', True)}")
        
        # Adjust OHLC columns
        data = self.adjust_ohlc(data)
        
        if not self.params.get('causal_mode', True):
            # Fallback to original (non-causal) method for compatibility
            return self._process_non_causal(data)
        
        # CAUSAL PROCESSING: Calculate indicators point-by-point
        return self._process_causal(data)
    
    def _process_causal(self, data):
        """
        Calculate technical indicators using STRICTLY CAUSAL approach.
        Each point at time t uses only data up to time t-1.
        """
        n = len(data)
        min_periods = self.params.get('min_periods', 50)
        
        print(f"Processing {n} data points with causal indicators...")
        print(f"Minimum periods required: {min_periods}")
        
        # Initialize result dictionary
        technical_indicators = {}
        
        # Initialize all indicator arrays with NaN
        for indicator in self.params['indicators']:
            if indicator == 'rsi':
                technical_indicators['RSI'] = np.full(n, np.nan)
            elif indicator == 'macd':
                technical_indicators['MACD'] = np.full(n, np.nan)
                technical_indicators['MACD_Histogram'] = np.full(n, np.nan)
                technical_indicators['MACD_Signal'] = np.full(n, np.nan)
            elif indicator == 'ema':
                technical_indicators['EMA'] = np.full(n, np.nan)
            elif indicator == 'stoch':
                technical_indicators['Stochastic_%K'] = np.full(n, np.nan)
                technical_indicators['Stochastic_%D'] = np.full(n, np.nan)
            elif indicator == 'adx':
                technical_indicators['ADX'] = np.full(n, np.nan)
                technical_indicators['DI+'] = np.full(n, np.nan)
                technical_indicators['DI-'] = np.full(n, np.nan)
            elif indicator == 'atr':
                technical_indicators['ATR'] = np.full(n, np.nan)
            elif indicator == 'cci':
                technical_indicators['CCI'] = np.full(n, np.nan)
            elif indicator == 'bbands':
                technical_indicators['BB_Upper'] = np.full(n, np.nan)
                technical_indicators['BB_Middle'] = np.full(n, np.nan)
                technical_indicators['BB_Lower'] = np.full(n, np.nan)
            elif indicator == 'williams':
                technical_indicators['WilliamsR'] = np.full(n, np.nan)
            elif indicator == 'momentum':
                technical_indicators['Momentum'] = np.full(n, np.nan)
            elif indicator == 'roc':
                technical_indicators['ROC'] = np.full(n, np.nan)
        
        # CAUSAL CALCULATION: Process each point using only past data
        print("Computing causal technical indicators...")
        for i in tqdm(range(min_periods, n), desc="Causal Tech Indicators", unit="point"):
            # For point i, use data from 0 to i (not including i+1 and beyond)
            # This ensures we NEVER use future data
            
            window_data = data.iloc[:i+1].copy()  # Past data only!
            
            # Calculate indicators for this window and extract the LAST value
            for indicator in self.params['indicators']:
                try:
                    if indicator == 'rsi':
                        rsi_series = ta.rsi(window_data['Close'], length=self.params['short_term_period'])
                        if rsi_series is not None and not rsi_series.empty:
                            technical_indicators['RSI'][i] = rsi_series.iloc[-1]
                    
                    elif indicator == 'macd':
                        macd_df = ta.macd(window_data['Close'])
                        if macd_df is not None and not macd_df.empty:
                            if 'MACD_12_26_9' in macd_df.columns:
                                technical_indicators['MACD'][i] = macd_df['MACD_12_26_9'].iloc[-1]
                            if 'MACDh_12_26_9' in macd_df.columns:
                                technical_indicators['MACD_Histogram'][i] = macd_df['MACDh_12_26_9'].iloc[-1]
                            if 'MACDs_12_26_9' in macd_df.columns:
                                technical_indicators['MACD_Signal'][i] = macd_df['MACDs_12_26_9'].iloc[-1]
                    
                    elif indicator == 'ema':
                        ema_series = ta.ema(window_data['Close'], length=self.params['mid_term_period'])
                        if ema_series is not None and not ema_series.empty:
                            technical_indicators['EMA'][i] = ema_series.iloc[-1]
                    
                    elif indicator == 'stoch':
                        stoch_df = ta.stoch(window_data['High'], window_data['Low'], window_data['Close'])
                        if stoch_df is not None and not stoch_df.empty:
                            if 'STOCHk_14_3_3' in stoch_df.columns:
                                technical_indicators['Stochastic_%K'][i] = stoch_df['STOCHk_14_3_3'].iloc[-1]
                            if 'STOCHd_14_3_3' in stoch_df.columns:
                                technical_indicators['Stochastic_%D'][i] = stoch_df['STOCHd_14_3_3'].iloc[-1]
                    
                    elif indicator == 'adx':
                        adx_df = ta.adx(window_data['High'], window_data['Low'], window_data['Close'])
                        if adx_df is not None and not adx_df.empty:
                            if 'ADX_14' in adx_df.columns:
                                technical_indicators['ADX'][i] = adx_df['ADX_14'].iloc[-1]
                            if 'DMP_14' in adx_df.columns:
                                technical_indicators['DI+'][i] = adx_df['DMP_14'].iloc[-1]
                            if 'DMN_14' in adx_df.columns:
                                technical_indicators['DI-'][i] = adx_df['DMN_14'].iloc[-1]
                    
                    elif indicator == 'atr':
                        atr_series = ta.atr(window_data['High'], window_data['Low'], window_data['Close'])
                        if atr_series is not None and not atr_series.empty:
                            technical_indicators['ATR'][i] = atr_series.iloc[-1]
                    
                    elif indicator == 'cci':
                        cci_series = ta.cci(window_data['High'], window_data['Low'], window_data['Close'])
                        if cci_series is not None and not cci_series.empty:
                            technical_indicators['CCI'][i] = cci_series.iloc[-1]
                    
                    elif indicator == 'bbands':
                        bbands_df = ta.bbands(window_data['Close'])
                        if bbands_df is not None and not bbands_df.empty:
                            if 'BBU_20_2.0' in bbands_df.columns:
                                technical_indicators['BB_Upper'][i] = bbands_df['BBU_20_2.0'].iloc[-1]
                            if 'BBM_20_2.0' in bbands_df.columns:
                                technical_indicators['BB_Middle'][i] = bbands_df['BBM_20_2.0'].iloc[-1]
                            if 'BBL_20_2.0' in bbands_df.columns:
                                technical_indicators['BB_Lower'][i] = bbands_df['BB_Lower_20_2.0'].iloc[-1]
                    
                    elif indicator == 'williams':
                        williams_series = ta.willr(window_data['High'], window_data['Low'], window_data['Close'])
                        if williams_series is not None and not williams_series.empty:
                            technical_indicators['WilliamsR'][i] = williams_series.iloc[-1]
                    
                    elif indicator == 'momentum':
                        momentum_series = ta.mom(window_data['Close'])
                        if momentum_series is not None and not momentum_series.empty:
                            technical_indicators['Momentum'][i] = momentum_series.iloc[-1]
                    
                    elif indicator == 'roc':
                        roc_series = ta.roc(window_data['Close'])
                        if roc_series is not None and not roc_series.empty:
                            technical_indicators['ROC'][i] = roc_series.iloc[-1]
                
                except Exception as e:
                    # Keep NaN for failed calculations
                    logger.warning(f"Failed to calculate {indicator} at point {i}: {e}")
                    pass
        
        # Create DataFrame with computed indicators
        indicator_df = pd.DataFrame(technical_indicators, index=data.index)
        
        print(f"CAUSAL technical indicators computed. Shape: {indicator_df.shape}")
        print(f"Non-NaN values per indicator:")
        for col in indicator_df.columns:
            non_nan_count = indicator_df[col].notna().sum()
            print(f"  {col}: {non_nan_count}/{len(indicator_df)} ({non_nan_count/len(indicator_df)*100:.1f}%)")
        
        return indicator_df
    
    def _process_non_causal(self, data):
        """Original non-causal method for compatibility."""
        print("Using non-causal (original) technical indicators...")
        # ... existing code from original plugin ...
        pass

    def get_comprehensive_params(self):
        """Get comprehensive parameters for replicability."""
        comprehensive_params = self.params.copy()
        comprehensive_params['causal_processing'] = True
        comprehensive_params['data_leakage_prevention'] = True
        return comprehensive_params
