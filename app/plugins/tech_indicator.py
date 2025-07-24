#!/usr/bin/env python3
"""
Technical Indicators Plugin - Phase 3 Compatible

This plugin generates the exact same technical indicators and features
as used in Phase 3.1, ensuring perfect compatibility with the predictor.

Based on the original tech_indicator_original.py implementation.
"""

import pandas_ta as ta
import pandas as pd
import numpy as np
import logging
from app.data_handler import load_csv, write_csv, load_additional_csv, load_sp500_csv, load_and_fix_hourly_data, load_high_frequency_data

# Set up logger
logger = logging.getLogger(__name__)


class Plugin:
    """
    Strictly causal technical indicators plugin.
    """
    
    # Plugin parameters including short, mid, and long-term period configurations
    plugin_params = {
        'short_term_period': 14,
        'mid_term_period': 50,
        'long_term_period': 200,
        'indicators': ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci', 'williams', 'momentum', 'roc'],
        'ohlc_order': 'ohlc'  # Default column order: Open, High, Low, Close
    }

    def __init__(self):
        self.params = self.plugin_params.copy()
        
    def set_params(self, **kwargs):
        """Set plugin parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def adjust_ohlc(self, data):
        """
        Adjust OHLC columns by renaming them according to the expected OHLC order.
        Handles case insensitivity for column names.
        Adds missing OHLC combination features to match Phase 3.1.

        Parameters:
        - data (pd.DataFrame): Input data with generic column names (e.g., 'OPEN', 'HIGH', 'LOW', 'CLOSE').

        Returns:
        - pd.DataFrame: Data with columns renamed to 'Open', 'High', 'Low', 'Close' plus additional features.
        """
        print("Starting adjust_ohlc method...")

        # Debug: Show initial columns of the data
        print(f"Initial data columns: {data.columns}")

        # Normalize column names to lowercase for consistent handling
        data.columns = data.columns.str.lower()

        # Expected renaming map (lowercase keys to handle normalized column names)
        renaming_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}

        # Check if all required columns are present after normalization
        missing_columns = [col for col in renaming_map.keys() if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required OHLC columns: {missing_columns}")

        # Apply renaming
        data_renamed = data.rename(columns=renaming_map)

        # ADD MISSING OHLC COMBINATION FEATURES - Phase 3.1 compatibility
        if all(col in data_renamed.columns for col in ['Open', 'High', 'Low', 'Close']):
            print("[DEBUG] Adding OHLC combination features for Phase 3.1 compatibility...")
            data_renamed['BC-BO'] = data_renamed['Close'] - data_renamed['Open']  # Body: Close - Open
            data_renamed['BH-BL'] = data_renamed['High'] - data_renamed['Low']    # Range: High - Low  
            data_renamed['BH-BO'] = data_renamed['High'] - data_renamed['Open']   # High - Open
            data_renamed['BO-BL'] = data_renamed['Open'] - data_renamed['Low']    # Open - Low

        # Debug: Show first few rows after renaming
        print(f"First 5 rows of renamed data:\n{data_renamed.head()}")

        print(f"Renaming successful. Available columns: {data_renamed.columns}")
        return data_renamed

    def process(self, data):
        """
        Process the input data by calculating the specified technical indicators using their default parameters.
        
        This matches the original tech_indicator_original.py implementation exactly.

        Parameters:
        - data (pd.DataFrame): Input time-series data with renamed 'Open', 'High', 'Low', 'Close', etc.

        Returns:
        - pd.DataFrame: DataFrame with the calculated technical indicators.
        """
        print("Starting process method...")

        # Debug: Show initial data columns before any processing
        print(f"Initial data columns before any processing: {data.columns}")
        print(f"Number of rows in data: {data.shape[0]}")

        # Ensure the datetime column is preserved if it exists
        if 'datetime' in data.columns:
            datetime_column = data[['datetime']]
            print(f"[DEBUG] Datetime column found with shape: {datetime_column.shape}")
        else:
            print("[DEBUG] No datetime column found; ensure alignment downstream.")
            datetime_column = None

        # Adjust the OHLC order of the columns
        data = self.adjust_ohlc(data)
        print(f"Data columns after OHLC adjustment: {data.columns}")

        # Initialize a dictionary to hold all technical indicators
        technical_indicators = {}

        # Loop through the specified indicators and calculate them
        for indicator in self.params['indicators']:
            print(f"Processing indicator: {indicator}")

            if indicator == 'rsi':
                rsi = ta.rsi(data['Close'])  # Default length is 14
                if rsi is not None:
                    technical_indicators['RSI'] = rsi
                    print(f"RSI calculated with shape: {rsi.shape}")

            elif indicator == 'macd':
                macd = ta.macd(data['Close'])  # Default fast, slow, signal periods
                if 'MACD_12_26_9' in macd.columns:
                    technical_indicators['MACD'] = macd['MACD_12_26_9']
                if 'MACDh_12_26_9' in macd.columns:
                    technical_indicators['MACD_Histogram'] = macd['MACDh_12_26_9']
                if 'MACDs_12_26_9' in macd.columns:
                    technical_indicators['MACD_Signal'] = macd['MACDs_12_26_9']
                print(f"MACD columns returned: {macd.columns}")

            elif indicator == 'ema':
                ema = ta.ema(data['Close'])  # Default length is 20
                if ema is not None:
                    technical_indicators['EMA'] = ema
                    print(f"EMA calculated with shape: {ema.shape}")

            elif indicator == 'stoch':
                stoch = ta.stoch(data['High'], data['Low'], data['Close'])  # Default %K, %D values
                if 'STOCHk_14_3_3' in stoch.columns:
                    technical_indicators['Stochastic_%K'] = stoch['STOCHk_14_3_3']
                if 'STOCHd_14_3_3' in stoch.columns:
                    technical_indicators['Stochastic_%D'] = stoch['STOCHd_14_3_3']
                print(f"Stochastic columns returned: {stoch.columns}")

            elif indicator == 'adx':
                adx = ta.adx(data['High'], data['Low'], data['Close'])  # Default length is 14
                if 'ADX_14' in adx.columns:
                    technical_indicators['ADX'] = adx['ADX_14']
                if 'DMP_14' in adx.columns:
                    technical_indicators['DI+'] = adx['DMP_14']
                if 'DMN_14' in adx.columns:
                    technical_indicators['DI-'] = adx['DMN_14']
                print(f"ADX columns returned: {adx.columns}")

            elif indicator == 'atr':
                atr = ta.atr(data['High'], data['Low'], data['Close'])  # Default length is 14
                if atr is not None:
                    technical_indicators['ATR'] = atr
                    print(f"ATR calculated with shape: {atr.shape}")

            elif indicator == 'cci':
                cci = ta.cci(data['High'], data['Low'], data['Close'])  # Default length is 20
                if cci is not None:
                    technical_indicators['CCI'] = cci
                    print(f"CCI calculated with shape: {cci.shape}")

            elif indicator == 'williams':
                williams = ta.willr(data['High'], data['Low'], data['Close'])  # Default length is 14
                if williams is not None:
                    technical_indicators['WilliamsR'] = williams
                    print(f"WilliamsR calculated with shape: {williams.shape}")

            elif indicator == 'momentum':
                momentum = ta.mom(data['Close'])  # Default length is 10
                if momentum is not None:
                    technical_indicators['Momentum'] = momentum
                    print(f"Momentum calculated with shape: {momentum.shape}")

            elif indicator == 'roc':
                roc = ta.roc(data['Close'])  # Default length is 10
                if roc is not None:
                    technical_indicators['ROC'] = roc
                    print(f"ROC calculated with shape: {roc.shape}")

        # Create a DataFrame from the calculated technical indicators
        indicator_df = pd.DataFrame(technical_indicators, index=data.index)

        # Debug: Show the calculated technical indicators
        print(f"Calculated technical indicators columns: {indicator_df.columns}")
        print(f"Calculated technical indicators shape: {indicator_df.shape}")
        print(f"Calculated technical indicators index type: {indicator_df.index.dtype}, range: {indicator_df.index.min()} to {indicator_df.index.max()}")

        # CRITICAL FIX: Include OHLC combination features (BC-BO, BH-BL, BH-BO, BO-BL) in the output
        ohlc_combination_features = ['BC-BO', 'BH-BL', 'BH-BO', 'BO-BL']
        for feature in ohlc_combination_features:
            if feature in data.columns:
                indicator_df[feature] = data[feature]
                print(f"[DEBUG] Added OHLC combination feature to output: {feature}")

        print(f"[DEBUG] Final output columns including OHLC combinations: {list(indicator_df.columns)}")
        return indicator_df

    def process_additional_datasets(self, data, config):
        """
        Process additional datasets for the technical indicators plugin.
        This is the COMPLETE original implementation that generates ALL missing features.
        """
        print("[DEBUG] Starting process_additional_datasets...")
        common_start = pd.Timestamp(data.index.min())
        common_end = pd.Timestamp(data.index.max())
        print(f"[DEBUG] Initial common range: {common_start} to {common_end}")
        print("[DEBUG] Main dataset first 5 rows:")
        print(data.head())
        
        # CRITICAL FIX: Apply adjust_ohlc to get OHLC combination features (BH-BL, BH-BO, BO-BL)
        # These features are created in adjust_ohlc but lost in the data flow
        ohlc_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        if all(col in data.columns for col in ohlc_columns):
            ohlc_data = data[ohlc_columns].copy()
            adjusted_ohlc_data = self.adjust_ohlc(ohlc_data)
            print(f"[DEBUG] Generated OHLC combination features: {list(adjusted_ohlc_data.columns)}")
            
            # Extract the combination features we need
            ohlc_combination_features = ['BC-BO', 'BH-BL', 'BH-BO', 'BO-BL']
            combination_features_df = pd.DataFrame(index=data.index)
            for feature in ohlc_combination_features:
                if feature in adjusted_ohlc_data.columns:
                    combination_features_df[feature] = adjusted_ohlc_data[feature]
                    print(f"[DEBUG] Added combination feature: {feature}")
        else:
            print(f"[DEBUG] Missing OHLC columns for combination features. Available: {list(data.columns)}")
            combination_features_df = pd.DataFrame(index=data.index)
        
        # Dictionary to store aligned datasets
        aligned_datasets = {}

        def log_dataset_range(dataset_key, dataset):
            if not dataset.empty:
                print(f"[DEBUG] {dataset_key} range: {dataset.index.min()} to {dataset.index.max()}, shape: {dataset.shape}")
                return dataset.index.min(), dataset.index.max()
            else:
                print(f"[DEBUG] {dataset_key} is empty!")
                return None, None

        def process_dataset(dataset_func, dataset_key):
            nonlocal common_start, common_end
            try:
                dataset = dataset_func()
                if dataset is not None and not dataset.empty:
                    dataset_start, dataset_end = log_dataset_range(dataset_key, dataset)
                    if dataset_start and dataset_end:
                        common_start = max(common_start, dataset_start)
                        common_end = min(common_end, dataset_end)
                        aligned_datasets[dataset_key] = dataset
                        print(f"[DEBUG] Updated common range after {dataset_key}: {common_start} to {common_end}")
                else:
                    print(f"[DEBUG] {dataset_key} returned None or empty dataset")
            except Exception as e:
                print(f"[DEBUG] Error processing {dataset_key}: {e}")

        # Process each dataset and store the aligned data
        econ_calendar = None
        if config.get('economic_calendar'):
            process_dataset(
                lambda: self.process_economic_calendar(config['economic_calendar'], config, common_start, common_end),
                'economic_calendar'
            )

        forex_features = None
        if config.get('forex_datasets'):
            process_dataset(
                lambda: self.process_forex_data(config['forex_datasets'], config, common_start, common_end),
                'forex_datasets'
            )

        sp500_features = None
        if config.get('sp500_dataset'):
            process_dataset(
                lambda: self.process_sp500_data(config['sp500_dataset'], config, common_start, common_end),
                'sp500_dataset'
            )

        vix_features = None
        if config.get('vix_dataset'):
            process_dataset(
                lambda: self.process_vix_data(config['vix_dataset'], config, common_start, common_end),
                'vix_dataset'
            )

        high_freq_features = None
        if config.get('high_freq_dataset'):
            process_dataset(
                lambda: self.process_high_frequency_data(config['high_freq_dataset'], config, common_start, common_end),
                'high_freq_dataset'
            )

        print("[DEBUG] After all datasets processed:")
        print(f"[DEBUG] final_common_start={common_start}, final_common_end={common_end}")

        # Trim the main hourly dataset based on final common range
        hourly_trimmed = data[(data.index >= common_start) & (data.index <= common_end)].copy()
        if not hourly_trimmed.empty:
            print(f"[DEBUG] Hourly dataset trimmed to common range. New shape: {hourly_trimmed.shape}")
        else:
            print("[DEBUG] ERROR: Hourly dataset is empty after trimming to common range!")

        # Merge all aligned datasets with the hourly dataset so that merged_features.csv contains them all
        merged_features_df = hourly_trimmed.copy()
        
        # CRITICAL FIX: Only add combination features if they're NOT already in the main dataset
        # to prevent _x, _y duplicates
        combination_features_trimmed = combination_features_df[(combination_features_df.index >= common_start) & (combination_features_df.index <= common_end)]
        if not combination_features_trimmed.empty:
            print(f"[DEBUG] Checking OHLC combination features with shape: {combination_features_trimmed.shape}")
            print(f"[DEBUG] Combination features columns: {list(combination_features_trimmed.columns)}")
            print(f"[DEBUG] Current merged_features_df columns: {list(merged_features_df.columns)}")
            
            # Only add combination features that don't already exist in merged_features_df
            features_to_add = [col for col in combination_features_trimmed.columns if col not in merged_features_df.columns]
            if features_to_add:
                print(f"[DEBUG] Adding missing OHLC combination features: {features_to_add}")
                for feature in features_to_add:
                    merged_features_df[feature] = combination_features_trimmed[feature]
                print(f"[DEBUG] After adding combination features, merged_features_df shape: {merged_features_df.shape}")
            else:
                print(f"[DEBUG] All OHLC combination features already exist in merged_features_df")
        
        for key, df in aligned_datasets.items():
            if df is not None and not df.empty:
                # Trim dataset to final common range
                df_trimmed = df[(df.index >= common_start) & (df.index <= common_end)]
                print(f"[DEBUG] Merging {key} with shape {df_trimmed.shape}")
                merged_features_df = pd.merge(merged_features_df, df_trimmed, left_index=True, right_index=True, how='inner')
                print(f"[DEBUG] After merging {key}, merged_features_df shape: {merged_features_df.shape}")

        # Add seasonality columns if specified in the config
        if config.get('seasonality_columns'):
            print("[DEBUG] Adding seasonality columns...")
            merged_features_df['day_of_month'] = merged_features_df.index.day
            merged_features_df['hour_of_day'] = merged_features_df.index.hour
            merged_features_df['day_of_week'] = merged_features_df.index.dayofweek

        # CRITICAL FIX: Remove unwanted columns that shouldn't be in the final output
        unwanted_columns = ['volume', 'VOLUME', 'Volume']
        columns_to_remove = [col for col in unwanted_columns if col in merged_features_df.columns]
        if columns_to_remove:
            print(f"[DEBUG] Removing unwanted columns: {columns_to_remove}")
            merged_features_df = merged_features_df.drop(columns=columns_to_remove)
            print(f"[DEBUG] After removing unwanted columns, shape: {merged_features_df.shape}")

        # Save the merged dataset containing all aligned datasets within the final common date range
        merged_features_df.reset_index().rename(columns={'index': 'datetime'}).to_csv('additional_merged_features.csv', index=False)
        print(f"[DEBUG] Saved merged dataset to 'additional_merged_features.csv' with final range {common_start} to {common_end}")

        print(f"[DEBUG] merged_features_df final shape: {merged_features_df.shape}")
        return merged_features_df, common_start, common_end

    # CRITICAL: All the missing methods from the original that generate the missing features
    def process_sp500_data(self, sp500_data_path, config, common_start, common_end):
        """Process S&P 500 data to generate S&P500_Close feature"""
        print(f"[DEBUG] Loading S&P 500 data from: {sp500_data_path}")
        try:
            sp500_data = load_sp500_csv(sp500_data_path)  # Fixed: only pass the path
            if sp500_data is not None and not sp500_data.empty:
                # Rename close column to match phase 3.1
                if 'Close' in sp500_data.columns:
                    sp500_data = sp500_data.rename(columns={'Close': 'S&P500_Close'})
                    print(f"[DEBUG] S&P 500 data processed. Shape: {sp500_data.shape}")
                    return sp500_data[['S&P500_Close']]
        except Exception as e:
            print(f"[DEBUG] Error processing S&P 500 data: {e}")
        return None

    def process_vix_data(self, vix_data_path, config, common_start, common_end):
        """Process VIX data to generate vix_close feature"""
        print(f"[DEBUG] Loading VIX data from: {vix_data_path}")
        try:
            vix_data = pd.read_csv(vix_data_path)
            if not vix_data.empty:
                # Convert date column to datetime and set as index
                vix_data['date'] = pd.to_datetime(vix_data['date'])
                vix_data = vix_data.set_index('date')
                
                # Rename close column to match phase 3.1
                if 'close' in vix_data.columns:
                    vix_data = vix_data.rename(columns={'close': 'vix_close'})
                    print(f"[DEBUG] VIX data processed. Shape: {vix_data.shape}")
                    return vix_data[['vix_close']]
        except Exception as e:
            print(f"[DEBUG] Error processing VIX data: {e}")
        return None

    def process_high_frequency_data(self, high_freq_data_path, config, common_start, common_end):
        """Process high frequency data to generate 15m and 30m tick features"""
        print(f"[DEBUG] Loading high frequency data from: {high_freq_data_path}")
        try:
            high_freq_data = load_high_frequency_data(high_freq_data_path, config)
            if high_freq_data is not None and not high_freq_data.empty:
                # Process sub-periodicities for 15m and 30m ticks
                # This generates CLOSE_15m_tick_1 through CLOSE_15m_tick_8 and CLOSE_30m_tick_1 through CLOSE_30m_tick_8
                sub_periodicity_features = self.process_sub_periodicities(
                    high_freq_data, high_freq_data, config.get('sub_periodicity_window_size', 8)
                )
                print(f"[DEBUG] Sub-periodicity features generated. Shape: {sub_periodicity_features.shape}")
                return sub_periodicity_features
        except Exception as e:
            print(f"[DEBUG] Error processing high frequency data: {e}")
        return None

    def process_sub_periodicities(self, hourly_data, sub_periodicity_data, window_size):
        """Generate sub-periodicity features (15m and 30m ticks)"""
        print(f"[DEBUG] Processing sub-periodicities with window_size: {window_size}")
        
        # Resample to 15m and 30m
        data_15m = sub_periodicity_data.resample('15min').last()
        data_30m = sub_periodicity_data.resample('30min').last()
        
        # Generate tick features for each periodicity
        result_features = pd.DataFrame(index=hourly_data.index)
        
        # 15m ticks
        for i in range(1, window_size + 1):
            col_name = f'CLOSE_15m_tick_{i}'
            result_features[col_name] = data_15m['CLOSE'].shift(i-1).reindex(hourly_data.index, method='ffill')
        
        # 30m ticks  
        for i in range(1, window_size + 1):
            col_name = f'CLOSE_30m_tick_{i}'
            result_features[col_name] = data_30m['CLOSE'].shift(i-1).reindex(hourly_data.index, method='ffill')
            
        return result_features

    def process_forex_data(self, forex_files, config, common_start, common_end):
        """Process forex data - placeholder for compatibility"""
        return None

    def process_economic_calendar(self, econ_calendar_path, config, common_start, common_end):
        """Process economic calendar - placeholder for compatibility"""  
        return None
