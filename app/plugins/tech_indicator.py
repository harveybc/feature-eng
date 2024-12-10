import pandas_ta as ta
import pandas as pd
import numpy as np
from app.data_handler import load_csv, write_csv, load_additional_csv,load_sp500_csv, load_and_fix_hourly_data, load_high_frequency_data
from app.positional_encoding import generate_positional_encoding
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv1D, Dense, GlobalAveragePooling1D, Flatten
from keras.optimizers import Adam
import os
from tqdm import tqdm  # For progress indication

class Plugin:
    """
    A feature-engineering plugin using technical indicators.
    """

    # Plugin parameters including short, mid, and long-term period configurations
    plugin_params = {
        'short_term_period': 14,
        'mid_term_period': 50,
        'long_term_period': 200,
        'indicators': ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci', 'bbands', 'williams', 'momentum', 'roc'],
        'ohlc_order': 'ohlc'  # Default column order: Open, High, Low, Close
    }

    # Debug variables to track important parameters and results
    plugin_debug_vars = ['short_term_period', 'mid_term_period', 'long_term_period', 'output_columns', 'ohlc_order']

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var, None) for var in self.plugin_debug_vars}



    def adjust_ohlc(self, data):
        """
        Adjust OHLC columns by renaming them according to the expected OHLC order.
        Handles case insensitivity for column names.

        Parameters:
        - data (pd.DataFrame): Input data with generic column names (e.g., 'OPEN', 'HIGH', 'LOW', 'CLOSE').

        Returns:
        - pd.DataFrame: Data with columns renamed to 'Open', 'High', 'Low', 'Close'.
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
            print(f"Error: Missing columns after renaming - {missing_columns}")
            print(f"Available columns: {data.columns}")
            raise KeyError(f"Missing columns after renaming: {missing_columns}")

        # Apply renaming
        data_renamed = data.rename(columns=renaming_map)

        # Debug: Show first few rows after renaming
        print(f"First 5 rows of renamed data:\n{data_renamed.head()}")

        print(f"Renaming successful. Available columns: {data_renamed.columns}")
        return data_renamed


    def process(self, data):
        """
        Process the input data by calculating the specified technical indicators using their default parameters.

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

            elif indicator == 'bbands':
                bbands = ta.bbands(data['Close'])  # Default length is 20
                if 'BBU_20_2.0' in bbands.columns:
                    technical_indicators['BB_Upper'] = bbands['BBU_20_2.0']
                if 'BBM_20_2.0' in bbands.columns:
                    technical_indicators['BB_Middle'] = bbands['BBM_20_2.0']
                if 'BBL_20_2.0' in bbands.columns:
                    technical_indicators['BB_Lower'] = bbands['BBL_20_2.0']
                print(f"Bollinger Bands columns returned: {bbands.columns}")

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

        return indicator_df


    def process_additional_datasets(self, data, config):
        print("[DEBUG] Starting process_additional_datasets...")
        common_start = pd.Timestamp(data.index.min())
        common_end = pd.Timestamp(data.index.max())
        print(f"[DEBUG] Initial common range: {common_start} to {common_end}")
        print("[DEBUG] Main dataset first 5 rows:")
        print(data.head())

        def log_dataset_range(dataset_key, dataset):
            print(f"[DEBUG] Dataset: {dataset_key}")
            if dataset is not None and not dataset.empty:
                print(f"    [DEBUG] {dataset_key} original range: {dataset.index.min()} to {dataset.index.max()}")
                print(f"    [DEBUG] {dataset_key} shape: {dataset.shape}")
                print(f"    [DEBUG] {dataset_key} first 5 rows:\n{dataset.head()}")
            else:
                print(f"[ERROR] {dataset_key} is empty or invalid.")

        def process_dataset(dataset_func, dataset_key):
            nonlocal common_start, common_end
            print(f"[DEBUG] Processing {dataset_key} with current range {common_start} to {common_end}")
            dataset = dataset_func(config[dataset_key], config, common_start, common_end)
            log_dataset_range(dataset_key, dataset)

            if dataset is not None and not dataset.empty:
                aligned_dataset = dataset[(dataset.index >= common_start) & (dataset.index <= common_end)]
                if aligned_dataset.empty:
                    print(f"[ERROR] After alignment, {dataset_key} dataset is empty. Check the alignment logic.")
                else:
                    print(f"[DEBUG] {dataset_key} after alignment: {aligned_dataset.index.min()} to {aligned_dataset.index.max()}")
                    # Update common_start and common_end based on aligned dataset
                    old_common_start, old_common_end = common_start, common_end
                    common_start = max(common_start, aligned_dataset.index.min())
                    common_end = min(common_end, aligned_dataset.index.max())
                    print(f"[DEBUG] Updated common range: {old_common_start} to {old_common_end} -> {common_start} to {common_end}")
                return aligned_dataset
            else:
                print(f"[DEBUG] {dataset_key} is empty or no data returned, skipping alignment update.")
            return None

        additional_features = {}

        # Process economic_calendar first to set the correct common_start
        if config.get('economic_calendar'):
            econ_calendar = process_dataset(self.process_economic_calendar, 'economic_calendar')
            if econ_calendar is not None and not econ_calendar.empty:
                additional_features.update(econ_calendar.to_dict(orient='series'))

        # Process other datasets after economic_calendar to maintain the updated common_start and common_end
        if config.get('forex_datasets'):
            forex_features = process_dataset(self.process_forex_data, 'forex_datasets')
            if forex_features is not None and not forex_features.empty:
                additional_features.update(forex_features.to_dict(orient='series'))

        if config.get('sp500_dataset'):
            sp500_features = process_dataset(self.process_sp500_data, 'sp500_dataset')
            if sp500_features is not None and not sp500_features.empty:
                additional_features.update(sp500_features.to_dict(orient='series'))

        if config.get('vix_dataset'):
            vix_features = process_dataset(self.process_vix_data, 'vix_dataset')
            if vix_features is not None and not vix_features.empty:
                additional_features.update(vix_features.to_dict(orient='series'))

        if config.get('high_freq_dataset'):
            high_freq_features = process_dataset(self.process_high_frequency_data, 'high_freq_dataset')
            if high_freq_features is not None and not high_freq_features.empty:
                additional_features.update(high_freq_features.to_dict(orient='series'))

        print("[DEBUG] After all datasets processed:")
        print(f"[DEBUG] final_common_start={common_start}, final_common_end={common_end}")
        additional_features_df = pd.DataFrame(additional_features)
        if not additional_features_df.empty:
            additional_features_df = additional_features_df[(additional_features_df.index >= common_start) & (additional_features_df.index <= common_end)]
        print(f"[DEBUG] additional_features_df final shape: {additional_features_df.shape}")
        return additional_features_df, common_start, common_end


    def process_economic_calendar(self, econ_calendar_path, config, common_start, common_end):
        """
        Process the economic calendar dataset:
        - Load and preprocess data
        - Filter duplicates using priority rules
        - Align and resample to hourly based on existing hourly data (excluding weekends)
        - Fill missing hours with sentinel (-1)
        """
        print("Processing economic calendar data...")

        # Load the hourly dataset (main hourly_data)
        hourly_data = load_and_fix_hourly_data(config['input_file'], config)
        print(f"[DEBUG] Hourly dataset loaded. Index range: {hourly_data.index.min()} to {hourly_data.index.max()}")

        # Load the economic calendar dataset
        econ_data = pd.read_csv(
            econ_calendar_path,
            header=None,
            names=['event_date', 'event_time', 'country', 'volatility',
                'description', 'evaluation', 'data_format',
                'actual', 'forecast', 'previous'],
            dtype={'event_date': str, 'event_time': str},
        )

        print("[DEBUG] Economic calendar loaded successfully.")
        print(f"[DEBUG] Economic calendar column types:\n{econ_data.dtypes}")

        # Combine 'event_date' and 'event_time' into 'datetime'
        econ_data['datetime'] = pd.to_datetime(
            econ_data['event_date'].str.strip() + ' ' + econ_data['event_time'].str.strip(),
            errors='coerce'
        )
        print(f"[DEBUG] Combined datetime column (first 5):\n{econ_data['datetime'].head()}")

        # Drop rows with invalid datetime and set as index
        econ_data.dropna(subset=['datetime'], inplace=True)
        econ_data.set_index('datetime', inplace=True)
        print("[DEBUG] Economic calendar index set successfully.")
        print(f"[DEBUG] Economic calendar index range: {econ_data.index.min()} to {econ_data.index.max()}")

        # Preprocess econ_data
        econ_data = self._preprocess_economic_calendar_data(econ_data)
        print("[DEBUG] Economic calendar data preprocessing complete.")
        print(f"[DEBUG] Processed economic calendar datetime range: {econ_data.index.min()} to {econ_data.index.max()}")

        # Update common_start to the start of econ_data
        econ_start = econ_data.index.min()
        print(f"[DEBUG] Updated common_start to economic calendar start date: {econ_start}")

        # Adjust common_end if necessary
        econ_end = econ_data.index.max()
        if common_end > econ_end:
            common_end = econ_end
            print(f"[DEBUG] Adjusted common_end to economic calendar end date: {common_end}")

        print("[DEBUG] Starting alignment process for economic calendar.")

        # Filter duplicates with a progress meter
        grouped = econ_data.groupby(econ_data.index)
        unique_timestamps = grouped.ngroups
        print("[DEBUG] Filtering duplicates...")
        filtered_rows = []
        for timestamp, group in tqdm(grouped, desc="Filtering duplicates", unit="group", total=unique_timestamps):
            filtered = self._filter_duplicate_events(group)
            filtered_rows.append(filtered)

        # Concatenate all filtered single-row DataFrames
        econ_data = pd.concat(filtered_rows, ignore_index=True)

        # Ensure 'datetime' column exists before setting index
        if 'datetime' not in econ_data.columns:
            raise ValueError("After filtering duplicates, 'datetime' column not found. Ensure it exists.")

        econ_data.set_index('datetime', inplace=True)
        print("[DEBUG] After filtering duplicates, only one event per timestamp remains.")
        print(f"[DEBUG] econ_data index range after filtering: {econ_data.index.min()} to {econ_data.index.max()}")
        print(f"[DEBUG] econ_data shape: {econ_data.shape}")

        # Align econ_data to the updated common range
        econ_data = econ_data[(econ_data.index >= econ_start) & (econ_data.index <= common_end)]
        print(f"[DEBUG] Econ data rows after alignment: {len(econ_data)}")
        if not econ_data.empty:
            print(f"[DEBUG] Econ data index range after alignment: {econ_data.index.min()} to {econ_data.index.max()}")
        else:
            print("[ERROR] Alignment resulted in an empty dataset.")
            raise ValueError("[ERROR] Economic calendar dataset is empty after alignment.")

        # Slice hourly_data to the updated common range
        final_hourly_data = hourly_data[(hourly_data.index >= econ_start) & (hourly_data.index <= common_end)]
        print(f"[DEBUG] Final hourly data rows after slicing: {len(final_hourly_data)}")
        if final_hourly_data.empty:
            print("[ERROR] Final hourly data is empty after slicing.")
            raise ValueError("[ERROR] Hourly data is empty after slicing to the common range.")

        # Reindex econ_data to match the existing hourly_data indices (excluding weekends)
        econ_data_aligned = econ_data.reindex(final_hourly_data.index, fill_value=-1)
        print(f"[DEBUG] econ_data_aligned range: {econ_data_aligned.index.min()} to {econ_data_aligned.index.max()} (len={len(econ_data_aligned)})")
        print(f"[DEBUG] final_hourly_data range: {final_hourly_data.index.min()} to {final_hourly_data.index.max()} (len={len(final_hourly_data)})")

        # Verify alignment
        if len(econ_data_aligned) != len(final_hourly_data):
            print("[ERROR] Length mismatch after alignment:")
            print(f"[DEBUG] econ_data_aligned length: {len(econ_data_aligned)}, final_hourly_data length: {len(final_hourly_data)}")
            print("[DEBUG] econ_data_aligned first 5 rows:\n", econ_data_aligned.head())
            print("[DEBUG] final_hourly_data first 5 rows:\n", final_hourly_data.head())
            raise ValueError("econ_data_aligned and final_hourly_data must have the same number of rows and alignment.")

        # Generate sliding window features
        window_size = config['calendar_window_size']
        econ_features = self._generate_sliding_window_features(econ_data_aligned, final_hourly_data, window_size)
        print(f"[DEBUG] Sliding window feature generation complete. Shape: {econ_features.shape}")

        # Generate training signals for trend and volatility
        trend_signal, volatility_signal = self._generate_training_signals(final_hourly_data, config)
        print("[DEBUG] Training signals for trend and volatility generated.")

        # Slice the training signals to match the number of sliding windows
        # Each window predicts the trend and volatility at the end of the window
        trend_signal = trend_signal[window_size:]
        volatility_signal = volatility_signal[window_size:]
        print(f"[DEBUG] Sliced trend_signal shape: {trend_signal.shape}")
        print(f"[DEBUG] Sliced volatility_signal shape: {volatility_signal.shape}")

        # Confirm that features and labels have the same number of samples
        if econ_features.shape[0] != len(trend_signal) or econ_features.shape[0] != len(volatility_signal):
            raise ValueError("Features and training signals have different number of samples.")

        # Prepare target variables
        y = np.stack([trend_signal, volatility_signal], axis=1)

        # Train and predict
        predictions = self._predict_trend_and_volatility_with_conv1d(
            econ_features=econ_features,
            training_signals=y,  # Pass the sliced and stacked y directly
            window_size=window_size
        )
        print(f"[DEBUG] Predictions generated. Predictions shape: {predictions.shape}")

        # Assign predictions to trend and volatility
        predicted_trend, predicted_volatility = predictions[:, 0], predictions[:, 1]
        print(f"[DEBUG] Predicted trend length: {len(predicted_trend)}")
        print(f"[DEBUG] Predicted volatility length: {len(predicted_volatility)}")

        # Adjusted index corresponds to the end of each sliding window
        adjusted_index = final_hourly_data.index[window_size:]
        print(f"[DEBUG] Adjusted index length: {len(adjusted_index)}")
        print(f"[DEBUG] Adjusted index datetime range: {adjusted_index.min()} to {adjusted_index.max()}")

        # Ensure that the number of predictions matches the adjusted index
        if len(predicted_trend) != len(adjusted_index):
            raise ValueError("Length mismatch after window adjustment.")

        # Create aligned Series
        aligned_trend = pd.Series(predicted_trend, index=adjusted_index, name="Predicted_Trend")
        aligned_volatility = pd.Series(predicted_volatility, index=adjusted_index, name="Predicted_Volatility")

        # Save the trained model
        self._save_trained_model('trained_conv1d_model.h5', self.latest_model)

        return pd.concat([aligned_trend, aligned_volatility], axis=1)






    def _preprocess_economic_calendar_data(self, econ_data):
        print("Preprocessing economic calendar data...")
        str_cols = econ_data.select_dtypes(include=['object']).columns
        for col in str_cols:
            econ_data[col] = econ_data[col].astype(str).str.strip()

        volatility_mapping = {
            'Low Volatility Expected': 1,
            'Moderate Volatility Expected': 2,
            'High Volatility Expected': 3
        }
        econ_data['volatility'] = econ_data['volatility'].map(volatility_mapping)

        for col in ['actual', 'forecast', 'previous']:
            econ_data[col] = econ_data[col].str.replace(',', '', regex=False)
            econ_data[col] = econ_data[col].str.replace('[kK]$', '', regex=True)

        for col in ['forecast', 'actual', 'volatility', 'previous']:
            econ_data[col] = pd.to_numeric(econ_data[col], errors='coerce')

        econ_data.dropna(subset=['forecast', 'actual', 'volatility'], inplace=True)

        econ_data['forecast_diff'] = econ_data['actual'] - econ_data['forecast']
        econ_data['volatility_weighted_diff'] = econ_data['forecast_diff'] * econ_data['volatility']

        numerical_cols = ['forecast', 'actual', 'volatility', 'forecast_diff', 'volatility_weighted_diff']
        for col in numerical_cols:
            col_min = econ_data[col].min()
            col_max = econ_data[col].max()
            if col_max != col_min:
                econ_data[col] = (econ_data[col] - col_min) / (col_max - col_min)
            else:
                econ_data[col] = 0.0

        econ_data['country_encoded'] = econ_data['country'].astype('category').cat.codes
        econ_data['description_encoded'] = econ_data['description'].astype('category').cat.codes

        print("Economic calendar data preprocessing complete.")
        return econ_data

    def _generate_training_signals(self, hourly_data, config):
        print("Generating training signals...")
        short_term_window = config['calendar_window_size'] // 10
        print(f"[DEBUG] Short-term window size for training signals: {short_term_window}")

        trend_signal = (
            hourly_data['close']
            .ewm(span=short_term_window)
            .mean()
            .diff()
            .fillna(0)
            .values
        )

        volatility_signal = (
            hourly_data['close']
            .rolling(window=short_term_window)
            .std()
            .fillna(0)
            .values
        )
        print("Training signals generated.")
        return trend_signal, volatility_signal


    def _filter_duplicate_events(self, events):
        events_sorted = events.sort_values(by='volatility', ascending=False)
        max_vol = events_sorted['volatility'].iloc[0]
        top_events = events_sorted[events_sorted['volatility'] == max_vol]

        if len(top_events) == 1:
            chosen = top_events.iloc[0]
        else:
            usa_events = top_events[top_events['country'].str.upper() == 'USA']
            if len(usa_events) == 1:
                chosen = usa_events.iloc[0]
            elif len(usa_events) > 1:
                chosen = usa_events.sample(1).iloc[0]
            else:
                chosen = top_events.sample(1).iloc[0]

        timestamp = events.index[0]
        chosen_df = pd.DataFrame([chosen])
        chosen_df['datetime'] = timestamp
        return chosen_df

    def _generate_sliding_window_features(self, econ_data_aligned, final_hourly_data, window_size):
        import numpy as np

        numeric_cols = ['forecast', 'actual', 'volatility', 'forecast_diff', 'volatility_weighted_diff']
        cat_cols = ['country_encoded', 'description_encoded']
        all_cols = numeric_cols + cat_cols

        # Convert econ_data_aligned to numpy array
        econ_array = econ_data_aligned[all_cols].to_numpy()
        N = econ_array.shape[0]
        M = len(all_cols)

        print(f"[DEBUG] econ_array shape: {econ_array.shape}")
        print(f"[DEBUG] window_size: {window_size}")

        if N < window_size:
            print("[ERROR] Not enough data points to form a full window.")
            raise ValueError("Not enough data points to form even one full window.")

        num_windows = N - window_size
        windows = np.empty((num_windows, window_size, M), dtype=econ_array.dtype)

        for i in range(num_windows):
            windows[i] = econ_array[i:i+window_size]

        print(f"[DEBUG] windows shape after manual sliding window: {windows.shape}")  # Expected: (12308,128,7)

        # Extract numeric and categorical data
        numeric_data = windows[..., :5]  # Shape: (12308,128,5)
        cat_data = windows[..., 5:7]      # Shape: (12308,128,2)

        # Generate event mask
        no_event = (numeric_data == -1).all(axis=-1)  # Shape: (12308,128)
        event_mask = (~no_event).astype(np.float32)[..., np.newaxis]  # Shape: (12308,128,1)

        # Concatenate features
        features = np.concatenate([numeric_data, cat_data, event_mask], axis=-1)  # Shape: (12308,128,8)
        print(f"[DEBUG] features shape after concatenation: {features.shape}")  # Expected: (12308,128,8)

        print(f"[DEBUG] Sliding window feature generation complete. Final features shape: {features.shape}")
        print("[DEBUG] First window event_mask values:", features[0, :, -1] if features.shape[0] > 0 else "No features")
        return features





    def _predict_trend_and_volatility_with_conv1d(self, econ_features, training_signals, window_size):
        """
        Train and use a Conv1D model to predict short-term trend and volatility.

        Parameters:
        - econ_features (np.ndarray): Features generated from the sliding window.
        - training_signals (np.ndarray): Numpy array with shape (samples, 2) containing 'trend' and 'volatility' signals.
        - window_size (int): Size of the sliding window.

        Returns:
        - np.ndarray: Predictions for trend and volatility for each hourly tick.
        """
        from keras.models import Sequential
        from keras.layers import Conv1D, Dense, Flatten, BatchNormalization, Dropout
        from keras.optimizers import Adam
        from sklearn.model_selection import train_test_split
        from keras.callbacks import EarlyStopping, ModelCheckpoint

        print("Training Conv1D model for trend and volatility predictions...")

        # Prepare target variables
        y = training_signals  # Already sliced and stacked in process_economic_calendar

        # Verify that y has the same number of samples as econ_features
        if len(y) != econ_features.shape[0]:
            raise ValueError(f"Number of labels ({len(y)}) does not match number of features ({econ_features.shape[0]}).")

        print(f"[DEBUG] econ_features shape: {econ_features.shape}")
        print(f"[DEBUG] y shape: {y.shape}")

        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            econ_features, 
            y, 
            test_size=0.2, 
            random_state=42
        )

        print(f"[DEBUG] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"[DEBUG] y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_conv1d_model.h5', monitor='val_loss', save_best_only=True)

        # Build Conv1D model
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(window_size, econ_features.shape[2])),
            BatchNormalization(),
            Conv1D(64, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(2, activation='linear')  # Two outputs: trend and volatility
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        print("[DEBUG] Conv1D model compiled successfully.")

        # Train the model with callbacks
        history = model.fit(
            X_train, 
            y_train, 
            validation_data=(X_test, y_test), 
            epochs=5,  # Increased epochs for better learning
            batch_size=32, 
            verbose=1,
            callbacks=[early_stop, checkpoint]
        )

        print("Conv1D model training complete.")

        # Load the best model
        model.load_weights('best_conv1d_model.h5')
        print("[DEBUG] Loaded the best model weights from checkpoint.")

        # Save the trained model
        self.latest_model = model  # Store the latest model for saving
        # Alternatively, return the model if needed elsewhere

        # Predict on all data
        predictions = model.predict(econ_features)
        print(f"Predictions shape: {predictions.shape}")
        return predictions




    
    def train_economic_calendar_model(self, econ_calendar_path, hourly_data_path, config):
        """
        Train the Conv1D model for predicting short-term volatility and trend variation.

        Parameters:
        - econ_calendar_path (str): Path to the economic calendar dataset.
        - hourly_data_path (str): Path to the hourly dataset for calculating targets.
        - config (dict): Configuration dictionary.

        Returns:
        - keras.Model: Trained Conv1D model.
        """
        print("Training economic calendar Conv1D model...")
        
        # Load and preprocess the economic calendar data
        econ_calendar_data = self.load_economic_calendar(econ_calendar_path, config)
        hourly_data = load_and_fix_hourly_data(hourly_data_path, config)
        
        # Generate sliding window data
        window_size = config['calendar_window_size']
        X, y_volatility, y_trend = self.generate_training_data(econ_calendar_data, hourly_data, window_size, config)

        # Build the Conv1D model
        input_shape = X.shape[1:]  # (window_size, num_features)
        model = self.build_conv1d_model(input_shape)

        # Compile the model
        model.compile(
            optimizer='adam',
            loss={'volatility_output': 'mse', 'trend_output': 'mse'},
            metrics={'volatility_output': 'mae', 'trend_output': 'mae'}
        )
        
        # Train the model
        model.fit(
            X,
            {'volatility_output': y_volatility, 'trend_output': y_trend},
            epochs=config.get('epochs', 5),
            batch_size=config.get('batch_size', 32),
            validation_split=0.2,
            verbose=1
        )
        print("Model training completed.")
        return model

    def generate_training_data(self, econ_calendar_data, hourly_data, window_size, config):
        """
        Generate training data for Conv1D model using economic calendar and hourly data.

        Parameters:
        - econ_calendar_data (pd.DataFrame): Processed economic calendar data.
        - hourly_data (pd.DataFrame): Hourly dataset with close prices.
        - window_size (int): Number of ticks in the sliding window.
        - config (dict): Configuration dictionary.

        Returns:
        - X (np.ndarray): Input features for the Conv1D model.
        - y_volatility (np.ndarray): Target volatility values.
        - y_trend (np.ndarray): Target trend variation values.
        """
        print("Generating training data for economic calendar model...")
        
        event_features = econ_calendar_data.values  # Event data as array
        prices = hourly_data['close'].values  # Close prices
        ema = hourly_data['close'].ewm(span=window_size).mean().values  # EMA for trend calculation

        X, y_volatility, y_trend = [], [], []
        
        for i in range(window_size, len(hourly_data)):
            # Input features from economic calendar events in the window
            X_window = event_features[i - window_size:i]
            
            # Target: Volatility in the short-term window
            price_window = prices[i - window_size:i]
            volatility = price_window.max() - price_window.min()
            
            # Target: Trend variation (relative change in EMA)
            trend_variation = (ema[i] - ema[i - window_size]) / ema[i - window_size]
            
            X.append(X_window)
            y_volatility.append(volatility)
            y_trend.append(trend_variation)
        
        return np.array(X), np.array(y_volatility), np.array(y_trend)

    def build_conv1d_model(self, input_shape):
        from tensorflow.keras import layers, Model

        input_layer = layers.Input(shape=input_shape)

        # Conv1D layers
        x = layers.Conv1D(filters=16, kernel_size=3, activation='relu')(input_layer)      # Filters=16
        x = layers.BatchNormalization()(x)

        x = layers.Conv1D(filters=8, kernel_size=3, activation='relu')(x)               # Filters=8
        x = layers.BatchNormalization()(x)

        # Flatten and Dense layers
        x = layers.Flatten()(x)

        x = layers.Dense(48, activation='relu')(x)                                      # Units=48
        x = layers.BatchNormalization()(x)

        x = layers.Dense(24, activation='relu')(x)                                      # Units=24
        x = layers.BatchNormalization()(x)

        # Output layers
        volatility_output = layers.Dense(1, name='volatility_output')(x)               # Output for Volatility
        trend_output = layers.Dense(1, name='trend_output')(x)                         # Output for Trend

        # Model Definition
        model = Model(inputs=input_layer, outputs=[volatility_output, trend_output])
        print("Conv1D model built successfully with Flatten layer and ~50,000 parameters.")
        model.summary()
        return model


    def process_forex_data(self, forex_files, config, common_start, common_end):
        """
        Processes and aligns multiple Forex rate datasets with the hourly dataset.

        Parameters:
        - forex_files (list): List of file paths for Forex rate datasets.
        - config (dict): Configuration settings.
        - common_start (str or pd.Timestamp): The common start date for alignment.
        - common_end (str or pd.Timestamp): The common end date for alignment.

        Returns:
        - pd.DataFrame: Processed Forex CLOSE features aligned with the hourly dataset.
        """
        print("Processing multiple Forex datasets...")

        # Step 1: Load the hourly dataset from config['input_file']
        hourly_data = load_csv(config['input_file'], config=config)

        # Ensure the timestamp column is named 'datetime'
        if 'DATE_TIME' in hourly_data.columns:
            hourly_data.rename(columns={'DATE_TIME': 'datetime'}, inplace=True)

        # Ensure the timestamp column exists
        if 'datetime' not in hourly_data.columns:
            raise ValueError("Hourly dataset must contain a 'datetime' column.")

        # Parse the 'datetime' column and set as index
        hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'], errors='coerce')
        hourly_data.dropna(subset=['datetime'], inplace=True)
        hourly_data.set_index('datetime', inplace=True)

        # Ensure hourly data has a valid DatetimeIndex
        if not isinstance(hourly_data.index, pd.DatetimeIndex):
            raise ValueError("Hourly data must have a valid DatetimeIndex.")

        print(f"Hourly data index (first 5): {hourly_data.index[:5]}")
        print(f"Hourly data range: {hourly_data.index.min()} to {hourly_data.index.max()}")

        # Initialize an empty DataFrame for storing Forex CLOSE features
        forex_features = pd.DataFrame(index=hourly_data.index)

        # Step 2: Loop through each Forex dataset and process it
        for file_path in forex_files:
            print(f"Processing Forex dataset: {file_path}")

            try:
                # Load the Forex dataset
                forex_data = load_additional_csv(file_path, dataset_type='forex_15m', config=config)

                # Ensure the Forex dataset has a valid DatetimeIndex
                if not isinstance(forex_data.index, pd.DatetimeIndex):
                    raise ValueError(f"Forex data from {file_path} does not have a valid DatetimeIndex.")

                # Validate the presence of the 'CLOSE' column
                if 'CLOSE' not in forex_data.columns:
                    raise ValueError(f"Forex dataset {file_path} does not contain a 'CLOSE' column.")

                print(f"Loaded Forex data (first 5 rows):\n{forex_data.head()}")
                print(f"Forex data index (first 5): {forex_data.index[:5]}")

                # Resample the CLOSE column to hourly frequency
                forex_close_hourly = forex_data['CLOSE'].resample('1H').ffill()
                print(f"Resampled Forex CLOSE data (first 5 rows):\n{forex_close_hourly.head()}")

                # Align with the hourly dataset
                aligned_forex = forex_close_hourly.reindex(hourly_data.index, method='ffill').fillna(0)
                print(f"Aligned Forex CLOSE data (first 5 rows):\n{aligned_forex.head()}")

                # Add the aligned data to the output DataFrame
                column_name = f"{file_path.split('/')[-1].split('.')[0]}_CLOSE"
                forex_features[column_name] = aligned_forex.values

            except Exception as e:
                print(f"Error processing Forex dataset {file_path}: {e}")
                continue

        # Apply common start and end date range filter
        forex_features = forex_features[(forex_features.index >= common_start) & (forex_features.index <= common_end)]

        if forex_features.empty:
            print("Warning: The processed Forex features are empty after applying the date filter.")

        print(f"Processed Forex CLOSE features (first 5 rows):\n{forex_features.head()}")
        return forex_features

    
    def process_high_frequency_data(self, high_freq_data_path, config, common_start, common_end):
        """
        Processes the high-frequency dataset and aligns it with the hourly dataset.

        Parameters:
        - high_freq_data_path (str): Path to the high-frequency dataset.
        - config (dict): Configuration settings.
        - common_start (str or pd.Timestamp): The common start date for alignment.
        - common_end (str or pd.Timestamp): The common end date for alignment.

        Returns:
        - pd.DataFrame: Aligned high-frequency features.
        """
        print(f"Processing high-frequency EUR/USD dataset...")

        # Load and fix the hourly data
        hourly_data = load_and_fix_hourly_data(config['input_file'], config)

        # Ensure hourly data has a valid DatetimeIndex
        if not isinstance(hourly_data.index, pd.DatetimeIndex):
            raise ValueError("Hourly data must have a valid DatetimeIndex.")

        print(f"Hourly data index (first 5): {hourly_data.index[:5]}")
        print(f"Hourly data range: {hourly_data.index.min()} to {hourly_data.index.max()}")

        # Load the high-frequency data
        print(f"Loading high-frequency dataset: {high_freq_data_path}")
        high_freq_data = load_high_frequency_data(high_freq_data_path, config)

        # Ensure high-frequency data has a valid DatetimeIndex
        if not isinstance(high_freq_data.index, pd.DatetimeIndex):
            raise ValueError("High-frequency dataset must have a valid DatetimeIndex.")

        # Validate the presence of the 'CLOSE' column
        if 'CLOSE' not in high_freq_data.columns:
            raise ValueError("High-frequency dataset must contain a 'CLOSE' column.")

        print(f"High-frequency dataset successfully loaded. Index range: {high_freq_data.index.min()} to {high_freq_data.index.max()}")

        # Resample high-frequency data to 15m and 30m
        high_freq_15m = high_freq_data['CLOSE'].resample('15T').ffill()
        high_freq_30m = high_freq_data['CLOSE'].resample('30T').ffill()

        # Debug: Print resampled data summaries
        print("Resampled 15m CLOSE data (first 5 rows):")
        print(high_freq_15m.head())
        print("Resampled 30m CLOSE data (first 5 rows):")
        print(high_freq_30m.head())

        # Construct high-frequency feature DataFrame aligned with hourly data
        high_freq_features = pd.DataFrame(index=hourly_data.index)
        try:
            for i in range(1, config['sub_periodicity_window_size'] + 1):
                high_freq_features[f'CLOSE_15m_tick_{i}'] = (
                    high_freq_15m.shift(i).reindex(hourly_data.index).fillna(0)
                )
                high_freq_features[f'CLOSE_30m_tick_{i}'] = (
                    high_freq_30m.shift(i).reindex(hourly_data.index).fillna(0)
                )
        except Exception as e:
            print(f"Error during high-frequency feature alignment: {e}")
            raise

        # Apply common start and end date range filter
        high_freq_features = high_freq_features[(high_freq_features.index >= common_start) & (high_freq_features.index <= common_end)]

        if high_freq_features.empty:
            print("Warning: The processed high-frequency features are empty after applying the date filter.")

        # Debug: Print processed features
        print("Processed high-frequency features (first 5 rows):")
        print(high_freq_features.head())

        return high_freq_features


    def process_sub_periodicities(self, hourly_data, sub_periodicity_data, window_size):
        """
        Processes sub-periodicity data for integration with the hourly dataset.

        Parameters:
        - hourly_data (pd.DataFrame): The hourly dataset.
        - sub_periodicity_data (pd.DataFrame): Sub-periodicity data (e.g., 15m, 30m).
        - window_size (int): Number of previous ticks to include for each hourly tick.

        Returns:
        - dict: Dictionary with sub-periodicity feature columns.
        """
        print(f"Processing sub-periodicities with window size: {window_size}...")

        # Ensure datetime index for both datasets
        hourly_data.index = pd.to_datetime(hourly_data.index)
        sub_periodicity_data.index = pd.to_datetime(sub_periodicity_data.index)

        sub_periodicity_features = {}

        # Iterate over each hourly tick
        for timestamp in hourly_data.index:
            # Get the current hour's sub-periodicity data
            window = sub_periodicity_data.loc[:timestamp].tail(window_size)

            # Pad with NaN if the window is incomplete
            if len(window) < window_size:
                padding = pd.DataFrame(index=range(window_size - len(window)))
                window = pd.concat([padding, window])

            # Add columns for each tick in the window
            for i, col in enumerate(window.columns):
                sub_periodicity_features[f"{col}_{i+1}"] = window[col].values

        print("Sub-periodicities processed successfully.")
        return sub_periodicity_features


    def process_sp500_data(self, sp500_data_path, config, common_start, common_end):
        """
        Processes S&P 500 data and aligns it with the hourly dataset.

        Parameters:
        - sp500_data_path (str): Path to the S&P 500 dataset.
        - config (dict): Configuration settings.
        - common_start (str or pd.Timestamp): The common start date for alignment.
        - common_end (str or pd.Timestamp): The common end date for alignment.

        Returns:
        - pd.DataFrame: Aligned S&P 500 features or None if processing fails.
        """
        print("Processing S&P 500 data...")

        # Load the hourly dataset
        hourly_data = load_csv(config['input_file'], config=config)

        # Ensure the timestamp column is named 'datetime'
        if 'DATE_TIME' in hourly_data.columns:
            hourly_data.rename(columns={'DATE_TIME': 'datetime'}, inplace=True)

        # Ensure the timestamp column exists
        if 'datetime' not in hourly_data.columns:
            raise ValueError("Hourly dataset must contain a 'datetime' column.")

        # Parse the 'datetime' column and set as index
        hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'], errors='coerce')
        hourly_data.dropna(subset=['datetime'], inplace=True)
        hourly_data.set_index('datetime', inplace=True)

        # Ensure hourly data has a valid DatetimeIndex
        if not isinstance(hourly_data.index, pd.DatetimeIndex):
            raise ValueError("Hourly data must have a valid DatetimeIndex.")

        print(f"Hourly data range: {hourly_data.index.min()} to {hourly_data.index.max()}")

        # Load the S&P 500 dataset
        sp500_data = load_csv(sp500_data_path, config=config)

        # Ensure the 'Date' column is properly parsed and set as index
        if 'Date' in sp500_data.columns:
            sp500_data['Date'] = pd.to_datetime(sp500_data['Date'], errors='coerce')
            sp500_data.dropna(subset=['Date'], inplace=True)
            sp500_data.set_index('Date', inplace=True)

        # Ensure S&P 500 data has a valid DatetimeIndex
        if not isinstance(sp500_data.index, pd.DatetimeIndex):
            raise ValueError("S&P 500 data must have a valid DatetimeIndex.")

        print(f"S&P 500 data range: {sp500_data.index.min()} to {sp500_data.index.max()}")

        # Resample the S&P 500 data to hourly frequency
        sp500_close = sp500_data['Close'].resample('1H').ffill()

        print(f"S&P 500 resampled range: {sp500_close.index.min()} to {sp500_close.index.max()}")

        # Align with the hourly dataset
        aligned_sp500 = sp500_close.reindex(hourly_data.index, method='ffill').fillna(0)

        print(f"Aligned S&P 500 range before filtering: {aligned_sp500.index.min()} to {aligned_sp500.index.max()}")

        # Apply common start and end date range filter
        aligned_sp500 = aligned_sp500[(aligned_sp500.index >= common_start) & (aligned_sp500.index <= common_end)]

        if aligned_sp500.empty:
            print("Warning: The aligned S&P 500 data is empty after applying the date filter.")
            return None

        print(f"Aligned S&P 500 CLOSE data (first 5 rows):\n{aligned_sp500.head()}")

        # Convert the aligned Series to a DataFrame
        aligned_sp500_df = aligned_sp500.to_frame(name='S&P500_Close')

        return aligned_sp500_df






    def process_vix_data(self, vix_data_path, config, common_start, common_end):
        """
        Processes VIX data and aligns it with the hourly dataset.

        Parameters:
        - vix_data_path (str): Path to the VIX dataset.
        - config (dict): Configuration settings.
        - common_start (str or pd.Timestamp): The common start date for alignment.
        - common_end (str or pd.Timestamp): The common end date for alignment.

        Returns:
        - pd.DataFrame: Aligned VIX features.
        """
        print("Processing VIX data...")

        # Load the VIX data
        vix_data = load_additional_csv(vix_data_path, dataset_type='vix', config=config)

        # Ensure the 'date' column is parsed and set as the index
        if 'date' in vix_data.columns:
            vix_data['date'] = pd.to_datetime(vix_data['date'], errors='coerce')
            vix_data.dropna(subset=['date'], inplace=True)
            vix_data.set_index('date', inplace=True)

        # Validate that VIX data has a proper DatetimeIndex
        if not isinstance(vix_data.index, pd.DatetimeIndex):
            raise ValueError("The VIX dataset must have a valid DatetimeIndex as its index.")

        print(f"VIX data range: {vix_data.index.min()} to {vix_data.index.max()}")

        # Extract the 'close' column and resample to hourly resolution
        if 'close' not in vix_data.columns:
            raise ValueError("The VIX dataset must contain a 'close' column.")
        vix_close = vix_data['close'].resample('1H').ffill()

        print(f"Resampled VIX CLOSE data (first 5 rows):\n{vix_close.head()}")

        # Load and validate the hourly data
        hourly_data = load_and_fix_hourly_data(config['input_file'], config)
        print(f"Hourly data range: {hourly_data.index.min()} to {hourly_data.index.max()}")

        # Align with the hourly dataset
        aligned_vix = vix_close.reindex(hourly_data.index, method='ffill').fillna(0)
        print(f"Aligned VIX CLOSE data range: {aligned_vix.index.min()} to {aligned_vix.index.max()}")

        # Apply common start and end date range filter
        aligned_vix = aligned_vix[(aligned_vix.index >= common_start) & (aligned_vix.index <= common_end)]

        if aligned_vix.empty:
            print("Warning: The aligned VIX data is empty after applying the date filter.")
            return None

        print(f"Aligned VIX CLOSE data (first 5 rows):\n{aligned_vix.head()}")

        # Return the aligned VIX features as a DataFrame
        return pd.DataFrame({'vix_close': aligned_vix})




    def clean_and_filter_economic_calendar(self, file_path, hourly_data, config):
        """
        Cleans and filters the economic calendar, aligning it with the hourly dataset.

        Parameters:
        - file_path (str): Path to the economic calendar dataset.
        - hourly_data (pd.DataFrame): The hourly dataset.
        - config (dict): Configuration settings.

        Returns:
        - pd.DataFrame: Processed economic calendar features aligned with the hourly dataset.
        """
        print("Cleaning and filtering economic calendar...")
        
        # Load dataset
        try:
            econ_data = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            raise RuntimeError(f"Error loading file: {e}")

        # Drop rows with too many missing fields
        econ_data.dropna(thresh=len(econ_data.columns) * 0.5, inplace=True)

        # Filter by relevant countries
        relevant_countries = config.get('relevant_countries', ['United States', 'Euro Zone'])
        econ_data = econ_data[econ_data['country'].isin(relevant_countries)]
        print(f"Filtered by relevant countries: {relevant_countries}")

        # Filter by volatility
        if config.get('filter_by_volatility', True):
            econ_data = econ_data[econ_data['volatility'].isin(['Moderate Volatility Expected', 'High Volatility Expected'])]
            print("Filtered by moderate/high volatility.")

        # Handle numeric columns
        numeric_columns = ['Actual', 'Previous', 'Forecast']
        for col in numeric_columns:
            econ_data[col] = pd.to_numeric(econ_data[col], errors='coerce').fillna(econ_data[col].mean())

        # Add derived features
        econ_data['actual_minus_forecast'] = econ_data['Actual'] - econ_data['Forecast']
        econ_data['actual_minus_previous'] = econ_data['Actual'] - econ_data['Previous']

        # Align with hourly dataset
        econ_data['datetime'] = pd.to_datetime(econ_data['event_date'] + ' ' + econ_data['event_time'])
        econ_data.set_index('datetime', inplace=True)

        # Aggregate to hourly resolution
        aggregated = econ_data.resample('1H').mean()

        # Align with the hourly dataset
        aligned = aggregated.reindex(hourly_data.index, method='ffill').fillna(0)
        print("Economic calendar aligned successfully.")
        return aligned

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)
        
    def _save_trained_model(self, filepath, model):
        """
        Save the trained Conv1D model to a file.

        Parameters:
        - filepath (str): Path to save the model.
        - model (keras.Model): Trained Keras model.
        """
        model.save(filepath)
        print(f"[DEBUG] Trained model saved at {filepath}")

