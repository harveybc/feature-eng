import pandas_ta as ta
import pandas as pd
import numpy as np
from app.data_handler import load_csv, write_csv, load_additional_csv,load_sp500_csv
from app.positional_encoding import generate_positional_encoding

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

        # Adjust the OHLC order of the columns
        data = self.adjust_ohlc(data)

        # Debug: Show the columns after adjustment
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
        indicator_df = pd.DataFrame(technical_indicators)

        # Debug: Show the calculated technical indicators
        print(f"Calculated technical indicators: {indicator_df.columns}")

        return indicator_df



    def process_additional_datasets(self, data, config):
        """
        Processes additional datasets (e.g., economic calendar, sub-periodicities, S&P 500, VIX, and Forex pairs).

        Parameters:
        - data (pd.DataFrame): Full dataset (hourly resolution).
        - config (dict): Configuration settings.

        Returns:
        - pd.DataFrame: Additional features DataFrame.
        """
        print("Processing additional datasets...")
            # Ensure hourly_data has a valid DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index, errors='coerce')
                if data.index.isna().any():
                    raise ValueError("Hourly data contains invalid datetime values.")
            except Exception as e:
                raise ValueError(f"Failed to set DatetimeIndex for hourly data: {e}")
        additional_features = {}

        # Process Forex Datasets
        if config.get('forex_datasets'):
            forex_features = self.process_forex_data(config['forex_datasets'], config=config)
            additional_features.update(forex_features)

        # Process S&P 500 Data
        if config.get('sp500_dataset'):
            print("Processing S&P 500 data...")
            sp500_features = self.process_sp500_data(config['sp500_dataset'], data, config=config)
            additional_features.update(sp500_features.to_dict(orient="list"))

            
        # Process VIX Data
        if config.get('vix_dataset'):
            print("Processing VIX data...")
            vix_features = self.process_vix_data(config['vix_dataset'], data, config=config)
            additional_features.update(vix_features.to_dict(orient="list"))

        # Process High-Frequency EUR/USD Dataset
        if config.get('high_freq_dataset'):
            print("Processing high-frequency EUR/USD dataset...")
            high_freq_features = self.process_high_frequency_data(
                config['high_freq_dataset'], data, config
            )
            additional_features.update(high_freq_features.to_dict(orient="list"))


        # Process Economic Calendar Data
        if config.get('economic_calendar'):
            print("Processing economic calendar data...")
            econ_calendar = self.process_economic_calendar(
                config['economic_calendar'], data, config
            )
            additional_features.update(econ_calendar.to_dict(orient="list"))

        # Combine into a DataFrame
        additional_features_df = pd.DataFrame(additional_features)
        print(f"Additional features processed: {additional_features_df.columns}")
        return additional_features_df




    def process_high_frequency_data(self, hf_data_path, hourly_data, config):
        """
        Processes high-frequency EUR/USD data (e.g., 15m and 30m tick data) and aligns it with hourly data.

        Parameters:
        - hf_data_path (str): Path to the high-frequency EUR/USD dataset.
        - hourly_data (pd.DataFrame): Hourly dataset.
        - config (dict): Configuration settings.

        Returns:
        - pd.DataFrame: Aligned high-frequency features with previous 8 ticks.
        """
        print(f"Processing high-frequency dataset: {hf_data_path}")

        # Load high-frequency data
        high_freq_data = load_additional_csv(hf_data_path, dataset_type='forex_15m', config=config)

        # Ensure 'CLOSE' column exists
        if 'CLOSE' not in high_freq_data.columns:
            raise KeyError(f"The high-frequency EUR/USD data must contain a 'CLOSE' column, but found {high_freq_data.columns}")

        print(f"Loaded high-frequency data columns: {list(high_freq_data.columns)}")
        print(f"First 5 rows of high-frequency data:\n{high_freq_data.head()}")

        # Resample to 15m and 30m periodicity
        high_freq_15m = high_freq_data[['CLOSE']].resample('15T').mean()
        high_freq_30m = high_freq_data[['CLOSE']].resample('30T').mean()

        print(f"Resampled 15m CLOSE data (first 5 rows):\n{high_freq_15m.head()}")
        print(f"Resampled 30m CLOSE data (first 5 rows):\n{high_freq_30m.head()}")

        # Create columns for the last 8 ticks of each periodicity
        features = {}
        for i in range(1, 9):
            features[f'CLOSE_15m_tick_{i}'] = high_freq_15m['CLOSE'].shift(i).reindex(hourly_data.index, method='ffill').fillna(0)
            features[f'CLOSE_30m_tick_{i}'] = high_freq_30m['CLOSE'].shift(i).reindex(hourly_data.index, method='ffill').fillna(0)

        # Combine features into a DataFrame
        high_freq_features = pd.DataFrame(features, index=hourly_data.index)

        print(f"Processed high-frequency features (first 5 rows):\n{high_freq_features.head()}")
        print(f"High-frequency features processed successfully. Shape: {high_freq_features.shape}")

        return high_freq_features



    def process_economic_calendar(self, econ_data_path, hourly_data, config):
        """
        Processes economic calendar data into a time-series aligned with hourly data.

        Parameters:
        - econ_data_path (str): Path to the economic calendar CSV file.
        - hourly_data (pd.DataFrame): Hourly dataset.
        - config (dict): Configuration settings.

        Returns:
        - pd.DataFrame: Processed economic calendar features aligned to hourly data.
        """
        print("Processing economic calendar data...")

        econ_columns = [
            'Event date', 'Event time', 'Country', 'Volatility',
            'Description', 'Evaluation', 'Data format',
            'Actual', 'Forecast', 'Previous'
        ]

        # Load the economic calendar data
        econ_data = load_csv(
            econ_data_path,
            has_headers=False,  # Economic calendar has no headers
            columns=econ_columns
        )

        # Combine event date and time into a single datetime column
        econ_data['datetime'] = pd.to_datetime(
            econ_data['Event date'] + ' ' + econ_data['Event time'],
            format='%Y/%m/%d %H:%M:%S',
            errors='coerce'
        )
        econ_data.set_index('datetime', inplace=True)

        # Process and align the economic calendar data
        aligned_econ = self.events_to_hourly_timeseries(
            econ_data, hourly_data.index, config['window_size'], config['temporal_decay']
        )

        print("Economic calendar processed successfully.")
        return pd.DataFrame(aligned_econ, index=hourly_data.index)


    

    def process_economic_calendar_with_attention(self, econ_data_path, hourly_data, config):
        """
        Processes economic calendar using temporal attention with positional encoding.

        Parameters:
        - econ_data_path (str): Path to the economic calendar CSV file.
        - hourly_data (pd.DataFrame): Hourly dataset.
        - config (dict): Configuration settings for processing.

        Returns:
        - pd.DataFrame: Aligned event impact features with positional encodings.
        """
        print("Processing economic calendar with attention and positional encoding...")

        # Column names for the economic calendar dataset
        column_names = [
            'event_date', 'event_time', 'country', 'volatility', 'description',
            'evaluation', 'data_format', 'actual', 'forecast', 'previous'
        ]

        # Load the economic calendar dataset
        econ_data = pd.read_csv(
            econ_data_path,
            header=None,
            names=column_names
        )

        # Parse datetime and set as index
        econ_data['datetime'] = pd.to_datetime(
            econ_data['event_date'] + ' ' + econ_data['event_time'],
            format='%Y/%m/%d %H:%M:%S',
            errors='coerce'
        )
        econ_data.dropna(subset=['datetime'], inplace=True)
        econ_data.set_index('datetime', inplace=True)

        # Ensure econ_data.index is a DatetimeIndex
        if not isinstance(econ_data.index, pd.DatetimeIndex):
            raise ValueError("econ_data index is not a valid DatetimeIndex after datetime parsing.")

        # Ensure hourly_data.index is a DatetimeIndex
        hourly_data.index = pd.to_datetime(hourly_data.index, errors='coerce')

        # Validate hourly_data index type
        if not isinstance(hourly_data.index, pd.DatetimeIndex):
            raise ValueError("hourly_data index is not a valid DatetimeIndex.")

        # Generate positional encodings for the events
        max_time = hourly_data.index.max()  # Get maximum timestamp
        econ_data['position'] = econ_data.index.map(lambda t: (max_time - t).total_seconds() / 3600)  # Hours from max_time
        num_features = config.get('positional_encoding_dim', 8)  # Positional encoding dimension
        econ_data_positional_encoding = generate_positional_encoding(len(econ_data), num_features)
        positional_encoding_df = pd.DataFrame(
            econ_data_positional_encoding,
            index=econ_data.index,
            columns=[f'pos_enc_{i}' for i in range(num_features)]
        )
        econ_data = pd.concat([econ_data, positional_encoding_df], axis=1)

        # Filter relevant countries and volatility levels
        relevant_countries = config.get('relevant_countries', ['United States', 'Euro Zone'])
        econ_data = econ_data[econ_data['country'].isin(relevant_countries)]
        econ_data = econ_data[econ_data['volatility'].isin(['Moderate Volatility Expected', 'High Volatility Expected'])]

        # Temporal weighting mechanism
        def apply_attention_weights(window, current_time):
            """
            Assigns weights to events in the window based on their temporal proximity.
            """
            time_diff = (current_time - window.index).total_seconds() / 3600  # Convert to hours
            weights = np.exp(-time_diff / config.get('temporal_decay', 24))  # Exponential decay
            weighted_values = window[['actual', 'forecast', 'previous']].apply(pd.to_numeric, errors='coerce').multiply(weights, axis=0)
            return weighted_values.sum()

        # Rolling window processing
        window_size = config.get('event_window_size', 8)
        processed_features = []

        for timestamp in hourly_data.index:
            # Convert timestamp to datetime if necessary
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.Timestamp(timestamp)

            # Ensure timestamp is within the range of econ_data.index
            if timestamp not in econ_data.index:
                print(f"Warning: {timestamp} is not in the economic data index.")
                continue  # Skip this timestamp if not found in econ_data

            # Get rolling window of events up to the current timestamp
            window = econ_data.loc[:timestamp].tail(window_size)

            if not window.empty:
                weighted_features = apply_attention_weights(window, timestamp)
            else:
                # Fill with zeros if no events in the window
                weighted_features = pd.Series(index=['actual', 'forecast', 'previous'], dtype='float64').fillna(0)

            # Add timestamp for alignment
            weighted_features['timestamp'] = timestamp
            processed_features.append(weighted_features)

        # Check the structure of processed_features before creating DataFrame
        if not processed_features:
            raise ValueError("No processed features were generated. Check the processing loop.")

        # Create DataFrame from processed features
        processed_df = pd.DataFrame(processed_features)

        # Ensure the 'timestamp' column is present
        if 'timestamp' not in processed_df.columns:
            raise KeyError("'timestamp' column is missing from processed features.")

        processed_df.set_index('timestamp', inplace=True)

        # Align with the hourly dataset
        processed_df = processed_df.reindex(hourly_data.index).fillna(0)

        print(f"Processed economic calendar features with shape: {processed_df.shape}")
        return processed_df



    
    def compute_temporal_impact(self, econ_data, hourly_index, impact_window):
        """
        Computes the temporal impact of economic events within a given window.

        Parameters:
        - econ_data (pd.DataFrame): Economic calendar data.
        - hourly_index (pd.DatetimeIndex): Target hourly index for synchronization.
        - impact_window (int): Window size in hours for temporal impact calculation.

        Returns:
        - dict: Dictionary with temporal impact features.
        """
        print("Computing temporal impact...")

        temporal_impact = {}

        for timestamp in hourly_index:
            relevant_events = econ_data.loc[timestamp - pd.Timedelta(hours=impact_window):timestamp]

            # Weight by recency (e.g., exponential decay)
            weights = np.exp(-((timestamp - relevant_events.index).total_seconds() / 3600) / impact_window)

            # Weighted averages for numerical features
            for col in ['actual_value', 'forecast_value', 'previous_value']:
                if col in relevant_events:
                    weighted_avg = np.sum(relevant_events[col] * weights) / np.sum(weights)
                    temporal_impact[f"{col}_impact"] = temporal_impact.get(f"{col}_impact", []) + [weighted_avg]

            # Aggregate one-hot encoded categorical features
            for cat_col in ['country', 'volatility', 'data_format']:
                if cat_col in relevant_events:
                    counts = relevant_events[cat_col].value_counts(normalize=True) * weights.sum()
                    for value, count in counts.items():
                        feature_name = f"{cat_col}_{value}_impact"
                        temporal_impact[feature_name] = temporal_impact.get(feature_name, []) + [count]

        print("Temporal impact computed successfully.")
        return temporal_impact



    def events_to_hourly_timeseries(self, events, hourly_index, window_size, decay_rate):
        """
        Converts sparse event data into a continuous hourly time series.

        Parameters:
        - events (pd.DataFrame): Event dataset with datetime index.
        - hourly_index (pd.DatetimeIndex): Hourly timestamps for alignment.
        - window_size (int): Number of hours to look back for events.
        - decay_rate (float): Decay rate for time-based weighting.

        Returns:
        - dict: Hourly-aligned features dictionary.
        """
        print("Transforming sparse events to dense hourly features...")

        # Ensure numeric columns are correctly converted
        for col in ['actual_minus_forecast', 'actual_minus_previous', 'actual_value', 'forecast_value']:
            events[col] = pd.to_numeric(events[col], errors='coerce')

        # Initialize the result dictionary with empty lists for each feature
        result = {f'{col}_weighted_mean': [] for col in ['actual_minus_forecast', 'actual_minus_previous', 'actual_value', 'forecast_value']}
        result['volatility_weighted'] = []

        for current_time in hourly_index:
            # Define the time window
            window_start = current_time - pd.Timedelta(hours=window_size)
            window_events = events.loc[window_start:current_time]

            # Skip if no events in the window
            if window_events.empty:
                # Append default values (zeros) for this timestamp
                for col in ['actual_minus_forecast', 'actual_minus_previous', 'actual_value', 'forecast_value']:
                    result[f'{col}_weighted_mean'].append(0)
                result['volatility_weighted'].append(0)
                continue

            # Calculate time difference and weights
            time_diff = (current_time - window_events.index).total_seconds() / 3600  # Convert to hours
            weights = np.exp(-decay_rate * time_diff)

            # Ensure weights sum is non-zero
            weight_sum = weights.sum()
            if weight_sum == 0:
                weight_sum = 1  # Avoid division by zero

            # Aggregate numerical features with weights
            for col in ['actual_minus_forecast', 'actual_minus_previous', 'actual_value', 'forecast_value']:
                weighted_mean = (window_events[col] * weights).sum() / weight_sum
                result[f'{col}_weighted_mean'].append(weighted_mean)

            # Aggregate volatility using weighted mode
            volatility_counts = window_events['volatility'].value_counts(normalize=True) * weight_sum
            most_relevant_volatility = volatility_counts.idxmax() if not volatility_counts.empty else 0
            result['volatility_weighted'].append(most_relevant_volatility)

        # Ensure alignment with hourly index
        for key, values in result.items():
            while len(values) < len(hourly_index):
                values.append(0)  # Fill with zero if no events are present

        print("Sparse events successfully transformed to dense hourly features.")
        return result



    def process_forex_data(self, forex_files, config):
        """
        Processes and aligns multiple Forex rate datasets with the hourly dataset.

        Parameters:
        - forex_files (list): List of file paths for Forex rate datasets.
        - config (dict): Configuration settings.

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

            # Load the Forex dataset
            forex_data = load_additional_csv(file_path, dataset_type='forex_15m', config=config)
            print(f"Loaded Forex data (first 5 rows):\n{forex_data.head()}")
            print(f"Forex data index (first 5): {forex_data.index[:5]}")

            # Ensure the Forex dataset has a valid DatetimeIndex
            if not isinstance(forex_data.index, pd.DatetimeIndex):
                raise ValueError(f"Forex data from {file_path} does not have a valid DatetimeIndex.")

            # Resample the CLOSE column to hourly frequency
            forex_close_hourly = forex_data['CLOSE'].resample('1H').ffill()
            print(f"Resampled Forex CLOSE data (first 5 rows):\n{forex_close_hourly.head()}")

            # Align with the hourly dataset
            aligned_forex = forex_close_hourly.reindex(hourly_data.index, method='ffill').fillna(0)
            print(f"Aligned Forex CLOSE data (first 5 rows):\n{aligned_forex.head()}")

            # Add the aligned data to the output DataFrame
            column_name = f"{file_path.split('/')[-1].split('.')[0]}_CLOSE"
            forex_features[column_name] = aligned_forex.values

        print(f"Processed Forex CLOSE features (first 5 rows):\n{forex_features.head()}")
        return forex_features


    def align_datasets(self, base_data, additional_datasets):
        """
        Align multiple datasets by their common date range and base index.

        Parameters:
        - base_data (pd.DataFrame): Base dataset (e.g., EUR/USD hourly).
        - additional_datasets (list): List of additional datasets to align.

        Returns:
        - list: List of aligned datasets.
        """
        print("Aligning datasets by common date range...")
        common_start = base_data.index.min()
        common_end = base_data.index.max()

        aligned_datasets = []
        for dataset in additional_datasets:
            dataset = dataset[(dataset.index >= common_start) & (dataset.index <= common_end)]
            aligned_dataset = dataset.reindex(base_data.index, method='ffill').fillna(0)
            aligned_datasets.append(aligned_dataset)

        print("Datasets aligned successfully.")
        return aligned_datasets


    


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


    def process_sp500_data(self, sp500_data_path, hourly_data, config=None):
        """
        Processes S&P 500 data and aligns it with the hourly dataset.

        Parameters:
        - sp500_data_path (str): Path to the S&P 500 dataset.
        - hourly_data (pd.DataFrame): Hourly dataset.
        - config (dict): Configuration settings (optional).

        Returns:
        - pd.DataFrame: Aligned S&P 500 CLOSE values as a single column DataFrame.
        """
        print("Processing S&P 500 data...")

        # Load the S&P 500 data using a dedicated loader
        sp500_data = load_sp500_csv(sp500_data_path)

        # Ensure the 'Date' column is properly parsed and the index is set
        if not isinstance(sp500_data.index, pd.DatetimeIndex):
            raise ValueError("S&P 500 data must have a valid DatetimeIndex.")

        # Extract the 'Close' column and resample to hourly resolution
        sp500_close = sp500_data['Close'].resample('1H').ffill()

        # Align with the hourly dataset's index
        aligned_sp500 = sp500_close.reindex(hourly_data.index, method='ffill').fillna(0)

        # Convert the result into a DataFrame with a single column
        aligned_sp500_df = pd.DataFrame({'S&P500_CLOSE': aligned_sp500})

        print("S&P 500 data aligned with hourly dataset successfully.")
        print(f"First 5 rows of aligned S&P 500 CLOSE data:\n{aligned_sp500_df.head()}")

        return aligned_sp500_df



    def process_vix_data(self, vix_data_path, hourly_data, config):
        """
        Processes VIX data and aligns it with the hourly dataset.

        Parameters:
        - vix_data_path (str): Path to the VIX dataset.
        - hourly_data (pd.DataFrame): Hourly dataset.
        - config (dict): Configuration settings.

        Returns:
        - pd.DataFrame: Aligned VIX CLOSE feature.
        """
        print("Processing VIX data...")

        # Load the VIX data
        vix_data = load_additional_csv(vix_data_path, dataset_type='vix', config=config)

        # Ensure the 'close' column exists
        if 'close' not in vix_data.columns:
            raise KeyError("The VIX dataset must contain a 'close' column.")

        # Resample to hourly resolution
        vix_close = vix_data['close'].resample('1H').ffill()

        # Align with the hourly dataset
        aligned_vix_close = vix_close.reindex(hourly_data.index, method='ffill').fillna(0)

        # Convert to DataFrame
        vix_features_df = pd.DataFrame({'VIX_CLOSE': aligned_vix_close}, index=hourly_data.index)

        print(f"Processed VIX CLOSE feature (first 5 rows):\n{vix_features_df.head()}")
        print(f"VIX feature processed successfully. Shape: {vix_features_df.shape}")
        return vix_features_df






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
