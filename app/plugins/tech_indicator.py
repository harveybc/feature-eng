import pandas_ta as ta
import pandas as pd
import numpy as np
from app.data_handler import load_csv, write_csv
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
        additional_features = {}

        # Process Economic Calendar Data
        if config.get('economic_calendar'):
            print("Processing economic calendar data...")
            econ_calendar = self.process_economic_calendar_with_attention(
                config['economic_calendar'], data, config
            )
            additional_features.update(econ_calendar.to_dict(orient="list"))

        # Process High-Frequency EUR/USD Dataset
        if config.get('high_freq_dataset'):
            print("Processing high-frequency EUR/USD dataset...")
            high_freq_features = self.process_high_frequency_data(
                config['high_freq_dataset'], data, config
            )
            additional_features.update(high_freq_features.to_dict(orient="list"))

        # Process Forex Datasets
        if config.get('forex_datasets'):
            print("Processing Forex datasets...")
            forex_features = self.process_forex_data(config['forex_datasets'], data, config=config)
            additional_features.update(forex_features)

        # Process S&P 500 Data
        if config.get('sp500_dataset'):
            print("Processing S&P 500 data...")
            sp500_data = load_csv(config['sp500_dataset'], config)
            sp500_features = self.process_sp500_data(sp500_data, data)
            additional_features.update(sp500_features.to_dict(orient="list"))

        # Process VIX Data
        if config.get('vix_dataset'):
            print("Processing VIX data...")
            vix_data = load_csv(config['vix_dataset'], config)
            vix_features = self.process_vix_data(vix_data, data)
            additional_features.update(vix_features.to_dict(orient="list"))

        # Combine into a DataFrame
        additional_features_df = pd.DataFrame(additional_features)
        print(f"Additional features processed: {additional_features_df.columns}")
        return additional_features_df



    def process_high_frequency_data(self, high_freq_file, hourly_data, config):
        """
        Processes high-frequency data (e.g., 15-minute EUR/USD dataset) and aligns it with the hourly dataset.

        Parameters:
        - high_freq_file (str): Path to the high-frequency dataset.
        - hourly_data (pd.DataFrame): Hourly dataset.
        - config (dict): Configuration settings.

        Returns:
        - pd.DataFrame: Processed high-frequency features.
        """
        print(f"Processing high-frequency dataset: {high_freq_file}")

        # Load the high-frequency data
        high_freq_data = load_csv(
            high_freq_file,
            has_headers=True,
            config= config
        )

        # Resample to hourly resolution
        high_freq_data = high_freq_data.resample('1H').mean()

        # Align with the hourly dataset
        aligned_high_freq = high_freq_data.reindex(hourly_data.index, method='ffill').fillna(0)

        print(f"High-frequency dataset aligned with hourly dataset. Shape: {aligned_high_freq.shape}")
        return aligned_high_freq


    def process_economic_calendar(self, econ_data, hourly_data, config):
        """
        Processes economic calendar data into a time-series aligned with hourly data.

        Parameters:
        - econ_data (pd.DataFrame): Economic calendar raw data (no headers).
        - hourly_data (pd.DataFrame): Hourly dataset.
        - config (dict): Configuration settings.

        Returns:
        - pd.DataFrame: Processed economic calendar features aligned to hourly data.
        """
        print("Processing economic calendar...")

        # Assign headers explicitly
        econ_data.columns = [
            'event_date', 'event_time', 'country', 'volatility', 'description',
            'evaluation', 'data_format', 'actual_value', 'forecast_value', 'previous_value'
        ]

        # Combine event date and time into a single datetime column
        econ_data['datetime'] = pd.to_datetime(
            econ_data['event_date'] + ' ' + econ_data['event_time'], format='%Y/%m/%d %H:%M:%S'
        )
        econ_data.set_index('datetime', inplace=True)

        # Filter relevant events (by countries or volatility)
        relevant_countries = config.get('relevant_countries', ['United States', 'Euro Zone'])
        econ_data = econ_data[econ_data['country'].isin(relevant_countries)]

        if config.get('filter_by_volatility', True):
            econ_data = econ_data[econ_data['volatility'].isin(['Moderate Volatility Expected', 'High Volatility Expected'])]

        # Temporal weighting based on event recency (e.g., 24-hour window)
        window_size = config.get('temporal_window_size', 24)
        hourly_features = self.temporal_weighting(econ_data, hourly_data.index, window_size)

        print("Economic calendar processed and aligned.")
        return hourly_features
    

    import pandas as pd

    def process_economic_calendar_with_attention(self, hourly_data, econ_data, window_size):
        """
        Processes economic calendar data with attention weights, aligning it with hourly data.
        
        Args:
            hourly_data (pd.DataFrame): The hourly data to align with the economic calendar.
            econ_data (pd.DataFrame): The economic calendar data.
            window_size (int): The size of the window to consider for attention weighting.
        
        Returns:
            pd.DataFrame: A DataFrame containing the processed features with attention applied.
        """
        # Ensure hourly_data.index is a DatetimeIndex
        try:
            hourly_data.index = pd.to_datetime(hourly_data.index, errors='raise')
        except Exception as e:
            print(f"Error in parsing hourly_data index: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of error

        # Check the ranges of the two datasets
        print(f"Economic data range: {econ_data.index.min()} to {econ_data.index.max()}")
        print(f"Hourly data range: {hourly_data.index.min()} to {hourly_data.index.max()}")

        # Initialize an empty list to store processed features
        processed_features = []

        # Loop through each timestamp in hourly_data
        for timestamp in hourly_data.index:
            print(f"Processing timestamp: {timestamp}")

            # Check if timestamp exists in the economic data index
            if timestamp not in econ_data.index:
                print(f"Warning: {timestamp} is not in the economic data index.")
                continue  # Skip this timestamp if not found in econ_data

            # Get rolling window of events up to the current timestamp
            window = econ_data.loc[:timestamp].tail(window_size)

            # If no events are found in the window, skip this timestamp
            if window.empty:
                print(f"No events found for timestamp {timestamp}. Skipping...")
                continue

            print(f"Found {len(window)} events for timestamp {timestamp}. Applying attention weights...")

            # Apply temporal attention weights (this function should be defined elsewhere)
            try:
                weighted_features = self.apply_attention_weights(window, timestamp)
            except Exception as e:
                print(f"Error in applying attention weights at timestamp {timestamp}: {e}")
                continue  # Skip this timestamp in case of an error in applying weights

            # Add timestamp to the processed features
            weighted_features['timestamp'] = timestamp
            processed_features.append(weighted_features)

        # Check if any features were processed
        if not processed_features:
            print("No features were generated after processing all timestamps.")
            raise ValueError("No processed features were generated. Check the processing loop.")

        # Convert the list of processed features into a DataFrame and set the timestamp as the index
        processed_df = pd.DataFrame(processed_features)

        # Ensure that the 'timestamp' column exists before setting it as the index
        if 'timestamp' not in processed_df.columns:
            print("Error: 'timestamp' column is missing from the processed features.")
            return pd.DataFrame()  # Return empty DataFrame if 'timestamp' column is missing

        processed_df.set_index('timestamp', inplace=True)

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

        result = {}
        for current_time in hourly_index:
            # Define the time window
            window_start = current_time - pd.Timedelta(hours=window_size)
            window_events = events.loc[window_start:current_time]

            # Skip if no events in the window
            if window_events.empty:
                continue

            # Calculate time difference and weights
            window_events['time_diff'] = (current_time - window_events.index).total_seconds() / 3600
            window_events['weight'] = np.exp(-decay_rate * window_events['time_diff'])

            # Aggregate numerical features with weights
            for col in ['actual_minus_forecast', 'actual_minus_previous', 'actual_value', 'forecast_value']:
                result.setdefault(f'{col}_weighted_mean', []).append(
                    (window_events[col] * window_events['weight']).sum() / window_events['weight'].sum()
                )

            # Aggregate volatility using weighted frequency
            volatility_one_hot = pd.get_dummies(window_events['volatility'])
            weighted_volatility = volatility_one_hot.mul(window_events['weight'], axis=0)
            most_relevant_volatility = weighted_volatility.sum().idxmax()
            result.setdefault('volatility_weighted', []).append(most_relevant_volatility)

        # Ensure alignment with hourly index
        for key in result.keys():
            while len(result[key]) < len(hourly_index):
                result[key].append(0)  # Fill with zero if no events are present

        return result


    def process_forex_data(self, forex_files, hourly_data, config):
        """
        Processes and aligns multiple Forex rate datasets with the hourly dataset.

        Parameters:
        - forex_files (list): List of file paths for Forex rate datasets.
        - hourly_data (pd.DataFrame): Hourly dataset.

        Returns:
        - dict: Processed Forex features.
        """
        print("Processing multiple Forex datasets...")

        forex_features = {}
        for file_path in forex_files:
            print(f"Processing Forex dataset: {file_path}")
            
            # Load the Forex data
            forex_data = load_csv(
                file_path,
                has_headers=True,
                config = config
            )
            
            # Resample to hourly resolution
            forex_data = forex_data.resample('1H').mean()
            
            # Align with the hourly dataset
            forex_data = forex_data.reindex(hourly_data.index, method='ffill').fillna(0)

            # Add processed Forex features to the output
            for col in forex_data.columns:
                forex_features[f"{file_path.split('/')[-1].split('.')[0]}_{col}"] = forex_data[col].values

        print(f"Processed Forex datasets: {list(forex_features.keys())}")
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


    def process_sp500_data(self, sp500_data, hourly_data):
        """
        Processes S&P 500 data and aligns it with the hourly dataset.

        Parameters:
        - sp500_data (pd.DataFrame): S&P 500 dataset (daily resolution).
        - hourly_data (pd.DataFrame): Hourly dataset.

        Returns:
        - pd.DataFrame: Aligned S&P 500 features.
        """
        print("Processing S&P 500 data...")

        # Ensure datetime parsing and alignment
        sp500_data['date'] = pd.to_datetime(sp500_data['date'])
        sp500_data.set_index('date', inplace=True)

        # Forward-fill daily data to hourly resolution
        sp500_data = sp500_data.resample('1H').ffill()

        # Align with the hourly dataset
        aligned_sp500 = sp500_data.reindex(hourly_data.index, method='ffill').fillna(0)
        print("S&P 500 data aligned with hourly dataset.")
        return aligned_sp500

    def process_vix_data(self, vix_data, hourly_data):
        """
        Processes VIX data and aligns it with the hourly dataset.

        Parameters:
        - vix_data (pd.DataFrame): VIX dataset (daily resolution).
        - hourly_data (pd.DataFrame): Hourly dataset.

        Returns:
        - pd.DataFrame: Aligned VIX features.
        """
        print("Processing VIX data...")

        # Ensure datetime parsing and alignment
        vix_data['date'] = pd.to_datetime(vix_data['date'])
        vix_data.set_index('date', inplace=True)

        # Forward-fill daily data to hourly resolution
        vix_data = vix_data.resample('1H').ffill()

        # Align with the hourly dataset
        aligned_vix = vix_data.reindex(hourly_data.index, method='ffill').fillna(0)
        print("VIX data aligned with hourly dataset.")
        return aligned_vix


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
