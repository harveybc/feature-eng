import pandas_ta as ta
import pandas as pd

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


    # File: tech_indicator.py

    def adjust_ohlc(self, data):
        """
        Adjust OHLC columns by renaming them according to the expected OHLC order.
        Parameters:
        - data (pd.DataFrame): Input data with generic column names (c1, c2, c3, c4, etc.)

        Returns:
        - pd.DataFrame: Data with columns renamed to 'Open', 'High', 'Low', 'Close'
        """
        print("Starting adjust_ohlc method...")

        # Debug: Show initial columns of the data
        print(f"Initial data columns: {data.columns}")

        # Expected renaming map
        renaming_map = {'c1': 'Open', 'c2': 'High', 'c3': 'Low', 'c4': 'Close'}

        # Check if 'c1' is in the data and ensure it's numeric
        if 'c1' in data.columns:
            data['c1'] = pd.to_numeric(data['c1'], errors='coerce')  # Convert 'c1' to numeric if not
            print(f"Column 'c1' converted to numeric. First few values:\n{data['c1'].head()}")

        # Debug: Show renaming map
        print(f"Renaming columns map: {renaming_map}")

        # Apply renaming
        data_renamed = data.rename(columns=renaming_map)

        # Debug: Show first few rows after renaming
        print(f"First 5 rows of renamed data:\n{data_renamed.head()}")

        # Check if all expected columns exist
        expected_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in expected_columns if col not in data_renamed.columns]

        # If any columns are missing, raise an error
        if missing_columns:
            print(f"Error: Missing columns after renaming - {missing_columns}")
            print(f"Available columns: {data_renamed.columns}")
            raise KeyError(f"Missing columns after renaming: {missing_columns}")

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
        Process additional datasets (e.g., sub-periodicities, S&P, VIX, economic calendar, positional encoding).

        Parameters:
        - data (pd.DataFrame): Full dataset including additional time-series data.
        - config (dict): Configuration settings for processing.

        Returns:
        - pd.DataFrame: DataFrame with additional features.
        """
        print("Processing additional datasets...")

        additional_features = {}

        # Sub-Periodicities
        if config.get('high_freq_dataset'):
            print("Processing high-frequency data...")
            high_freq_data = load_csv(config['high_freq_dataset'])
            high_freq_data.index = pd.to_datetime(high_freq_data['datetime'])
            for col in ['Close']:
                additional_features[f'{col}_15m'] = high_freq_data[col].resample('15T').last().resample('1H').ffill()
                additional_features[f'{col}_30m'] = high_freq_data[col].resample('30T').last().resample('1H').ffill()

        # S&P Features
        if config.get('sp_300_daily_dataset'):
            print("Processing S&P data...")
            sp_data = load_csv(config['sp_300_daily_dataset'])
            sp_data.index = pd.to_datetime(sp_data['datetime'])
            sp_data = sp_data.resample('1H').ffill()
            sp_mean = sp_data['sp_close'].rolling(window=7).mean()
            additional_features['SP_rolling_mean'] = sp_mean

        # VIX Features
        if config.get('vix_daily_dataset'):
            print("Processing VIX data...")
            vix_data = load_csv(config['vix_daily_dataset'])
            vix_data.index = pd.to_datetime(vix_data['datetime'])
            vix_data = vix_data.resample('1H').ffill()
            vix_std = vix_data['vix_close'].rolling(window=7).std()
            additional_features['VIX_rolling_std'] = vix_std

        # Economic Calendar
        if config.get('economic_calendar'):
            print("Processing economic calendar data...")
            econ_data = load_csv(config['economic_calendar'])
            econ_data.index = pd.to_datetime(econ_data['datetime'])
            econ_emb = generate_economic_embeddings(econ_data)
            additional_features.update(econ_emb.to_dict(orient='list'))

        # Positional Encoding
        if 'pos_enc_0' in data:
            print("Adding positional encoding...")
            pos_enc = data.filter(like='pos_enc')
            additional_features.update(pos_enc.to_dict(orient='list'))

        # Combine all additional features into a DataFrame
        additional_features_df = pd.DataFrame(additional_features)

        print(f"Additional features processed: {additional_features_df.columns}")
        return additional_features_df







    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)
