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











    # File: tech_indicator.py

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
                rsi = ta.rsi(data['Close'])
                if rsi is not None:
                    technical_indicators['RSI'] = rsi
                    print(f"RSI calculated with shape: {rsi.shape}")

            elif indicator == 'macd':
                macd = ta.macd(data['Close'])
                if 'MACD_12_26_9' in macd.columns:
                    technical_indicators['MACD'] = macd['MACD_12_26_9']
                print(f"MACD columns returned: {macd.columns}")

            # Add more indicators processing here...
        
        # Create a DataFrame from the calculated technical indicators
        indicator_df = pd.DataFrame(technical_indicators)

        # Debug: Show the calculated technical indicators
        print(f"Calculated technical indicators: {indicator_df.columns}")

        return indicator_df









    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)
