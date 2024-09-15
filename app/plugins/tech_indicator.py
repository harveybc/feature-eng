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



    def adjust_ohlc(self, data):
        """
        Adjust the OHLC columns based on the ohlc_order parameter.
        This method renames 'c1', 'c2', 'c3', and 'c4' to 'Open', 'High', 'Low', and 'Close'
        according to the OHLC order provided, and then ensures the correct columns are selected.

        Parameters:
        data (pd.DataFrame): Input DataFrame with the columns ['c1', 'c2', 'c3', 'c4', ...]

        Returns:
        pd.DataFrame: DataFrame with the OHLC columns renamed and ordered.
        """
        print("Starting adjust_ohlc method...")
        
        # Step 1: Print the current columns before any renaming to verify 'c1' exists
        print(f"Initial data columns: {data.columns}")
        
        # Step 2: Define the renaming order based on OHLC parameter
        print(f"Received OHLC order: {self.params['ohlc_order']}")
        if self.params['ohlc_order'] == 'ohlc':
            ordered_columns = ['Open', 'High', 'Low', 'Close']
        elif self.params['ohlc_order'] == 'olhc':
            ordered_columns = ['Open', 'Low', 'High', 'Close']
        else:
            print("Invalid OHLC order specified. Raising ValueError.")
            raise ValueError("Invalid OHLC order specified")
        
        # Step 3: Define the renaming map
        rename_map = {
            'c1': ordered_columns[0],
            'c2': ordered_columns[1],
            'c3': ordered_columns[2],
            'c4': ordered_columns[3]
        }
        
        # Debugging: Print renaming map before applying
        print(f"Renaming columns map: {rename_map}")
        
        # Step 4: Rename columns and store the result
        try:
            data_renamed = data.rename(columns=rename_map)
        except Exception as e:
            print(f"Error during renaming columns: {e}")
            raise

        # Debugging: Print first few rows of the renamed data
        print("First 5 rows of renamed data:")
        print(data_renamed.head())
        
        # Step 5: Check if all renamed columns are in the dataset
        print(f"Checking if the renamed columns exist in the DataFrame...")
        print(f"Expected columns after renaming: {ordered_columns}")
        missing_columns = [col for col in ordered_columns if col not in data_renamed.columns]
        
        if missing_columns:
            print(f"Error: Missing columns after renaming - {missing_columns}")
            print(f"Available columns: {data_renamed.columns}")
            raise KeyError(f"Missing columns after renaming: {missing_columns}")
        
        # Debugging: Confirm all expected columns are present
        print(f"All expected columns found: {ordered_columns}")

        # Step 6: Return data with columns ordered according to the OHLC order
        try:
            result = data_renamed[ordered_columns]
        except KeyError as e:
            print(f"KeyError when selecting ordered columns: {e}")
            print(f"Available columns: {data_renamed.columns}")
            raise
        
        # Debugging: Print the shape of the final DataFrame and column names
        print(f"Final data shape: {result.shape}")
        print(f"Final column names: {result.columns}")

        return result










    def process(self, data):
        """
        Process the input data by calculating the specified technical indicators using their default parameters.
        
        Parameters:
        data (pd.DataFrame): Input time-series data with renamed 'Open', 'High', 'Low', 'Close', etc.
        
        Returns:
        pd.DataFrame: DataFrame with the calculated technical indicators.
        """
        print(f"Starting process method...")
        
        # Step 1: Print the initial data columns
        print(f"Initial data columns before any processing: {data.columns}")
        
        # Step 2: Adjust the OHLC order of the columns
        try:
            data = self.adjust_ohlc(data)
            print(f"Data columns after adjust_ohlc: {data.columns}")
        except KeyError as e:
            print(f"Error in adjust_ohlc: {e}")
            raise
        
        # Debug: Print the first 50 rows of the data to verify input
        print(f"First 50 rows of data after OHLC adjustment:\n{data.head(50)}")
        
        # Initialize a dictionary to hold all technical indicators
        technical_indicators = {}
        
        # Step 3: Start calculating each indicator and print debug messages
        for indicator in self.params['indicators']:
            print(f"Calculating {indicator}...")
            
            if indicator == 'rsi':
                try:
                    rsi = ta.rsi(data['Close'])  # Using default length of 14
                    technical_indicators['RSI'] = rsi
                    print(f"RSI calculated with shape: {rsi.shape}")
                except Exception as e:
                    print(f"Error calculating RSI: {e}")
            
            elif indicator == 'macd':
                try:
                    macd = ta.macd(data['Close'])  # Using default fast, slow, and signal periods
                    technical_indicators['MACD'] = macd['MACD_12_26_9']
                    technical_indicators['MACD_signal'] = macd['MACDs_12_26_9']
                    print(f"MACD columns returned: {macd.columns}")
                except Exception as e:
                    print(f"Error calculating MACD: {e}")
            
            elif indicator == 'ema':
                try:
                    ema = ta.ema(data['Close'])  # Using default length of 20
                    technical_indicators['EMA'] = ema
                    print(f"EMA calculated with shape: {ema.shape}")
                except Exception as e:
                    print(f"Error calculating EMA: {e}")
            
            # Continue for other indicators, printing the results similarly
            
        # Step 4: Create a DataFrame from the calculated technical indicators
        try:
            indicator_df = pd.DataFrame(technical_indicators)
            print(f"Indicator DataFrame created with columns: {indicator_df.columns}")
        except Exception as e:
            print(f"Error creating DataFrame for technical indicators: {e}")
            raise
        
        # Debug: Print the shape and column names of the final DataFrame
        print(f"Final indicator DataFrame shape: {indicator_df.shape}")
        print(f"Final indicator DataFrame columns: {indicator_df.columns}")
        
        return indicator_df








    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)
