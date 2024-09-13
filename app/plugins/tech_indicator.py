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
        'indicators': ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci', 'bbands', 'williams', 'momentum', 'roc', 'ichimoku'],
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
        Adjusts the input data based on the specified OHLC order by renaming the columns accordingly.

        Parameters:
        data (pd.DataFrame): Input time-series data (excluding the date column).

        Returns:
        pd.DataFrame: Reordered DataFrame with renamed columns (Open, High, Low, Close).
        """
        ohlc_order = self.params['ohlc_order']
        columns_map = {
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close'
        }

        if len(ohlc_order) != 4 or not set(ohlc_order).issubset(set(columns_map.keys())):
            raise ValueError("Invalid 'ohlc_order' format. It must be a string with the exact four characters: 'o', 'h', 'l', 'c'.")

        # Map the current columns (e.g., c1, c2, c3, c4) to the corresponding OHLC names
        current_columns = data.columns[:4]  # Assuming the first 4 columns represent OHLC values (without the date)
        ordered_columns = [columns_map[col] for col in ohlc_order]
        column_mapping = {current_columns[i]: ordered_columns[i] for i in range(4)}

        print(f"Renaming columns to match OHLC order: {column_mapping}")
        data = data.rename(columns=column_mapping)

        return data


    def process(self, data):
        """
        Process the input data by calculating the specified technical indicators.

        Parameters:
        data (pd.DataFrame): Input time-series data with renamed 'Open', 'High', 'Low', 'Close', etc.

        Returns:
        pd.DataFrame: DataFrame with the calculated technical indicators.
        """
        print(f"Calculating technical indicators with short_term={self.params['short_term_period']}, mid_term={self.params['mid_term_period']}, long_term={self.params['long_term_period']}")

        # Adjust the OHLC order of the columns
        data = self.adjust_ohlc(data)

        # Initialize a dictionary to hold all technical indicators
        technical_indicators = {}

        # Calculate each indicator based on the type and the periods
        for indicator in self.params['indicators']:
            if indicator == 'rsi':
                technical_indicators['RSI'] = ta.rsi(data['Close'], length=self.params['short_term_period'])
            
            elif indicator == 'macd':
                macd = ta.macd(data['Close'], fast=self.params['short_term_period'], slow=self.params['mid_term_period'])
                print(f"MACD columns returned: {macd.columns}")  # Add this to inspect the actual column names
                
                # Store each MACD component as a separate column
                technical_indicators['MACD'] = macd.iloc[:, 0]
                technical_indicators['MACD_signal'] = macd.iloc[:, 2]  # Use the signal line
            
            elif indicator == 'ema':
                technical_indicators['EMA'] = ta.ema(data['Close'], length=self.params['mid_term_period'])
            
            elif indicator == 'stoch':
                stoch = ta.stoch(data['High'], data['Low'], data['Close'])
                technical_indicators['StochK'] = stoch['STOCHk_14_3_3']
                technical_indicators['StochD'] = stoch['STOCHd_14_3_3']
            
            elif indicator == 'adx':
                adx = ta.adx(data['High'], data['Low'], data['Close'], length=self.params['mid_term_period'])
                print(f"ADX columns returned: {adx.columns}")  # Print the actual ADX column names
                
                # Store each ADX component as a separate column
                technical_indicators['ADX'] = adx.iloc[:, 0]  # ADX
                technical_indicators['DMP'] = adx.iloc[:, 1]  # D+ (Directional Movement Plus)
                technical_indicators['DMN'] = adx.iloc[:, 2]  # D- (Directional Movement Minus)
            
            elif indicator == 'atr':
                technical_indicators['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=self.params['short_term_period'])
            
            elif indicator == 'cci':
                technical_indicators['CCI'] = ta.cci(data['High'], data['Low'], data['Close'], length=self.params['short_term_period'])
            
            elif indicator == 'bbands':
                bbands = ta.bbands(data['Close'], length=self.params['short_term_period'])
                technical_indicators['BB_Upper'] = bbands['BBU_14_2.0']
                technical_indicators['BB_Lower'] = bbands['BBL_14_2.0']
            
            elif indicator == 'williams':
                technical_indicators['WilliamsR'] = ta.willr(data['High'], data['Low'], data['Close'], length=self.params['short_term_period'])
            
            elif indicator == 'momentum':
                technical_indicators['Momentum'] = ta.mom(data['Close'], length=self.params['short_term_period'])
            
            elif indicator == 'roc':
                technical_indicators['ROC'] = ta.roc(data['Close'], length=self.params['short_term_period'])
            
            elif indicator == 'ichimoku':
                ichimoku = ta.ichimoku(data['High'], data['Low'], data['Close'], tenkan=self.params['short_term_period'], kijun=self.params['mid_term_period'], senkou=self.params['long_term_period'])
                technical_indicators['IchimokuA'] = ichimoku[0]  # Conversion line (Tenkan-sen)
                technical_indicators['IchimokuB'] = ichimoku[1]  # Base line (Kijun-sen)

        # Flatten and split multi-column indicators to separate columns
        for key, value in technical_indicators.items():
            if isinstance(value, pd.DataFrame):
                for column in value.columns:
                    technical_indicators[f"{key}_{column}"] = value[column]
                del technical_indicators[key]

        # Create a DataFrame from the calculated technical indicators
        indicator_df = pd.DataFrame(technical_indicators)

        # Update debug info with the names of the output columns
        self.params['output_columns'] = list(indicator_df.columns)
        print(f"Calculated technical indicators: {self.params['output_columns']}")

        return indicator_df




    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)
