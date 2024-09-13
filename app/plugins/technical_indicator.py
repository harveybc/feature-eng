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
        'indicators': ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci', 'bbands', 'williams', 'momentum', 'roc', 'pvt', 'cmf', 'obv', 'ichimoku'],
    }

    # Debug variables to track important parameters and results
    plugin_debug_vars = ['short_term_period', 'mid_term_period', 'long_term_period', 'output_columns']

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var, None) for var in self.plugin_debug_vars}

    def process(self, data):
        """
        Process the input data by calculating the specified technical indicators.

        Parameters:
        data (pd.DataFrame): Input time-series data with 'Close', 'High', 'Low', 'Volume', etc.

        Returns:
        pd.DataFrame: DataFrame with the calculated technical indicators.
        """
        print(f"Calculating technical indicators with short_term={self.params['short_term_period']}, mid_term={self.params['mid_term_period']}, long_term={self.params['long_term_period']}")

        # Initialize a dictionary to hold all technical indicators
        technical_indicators = {}

        # Calculate each indicator based on the type and the periods
        for indicator in self.params['indicators']:
            if indicator == 'rsi':
                technical_indicators['RSI'] = ta.rsi(data['Close'], length=self.params['short_term_period'])
            elif indicator == 'macd':
                macd = ta.macd(data['Close'], fast=self.params['short_term_period'], slow=self.params['mid_term_period'])
                technical_indicators['MACD'] = macd['MACD_12_26_9']
                technical_indicators['MACD_signal'] = macd['MACDs_12_26_9']
            elif indicator == 'ema':
                technical_indicators['EMA'] = ta.ema(data['Close'], length=self.params['mid_term_period'])
            elif indicator == 'stoch':
                stoch = ta.stoch(data['High'], data['Low'], data['Close'])
                technical_indicators['StochK'] = stoch['STOCHk_14_3_3']
                technical_indicators['StochD'] = stoch['STOCHd_14_3_3']
            elif indicator == 'adx':
                technical_indicators['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=self.params['mid_term_period'])['ADX_14']
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
            elif indicator == 'pvt':
                technical_indicators['PVT'] = ta.pvt(data['Close'], data['Volume'])
            elif indicator == 'cmf':
                technical_indicators['CMF'] = ta.cmf(data['High'], data['Low'], data['Close'], data['Volume'], length=self.params['short_term_period'])
            elif indicator == 'obv':
                technical_indicators['OBV'] = ta.obv(data['Close'], data['Volume'])
            elif indicator == 'ichimoku':
                ichimoku = ta.ichimoku(data['High'], data['Low'], data['Close'], tenkan=self.params['short_term_period'], kijun=self.params['mid_term_period'], senkou=self.params['long_term_period'])
                technical_indicators['IchimokuA'] = ichimoku[0]  # Conversion line (Tenkan-sen)
                technical_indicators['IchimokuB'] = ichimoku[1]  # Base line (Kijun-sen)

        # Create a DataFrame from the calculated technical indicators
        indicator_df = pd.DataFrame(technical_indicators)

        # Update debug info with the names of the output columns
        self.params['output_columns'] = list(indicator_df.columns)
        print(f"Calculated technical indicators: {self.params['output_columns']}")

        return indicator_df

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)
