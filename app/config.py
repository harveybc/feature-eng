# Feature Engineering Configuration Export/Import filename
FE_CONFIG_FILENAME = 'fe_config.json'

DEFAULT_VALUES = {
    'output_file': './feature_eng_output.csv',
    'include_original_5': True,
    'feature_plugins': ['base_features', 'technical_features', 'fundamental_features', 'seasonal_features', 'high_frequency_features'],
    'pipeline_plugin': 'default',
    'aligner_plugin': 'default',
    'post_processor_plugin': 'decomposition',

    'correlation_analysis': False,
    'distribution_plot': True,
    'quiet_mode': False,
    'save_log': './debug_log.json',
    'username': None,
    'password': None,
    'remote_save_config': None,
    'remote_load_config': None,
    'remote_log': None,
    'load_config': None,
    'save_config': './output_config.json',
    'fe_config_export': f'./{FE_CONFIG_FILENAME}',  # Export comprehensive feature engineering configuration for replicability
    'headers': True,
    'base_features_max_rows': 1000000,  # Limit for CSV reading

    # Decomposition settings - PHASE 3.1 COMPATIBILITY: STL + WAVELET + MTM
    'decomp_features': ['CLOSE'],  # List of feature names to decompose using STL, wavelet, and MTM methods

    # Additional datasets - REQUIRED FOR PHASE 3 COMPATIBILITY
    'economic_calendar': None,  # Disabled for phase 2.6 compatibility
    'add_log_return': False,     # Enable calculation and inclusion of log return column
    'apply_log_transform': False,  # Disable log transformation analysis to match reference data
    
    # CRITICAL: Enable all forex sub-periodicities for 15m and 30m features
    'forex_datasets': [
        'tests/data/USDCAD-2000-2020-15m.csv',
        'tests/data/USDJPY-2000-2020-15m.csv', 
        'tests/data/EURCHF-2000-2020-15m.csv',
        'tests/data/AUDUSD-2000-2020-15m.csv'
    ],

    # General configurations
    'sub_periodicity_window_size': 8,  # Default window size for sub-periodicities
    'output_resample_frequency': '1H',  # Target frequency for resampling
    'ohlc_columns': ['open', 'high', 'low', 'close'],

    # Header mappings for each dataset type
    'header_mappings': {
        'forex_15m': {
            'datetime': 'DATE_TIME',
            'open': 'OPEN',
            'high': 'HIGH',
            'low': 'LOW',
            'close': 'CLOSE'
        },
        'sp500': {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adj_close': 'Adj Close',
            'volume': 'Volume'
        },
        'vix': {
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        },
        'economic_calendar': {
            'event_date': 'c1',
            'event_time': 'c2',
            'country': 'c3',
            'volatility': 'c4',
            'description': 'c5',
            'evaluation': 'c6',
            'data_format': 'c7',
            'actual': 'c8',
            'forecast': 'c9',
            'previous': 'c10'
        },
        'hourly': {
        'datetime': 'datetime',  # The column name for the datetime index
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close'
        },
    },

    # Dataset types for loading
    'dataset_type': 'forex_15m',  # Default dataset type
    'dataset_types': {
        'forex': ['forex_15m'],
        'stock': ['sp500'],
        'volatility': ['vix'],
        'calendar': ['economic_calendar']
    },

    # Economic calendar settings
    'calendar_window_size': 128,  # Default sliding window for events
    'calendar_window_size_divisor': 5,  # Divisor of the main window size to calculate the short-term trend and the volatility 
    'temporal_decay': 0.1,  # Decay rate for temporal weighting
    'relevant_countries': ['United States', 'Euro Zone'],  # Filter countries
    'filter_by_volatility': True,  # Only include moderate/high volatility events
    'default_positional_encoding_dim': 8  # Dimension for positional encoding
}
