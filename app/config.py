DEFAULT_VALUES = {
    'input_file': 'tests/data/eurusd_hour_2005_2020_ohlc.csv',
    'output_file': './indicators_output.csv',
    'include_original_5': True,
    'plugin': 'tech_indicator',
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
    'headers': True,

    # Additional datasets
    'high_freq_dataset': 'tests/data/EURUSD-2000-2020-15m.csv',
    'sp500_dataset': 'tests/data/SP500-2000-2020.csv',
    'vix_dataset': 'tests/data/VIX-2000-2020.csv',
    'economic_calendar': 'tests/data/Economic_Calendar.csv',

    # Forex datasets (15-minute data)
    'forex_datasets': [
        'tests/data/USDCAD-2000-2020-15m.csv',
        'tests/data/USDJPY-2000-2020-15m.csv',
        'tests/data/EURCHF-2000-2020-15m.csv',
        'tests/data/AUDUSD-2000-2020-15m.csv'
    ],

    'sub_periodicity_window_size': 8,  # Default window size for sub-periodicities
    'output_resample_frequency': '1H',  # Target frequency for resampling
    'ohlc_columns': ['open', 'high', 'low', 'close'],

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
            'close': 'Close'
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
        }
    },
    'dataset_type': 'forex_15m',  # Example default dataset type

}
