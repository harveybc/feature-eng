DEFAULT_VALUES = {
    'input_file': 'tests/data/eurusd_hour_2005_2020_ohlc',
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
}
