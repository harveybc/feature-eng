# config.py

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

    # New defaults for additional datasets
    'high_freq_dataset': 'tests/data/eurusd_5m_2002_2020.csv',
    'sp_300_daily_dataset': 'tests/data/sp_300_day_1927_2020.csv',
    'vix_daily_dataset': 'tests/data/vix_day_1990_2024.csv',
    'economic_calendar': 'tests/data/economic_calendar_2011_2021.csv'
}
