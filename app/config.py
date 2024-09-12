# config.py

DEFAULT_VALUES = {
    'x_train_file': './tests/data/encoder_eval.csv',
    'y_train_file': './tests/data/close-open_eurusd_1h_norm_10y.csv',
    'x_validation_file': None,
    'y_validation_file': None,
    'target_column': None,
    'output_file': './csv_output.csv',
    'save_model': './predictor_model.keras',
    'load_model': None,
    'evaluate_file': './model_eval.csv',
    'plugin': 'ann',
    'time_horizon': 1,
    'threshold_error': 0.00004,
    'remote_log': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'username': None,
    'password': None,
    'load_config': None,
    'save_config': './config_out.json',
    'save_log': './debug_out.json',
    'quiet_mode': False,
    'force_date': False,
    'headers': True,
    'input_offset': 256  
}
