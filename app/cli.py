import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Engineering System: A tool for generating and selecting features from time-series data with plugin support."
    )

    # Required argument
    parser.add_argument('--input_file', type=str, help='Path to the input CSV file.')

    # Optional arguments
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file (optional).')
    parser.add_argument('--include_original_5', action='store_true', help='Include first 5 columns of input dataset: date, open, low, high, and close values in the output dataset.')

    # Existing arguments
    parser.add_argument('--plugin', type=str, help='Name of the plugin to use for feature generation.')
    parser.add_argument('--correlation_analysis', action='store_true', help='Compute and display Pearson and Spearman correlation matrices.')
    parser.add_argument('--distribution_plot', action='store_true', help='Plot the distributions of the generated features.')
    parser.add_argument('--quiet_mode', action='store_true', help='Suppress output messages to reduce verbosity.')
    parser.add_argument('--save_log', type=str, help='Path to save the current debug log.')
    parser.add_argument('--username', type=str, help='Username for the remote API endpoint.')
    parser.add_argument('--password', type=str, help='Password for the remote API endpoint.')
    parser.add_argument('--remote_save_config', type=str, help='URL of a remote API endpoint for saving the configuration in JSON format.')
    parser.add_argument('--remote_load_config', type=str, help='URL of a remote JSON configuration file to download and execute.')
    parser.add_argument('--remote_log', type=str, help='URL of a remote API endpoint for saving debug variables in JSON format.')
    parser.add_argument('--load_config', type=str, help='Path to load a configuration file.')
    parser.add_argument('--save_config', type=str, help='Path to save the current configuration.')
    parser.add_argument('--headers', action='store_true', help='Input includes headers.')

    # Additional datasets arguments
    parser.add_argument('--high_freq_dataset', type=str, help='Path to the high-frequency dataset (default: set in config.py).')
    parser.add_argument('--sp500_dataset', type=str, help='Path to the S&P 500 daily dataset (default: set in config.py).')
    parser.add_argument('--vix_dataset', type=str, help='Path to the VIX daily dataset (default: set in config.py).')
    parser.add_argument('--economic_calendar', type=str, help='Path to the economic calendar dataset (default: set in config.py).')
    parser.add_argument('--forex_datasets', type=str, nargs='+', help='Paths to the Forex datasets (default: set in config.py).')

    # Additional configuration arguments
    parser.add_argument('--sub_periodicity_window_size', type=int, help='Window size for sub-periodicities (default: 8).')
    parser.add_argument('--output_resample_frequency', type=str, help='Target frequency for resampling (default: 1H).')

    return parser.parse_known_args()  # Returns (args, unknown_args)
