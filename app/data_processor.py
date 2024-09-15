import pandas as pd
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def normalize_column(data_column):
    """
    Normalizes a column based on its distribution:
    - Applies z-score normalization if the column is normally distributed.
    - Applies min-max normalization if the column is not normally distributed.
    - Applies log transformation if the column has a right-tailed distribution.
    """
    # Test for normality using the Shapiro-Wilk test (p-value threshold of 0.05)
    stat, p_value = shapiro(data_column)
    skewness = skew(data_column)

    print(f"Column {data_column.name}: Shapiro-Wilk p-value = {p_value}, skewness = {skewness}")

    # Log transformation for right-skewed distributions
    if skewness > 1:
        print(f"Applying log transformation to {data_column.name} due to right skewness.")
        data_column = np.log1p(data_column - data_column.min() + 1)  # Ensure positive values for log transformation

    # Re-test normality after log transformation if applied
    stat, p_value = shapiro(data_column)
    if p_value > 0.05:
        print(f"{data_column.name} is normally distributed. Applying z-score normalization.")
        # Apply z-score normalization
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_column.values.reshape(-1, 1)).flatten()
    else:
        print(f"{data_column.name} is not normally distributed. Applying min-max normalization.")
        # Apply min-max normalization
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data_column.values.reshape(-1, 1)).flatten()

    return pd.Series(normalized_data, name=data_column.name)


def process_data(data, plugin, config):
    """
    Processes the data using the specified plugin and performs additional analysis
    if distribution_plot or correlation_analysis are set to True.
    """
    print("Processing data using plugin...")

    # Keep the date column separate
    if 'date' in data.columns:
        date_column = data['date']
    else:
        date_column = data.index

    # Debugging: Show the data columns before processing
    print(f"Data columns before processing: {data.columns}")

    # Select OHLC columns by name explicitly (or the expected columns)
    ohlc_columns = ['c1', 'c2', 'c3', 'c4']  # These are placeholders for OHLC
    if all(col in data.columns for col in ohlc_columns):
        numeric_data = data[ohlc_columns]
    else:
        raise KeyError(f"Missing expected OHLC columns: {ohlc_columns}")

    # Ensure input data is numeric
    numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Use the plugin to process the numeric data (e.g., feature extraction)
    processed_data = plugin.process(numeric_data)

    # Debugging message to confirm the shape of the processed data
    print(f"Processed data shape: {processed_data.shape}")

    # Apply normalization to the processed data
    print("Normalizing processed data based on distribution characteristics...")
    for column in processed_data.columns:
        processed_data[column] = normalize_column(processed_data[column])

    # Check if distribution_plot is set to True in config
    if config.get('distribution_plot', False):
        print("Generating distribution plots for each technical indicator...")
        for column in processed_data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(processed_data[column], kde=True)
            plt.title(f"Distribution of {column}")
            plt.show()

    # Check if correlation_analysis is set to True in config
    if config.get('correlation_analysis', False):
        print("Performing correlation analysis...")
        corr_matrix = processed_data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f')
        plt.title("Correlation Matrix of Technical Indicators")
        plt.show()

    return processed_data


def run_feature_engineering_pipeline(config, plugin):
    """
    Runs the feature-engineering pipeline using the plugin.
    """
    start_time = time.time()

    # Load the data
    print(f"Loading data from {config['input_file']}...")
    data = load_csv(config['input_file'])
    print(f"Data loaded with shape: {data.shape}")

    # Process the data
    processed_data = process_data(data, plugin, config)

    # Save the processed data to the output file if specified
    if config['output_file']:
        processed_data.to_csv(config['output_file'], index=False)
        print(f"Processed data saved to {config['output_file']}.")

    # Save final configuration and debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {
        'execution_time': float(execution_time)
    }

    # Save debug info if specified
    if config.get('save_log'):
        save_debug_info(debug_info, config['save_log'])
        print(f"Debug info saved to {config['save_log']}.")

    # Remote log debug info and config if specified
    if config.get('remote_log'):
        remote_log(config, debug_info, config['remote_log'], config['username'], config['password'])
        print(f"Debug info saved to {config['remote_log']}.")

    print(f"Execution time: {execution_time} seconds")
