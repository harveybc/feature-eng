import pandas as pd
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro, skew, kurtosis
import numpy as np

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
    
    # Analyze variability and normality
    processed_data = analyze_variability_and_normality(processed_data)

    # Check if distribution_plot is set to True in config
    if config.get('distribution_plot', False):
        plot_distributions(processed_data)

    # Check if correlation_analysis is set to True in config
    if config.get('correlation_analysis', False):
        perform_correlation_analysis(processed_data)

    return processed_data


def analyze_variability_and_normality(data):
    """
    Analyzes each column's variability, normality, and skewness.
    Applies log transformation, z-score normalization, or min-max normalization.
    Returns transformed data with modified column names.
    """
    print("Analyzing variability and normality of each column...")

    transformed_data = data.copy()
    num_columns = len(data.columns)
    num_rows = (num_columns + 3) // 4  # Adjust to show 4 columns of plots
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    plot_index = 0

    for column in list(data.columns):  # Create a list of original columns to iterate over
        original_column = column  # Keep track of the original column name
        
        # Handle missing values by filling with mean for analysis
        if data[column].isna().sum() > 0:
            print(f"{column} contains NaN values. Filling with column mean for analysis.")
            data[column] = data[column].fillna(data[column].mean())

        # Variability (standard deviation)
        variability = np.std(data[column])

        # Normality test using D'Agostino's K^2 and Shapiro-Wilk
        dagostino_result = normaltest(data[column])
        shapiro_result = shapiro(data[column])

        # p-values
        p_value_normaltest = dagostino_result.pvalue
        p_value_shapiro = shapiro_result.pvalue

        # Skewness and Kurtosis
        column_skewness = skew(data[column])
        column_kurtosis = kurtosis(data[column])

        # Log transformation criteria (for skewed columns)
        if abs(column_skewness) > 0.5:
            print(f"Applying log transformation to {column} due to high skewness.")
            transformed_data[f"Log_{column}"] = np.log1p(data[column].abs())  # Log-transformation of absolute values
            column = f"Log_{column}"  # Now refer to the log-transformed column
            column_skewness = skew(transformed_data[column])
            column_kurtosis = kurtosis(transformed_data[column])

        # Refined Normality Decision Logic with Expanded Kurtosis Threshold [-1, 6]
        if -0.5 < column_skewness < 0.5 and -1.0 < column_kurtosis < 6.0:
            print(f"{original_column} is almost normally distributed because skewness is {column_skewness:.5f} in [-0.5, 0.5] and kurtosis is {column_kurtosis:.5f} in [-1, 6]. Applying z-score normalization.")
            transformed_data[f"Standardized_{column}"] = (transformed_data[column] - transformed_data[column].mean()) / transformed_data[column].std()
            transformed_data.drop(columns=[column], inplace=True)  # Drop the old column
        else:
            print(f"{original_column} is not normally distributed because D'Agostino p-value is {p_value_normaltest:.5f} <= 0.05 or Shapiro-Wilk p-value is {p_value_shapiro:.5f} <= 0.05, and skewness is {column_skewness:.5f}, kurtosis is {column_kurtosis:.5f}. Applying min-max normalization.")
            transformed_data[f"Normalized_{column}"] = (transformed_data[column] - transformed_data[column].min()) / (transformed_data[column].max() - transformed_data[column].min())
            transformed_data.drop(columns=[column], inplace=True)

        # Plotting the transformed distribution (use the final transformed column name)
        final_column_name = f"Standardized_{column}" if f"Standardized_{column}" in transformed_data.columns else f"Normalized_{column}"
        sns.histplot(transformed_data[final_column_name], kde=True, ax=axes[plot_index])
        axes[plot_index].set_title(f"{final_column_name} (Transformed)", fontsize=10)
        plot_index += 1

    # Adjust layout and vertical separation
    plt.tight_layout(h_pad=3)
    plt.subplots_adjust(top=0.9, bottom=0.1)  # Ensure sufficient space for titles and labels
    plt.show()

    return transformed_data


# For plotting the distributions after normalization
def plot_distributions(data):
    """
    Plots the distributions of the normalized data.
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, column in enumerate(data.columns):
        ax = axes[idx]
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")

    plt.tight_layout(h_pad=3)
    plt.show()

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
