import pandas as pd
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro, skew, kurtosis, anderson
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
    transformed_data = analyze_variability_and_normality(processed_data)

    # Check if distribution_plot is set to True in config
    if config.get('distribution_plot', False):
        plot_distributions(transformed_data)

    # Check if correlation_analysis is set to True in config
    if config.get('correlation_analysis', False):
        perform_correlation_analysis(transformed_data)

    return transformed_data

def analyze_variability_and_normality(data):
    """
    Analyzes each column's variability, normality, and skewness.
    Based on the metrics, applies log transformation, z-score normalization, or min-max normalization.
    Prints a detailed explanation for each decision in one line with all calculated values.
    Returns the transformed data with renamed columns based on the transformation applied.
    """
    print("Analyzing variability and normality of each column...")

    transformed_data = data.copy()

    for column in data.columns:
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

        # Adjustments based on skewness, kurtosis and log transformation for high skew
        if abs(column_skewness) > 0.5:  # Apply log transform for high skewness
            print(f"Applying log transformation to {column} due to high skewness.")
            transformed_data[f"Log_{column}"] = np.log1p(data[column] - data[column].min() + 1)
            column = f"Log_{column}"  # Update the column name after log transformation
            column_skewness = skew(transformed_data[column])
            column_kurtosis = kurtosis(transformed_data[column])

        # Refined Normality Decision Logic
        if -0.5 < column_skewness < 0.5 and -0.5 < column_kurtosis < 6.0:
            print(f"{original_column} is almost normally distributed because skewness is {column_skewness:.5f} in [-0.5, 0.5] and kurtosis is {column_kurtosis:.5f} in [-0.5, 6]. Applying z-score normalization.")
            transformed_data[f"Standardized_{column}"] = (transformed_data[column] - transformed_data[column].mean()) / transformed_data[column].std()
            transformed_data.drop(columns=[column], inplace=True)  # Drop the old column
        else:
            print(f"{original_column} is not normally distributed because D'Agostino p-value is {p_value_normaltest:.5f} <= 0.05 or Shapiro-Wilk p-value is {p_value_shapiro:.5f} <= 0.05, and skewness is {column_skewness:.5f}, kurtosis is {column_kurtosis:.5f}. Applying min-max normalization.")
            transformed_data[f"Normalized_{column}"] = (transformed_data[column] - transformed_data[column].min()) / (transformed_data[column].max() - transformed_data[column].min())
            transformed_data.drop(columns=[column], inplace=True)

    return transformed_data

# For plotting the distributions after normalization with the updated column names
def plot_distributions(data):
    """
    Plots the distributions of the normalized and transformed data.
    """
    num_columns = len(data.columns)
    num_rows = (num_columns + 3) // 4  # Adjust to show 4 columns of plots
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    for idx, column in enumerate(data.columns):
        sns.histplot(data[column], kde=True, ax=axes[idx])
        axes[idx].set_title(f"{column} (Transformed)", fontsize=10)

    # Adjust layout and vertical separation
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
