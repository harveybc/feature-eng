import pandas as pd
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, normaltest, shapiro
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
    analyze_variability_and_normality(processed_data)

    # Check if distribution_plot is set to True in config
    if config.get('distribution_plot', False):
        plot_distributions(processed_data)

    # Check if correlation_analysis is set to True in config
    if config.get('correlation_analysis', False):
        perform_correlation_analysis(processed_data)

    return processed_data

def analyze_variability_and_normality(data):
    """
    Analyzes each column's variability, normality (p-value, skewness, kurtosis), 
    and determines appropriate normalization based on thresholds.
    """
    print("Analyzing variability and normality of each column...")
    
    for column in data.columns:
        # Handle missing values by filling with mean for analysis
        if data[column].isna().sum() > 0:
            print(f"{column} contains NaN values. Filling with column mean for analysis.")
            data[column] = data[column].fillna(data[column].mean())

        # Variability (standard deviation)
        variability = np.std(data[column])
        print(f"Variability for {column}: {variability}")

        # Normality test using Shapiro-Wilk and D'Agostino's K^2
        p_value_normaltest = normaltest(data[column]).pvalue
        p_value_shapiro = shapiro(data[column]).pvalue

        # Skewness and Kurtosis
        column_skewness = skew(data[column])
        column_kurtosis = kurtosis(data[column])

        print(f"{column} p-value from D'Agostino's K^2 normality test: {p_value_normaltest}")
        print(f"{column} p-value from Shapiro-Wilk normality test: {p_value_shapiro}")
        print(f"{column} skewness: {column_skewness}")
        print(f"{column} kurtosis: {column_kurtosis}")

        # Normality decision based on p-values and skewness
        if p_value_normaltest > 0.001 or p_value_shapiro > 0.001:  # Less strict
            if -0.5 < column_skewness < 0.5:  # Consider almost normal based on skewness
                print(f"{column} is almost normal. Applying z-score normalization.")
                data[column] = (data[column] - data[column].mean()) / data[column].std()
            else:
                print(f"{column} is right/left skewed. Applying log transformation and z-score normalization.")
                data[column] = np.log1p(data[column] - min(data[column]) + 1)
                data[column] = (data[column] - data[column].mean()) / data[column].std()
        else:
            print(f"{column} is not normally distributed. Applying min-max normalization.")
            data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

    return data

def plot_distributions(data):
    """
    Plots the distribution of each column.
    """
    num_columns = data.shape[1]
    num_rows = (num_columns // 4) + (num_columns % 4 > 0)  # Calculate rows for 4 plots per row
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 12))  # Adjust the figure size

    # Flatten axes array to easily iterate
    axes = axes.flatten()

    for idx, column in enumerate(data.columns):
        sns.histplot(data[column], kde=True, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {column}")

    # Hide empty subplots if the number of columns is not a multiple of 4
    for i in range(len(data.columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def perform_correlation_analysis(data):
    """
    Performs a correlation analysis of the processed data.
    """
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f')
    plt.title("Correlation Matrix of Technical Indicators")
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
