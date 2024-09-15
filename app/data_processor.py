import pandas as pd
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, skew

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
    
    return processed_data

def is_normal(data, alpha=0.05):
    """ 
    Perform Shapiro-Wilk test to check if the data is normally distributed. 
    Returns True if data is normally distributed.
    """
    stat, p_value = shapiro(data)
    return p_value > alpha

def analyze_variability_and_normality(data, column):
    """
    Analyze the variability and normality of a given column.
    
    Returns:
    - dict with information about variability, normality, and transformations.
    """
    result = {
        'high_variability': None,
        'normal_distribution': None,
        'log_transform_applied': False,
        'normalization_used': None
    }

    # Analyze variability: Calculate standard deviation
    std_dev = data[column].std()
    
    # Assume a high threshold for low variability (can be tuned based on domain knowledge)
    result['high_variability'] = std_dev > 0.05  # Example threshold
    
    # Check normality with Shapiro-Wilk test
    result['normal_distribution'] = is_normal(data[column])
    
    # Log transformation for right-skewed data
    if result['normal_distribution'] is False and skew(data[column]) > 0:
        print(f"Log transformation applied to column: {column}")
        data[column] = np.log1p(data[column] - data[column].min() + 1)  # Shift to avoid negative/zero values
        result['log_transform_applied'] = True
    
    # After log transformation, check normality again
    result['normal_distribution'] = is_normal(data[column])
    
    # Apply normalization based on the final distribution
    if result['normal_distribution']:
        print(f"Z-score normalization applied to column: {column}")
        data[column] = (data[column] - data[column].mean()) / data[column].std()
        result['normalization_used'] = 'z-score'
    else:
        print(f"Min-Max normalization applied to column: {column}")
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
        result['normalization_used'] = 'min-max'
    
    return result

def run_feature_engineering_pipeline(config, plugin):
    """
    Runs the feature-engineering pipeline using the plugin.
    Includes variability, normality analysis, and automatic normalization.
    """
    start_time = time.time()

    # Load the data
    print(f"Loading data from {config['input_file']}...")
    data = load_csv(config['input_file'])
    print(f"Data loaded with shape: {data.shape}")

    # Process the data with the plugin
    processed_data = process_data(data, plugin, config)

    # Dictionary to store the analysis results
    analysis_results = {}

    # Perform variability, normality analysis, and normalization for each column
    if config.get('distribution_plot', False):
        print("Analyzing variability, normality, and applying transformations for each technical indicator...")

        for column in processed_data.columns:
            print(f"Analyzing column: {column}")
            analysis_results[column] = analyze_variability_and_normality(processed_data, column)
        
        # Output analysis results
        for column, result in analysis_results.items():
            print(f"\nColumn: {column}")
            print(f"  High Variability: {'Yes' if result['high_variability'] else 'No'}")
            print(f"  Normal Distribution: {'Yes' if result['normal_distribution'] else 'No'}")
            print(f"  Log Transformation Applied: {'Yes' if result['log_transform_applied'] else 'No'}")
            print(f"  Normalization Used: {result['normalization_used']}")

        # Generate distribution plots for each technical indicator
        print("Generating distribution plots for each technical indicator...")
        for column in processed_data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(processed_data[column], kde=True)
            plt.title(f"Distribution of {column}")
            plt.show()

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
