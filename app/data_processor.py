import pandas as pd
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro, skew, kurtosis, anderson
import numpy as np
import warnings
# Suppress the specific UserWarning from scipy.stats about p-values for large datasets
warnings.filterwarnings("ignore", message="p-value may not be accurate for N > 5000.")


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
    
    # analyze_variability_and_normality
    transformed_data = analyze_variability_and_normality(processed_data, config)
    # If the paarameter include_original_5 in the config is set to True, include the original firsst 5 columns(starting by date,c1,c2,c3,c4) in the processed data
    if config.get('include_original_5'):
        # Add the columns date, c1,c2,c3,c4 to  processed_data columns 
        transformed_data = pd.concat([date_column, data[ohlc_columns], transformed_data], axis=1)

    return transformed_data




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, shapiro, skew, kurtosis

def analyze_variability_and_normality(data, config):
    """
    Analyzes each column's variability, normality, and skewness.
    Applies log transformation if it improves normality.
    Returns transformed data with modified column names.
    """
    print("Analyzing variability and normality of each column...")

    transformed_columns = {}  # Dictionary to store transformed column names and their data
    num_columns = len(data.columns)
    num_rows = (num_columns + 3) // 4  # Adjust to show 4 columns of plots
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    plot_index = 0


      

    for column in data.columns:
        # Handle missing values by filling with mean for analysis (silent operation)
        if data[column].isna().sum() > 0:
            data[column] = data[column].fillna(data[column].mean())

        # Analyze original data
        original_data = data[column]

        # Variability (standard deviation)
        variability_original = np.std(original_data)

        # Normality test using D'Agostino's K^2 and Shapiro-Wilk
        dagostino_result_original = normaltest(original_data)
        shapiro_result_original = shapiro(original_data)

        # p-values
        p_value_normaltest_original = dagostino_result_original.pvalue
        p_value_shapiro_original = shapiro_result_original.pvalue

        # Skewness and Kurtosis
        skewness_original = skew(original_data)
        kurtosis_original = kurtosis(original_data)

        # Apply log transformation
        print(f"Analyzing {column} for possible log transformation...")
        # For log transformation, handle zeros and negative values appropriately
        if (original_data <= 0).any():
            # Since log cannot handle zero or negative values, we can shift the data if needed
            min_value = original_data.min()
            if min_value <= 0:
                shifted_data = original_data - min_value + 1  # Shift data to make it all positive
            else:
                shifted_data = original_data
        else:
            shifted_data = original_data

        log_transformed_data = np.log(shifted_data)

        # Analyze log-transformed data
        dagostino_result_log = normaltest(log_transformed_data)
        shapiro_result_log = shapiro(log_transformed_data)
        p_value_normaltest_log = dagostino_result_log.pvalue
        p_value_shapiro_log = shapiro_result_log.pvalue
        skewness_log = skew(log_transformed_data)
        kurtosis_log = kurtosis(log_transformed_data)

        # Decide whether to use log-transformed data or original data
        # Criteria: if log-transformed data has p-values closer to 1, or skewness closer to 0
        normality_score_original = abs(skewness_original) + abs(kurtosis_original)
        normality_score_log = abs(skewness_log) + abs(kurtosis_log)

        if normality_score_log < normality_score_original:
            print(f"Using log-transformed data for {column} (improved normality).")
            transformed_columns[column] = log_transformed_data
            # Plot log-transformed data
            sns.histplot(log_transformed_data, kde=True, ax=axes[plot_index])
            axes[plot_index].set_title(f"{column} (Log-Transformed)", fontsize=10)
        else:
            print(f"Using original data for {column} (log transform did not improve normality).")
            transformed_columns[column] = original_data
            # Plot original data
            sns.histplot(original_data, kde=True, ax=axes[plot_index])
            axes[plot_index].set_title(f"{column} (Original)", fontsize=10)
        plot_index += 1

    # Adjust layout and vertical separation
    plt.tight_layout(h_pad=10, pad=3)  # Added padding to prevent overlap
    plt.show()

    # Convert the transformed columns dictionary back into a DataFrame and return it
    transformed_data = pd.DataFrame(transformed_columns)
    
    return transformed_data












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
