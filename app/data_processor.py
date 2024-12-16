import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, shapiro, skew, kurtosis
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import warnings
from app.positional_encoding import generate_positional_encoding

# Suppress the specific UserWarning from scipy.stats about p-values for large datasets
warnings.filterwarnings("ignore", message="p-value may not be accurate for N > 5000.")


def analyze_variability_and_normality(data, config):
    """
    Analyzes each column's variability using the Coefficient of Variation (CV),
    normality, and skewness. Measures variability and prints whether it is high
    or low variability. Applies log transformation if it improves normality.
    Returns transformed data with modified column names.
    """
    print("Analyzing variability and normality of each column...")

    # First, compute Coefficient of Variation (CV) for all columns
    cvs = {}
    epsilon = 1e-8  # Small value to prevent division by zero
    for column in data.columns:
        # Handle missing values by filling with mean for analysis (silent operation)
        column_data = data[column].fillna(data[column].mean())
        mean = column_data.mean()
        std_dev = column_data.std()
        # Avoid division by zero or near-zero mean
        adjusted_mean = mean if abs(mean) > epsilon else epsilon
        cv = std_dev / abs(adjusted_mean)
        cvs[column] = cv

    # Compute the median CV
    median_cv = np.median(list(cvs.values()))

    # Now, proceed to analyze each column
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

        # Variability using CV
        cv = cvs[column]

        # Determine high or low variability
        if cv > median_cv:
            variability_status = "High Variability"
        else:
            variability_status = "Low Variability"

        print(f"{column} has CV {cv:.5f} ({variability_status}).")

        # Normality test using D'Agostino's K^2 and Shapiro-Wilk
        skewness_original = skew(original_data)
        kurtosis_original = kurtosis(original_data)

        # Apply log transformation
        print(f"Analyzing {column} for possible log transformation...")
        # For log transformation, handle zeros and negative values appropriately
        if (original_data <= 0).any():
            # Since log cannot handle zero or negative values, we can shift the data if needed
            min_value = original_data.min()
            shifted_data = original_data - min_value + 1  # Shift data to make it all positive
        else:
            shifted_data = original_data

        log_transformed_data = np.log(shifted_data)

        # Analyze log-transformed data
        skewness_log = skew(log_transformed_data)
        kurtosis_log = kurtosis(log_transformed_data)

        # Decide whether to use log-transformed data or original data
        # Criteria: if log-transformed data has a lower normality score
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
    transformed_data = pd.DataFrame(transformed_columns, index=data.index)
    print(f"Transformed data shape: {transformed_data.shape}")
    print(f"Transformed data columns: {transformed_data.columns}")
    print(f"Transformed data index type: {transformed_data.index.dtype}")
    print(f"Transformed data index range: {transformed_data.index.min()} to {transformed_data.index.max()}")

    return transformed_data


def process_data(data, plugin, config):
    print("[DEBUG] Starting process_data...")
    print(f"[DEBUG] Initial data shape: {data.shape}")
    print(f"[DEBUG] Initial data columns: {list(data.columns)}")
    print(f"[DEBUG] Initial data index type: {data.index.dtype}")

    date_column_name = 'DATE_TIME'
    if date_column_name in data.columns:
        data[date_column_name] = pd.to_datetime(data[date_column_name])
        data.set_index(date_column_name, inplace=True)
        print(f"[DEBUG] Set data index to {date_column_name}")
    else:
        raise KeyError(f"Date column '{date_column_name}' not found in data columns")

    print(f"[DEBUG] Data index type after setting datetime index: {data.index.dtype}")
    print(f"[DEBUG] Data index range: {data.index.min()} to {data.index.max()}")

    header_mappings = config.get('header_mappings', {})
    dataset_type = config.get('dataset_type', 'default')
    dataset_headers = header_mappings.get(dataset_type, {})
    ohlc_columns = [dataset_headers.get(k, k).upper() for k in ['open', 'high', 'low', 'close']]
    missing_columns = [col for col in ohlc_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"[ERROR] Missing expected OHLC columns: {missing_columns}. Available: {list(data.columns)}")
    print(f"[DEBUG] Mapped OHLC columns: {ohlc_columns}")

    numeric_data = data[ohlc_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"[DEBUG] Numeric OHLC data shape: {numeric_data.shape}")
    print("[DEBUG] Numeric OHLC data first 5 rows:")
    print(numeric_data.head())

    # Process the numeric OHLC data using the plugin
    processed_data = plugin.process(numeric_data)
    print(f"[DEBUG] After plugin.process, data range: {processed_data.index.min()} to {processed_data.index.max()} (len={len(processed_data)})")

    # Analyze variability and normality
    transformed_data = analyze_variability_and_normality(processed_data, config)
    print(f"[DEBUG] After analyze_variability_and_normality, data range: {transformed_data.index.min()} to {transformed_data.index.max()} (len={len(transformed_data)})")

    # Process additional datasets and determine the final valid date range
    additional_features_df, final_common_start, final_common_end = plugin.process_additional_datasets(data, config)
    print("[DEBUG] After process_additional_datasets:")
    print(f"[DEBUG] final_common_start: {final_common_start}, final_common_end: {final_common_end}")
    print(f"[DEBUG] additional_features_df shape: {additional_features_df.shape}")

    # Trim and save the technical indicators dataset to the final valid date range
    processed_data_trimmed = processed_data[(processed_data.index >= final_common_start) & (processed_data.index <= final_common_end)]
    processed_data_trimmed.reset_index().rename(columns={'index': 'datetime'}).to_csv('technical_indicators_aligned.csv', index=False)
    print("[DEBUG] Saved trimmed technical indicators dataset to 'technical_indicators_aligned.csv'.")

    # Re-slice transformed_data and additional_features_df to final_common_start and final_common_end
    transformed_data = transformed_data[(transformed_data.index >= final_common_start) & (transformed_data.index <= final_common_end)]
    additional_features_df = additional_features_df[(additional_features_df.index >= final_common_start) & (additional_features_df.index <= final_common_end)]
    print("[DEBUG] After re-slicing transformed_data and additional_features_df:")
    print(f"[DEBUG] transformed_data range: {transformed_data.index.min()} to {transformed_data.index.max()} (len={len(transformed_data)})")
    print(f"[DEBUG] additional_features_df range: {additional_features_df.index.min()} to {additional_features_df.index.max()} (len={len(additional_features_df)})")

    try:
        additional_features_df = additional_features_df.reindex(transformed_data.index, method='ffill').fillna(-1)
        print("[DEBUG] Successfully aligned additional features with transformed_data.")
    except Exception as e:
        print("[ERROR] Failed to align additional features with transformed_data.")
        print("[DEBUG] transformed_data first 5 rows:", transformed_data.head())
        print("[DEBUG] additional_features_df first 5 rows:", additional_features_df.head())
        raise e

    final_data = pd.concat([transformed_data, additional_features_df], axis=1)
    print("[DEBUG] Final combined data shape:", final_data.shape)
    print("[DEBUG] Final combined data first 5 rows:")
    print(final_data.head())

    final_data.reset_index(inplace=True)
    if 'datetime' not in final_data.columns:
        final_data.rename(columns={'index': 'datetime'}, inplace=True)
    print("[DEBUG] Final dataset with datetime column restored, first 5 rows:")
    print(final_data.head())

    return final_data




def run_feature_engineering_pipeline(config, plugin):
    """
    Runs the feature-engineering pipeline using the plugin.
    """
    start_time = time.time()

    # Load the data
    print(f"Loading data from {config['input_file']}...")
    data = load_csv(config['input_file'], config=config)
    print(f"Data loaded with shape: {data.shape}")
    print(f"Data index type: {data.index.dtype}, range: {data.index.min()} to {data.index.max()}")

    # Process the data
    processed_data = process_data(data, plugin, config)

    # Save the processed data to the output file if specified
    if config.get('output_file'):
        processed_data.to_csv(config['output_file'], index=True)  # Ensure index is saved as datetime
        print(f"Processed data saved to {config['output_file']}.")
    else:
        print("No output file specified; skipping save.")

    # Save final configuration and debug information
    end_time = time.time()
    execution_time = end_time - start_time
    debug_info = {'execution_time': float(execution_time)}

    # Save debug info if specified
    if config.get('save_log'):
        save_debug_info(debug_info, config['save_log'])
        print(f"Debug info saved to {config['save_log']}.")
    else:
        print("No log file specified; skipping save of debug info.")

    print(f"Execution time: {execution_time:.2f} seconds")



