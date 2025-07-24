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

    # Save the plot as a PNG image
    output_image_path = config.get('variability_normality_plot', 'variability_normality_analysis.png')
    plt.savefig(output_image_path)
    print(f"[DEBUG] Saved variability and normality analysis plot to '{output_image_path}'.")
    plt.close(fig)  # Close the plot to prevent it from displaying

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

    # Initialize decomposition processor as None
    decomp_processor = None

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

    # Extract the original CLOSE column from the raw input file for the final output
    original_close_column = None
    
    # Get the input file path from config and load the raw data
    input_file = config.get('input_file')
    max_rows = config.get('max_rows', None)
    if input_file:
        try:
            raw_data = load_csv(input_file, config)
            if max_rows and len(raw_data) > max_rows:
                raw_data = raw_data.head(max_rows)
                print(f"[DEBUG] Limited raw data to {max_rows} rows")
            print(f"[DEBUG] Loaded raw data from {input_file} with shape: {raw_data.shape}")
            print(f"[DEBUG] Raw data columns: {raw_data.columns.tolist()}")
            
            # Set the datetime index on raw data if needed
            if 'DATE_TIME' in raw_data.columns and raw_data.index.dtype != 'datetime64[ns]':
                raw_data['DATE_TIME'] = pd.to_datetime(raw_data['DATE_TIME'])
                raw_data.set_index('DATE_TIME', inplace=True)
                print("[DEBUG] Set datetime index on raw data")
            
            print(f"[DEBUG] Raw data index type: {raw_data.index.dtype}")
            print(f"[DEBUG] Raw data date range: {raw_data.index.min()} to {raw_data.index.max()}")
            
            if 'close' in raw_data.columns:
                original_close_column = raw_data['close']
                print("[DEBUG] Found original 'close' column in raw input data")
            elif 'CLOSE' in raw_data.columns:
                original_close_column = raw_data['CLOSE']
                print("[DEBUG] Found original 'CLOSE' column in raw input data")
            elif 'Close' in raw_data.columns:
                original_close_column = raw_data['Close']
                print("[DEBUG] Found original 'Close' column in raw input data")
        except Exception as e:
            print(f"[WARNING] Could not load original close column from input file: {e}")
            original_close_column = None
    
    # Combine the transformed data with additional features
    if config.get('tech_indicators'):
        # CRITICAL FIX: Remove duplicates to prevent _x, _y suffixes
        print(f"[DEBUG] transformed_data columns: {list(transformed_data.columns)}")
        print(f"[DEBUG] additional_features_df columns: {list(additional_features_df.columns)}")
        
        # Find and remove duplicate columns from additional_features_df
        duplicate_columns = set(transformed_data.columns) & set(additional_features_df.columns)
        if duplicate_columns:
            print(f"[DEBUG] Found duplicate columns: {duplicate_columns}")
            additional_features_df = additional_features_df.drop(columns=list(duplicate_columns))
            print(f"[DEBUG] Removed duplicates from additional_features_df. New columns: {list(additional_features_df.columns)}")
        
        final_data = pd.concat([transformed_data, additional_features_df], axis=1)
        print("[DEBUG] Final combined data shape:", final_data.shape)
        print("[DEBUG] Final combined data columns:", list(final_data.columns))
        print("[DEBUG] Final combined data first 5 rows:")
    else:
        final_data = additional_features_df
        print("[DEBUG] Final data shape:", final_data.shape)
        print("[DEBUG] Final data first 5 rows:")
    print(final_data.head())

    # Apply decomposition post-processing if configured
    print(f"[DEBUG] Config keys: {list(config.keys())}")
    print(f"[DEBUG] decomp_features in config: {config.get('decomp_features', 'NOT_FOUND')}")
    print(f"[DEBUG] use_stl_decomp in config: {config.get('use_stl_decomp', 'NOT_FOUND')}")
    print(f"[DEBUG] use_wavelet_decomp in config: {config.get('use_wavelet_decomp', 'NOT_FOUND')}")
    decomp_features = config.get('decomp_features', [])
    print(f"[DEBUG] decomp_features value: {decomp_features}, type: {type(decomp_features)}")
    if decomp_features:
        print(f"[DEBUG] Applying decomposition post-processing to features: {decomp_features}")
        print(f"[DEBUG] Available columns for decomposition: {list(final_data.columns)}")
        print(f"[DEBUG] Final data shape before decomposition: {final_data.shape}")
        try:
            from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
            
            # Create post-processor with configuration
            decomp_params = {
                'decomp_features': decomp_features,
                'use_stl_decomp': config.get('use_stl_decomp', True),
                'use_wavelet_decomp': config.get('use_wavelet_decomp', True),
                'use_mtm_decomp': config.get('use_mtm_decomp', True),  # PHASE 3.1 REQUIRES MTM
                'normalize_decomposed_features': True,
                'replace_original': True,
                'keep_original': False,
                # MTM parameters for proper decomposition
                'mtm_window_len': 168,
                'mtm_step': 1,
                'mtm_time_bandwidth': 5.0,
                'mtm_freq_bands': [(0, 0.01), (0.01, 0.06), (0.06, 0.2), (0.2, 0.5)],
                # Wavelet parameters
                'wavelet_name': 'db4',
                'wavelet_levels': 2,
                'wavelet_mode': 'symmetric',
                # STL parameters
                'stl_period': 24,
                'stl_window': None,  # Will be auto-calculated
                'stl_trend': None    # Will be auto-calculated
            }
            
            print(f"[DEBUG] Decomposition parameters being passed: {decomp_params}")
            
            decomp_processor = DecompositionPostProcessor(decomp_params)
            
            # Set the index back to datetime for decomposition processing
            final_data_with_index = final_data.copy()
            if 'DATE_TIME' not in final_data_with_index.columns:
                # DATE_TIME is already the index
                pass
            else:
                # DATE_TIME is a column, convert to datetime and set as index
                final_data_with_index['DATE_TIME'] = pd.to_datetime(final_data_with_index['DATE_TIME'])
                final_data_with_index.set_index('DATE_TIME', inplace=True)
            
            # Apply decomposition
            decomposed_data = decomp_processor.process_features(final_data_with_index)
            
            # Reset index again for final output
            decomposed_data.reset_index(inplace=True)
            decomposed_data.rename(columns={'DATE_TIME': 'DATE_TIME'}, inplace=True)
            
            final_data = decomposed_data
            print(f"[DEBUG] After decomposition post-processing, final data shape: {final_data.shape}")
            print(f"[DEBUG] Final decomposed data columns: {list(final_data.columns)}")
            
        except Exception as e:
            print(f"[ERROR] Decomposition post-processing failed: {e}")
            print("[DEBUG] Continuing with original data...")
    
    # Note: CLOSE column is now handled by the decomposition post-processor in the correct order
    # No need to add extra CLOSE column here

    # Reset the index for the final dataset
    if 'DATE_TIME' not in final_data.columns:
        final_data.reset_index(inplace=True)
        if final_data.columns[0] == 'index':
            final_data.rename(columns={'index': 'DATE_TIME'}, inplace=True)
    print("[DEBUG] Final dataset with DATE_TIME column restored, first 5 rows:")
    print(final_data.head())

    # Remove rows with any NaN values before saving
    rows_before = len(final_data)
    final_data = final_data.dropna()
    rows_after = len(final_data)
    rows_removed = rows_before - rows_after
    
    if rows_removed > 0:
        print(f"[DEBUG] Removed {rows_removed} rows containing NaN values ({rows_before} -> {rows_after})")
    else:
        print(f"[DEBUG] No NaN values found, all {rows_before} rows retained")

    return final_data, decomp_processor





def run_feature_engineering_pipeline(config, plugin):
    """
    Runs the feature-engineering pipeline using the plugin.
    """
    start_time = time.time()

    # Load the data with max_rows limit for faster testing
    print(f"Loading data from {config['input_file']}...")
    max_rows = config.get('max_rows', None)
    if max_rows:
        print(f"[DEBUG] Limiting data to {max_rows} rows for faster processing")
    data = load_csv(config['input_file'], config=config)
    if max_rows and len(data) > max_rows:
        data = data.head(max_rows)
        print(f"[DEBUG] Limited data to {max_rows} rows")
    print(f"Data loaded with shape: {data.shape}")
    print(f"Data index type: {data.index.dtype}, range: {data.index.min()} to {data.index.max()}")

    # Process the data and capture plugin instances for FE config export
    processed_data, decomp_processor = process_data(data, plugin, config)

    # Export comprehensive FE configuration for perfect replicability if specified
    if config.get('fe_config_export'):
        try:
            from app.fe_config_manager import FeConfigManager
            
            # Create FE config manager and export comprehensive configuration
            fe_manager = FeConfigManager()
            exported_config = fe_manager.export_comprehensive_config(plugin, decomp_processor, config)
            fe_config_path = fe_manager.save_fe_config(exported_config, config['fe_config_export'])
            
            print(f"[FE_CONFIG] ✅ PERFECT REPLICABILITY CONFIG EXPORTED: {fe_config_path}")
            print(f"[FE_CONFIG] ✅ Contains ALL tech indicator parameters and decomposition settings")
            print(f"[FE_CONFIG] ✅ Ready for exact replication in prediction_provider repo")
            
        except Exception as e:
            print(f"[WARNING] ❌ Failed to export FE configuration: {e}")
            import traceback
            traceback.print_exc()

    # Save the processed data to the output file if specified
    if config.get('output_file'):
        # Remove rows with any NaN values before final export
        rows_before = len(processed_data)
        processed_data = processed_data.dropna()
        rows_after = len(processed_data)
        rows_removed = rows_before - rows_after
        
        if rows_removed > 0:
            print(f"[FINAL EXPORT] Removed {rows_removed} rows containing NaN values ({rows_before} -> {rows_after})")
        else:
            print(f"[FINAL EXPORT] No NaN values found, all {rows_before} rows retained")
        
        # Remove the first 168 rows before final export
        processed_data = processed_data.iloc[168:]
        print(f"[FINAL EXPORT] Removed first 168 rows. Final dataset shape: {processed_data.shape}")
        
        processed_data.to_csv(config['output_file'], index=False) 
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



