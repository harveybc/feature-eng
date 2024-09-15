import pandas as pd
import time
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, normaltest, shapiro
import numpy as np

def analyze_variability_and_normality(data):
    """
    Analyzes each column's variability, normality, and skewness.
    Based on the metrics, applies log transformation, z-score normalization, or min-max normalization.
    Prints a detailed explanation for each decision in one line with all calculated values.
    """
    print("Analyzing variability and normality of each column...")

    for column in data.columns:
        # Handle missing values by filling with mean for analysis
        if data[column].isna().sum() > 0:
            print(f"{column} contains NaN values. Filling with column mean for analysis.")
            data[column] = data[column].fillna(data[column].mean())

        # Apply log transformation to highly skewed columns (e.g., ADX, DI+, DI-, ATR)
        if column in ['ADX', 'DI+', 'DI-', 'ATR']:
            print(f"Applying log transformation to {column} due to high skewness.")
            data[column] = np.log1p(data[column])  # Log(1 + x) to avoid log(0) issue

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

        # Refined Normality Decision Logic with Expanded Kurtosis Threshold [-1, 6]
        if -0.5 < column_skewness < 0.5 and -1.0 < column_kurtosis < 6.0:
            print(f"{column} is almost normally distributed because skewness is {column_skewness:.5f} in [-0.5, 0.5] and kurtosis is {column_kurtosis:.5f} in [-1, 6]. Applying z-score normalization.")
            data[column] = (data[column] - data[column].mean()) / data[column].std()
        else:
            print(f"{column} is not normally distributed because D'Agostino p-value is {p_value_normaltest:.5f} <= 0.05 or Shapiro-Wilk p-value is {p_value_shapiro:.5f} <= 0.05, and skewness is {column_skewness:.5f}, kurtosis is {column_kurtosis:.5f}. Applying min-max normalization.")
            data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

    return data

# Example usage with a dummy dataset (replace with actual dataset)
# df = pd.read_csv('your_data.csv')
# df = analyze_variability_and_normality(df)

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

    plt.tight_layout()
    plt.show()



v










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
