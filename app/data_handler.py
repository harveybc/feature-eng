import pandas as pd

def load_csv(file_path, config=None):
    """
    Load a CSV file dynamically based on header mappings and configurations.

    Parameters:
    - file_path (str): Path to the file.
    - config (dict): Configuration for header mappings.

    Returns:
    - pd.DataFrame: Loaded and processed data.
    """
    try:
        # Load header mappings from config if available
        header_mappings = config.get('header_mappings', {}) if config else {}

        # Check for dataset type
        dataset_type = config.get('dataset_type', 'default') if config else 'default'
        column_map = header_mappings.get(dataset_type, {})

        # Load the CSV file
        data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)

        # Apply column mappings
        if column_map:
            data.rename(columns=column_map, inplace=True)

        # Ensure date column is parsed
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data.set_index('date', inplace=True)

        # Convert numeric columns
        for col in data.select_dtypes(include=['object', 'category']).columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"Loaded data columns: {list(data.columns)}")
        print(f"First 5 rows of data:\n{data.head()}")

    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        raise

    return data


def load_additional_csv(file_path, config=None, dataset_type=None):
    """
    Load an additional dataset (e.g., S&P 500, VIX) with appropriate parsing and column handling.

    Parameters:
    - file_path (str): Path to the file.
    - config (dict): Configuration for header mappings and date parsing.
    - dataset_type (str): Dataset type (e.g., 'sp500', 'vix', etc.) for specific parsing rules.

    Returns:
    - pd.DataFrame: Loaded and processed DataFrame.
    """
    try:
        # Retrieve header mappings for the specified dataset type
        header_mappings = config.get('header_mappings', {}).get(dataset_type, {}) if config else {}

        # Load the dataset
        data = pd.read_csv(file_path, sep=',', header=0)

        # Apply column mappings (if specified)
        if header_mappings:
            data.rename(columns=header_mappings, inplace=True)

        # Parse the date column based on dataset type
        if dataset_type == 'sp500' and 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce', format='%Y-%m-%d')

        elif dataset_type == 'vix' and 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce', format='%Y-%m-%d')

        # Drop rows with invalid dates
        if dataset_type in ['sp500', 'vix']:
            date_column = 'Date' if 'Date' in data.columns else 'date'
            data.dropna(subset=[date_column], inplace=True)

        # Set the date column as the index
        if dataset_type in ['sp500', 'vix']:
            data.set_index(date_column, inplace=True)

        # Print debug information
        print(f"Loaded data columns: {list(data.columns)}")
        print(f"First 5 rows of data:\n{data.head()}")

    except Exception as e:
        print(f"An error occurred while loading the CSV for {dataset_type}: {e}")
        raise

    return data


def write_csv(file_path, data, include_date=True, headers=True):
    """
    Write a DataFrame to a CSV file, optionally including the date column and headers.
    
    Parameters:
    - file_path: str: Path to the output CSV file
    - data: pd.DataFrame: DataFrame to save
    - include_date: bool: Whether to include the 'date' column in the output
    - headers: bool: Whether to include the column headers in the output
    """
    try:
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
        else:
            data.to_csv(file_path, index=False, header=headers)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise
