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

        # Get the date format from the config if specified
        date_format = config.get('date_format', None)

        # Load the CSV file with or without a custom date parser
        if date_format:
            data = pd.read_csv(
                file_path, sep=',', parse_dates=[0],
                date_parser=lambda x: pd.to_datetime(x, format=date_format, errors='coerce')
            )
        else:
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)

        # Apply column mappings if specified
        if column_map:
            data.rename(columns=column_map, inplace=True)

        # Ensure the 'date' column is parsed as datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data.set_index('date', inplace=True)

        # Convert numeric columns to appropriate types
        for col in data.select_dtypes(include=['object', 'category']).columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"Loaded data columns: {list(data.columns)}")
        print(f"First 5 rows of data:\n{data.head()}")

    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
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
