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
        data = pd.read_csv(file_path, sep=',', encoding='utf-8', parse_dates=False)

        # Apply column mappings
        if column_map:
            data.rename(columns=column_map, inplace=True)

        # Debug raw data
        print(f"Raw data loaded from {file_path}:\n{data.head()}")
        print(f"Data columns: {data.columns}")

        # Ensure DATE_TIME column is parsed correctly for Forex datasets
        if 'DATE_TIME' in data.columns:
            data['DATE_TIME'] = data['DATE_TIME'].str.strip()  # Remove spaces
            data['DATE_TIME'] = pd.to_datetime(
                data['DATE_TIME'],
                format='%Y.%m.%d %H:%M:%S',  # Adjust format as needed
                errors='coerce'  # Handle errors gracefully
            )
            invalid_dates = data['DATE_TIME'].isna().sum()
            if invalid_dates > 0:
                print(f"Warning: Found {invalid_dates} rows with invalid DATE_TIME values.")
                data.dropna(subset=['DATE_TIME'], inplace=True)
            data.set_index('DATE_TIME', inplace=True)

        # Handle other date columns
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data.set_index('date', inplace=True)

        # Convert numeric columns
        for col in data.select_dtypes(include=['object', 'category']).columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"Processed data columns: {list(data.columns)}")
        print(f"First 5 rows of processed data:\n{data.head()}")

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
