import pandas as pd

def load_csv(file_path, has_headers=True, dataset_type=None, config=None):
    """
    Load a CSV file with dynamic column mapping based on dataset type.

    Parameters:
    - file_path (str): Path to the CSV file.
    - has_headers (bool): Whether the CSV file includes headers.
    - dataset_type (str): The type of dataset being loaded (e.g., 'forex_15m').
    - config (dict): Configuration for header mappings.

    Returns:
    - pd.DataFrame: Processed DataFrame with standardized column names.
    """
    try:
        header_mappings = config.get('header_mappings', {})
        column_map = header_mappings.get(dataset_type, {})

        # Read CSV
        if has_headers:
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
            data.rename(columns=column_map, inplace=True)
        else:
            column_names = list(column_map.values())
            data = pd.read_csv(file_path, sep=',', header=None, names=column_names, parse_dates=[0], dayfirst=True)

        # Ensure 'date' or 'datetime' column is set as index
        date_col = column_map.get('date', 'date') if 'date' in column_map else column_map.get('datetime', 'datetime')
        data.set_index(date_col, inplace=True)

        print(f"Loaded data columns: {data.columns}")  # Debugging line
        
        # Convert all non-date columns to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

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
