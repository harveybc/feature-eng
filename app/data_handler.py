import pandas as pd

def load_csv(file_path, has_headers=True, columns=None, config=None):
    """
    Load a CSV file dynamically based on headers and column definitions.

    Parameters:
    - file_path (str): Path to the file.
    - has_headers (bool): Whether the CSV file includes headers.
    - columns (list): Column names to assign if headers are missing.
    - config (dict): Configuration for header mappings.

    Returns:
    - pd.DataFrame: Loaded and processed data.
    """
    try:
        # Load the CSV file with or without headers
        if has_headers:
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
        else:
            data = pd.read_csv(file_path, sep=',', header=None)
        
        # Assign custom column names if provided
        if columns:
            data.columns = columns
        
        # Load header mappings from config if available
        if config:
            header_mappings = config.get('header_mappings', {})
            dataset_type = config.get('dataset_type', 'default')
            column_map = header_mappings.get(dataset_type, {})
            if column_map:
                data.rename(columns=column_map, inplace=True)
        
        # Ensure date column is parsed and set as index
        if 'date' in data.columns or 'DATE_TIME' in data.columns:
            date_col = 'date' if 'date' in data.columns else 'DATE_TIME'
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data.set_index(date_col, inplace=True)

        # Convert numeric columns
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
