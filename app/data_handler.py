import pandas as pd

def load_csv(file_path, has_headers=True, header=None, column_map=None):
    """
    Load a CSV file with optional handling for missing headers and column renaming.

    Parameters:
    - file_path (str): Path to the CSV file.
    - has_headers (bool): Whether the CSV file has headers. Default is True.
    - header (list): If `has_headers=False`, provide a list of column names.
    - column_map (dict): Mapping of current column names to standardized names.

    Returns:
    - pd.DataFrame: Loaded and processed DataFrame.
    """
    try:
        # Read CSV with or without headers
        if has_headers:
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
        else:
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True, header=None)
            if header:
                data.columns = header
        
        # Apply column renaming if provided
        if column_map:
            data.rename(columns=column_map, inplace=True)

        # Standardize column names (lowercase)
        data.columns = [col.lower() for col in data.columns]

        # Ensure the first column is datetime
        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
        data.set_index(data.columns[0], inplace=True)

        # Convert all non-datetime columns to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"Loaded data columns: {data.columns}")
        print(f"First 5 rows of the data:\n{data.head()}")

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
