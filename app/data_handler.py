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


def load_additional_csv(file_path, dataset_type, config=None):
    """
    Load additional datasets with specific parsing requirements.

    Parameters:
    - file_path (str): Path to the file.
    - dataset_type (str): Type of the dataset ('forex_15m', 'sp500', 'vix', 'economic_calendar').
    - config (dict): Configuration for header mappings.

    Returns:
    - pd.DataFrame: Loaded and processed data.
    """
    try:
        # Load header mappings from config
        header_mappings = config.get('header_mappings', {}) if config else {}
        column_map = header_mappings.get(dataset_type, {})

        # Load the CSV file
        data = pd.read_csv(file_path, sep=',', encoding='utf-8')

        # Apply column mappings
        if column_map:
            data.rename(columns=column_map, inplace=True)

        # Parse 'DATE_TIME' for Forex datasets
        if 'DATE_TIME' in data.columns:
            data['DATE_TIME'] = pd.to_datetime(
                data['DATE_TIME'].str.strip(),
                format='%Y.%m.%d %H:%M:%S',
                errors='coerce'
            )
            invalid_rows = data['DATE_TIME'].isna().sum()
            if invalid_rows > 0:
                print(f"Warning: Found {invalid_rows} rows with invalid DATE_TIME values. Dropping them.")
                data = data.dropna(subset=['DATE_TIME'])
            data.set_index('DATE_TIME', inplace=True)

        # Parse 'date' for SP500 and VIX datasets
        elif 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            invalid_rows = data['date'].isna().sum()
            if invalid_rows > 0:
                print(f"Warning: Found {invalid_rows} rows with invalid date values. Dropping them.")
                data = data.dropna(subset=['date'])
            data.set_index('date', inplace=True)

        # Parse economic calendar with positional encoding
        elif dataset_type == 'economic_calendar':
            data.columns = [
                'event_date', 'event_time', 'country', 'volatility', 'description',
                'evaluation', 'data_format', 'actual', 'forecast', 'previous'
            ]
            data['datetime'] = pd.to_datetime(
                data['event_date'] + ' ' + data['event_time'], format='%Y/%m/%d %H:%M:%S', errors='coerce'
            )
            invalid_rows = data['datetime'].isna().sum()
            if invalid_rows > 0:
                print(f"Warning: Found {invalid_rows} rows with invalid datetime values. Dropping them.")
                data = data.dropna(subset=['datetime'])
            data.set_index('datetime', inplace=True)

        # Convert numeric columns
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"Loaded data columns: {list(data.columns)}")
        print(f"First 5 rows of data:\n{data.head()}")

    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        raise

    return data

def load_sp500_csv(file_path):
    """
    Load and process S&P 500 dataset specifically.

    Parameters:
    - file_path (str): Path to the S&P 500 CSV file.

    Returns:
    - pd.DataFrame: Loaded and processed data with a valid DatetimeIndex.
    """
    try:
        # Load the CSV
        data = pd.read_csv(file_path, sep=',', encoding='utf-8')

        # Ensure the 'Date' column is properly parsed
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
            invalid_dates = data['Date'].isna().sum()
            if invalid_dates > 0:
                print(f"Warning: Found {invalid_dates} rows with invalid Date values. Dropping them.")
                data = data.dropna(subset=['Date'])
            data.set_index('Date', inplace=True)

        # Convert numeric columns
        for col in data.select_dtypes(include='object').columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        print(f"Loaded S&P 500 data columns: {list(data.columns)}")
        print(f"First 5 rows of S&P 500 data:\n{data.head()}")

    except Exception as e:
        print(f"An error occurred while loading the S&P 500 CSV: {e}")
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
