import pandas as pd

def load_csv(file_path):
    """
    Load a CSV file assuming it has a 'date' column at the beginning and headers.
    Renames the columns from the headers to 'date', 'c1', 'c2', etc.
    """
    try:
        # Read CSV with headers and date parsing for the first column
        data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
        
        # Correctly set column names including 'date'
        data.columns = ['date'] + [f'c{i}' for i in range(1, len(data.columns))]
        data.set_index('date', inplace=True)

        print(f"Loaded data columns: {data.columns}")  # Debugging line
        
        # Convert all non-date columns to numeric, coercing errors to NaN
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
