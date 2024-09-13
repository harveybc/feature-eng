import pandas as pd

def load_csv(file_path):
    """
    Load a CSV file assuming it has a 'date' column at the beginning and headers.
    """
    try:
        print(f"Loading CSV from: {file_path}")
        # Read CSV with headers and date parsing for the first column
        data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
        
        # Assume the first column is the 'date' column
        data.columns = ['date'] + [f'c{i}' for i in range(1, len(data.columns))]
        data.set_index('date', inplace=True)
        
        # Print the loaded columns to debug any issues with initial loading
        print(f"Loaded data columns: {data.columns}")
        print(f"First 5 rows of the data:\n{data.head()}")

        # Ensure only numeric data is processed for technical indicators
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Debugging numeric conversion
        print(f"Numeric data after conversion:\n{data.head()}")

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
