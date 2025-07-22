import pandas as pd
import sys

def concatenate_csv_files(file1, file2, output_file):
    """
    Concatenate two CSV files column-wise and save the result to a new CSV file.
    
    Parameters:
    file1 (str): Path to the first CSV file.
    file2 (str): Path to the second CSV file.
    output_file (str): Path to the output CSV file.
    """
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Concatenate the DataFrames column-wise
    concatenated_df = pd.concat([df1, df2], axis=1)
    
    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python concatenate_csv.py <file1> <file2> <output_file>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]
    
    concatenate_csv_files(file1, file2, output_file)
    print(f"Concatenated file saved to {output_file}")
