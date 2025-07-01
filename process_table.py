#!/usr/bin/env python3
"""
Script to remove the first column from a CSV file and check for non-empty binomial_name values.
"""

import pandas as pd
import sys
import argparse


def process_csv_file(input_file, output_file=None, delimiter=';'):
    """
    Remove the first column from a CSV file and check for non-empty binomial_name values.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file (optional)
        delimiter (str): CSV delimiter (default: ';')
    
    Returns:
        tuple: (success, message, empty_binomial_count)
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file, delimiter=delimiter)
        
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Remove the first column
        if len(df.columns) > 0:
            first_column = df.columns[0]
            df_modified = df.drop(columns=[first_column])
            print(f"Removed first column: '{first_column}'")
            print(f"New shape: {df_modified.shape}")
            print(f"Remaining columns: {list(df_modified.columns)}")
        else:
            return False, "CSV file has no columns", 0
        
        # Check if binomial_name column exists
        if 'binomial_name' not in df_modified.columns:
            return False, "No 'binomial_name' column found after removing first column", 0
        
        # Check for empty binomial_name values
        binomial_name_series = df_modified['binomial_name']
        
        # Count empty, null, or whitespace-only values
        empty_mask = (
            binomial_name_series.isna() | 
            binomial_name_series.isnull() | 
            (binomial_name_series.astype(str).str.strip() == '')
        )
        empty_count = empty_mask.sum()
        total_count = len(df_modified)
        
        print(f"\nBinomial name analysis:")
        print(f"Total entries: {total_count}")
        print(f"Empty binomial_name entries: {empty_count}")
        print(f"Non-empty binomial_name entries: {total_count - empty_count}")
        
        if empty_count > 0:
            print(f"\nRows with empty binomial_name:")
            empty_rows = df_modified[empty_mask]
            print(empty_rows.head(10))  # Show first 10 empty rows
            if len(empty_rows) > 10:
                print(f"... and {len(empty_rows) - 10} more")
        
        # Save the modified CSV if output file is specified
        if output_file:
            df_modified.to_csv(output_file, sep=delimiter, index=False)
            print(f"\nModified CSV saved to: {output_file}")
        
        # Return success status
        if empty_count == 0:
            return True, f"SUCCESS: All {total_count} entries have non-empty binomial_name values", empty_count
        else:
            return False, f"FAILURE: {empty_count} out of {total_count} entries have empty binomial_name values", empty_count
            
    except Exception as e:
        return False, f"Error processing file: {str(e)}", 0


def main():
    parser = argparse.ArgumentParser(description='Remove first column and check binomial_name values')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('-o', '--output', help='Path to output CSV file (optional)')
    parser.add_argument('-d', '--delimiter', default=';', help='CSV delimiter (default: ";")')
    
    args = parser.parse_args()
    
    # Process the file
    success, message, empty_count = process_csv_file(args.input_file, args.output, args.delimiter)
    
    print(f"\n{'='*60}")
    print(f"RESULT: {message}")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 