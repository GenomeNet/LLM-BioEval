#!/usr/bin/env python3
"""
Flexible script to process CSV files and check binomial_name integrity.
Can handle different scenarios based on file structure.
"""

import pandas as pd
import sys
import argparse


def process_csv_file(input_file, output_file=None, delimiter=';', force_remove_first=False):
    """
    Process CSV file based on its structure and check binomial_name integrity.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file (optional)
        delimiter (str): CSV delimiter (default: ';')
        force_remove_first (bool): Force removal of first column regardless of name
    
    Returns:
        tuple: (success, message, empty_binomial_count)
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file, delimiter=delimiter)
        
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Determine processing strategy
        first_column = df.columns[0] if len(df.columns) > 0 else None
        has_binomial_name = 'binomial_name' in df.columns
        
        if not has_binomial_name:
            return False, "No 'binomial_name' column found in the file", 0
        
        df_modified = df.copy()
        
        # Strategy 1: First column is binomial_name - just check integrity
        if first_column == 'binomial_name' and not force_remove_first:
            print(f"\nStrategy: First column is 'binomial_name' - checking integrity only")
            
        # Strategy 2: First column is not binomial_name or forced removal
        elif first_column != 'binomial_name' or force_remove_first:
            if force_remove_first or (first_column and first_column.lower() in ['index', 'id', 'row_id', '']):
                df_modified = df.drop(columns=[first_column])
                print(f"\nStrategy: Removed first column '{first_column}'")
                print(f"New shape: {df_modified.shape}")
                print(f"Remaining columns: {list(df_modified.columns)}")
                
                if 'binomial_name' not in df_modified.columns:
                    return False, f"No 'binomial_name' column found after removing '{first_column}'", 0
            else:
                print(f"\nStrategy: Keeping all columns (first column '{first_column}' doesn't look like an index)")
        
        # Check binomial_name integrity
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
        
        # Show sample of non-empty values
        non_empty_sample = binomial_name_series[~empty_mask].head(5)
        print(f"\nSample of valid binomial names:")
        for i, name in enumerate(non_empty_sample, 1):
            print(f"  {i}. {name}")
        
        if empty_count > 0:
            print(f"\nRows with empty binomial_name:")
            empty_rows = df_modified[empty_mask]
            for i, (idx, row) in enumerate(empty_rows.head(10).iterrows()):
                print(f"  Row {idx}: binomial_name = '{row['binomial_name']}'")
            if len(empty_rows) > 10:
                print(f"  ... and {len(empty_rows) - 10} more")
        
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
    parser = argparse.ArgumentParser(description='Process CSV and check binomial_name integrity')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('-o', '--output', help='Path to output CSV file (optional)')
    parser.add_argument('-d', '--delimiter', default=';', help='CSV delimiter (default: ";")')
    parser.add_argument('-f', '--force-remove-first', action='store_true', 
                        help='Force removal of first column regardless of its name')
    
    args = parser.parse_args()
    
    # Process the file
    success, message, empty_count = process_csv_file(
        args.input_file, 
        args.output, 
        args.delimiter, 
        args.force_remove_first
    )
    
    print(f"\n{'='*60}")
    print(f"RESULT: {message}")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 