#!/usr/bin/env python3
"""
Script to filter ground truth dataset for well-annotated samples.
Keeps only rows with at least 8 non-NA annotations.
"""

import pandas as pd
import sys

def filter_well_annotated(input_file, output_file, min_annotations=8):
    """
    Filter CSV file to keep only rows with at least min_annotations non-NA values.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        min_annotations (int): Minimum number of non-NA annotations required
    """
    
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file with semicolon delimiter
    df = pd.read_csv(input_file, delimiter=';')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get annotation columns (exclude the first column which is binomial_name)
    annotation_columns = df.columns[1:]  # All columns except the first one
    print(f"Number of annotation columns: {len(annotation_columns)}")
    
    # Count non-NA values in annotation columns for each row
    non_na_counts = df[annotation_columns].notna().sum(axis=1)
    
    print(f"Non-NA annotation counts statistics:")
    print(f"  Mean: {non_na_counts.mean():.2f}")
    print(f"  Median: {non_na_counts.median():.2f}")
    print(f"  Min: {non_na_counts.min()}")
    print(f"  Max: {non_na_counts.max()}")
    
    # Filter rows with at least min_annotations non-NA values
    well_annotated_mask = non_na_counts >= min_annotations
    filtered_df = df[well_annotated_mask].copy()
    
    print(f"\nFiltering results:")
    print(f"  Rows with >= {min_annotations} annotations: {well_annotated_mask.sum()}")
    print(f"  Filtered dataset shape: {filtered_df.shape}")
    print(f"  Retention rate: {(len(filtered_df) / len(df)) * 100:.1f}%")
    
    # Save the filtered dataset
    filtered_df.to_csv(output_file, sep=';', index=False)
    print(f"\nFiltered dataset saved to: {output_file}")
    
    # Show distribution of non-NA counts in filtered data
    filtered_counts = filtered_df[annotation_columns].notna().sum(axis=1)
    print(f"\nFiltered data non-NA annotation distribution:")
    for count in sorted(filtered_counts.unique()):
        num_rows = (filtered_counts == count).sum()
        print(f"  {count} annotations: {num_rows} rows")

def main():
    input_file = "data/ground_truth1.csv"
    output_file = "data/subset_well_annotated.csv"
    min_annotations = 8
    
    try:
        filter_well_annotated(input_file, output_file, min_annotations)
        print(f"\n✓ Successfully created {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 