#!/usr/bin/env python3
"""
Script to filter ground truth dataset for well-annotated samples.
Keeps only rows with at least 8 non-NA annotations AND non-empty binomial_name.
"""

import pandas as pd
import sys

def filter_well_annotated(input_file, output_file, min_annotations=8):
    """
    Filter CSV file to keep only rows with at least min_annotations non-NA values
    AND non-empty binomial_name.
    
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
    
    # Check binomial_name integrity first
    if 'binomial_name' not in df.columns:
        print("Error: No 'binomial_name' column found!")
        return
    
    binomial_name_series = df['binomial_name']
    empty_binomial_mask = (
        binomial_name_series.isna() | 
        binomial_name_series.isnull() | 
        (binomial_name_series.astype(str).str.strip() == '') |
        (binomial_name_series.astype(str) == 'nan')
    )
    empty_binomial_count = empty_binomial_mask.sum()
    
    print(f"\nBinomial name analysis:")
    print(f"  Total entries: {len(df)}")
    print(f"  Empty binomial_name entries: {empty_binomial_count}")
    print(f"  Valid binomial_name entries: {len(df) - empty_binomial_count}")
    
    if empty_binomial_count > 0:
        print(f"  -> Will exclude {empty_binomial_count} rows with empty binomial_name")
    
    # Get annotation columns (exclude the first column which is binomial_name)
    annotation_columns = df.columns[1:]  # All columns except the first one
    print(f"\nNumber of annotation columns: {len(annotation_columns)}")
    
    # Count non-NA values in annotation columns for each row
    non_na_counts = df[annotation_columns].notna().sum(axis=1)
    
    print(f"Non-NA annotation counts statistics:")
    print(f"  Mean: {non_na_counts.mean():.2f}")
    print(f"  Median: {non_na_counts.median():.2f}")
    print(f"  Min: {non_na_counts.min()}")
    print(f"  Max: {non_na_counts.max()}")
    
    # Apply BOTH filters:
    # 1. At least min_annotations non-NA values
    # 2. Non-empty binomial_name
    annotation_filter = non_na_counts >= min_annotations
    binomial_name_filter = ~empty_binomial_mask
    combined_filter = annotation_filter & binomial_name_filter
    
    filtered_df = df[combined_filter].copy()
    
    print(f"\nFiltering results:")
    print(f"  Rows with >= {min_annotations} annotations: {annotation_filter.sum()}")
    print(f"  Rows with valid binomial_name: {binomial_name_filter.sum()}")
    print(f"  Rows passing BOTH filters: {combined_filter.sum()}")
    print(f"  Filtered dataset shape: {filtered_df.shape}")
    print(f"  Overall retention rate: {(len(filtered_df) / len(df)) * 100:.1f}%")
    
    # Verify no empty binomial_name in filtered data
    filtered_empty_count = (
        filtered_df['binomial_name'].isna() | 
        filtered_df['binomial_name'].isnull() | 
        (filtered_df['binomial_name'].astype(str).str.strip() == '') |
        (filtered_df['binomial_name'].astype(str) == 'nan')
    ).sum()
    
    print(f"\nQuality check on filtered data:")
    print(f"  Empty binomial_name entries: {filtered_empty_count}")
    
    if filtered_empty_count == 0:
        print("  ✅ All entries have valid binomial_name!")
    else:
        print(f"  ❌ Warning: {filtered_empty_count} entries still have empty binomial_name")
    
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
    output_file = "data/subset_well_annotated_clean.csv"
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