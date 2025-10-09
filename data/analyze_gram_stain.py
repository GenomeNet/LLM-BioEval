#!/usr/bin/env python3
"""
Analyze Gram_stain column in bugphyzz_all 2.csv dataset.
Lists unique values and their counts.
"""

import pandas as pd
from pathlib import Path

def analyze_gram_stain():
    # Read the CSV file
    csv_path = Path(__file__).parent / "bugphyzz_all 2.csv"
    df = pd.read_csv(csv_path)

    # Check if Gram_stain column exists
    if 'Gram_stain' not in df.columns:
        print("Error: 'Gram_stain' column not found in the dataset")
        print(f"Available columns: {', '.join(df.columns)}")
        return

    # Get value counts for Gram_stain column
    gram_stain_counts = df['Gram_stain'].value_counts(dropna=False)

    # Print summary statistics
    print("=" * 50)
    print("GRAM STAIN ANALYSIS")
    print("=" * 50)
    print(f"\nTotal entries: {len(df)}")
    print(f"Unique values in Gram_stain: {df['Gram_stain'].nunique(dropna=False)}")

    print("\n" + "-" * 50)
    print("VALUE COUNTS:")
    print("-" * 50)

    # Print each unique value and its count
    for value, count in gram_stain_counts.items():
        percentage = (count / len(df)) * 100
        # Handle NaN values for display
        display_value = "NA/Missing" if pd.isna(value) else value
        print(f"{display_value:20s}: {count:6d} ({percentage:6.2f}%)")

    # Additional statistics
    print("\n" + "-" * 50)
    print("SUMMARY:")
    print("-" * 50)

    non_na_count = df['Gram_stain'].notna().sum()
    na_count = df['Gram_stain'].isna().sum()

    print(f"Non-NA entries: {non_na_count} ({(non_na_count/len(df))*100:.2f}%)")
    print(f"NA entries:     {na_count} ({(na_count/len(df))*100:.2f}%)")

    # List actual unique non-NA values
    unique_values = df['Gram_stain'].dropna().unique()
    if len(unique_values) > 0:
        print(f"\nUnique non-NA values: {sorted(unique_values)}")

if __name__ == "__main__":
    analyze_gram_stain()