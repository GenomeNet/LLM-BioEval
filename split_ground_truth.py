#!/usr/bin/env python3
"""
Script to split ground_truth1_cleaned.csv into two subsets:
- WA (Well Annotated): species with 5 or fewer missing trait entries
- LA (Less Annotated): species with more than 5 missing trait entries

Missing values are represented as "NA" in the CSV.
"""

import csv
import sys
import os
from pathlib import Path


def count_missing_values(row, exclude_first_column=True):
    """
    Count the number of 'NA' values in a row.
    
    Args:
        row: List of values from a CSV row
        exclude_first_column: If True, exclude the first column (species name) from counting
        
    Returns:
        Number of 'NA' values
    """
    start_index = 1 if exclude_first_column else 0
    return sum(1 for value in row[start_index:] if value.strip() == 'NA')


def split_ground_truth(input_file, output_dir='data'):
    """
    Split the ground truth CSV into WA and LA subsets.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the output files
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Define output files
    wa_file = output_path / 'ground_truth_WA.csv'
    la_file = output_path / 'ground_truth_LA.csv'
    
    # Statistics
    total_rows = 0
    wa_count = 0
    la_count = 0
    
    print(f"Reading from: {input_path}")
    print(f"Output directory: {output_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(wa_file, 'w', encoding='utf-8', newline='') as wa_out, \
             open(la_file, 'w', encoding='utf-8', newline='') as la_out:
            
            # Create CSV readers and writers
            reader = csv.reader(infile, delimiter=';')
            wa_writer = csv.writer(wa_out, delimiter=';')
            la_writer = csv.writer(la_out, delimiter=';')
            
            # Process header
            header = next(reader)
            wa_writer.writerow(header)
            la_writer.writerow(header)
            
            print(f"Header: {header}")
            print(f"Total columns: {len(header)}")
            print(f"Trait columns (excluding species name): {len(header) - 1}")
            print()
            
            # Process data rows
            for row in reader:
                total_rows += 1
                
                # Skip empty rows
                if not row or len(row) != len(header):
                    continue
                
                # Count missing values (excluding species name column)
                missing_count = count_missing_values(row, exclude_first_column=True)
                
                # Classify as WA (≤5 missing) or LA (>5 missing)
                if missing_count <= 5:
                    wa_writer.writerow(row)
                    wa_count += 1
                else:
                    la_writer.writerow(row)
                    la_count += 1
                
                # Progress indicator
                if total_rows % 1000 == 0:
                    print(f"Processed {total_rows} rows...")
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SPLITTING COMPLETE")
    print("="*50)
    print(f"Total rows processed: {total_rows}")
    print(f"WA (Well Annotated) subset: {wa_count} species (≤5 missing traits)")
    print(f"LA (Less Annotated) subset: {la_count} species (>5 missing traits)")
    print(f"WA percentage: {wa_count/total_rows*100:.1f}%")
    print(f"LA percentage: {la_count/total_rows*100:.1f}%")
    print()
    print("Output files:")
    print(f"- WA subset: {wa_file}")
    print(f"- LA subset: {la_file}")
    
    # Additional statistics
    print("\n" + "="*50)
    print("MISSING VALUE DISTRIBUTION SAMPLE")
    print("="*50)
    
    # Analyze distribution of missing values
    missing_distribution = {}
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=';')
        next(reader)  # Skip header
        
        for row in reader:
            if row and len(row) == len(header):
                missing_count = count_missing_values(row, exclude_first_column=True)
                missing_distribution[missing_count] = missing_distribution.get(missing_count, 0) + 1
    
    print("Missing values distribution:")
    for missing_count in sorted(missing_distribution.keys()):
        count = missing_distribution[missing_count]
        percentage = count / total_rows * 100
        marker = "WA" if missing_count <= 5 else "LA"
        print(f"  {missing_count:2d} missing: {count:5d} species ({percentage:5.1f}%) [{marker}]")


def main():
    """Main function to run the splitting script."""
    input_file = 'data/ground_truth1_cleaned.csv'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        print("Usage: python split_ground_truth.py [input_file]")
        print(f"Default input file: {input_file}")
        sys.exit(1)
    
    split_ground_truth(input_file)


if __name__ == "__main__":
    main() 