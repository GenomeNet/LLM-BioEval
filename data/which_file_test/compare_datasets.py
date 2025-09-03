#!/usr/bin/env python3
"""
Compare two phenotype datasets to identify differences in columns, 
value distributions, and binomial name overlaps.
"""
import pandas as pd
import os
from datetime import datetime

def analyze_dataset(df, dataset_name):
    """Analyze a single dataset and return statistics"""
    stats = {
        'name': dataset_name,
        'shape': df.shape,
        'columns': list(df.columns),
        'column_count': len(df.columns),
        'row_count': len(df),
        'binomial_names': set(),
        'column_distributions': {}
    }
    
    # Get binomial names
    if 'Binomial name' in df.columns:
        stats['binomial_names'] = set(df['Binomial name'].dropna().unique())
    elif 'binomial_name' in df.columns:
        stats['binomial_names'] = set(df['binomial_name'].dropna().unique())
    
    # Analyze each column's distribution
    for col in df.columns:
        if col in ['Binomial name', 'binomial_name']:
            continue
            
        # Get value counts
        value_counts = df[col].value_counts(dropna=False)
        null_count = df[col].isna().sum()
        unique_count = df[col].nunique(dropna=False)
        
        stats['column_distributions'][col] = {
            'unique_values': unique_count,
            'null_count': null_count,
            'null_percentage': (null_count / len(df)) * 100,
            'value_counts': value_counts.to_dict(),
            'is_binary': unique_count <= 3  # Likely binary if <=3 unique values (including NaN)
        }
    
    return stats

def compare_datasets(file1, file2):
    """Compare two datasets and generate a report"""
    
    # Read the datasets
    print(f"Reading {file1}...")
    df1 = pd.read_csv(file1, sep='\t' if file1.endswith('.tsv') else ',')
    
    print(f"Reading {file2}...")
    df2 = pd.read_csv(file2, sep='\t' if file2.endswith('.tsv') else ',')
    
    # Analyze each dataset
    stats1 = analyze_dataset(df1, os.path.basename(file1))
    stats2 = analyze_dataset(df2, os.path.basename(file2))
    
    # Generate comparison report
    report = []
    report.append("=" * 80)
    report.append("DATASET COMPARISON REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Basic statistics
    report.append("DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"Dataset 1: {stats1['name']}")
    report.append(f"  - Shape: {stats1['shape'][0]} rows × {stats1['shape'][1]} columns")
    report.append(f"  - Unique binomial names: {len(stats1['binomial_names'])}")
    report.append("")
    report.append(f"Dataset 2: {stats2['name']}")
    report.append(f"  - Shape: {stats2['shape'][0]} rows × {stats2['shape'][1]} columns")
    report.append(f"  - Unique binomial names: {len(stats2['binomial_names'])}")
    report.append("")
    
    # Column comparison
    report.append("COLUMN COMPARISON")
    report.append("-" * 40)
    
    cols1 = set(stats1['columns'])
    cols2 = set(stats2['columns'])
    
    common_cols = cols1.intersection(cols2)
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    report.append(f"Common columns: {len(common_cols)}")
    report.append(f"Columns only in {stats1['name']}: {len(only_in_1)}")
    if only_in_1:
        for col in sorted(only_in_1):
            report.append(f"  - {col}")
    
    report.append(f"\nColumns only in {stats2['name']}: {len(only_in_2)}")
    if only_in_2:
        for col in sorted(only_in_2):
            report.append(f"  - {col}")
    report.append("")
    
    # Binomial name comparison
    report.append("BINOMIAL NAME COMPARISON")
    report.append("-" * 40)
    
    common_names = stats1['binomial_names'].intersection(stats2['binomial_names'])
    only_names_1 = stats1['binomial_names'] - stats2['binomial_names']
    only_names_2 = stats2['binomial_names'] - stats1['binomial_names']
    
    report.append(f"Common binomial names: {len(common_names)}")
    report.append(f"Names only in {stats1['name']}: {len(only_names_1)}")
    report.append(f"Names only in {stats2['name']}: {len(only_names_2)}")
    
    if only_names_1 and len(only_names_1) <= 10:
        report.append(f"\nSample of names only in {stats1['name']}:")
        for name in sorted(list(only_names_1)[:10]):
            report.append(f"  - {name}")
    
    if only_names_2 and len(only_names_2) <= 10:
        report.append(f"\nSample of names only in {stats2['name']}:")
        for name in sorted(list(only_names_2)[:10]):
            report.append(f"  - {name}")
    report.append("")
    
    # Column distribution comparison for common columns
    report.append("VALUE DISTRIBUTIONS FOR COMMON COLUMNS")
    report.append("-" * 40)
    
    # Sort columns by type (binary first, then by name)
    binary_cols = []
    categorical_cols = []
    
    for col in sorted(common_cols):
        if col in ['Binomial name', 'binomial_name']:
            continue
        
        is_binary_1 = stats1['column_distributions'].get(col, {}).get('is_binary', False)
        is_binary_2 = stats2['column_distributions'].get(col, {}).get('is_binary', False)
        
        if is_binary_1 or is_binary_2:
            binary_cols.append(col)
        else:
            categorical_cols.append(col)
    
    # Report binary columns
    if binary_cols:
        report.append("\nBINARY/BOOLEAN COLUMNS:")
        for col in binary_cols:
            report.append(f"\n  {col}:")
            
            dist1 = stats1['column_distributions'].get(col, {})
            dist2 = stats2['column_distributions'].get(col, {})
            
            report.append(f"    In {stats1['name']}:")
            if dist1:
                report.append(f"      - Null/NaN: {dist1['null_count']} ({dist1['null_percentage']:.1f}%)")
                for value, count in sorted(dist1['value_counts'].items(), key=lambda x: -x[1]):
                    if pd.notna(value):
                        pct = (count / stats1['row_count']) * 100
                        report.append(f"      - {value}: {count} ({pct:.1f}%)")
            
            report.append(f"    In {stats2['name']}:")
            if dist2:
                report.append(f"      - Null/NaN: {dist2['null_count']} ({dist2['null_percentage']:.1f}%)")
                for value, count in sorted(dist2['value_counts'].items(), key=lambda x: -x[1]):
                    if pd.notna(value):
                        pct = (count / stats2['row_count']) * 100
                        report.append(f"      - {value}: {count} ({pct:.1f}%)")
    
    # Report categorical columns
    if categorical_cols:
        report.append("\nCATEGORICAL COLUMNS:")
        for col in categorical_cols:
            report.append(f"\n  {col}:")
            
            dist1 = stats1['column_distributions'].get(col, {})
            dist2 = stats2['column_distributions'].get(col, {})
            
            report.append(f"    In {stats1['name']}:")
            if dist1:
                report.append(f"      - Unique values: {dist1['unique_values']}")
                report.append(f"      - Null/NaN: {dist1['null_count']} ({dist1['null_percentage']:.1f}%)")
                # Show top 5 values
                top_values = sorted(dist1['value_counts'].items(), key=lambda x: -x[1])[:5]
                for value, count in top_values:
                    if pd.notna(value):
                        pct = (count / stats1['row_count']) * 100
                        report.append(f"      - {value}: {count} ({pct:.1f}%)")
                if len(dist1['value_counts']) > 5:
                    report.append(f"      ... and {len(dist1['value_counts']) - 5} more values")
            
            report.append(f"    In {stats2['name']}:")
            if dist2:
                report.append(f"      - Unique values: {dist2['unique_values']}")
                report.append(f"      - Null/NaN: {dist2['null_count']} ({dist2['null_percentage']:.1f}%)")
                # Show top 5 values
                top_values = sorted(dist2['value_counts'].items(), key=lambda x: -x[1])[:5]
                for value, count in top_values:
                    if pd.notna(value):
                        pct = (count / stats2['row_count']) * 100
                        report.append(f"      - {value}: {count} ({pct:.1f}%)")
                if len(dist2['value_counts']) > 5:
                    report.append(f"      ... and {len(dist2['value_counts']) - 5} more values")
    
    # Columns unique to each dataset
    if only_in_1:
        report.append("\n" + "=" * 40)
        report.append(f"COLUMNS UNIQUE TO {stats1['name'].upper()}")
        report.append("-" * 40)
        
        for col in sorted(only_in_1):
            dist = stats1['column_distributions'].get(col, {})
            if dist:
                report.append(f"\n  {col}:")
                report.append(f"    - Unique values: {dist['unique_values']}")
                report.append(f"    - Null/NaN: {dist['null_count']} ({dist['null_percentage']:.1f}%)")
                # Show top 3 values
                top_values = sorted(dist['value_counts'].items(), key=lambda x: -x[1])[:3]
                for value, count in top_values:
                    if pd.notna(value):
                        pct = (count / stats1['row_count']) * 100
                        report.append(f"    - {value}: {count} ({pct:.1f}%)")
    
    if only_in_2:
        report.append("\n" + "=" * 40)
        report.append(f"COLUMNS UNIQUE TO {stats2['name'].upper()}")
        report.append("-" * 40)
        
        for col in sorted(only_in_2):
            dist = stats2['column_distributions'].get(col, {})
            if dist:
                report.append(f"\n  {col}:")
                report.append(f"    - Unique values: {dist['unique_values']}")
                report.append(f"    - Null/NaN: {dist['null_count']} ({dist['null_percentage']:.1f}%)")
                # Show top 3 values
                top_values = sorted(dist['value_counts'].items(), key=lambda x: -x[1])[:3]
                for value, count in top_values:
                    if pd.notna(value):
                        pct = (count / stats2['row_count']) * 100
                        report.append(f"    - {value}: {count} ({pct:.1f}%)")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    # File paths
    file1 = "s6_missing_phenotypes.tsv"
    file2 = "s6_old.tsv"
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"Error: {file1} not found!")
        return
    
    if not os.path.exists(file2):
        print(f"Error: {file2} not found!")
        return
    
    # Generate report
    print("Comparing datasets...")
    report = compare_datasets(file1, file2)
    
    # Save report
    output_file = "dataset_comparison_report.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")
    print("\n" + "=" * 40)
    print("SUMMARY:")
    
    # Also print to console
    print(report)

if __name__ == "__main__":
    main()