#!/usr/bin/env python3
"""
Merge Animal pathogenicity column from s6_old.tsv into s6_missing_phenotypes.tsv
"""
import pandas as pd
import sys

def merge_animal_pathogenicity():
    """Merge the missing Animal pathogenicity column"""
    
    print("Reading s6_missing_phenotypes.tsv...")
    # Read with low_memory=False to avoid dtype warnings
    df_missing = pd.read_csv('s6_missing_phenotypes.tsv', sep='\t', low_memory=False)
    
    print("Reading s6_old.tsv...")
    df_old = pd.read_csv('s6_old.tsv', sep='\t', low_memory=False)
    
    # Check if Animal pathogenicity exists in old file
    if 'Animal pathogenicity' not in df_old.columns:
        print("Error: 'Animal pathogenicity' column not found in s6_old.tsv")
        return False
    
    # Check if the datasets align by binomial name and model
    print("\nVerifying data alignment...")
    
    # Create a unique key for matching rows
    df_missing['_key'] = df_missing['Binomial name'].astype(str) + '|' + df_missing['Model'].astype(str)
    df_old['_key'] = df_old['Binomial name'].astype(str) + '|' + df_old['Model'].astype(str)
    
    # Check if all keys match
    missing_keys = set(df_missing['_key'])
    old_keys = set(df_old['_key'])
    
    if missing_keys != old_keys:
        print(f"Warning: Keys don't match perfectly!")
        print(f"  Keys in s6_missing_phenotypes but not in s6_old: {len(missing_keys - old_keys)}")
        print(f"  Keys in s6_old but not in s6_missing_phenotypes: {len(old_keys - missing_keys)}")
    else:
        print("✓ All binomial name + model combinations match between files")
    
    # Sort both dataframes by the key to ensure alignment
    df_missing = df_missing.sort_values('_key').reset_index(drop=True)
    df_old = df_old.sort_values('_key').reset_index(drop=True)
    
    # Verify row-by-row alignment
    alignment_check = (df_missing['_key'] == df_old['_key']).all()
    if alignment_check:
        print("✓ Rows are properly aligned")
    else:
        print("Error: Rows are not aligned after sorting!")
        return False
    
    # Copy the Animal pathogenicity column
    print("\nCopying Animal pathogenicity column...")
    df_missing['Animal pathogenicity'] = df_old['Animal pathogenicity']
    
    # Remove the temporary key column
    df_missing = df_missing.drop('_key', axis=1)
    
    # Remove the "Unnamed: 1" column if it exists (it's all NaN)
    if 'Unnamed: 1' in df_missing.columns:
        print("Removing empty 'Unnamed: 1' column...")
        df_missing = df_missing.drop('Unnamed: 1', axis=1)
    
    # Verify the merge
    print("\nVerifying merge:")
    print(f"  Total rows: {len(df_missing)}")
    print(f"  Total columns: {len(df_missing.columns)}")
    print(f"  Animal pathogenicity distribution:")
    print(f"    - True: {(df_missing['Animal pathogenicity'] == True).sum()}")
    print(f"    - False: {(df_missing['Animal pathogenicity'] == False).sum()}")
    print(f"    - NaN: {df_missing['Animal pathogenicity'].isna().sum()}")
    
    # Save the merged file
    output_file = 's6_complete.tsv'
    print(f"\nSaving merged data to {output_file}...")
    df_missing.to_csv(output_file, sep='\t', index=False)
    
    print(f"✓ Successfully created {output_file} with Animal pathogenicity column added")
    
    # Display column list
    print(f"\nFinal columns ({len(df_missing.columns)}):")
    for i, col in enumerate(df_missing.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return True

if __name__ == "__main__":
    success = merge_animal_pathogenicity()
    if not success:
        sys.exit(1)