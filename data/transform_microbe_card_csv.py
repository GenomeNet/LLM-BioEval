#!/usr/bin/env python3
"""
Transform microbe.card CSV export to MicrobeLLM import format
"""
import pandas as pd
import sys
import os

def transform_csv(input_file, output_file=None):
    """Transform microbe.card CSV to MicrobeLLM import format"""
    
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Create new dataframe with correct column names
    transformed = pd.DataFrame()
    
    # Map column names (handle case and spacing)
    column_mapping = {
        'Binomial name': 'binomial_name',
        'Gram staining': 'gram_staining',
        'Motility': 'motility',
        'Aerophilicity': 'aerophilicity',
        'Extreme environment tolerance': 'extreme_environment_tolerance',
        'Biofilm formation': 'biofilm_formation',
        'Biosafety level': 'biosafety_level',
        'Host association': 'host_association',
        'Health association': 'health_association',
        'Plant pathogenicity': 'plant_pathogenicity',
        'Spore formation': 'spore_formation',
        'Hemolysis': 'hemolysis',
        'Cell shape': 'cell_shape',
        'Model': 'model',
        'Query template': 'user_template',
        'Infrence date and time': 'inference_datetime'
    }
    
    # Copy and rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            transformed[new_col] = df[old_col]
    
    # Fix boolean columns - convert True/False/1/0 to TRUE/FALSE strings
    boolean_columns = [
        'motility', 'extreme_environment_tolerance', 'biofilm_formation',
        'animal_pathogenicity', 'health_association', 'host_association',
        'plant_pathogenicity', 'spore_formation'
    ]
    
    for col in boolean_columns:
        if col in transformed.columns:
            # Log original value distribution
            orig_values = transformed[col].value_counts(dropna=False)
            print(f"\n{col} - Original values:")
            print(orig_values.head(10))
            
            # Convert various boolean representations to TRUE/FALSE
            transformed[col] = transformed[col].apply(lambda x: 
                'TRUE' if str(x).lower() in ['true', '1', 't', 'yes'] 
                else 'FALSE' if str(x).lower() in ['false', '0', 'f', 'no']
                else None  # Convert NaN and other values to None
            )
            
            # Log converted value distribution
            new_values = transformed[col].value_counts(dropna=False)
            print(f"\n{col} - After conversion:")
            print(new_values)
    
    # Add required columns
    transformed['status'] = 'completed'
    transformed['species_file'] = 'wa_with_gcount.txt'  # Match existing WA data
    
    # Transform aerophilicity to list format
    def format_aerophilicity(value):
        if pd.isna(value):
            return value
        # Handle multiple values separated by comma
        if ',' in str(value):
            values = [v.strip() for v in str(value).split(',')]
            return str(values).replace("'", "'")
        else:
            return f"['{value}']"
    
    transformed['aerophilicity'] = transformed['aerophilicity'].apply(format_aerophilicity)
    
    # Transform template paths
    def transform_template(value):
        if pd.isna(value):
            return value
        # Map old template names to new ones
        template_mapping = {
            'templates/query_template_system_pred1.txt': 'templates/user/template1_phenotype.txt',
            'templates/query_template_system_pred2.txt': 'templates/user/template2_phenotype.txt',
            # Add more mappings as needed
        }
        return template_mapping.get(value, value)
    
    transformed['user_template'] = transformed['user_template'].apply(transform_template)
    transformed['system_template'] = 'templates/system/template1_phenotype.txt'  # Match existing format
    
    # Reorder columns to match database schema exactly
    # This order matches the processing_results table structure
    column_order = [
        'binomial_name', 'model', 'status', 'species_file',
        'system_template', 'user_template',
        'gram_staining', 'motility', 'aerophilicity',
        'extreme_environment_tolerance', 'biofilm_formation',
        'animal_pathogenicity', 'biosafety_level', 'health_association',
        'host_association', 'plant_pathogenicity', 'spore_formation',
        'hemolysis', 'cell_shape'
    ]
    
    # Add any remaining columns not in the standard order
    remaining_cols = [col for col in transformed.columns if col not in column_order]
    final_order = column_order + remaining_cols
    
    # Only include columns that exist in the dataframe
    final_order = [col for col in final_order if col in transformed.columns]
    transformed = transformed[final_order]
    
    # Remove rows with missing binomial_name or model
    transformed = transformed.dropna(subset=['binomial_name', 'model'])
    
    # Save output
    if output_file is None:
        output_file = input_file.replace('.csv', '_transformed.csv')
    
    transformed.to_csv(output_file, index=False)
    
    print(f"\n" + "="*80)
    print(f"TRANSFORMATION COMPLETE")
    print(f"="*80)
    print(f"Transformed CSV saved to: {output_file}")
    print(f"Total rows: {len(transformed)}")
    
    # Log summary statistics
    print(f"\n--- Data Quality Summary ---")
    for col in boolean_columns:
        if col in transformed.columns:
            true_count = (transformed[col] == 'TRUE').sum()
            false_count = (transformed[col] == 'FALSE').sum()
            null_count = transformed[col].isna().sum()
            total = len(transformed)
            print(f"{col:30} TRUE: {true_count:6} ({true_count/total*100:.1f}%), FALSE: {false_count:6} ({false_count/total*100:.1f}%), NULL: {null_count:6} ({null_count/total*100:.1f}%)")
    
    print(f"\nFirst few rows:")
    print(transformed[['binomial_name', 'model', 'status', 'motility', 'spore_formation']].head())
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transform_microbe_card_csv.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    transform_csv(input_file, output_file)