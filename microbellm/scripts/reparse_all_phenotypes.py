#!/usr/bin/env python3
"""
Script to re-parse all phenotype data in the database with improved validation.
This will clean up invalid values and properly mark them.
"""

import sqlite3
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from microbellm.utils import parse_response, detect_template_type
from microbellm.config import DATABASE_PATH

def reparse_all_phenotype_data():
    """Re-parse all phenotype data with improved validation"""
    db_path = DATABASE_PATH
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get all phenotype results with raw responses
        cursor.execute("""
            SELECT r.id, r.binomial_name, r.result, r.user_template, r.model, r.species_file
            FROM results r
            WHERE r.result IS NOT NULL AND r.result != ''
                  AND r.status = 'completed'
                  AND (r.user_template LIKE '%phenotype%' 
                       OR r.gram_staining IS NOT NULL 
                       OR r.motility IS NOT NULL)
            ORDER BY r.species_file, r.model, r.binomial_name
        """)
        
        results = cursor.fetchall()
        total_results = len(results)
        
        print(f"Found {total_results} phenotype results to re-parse")
        
        if total_results == 0:
            print("No phenotype results found to re-parse")
            return
        
        # Track statistics
        updated_count = 0
        failed_count = 0
        invalid_fields_count = 0
        
        # Process each result
        for idx, (result_id, binomial_name, raw_response, user_template, model, species_file) in enumerate(results):
            if idx % 100 == 0:
                print(f"Processing {idx}/{total_results} ({idx/total_results*100:.1f}%)...")
            
            try:
                # Verify this is a phenotype template
                if detect_template_type(user_template) != 'phenotype':
                    continue
                
                # Parse the raw response using template validation
                parsed_result = parse_response(raw_response, user_template)
                
                if parsed_result:
                    # Check if there were any invalid fields
                    if 'invalid_fields' in parsed_result:
                        invalid_fields_count += 1
                        invalid_info = ", ".join([f"{f['field']}={f['value']}" for f in parsed_result['invalid_fields']])
                        print(f"  Warning: {binomial_name} ({model}) has invalid fields: {invalid_info}")
                    
                    # Handle aerophilicity as array - convert to string for database storage
                    aerophilicity = parsed_result.get('aerophilicity')
                    if isinstance(aerophilicity, list):
                        aerophilicity_str = str(aerophilicity)
                    else:
                        aerophilicity_str = aerophilicity
                    
                    # Update the phenotype fields in the database
                    cursor.execute("""
                        UPDATE results 
                        SET gram_staining = ?,
                            motility = ?,
                            aerophilicity = ?,
                            extreme_environment_tolerance = ?,
                            biofilm_formation = ?,
                            animal_pathogenicity = ?,
                            biosafety_level = ?,
                            health_association = ?,
                            host_association = ?,
                            plant_pathogenicity = ?,
                            spore_formation = ?,
                            hemolysis = ?,
                            cell_shape = ?
                        WHERE id = ?
                    """, (
                        parsed_result.get('gram_staining'),
                        parsed_result.get('motility'),
                        aerophilicity_str,
                        parsed_result.get('extreme_environment_tolerance'),
                        parsed_result.get('biofilm_formation'),
                        parsed_result.get('animal_pathogenicity'),
                        parsed_result.get('biosafety_level'),
                        parsed_result.get('health_association'),
                        parsed_result.get('host_association'),
                        parsed_result.get('plant_pathogenicity'),
                        parsed_result.get('spore_formation'),
                        parsed_result.get('hemolysis'),
                        parsed_result.get('cell_shape'),
                        result_id
                    ))
                    updated_count += 1
                else:
                    failed_count += 1
                    print(f"  Error: Failed to parse result for {binomial_name} ({model})")
                    
            except Exception as e:
                print(f"  Error re-parsing {binomial_name} ({model}): {e}")
                failed_count += 1
        
        # Commit changes
        conn.commit()
        
        print("\n" + "="*60)
        print("Re-parsing complete!")
        print(f"Total results processed: {total_results}")
        print(f"Successfully updated: {updated_count}")
        print(f"Failed to parse: {failed_count}")
        print(f"Results with invalid fields: {invalid_fields_count}")
        print("="*60)
        
    finally:
        conn.close()

if __name__ == "__main__":
    reparse_all_phenotype_data() 