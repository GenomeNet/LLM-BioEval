#!/usr/bin/env python3
"""
Direct import of microbe.cards data into the database, preserving TRUE/FALSE values correctly
"""
import pandas as pd
import sqlite3
import uuid
from datetime import datetime

def import_csv_to_db(csv_file, db_path='../microbellm.db'):
    """Import CSV directly to database preserving all values"""
    
    # Read CSV - force boolean columns to be read as strings
    print(f"Reading CSV file: {csv_file}")
    
    # Define columns that should be read as strings, not converted to boolean
    bool_cols = ['motility', 'extreme_environment_tolerance', 'biofilm_formation',
                 'animal_pathogenicity', 'health_association', 'host_association',
                 'plant_pathogenicity', 'spore_formation']
    
    # Read with dtype specification
    dtype_dict = {col: str for col in bool_cols}
    df = pd.read_csv(csv_file, dtype=dtype_dict)
    
    print(f"Loaded {len(df)} rows")
    
    # Verify the values were read correctly
    if 'spore_formation' in df.columns:
        print(f"Spore formation values: {df['spore_formation'].value_counts().head()}")
    
    # Create job_id for this import
    job_id = f"import_{uuid.uuid4().hex[:8]}"
    print(f"Created import job_id: {job_id}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First, delete any existing import data for these models
    models_to_update = df['model'].unique()
    placeholders = ','.join(['?' for _ in models_to_update])
    cursor.execute(f"""
        DELETE FROM processing_results 
        WHERE species_file = 'wa_with_gcount.txt' 
        AND model IN ({placeholders})
        AND job_id LIKE 'import%'
    """, models_to_update.tolist())
    print(f"Cleaned up old import data for {len(models_to_update)} models")
    
    # Prepare data for insertion
    inserted = 0
    failed = 0
    
    for _, row in df.iterrows():
        try:
            # Don't convert boolean values - keep them as-is from CSV
            cursor.execute("""
                INSERT INTO processing_results (
                    job_id, binomial_name, model, status, species_file,
                    system_template, user_template,
                    gram_staining, motility, aerophilicity,
                    extreme_environment_tolerance, biofilm_formation,
                    animal_pathogenicity, biosafety_level, health_association,
                    host_association, plant_pathogenicity, spore_formation,
                    hemolysis, cell_shape,
                    job_status, job_created_at
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    'completed', ?
                )
            """, (
                job_id,
                row['binomial_name'],
                row['model'],
                row.get('status', 'completed'),
                row['species_file'],
                row.get('system_template'),
                row.get('user_template'),
                row.get('gram_staining'),
                row.get('motility'),
                row.get('aerophilicity'),
                row.get('extreme_environment_tolerance'),
                row.get('biofilm_formation'),
                row.get('animal_pathogenicity'),
                row.get('biosafety_level'),
                row.get('health_association'),
                row.get('host_association'),
                row.get('plant_pathogenicity'),
                row.get('spore_formation'),
                row.get('hemolysis'),
                row.get('cell_shape'),
                datetime.now()
            ))
            inserted += 1
            
            if inserted % 1000 == 0:
                print(f"Inserted {inserted} rows...")
                conn.commit()
                
        except Exception as e:
            print(f"Failed to insert row for {row['binomial_name']}: {e}")
            failed += 1
    
    # Final commit
    conn.commit()
    
    # Verify the import
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT model) as models,
            COUNT(CASE WHEN spore_formation = 'TRUE' THEN 1 END) as spore_true,
            COUNT(CASE WHEN spore_formation = 'FALSE' THEN 1 END) as spore_false,
            COUNT(CASE WHEN motility = 'TRUE' THEN 1 END) as motility_true,
            COUNT(CASE WHEN motility = 'FALSE' THEN 1 END) as motility_false
        FROM processing_results 
        WHERE job_id = ?
    """, (job_id,))
    
    stats = cursor.fetchone()
    
    conn.close()
    
    print(f"\n" + "="*60)
    print(f"IMPORT COMPLETE")
    print(f"="*60)
    print(f"Successfully inserted: {inserted} rows")
    print(f"Failed: {failed} rows")
    print(f"\nVerification:")
    print(f"Total in DB: {stats[0]}")
    print(f"Models: {stats[1]}")
    print(f"Spore Formation - TRUE: {stats[2]}, FALSE: {stats[3]}")
    print(f"Motility - TRUE: {stats[4]}, FALSE: {stats[5]}")
    
    return job_id

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python direct_import.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    import_csv_to_db(csv_file)