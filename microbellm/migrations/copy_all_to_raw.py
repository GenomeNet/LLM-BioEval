#!/usr/bin/env python
"""
One-time migration to copy ALL existing data to raw columns.
This ensures we have a backup of all original data before validation.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

def copy_all_to_raw(db_path):
    """Copy all existing phenotype data to raw columns."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print(f"Starting migration at {datetime.now()}")
        print("This will copy ALL existing data to raw columns...")
        
        # First, count how many records we need to update
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processing_results 
            WHERE raw_data_preserved = 0 OR raw_data_preserved IS NULL
        """)
        total_records = cursor.fetchone()[0]
        print(f"Found {total_records} records to update")
        
        if total_records == 0:
            print("All records already have raw data preserved!")
            return True
        
        # Copy all phenotype data to raw columns
        print(f"Copying data to raw columns...")
        cursor.execute("""
            UPDATE processing_results
            SET 
                raw_knowledge_group = COALESCE(raw_knowledge_group, knowledge_group),
                raw_gram_staining = COALESCE(raw_gram_staining, gram_staining),
                raw_motility = COALESCE(raw_motility, motility),
                raw_aerophilicity = COALESCE(raw_aerophilicity, aerophilicity),
                raw_extreme_environment_tolerance = COALESCE(raw_extreme_environment_tolerance, extreme_environment_tolerance),
                raw_biofilm_formation = COALESCE(raw_biofilm_formation, biofilm_formation),
                raw_animal_pathogenicity = COALESCE(raw_animal_pathogenicity, animal_pathogenicity),
                raw_biosafety_level = COALESCE(raw_biosafety_level, biosafety_level),
                raw_health_association = COALESCE(raw_health_association, health_association),
                raw_host_association = COALESCE(raw_host_association, host_association),
                raw_plant_pathogenicity = COALESCE(raw_plant_pathogenicity, plant_pathogenicity),
                raw_spore_formation = COALESCE(raw_spore_formation, spore_formation),
                raw_hemolysis = COALESCE(raw_hemolysis, hemolysis),
                raw_cell_shape = COALESCE(raw_cell_shape, cell_shape),
                raw_data_preserved = 1
            WHERE raw_data_preserved = 0 OR raw_data_preserved IS NULL
        """)
        
        rows_updated = cursor.rowcount
        print(f"✓ Copied data for {rows_updated} records")
        
        # Verify the update
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processing_results 
            WHERE raw_data_preserved = 1
        """)
        preserved_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processing_results
        """)
        total_count = cursor.fetchone()[0]
        
        print(f"\nVerification:")
        print(f"  Total records: {total_count}")
        print(f"  Records with raw data preserved: {preserved_count}")
        print(f"  Records without raw data: {total_count - preserved_count}")
        
        conn.commit()
        print(f"\n✓ Migration completed successfully at {datetime.now()}")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        return False
    finally:
        conn.close()
    
    return True


if __name__ == "__main__":
    # Get database path
    db_path = Path(__file__).parent.parent.parent / "microbellm.db"
    
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        sys.exit(1)
    
    print(f"Migrating database: {db_path}")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("This will copy ALL existing data to raw columns. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Migration cancelled.")
        sys.exit(0)
    
    success = copy_all_to_raw(str(db_path))
    
    if not success:
        sys.exit(1)