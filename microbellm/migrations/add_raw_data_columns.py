#!/usr/bin/env python
"""
Database migration to add raw data preservation columns to processing_results table.
This allows us to keep original predictions while storing normalized versions.
"""

import sqlite3
import sys
from pathlib import Path

def add_raw_data_columns(db_path):
    """Add raw data preservation columns to processing_results table."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(processing_results)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # Define raw data columns to add (matching each phenotype field)
        raw_columns = [
            ("raw_knowledge_group", "TEXT"),
            ("raw_gram_staining", "TEXT"),
            ("raw_motility", "TEXT"),
            ("raw_aerophilicity", "TEXT"),
            ("raw_extreme_environment_tolerance", "TEXT"),
            ("raw_biofilm_formation", "TEXT"),
            ("raw_animal_pathogenicity", "TEXT"),
            ("raw_biosafety_level", "TEXT"),
            ("raw_health_association", "TEXT"),
            ("raw_host_association", "TEXT"),
            ("raw_plant_pathogenicity", "TEXT"),
            ("raw_spore_formation", "TEXT"),
            ("raw_hemolysis", "TEXT"),
            ("raw_cell_shape", "TEXT"),
            ("raw_data_preserved", "INTEGER DEFAULT 0")  # Flag to indicate if raw data was preserved
        ]
        
        for column_name, column_def in raw_columns:
            if column_name not in existing_columns:
                print(f"Adding column: {column_name}")
                cursor.execute(f"ALTER TABLE processing_results ADD COLUMN {column_name} {column_def}")
                print(f"  ✓ Column {column_name} added successfully")
            else:
                print(f"  ⚠ Column {column_name} already exists, skipping")
        
        # Copy existing data to raw columns for records that haven't been validated yet
        print("\nPreserving existing unvalidated data as raw data...")
        cursor.execute("""
            UPDATE processing_results
            SET 
                raw_knowledge_group = knowledge_group,
                raw_gram_staining = gram_staining,
                raw_motility = motility,
                raw_aerophilicity = aerophilicity,
                raw_extreme_environment_tolerance = extreme_environment_tolerance,
                raw_biofilm_formation = biofilm_formation,
                raw_animal_pathogenicity = animal_pathogenicity,
                raw_biosafety_level = biosafety_level,
                raw_health_association = health_association,
                raw_host_association = host_association,
                raw_plant_pathogenicity = plant_pathogenicity,
                raw_spore_formation = spore_formation,
                raw_hemolysis = hemolysis,
                raw_cell_shape = cell_shape,
                raw_data_preserved = 1
            WHERE (validation_status IS NULL OR validation_status = 'unvalidated')
            AND raw_data_preserved = 0
        """)
        
        rows_updated = cursor.rowcount
        print(f"  ✓ Preserved raw data for {rows_updated} unvalidated records")
        
        # Create index on raw_data_preserved for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_data_preserved 
            ON processing_results(raw_data_preserved)
        """)
        print("✓ Index on raw_data_preserved created/verified")
        
        conn.commit()
        print("✓ Migration completed successfully")
        
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
    success = add_raw_data_columns(str(db_path))
    
    if not success:
        sys.exit(1)