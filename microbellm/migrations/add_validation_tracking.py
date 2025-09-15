#!/usr/bin/env python
"""
Database migration to add validation tracking columns to processing_results table.
"""

import sqlite3
import sys
from pathlib import Path

def add_validation_columns(db_path):
    """Add validation tracking columns to processing_results table."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(processing_results)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        columns_to_add = [
            ("validation_status", "TEXT DEFAULT 'unvalidated'"),
            ("validation_date", "TIMESTAMP"),
            ("validation_log", "TEXT")
        ]
        
        for column_name, column_def in columns_to_add:
            if column_name not in existing_columns:
                print(f"Adding column: {column_name}")
                cursor.execute(f"ALTER TABLE processing_results ADD COLUMN {column_name} {column_def}")
                print(f"  ✓ Column {column_name} added successfully")
            else:
                print(f"  ⚠ Column {column_name} already exists, skipping")
        
        # Create index on validation_status for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_validation_status 
            ON processing_results(validation_status)
        """)
        print("✓ Index on validation_status created/verified")
        
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
    success = add_validation_columns(str(db_path))
    
    if not success:
        sys.exit(1)