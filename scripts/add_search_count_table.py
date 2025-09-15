#!/usr/bin/env python3
"""
Script to add search_count table to the database and import data from wa_with_gcount.txt
"""

import sqlite3
import csv
from pathlib import Path

# Get database path
DB_PATH = Path(__file__).parent.parent / "microbellm.db"
DATA_PATH = Path(__file__).parent.parent / "data" / "wa_with_gcount.txt"

def create_search_count_table():
    """Create the search_count table if it doesn't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_count (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            binomial_name TEXT UNIQUE NOT NULL,
            search_count INTEGER NOT NULL,
            import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_search_count_binomial 
        ON search_count(binomial_name)
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ Created search_count table in {DB_PATH}")

def import_search_count_data():
    """Import data from wa_with_gcount.txt"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("DELETE FROM search_count")
    
    # Read and import CSV data
    with open(DATA_PATH, 'r') as f:
        reader = csv.DictReader(f)
        row_count = 0
        
        for row in reader:
            binomial_name = row['Binomial name'].strip()
            search_count_str = row['Search count'].strip()
            
            # Skip empty search counts
            if not search_count_str:
                continue
                
            try:
                search_count = int(search_count_str)
            except ValueError:
                print(f"  Warning: Skipping invalid search count for {binomial_name}: '{search_count_str}'")
                continue
            
            cursor.execute("""
                INSERT INTO search_count (binomial_name, search_count)
                VALUES (?, ?)
            """, (binomial_name, search_count))
            
            row_count += 1
    
    conn.commit()
    
    # Verify import
    cursor.execute("SELECT COUNT(*) FROM search_count")
    total_count = cursor.fetchone()[0]
    
    # Get some statistics
    cursor.execute("""
        SELECT 
            MIN(search_count) as min_count,
            MAX(search_count) as max_count,
            AVG(search_count) as avg_count
        FROM search_count
    """)
    stats = cursor.fetchone()
    
    conn.close()
    
    print(f"✓ Imported {total_count} species with search counts")
    print(f"  Min count: {stats[0]:,}")
    print(f"  Max count: {stats[1]:,}")
    print(f"  Avg count: {stats[2]:,.0f}")

if __name__ == "__main__":
    print(f"Database: {DB_PATH}")
    print(f"Data file: {DATA_PATH}")
    
    create_search_count_table()
    import_search_count_data()
    
    print("\n✓ Search count data successfully added to database!")