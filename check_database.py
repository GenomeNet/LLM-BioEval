#!/usr/bin/env python3
"""Script to investigate why knowledge analysis shows results but dashboard doesn't."""

import sqlite3
import os

# Database path - check multiple possible locations
possible_paths = [
    os.path.join(os.path.dirname(__file__), 'microbellm', 'microbellm.db'),
    os.path.join(os.path.dirname(__file__), 'microbellm.db'),
    os.path.join(os.path.dirname(__file__), 'microbebench.db'),
]

db_path = None
for path in possible_paths:
    if os.path.exists(path):
        db_path = path
        print(f"Found database at: {path}")
        break

def check_database():
    """Check the database for results and combinations."""
    
    if not db_path or not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Check if results table has data
    print("=== Checking RESULTS table ===")
    cursor.execute("SELECT COUNT(*) FROM results")
    total_results = cursor.fetchone()[0]
    print(f"Total results: {total_results}")
    
    # Check results with knowledge_group
    cursor.execute("SELECT COUNT(*) FROM results WHERE knowledge_group IS NOT NULL")
    knowledge_results = cursor.fetchone()[0]
    print(f"Results with knowledge_group: {knowledge_results}")
    
    # Get sample of species files from results
    cursor.execute("SELECT DISTINCT species_file FROM results LIMIT 10")
    result_species_files = cursor.fetchall()
    print(f"\nSample species files in results:")
    for file in result_species_files:
        print(f"  - {file[0]}")
    
    # 2. Check if combinations table has data
    print("\n=== Checking COMBINATIONS table ===")
    cursor.execute("SELECT COUNT(*) FROM combinations")
    total_combinations = cursor.fetchone()[0]
    print(f"Total combinations: {total_combinations}")
    
    # Get sample of species files from combinations
    cursor.execute("SELECT DISTINCT species_file FROM combinations LIMIT 10")
    combo_species_files = cursor.fetchall()
    print(f"\nSample species files in combinations:")
    for file in combo_species_files:
        print(f"  - {file[0]}")
    
    # 3. Check for mismatches
    print("\n=== Checking for mismatches ===")
    
    # Find results without corresponding combinations
    cursor.execute("""
        SELECT DISTINCT r.species_file, r.model, r.system_template, r.user_template
        FROM results r
        LEFT JOIN combinations c 
        ON r.species_file = c.species_file 
        AND r.model = c.model 
        AND r.system_template = c.system_template 
        AND r.user_template = c.user_template
        WHERE c.id IS NULL
        LIMIT 10
    """)
    orphan_results = cursor.fetchall()
    
    if orphan_results:
        print(f"\nFound {len(orphan_results)} result combinations without matching combination entries:")
        for result in orphan_results:
            print(f"  - File: {result[0]}, Model: {result[1]}")
            print(f"    System: {result[2]}")
            print(f"    User: {result[3]}")
    else:
        print("All results have matching combination entries")
    
    # 4. Check template paths
    print("\n=== Checking template paths ===")
    cursor.execute("""
        SELECT DISTINCT system_template, user_template 
        FROM results 
        WHERE knowledge_group IS NOT NULL 
        LIMIT 5
    """)
    templates = cursor.fetchall()
    print("Sample templates from results with knowledge data:")
    for system, user in templates:
        print(f"  System: {system}")
        print(f"  User: {user}")
        print()
    
    conn.close()

if __name__ == "__main__":
    check_database()