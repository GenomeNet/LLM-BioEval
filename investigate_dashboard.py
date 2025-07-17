#!/usr/bin/env python3
"""Investigate why the dashboard shows no results."""

import sqlite3
import os
import sys

# Add the microbellm module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microbellm import config

# Database path from config
db_path = config.DATABASE_PATH

def investigate_dashboard():
    """Check why dashboard might show no results."""
    
    print(f"Database path: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Check combinations table
    print("\n=== COMBINATIONS TABLE ===")
    cursor.execute("SELECT COUNT(*) FROM combinations")
    total_combinations = cursor.fetchone()[0]
    print(f"Total combinations: {total_combinations}")
    
    # Check status distribution
    cursor.execute("SELECT status, COUNT(*) FROM combinations GROUP BY status")
    status_counts = cursor.fetchall()
    print("\nStatus distribution:")
    for status, count in status_counts:
        print(f"  {status}: {count}")
    
    # Check sample combinations
    cursor.execute("""
        SELECT species_file, model, system_template, user_template, status, 
               completed_species, total_species
        FROM combinations
        LIMIT 5
    """)
    print("\nSample combinations:")
    for row in cursor.fetchall():
        print(f"  File: {row[0]}, Model: {row[1]}")
        print(f"    Templates: {row[2]} / {row[3]}")
        print(f"    Status: {row[4]}, Progress: {row[5]}/{row[6]}")
    
    # 2. Check results table
    print("\n=== RESULTS TABLE ===")
    cursor.execute("SELECT COUNT(*) FROM results")
    total_results = cursor.fetchone()[0]
    print(f"Total results: {total_results}")
    
    # Check results with knowledge_group
    cursor.execute("SELECT COUNT(*) FROM results WHERE knowledge_group IS NOT NULL")
    knowledge_results = cursor.fetchone()[0]
    print(f"Results with knowledge_group: {knowledge_results}")
    
    # Check unique combinations in results
    cursor.execute("""
        SELECT COUNT(DISTINCT species_file || '|' || model || '|' || 
               system_template || '|' || user_template) 
        FROM results
    """)
    unique_result_combos = cursor.fetchone()[0]
    print(f"Unique combinations in results: {unique_result_combos}")
    
    # 3. Check for orphan results (results without combinations)
    print("\n=== CHECKING FOR ORPHAN RESULTS ===")
    cursor.execute("""
        SELECT COUNT(DISTINCT r.species_file || '|' || r.model || '|' || 
               r.system_template || '|' || r.user_template)
        FROM results r
        LEFT JOIN combinations c 
        ON r.species_file = c.species_file 
        AND r.model = c.model 
        AND r.system_template = c.system_template 
        AND r.user_template = c.user_template
        WHERE c.id IS NULL
    """)
    orphan_count = cursor.fetchone()[0]
    print(f"Combinations in results but not in combinations table: {orphan_count}")
    
    if orphan_count > 0:
        # Show some examples
        cursor.execute("""
            SELECT DISTINCT r.species_file, r.model, r.system_template, r.user_template,
                   COUNT(*) as result_count
            FROM results r
            LEFT JOIN combinations c 
            ON r.species_file = c.species_file 
            AND r.model = c.model 
            AND r.system_template = c.system_template 
            AND r.user_template = c.user_template
            WHERE c.id IS NULL
            GROUP BY r.species_file, r.model, r.system_template, r.user_template
            LIMIT 5
        """)
        print("\nExample orphan result combinations:")
        for row in cursor.fetchall():
            print(f"  File: {row[0]}, Model: {row[1]}, Count: {row[4]}")
            print(f"    System: {row[2]}")
            print(f"    User: {row[3]}")
    
    # 4. Check managed models and species files
    print("\n=== MANAGED ENTITIES ===")
    cursor.execute("SELECT COUNT(*) FROM managed_models")
    managed_models = cursor.fetchone()[0]
    print(f"Managed models: {managed_models}")
    
    cursor.execute("SELECT COUNT(*) FROM managed_species_files")
    managed_files = cursor.fetchone()[0]
    print(f"Managed species files: {managed_files}")
    
    # 5. Check if the issue is related to knowledge templates
    print("\n=== KNOWLEDGE TEMPLATE ANALYSIS ===")
    cursor.execute("""
        SELECT system_template, user_template, COUNT(*) as count
        FROM results
        WHERE knowledge_group IS NOT NULL
        GROUP BY system_template, user_template
        LIMIT 5
    """)
    print("Templates with knowledge data:")
    for row in cursor.fetchall():
        print(f"  System: {row[0]}")
        print(f"  User: {row[1]}")
        print(f"  Count: {row[2]}")
    
    conn.close()

if __name__ == "__main__":
    investigate_dashboard()