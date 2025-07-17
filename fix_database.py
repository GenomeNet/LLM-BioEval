#!/usr/bin/env python3
"""Fix the database schema issue - ensure results table exists."""

import sqlite3
import os
import sys

# Add the microbellm module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microbellm import config

# Database path from config
db_path = config.DATABASE_PATH

def fix_database():
    """Fix the database schema to match what the web app expects."""
    
    print(f"Database path: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        print("Creating new database...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = [row[0] for row in cursor.fetchall()]
    print(f"\nExisting tables: {existing_tables}")
    
    # Create results table if it doesn't exist
    if 'results' not in existing_tables:
        print("\nCreating 'results' table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_file TEXT,
                binomial_name TEXT,
                model TEXT,
                system_template TEXT,
                user_template TEXT,
                status TEXT,
                result TEXT,
                error TEXT,
                timestamp TIMESTAMP,
                -- Parsed phenotype predictions
                gram_staining TEXT,
                motility TEXT,
                aerophilicity TEXT,
                extreme_environment_tolerance TEXT,
                biofilm_formation TEXT,
                animal_pathogenicity TEXT,
                biosafety_level TEXT,
                health_association TEXT,
                host_association TEXT,
                plant_pathogenicity TEXT,
                spore_formation TEXT,
                hemolysis TEXT,
                cell_shape TEXT,
                knowledge_group TEXT
            )
        ''')
        print("Created 'results' table")
    
    # Create combinations table if it doesn't exist
    if 'combinations' not in existing_tables:
        print("\nCreating 'combinations' table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS combinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_file TEXT,
                model TEXT,
                system_template TEXT,
                user_template TEXT,
                status TEXT DEFAULT 'pending',
                total_species INTEGER DEFAULT 0,
                completed_species INTEGER DEFAULT 0,
                failed_species INTEGER DEFAULT 0,
                submitted_species INTEGER DEFAULT 0,
                received_species INTEGER DEFAULT 0,
                successful_species INTEGER DEFAULT 0,
                retried_species INTEGER DEFAULT 0,
                timeout_species INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                UNIQUE(species_file, model, system_template, user_template)
            )
        ''')
        print("Created 'combinations' table")
    
    # Create other required tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS managed_models (
            model TEXT PRIMARY KEY
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS managed_species_files (
            species_file TEXT PRIMARY KEY
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS template_metadata (
            system_template TEXT,
            user_template TEXT,
            display_name TEXT,
            description TEXT,
            template_type TEXT DEFAULT 'phenotype',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (system_template, user_template)
        )
    ''')
    
    conn.commit()
    
    # Check if species_results table exists and has data
    if 'species_results' in existing_tables:
        cursor.execute("SELECT COUNT(*) FROM species_results")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"\nFound {count} rows in 'species_results' table")
            print("Note: You may need to migrate data from 'species_results' to 'results' table")
            print("This script does not automatically migrate data to prevent data loss")
            
            # Show sample data
            cursor.execute("SELECT * FROM species_results LIMIT 1")
            columns = [description[0] for description in cursor.description]
            print(f"\nColumns in species_results: {columns}")
    
    # Check final state
    cursor.execute("SELECT COUNT(*) FROM results")
    results_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM combinations")
    combinations_count = cursor.fetchone()[0]
    
    print(f"\nFinal state:")
    print(f"- results table: {results_count} rows")
    print(f"- combinations table: {combinations_count} rows")
    
    conn.close()
    print("\nDatabase schema fixed!")

if __name__ == "__main__":
    fix_database()