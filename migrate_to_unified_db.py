#!/usr/bin/env python3
"""
Migrate data from species_results table to results table for unified database architecture.
This ensures a single source of truth where both admin dashboard and components use the same table.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Database path
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "microbellm_jobs.db"

def migrate_species_results_to_results():
    """Migrate all data from species_results table to results table"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # First, check if species_results table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='species_results'
        """)
        if not cursor.fetchone():
            print("species_results table does not exist. Nothing to migrate.")
            return
        
        # Get count of records in species_results
        cursor.execute("SELECT COUNT(*) FROM species_results")
        total_count = cursor.fetchone()[0]
        print(f"Found {total_count} records in species_results table")
        
        if total_count == 0:
            print("No records to migrate.")
            return
        
        # Get all species_results with their combination info
        cursor.execute("""
            SELECT sr.id, sr.combination_id, sr.binomial_name, sr.result, 
                   sr.status, sr.error, sr.created_at,
                   c.species_file, c.model, c.system_template, c.user_template
            FROM species_results sr
            JOIN combinations c ON sr.combination_id = c.id
            ORDER BY sr.id
        """)
        
        all_results = cursor.fetchall()
        migrated = 0
        skipped = 0
        errors = 0
        
        for row in all_results:
            sr_id, combo_id, binomial_name, result_json, status, error, created_at, \
                species_file, model, system_template, user_template = row
            
            # Check if this record already exists in results table
            cursor.execute("""
                SELECT COUNT(*) FROM results 
                WHERE species_file = ? AND binomial_name = ? AND model = ? 
                AND system_template = ? AND user_template = ?
            """, (species_file, binomial_name, model, system_template, user_template))
            
            if cursor.fetchone()[0] > 0:
                print(f"  Skipping {binomial_name} - already exists in results table")
                skipped += 1
                continue
            
            # Parse the result JSON to extract knowledge_group and phenotypes
            knowledge_group = None
            phenotype_data = {}
            
            if result_json and status == 'completed':
                try:
                    result_dict = json.loads(result_json)
                    
                    # Extract knowledge_group
                    knowledge_group = result_dict.get('knowledge_group') or result_dict.get('knowledge_level')
                    
                    # Extract phenotypes
                    if 'phenotypes' in result_dict:
                        phenotype_data = result_dict['phenotypes']
                    
                    # Also check for individual phenotype fields at top level
                    phenotype_fields = ['gram_staining', 'motility', 'aerophilicity', 
                                      'extreme_environment_tolerance', 'biofilm_formation',
                                      'animal_pathogenicity', 'biosafety_level', 
                                      'health_association', 'host_association',
                                      'plant_pathogenicity', 'spore_formation', 
                                      'hemolysis', 'cell_shape']
                    
                    for field in phenotype_fields:
                        if field in result_dict and field not in phenotype_data:
                            phenotype_data[field] = result_dict[field]
                            
                except json.JSONDecodeError:
                    print(f"  Warning: Could not parse JSON for {binomial_name}")
            
            try:
                # Insert into results table
                cursor.execute("""
                    INSERT INTO results (
                        species_file, binomial_name, model, system_template, user_template,
                        status, result, error, knowledge_group, created_at,
                        gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                        biofilm_formation, animal_pathogenicity, biosafety_level,
                        health_association, host_association, plant_pathogenicity,
                        spore_formation, hemolysis, cell_shape
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    species_file, binomial_name, model, system_template, user_template,
                    status, result_json, error, knowledge_group, created_at or datetime.now(),
                    phenotype_data.get('gram_staining'),
                    phenotype_data.get('motility'),
                    phenotype_data.get('aerophilicity'),
                    phenotype_data.get('extreme_environment_tolerance'),
                    phenotype_data.get('biofilm_formation'),
                    phenotype_data.get('animal_pathogenicity'),
                    phenotype_data.get('biosafety_level'),
                    phenotype_data.get('health_association'),
                    phenotype_data.get('host_association'),
                    phenotype_data.get('plant_pathogenicity'),
                    phenotype_data.get('spore_formation'),
                    phenotype_data.get('hemolysis'),
                    phenotype_data.get('cell_shape')
                ))
                
                migrated += 1
                if migrated % 100 == 0:
                    print(f"  Migrated {migrated} records...")
                    
            except Exception as e:
                print(f"  Error migrating {binomial_name}: {str(e)}")
                errors += 1
        
        # Commit all changes
        conn.commit()
        
        print(f"\nMigration completed:")
        print(f"  Total records: {total_count}")
        print(f"  Migrated: {migrated}")
        print(f"  Skipped (already existed): {skipped}")
        print(f"  Errors: {errors}")
        
        # Verify migration
        cursor.execute("SELECT COUNT(*) FROM results")
        results_count = cursor.fetchone()[0]
        print(f"\nTotal records in results table: {results_count}")
        
        if errors == 0 and (migrated + skipped) == total_count:
            print("\n✓ Migration successful! You can now safely remove the species_results table.")
            print("To remove the old table, run:")
            print("  sqlite3 microbellm_jobs.db 'DROP TABLE species_results;'")
        else:
            print("\n⚠ Migration completed with issues. Please review before removing species_results table.")
        
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def backup_database():
    """Create a backup of the database before migration"""
    import shutil
    backup_path = DB_PATH.with_suffix('.db.backup')
    
    print(f"Creating backup at {backup_path}...")
    shutil.copy2(DB_PATH, backup_path)
    print("Backup created successfully.")

if __name__ == "__main__":
    print("=== Database Migration: species_results → results ===")
    print("This script will migrate all data from species_results table to results table")
    print("to establish a single source of truth for the admin dashboard and components.\n")
    
    # Create backup first
    backup_database()
    
    # Perform migration
    migrate_species_results_to_results()