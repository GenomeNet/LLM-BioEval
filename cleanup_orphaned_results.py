#!/usr/bin/env python3
"""
Clean up orphaned results in the results table that don't have corresponding combinations.
This can happen when combinations are deleted but results remain.
"""

import sqlite3
from pathlib import Path

# Database path
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "microbellm_jobs.db"

def cleanup_orphaned_results():
    """Remove results that don't have corresponding combinations"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # First, count orphaned results
        cursor.execute("""
            SELECT COUNT(*)
            FROM results r
            WHERE NOT EXISTS (
                SELECT 1 FROM combinations c 
                WHERE c.species_file = r.species_file 
                AND c.model = r.model 
                AND c.system_template = r.system_template 
                AND c.user_template = r.user_template
            )
        """)
        
        orphan_count = cursor.fetchone()[0]
        print(f"Found {orphan_count} orphaned results")
        
        if orphan_count == 0:
            print("No orphaned results to clean up.")
            return
        
        # Show what will be deleted
        cursor.execute("""
            SELECT DISTINCT species_file, model, system_template, user_template, COUNT(*) as count
            FROM results r
            WHERE NOT EXISTS (
                SELECT 1 FROM combinations c 
                WHERE c.species_file = r.species_file 
                AND c.model = r.model 
                AND c.system_template = r.system_template 
                AND c.user_template = r.user_template
            )
            GROUP BY species_file, model, system_template, user_template
            ORDER BY model, species_file
        """)
        
        print("\nOrphaned results to be deleted:")
        print("-" * 80)
        for row in cursor.fetchall():
            species_file, model, sys_tmpl, usr_tmpl, count = row
            print(f"{model} | {species_file} | {Path(sys_tmpl).name} | {count} results")
        
        # Ask for confirmation
        response = input(f"\nDelete {orphan_count} orphaned results? (yes/no): ")
        
        if response.lower() == 'yes':
            # Delete orphaned results
            cursor.execute("""
                DELETE FROM results
                WHERE NOT EXISTS (
                    SELECT 1 FROM combinations c 
                    WHERE c.species_file = results.species_file 
                    AND c.model = results.model 
                    AND c.system_template = results.system_template 
                    AND c.user_template = results.user_template
                )
            """)
            
            deleted = cursor.rowcount
            conn.commit()
            print(f"\n✓ Deleted {deleted} orphaned results")
        else:
            print("\nCancelled - no changes made")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def check_specific_model(model_name):
    """Check status of a specific model"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        print(f"\nChecking status for model: {model_name}")
        print("-" * 80)
        
        # Check in combinations
        cursor.execute("""
            SELECT species_file, system_template, user_template, status
            FROM combinations
            WHERE model = ?
        """, (model_name,))
        
        combos = cursor.fetchall()
        print(f"Found {len(combos)} combinations")
        
        # Check in results
        cursor.execute("""
            SELECT species_file, system_template, user_template, COUNT(*) as count
            FROM results
            WHERE model = ?
            GROUP BY species_file, system_template, user_template
        """, (model_name,))
        
        results = cursor.fetchall()
        print(f"Found {len(results)} result groups")
        
        # Check for orphans
        cursor.execute("""
            SELECT COUNT(*)
            FROM results r
            WHERE model = ?
            AND NOT EXISTS (
                SELECT 1 FROM combinations c 
                WHERE c.species_file = r.species_file 
                AND c.model = r.model 
                AND c.system_template = r.system_template 
                AND c.user_template = r.user_template
            )
        """, (model_name,))
        
        orphan_count = cursor.fetchone()[0]
        print(f"Orphaned results: {orphan_count}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        # Check specific model
        if len(sys.argv) > 2:
            check_specific_model(sys.argv[2])
        else:
            check_specific_model('x-ai/grok-4')
    else:
        # Run cleanup
        cleanup_orphaned_results()