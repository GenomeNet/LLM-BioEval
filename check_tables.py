#!/usr/bin/env python3
"""Check what tables exist in the database."""

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
        print(f"Found database at: {path}")
        db_path = path
        
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"Tables in {os.path.basename(path)}:")
        for table in tables:
            print(f"  - {table[0]}")
            
            # Get row count for each table
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"    Row count: {count}")
            except:
                print(f"    Could not count rows")
        
        conn.close()
        print()

if not db_path:
    print("No database found!")