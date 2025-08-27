"""
Shared utilities for both public and admin web applications
"""

import os
import sqlite3
from pathlib import Path
import yaml
from microbellm import config

# Database path
# Use absolute path relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
# Use the database path from config (microbellm.db)
DATABASE_PATH = str(PROJECT_ROOT / config.DATABASE_PATH)

# Cache configuration
CACHE_DURATION = 300  # 5 minutes in seconds

def get_db_connection(db_path=None):
    """Get a database connection"""
    if db_path is None:
        db_path = DATABASE_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_page_manifest(page_name):
    """Load manifest file for a research page"""
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'templates', 'research', page_name, 'manifest.yaml'
    )
    
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def reset_running_jobs_on_startup(db_path=None):
    """Set status of 'running' jobs to 'interrupted' on application start."""
    if db_path is None:
        db_path = DATABASE_PATH
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find all jobs that were running
    cursor.execute("SELECT id FROM combinations WHERE status = 'running'")
    running_jobs = cursor.fetchall()
    
    if running_jobs:
        print(f"Found {len(running_jobs)} jobs with 'running' status on startup. Setting to 'interrupted'.")
        # Update their status to 'interrupted'
        cursor.execute("UPDATE combinations SET status = 'interrupted' WHERE status = 'running'")
        conn.commit()
        
    conn.close()

def init_database(db_path=None):
    """Initialize database tables"""
    if db_path is None:
        db_path = DATABASE_PATH
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS combinations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species_file TEXT,
            model TEXT,
            system_template TEXT,
            user_template TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            total_species INTEGER DEFAULT 0,
            successful_species INTEGER DEFAULT 0,
            failed_species INTEGER DEFAULT 0,
            timeout_species INTEGER DEFAULT 0,
            submitted_species INTEGER DEFAULT 0,
            received_species INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS species_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            combination_id INTEGER,
            binomial_name TEXT,
            result TEXT,
            status TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(combination_id) REFERENCES combinations(id)
        )
    ''')
    
    # Create results table for storing individual species results with full data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            binomial_name TEXT,
            result TEXT,
            status TEXT,
            error TEXT,
            species_file TEXT,
            model TEXT,
            system_template TEXT,
            user_template TEXT,
            knowledge_group TEXT,
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create managed_models table for storing added models
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS managed_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create managed_species_files table for storing added species files
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS managed_species_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species_file TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add missing columns to combinations table if they don't exist
    try:
        cursor.execute("SELECT total_species FROM combinations LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        cursor.execute("ALTER TABLE combinations ADD COLUMN total_species INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE combinations ADD COLUMN successful_species INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE combinations ADD COLUMN failed_species INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE combinations ADD COLUMN timeout_species INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE combinations ADD COLUMN submitted_species INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE combinations ADD COLUMN received_species INTEGER DEFAULT 0")
    
    conn.commit()
    conn.close()