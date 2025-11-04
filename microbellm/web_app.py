#!/usr/bin/env python

import os
import io
import json
import csv
import threading
import time
import math
import re
import secrets
import logging
from typing import Optional
from collections import defaultdict
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from markupsafe import escape
from flask_socketio import SocketIO, emit
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from microbellm.utils import read_template_from_file
from microbellm.predict import predict_binomial_name
import sqlite3
from pathlib import Path
from microbellm import config
from microbellm.utils import detect_template_type
from microbellm.utils import (
    create_ground_truth_tables, import_ground_truth_csv, get_ground_truth_datasets,
    get_ground_truth_data, calculate_model_accuracy, delete_ground_truth_dataset,
    normalize_value
)
from microbellm.research_config import (
    RESEARCH_PROJECTS, get_project_by_id, get_project_by_route, get_projects_for_page
)
import yaml
from microbellm.unified_db import UnifiedDB
from microbellm.validation import PredictionValidator

def _load_env_file(name: str) -> None:
    """Populate os.environ with variables from a simple KEY=VALUE file if present"""
    env_path = Path(__file__).resolve().parent.parent / name
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            key, sep, value = stripped.partition('=')
            if not sep:
                continue

            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            os.environ.setdefault(key, value)
    except OSError as exc:
        logging.getLogger(__name__).warning("Unable to load environment variables from %s: %s", env_path, exc)


if not os.getenv('MICROBELLM_SECRET_KEY'):
    _load_env_file('.env.local')


def _configure_logging() -> logging.Logger:
    """Configure application logging with an overridable log level."""
    level_name = os.getenv('MICROBELLM_LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    # Reduce noisy Flask/Werkzeug request logs unless explicitly enabled
    if level > logging.DEBUG:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    return logging.getLogger(__name__)


logger = _configure_logging()


def _resolve_secret_key() -> str:
    """Return Flask secret key from environment, falling back to an ephemeral value"""
    secret_key = os.getenv('MICROBELLM_SECRET_KEY')
    if secret_key:
        return secret_key

    fallback = os.getenv('FLASK_SECRET_KEY')
    if fallback:
        logger.warning("Using FLASK_SECRET_KEY fallback; set MICROBELLM_SECRET_KEY for consistency.")
        return fallback

    ephemeral = secrets.token_urlsafe(32)
    logger.warning("MICROBELLM_SECRET_KEY not set. Generated ephemeral secret key; sessions reset on restart.")
    return ephemeral


app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = _resolve_secret_key()
socketio = SocketIO(app, cors_allowed_origins="*")

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

# Global variables for job management
processing_manager = None
# Use the unified database
from microbellm.shared import DATABASE_PATH
db_path = DATABASE_PATH

# Cache for search count correlation data
_search_correlation_cache = {}
_search_correlation_cache_timestamp = 0

# Cache for knowledge analysis data
_knowledge_analysis_cache = {}
_knowledge_analysis_cache_timestamp = 0

# Cache for ground truth phenotype statistics (per dataset)
_ground_truth_stats_cache = {}
_ground_truth_stats_cache_lock = threading.Lock()

# Ground truth phenotype metadata used by statistics endpoints
GROUND_TRUTH_PHENOTYPE_DEFINITIONS = {
    'motility': {
        'label': 'Motility',
        'type': 'Binary',
        'targets': ['TRUE', 'FALSE']
    },
    'spore_formation': {
        'label': 'Spore formation',
        'type': 'Binary',
        'targets': ['TRUE', 'FALSE']
    },
    'gram_staining': {
        'label': 'Gram staining',
        'type': 'Multi-class',
        'targets': ['gram stain negative', 'gram stain positive', 'gram stain variable']
    },
    'cell_shape': {
        'label': 'Cell shape',
        'type': 'Multi-class',
        'targets': ['bacillus', 'coccus', 'spirillum', 'tail']
    },
    'health_association': {
        'label': 'Health Association',
        'type': 'Binary',
        'targets': ['TRUE', 'FALSE']
    },
    'host_association': {
        'label': 'Host Association',
        'type': 'Binary',
        'targets': ['TRUE', 'FALSE']
    },
    'plant_pathogenicity': {
        'label': 'Plant pathogenicity',
        'type': 'Binary',
        'targets': ['TRUE', 'FALSE']
    },
    'biosafety_level': {
        'label': 'Biosafety level',
        'type': 'Multi-class',
        'targets': ['biosafety level 1', 'biosafety level 2', 'biosafety level 3']
    },
    'extreme_environment_tolerance': {
        'label': 'Extreme environment tolerance',
        'type': 'Binary',
        'targets': ['TRUE', 'FALSE']
    },
    'animal_pathogenicity': {
        'label': 'Animal Pathogenicity',
        'type': 'Binary',
        'targets': ['TRUE', 'FALSE']
    },
    'hemolysis': {
        'label': 'Hemolysis',
        'type': 'Multi-class',
        'targets': ['alpha', 'beta', 'gamma']
    },
    'aerophilicity': {
        'label': 'Aerophilicity',
        'type': 'Multi-class',
        'targets': ['aerobic', 'aerotolerant', 'anaerobic', 'facultatively anaerobic']
    },
    'biofilm_formation': {
        'label': 'Biofilm formation',
        'type': 'Binary',
        'targets': ['TRUE', 'FALSE']
    }
}

# Cached model accuracy snapshots (per dataset)
_model_accuracy_cache = {}
_model_accuracy_cache_lock = threading.Lock()

# Dataset to species file mapping used for model accuracy calculations
DATASET_SPECIES_FILE_MAP = {
    'WA_Test_Dataset': 'wa_with_gcount.txt',
    'LA_Test_Dataset': 'la.txt'
}

# Cached knowledge accuracy snapshots (per dataset)
_knowledge_accuracy_cache = {}
_knowledge_accuracy_cache_lock = threading.Lock()

# Cached model performance by year snapshots (per dataset)
_model_performance_year_cache = {}
_model_performance_year_cache_lock = threading.Lock()

# Cached model metadata index derived from year_size.tsv
_model_metadata_index = None
_model_metadata_index_lock = threading.Lock()

# Track whether persistence tables are writable in this environment
_ground_truth_persistence_available: Optional[bool] = None

_MODEL_METADATA_ALIAS_MAP = {
    'anthropicclaude3opus': 'claude3opus',
    'anthropicclaude3sonnet': 'claude3sonnet',
    'anthropicclaude3haiku': 'claude3haiku',
    'openaigpt4': 'gpt4',
    'openaigpt4o': 'gpt4o',
    'openaigpt4omini': 'gpt4omini',
    'openaigpt45': 'gpt45',
    'openaigpt5': 'gpt5',
    'googlegemini15pro': 'gemini15pro',
    'googlegemini15flash': 'gemini15flash',
    'metallamallama370b': 'llama370b',
    'metallamallama38b': 'llama38b',
    'metallamallama3340b': 'llama3340b',
    'metallamallama312b': 'llama312b',
    'mistralaimistral7b': 'mistral7b',
    'mistralaimixtral8x7b': 'mixtral8x7b',
    'coherecommandrplus': 'commandrplus',
    'coherecommandr': 'commandr',
    'xaigrok15': 'grok15'
}

AVERAGE_SAMPLE_SIZE_THRESHOLD = 100

CACHE_DURATION = 300  # 5 minutes in seconds

def reset_running_jobs_on_startup():
    """Set status of 'running' jobs to 'interrupted' on application start."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find all jobs that were running
    cursor.execute("SELECT id FROM combinations WHERE status = 'running'")
    running_jobs = cursor.fetchall()
    
    if running_jobs:
        logger.info("Found %d jobs with 'running' status on startup. Setting to 'interrupted'.", len(running_jobs))
        try:
            cursor.execute("UPDATE combinations SET status = 'interrupted' WHERE status = 'running'")
            conn.commit()
        except sqlite3.OperationalError as exc:
            if 'readonly' in str(exc).lower():
                logger.warning("Database is read-only; skipping job reset on startup.")
            else:
                raise
        
    conn.close()

class ProcessingManager:
    def __init__(self):
        self.unified_db = UnifiedDB(db_path)
        self.running_combinations = {}
        self.job_queue = []
        self.paused_combinations = set()
        self.stopped_combinations = set()
        self.requests_per_second = 30.0  # Default rate limit
        self.last_request_time = {}
        self.max_concurrent_requests = 30  # Default concurrent requests
        self.executor = None  # Thread pool executor
        self.init_database()
    
    def set_max_concurrent_requests(self, max_concurrent):
        """Set the maximum number of concurrent API requests"""
        self.max_concurrent_requests = max(1, max_concurrent)
        # Restart executor with new thread count if it exists
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
    
    def _get_or_create_executor(self):
        """Get or create the thread pool executor"""
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        return self.executor
    
    def init_database(self):
        """Initialize SQLite database for tracking species x model combinations"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables for combination tracking
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
        
        # Create tables for managing dashboard view
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
        
        # Create table for template metadata
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
        
        # Add template_type column to template_metadata table if it doesn't exist
        cursor.execute("PRAGMA table_info(template_metadata)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'template_type' not in columns:
            cursor.execute("ALTER TABLE template_metadata ADD COLUMN template_type TEXT DEFAULT 'phenotype'")
            logger.info("Added column template_type to template_metadata table")

        # Migration: Add phenotype columns if they don't exist
        cursor.execute("PRAGMA table_info(results)")
        columns = [column[1] for column in cursor.fetchall()]
        
        phenotype_columns = [
            ('gram_staining', 'TEXT'),
            ('motility', 'TEXT'),
            ('aerophilicity', 'TEXT'),
            ('extreme_environment_tolerance', 'TEXT'),
            ('biofilm_formation', 'TEXT'),
            ('animal_pathogenicity', 'TEXT'),
            ('biosafety_level', 'TEXT'),
            ('health_association', 'TEXT'),
            ('host_association', 'TEXT'),
            ('plant_pathogenicity', 'TEXT'),
            ('spore_formation', 'TEXT'),
            ('hemolysis', 'TEXT'),
            ('cell_shape', 'TEXT'),
            ('knowledge_group', 'TEXT')
        ]
        
        for column_name, column_type in phenotype_columns:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE results ADD COLUMN {column_name} {column_type}')
                logger.info("Added column %s to results table", column_name)
        
        # Migration: Add new progress tracking columns if they don't exist
        cursor.execute("PRAGMA table_info(combinations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        progress_columns = [
            ('submitted_species', 'INTEGER DEFAULT 0'),
            ('received_species', 'INTEGER DEFAULT 0'),
            ('successful_species', 'INTEGER DEFAULT 0'),
            ('retried_species', 'INTEGER DEFAULT 0'),
            ('timeout_species', 'INTEGER DEFAULT 0')
        ]
        
        for column_name, column_type in progress_columns:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE combinations ADD COLUMN {column_name} {column_type}')
                logger.info("Added column %s to combinations table", column_name)
        
        conn.commit()
        conn.close()
    
    def create_combinations(self, species_file, models, template_pairs):
        """Create combinations for processing"""
        # Read species from file
        species_list = self._read_species_from_file(os.path.join(config.SPECIES_DIR, species_file))
        total_species = len(species_list)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        created_combinations = []
        
        # Create combinations for each model and template pair
        for model in models:
            for template_name, template_paths in template_pairs.items():
                system_template = template_paths['system']
                user_template = template_paths['user']
                
                # Check if combination already exists
                cursor.execute('''
                    SELECT id FROM combinations 
                    WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
                ''', (species_file, model, system_template, user_template))
                
                if not cursor.fetchone():
                    # Create new combination
                    cursor.execute('''
                        INSERT INTO combinations 
                        (species_file, model, system_template, user_template, total_species, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (species_file, model, system_template, user_template, total_species, datetime.now()))
                    
                    created_combinations.append({
                        'species_file': species_file,
                        'model': model,
                        'system_template': system_template,
                        'user_template': user_template,
                        'template_name': template_name
                    })
        
        conn.commit()
        conn.close()
        
        return created_combinations
    
    def _read_species_from_file(self, file_path):
        """Read species list from a file, filtering out headers"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # Import the header filtering function
                from microbellm.utils import filter_species_list
                species = filter_species_list(lines)
                return species
        except Exception as e:
            logger.error("Error reading species file %s: %s", file_path, e)
            return []
    
    def start_combination(self, combination_id):
        """Start processing a combination."""
        logger.debug("start_combination called for ID %s", combination_id)
        
        # Check if API key is configured before starting
        api_key = os.getenv('OPENROUTER_API_KEY')
        logger.debug("API key status: %s", 'SET' if api_key else 'NOT SET')
        
        if not api_key:
            logger.error("No API key found for combination %s", combination_id)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE combinations SET status = 'failed' WHERE id = ?", (combination_id,))
            conn.commit()
            conn.close()
            self._emit_log(combination_id, "Error: OPENROUTER_API_KEY is not configured. Please set your API key in Settings.")
            socketio.emit('job_update', {'combination_id': combination_id, 'status': 'failed', 'error': 'API key not configured'})
            return False
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check current status
        cursor.execute("SELECT status FROM combinations WHERE id = ?", (combination_id,))
        result = cursor.fetchone()
        if not result:
            logger.error("Combination %s not found in database", combination_id)
            conn.close()
            return False
            
        current_status = result[0]
        logger.debug("Current status for combination %s: %s", combination_id, current_status)
        
        if current_status not in ['pending', 'interrupted']:
            logger.warning("Cannot start combination %s - status is %s, not pending/interrupted", combination_id, current_status)
            conn.close()
            return False
        
        # Remove from paused/stopped sets if it was there
        self.paused_combinations.discard(combination_id)
        self.stopped_combinations.discard(combination_id)
        
        # Check if we can start immediately or need to queue
        can_start = self.can_start_new_job()
        logger.debug("Can start new job: %s", can_start)
        
        if not can_start:
            if combination_id not in self.job_queue:
                self.job_queue.append(combination_id)
                logger.info("Job %s queued - position: %d", combination_id, len(self.job_queue))
                self._emit_log(combination_id, f"Job queued - waiting for slot. Queue position: {len(self.job_queue)}")
                socketio.emit('job_update', {'combination_id': combination_id, 'status': 'queued'})
            return True

        # Get combination details
        cursor.execute('''
            SELECT species_file, model, system_template, user_template 
            FROM combinations WHERE id = ?
        ''', (combination_id,))

        result = cursor.fetchone()
        if not result:
            logger.error("Could not retrieve combination details for %s", combination_id)
            conn.close()
            return False
        
        species_file, model, system_template, user_template = result
        logger.debug("Starting combination %s: %s + %s", combination_id, species_file, model)
        logger.debug("Templates for combination %s: %s + %s", combination_id, system_template, user_template)
        
        # Update status to running
        cursor.execute('''
            UPDATE combinations SET status = ?, started_at = ? WHERE id = ?
        ''', ('running', datetime.now(), combination_id))
        conn.commit()
        conn.close()

        logger.info("Starting processing thread for combination %s", combination_id)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self._process_combination, 
                                 args=(combination_id, species_file, model, system_template, user_template))
        thread.daemon = True
        thread.start()
        
        self.running_combinations[combination_id] = {
            'species_file': species_file,
            'model': model,
            'system_template': system_template,
            'user_template': user_template,
            'status': 'running'
        }
        
        logger.info("Combination %s started successfully", combination_id)
        return True

    def pause_combination(self, combination_id):
        """Pause a running combination"""
        if combination_id in self.running_combinations:
            self.paused_combinations.add(combination_id)
            self._emit_log(combination_id, "Job paused by user")
            
            # Update database status
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE combinations SET status = 'interrupted' WHERE id = ?", (combination_id,))
            conn.commit()
            conn.close()
            
            socketio.emit('job_update', {'combination_id': combination_id, 'status': 'interrupted'})
            return True
        return False

    def stop_combination(self, combination_id):
        """Stop a running combination"""
        if combination_id in self.running_combinations:
            self.stopped_combinations.add(combination_id)
            self.paused_combinations.discard(combination_id)  # Remove from paused if it was there
            self._emit_log(combination_id, "Job stopped by user")
            
            # Update database status
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE combinations SET status = 'interrupted' WHERE id = ?", (combination_id,))
            conn.commit()
            conn.close()
            
            socketio.emit('job_update', {'combination_id': combination_id, 'status': 'interrupted'})
            return True
        return False

    def _wait_for_rate_limit(self, combination_id):
        """Wait if necessary to respect rate limiting"""
        if combination_id not in self.last_request_time:
            self.last_request_time[combination_id] = 0
        
        time_since_last = time.time() - self.last_request_time[combination_id]
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time[combination_id] = time.time()

    def _process_combination(self, combination_id, species_file, model, system_template, user_template):
        """The actual processing logic for a combination with parallel processing."""
        
        logger.debug("_process_combination started for %s", combination_id)
        self._emit_log(combination_id, f"Starting processing for {species_file} with model {model}")
        
        try:
            species_list = self._read_species_from_file(os.path.join(config.SPECIES_DIR, species_file))
            if not species_list:
                self._emit_log(combination_id, f"Error: Could not read or empty species file: {species_file}")
                self._mark_combination_failed(combination_id)
                return

            # Get already processed species for smart resume
            already_processed = self.get_already_processed_species(species_file, model, system_template, user_template)
            remaining_species = [species for species in species_list if species not in already_processed]
            
            if not remaining_species:
                self._emit_log(combination_id, f"All species already processed! ({len(already_processed)}/{len(species_list)})")
                self._mark_combination_completed(combination_id)
                return
            
            self._emit_log(combination_id, f"Smart resume: {len(already_processed)} already done, processing {len(remaining_species)} remaining species...")
            self._emit_log(combination_id, f"Using {self.max_concurrent_requests} parallel workers...")

            system_template_content = read_template_from_file(system_template)
            user_template_content = read_template_from_file(user_template)

            # Create executor for parallel processing
            executor = self._get_or_create_executor()
            futures = {}
            
            # Submit initial batch of species
            for i, species_name in enumerate(remaining_species[:self.max_concurrent_requests]):
                if combination_id in self.paused_combinations or combination_id in self.stopped_combinations:
                    break
                    
                future = executor.submit(
                    self._process_single_species,
                    combination_id, species_file, species_name, 
                    model, system_template, user_template,
                    system_template_content, user_template_content,
                    len(already_processed) + i + 1, len(species_list)
                )
                futures[future] = species_name
            
            # Process remaining species as workers complete
            next_species_idx = len(futures)
            
            while futures and combination_id not in self.stopped_combinations:
                # Check for paused state
                if combination_id in self.paused_combinations:
                    self._emit_log(combination_id, "Job paused.")
                    # Cancel pending futures
                    for future in futures:
                        future.cancel()
                    break
                
                # Wait for any future to complete with timeout
                import concurrent.futures
                done, pending = concurrent.futures.wait(futures.keys(), timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for future in done:
                    species_name = futures.pop(future)
                    try:
                        # Get result (this will raise exception if processing failed)
                        future.result()
                    except Exception as e:
                        self._emit_log(combination_id, f"Error processing {species_name}: {str(e)}")
                    
                    # Submit next species if available
                    if next_species_idx < len(remaining_species) and combination_id not in self.paused_combinations:
                        next_species_name = remaining_species[next_species_idx]
                        new_future = executor.submit(
                            self._process_single_species,
                            combination_id, species_file, next_species_name,
                            model, system_template, user_template,
                            system_template_content, user_template_content,
                            len(already_processed) + next_species_idx + 1, len(species_list)
                        )
                        futures[new_future] = next_species_name
                        next_species_idx += 1
                
                # Emit progress update
                self._emit_progress_update(combination_id)

            # Wait for remaining futures to complete
            for future in futures:
                try:
                    future.result()
                except:
                    pass

            # Final status update
            self._finalize_combination(combination_id)
            
        except Exception as e:
            self._emit_log(combination_id, f"Critical error in processing: {str(e)}")
            self._mark_combination_failed(combination_id)
        finally:
            # Clean up and start next job in queue
            self._cleanup_completed_job(combination_id)

    def _process_single_species(self, combination_id, species_file, species_name, 
                               model, system_template, user_template,
                               system_template_content, user_template_content,
                               current_idx, total_species):
        """Process a single species (called by thread pool)"""
        try:
            self._emit_log(combination_id, f"Processing species {current_idx}/{total_species}: {species_name}")
            
            # Apply rate limiting (shared across threads)
            self._wait_for_rate_limit(combination_id)
            
            # Track submission
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE combinations SET submitted_species = submitted_species + 1 WHERE id = ?", (combination_id,))
            conn.commit()
            conn.close()
            
            # Make prediction
            result = predict_binomial_name(
                species_name,
                system_template_content,
                user_template_content,
                model,
                temperature=0.0,
                verbose=False,
                user_template_path=user_template
            )
            
            # Process result
            self._process_species_result(combination_id, species_file, species_name, model, 
                                       system_template, user_template, result)
                                       
        except TimeoutError as e:
            self._process_species_timeout(combination_id, species_file, species_name, model, 
                                        system_template, user_template, str(e))
        except Exception as e:
            self._process_species_error(combination_id, species_file, species_name, model, 
                                      system_template, user_template, str(e))

    def _mark_combination_failed(self, combination_id):
        """Mark combination as failed"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE combinations SET status = 'failed' WHERE id = ?", (combination_id,))
        conn.commit()
        conn.close()
        socketio.emit('job_update', {'combination_id': combination_id, 'status': 'failed'})

    def _mark_combination_completed(self, combination_id):
        """Mark combination as completed"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE combinations SET status = 'completed', completed_at = ? WHERE id = ?", (datetime.now(), combination_id))
        conn.commit()
        conn.close()
        socketio.emit('job_update', {'combination_id': combination_id, 'status': 'completed'})

    def _cleanup_completed_job(self, combination_id):
        """Clean up after job completion and start next queued job"""
        # Remove from running combinations
        if combination_id in self.running_combinations:
            del self.running_combinations[combination_id]
        
        # Remove from tracking sets
        self.paused_combinations.discard(combination_id)
        self.stopped_combinations.discard(combination_id)
        
        # Start next job in queue if any
        if self.job_queue and self.can_start_new_job():
            next_job_id = self.job_queue.pop(0)
            self._emit_log(next_job_id, "Starting from queue...")
            threading.Thread(target=lambda: self.start_combination(next_job_id), daemon=True).start()
    
    def can_start_new_job(self):
        """Check if we can start a new job - for now, allow multiple jobs"""
        return True  # Allow multiple jobs to run concurrently
    
    def shutdown(self):
        """Shutdown the executor cleanly"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

    def restart_combination(self, combination_id):
        """Restart a failed or interrupted combination."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check current status
        cursor.execute("SELECT status FROM combinations WHERE id = ?", (combination_id,))
        result = cursor.fetchone()
        if not result or result[0] not in ['failed', 'interrupted', 'completed']:
            conn.close()
            return False, "Job is not in a restartable state."

        # Get combination details to clear old results if requested
        cursor.execute("SELECT species_file, model, system_template, user_template FROM combinations WHERE id = ?", (combination_id,))
        combo = cursor.fetchone()

        # Reset the combination status and some progress counters (keep completed work)
        cursor.execute('''
            UPDATE combinations 
            SET status = 'pending', started_at = NULL, completed_at = NULL 
            WHERE id = ?
        ''', (combination_id,))
        
        conn.commit()
        conn.close()

        # Start the combination
        self.start_combination(combination_id)
        return True, "Combination restarted successfully."

    def _emit_log(self, combination_id, message):
        """Emit a log message to the client and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # Server-side log for troubleshooting (debug by default)
        logger.debug("[COMBO-%s] %s", combination_id, log_message)
        
        # Emit to web client
        socketio.emit('log_message', {
            'combination_id': combination_id,
            'log': log_message
        })

    def _process_species_result(self, combination_id, species_file, species_name, model, system_template, user_template, result):
        """Process the result from a species prediction"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if result:
            # We received a response from the LLM
            cursor.execute("UPDATE combinations SET received_species = received_species + 1 WHERE id = ?", (combination_id,))
            
            # Detect if this is a knowledge level prediction
            is_knowledge_prediction = detect_template_type(user_template) == 'knowledge'
            
            # Check if the result contains valid data based on template type
            if is_knowledge_prediction:
                # For knowledge predictions, check for knowledge_group
                has_valid_data = result.get('knowledge_group') is not None
            else:
                # For phenotype predictions, check for phenotype data
                has_valid_data = any([
                    result.get('gram_staining'),
                    result.get('motility'),
                    result.get('aerophilicity'),
                    result.get('extreme_environment_tolerance'),
                    result.get('biofilm_formation'),
                    result.get('animal_pathogenicity'),
                    result.get('biosafety_level'),
                    result.get('health_association'),
                    result.get('host_association'),
                    result.get('plant_pathogenicity'),
                    result.get('spore_formation'),
                    result.get('hemolysis'),
                    result.get('cell_shape')
                ])

            if has_valid_data:
                # Successfully parsed result with valid data
                if is_knowledge_prediction:
                    knowledge_level = result.get('knowledge_group', 'unknown')
                    self._emit_log(combination_id, f"Success: Knowledge level for {species_name}: {knowledge_level}")
                else:
                    self._emit_log(combination_id, f"Success: Successfully processed {species_name} with phenotype data")
                cursor.execute("UPDATE combinations SET successful_species = successful_species + 1, completed_species = completed_species + 1 WHERE id = ?", (combination_id,))
                result_status = 'completed'
            else:
                # Got a response but no valid data extracted
                data_type = "knowledge level" if is_knowledge_prediction else "phenotype data"
                self._emit_log(combination_id, f"Warning: Received response for {species_name} but no valid {data_type} extracted")
                cursor.execute("UPDATE combinations SET failed_species = failed_species + 1, completed_species = completed_species + 1 WHERE id = ?", (combination_id,))
                result_status = 'failed'
            
            # Prepare data for insertion
            result_data = {
                'species_file': species_file,
                'binomial_name': species_name,
                'model': model,
                'system_template': system_template,
                'user_template': user_template,
                'status': result_status,
                'result': result.get('raw_response', ''),
                'error': '',
                'timestamp': datetime.now()
            }
            
            # Add data based on template type
            if is_knowledge_prediction:
                # For knowledge predictions, store normalized knowledge_group
                from microbellm.utils import normalize_knowledge_level
                raw_knowledge = result.get('knowledge_group')
                result_data['knowledge_group'] = normalize_knowledge_level(raw_knowledge)
                # Set all phenotype fields to NULL
                for field in ['gram_staining', 'motility', 'aerophilicity', 'extreme_environment_tolerance',
                             'biofilm_formation', 'animal_pathogenicity', 'biosafety_level', 'health_association',
                             'host_association', 'plant_pathogenicity', 'spore_formation', 'hemolysis', 'cell_shape']:
                    result_data[field] = None
            else:
                # For phenotype predictions, store phenotype data
                result_data['knowledge_group'] = None
                
                # Process each field - if it contains INVALID: prefix, save it as is for tracking
                # The validation has already been done in parse_response
                result_data['gram_staining'] = result.get('gram_staining')
                result_data['motility'] = result.get('motility')
                
                # Handle aerophilicity as array - convert to string for database storage
                aerophilicity = result.get('aerophilicity')
                if isinstance(aerophilicity, list):
                    result_data['aerophilicity'] = str(aerophilicity)
                else:
                    result_data['aerophilicity'] = aerophilicity
                    
                result_data['extreme_environment_tolerance'] = result.get('extreme_environment_tolerance')
                result_data['biofilm_formation'] = result.get('biofilm_formation')
                result_data['animal_pathogenicity'] = result.get('animal_pathogenicity')
                result_data['biosafety_level'] = result.get('biosafety_level')
                result_data['health_association'] = result.get('health_association')
                result_data['host_association'] = result.get('host_association')
                result_data['plant_pathogenicity'] = result.get('plant_pathogenicity')
                result_data['spore_formation'] = result.get('spore_formation')
                result_data['hemolysis'] = result.get('hemolysis')
                result_data['cell_shape'] = result.get('cell_shape')
                
                # Log if there were invalid fields
                if 'invalid_fields' in result:
                    invalid_info = ", ".join([f"{f['field']}={f['value']}" for f in result['invalid_fields']])
                    self._emit_log(combination_id, f"Validation warning for {species_name}: {invalid_info}")
            
            cursor.execute('''
                INSERT INTO results (species_file, binomial_name, model, system_template, user_template, status, result, error, timestamp, gram_staining, motility, aerophilicity, extreme_environment_tolerance, biofilm_formation, animal_pathogenicity, biosafety_level, health_association, host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape, knowledge_group)
                VALUES (:species_file, :binomial_name, :model, :system_template, :user_template, :status, :result, :error, :timestamp, :gram_staining, :motility, :aerophilicity, :extreme_environment_tolerance, :biofilm_formation, :animal_pathogenicity, :biosafety_level, :health_association, :host_association, :plant_pathogenicity, :spore_formation, :hemolysis, :cell_shape, :knowledge_group)
            ''', result_data)
            
            # Invalidate all caches when new results are added
            _invalidate_all_caches()
            
            # CRITICAL FIX: Commit the transaction for successful results
            conn.commit()
            conn.close()

        else:
            # No response from API at all
            self._emit_log(combination_id, f"Error: Failed to get any response for {species_name}")
            cursor.execute('''
                INSERT INTO results (species_file, binomial_name, model, system_template, user_template, status, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (species_file, species_name, model, system_template, user_template, 'failed', 'No response from API', datetime.now()))
            cursor.execute("UPDATE combinations SET failed_species = failed_species + 1, completed_species = completed_species + 1 WHERE id = ?", (combination_id,))
            conn.commit()
            conn.close()
        
        # Emit progress update after processing each species
        self._emit_progress_update(combination_id)

    def _process_species_error(self, combination_id, species_file, species_name, model, system_template, user_template, error_message):
        """Process an error during species prediction"""
        self._emit_log(combination_id, f"Error: Exception processing {species_name}: {error_message}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results (species_file, binomial_name, model, system_template, user_template, status, error, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (species_file, species_name, model, system_template, user_template, 'failed', error_message, datetime.now()))
        cursor.execute("UPDATE combinations SET failed_species = failed_species + 1, completed_species = completed_species + 1 WHERE id = ?", (combination_id,))
        conn.commit()
        conn.close()
        
        # Emit progress update after processing each species
        self._emit_progress_update(combination_id)

    def _process_species_timeout(self, combination_id, species_file, species_name, model, system_template, user_template, error_message):
        """Process a timeout during species prediction"""
        self._emit_log(combination_id, f"⏱️ Timeout processing {species_name}: {error_message}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results (species_file, binomial_name, model, system_template, user_template, status, error, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (species_file, species_name, model, system_template, user_template, 'timeout', error_message, datetime.now()))
        cursor.execute("UPDATE combinations SET timeout_species = timeout_species + 1, completed_species = completed_species + 1 WHERE id = ?", (combination_id,))
        conn.commit()
        conn.close()
        
        # Emit progress update after processing each species
        self._emit_progress_update(combination_id)

    def _emit_progress_update(self, combination_id):
        """Emit detailed progress update"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT total_species, submitted_species, received_species, successful_species, failed_species, completed_species, timeout_species, status
            FROM combinations WHERE id = ?
        """, (combination_id,))
        progress = cursor.fetchone()
        conn.close()

        if progress:
            socketio.emit('job_update', {
                'combination_id': combination_id, 
                'status': progress[7],
                'total': progress[0],
                'submitted': progress[1],
                'received': progress[2],
                'successful': progress[3],
                'failed': progress[4],
                'completed': progress[5],
                'timeouts': progress[6]
            })

    def _finalize_combination(self, combination_id):
        """Finalize combination status based on results"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get full progress data for final update
        cursor.execute("""
            SELECT total_species, submitted_species, received_species, successful_species, failed_species, completed_species, timeout_species 
            FROM combinations WHERE id = ?
        """, (combination_id,))
        progress = cursor.fetchone()
        
        if not progress:
            conn.close()
            return
        
        successful_species, total_species = progress[3], progress[0]
        logger.debug("_finalize_combination for %s: successful=%s, total=%s", combination_id, successful_species, total_species)
        logger.debug("Full progress snapshot for %s: %s", combination_id, progress)
        
        if combination_id in self.paused_combinations:
            final_status = 'interrupted'
            self._emit_log(combination_id, f"Job paused - processed {successful_species} out of {total_species} species successfully")
        elif combination_id in self.stopped_combinations:
            final_status = 'interrupted'
            self._emit_log(combination_id, f"Job stopped - processed {successful_species} out of {total_species} species successfully")
        elif successful_species > 0:
            final_status = 'completed'
            self._emit_log(combination_id, f"Job completed - processed {successful_species} out of {total_species} species successfully")
        else:
            final_status = 'failed'
            self._emit_log(combination_id, f"Job failed - no species were processed successfully out of {total_species} total")
        
        cursor.execute("UPDATE combinations SET status = ?, completed_at = ? WHERE id = ?", (final_status, datetime.now(), combination_id))
        conn.commit()
        conn.close()
        
        # Send final update with complete progress data
        socketio.emit('job_update', {
            'combination_id': combination_id, 
            'status': final_status,
            'total': progress[0],
            'submitted': progress[1],
            'received': progress[2],
            'successful': progress[3],
            'failed': progress[4],
            'completed': progress[5],
            'timeouts': progress[6]
        })

    def get_dashboard_data(self):
        """Get data for the dashboard matrix view"""
        # Use unified database instead of old combinations table
        return self.unified_db.get_dashboard_data()

    def export_results_to_csv(self, species_file=None, model=None):
        """Export results to CSV format matching the example predictions.csv"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query with optional filters
        query = '''
            SELECT binomial_name, species_file, gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                   biofilm_formation, animal_pathogenicity, biosafety_level, health_association,
                   host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape,
                   model, system_template, timestamp
            FROM results 
            WHERE status = "completed"
        '''
        params = []
        
        if species_file:
            query += ' AND species_file = ?'
            params.append(species_file)
        
        if model:
            query += ' AND model = ?'
            params.append(model)
        
        query += ' ORDER BY species_file, binomial_name, model, timestamp'
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        # Generate CSV content with proper header
        if not results:
            return "binomial_name;species_file;unknown;gram_staining;motility;aerophilicity;extreme_environment_tolerance;biofilm_formation;animal_pathogenicity;biosafety_level;health_association;host_association;plant_pathogenicity;spore_formation;hemolysis;cell_shape;model;system_template;inference_timestamp"
        
        # Import cleaning functions
        from microbellm.utils import clean_csv_field
        
        csv_lines = []
        # Add header
        header = "binomial_name;species_file;unknown;gram_staining;motility;aerophilicity;extreme_environment_tolerance;biofilm_formation;animal_pathogenicity;biosafety_level;health_association;host_association;plant_pathogenicity;spore_formation;hemolysis;cell_shape;model;system_template;inference_timestamp"
        csv_lines.append(header)
        
        for row in results:
            (binomial_name, species_file_path, gram_staining, motility, aerophilicity, extreme_environment_tolerance,
             biofilm_formation, animal_pathogenicity, biosafety_level, health_association,
             host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape,
             model, system_template, timestamp) = row
            
            # Extract just the filename from the full path for cleaner CSV
            species_filename = species_file_path.split('/')[-1] if species_file_path else ''
            
            # Format timestamp for better readability
            formatted_timestamp = timestamp
            if timestamp:
                try:
                    # Parse and reformat timestamp
                    from datetime import datetime
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                    formatted_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    # Keep original if parsing fails
                    formatted_timestamp = str(timestamp)
            
            # Clean all fields for CSV export
            cleaned_fields = [
                clean_csv_field(binomial_name),
                clean_csv_field(species_filename),
                "0",  # unknown field placeholder
                clean_csv_field(gram_staining),
                clean_csv_field(motility),
                clean_csv_field(aerophilicity),
                clean_csv_field(extreme_environment_tolerance),
                clean_csv_field(biofilm_formation),
                clean_csv_field(animal_pathogenicity),
                clean_csv_field(biosafety_level),
                clean_csv_field(health_association),
                clean_csv_field(host_association),
                clean_csv_field(plant_pathogenicity),
                clean_csv_field(spore_formation),
                clean_csv_field(hemolysis),
                clean_csv_field(cell_shape),
                clean_csv_field(model),
                clean_csv_field(system_template),
                clean_csv_field(formatted_timestamp)
            ]
            
            # Join fields with semicolon delimiter
            csv_line = ";".join(cleaned_fields)
            csv_lines.append(csv_line)
        
        return '\n'.join(csv_lines)

    def import_results_from_csv(self, csv_content):
        """Import results from CSV content"""
        import csv
        from io import StringIO
        
        # Initialize counters
        results = {
            'total_entries': 0,
            'imported': 0,
            'skipped': 0,
            'overwritten_agreeing': 0,
            'overwritten_conflicting': 0,
            'errors': [],
            'conflicts': []
        }
        
        try:
            # Parse CSV with semicolon delimiter
            csv_file = StringIO(csv_content)
            reader = csv.DictReader(csv_file, delimiter=';')
            
            # Validate header
            expected_fields = [
                'binomial_name', 'species_file', 'unknown', 'gram_staining', 'motility', 
                'aerophilicity', 'extreme_environment_tolerance', 'biofilm_formation', 
                'animal_pathogenicity', 'biosafety_level', 'health_association', 
                'host_association', 'plant_pathogenicity', 'spore_formation', 
                'hemolysis', 'cell_shape', 'model', 'system_template', 'inference_timestamp'
            ]
            
            if not reader.fieldnames:
                results['errors'].append("CSV file appears to be empty")
                return results
            
            missing_fields = set(expected_fields) - set(reader.fieldnames)
            if missing_fields:
                results['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
                return results
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                results['total_entries'] += 1
                
                try:
                    # Extract data from row
                    binomial_name = row['binomial_name'].strip()
                    species_file = row['species_file'].strip()
                    model = row['model'].strip()
                    system_template = row['system_template'].strip()
                    timestamp_str = row.get('inference_timestamp', '').strip()
                    
                    # Skip empty rows
                    if not binomial_name or not model:
                        continue
                    
                    # Parse timestamp
                    timestamp = None
                    if timestamp_str:
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        except:
                            # Try other formats
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            except:
                                timestamp = datetime.now()
                    else:
                        timestamp = datetime.now()
                    
                    # Extract phenotype data
                    phenotype_data = {
                        'gram_staining': row.get('gram_staining', '').strip() or None,
                        'motility': row.get('motility', '').strip() or None,
                        'aerophilicity': row.get('aerophilicity', '').strip() or None,
                        'extreme_environment_tolerance': row.get('extreme_environment_tolerance', '').strip() or None,
                        'biofilm_formation': row.get('biofilm_formation', '').strip() or None,
                        'animal_pathogenicity': row.get('animal_pathogenicity', '').strip() or None,
                        'biosafety_level': row.get('biosafety_level', '').strip() or None,
                        'health_association': row.get('health_association', '').strip() or None,
                        'host_association': row.get('host_association', '').strip() or None,
                        'plant_pathogenicity': row.get('plant_pathogenicity', '').strip() or None,
                        'spore_formation': row.get('spore_formation', '').strip() or None,
                        'hemolysis': row.get('hemolysis', '').strip() or None,
                        'cell_shape': row.get('cell_shape', '').strip() or None,
                        'knowledge_group': row.get('knowledge_group', '').strip() or None
                    }
                    
                    # Derive user_template from system_template (assuming same filename pattern)
                    user_template = system_template.replace('/system/', '/user/')
                    
                    # Check if entry already exists
                    cursor.execute('''
                        SELECT id, gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                               biofilm_formation, animal_pathogenicity, biosafety_level, health_association,
                               host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape,
                               knowledge_group
                        FROM results 
                        WHERE binomial_name = ? AND species_file = ? AND model = ? 
                        AND system_template = ? AND user_template = ?
                    ''', (binomial_name, species_file, model, system_template, user_template))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Compare phenotype predictions
                        existing_data = {
                            'gram_staining': existing[1],
                            'motility': existing[2],
                            'aerophilicity': existing[3],
                            'extreme_environment_tolerance': existing[4],
                            'biofilm_formation': existing[5],
                            'animal_pathogenicity': existing[6],
                            'biosafety_level': existing[7],
                            'health_association': existing[8],
                            'host_association': existing[9],
                            'plant_pathogenicity': existing[10],
                            'spore_formation': existing[11],
                            'hemolysis': existing[12],
                            'cell_shape': existing[13],
                            'knowledge_group': existing[14]
                        }
                        
                        # Check if predictions agree
                        conflicts = []
                        for field, new_value in phenotype_data.items():
                            old_value = existing_data.get(field)
                            if old_value != new_value and (old_value or new_value):  # Ignore if both are None/empty
                                conflicts.append({
                                    'field': field,
                                    'old_value': old_value or 'None',
                                    'new_value': new_value or 'None'
                                })
                        
                        if conflicts:
                            results['overwritten_conflicting'] += 1
                            results['conflicts'].append({
                                'species': binomial_name,
                                'model': model,
                                'conflicts': conflicts
                            })
                        else:
                            results['overwritten_agreeing'] += 1
                        
                        # Update existing entry
                        cursor.execute('''
                            UPDATE results 
                            SET gram_staining = ?, motility = ?, aerophilicity = ?, 
                                extreme_environment_tolerance = ?, biofilm_formation = ?, 
                                animal_pathogenicity = ?, biosafety_level = ?, health_association = ?,
                                host_association = ?, plant_pathogenicity = ?, spore_formation = ?,
                                hemolysis = ?, cell_shape = ?, knowledge_group = ?, timestamp = ?, status = "completed"
                            WHERE id = ?
                        ''', (
                            phenotype_data['gram_staining'],
                            phenotype_data['motility'],
                            phenotype_data['aerophilicity'],
                            phenotype_data['extreme_environment_tolerance'],
                            phenotype_data['biofilm_formation'],
                            phenotype_data['animal_pathogenicity'],
                            phenotype_data['biosafety_level'],
                            phenotype_data['health_association'],
                            phenotype_data['host_association'],
                            phenotype_data['plant_pathogenicity'],
                            phenotype_data['spore_formation'],
                            phenotype_data['hemolysis'],
                            phenotype_data['cell_shape'],
                            phenotype_data['knowledge_group'],
                            timestamp,
                            existing[0]
                        ))
                        
                    else:
                        # Insert new entry
                        cursor.execute('''
                            INSERT INTO results (species_file, binomial_name, model, system_template, 
                                               user_template, status, result, error, timestamp,
                                               gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                                               biofilm_formation, animal_pathogenicity, biosafety_level, 
                                               health_association, host_association, plant_pathogenicity, 
                                               spore_formation, hemolysis, cell_shape, knowledge_group)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            species_file, binomial_name, model, system_template, user_template,
                            'completed', '', '', timestamp,
                            phenotype_data['gram_staining'],
                            phenotype_data['motility'],
                            phenotype_data['aerophilicity'],
                            phenotype_data['extreme_environment_tolerance'],
                            phenotype_data['biofilm_formation'],
                            phenotype_data['animal_pathogenicity'],
                            phenotype_data['biosafety_level'],
                            phenotype_data['health_association'],
                            phenotype_data['host_association'],
                            phenotype_data['plant_pathogenicity'],
                            phenotype_data['spore_formation'],
                            phenotype_data['hemolysis'],
                            phenotype_data['cell_shape'],
                            phenotype_data['knowledge_group']
                        ))
                        results['imported'] += 1
                    
                    # Also ensure the combination exists
                    cursor.execute('''
                        INSERT OR IGNORE INTO combinations 
                        (species_file, model, system_template, user_template, status, created_at)
                        VALUES (?, ?, ?, ?, 'imported', ?)
                    ''', (species_file, model, system_template, user_template, datetime.now()))
                    
                except Exception as e:
                    results['errors'].append(f"Row {row_num}: {str(e)}")
                    results['skipped'] += 1
            
            conn.commit()
            
            # Update combination statistics after all imports
            self._update_combination_statistics_after_import(cursor)
            
            # Add any new species files and models to managed lists
            cursor.execute('SELECT DISTINCT species_file FROM results')
            for (species_file,) in cursor.fetchall():
                cursor.execute('INSERT OR IGNORE INTO managed_species_files (species_file) VALUES (?)', (species_file,))
            
            cursor.execute('SELECT DISTINCT model FROM results')
            for (model,) in cursor.fetchall():
                cursor.execute('INSERT OR IGNORE INTO managed_models (model) VALUES (?)', (model,))
            
            conn.commit()
            conn.close()
            
            # Invalidate all caches when results are imported
            _invalidate_all_caches()
            
        except Exception as e:
            results['errors'].append(f"CSV parsing error: {str(e)}")
        
        return results

    def _update_combination_statistics_after_import(self, cursor):
        """Update combination statistics based on imported results"""
        # Get all combinations that need updating
        cursor.execute('''
            SELECT DISTINCT c.id, c.species_file, c.model, c.system_template, c.user_template
            FROM combinations c
            WHERE c.status = 'imported' OR c.completed_species IS NULL OR c.completed_species = 0
        ''')
        
        combinations_to_update = cursor.fetchall()
        
        for combo in combinations_to_update:
            combo_id, species_file, model, system_template, user_template = combo
            
            # Count successful results for this combination
            cursor.execute('''
                SELECT COUNT(DISTINCT binomial_name) 
                FROM results 
                WHERE species_file = ? AND model = ? AND system_template = ? 
                AND user_template = ? AND status = 'completed'
            ''', (species_file, model, system_template, user_template))
            
            successful_count = cursor.fetchone()[0]
            
            # Count failed results
            cursor.execute('''
                SELECT COUNT(DISTINCT binomial_name) 
                FROM results 
                WHERE species_file = ? AND model = ? AND system_template = ? 
                AND user_template = ? AND status = 'failed'
            ''', (species_file, model, system_template, user_template))
            
            failed_count = cursor.fetchone()[0]
            
            # Get total species count from file if possible
            total_species = 0
            try:
                species_list = self._read_species_from_file(os.path.join(config.SPECIES_DIR, species_file))
                total_species = len(species_list)
            except:
                # If file doesn't exist, use the count of distinct species we have
                cursor.execute('''
                    SELECT COUNT(DISTINCT binomial_name) 
                    FROM results 
                    WHERE species_file = ? AND model = ? AND system_template = ? 
                    AND user_template = ?
                ''', (species_file, model, system_template, user_template))
                total_species = cursor.fetchone()[0]
            
            # Update combination with proper counts
            cursor.execute('''
                UPDATE combinations 
                SET total_species = ?, 
                    completed_species = ?, 
                    successful_species = ?,
                    failed_species = ?,
                    status = CASE 
                        WHEN ? = ? THEN 'completed' 
                        WHEN ? > 0 THEN 'interrupted'
                        ELSE status 
                    END
                WHERE id = ?
            ''', (
                total_species, 
                successful_count + failed_count,  # completed = successful + failed
                successful_count,
                failed_count,
                successful_count + failed_count, total_species,  # for CASE: if all done, mark completed
                successful_count + failed_count,  # for CASE: if some done, mark interrupted
                combo_id
            ))

    def set_rate_limit(self, requests_per_second):
        """Set the rate limit for API requests"""
        self.requests_per_second = max(0.1, min(100.0, requests_per_second))  # Clamp between 0.1 and 100 RPS
        logger.info("Rate limit set to %.2f requests per second", self.requests_per_second)
    
    def get_already_processed_species(self, species_file, model, system_template, user_template):
        """Get list of species already successfully processed for this combination"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT binomial_name FROM results 
            WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ? 
            AND status = "completed"
        ''', (species_file, model, system_template, user_template))
        
        already_processed = set([row[0] for row in cursor.fetchall()])
        conn.close()
        return already_processed

    def _rerun_single_species(self, combination_id, species_file, species_name, model, system_template, user_template, system_template_content, user_template_content):
        """Re-run a single failed species"""
        try:
            self._emit_log(combination_id, f"Re-running species: {species_name}")
            
            # Make prediction
            result = predict_binomial_name(
                species_name,
                system_template_content,
                user_template_content,
                model,
                temperature=0.0,
                verbose=False,
                user_template_path=user_template
            )
            
            # Process result
            self._process_species_result(combination_id, species_file, species_name, model, 
                                       system_template, user_template, result)
            
            # Update combination statistics - don't increment counters since species was already counted
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get current counts to recalculate properly
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_results,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_count,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                    SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) as timeout_count
                FROM results 
                WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
            """, (species_file, model, system_template, user_template))
            
            counts = cursor.fetchone()
            if counts:
                total_results, successful_count, failed_count, timeout_count = counts
                completed_count = successful_count + failed_count + timeout_count
                
                # Determine the combination status based on current results
                cursor.execute("SELECT total_species FROM combinations WHERE id = ?", (combination_id,))
                total_species_result = cursor.fetchone()
                total_species = total_species_result[0] if total_species_result else 0
                
                if completed_count >= total_species:
                    new_status = 'completed'
                elif completed_count > 0:
                    new_status = 'completed'  # Partially completed is still considered completed
                else:
                    new_status = 'pending'
                
                cursor.execute("""
                    UPDATE combinations 
                    SET successful_species = ?, failed_species = ?, timeout_species = ?, 
                        completed_species = ?, received_species = ?, status = ?
                    WHERE id = ?
                """, (successful_count, failed_count, timeout_count, completed_count, total_results, new_status, combination_id))
            
            conn.commit()
            conn.close()
            
            # Emit progress update to refresh the dashboard
            self._emit_progress_update(combination_id)
            
            # Log the updated statistics for debugging
            self._emit_log(combination_id, f"Updated statistics - Success: {successful_count}, Failed: {failed_count}, Timeouts: {timeout_count}, Status: {new_status}")
            
            return True
        
        except Exception as e:
            self._emit_log(combination_id, f"Error re-running species: {str(e)}")
            return False

    def _rerun_failed_species(self, combination_id, failed_species):
        """Re-run all failed species for a combination"""
        try:
            self._emit_log(combination_id, f"Re-running {len(failed_species)} failed species")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for species_name in failed_species:
                # Get combination details
                cursor.execute("""
                    SELECT species_file, model, system_template, user_template
                    FROM combinations WHERE id = ?
                """, (combination_id,))
                
                combo_info = cursor.fetchone()
                if not combo_info:
                    continue
                
                species_file, model, system_template, user_template = combo_info
                
                # Get the failed result
                cursor.execute("""
                    SELECT id, result, error, timestamp
                    FROM results 
                    WHERE species_file = ? AND binomial_name = ? AND model = ? 
                    AND system_template = ? AND user_template = ? AND status = 'failed'
                """, (species_file, species_name, model, system_template, user_template))
                
                result = cursor.fetchone()
                if not result:
                    continue
                
                id, result_data, error, timestamp = result
                
                # Re-run the failed species
                system_template_content = read_template_from_file(system_template)
                user_template_content = read_template_from_file(user_template)
                
                # Run in a separate thread
                thread = threading.Thread(
                    target=self._rerun_single_species,
                    args=(combination_id, species_file, species_name, model, 
                          system_template, user_template, system_template_content, user_template_content)
                )
                thread.daemon = True
                thread.start()
            
            conn.commit()
            conn.close()
            
            return True
        
        except Exception as e:
            self._emit_log(combination_id, f"Error re-running failed species: {str(e)}")
            return False

# Initialize processing manager
processing_manager = ProcessingManager()

def get_available_species_files():
    """Get a list of available species data files."""
    species_dir = Path(config.SPECIES_DIR)
    if not species_dir.exists():
        return []
    return [f.name for f in species_dir.iterdir() if f.is_file() and f.name.endswith(('.csv', '.txt'))]

def get_available_templates(template_type):
    """Get a list of available template files for a given type."""
    if template_type == 'system':
        template_dir = Path(config.SYSTEM_TEMPLATES_DIR)
    elif template_type == 'user':
        template_dir = Path(config.USER_TEMPLATES_DIR)
    else:
        return []
    
    if not template_dir.exists():
        return []
    
    return [f.name for f in template_dir.iterdir() if f.is_file() and f.name.endswith('.txt')]

def get_available_template_pairs():
    """
    Get available template pairs (system and user) based on matching filenames.
    Returns a dictionary of template pairs.
    """
    system_templates_dir = Path(config.SYSTEM_TEMPLATES_DIR)
    user_templates_dir = Path(config.USER_TEMPLATES_DIR)
    template_pairs = {}
    
    if not system_templates_dir.exists() or not user_templates_dir.exists():
        return {}
        
    for system_file in system_templates_dir.glob('*.txt'):
        user_file = user_templates_dir / system_file.name
        if user_file.exists():
            template_name = system_file.stem
            template_pairs[template_name] = {
                'system': str(system_file),
                'user': str(user_file)
            }
            
    return template_pairs

def get_template_metadata():
    """Get template metadata with custom names and descriptions."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT system_template, user_template, display_name, description 
        FROM template_metadata
    ''')
    
    metadata = {}
    for row in cursor.fetchall():
        system_template, user_template, display_name, description = row
        key = (system_template, user_template)
        metadata[key] = {
            'display_name': display_name,
            'description': description
        }
    
    conn.close()
    return metadata

def get_template_display_info(system_template, user_template):
    """Get display name and description for template pair from metadata table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT display_name, description, template_type 
        FROM template_metadata 
        WHERE system_template = ? AND user_template = ?
    ''', (system_template, user_template))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'display_name': result[0],
            'description': result[1],
            'template_type': result[2] if result[2] else 'phenotype'
        }
    else:
        # Return default values based on filename
        template_name = Path(system_template).stem
        # Detect template type
        template_type = detect_template_type(user_template)
        return {
            'display_name': template_name,
            'description': f'Template pair: {template_name}',
            'template_type': template_type
        }

def get_popular_models():
    """Get a list of popular models from config."""
    return config.POPULAR_MODELS

def get_available_models_from_db():
    """Get a list of all models used in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT model FROM combinations")
    models = [row[0] for row in cursor.fetchall()]
    conn.close()
    return models

@app.route('/')
def index():
    """Landing page"""
    projects = get_projects_for_page('index')
    return render_template('index.html', projects=projects)

@app.route('/research')
def research():
    """Research projects page"""
    projects = get_projects_for_page('research')
    return render_template('research.html', projects=projects)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/imprint')
def imprint():
    """Imprint page"""
    return render_template('imprint.html')

@app.route('/privacy')
def privacy():
    """Privacy policy page"""
    return render_template('privacy.html')

@app.route('/dashboard')
def dashboard():
    """Technical dashboard for managing processing jobs"""
    dashboard_data = processing_manager.get_dashboard_data()
    
    # Add additional data needed for the modals
    dashboard_data['available_species_files'] = get_available_species_files()
    dashboard_data['popular_models'] = get_popular_models()
    dashboard_data['available_template_pairs'] = get_available_template_pairs()
    
    return render_template('dashboard.html', dashboard_data=dashboard_data)

@app.route('/api/start_combination/<int:combination_id>', methods=['POST'])
def start_combination_api(combination_id):
    if processing_manager.start_combination(combination_id):
        return jsonify({'message': f'Combination {combination_id} started successfully.'})
    else:
        return jsonify({'error': 'Failed to start combination - may already be running or not found'}), 500

@app.route('/api/restart_combination/<int:combination_id>', methods=['POST'])
def restart_combination_api(combination_id):
    success, message = processing_manager.restart_combination(combination_id)
    if success:
        return jsonify({'message': message})
    else:
        return jsonify({'error': message}), 500

@app.route('/api/pause_combination/<int:combination_id>', methods=['POST'])
def pause_combination_api(combination_id):
    success = processing_manager.pause_combination(combination_id)
    if success:
        return jsonify({'message': f'Combination {combination_id} paused successfully.'})
    else:
        return jsonify({'error': 'Failed to pause combination - may not be running'}), 500

@app.route('/api/stop_combination/<int:combination_id>', methods=['POST'])
def stop_combination_api(combination_id):
    success = processing_manager.stop_combination(combination_id)
    if success:
        return jsonify({'message': f'Combination {combination_id} stopped successfully.'})
    else:
        return jsonify({'error': 'Failed to stop combination - may not be running'}), 500

@app.route('/api/set_rate_limit', methods=['POST'])
def set_rate_limit_api():
    data = request.get_json()
    requests_per_second = data.get('requests_per_second', 30.0)
    max_concurrent_requests = data.get('max_concurrent_requests')
    
    processing_manager.set_rate_limit(requests_per_second)
    
    if max_concurrent_requests is not None:
        processing_manager.set_max_concurrent_requests(max_concurrent_requests)
    
    return jsonify({
        'message': f'Settings updated successfully',
        'rate_limit': processing_manager.requests_per_second,
        'max_concurrent_requests': processing_manager.max_concurrent_requests
    })

@app.route('/api/get_settings')
def get_settings_api():
    return jsonify({
        'rate_limit': processing_manager.requests_per_second,
        'max_concurrent_requests': processing_manager.max_concurrent_requests,
        'queue_length': len(processing_manager.job_queue)
    })

@app.route('/api/dashboard_data')
def dashboard_data_api():
    """Get dashboard data via API"""
    data = processing_manager.get_dashboard_data()
    return jsonify(data)

@app.route('/api/validate_predictions', methods=['POST'])
def validate_predictions():
    """Validate and normalize all unvalidated predictions."""
    try:
        validator = PredictionValidator()
        
        # Check if specific job_id provided
        data = request.get_json() or {}
        job_id = data.get('job_id')
        
        if job_id:
            # Validate specific job
            result = validator.validate_job_predictions(job_id, db_path)
        else:
            # Validate all unvalidated predictions
            result = validator.validate_all_unvalidated(db_path)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validation_stats', methods=['GET'])
def get_validation_stats():
    """Get current validation statistics."""
    try:
        validator = PredictionValidator()
        stats = validator.get_validation_stats(db_path)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/template/phenotype')
def get_phenotype_template():
    """Serve the phenotype template JSON file"""
    import json
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'templates', 'validation', 'template1_phenotype.json')
    try:
        with open(template_path, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/database')
def database_browser():
    """Database browser page"""
    return render_template('database_browser.html')

@app.route('/api/database/info')
def get_database_info():
    """Get overview information about all tables"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        tables_info = {}
        
        # Get info for each table
        tables = ['processing_results', 'ground_truth', 'template_metadata', 'managed_models']
        
        for table in tables:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            # Get column count
            cursor.execute(f"PRAGMA table_info({table})")
            columns = len(cursor.fetchall())
            
            table_info = {
                'count': count,
                'columns': columns
            }
            
            # Special handling for processing_results
            if table == 'processing_results':
                cursor.execute("""
                    SELECT 
                        SUM(CASE WHEN validation_status = 'validated' THEN 1 ELSE 0 END) as validated,
                        SUM(CASE WHEN validation_status != 'validated' OR validation_status IS NULL THEN 1 ELSE 0 END) as unvalidated
                    FROM processing_results
                """)
                row = cursor.fetchone()
                table_info['validated'] = row[0] or 0
                table_info['unvalidated'] = row[1] or 0
            
            tables_info[table] = table_info
        
        # Get database file size
        import os
        db_size = os.path.getsize(db_path)
        db_size_mb = round(db_size / (1024 * 1024), 2)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'database': 'microbellm.db',
            'size_mb': db_size_mb,
            'tables': tables_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/database/table/<table_name>')
def get_table_data(table_name):
    """Get data from a specific table"""
    try:
        # Validate table name to prevent SQL injection
        allowed_tables = ['processing_results', 'ground_truth', 'template_metadata', 'managed_models']
        if table_name not in allowed_tables:
            return jsonify({'success': False, 'error': 'Invalid table name'}), 400
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get limit from query params
        limit = request.args.get('limit', 1000, type=int)
        limit = min(limit, 10000)  # Cap at 10000 rows
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Get data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'table': table_name,
            'columns': columns,
            'rows': rows,
            'count': len(rows)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/delete_combination/<combination_id>', methods=['DELETE'])
def delete_combination_api(combination_id):
    """API endpoint to delete a combination and its results."""
    try:
        logger.debug("Deleting combination/job: %s", combination_id)
        
        # Check if job exists first
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processing_results WHERE job_id = ?", (combination_id,))
        count_before = cursor.fetchone()[0]
        logger.debug("Found %d entries for job %s", count_before, combination_id)
        conn.close()
        
        # Use unified database to delete the job
        from microbellm.unified_db import UnifiedDB
        unified_db = UnifiedDB(db_path)  # Use the same db_path as the API
        
        # Delete all entries for this job_id
        unified_db.delete_job(combination_id)
        
        # Verify deletion worked
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processing_results WHERE job_id = ?", (combination_id,))
        count_after = cursor.fetchone()[0]
        logger.debug("After deletion, found %d entries for job %s", count_after, combination_id)
        
        # Also clean up from legacy tables if they exist
        cursor.execute("DELETE FROM combinations WHERE id = ?", (combination_id,))
        cursor.execute("DELETE FROM results WHERE id = ?", (combination_id,))
        
        conn.commit()
        conn.close()
        
        socketio.emit('job_update', {'combination_id': combination_id, 'status': 'deleted'})
        return jsonify({'success': True, 'message': f'Job {combination_id} deleted. Removed {count_before} entries.'})
    except Exception as e:
        logger.exception("Error deleting job %s", combination_id)
        return jsonify({'success': False, 'message': f'Error deleting job: {e}'}), 500

@app.route('/export')
def export_page():
    """Export page with options"""
    dashboard_data = {
        'species_files': get_available_species_files(),
        'models': get_available_models_from_db()
    }
    return render_template('export.html', dashboard_data=dashboard_data)

@app.route('/import')
def import_page():
    """Import page for uploading CSV files"""
    return render_template('import.html')

@app.route('/compare')
def compare_page():
    """Compare results page"""
    # Get unique templates and species files for filters
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT system_template FROM results WHERE status = 'completed'")
    templates = [row[0].split('/')[-1].replace('.txt', '') for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT species_file FROM results WHERE status = 'completed'")
    species_files = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    return render_template('compare.html', templates=templates, species_files=species_files)

@app.route('/correlation')
def correlation_page():
    return render_template('correlation.html')


@app.route('/research/<page>/dynamic')
def research_dynamic_page(page):
    """Dynamic research page renderer using manifest-based approach"""
    # Load manifest
    manifest = load_page_manifest(page)
    if not manifest:
        abort(404)
    
    # Get project data
    project = get_project_by_id(manifest['page_config'].get('project_id', page))
    if not project:
        abort(404)
    
    # Read annotation data if this is knowledge calibration
    annotation_data = {}
    if page == 'knowledge_calibration':
        annotation_file = os.path.join(config.SPECIES_DIR, 'artificial_annotation.txt')
        try:
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Skip header
                        for line in lines[1:]:
                            line = line.strip()
                            if line and ';' in line:
                                parts = line.split(';')
                                if len(parts) >= 3:
                                    type_name = parts[0].strip()
                                    description = parts[1].strip()
                                    example = parts[2].strip()
                                    annotation_data[type_name] = {
                                        'description': description,
                                        'example': example
                                    }
        except Exception as e:
            logger.exception("Error reading annotation file: %s", e)
    
    return render_template('research_dynamic.html',
                         project=project,
                         manifest=manifest,
                         annotations=annotation_data,
                         project_path=f'research/{page}',
                         page_specific_css=f'css/research/{page}/page_specific.css')


@app.route('/search_correlation')
def search_correlation_page():
    return render_template('search_correlation.html')

@app.route('/components')
def components_index():
    """List all available components for testing"""
    components = {}
    
    # Scan research directories for manifests
    research_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'templates', 'research'
    )
    
    if os.path.exists(research_dir):
        for page_name in os.listdir(research_dir):
            page_dir = os.path.join(research_dir, page_name)
            if os.path.isdir(page_dir):
                manifest = load_page_manifest(page_name)
                if manifest:
                    components[page_name] = {
                        'title': manifest['page_config']['title'],
                        'sections': manifest['sections'],
                        'color_theme': manifest['page_config'].get('color_theme', 'purple')
                    }
    
    return render_template('components/index.html', components=components)

@app.route('/debug_layout')
def debug_layout():
    """Debug page for layout issues"""
    return render_template('debug_layout.html')

@app.route('/components/<page>/<section_id>')
def view_component(page, section_id):
    """View a single component in isolation"""
    manifest = load_page_manifest(page)
    if not manifest:
        return "Page not found", 404
        
    # Find the section
    section = None
    for s in manifest['sections']:
        if s['id'] == section_id:
            section = s
            break
    
    if not section:
        return "Section not found", 404
    
    # Get project data
    project = get_project_by_id(manifest['page_config'].get('project_id', page))
    
    # Read the raw content of the section file for all viewers
    section_raw_content = ""
    if section.get('file'):
        # Get absolute path to the template folder
        app_root = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(app_root, 'templates', 'research', page, section['file'])
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                section_raw_content = escape(f.read())
        except Exception as e:
            section_raw_content = f"Error reading file: {str(e)}"
    
    # Check if simple viewer requested (now the default for better dual view)
    if request.args.get('simple') != 'false':
        return render_template(
            'components/viewer_simple.html',
            section=section,
            page=page,
            manifest=manifest,
            project=project,
            section_raw_content=section_raw_content
        )
    
    # Check if debug viewer requested
    if request.args.get('debug') == 'true':
        return render_template(
            'components/viewer_debug.html',
            section=section,
            page=page,
            manifest=manifest,
            project=project
        )
    
    # Check if old viewer requested
    if request.args.get('old') == 'true':
        return render_template(
            'components/viewer.html', 
            section=section, 
            page=page, 
            manifest=manifest, 
            project=project,
            project_path='research/' + page,
            page_specific_css=f'css/research/{page}/page_specific.css',
            section_raw_content=section_raw_content
        )
    
    # Default to regular viewer with raw content
    return render_template(
        'components/viewer.html',
        section=section,
        page=page,
        manifest=manifest,
        project=project,
        section_raw_content=section_raw_content
    )

@app.route('/components/admin_interface_figure')
def admin_interface_figure():
    """Display the admin interface figure for the manuscript"""
    return render_template('components/admin_interface_figure.html')

@app.route('/settings')
def settings_page():
    """Settings page for API key and configuration management"""
    return render_template('settings.html')

@app.route('/templates')
def templates_page():
    """Templates page showing all template pairs side by side"""
    try:
        template_pairs = get_available_template_pairs()
        
        # Create template data for the view
        template_data = {}
        for template_name, paths in template_pairs.items():
            # Read system template
            with open(paths['system'], 'r', encoding='utf-8') as f:
                system_content = f.read()
            
            # Read user template  
            with open(paths['user'], 'r', encoding='utf-8') as f:
                user_content = f.read()
            
            # Try to read validation config
            validation_data = None
            try:
                from microbellm.template_config import find_validation_config_for_template, TemplateValidator
                config_path = find_validation_config_for_template(paths['user'])
                if config_path:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        validation_content = f.read()
                    
                    # Parse validation config to get template info
                    validator = TemplateValidator(config_path)
                    template_info = validator.get_template_info()
                    
                    # Get expected response info
                    expected_response = validator.config.get('expected_response', {}) if validator.config else {}
                    
                    validation_data = {
                        'path': config_path,
                        'content': validation_content,
                        'info': {
                            'type': template_info.get('type', 'unknown'),
                            'description': template_info.get('description', ''),
                            'purpose': template_info.get('purpose', ''),
                            'usage_context': template_info.get('usage_context', {}),
                            'interpretation_guide': template_info.get('interpretation_guide', {}),
                            'quality_indicators': template_info.get('quality_indicators', {}),
                            'required_fields': expected_response.get('required_fields', []),
                            'optional_fields': expected_response.get('optional_fields', [])
                        }
                    }
            except Exception as e:
                logger.warning("Could not load validation config for %s: %s", template_name, e)
            
            # Get display info
            display_info = get_template_display_info(paths['system'], paths['user'])
            
            template_data[template_name] = {
                'system': {
                    'path': paths['system'],
                    'content': system_content
                },
                'user': {
                    'path': paths['user'],
                    'content': user_content
                },
                'validation': validation_data,
                'display_name': display_info['display_name'],
                'description': display_info['description']
            }
        
        return render_template('view_template.html', template_data=template_data)
    
    except Exception as e:
        return f"Error loading templates: {str(e)}", 500

@app.route('/api/template_metadata')
def get_template_metadata_api():
    """API endpoint to get all template metadata"""
    try:
        template_pairs = get_available_template_pairs()
        metadata = []
        
        for template_name, paths in template_pairs.items():
            display_info = get_template_display_info(paths['system'], paths['user'])
            # Detect template type
            template_type = detect_template_type(paths['user'])
            
            metadata.append({
                'template_name': template_name,
                'system_template': paths['system'],
                'user_template': paths['user'],
                'display_name': display_info['display_name'],
                'description': display_info['description'],
                'template_type': display_info.get('template_type', template_type)
            })
        
        return jsonify({'success': True, 'metadata': metadata})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/template_validation_info')
def get_template_validation_info_api():
    """API endpoint to get template validation info from JSON files"""
    try:
        template_pairs = get_available_template_pairs()
        validation_info = {}
        
        for template_name, paths in template_pairs.items():
            # Try to read validation config
            try:
                from microbellm.template_config import find_validation_config_for_template, TemplateValidator
                config_path = find_validation_config_for_template(paths['user'])
                if config_path:
                    # Parse validation config to get template info
                    validator = TemplateValidator(config_path)
                    template_info = validator.get_template_info()
                    
                    validation_info[template_name] = {
                        'type': template_info.get('type', 'unknown'),
                        'description': template_info.get('description', ''),
                        'purpose': template_info.get('purpose', ''),
                        'usage_context': template_info.get('usage_context', {}),
                        'interpretation_guide': template_info.get('interpretation_guide', {}),
                        'quality_indicators': template_info.get('quality_indicators', {})
                    }
            except Exception as e:
                logger.warning("Could not load validation config for %s: %s", template_name, e)
                validation_info[template_name] = {
                    'type': 'unknown',
                    'description': 'Template for model evaluation',
                    'purpose': '',
                    'usage_context': {},
                    'interpretation_guide': {},
                    'quality_indicators': {}
                }
        
        return jsonify({'success': True, 'validation_info': validation_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/update_template_metadata', methods=['POST'])
def update_template_metadata_api():
    """API endpoint to update template metadata"""
    try:
        data = request.get_json()
        system_template = data.get('system_template')
        user_template = data.get('user_template')
        display_name = data.get('display_name')
        description = data.get('description')
        template_type = data.get('template_type', 'phenotype')
        
        if not all([system_template, user_template, display_name]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert or update template metadata
        cursor.execute('''
            INSERT OR REPLACE INTO template_metadata 
            (system_template, user_template, display_name, description, template_type, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (system_template, user_template, display_name, description or '', template_type))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Template metadata updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/comparison_data')
def get_comparison_data():
    """Get comparison data for multiple models on same species"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all species that have been processed by multiple models
        cursor.execute("""
            SELECT binomial_name, species_file, system_template, user_template, COUNT(DISTINCT model) as model_count
            FROM results
            WHERE status = 'completed'
            GROUP BY binomial_name, species_file, system_template, user_template
            HAVING model_count > 1
        """)
        
        multi_model_species = cursor.fetchall()
        
        comparisons = []
        
        for species_name, species_file, system_template, user_template, model_count in multi_model_species:
            # Get all results for this species across different models
            cursor.execute("""
                SELECT model, gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                       biofilm_formation, animal_pathogenicity, biosafety_level, health_association,
                       host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape
                FROM results
                WHERE binomial_name = ? AND species_file = ? AND system_template = ? AND user_template = ?
                AND status = 'completed'
            """, (species_name, species_file, system_template, user_template))
            
            model_results = cursor.fetchall()
            
            # Build phenotype comparison data
            phenotypes = {}
            models = []
            
            phenotype_names = [
                'gram_staining', 'motility', 'aerophilicity', 'extreme_environment_tolerance',
                'biofilm_formation', 'animal_pathogenicity', 'biosafety_level', 'health_association',
                'host_association', 'plant_pathogenicity', 'spore_formation', 'hemolysis', 'cell_shape'
            ]
            
            # Initialize phenotype data structure
            for phenotype in phenotype_names:
                phenotypes[phenotype] = {
                    'predictions': {},
                    'conflicts': {}
                }
            
            # Collect predictions from each model
            for result in model_results:
                model = result[0]
                models.append(model)
                
                for i, phenotype in enumerate(phenotype_names):
                    value = result[i + 1]  # +1 because model is at index 0
                    phenotypes[phenotype]['predictions'][model] = value
            
            # Identify conflicts for each phenotype
            for phenotype, data in phenotypes.items():
                predictions = data['predictions']
                unique_values = set(v for v in predictions.values() if v is not None)
                
                if len(unique_values) > 1:
                    # There's a conflict
                    for model, value in predictions.items():
                        if value is not None:
                            for other_model, other_value in predictions.items():
                                if model != other_model and other_value is not None and value != other_value:
                                    if model not in data['conflicts']:
                                        data['conflicts'][model] = []
                                    data['conflicts'][model].append({
                                        'conflicting_model': other_model,
                                        'this_value': value,
                                        'other_value': other_value
                                    })
            
            comparisons.append({
                'species': species_name,
                'species_file': species_file,
                'template': system_template.split('/')[-1].replace('.txt', ''),
                'models': sorted(models),
                'phenotypes': phenotypes
            })
        
        conn.close()
        
        return jsonify({
            'comparisons': comparisons,
            'total_comparisons': len(comparisons)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search_count_correlation')
def get_search_count_correlation():
    """Get correlation analysis between Google search counts and knowledge level predictions with caching"""
    import time
    import math
    
    try:
        # Check cache first
        current_time = time.time()
        if (_search_correlation_cache and 
            current_time - _search_correlation_cache_timestamp < CACHE_DURATION):
            # Add cache info to cached response
            cached_response = _search_correlation_cache.copy()
            cached_response['cache_info'] = {
                'cached': True,
                'calculated_at': _search_correlation_cache_timestamp,
                'cache_duration': CACHE_DURATION,
                'age_seconds': current_time - _search_correlation_cache_timestamp
            }
            return jsonify(cached_response)
        
        # Cache miss - need to recalculate
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all knowledge level results with template information (optimized query)
        cursor.execute("""
            SELECT r.binomial_name, r.species_file, r.model, r.knowledge_group, 
                   r.system_template, r.user_template
            FROM processing_results r
            WHERE r.knowledge_group IS NOT NULL AND r.status = 'completed'
            ORDER BY r.species_file, r.model
        """)
        
        knowledge_results = cursor.fetchall()
        conn.close()
        
        if not knowledge_results:
            empty_result = {
                'correlation_data': {},
                'files_with_search_counts': [],
                'total_files': 0,
                'cache_info': {
                    'cached': False,
                    'calculated_at': current_time,
                    'cache_duration': CACHE_DURATION
                }
            }
            _update_search_correlation_cache(empty_result, current_time)
            return jsonify(empty_result)
        
        # Process data to extract search count information and organize correlations
        correlation_data = {}
        
        # Cache for search count mappings to avoid re-reading files
        search_count_cache = {}
        
        # Helper to get search count from species file (with caching)
        def get_search_count_mapping(species_file):
            if species_file in search_count_cache:
                return search_count_cache[species_file]
                
            search_count_mapping = {}
            try:
                # Handle both full paths and just filenames
                if os.path.isabs(species_file):
                    # Full path provided - use it directly but normalize the key
                    species_file_path = species_file
                    # Use basename as cache key for consistency
                    cache_key = os.path.basename(species_file)
                else:
                    # Just filename provided - construct the path
                    species_file_path = os.path.join(config.SPECIES_DIR, species_file)
                    cache_key = species_file
                
                with open(species_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Check if this is a file with search count information (CSV or TSV)
                if len(lines) > 0:
                    header_line = lines[0].strip().lower()
                    # Check for both comma and tab separated formats
                    delimiter = None
                    if ',' in header_line and ('search count' in header_line or 'search_count' in header_line):
                        delimiter = ','
                    elif '\t' in header_line and ('search count' in header_line or 'search_count' in header_line):
                        delimiter = '\t'
                    
                    if delimiter:
                        # This is a file with search count information
                        header_parts = [h.strip() for h in header_line.split(delimiter)]
                        
                        # Find search count column index
                        search_count_col_idx = None
                        for i, col in enumerate(header_parts):
                            if 'search count' in col or 'search_count' in col:
                                search_count_col_idx = i
                                break
                        
                        if search_count_col_idx is not None:
                            # Process data lines efficiently
                            for line in lines[1:]:  # Skip header
                                line = line.strip()
                                if line and delimiter in line:
                                    parts = [p.strip() for p in line.split(delimiter)]
                                    if len(parts) > search_count_col_idx:
                                        species_name = parts[0]
                                        try:
                                            search_count = int(parts[search_count_col_idx])
                                            search_count_mapping[species_name] = search_count
                                        except (ValueError, IndexError):
                                            # Skip invalid search counts
                                            pass
                        
            except Exception as e:
                logger.warning("Error reading species file %s: %s", species_file, e)
            
            # Cache the result using normalized key
            search_count_cache[cache_key] = search_count_mapping
            # Also cache with original key for lookup
            if cache_key != species_file:
                search_count_cache[species_file] = search_count_mapping
            return search_count_mapping
        
        # Pre-fetch all search count mappings for files that appear in results
        unique_species_files = list(set(row[1] for row in knowledge_results))
        for species_file in unique_species_files:
            get_search_count_mapping(species_file)
        
        # Filter out files without search counts early
        files_with_search_counts = [f for f in unique_species_files if search_count_cache.get(f)]
        
        if not files_with_search_counts:
            empty_result = {
                'correlation_data': {},
                'files_with_search_counts': [],
                'total_files': 0,
                'cache_info': {
                    'cached': False,
                    'calculated_at': current_time,
                    'cache_duration': CACHE_DURATION
                }
            }
            _update_search_correlation_cache(empty_result, current_time)
            return jsonify(empty_result)
        
        # Cache template display info to avoid repeated calls
        template_display_cache = {}
        
        def get_cached_template_display_info(system_template, user_template):
            key = f"{system_template}|{user_template}"
            if key not in template_display_cache:
                template_display_cache[key] = get_template_display_info(system_template, user_template)
            return template_display_cache[key]
        
        # Process knowledge results and match with search counts (optimized)
        for binomial_name, species_file, model, knowledge_group, system_template, user_template in knowledge_results:
            # Skip files without search count data
            if species_file not in files_with_search_counts:
                continue
                
            # Get search count mapping for this species file
            search_count_mapping = search_count_cache[species_file]
            
            # Get search count for this species
            search_count = search_count_mapping.get(binomial_name)
            if search_count is None:
                # Fallback for old data where the full line might be the species name
                cleaned_name = binomial_name.split('\t')[0].strip()
                search_count = search_count_mapping.get(cleaned_name)

            if search_count is None:
                continue
            
            # Get template display info (cached)
            template_display = get_cached_template_display_info(system_template, user_template)
            display_name = template_display['display_name']
            template_type = template_display['template_type']
            
            # Only include knowledge templates
            if template_type != 'knowledge':
                continue
            
            # Initialize data structure efficiently
            if species_file not in correlation_data:
                correlation_data[species_file] = {}
            
            if display_name not in correlation_data[species_file]:
                correlation_data[species_file][display_name] = {}
            
            if model not in correlation_data[species_file][display_name]:
                correlation_data[species_file][display_name][model] = {
                    'data_points': [],
                    'species_count': 0,
                    'correlation_coefficient': 0,
                    'knowledge_distribution': {'limited': 0, 'moderate': 0, 'extensive': 0, 'NA': 0}
                }
            
            # Convert knowledge level to numerical score for correlation (optimized)
            normalized_knowledge = knowledge_group.lower().strip()
            knowledge_score = None
            
            if normalized_knowledge in ['limited', 'minimal', 'basic', 'low']:
                knowledge_score = 1
                correlation_data[species_file][display_name][model]['knowledge_distribution']['limited'] += 1
            elif normalized_knowledge in ['moderate', 'medium', 'intermediate']:
                knowledge_score = 2
                correlation_data[species_file][display_name][model]['knowledge_distribution']['moderate'] += 1
            elif normalized_knowledge in ['extensive', 'comprehensive', 'detailed', 'high', 'full']:
                knowledge_score = 3
                correlation_data[species_file][display_name][model]['knowledge_distribution']['extensive'] += 1
            elif normalized_knowledge in ['na', 'n/a', 'n.a.', 'not available', 'not applicable', 'unknown']:
                correlation_data[species_file][display_name][model]['knowledge_distribution']['NA'] += 1
                continue  # Skip NA values for correlation calculation
            
            # Add data point only if we have a valid knowledge score
            if knowledge_score is not None:
                correlation_data[species_file][display_name][model]['data_points'].append({
                    'species': binomial_name,
                    'search_count': search_count,
                    'knowledge_score': knowledge_score,
                    'knowledge_group': knowledge_group
                })
        
        # Calculate correlations for each model (optimized)
        def calculate_correlation(data_points):
            """Calculate Pearson correlation coefficient - optimized version"""
            if len(data_points) < 2:
                return 0, 0  # correlation, p_value placeholder
            
            n = len(data_points)
            
            # Pre-calculate log values
            x_log = [math.log10(point['search_count']) if point['search_count'] > 0 else 0 for point in data_points]
            y_values = [point['knowledge_score'] for point in data_points]
            
            # Calculate means
            x_mean = sum(x_log) / n
            y_mean = sum(y_values) / n
            
            # Calculate correlation coefficient in one pass
            numerator = 0
            x_variance = 0
            y_variance = 0
            
            for i in range(n):
                x_diff = x_log[i] - x_mean
                y_diff = y_values[i] - y_mean
                numerator += x_diff * y_diff
                x_variance += x_diff * x_diff
                y_variance += y_diff * y_diff
            
            if x_variance == 0 or y_variance == 0:
                return 0, 0
            
            correlation = numerator / (math.sqrt(x_variance) * math.sqrt(y_variance))
            
            # Simple p-value approximation (for display purposes)
            t_stat = correlation * math.sqrt((n - 2) / (1 - correlation ** 2)) if abs(correlation) < 1 else 0
            p_value = 0.05 if abs(t_stat) > 2 else 0.1  # Rough approximation
            
            return correlation, p_value
        
        # Calculate correlations and update data efficiently
        for species_file in correlation_data:
            for template_name in correlation_data[species_file]:
                for model in correlation_data[species_file][template_name]:
                    model_data = correlation_data[species_file][template_name][model]
                    correlation, p_value = calculate_correlation(model_data['data_points'])
                    model_data['correlation_coefficient'] = correlation
                    model_data['p_value'] = p_value
                    model_data['species_count'] = len(model_data['data_points'])
        
        result = {
            'correlation_data': correlation_data,
            'files_with_search_counts': files_with_search_counts,
            'total_files': len(correlation_data),
            'cache_info': {
                'cached': False,
                'calculated_at': current_time,
                'cache_duration': CACHE_DURATION
            }
        }
        
        # Update cache
        _update_search_correlation_cache(result, current_time)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _update_search_correlation_cache(data, timestamp):
    """Update the global search correlation cache"""
    global _search_correlation_cache, _search_correlation_cache_timestamp
    _search_correlation_cache = data
    _search_correlation_cache_timestamp = timestamp

def _invalidate_search_correlation_cache():
    """Invalidate the search correlation cache (call when results are updated)"""
    global _search_correlation_cache, _search_correlation_cache_timestamp
    _search_correlation_cache = {}
    _search_correlation_cache_timestamp = 0

def _invalidate_knowledge_analysis_cache():
    """Invalidate the knowledge analysis cache (call when results are updated)"""
    global _knowledge_analysis_cache, _knowledge_analysis_cache_timestamp
    _knowledge_analysis_cache = {}
    _knowledge_analysis_cache_timestamp = 0

def _invalidate_all_caches():
    """Invalidate all caches when results are updated"""
    _invalidate_search_correlation_cache()
    _invalidate_knowledge_analysis_cache()

def _ensure_ground_truth_persistence() -> bool:
    """Ensure ground-truth cache tables are available; degrade gracefully if read-only."""
    global _ground_truth_persistence_available

    if _ground_truth_persistence_available is True:
        return True
    if _ground_truth_persistence_available is False:
        return False

    try:
        create_ground_truth_tables()
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if 'readonly' in message:
            logger.warning("Ground truth cache persistence disabled: database is read-only.")
            _ground_truth_persistence_available = False
            return False
        logger.exception("Failed to ensure ground truth tables exist")
        _ground_truth_persistence_available = False
        return False
    except sqlite3.DatabaseError:
        logger.exception("Failed to ensure ground truth tables exist")
        _ground_truth_persistence_available = False
        return False

    _ground_truth_persistence_available = True
    return True

def _update_knowledge_analysis_cache(data, timestamp):
    """Update the global knowledge analysis cache"""
    global _knowledge_analysis_cache, _knowledge_analysis_cache_timestamp
    _knowledge_analysis_cache = data
    _knowledge_analysis_cache_timestamp = timestamp


def _load_persistent_ground_truth_stats(dataset_name):
    if not _ensure_ground_truth_persistence():
        return None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT payload, import_timestamp, computed_at
            FROM ground_truth_statistics_cache
            WHERE dataset_name = ?
            """,
            (dataset_name,)
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    payload_json, import_timestamp, computed_at = row
    if not payload_json:
        return None

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    return {
        'data': payload,
        'import_timestamp': float(import_timestamp or 0),
        'computed_at': float(computed_at or 0)
    }


def _save_persistent_ground_truth_stats(dataset_name, data, import_timestamp, computed_at):
    if not _ensure_ground_truth_persistence():
        logger.debug("Skipping save of ground truth stats for %s; persistence unavailable", dataset_name)
        return
    snapshot_json = json.dumps(data, ensure_ascii=False, default=str)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO ground_truth_statistics_cache (dataset_name, payload, import_timestamp, computed_at, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_name) DO UPDATE SET
            payload = excluded.payload,
            import_timestamp = excluded.import_timestamp,
            computed_at = excluded.computed_at,
            updated_at = CURRENT_TIMESTAMP
        """,
        (dataset_name, snapshot_json, float(import_timestamp or 0), float(computed_at or time.time()))
    )

    conn.commit()
    conn.close()


def _clear_persistent_ground_truth_stats(dataset_name=None):
    if not _ensure_ground_truth_persistence():
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if dataset_name:
        cursor.execute(
            "DELETE FROM ground_truth_statistics_cache WHERE dataset_name = ?",
            (dataset_name,)
        )
    else:
        cursor.execute("DELETE FROM ground_truth_statistics_cache")

    conn.commit()
    conn.close()


def _get_species_file_for_dataset(dataset_name):
    if not dataset_name:
        return 'wa_with_gcount.txt'

    if dataset_name in DATASET_SPECIES_FILE_MAP:
        return DATASET_SPECIES_FILE_MAP[dataset_name]

    normalized = dataset_name.lower()
    if 'artificial' in normalized:
        return 'artificial.txt'
    if 'la_test' in normalized or 'los angeles' in normalized or normalized.endswith('_la'):
        return 'la.txt'
    if 'wa_test' in normalized or 'washington' in normalized or normalized.endswith('_wa'):
        return 'wa_with_gcount.txt'

    return 'wa_with_gcount.txt'


def _normalize_model_key(name):
    if not name:
        return ''
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _load_model_metadata_index():
    global _model_metadata_index

    with _model_metadata_index_lock:
        if _model_metadata_index is not None:
            return _model_metadata_index

        metadata_path = Path(__file__).resolve().parent / 'static' / 'data' / 'year_size.tsv'
        index = {}

        if metadata_path.exists():
            try:
                with metadata_path.open('r', encoding='utf-8') as handle:
                    reader = csv.DictReader(handle, delimiter='\t')
                    for row in reader:
                        model_name = (row.get('Model') or '').strip()
                        if not model_name:
                            continue

                        entry = {
                            'model': model_name,
                            'organization': (row.get('Organization') or '').strip() or None,
                            'publication_date': (row.get('Publication date') or '').strip() or None,
                            'parameters': (row.get('Parameters') or '').strip() or None,
                            'reference': (row.get('Reference') or '').strip() or None,
                            'raw': row
                        }

                        keys = set()
                        keys.add(_normalize_model_key(model_name))
                        for part in re.split(r'[\s/]+', model_name):
                            normalized_part = _normalize_model_key(part)
                            if normalized_part:
                                keys.add(normalized_part)

                        for key in keys:
                            if key and key not in index:
                                index[key] = entry

            except Exception as exc:  # pragma: no cover - debug logging
                logger.debug("Failed to parse year_size.tsv metadata: %s", exc)

        _model_metadata_index = index
        return _model_metadata_index


def _normalize_publication_date(value):
    if not value:
        return None

    try:
        timestamp = pd.to_datetime(value, errors='coerce')
    except Exception:
        return None

    if pd.isna(timestamp):
        return None

    if hasattr(timestamp, 'to_pydatetime'):
        timestamp = timestamp.to_pydatetime()

    return timestamp.date().isoformat() if hasattr(timestamp, 'date') else None


def _match_model_metadata(model_name):
    if not model_name:
        return None

    metadata_index = _load_model_metadata_index()
    if not metadata_index:
        return None

    normalized_full = _normalize_model_key(model_name)
    prefixes = (
        'anthropic', 'openai', 'google', 'gemini', 'metaai', 'metallama',
        'mistralai', 'cohere', 'xai', 'ai21', 'amazon', 'baidu', 'together'
    )

    candidates = []
    seen = set()

    def add_candidate(key):
        if key and key not in seen:
            seen.add(key)
            candidates.append(key)

    add_candidate(normalized_full)

    if '/' in model_name:
        add_candidate(_normalize_model_key(model_name.split('/')[-1]))

    for part in re.split(r'[-_\s/]+', model_name):
        add_candidate(_normalize_model_key(part))

    for key in list(candidates):
        alias = _MODEL_METADATA_ALIAS_MAP.get(key)
        if alias:
            add_candidate(alias)

    for key in list(candidates):
        for prefix in prefixes:
            if key.startswith(prefix) and len(key) > len(prefix) + 2:
                add_candidate(key[len(prefix):])

    for key in candidates:
        if key and key in metadata_index:
            return metadata_index[key]

    for key, entry in metadata_index.items():
        if key and (key in normalized_full or normalized_full in key):
            return entry

    return None


def _load_persistent_model_accuracy(dataset_name):
    if not _ensure_ground_truth_persistence():
        return None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT payload, import_timestamp, computed_at
            FROM model_accuracy_cache
            WHERE dataset_name = ?
            """,
            (dataset_name,)
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    payload_json, import_timestamp, computed_at = row
    if not payload_json:
        return None

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    return {
        'data': payload,
        'import_timestamp': float(import_timestamp or 0),
        'computed_at': float(computed_at or 0)
    }


def _save_persistent_model_accuracy(dataset_name, data, import_timestamp, computed_at):
    if not _ensure_ground_truth_persistence():
        logger.debug("Skipping save of model accuracy cache for %s; persistence unavailable", dataset_name)
        return
    snapshot_json = json.dumps(data, ensure_ascii=False, default=str)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO model_accuracy_cache (dataset_name, payload, import_timestamp, computed_at, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_name) DO UPDATE SET
            payload = excluded.payload,
            import_timestamp = excluded.import_timestamp,
            computed_at = excluded.computed_at,
            updated_at = CURRENT_TIMESTAMP
        """,
        (dataset_name, snapshot_json, float(import_timestamp or 0), float(computed_at or time.time()))
    )

    conn.commit()
    conn.close()


def _clear_persistent_model_accuracy(dataset_name=None):
    if not _ensure_ground_truth_persistence():
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if dataset_name:
        cursor.execute(
            "DELETE FROM model_accuracy_cache WHERE dataset_name = ?",
            (dataset_name,)
        )
    else:
        cursor.execute("DELETE FROM model_accuracy_cache")

    conn.commit()
    conn.close()


def _load_persistent_performance_year(dataset_name):
    if not _ensure_ground_truth_persistence():
        return None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT payload, import_timestamp, computed_at
            FROM model_performance_year_cache
            WHERE dataset_name = ?
            """,
            (dataset_name,)
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    payload_json, import_timestamp, computed_at = row
    if not payload_json:
        return None

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    return {
        'data': payload,
        'import_timestamp': float(import_timestamp or 0),
        'computed_at': float(computed_at or 0)
    }


def _save_persistent_performance_year(dataset_name, data, import_timestamp, computed_at):
    if not _ensure_ground_truth_persistence():
        logger.debug("Skipping save of performance-by-year cache for %s; persistence unavailable", dataset_name)
        return
    snapshot_json = json.dumps(data, ensure_ascii=False, default=str)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO model_performance_year_cache (dataset_name, payload, import_timestamp, computed_at, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_name) DO UPDATE SET
            payload = excluded.payload,
            import_timestamp = excluded.import_timestamp,
            computed_at = excluded.computed_at,
            updated_at = CURRENT_TIMESTAMP
        """,
        (dataset_name, snapshot_json, float(import_timestamp or 0), float(computed_at or time.time()))
    )

    conn.commit()
    conn.close()


def _clear_persistent_performance_year(dataset_name=None):
    if not _ensure_ground_truth_persistence():
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if dataset_name:
        cursor.execute(
            "DELETE FROM model_performance_year_cache WHERE dataset_name = ?",
            (dataset_name,)
        )
    else:
        cursor.execute("DELETE FROM model_performance_year_cache")

    conn.commit()
    conn.close()


def _load_persistent_knowledge_accuracy(dataset_name):
    if not _ensure_ground_truth_persistence():
        return None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT payload, import_timestamp, computed_at
            FROM knowledge_accuracy_cache
            WHERE dataset_name = ?
            """,
            (dataset_name,)
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    payload_json, import_timestamp, computed_at = row
    if not payload_json:
        return None

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    return {
        'data': payload,
        'import_timestamp': float(import_timestamp or 0),
        'computed_at': float(computed_at or 0)
    }


def _save_persistent_knowledge_accuracy(dataset_name, data, import_timestamp, computed_at):
    if not _ensure_ground_truth_persistence():
        logger.debug("Skipping save of knowledge accuracy cache for %s; persistence unavailable", dataset_name)
        return
    snapshot_json = json.dumps(data, ensure_ascii=False, default=str)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO knowledge_accuracy_cache (dataset_name, payload, import_timestamp, computed_at, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_name) DO UPDATE SET
            payload = excluded.payload,
            import_timestamp = excluded.import_timestamp,
            computed_at = excluded.computed_at,
            updated_at = CURRENT_TIMESTAMP
        """,
        (dataset_name, snapshot_json, float(import_timestamp or 0), float(computed_at or time.time()))
    )

    conn.commit()
    conn.close()


def _clear_persistent_knowledge_accuracy(dataset_name=None):
    if not _ensure_ground_truth_persistence():
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if dataset_name:
        cursor.execute(
            "DELETE FROM knowledge_accuracy_cache WHERE dataset_name = ?",
            (dataset_name,)
        )
    else:
        cursor.execute("DELETE FROM knowledge_accuracy_cache")

    conn.commit()
    conn.close()


def _coerce_timestamp(value):
    """Convert a string or numeric timestamp into float seconds since epoch."""
    if value in (None, ''):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def _get_ground_truth_dataset_import_metadata(dataset_name, cursor=None):
    """Fetch import metadata for a ground truth dataset."""
    own_conn = None
    if cursor is None:
        own_conn = sqlite3.connect(db_path)
        cursor = own_conn.cursor()

    cursor.execute(
        """
        SELECT import_date, species_count
        FROM ground_truth_datasets
        WHERE dataset_name = ?
        """,
        (dataset_name,)
    )
    row = cursor.fetchone()

    if own_conn:
        own_conn.close()

    import_date = row[0] if row else None
    reported_species_count = row[1] if row and len(row) > 1 else None
    return {
        'import_date': import_date,
        'import_timestamp': _coerce_timestamp(import_date),
        'reported_species_count': int(reported_species_count) if reported_species_count is not None else None
    }


def _calculate_ground_truth_statistics(dataset_name, metadata=None):
    """Compute phenotype statistics for a ground truth dataset."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if metadata is None:
            metadata = _get_ground_truth_dataset_import_metadata(dataset_name, cursor)
        else:
            metadata = {
                'import_date': metadata.get('import_date'),
                'import_timestamp': _coerce_timestamp(metadata.get('import_timestamp')),
                'reported_species_count': (
                    int(metadata['reported_species_count'])
                    if metadata.get('reported_species_count') is not None
                    else None
                )
            }

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ground_truth'"
        )
        if not cursor.fetchone():
            return {
                'dataset_name': dataset_name,
                'total_species': 0,
                'statistics': [],
                'metadata': {
                    **metadata,
                    'calculated_at': datetime.utcnow().isoformat() + 'Z'
                }
            }

        cursor.execute(
            "SELECT COUNT(*) FROM ground_truth WHERE dataset_name = ?",
            (dataset_name,)
        )
        total_row = cursor.fetchone()
        total_species = int(total_row[0]) if total_row else 0

        statistics = []

        if total_species == 0:
            result = {
                'dataset_name': dataset_name,
                'total_species': 0,
                'statistics': [],
                'metadata': {
                    **metadata,
                    'calculated_at': datetime.utcnow().isoformat() + 'Z'
                }
            }
            return result

        for field, info in GROUND_TRUTH_PHENOTYPE_DEFINITIONS.items():
            cursor.execute(
                f"""
                SELECT COUNT(*) FROM ground_truth
                WHERE dataset_name = ?
                  AND {field} IS NOT NULL
                  AND {field} != ''
                  AND {field} != 'NA'
                """,
                (dataset_name,)
            )
            annotated_count = int(cursor.fetchone()[0])

            annotation_fraction = (annotated_count / total_species * 100) if total_species > 0 else 0

            cursor.execute(
                f"""
                SELECT {field}, COUNT(*) as count
                FROM ground_truth
                WHERE dataset_name = ?
                  AND {field} IS NOT NULL
                  AND {field} != ''
                  AND {field} != 'NA'
                GROUP BY {field}
                ORDER BY count DESC
                """,
                (dataset_name,)
            )

            value_distribution = {}
            rows = cursor.fetchall()
            for value, count in rows:
                label = '' if value is None else str(value)
                value_distribution[label] = {
                    'count': int(count),
                    'percentage': round((count / annotated_count * 100) if annotated_count > 0 else 0, 1)
                }

            statistics.append({
                'field': field,
                'label': info['label'],
                'type': info['type'],
                'targets': info['targets'],
                'total_species': total_species,
                'annotated_species': annotated_count,
                'missing_species': total_species - annotated_count,
                'annotation_fraction': round(annotation_fraction, 1),
                'value_distribution': value_distribution
            })

        statistics.sort(key=lambda item: item['annotation_fraction'], reverse=True)

        return {
            'dataset_name': dataset_name,
            'total_species': total_species,
            'statistics': statistics,
            'metadata': {
                **metadata,
                'calculated_at': datetime.utcnow().isoformat() + 'Z'
            }
        }

    finally:
        conn.close()


def _get_ground_truth_stats_cache_entry(dataset_name):
    with _ground_truth_stats_cache_lock:
        entry = _ground_truth_stats_cache.get(dataset_name)
        if entry:
            return {
                'data': entry['data'],
                'computed_at': entry.get('computed_at', 0),
                'import_timestamp': entry.get('import_timestamp', 0)
            }

    persistent_entry = _load_persistent_ground_truth_stats(dataset_name)
    if persistent_entry:
        with _ground_truth_stats_cache_lock:
            _ground_truth_stats_cache[dataset_name] = {
                'data': persistent_entry['data'],
                'computed_at': persistent_entry.get('computed_at', 0),
                'import_timestamp': persistent_entry.get('import_timestamp', 0)
            }
        return persistent_entry

    return None


def _update_ground_truth_stats_cache(dataset_name, data, import_timestamp, computed_at=None):
    timestamp = time.time() if computed_at is None else computed_at
    normalized_json = json.dumps(data, ensure_ascii=False, default=str)
    normalized_data = json.loads(normalized_json)

    with _ground_truth_stats_cache_lock:
        _ground_truth_stats_cache[dataset_name] = {
            'data': normalized_data,
            'computed_at': float(timestamp),
            'import_timestamp': float(import_timestamp or 0)
        }

    _save_persistent_ground_truth_stats(
        dataset_name,
        normalized_data,
        float(import_timestamp or 0),
        float(timestamp)
    )


def _invalidate_ground_truth_stats_cache(dataset_name=None):
    with _ground_truth_stats_cache_lock:
        if dataset_name:
            _ground_truth_stats_cache.pop(dataset_name, None)
        else:
            _ground_truth_stats_cache.clear()

    _clear_persistent_ground_truth_stats(dataset_name)


def _calculate_model_accuracy_metrics(dataset_name, metadata=None):
    """Compute model accuracy metrics for a dataset."""
    if metadata is None:
        metadata = _get_ground_truth_dataset_import_metadata(dataset_name)
    else:
        metadata = {
            'import_date': metadata.get('import_date'),
            'import_timestamp': _coerce_timestamp(metadata.get('import_timestamp')),
            'reported_species_count': (
                int(metadata['reported_species_count'])
                if metadata.get('reported_species_count') is not None
                else None
            )
        }

    species_file = _get_species_file_for_dataset(dataset_name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT r.binomial_name, r.model,
                   r.gram_staining, r.motility, r.aerophilicity,
                   r.extreme_environment_tolerance, r.biofilm_formation,
                   r.animal_pathogenicity, r.biosafety_level, r.health_association,
                   r.host_association, r.plant_pathogenicity, r.spore_formation,
                   r.hemolysis, r.cell_shape
            FROM processing_results r
            WHERE r.species_file = ?
              AND r.user_template LIKE '%phenotype%'
              AND r.status = 'completed'
              AND (
                   r.gram_staining IS NOT NULL OR r.motility IS NOT NULL OR
                   r.aerophilicity IS NOT NULL OR r.extreme_environment_tolerance IS NOT NULL OR
                   r.biofilm_formation IS NOT NULL OR r.animal_pathogenicity IS NOT NULL OR
                   r.biosafety_level IS NOT NULL OR r.health_association IS NOT NULL OR
                   r.host_association IS NOT NULL OR r.plant_pathogenicity IS NOT NULL OR
                   r.spore_formation IS NOT NULL OR r.hemolysis IS NOT NULL OR
                   r.cell_shape IS NOT NULL
              )
            ORDER BY r.model, r.binomial_name
            """,
            (species_file,)
        )

        prediction_rows = cursor.fetchall()
    finally:
        conn.close()

    ground_truth_records = get_ground_truth_data(dataset_name=dataset_name)
    ground_truth_map = {
        (record.get('binomial_name') or '').lower(): record
        for record in ground_truth_records
        if record.get('binomial_name')
    }

    total_species = len(ground_truth_records)

    predictions = []
    for row in prediction_rows:
        (binomial_name, model,
         gram_staining, motility, aerophilicity,
         extreme_environment_tolerance, biofilm_formation,
         animal_pathogenicity, biosafety_level, health_association,
         host_association, plant_pathogenicity, spore_formation,
         hemolysis, cell_shape) = row

        predictions.append({
            'binomial_name': binomial_name,
            'model': model,
            'gram_staining': gram_staining,
            'motility': motility,
            'aerophilicity': aerophilicity,
            'extreme_environment_tolerance': extreme_environment_tolerance,
            'biofilm_formation': biofilm_formation,
            'animal_pathogenicity': animal_pathogenicity,
            'biosafety_level': biosafety_level,
            'health_association': health_association,
            'host_association': host_association,
            'plant_pathogenicity': plant_pathogenicity,
            'spore_formation': spore_formation,
            'hemolysis': hemolysis,
            'cell_shape': cell_shape
        })

    models = sorted({pred['model'] for pred in predictions if pred.get('model')})
    phenotype_fields = list(GROUND_TRUTH_PHENOTYPE_DEFINITIONS.keys())

    predictions_by_model = {}
    for pred in predictions:
        model = pred.get('model')
        if not model:
            continue
        predictions_by_model.setdefault(model, []).append(pred)

    def normalize_metric_value(value):
        normalized = normalize_value(value)
        if not normalized or normalized == 'NA':
            return None
        cleaned = str(normalized).strip().lower()
        if ',' in cleaned or ';' in cleaned:
            parts = [part.strip() for part in re.split(r'[;,]', cleaned) if part.strip()]
            if not parts:
                return None
            cleaned = ','.join(sorted(parts))
        return cleaned

    def compute_classification_metrics(pred_list, truth_list):
        if not pred_list or not truth_list:
            return None

        labels = sorted(set(truth_list) | set(pred_list))
        if not labels:
            return None

        confusion = {label: {inner: 0 for inner in labels} for label in labels}
        for truth, pred in zip(truth_list, pred_list):
            if truth in confusion and pred in confusion[truth]:
                confusion[truth][pred] += 1

        recall_total = 0.0
        precision_total = 0.0
        f1_total = 0.0

        for label in labels:
            tp = confusion[label][label]
            fn = sum(confusion[label][other] for other in labels if other != label)
            fp = sum(confusion[other][label] for other in labels if other != label)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            recall_total += recall
            precision_total += precision
            f1_total += f1

        num_labels = len(labels)
        balanced_acc = recall_total / num_labels if num_labels else 0.0
        macro_precision = precision_total / num_labels if num_labels else 0.0
        macro_recall = recall_total / num_labels if num_labels else 0.0
        macro_f1 = f1_total / num_labels if num_labels else 0.0

        confusion_matrix = [
            [confusion[row_label][col_label] for col_label in labels]
            for row_label in labels
        ]

        return {
            'balancedAcc': balanced_acc,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
            'sampleSize': len(truth_list),
            'confusionMatrix': confusion_matrix,
            'labels': labels
        }

    def safe_metric_value(value):
        if value is None:
            return None
        try:
            if math.isnan(value):
                return None
        except TypeError:
            pass
        return round(float(value), 6)

    metrics = []

    for phenotype in phenotype_fields:
        for model, model_predictions in predictions_by_model.items():
            truth_values = []
            pred_values = []

            for pred in model_predictions:
                species_name = (pred.get('binomial_name') or '').lower()
                if not species_name:
                    continue
                ground_truth = ground_truth_map.get(species_name)
                if not ground_truth:
                    continue

                truth_norm = normalize_metric_value(ground_truth.get(phenotype))
                pred_norm = normalize_metric_value(pred.get(phenotype))

                if truth_norm is None or pred_norm is None:
                    continue

                truth_values.append(truth_norm)
                pred_values.append(pred_norm)

            if not truth_values:
                continue

            metric_result = compute_classification_metrics(pred_values, truth_values)
            if not metric_result:
                continue

            metrics.append({
                'model': model,
                'phenotype': phenotype,
                'balancedAcc': safe_metric_value(metric_result['balancedAcc']),
                'precision': safe_metric_value(metric_result['precision']),
                'recall': safe_metric_value(metric_result['recall']),
                'f1': safe_metric_value(metric_result['f1']),
                'sampleSize': metric_result['sampleSize'],
                'confusionMatrix': metric_result['confusionMatrix'],
                'labels': metric_result['labels']
            })

    metrics_generated_at = datetime.utcnow().isoformat() + 'Z'

    summary = {
        'model_count': len(models),
        'phenotype_count': len(phenotype_fields),
        'prediction_count': len(predictions),
        'ground_truth_species': total_species,
        'species_file': species_file,
        'metrics_count': len(metrics)
    }

    return {
        'dataset_name': dataset_name,
        'species_file': species_file,
        'metrics': metrics,
        'models': models,
        'phenotypes': phenotype_fields,
        'summary': summary,
        'metadata': {
            **metadata,
            'calculated_at': metrics_generated_at,
            'species_file': species_file
        }
    }


def _get_model_accuracy_cache_entry(dataset_name):
    with _model_accuracy_cache_lock:
        entry = _model_accuracy_cache.get(dataset_name)
        if entry:
            return {
                'data': entry['data'],
                'computed_at': entry.get('computed_at', 0),
                'import_timestamp': entry.get('import_timestamp', 0)
            }

    persistent_entry = _load_persistent_model_accuracy(dataset_name)
    if persistent_entry:
        with _model_accuracy_cache_lock:
            _model_accuracy_cache[dataset_name] = {
                'data': persistent_entry['data'],
                'computed_at': persistent_entry.get('computed_at', 0),
                'import_timestamp': persistent_entry.get('import_timestamp', 0)
            }
        return persistent_entry

    return None


def _update_model_accuracy_cache(dataset_name, data, import_timestamp, computed_at=None):
    timestamp = time.time() if computed_at is None else computed_at
    normalized_json = json.dumps(data, ensure_ascii=False, default=str)
    normalized_data = json.loads(normalized_json)

    with _model_accuracy_cache_lock:
        _model_accuracy_cache[dataset_name] = {
            'data': normalized_data,
            'computed_at': float(timestamp),
            'import_timestamp': float(import_timestamp or 0)
        }

    _save_persistent_model_accuracy(
        dataset_name,
        normalized_data,
        float(import_timestamp or 0),
        float(timestamp)
    )


def _invalidate_model_accuracy_cache(dataset_name=None):
    with _model_accuracy_cache_lock:
        if dataset_name:
            _model_accuracy_cache.pop(dataset_name, None)
        else:
            _model_accuracy_cache.clear()

    _clear_persistent_model_accuracy(dataset_name)


def _get_performance_year_cache_entry(dataset_name):
    with _model_performance_year_cache_lock:
        entry = _model_performance_year_cache.get(dataset_name)
        if entry:
            return {
                'data': entry['data'],
                'computed_at': entry.get('computed_at', 0),
                'import_timestamp': entry.get('import_timestamp', 0)
            }

    persistent_entry = _load_persistent_performance_year(dataset_name)
    if persistent_entry:
        with _model_performance_year_cache_lock:
            _model_performance_year_cache[dataset_name] = {
                'data': persistent_entry['data'],
                'computed_at': persistent_entry.get('computed_at', 0),
                'import_timestamp': persistent_entry.get('import_timestamp', 0)
            }
        return persistent_entry

    return None


def _update_performance_year_cache(dataset_name, data, import_timestamp, computed_at=None):
    timestamp = time.time() if computed_at is None else computed_at
    normalized_json = json.dumps(data, ensure_ascii=False, default=str)
    normalized_data = json.loads(normalized_json)

    with _model_performance_year_cache_lock:
        _model_performance_year_cache[dataset_name] = {
            'data': normalized_data,
            'computed_at': float(timestamp),
            'import_timestamp': float(import_timestamp or 0)
        }

    _save_persistent_performance_year(
        dataset_name,
        normalized_data,
        float(import_timestamp or 0),
        float(timestamp)
    )


def _invalidate_performance_year_cache(dataset_name=None):
    with _model_performance_year_cache_lock:
        if dataset_name:
            _model_performance_year_cache.pop(dataset_name, None)
        else:
            _model_performance_year_cache.clear()

    _clear_persistent_performance_year(dataset_name)


def _calculate_performance_by_year_metrics(dataset_name, metadata=None):
    if not dataset_name:
        raise ValueError('Dataset name is required')

    metadata = metadata or _get_ground_truth_dataset_import_metadata(dataset_name)
    import_timestamp = float((metadata or {}).get('import_timestamp') or 0)

    accuracy_entry = _get_model_accuracy_cache_entry(dataset_name)
    accuracy_data = None

    if accuracy_entry and import_timestamp <= accuracy_entry.get('import_timestamp', 0):
        accuracy_data = accuracy_entry['data']
    else:
        computed_at = time.time()
        accuracy_data = _calculate_model_accuracy_metrics(dataset_name, metadata=metadata)
        _update_model_accuracy_cache(
            dataset_name,
            accuracy_data,
            float(accuracy_data.get('metadata', {}).get('import_timestamp', 0)),
            computed_at=computed_at
        )

    metrics_list = accuracy_data.get('metrics') or []
    if not metrics_list:
        return {
            'dataset_name': dataset_name,
            'phenotypes': [],
            'metric_names': ['balanced_accuracy', 'precision', 'recall', 'f1'],
            'models': [],
            'summary': {
                'total_models': 0,
                'models_with_metadata': 0,
                'models_without_metadata': [],
                'metadata_source': 'static/data/year_size.tsv',
                'average_sample_threshold': AVERAGE_SAMPLE_SIZE_THRESHOLD
            },
            'metadata': {
                'import_date': (metadata or {}).get('import_date'),
                'import_timestamp': import_timestamp,
                'reported_species_count': (metadata or {}).get('reported_species_count'),
                'species_file': accuracy_data.get('species_file')
            }
        }

    def safe_float(value):
        if value in (None, ''):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    def safe_round(value):
        clean = safe_float(value)
        if clean is None:
            return None
        return round(clean, 4)

    phenotype_metrics = {}
    for entry in metrics_list:
        model = entry.get('model')
        phenotype = entry.get('phenotype')
        if not model or not phenotype:
            continue

        stats = phenotype_metrics.setdefault(model, {})
        stats[phenotype] = {
            'balanced_accuracy': safe_float(entry.get('balancedAcc')),
            'precision': safe_float(entry.get('precision')),
            'recall': safe_float(entry.get('recall')),
            'f1': safe_float(entry.get('f1')),
            'sample_size': int(entry.get('sampleSize') or 0)
        }

    metric_names = ['balanced_accuracy', 'precision', 'recall', 'f1']
    phenotypes = sorted({p for stats in phenotype_metrics.values() for p in stats.keys()})

    models_payload = []
    models_with_metadata = 0
    models_without_metadata = []

    for model_name, phenotype_values in phenotype_metrics.items():
        metrics_payload = {}
        average_sums = {metric: 0.0 for metric in metric_names}
        average_counts = {metric: 0 for metric in metric_names}
        included_phenotypes = []
        average_sample_sum = 0

        for phenotype, stats in phenotype_values.items():
            metric_entry = {
                'balanced_accuracy': safe_round(stats.get('balanced_accuracy')),
                'precision': safe_round(stats.get('precision')),
                'recall': safe_round(stats.get('recall')),
                'f1': safe_round(stats.get('f1')),
                'sample_size': int(stats.get('sample_size') or 0),
                'phenotype_count': 1
            }
            metrics_payload[phenotype] = metric_entry

            if metric_entry['sample_size'] >= AVERAGE_SAMPLE_SIZE_THRESHOLD:
                contributed = False
                for metric in metric_names:
                    value = stats.get(metric)
                    if value is None:
                        continue
                    average_sums[metric] += value
                    average_counts[metric] += 1
                    contributed = True
                if contributed:
                    included_phenotypes.append(phenotype)
                    average_sample_sum += metric_entry['sample_size']

        if included_phenotypes:
            average_entry = {
                metric: safe_round(average_sums[metric] / average_counts[metric])
                if average_counts[metric] else None
                for metric in metric_names
            }
            average_entry['phenotype_count'] = len(included_phenotypes)
            average_entry['sample_size'] = average_sample_sum
            metrics_payload['average'] = average_entry

        metadata_entry = _match_model_metadata(model_name)
        publication_date = _normalize_publication_date(metadata_entry.get('publication_date')) if metadata_entry else None
        organization = (metadata_entry or {}).get('organization')
        parameters = (metadata_entry or {}).get('parameters')
        display_name = (metadata_entry or {}).get('model') or model_name

        if publication_date:
            models_with_metadata += 1
        else:
            models_without_metadata.append(model_name)

        if not organization and '/' in model_name:
            organization = model_name.split('/')[0].replace('-', ' ').title()

        models_payload.append({
            'model': model_name,
            'display_name': display_name,
            'organization': organization or 'Unknown',
            'publication_date': publication_date,
            'parameters': parameters,
            'metrics': metrics_payload
        })

    result_metadata = {
        'import_date': (metadata or {}).get('import_date'),
        'import_timestamp': import_timestamp,
        'reported_species_count': (metadata or {}).get('reported_species_count'),
        'species_file': accuracy_data.get('species_file'),
        'accuracy_calculated_at': accuracy_data.get('metadata', {}).get('calculated_at')
    }

    if result_metadata['reported_species_count'] is not None:
        try:
            result_metadata['reported_species_count'] = int(result_metadata['reported_species_count'])
        except (TypeError, ValueError):
            result_metadata['reported_species_count'] = None

    ordered_phenotypes = ['average'] if any('average' in m['metrics'] for m in models_payload) else []
    ordered_phenotypes.extend([p for p in phenotypes if p])

    return {
        'dataset_name': dataset_name,
        'phenotypes': ordered_phenotypes,
        'metric_names': metric_names,
        'models': models_payload,
        'summary': {
            'total_models': len(models_payload),
            'models_with_metadata': models_with_metadata,
            'models_without_metadata': sorted(models_without_metadata),
            'metadata_source': 'static/data/year_size.tsv',
            'average_sample_threshold': AVERAGE_SAMPLE_SIZE_THRESHOLD
        },
        'metadata': result_metadata
    }


def _normalize_knowledge_value(phenotype, value):
    if value is None:
        return None

    str_value = str(value).strip().lower()
    if not str_value or str_value in {'na', 'n/a', '-', 'null', 'none', 'nan'}:
        return None

    if phenotype == 'gram_staining':
        if 'positive' in str_value:
            return 'positive'
        if 'negative' in str_value:
            return 'negative'
        if 'variable' in str_value:
            return 'variable'

    if phenotype == 'biosafety_level':
        if '1' in str_value:
            return 'level_1'
        if '2' in str_value:
            return 'level_2'
        if '3' in str_value:
            return 'level_3'

    if phenotype == 'cell_shape':
        if 'bacillus' in str_value:
            return 'bacillus'
        if 'coccus' in str_value:
            return 'coccus'
        if 'spirillum' in str_value:
            return 'spirillum'
        if 'tail' in str_value:
            return 'tail'
        if 'filamentous' in str_value:
            return 'filamentous'

    if str_value in {'true', 't', 'yes', '1'}:
        return True
    if str_value in {'false', 'f', 'no', '0'}:
        return False

    return str_value


def _calculate_balanced_accuracy_from_confusion(confusion):
    labels = set(confusion.keys())
    for truth_map in confusion.values():
        labels.update(truth_map.keys())

    labels = sorted(labels)
    if not labels:
        return 0.0

    recalls = []
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fn = sum(confusion.get(label, {}).values()) - tp
        fp = sum(confusion.get(other, {}).get(label, 0) for other in labels if other != label)
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        recalls.append(recall)

    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def _calculate_knowledge_accuracy_metrics(dataset_name, metadata=None):
    if metadata is None:
        metadata = _get_ground_truth_dataset_import_metadata(dataset_name)
    else:
        metadata = {
            'import_date': metadata.get('import_date'),
            'import_timestamp': _coerce_timestamp(metadata.get('import_timestamp')),
            'reported_species_count': (
                int(metadata['reported_species_count'])
                if metadata.get('reported_species_count') is not None
                else None
            )
        }

    species_file = _get_species_file_for_dataset(dataset_name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT 
                pheno.model,
                know.knowledge_group,
                pheno.binomial_name,
                pheno.gram_staining,
                pheno.motility,
                pheno.aerophilicity,
                pheno.extreme_environment_tolerance,
                pheno.biofilm_formation,
                pheno.animal_pathogenicity,
                pheno.biosafety_level,
                pheno.health_association,
                pheno.host_association,
                pheno.plant_pathogenicity,
                pheno.spore_formation,
                pheno.hemolysis,
                pheno.cell_shape
            FROM processing_results pheno
            INNER JOIN processing_results know 
                ON pheno.model = know.model 
                AND pheno.binomial_name = know.binomial_name 
                AND pheno.species_file = know.species_file
            WHERE 
                pheno.user_template LIKE '%template1_phenotype%'
                AND know.user_template LIKE '%template3_knowlege%'
                AND pheno.species_file = ?
                AND know.knowledge_group IS NOT NULL
            ORDER BY pheno.model, know.knowledge_group, pheno.binomial_name
            """,
            (species_file,)
        )

        rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT DISTINCT 
                LOWER(binomial_name) as binomial_name,
                gram_staining,
                motility,
                aerophilicity,
                extreme_environment_tolerance,
                biofilm_formation,
                animal_pathogenicity,
                biosafety_level,
                health_association,
                host_association,
                plant_pathogenicity,
                spore_formation,
                hemolysis,
                cell_shape
            FROM ground_truth
            WHERE dataset_name = ?
            """,
            (dataset_name,)
        )

        ground_truth_map = {}
        for row in cursor.fetchall():
            ground_truth_map[row[0]] = {
                'gram_staining': row[1],
                'motility': row[2],
                'aerophilicity': row[3],
                'extreme_environment_tolerance': row[4],
                'biofilm_formation': row[5],
                'animal_pathogenicity': row[6],
                'biosafety_level': row[7],
                'health_association': row[8],
                'host_association': row[9],
                'plant_pathogenicity': row[10],
                'spore_formation': row[11],
                'hemolysis': row[12],
                'cell_shape': row[13]
            }

    finally:
        conn.close()

    phenotype_fields = [
        'gram_staining',
        'motility',
        'extreme_environment_tolerance',
        'biofilm_formation',
        'animal_pathogenicity',
        'biosafety_level',
        'host_association',
        'plant_pathogenicity',
        'spore_formation',
        'cell_shape'
    ]

    MIN_SAMPLES_PER_PHENOTYPE = 30

    aggregates = {}
    models = set()
    knowledge_groups = set()

    for row in rows:
        model = row[0]
        knowledge_group = row[1] or 'NA'
        binomial_name = row[2]

        if not binomial_name:
            continue

        species_key = binomial_name.lower()
        ground_truth = ground_truth_map.get(species_key)
        if not ground_truth:
            continue

        key = (model, knowledge_group)
        if key not in aggregates:
            aggregates[key] = {
                'phenotypes': {field: {'confusion': defaultdict(lambda: defaultdict(int)), 'count': 0, 'correct': 0} for field in phenotype_fields},
                'species': set(),
                'total_predictions': 0,
                'total_correct': 0
            }

        agg = aggregates[key]
        agg['species'].add(species_key)
        models.add(model)
        knowledge_groups.add(knowledge_group)

        row_values = {
            'gram_staining': row[3],
            'motility': row[4],
            'aerophilicity': row[5],
            'extreme_environment_tolerance': row[6],
            'biofilm_formation': row[7],
            'animal_pathogenicity': row[8],
            'biosafety_level': row[9],
            'health_association': row[10],
            'host_association': row[11],
            'plant_pathogenicity': row[12],
            'spore_formation': row[13],
            'hemolysis': row[14],
            'cell_shape': row[15]
        }

        for field in phenotype_fields:
            predicted = _normalize_knowledge_value(field, row_values.get(field))
            truth = _normalize_knowledge_value(field, ground_truth.get(field))

            if predicted is None or truth is None:
                continue

            phen_stats = agg['phenotypes'][field]
            phen_stats['confusion'][truth][predicted] += 1
            phen_stats['count'] += 1
            agg['total_predictions'] += 1
            if predicted == truth:
                phen_stats['correct'] += 1
                agg['total_correct'] += 1

    entries = []
    phenotype_order = phenotype_fields[:]
    knowledge_order = ['NA', 'limited', 'moderate', 'extensive']

    for (model, knowledge_group), data in aggregates.items():
        phenotype_accuracies = {}
        per_phenotype_scores = []
        total_samples = 0

        for phenotype, stats in data['phenotypes'].items():
            if stats['count'] < MIN_SAMPLES_PER_PHENOTYPE:
                continue

            balanced_accuracy = _calculate_balanced_accuracy_from_confusion(stats['confusion'])
            accuracy_percentage = balanced_accuracy * 100 if math.isfinite(balanced_accuracy) else 0.0
            phenotype_accuracies[phenotype] = {
                'accuracy': accuracy_percentage,
                'correct': stats['correct'],
                'total': stats['count']
            }
            per_phenotype_scores.append(accuracy_percentage)
            total_samples += stats['count']

        if not phenotype_accuracies:
            continue

        overall_accuracy = sum(per_phenotype_scores) / len(per_phenotype_scores) if per_phenotype_scores else 0.0

        entries.append({
            'model': model,
            'knowledge_group': knowledge_group,
            'overall_accuracy': overall_accuracy,
            'sample_size': len(data['species']),
            'total_predictions': total_samples,
            'phenotype_accuracies': phenotype_accuracies
        })

    entries.sort(key=lambda item: (item['model'], knowledge_order.index(item['knowledge_group']) if item['knowledge_group'] in knowledge_order else 999))

    summary = {
        'model_count': len(models),
        'knowledge_group_count': len(knowledge_groups),
        'phenotype_count': len(phenotype_fields),
        'entry_count': len(entries),
        'ground_truth_species': len(ground_truth_map)
    }

    return {
        'dataset_name': dataset_name,
        'entries': entries,
        'models': sorted(models),
        'knowledge_groups': sorted(knowledge_groups, key=lambda g: knowledge_order.index(g) if g in knowledge_order else 999),
        'phenotypes': phenotype_order,
        'summary': summary,
        'metadata': {
            **metadata,
            'calculated_at': datetime.utcnow().isoformat() + 'Z',
            'species_file': species_file
        }
    }

def _get_knowledge_accuracy_cache_entry(dataset_name):
    with _knowledge_accuracy_cache_lock:
        entry = _knowledge_accuracy_cache.get(dataset_name)
        if entry:
            return {
                'data': entry['data'],
                'computed_at': entry.get('computed_at', 0),
                'import_timestamp': entry.get('import_timestamp', 0)
            }

    persistent_entry = _load_persistent_knowledge_accuracy(dataset_name)
    if persistent_entry:
        with _knowledge_accuracy_cache_lock:
            _knowledge_accuracy_cache[dataset_name] = {
                'data': persistent_entry['data'],
                'computed_at': persistent_entry.get('computed_at', 0),
                'import_timestamp': persistent_entry.get('import_timestamp', 0)
            }
        return persistent_entry

    return None


def _update_knowledge_accuracy_cache(dataset_name, data, import_timestamp, computed_at=None):
    timestamp = time.time() if computed_at is None else computed_at
    normalized_json = json.dumps(data, ensure_ascii=False, default=str)
    normalized_data = json.loads(normalized_json)

    with _knowledge_accuracy_cache_lock:
        _knowledge_accuracy_cache[dataset_name] = {
            'data': normalized_data,
            'computed_at': float(timestamp),
            'import_timestamp': float(import_timestamp or 0)
        }

    _save_persistent_knowledge_accuracy(
        dataset_name,
        normalized_data,
        float(import_timestamp or 0),
        float(timestamp)
    )


def _invalidate_knowledge_accuracy_cache(dataset_name=None):
    with _knowledge_accuracy_cache_lock:
        if dataset_name:
            _knowledge_accuracy_cache.pop(dataset_name, None)
        else:
            _knowledge_accuracy_cache.clear()

    _clear_persistent_knowledge_accuracy(dataset_name)

@app.route('/api/knowledge_analysis_data')
def get_knowledge_analysis_data():
    """Get knowledge level analysis data stratified by input type with caching"""
    import time
    
    try:
        # Check cache first
        current_time = time.time()
        if (_knowledge_analysis_cache and 
            current_time - _knowledge_analysis_cache_timestamp < CACHE_DURATION):
            # Add cache info to cached response
            cached_response = _knowledge_analysis_cache.copy()
            cached_response['cache_info'] = {
                'cached': True,
                'calculated_at': _knowledge_analysis_cache_timestamp,
                'cache_duration': CACHE_DURATION,
                'age_seconds': current_time - _knowledge_analysis_cache_timestamp
            }
            return jsonify(cached_response)
        
        # Cache miss - need to recalculate
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all knowledge level results with template information (including failed ones)
        cursor.execute("""
            SELECT r.binomial_name, r.species_file, r.model, r.knowledge_group, 
                   r.system_template, r.user_template, r.status, r.result
            FROM processing_results r
            WHERE (r.knowledge_group IS NOT NULL AND r.status = 'completed') 
               OR r.status = 'failed'
            ORDER BY r.species_file, r.binomial_name, r.model
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        # Process data to organize by species file and then by type within each file
        file_data = {}
        
        # Helper to get type from species file
        def get_species_type_mapping(species_file):
            type_mapping = {}
            has_type_column = False
            try:
                # Handle both full paths and just filenames
                if os.path.isabs(species_file):
                    # Full path provided - use it directly if it exists
                    species_file_path = species_file
                    # Also normalize the species_file to just the basename for consistency
                    species_file = os.path.basename(species_file)
                else:
                    # Just filename provided - construct the path
                    species_file_path = os.path.join(config.SPECIES_DIR, species_file)
                
                with open(species_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Check if this is a CSV file with type information
                if len(lines) > 0:
                    header_line = lines[0].strip().lower()
                    if ',' in header_line and ('type' in header_line or 'category' in header_line):
                        # This is a CSV with type information
                        header_parts = [h.strip() for h in header_line.split(',')]
                        
                        # Find type column index
                        type_col_idx = None
                        for i, col in enumerate(header_parts):
                            if 'type' in col or 'category' in col:
                                type_col_idx = i
                                break
                        
                        if type_col_idx is not None:
                            has_type_column = True
                            # Process data lines
                            for line in lines[1:]:  # Skip header
                                line = line.strip()
                                if line and ',' in line:
                                    parts = [p.strip() for p in line.split(',')]
                                    if len(parts) > max(0, type_col_idx):
                                        species_name = parts[0]
                                        species_type = parts[type_col_idx] if type_col_idx < len(parts) else 'unclassified'
                                        type_mapping[species_name] = species_type
                
                # If no type mapping found, assign all to the filename (without extension)
                if not type_mapping:
                    from microbellm.utils import filter_species_list
                    species_list = filter_species_list(lines)
                    file_label = os.path.splitext(species_file)[0]  # Use filename without extension
                    for species in species_list:
                        type_mapping[species] = file_label
                        
            except Exception as e:
                logger.warning("Error reading species file %s: %s", species_file, e)
                # Default to filename
                pass
            
            return type_mapping, has_type_column
        
        # Cache for species type mappings
        species_type_cache = {}
        file_has_types = {}
        
        for binomial_name, species_file, model, knowledge_group, system_template, user_template, status, raw_result in results:
            # Normalize species_file to just the basename for consistency
            # This handles both full paths and relative paths to ensure deduplication
            normalized_species_file = os.path.basename(species_file)
            
            # Get type mapping for this species file
            if normalized_species_file not in species_type_cache:
                species_type_cache[normalized_species_file], file_has_types[normalized_species_file] = get_species_type_mapping(species_file)
            
            species_type = species_type_cache[normalized_species_file].get(binomial_name, 'unclassified')
            
            # Get template info for categorization
            template_key = f"{system_template}|{user_template}"
            
            # Initialize nested structure: file -> type -> template -> model
            if normalized_species_file not in file_data:
                file_data[normalized_species_file] = {
                    'has_type_column': file_has_types[normalized_species_file],
                    'types': {}
                }
            
            if species_type not in file_data[normalized_species_file]['types']:
                file_data[normalized_species_file]['types'][species_type] = {}
            
            if template_key not in file_data[normalized_species_file]['types'][species_type]:
                file_data[normalized_species_file]['types'][species_type][template_key] = {}
                
            if model not in file_data[normalized_species_file]['types'][species_type][template_key]:
                file_data[normalized_species_file]['types'][species_type][template_key][model] = {
                    'limited': 0,
                    'moderate': 0, 
                    'extensive': 0,
                    'NA': 0,
                    'no_result': 0,
                    'inference_failed': 0,
                    'total': 0,
                    'samples': {
                        'limited': [],
                        'moderate': [],
                        'extensive': [],
                        'NA': [],
                        'no_result': [],
                        'inference_failed': []
                    }
                }
            
            # Create sample data for tooltip
            sample_data = {
                'species': binomial_name,
                'raw_response': raw_result or 'No response received',
                'knowledge_group': knowledge_group
            }
            
            # Handle failed inference
            if status == 'failed':
                file_data[normalized_species_file]['types'][species_type][template_key][model]['inference_failed'] += 1
                file_data[normalized_species_file]['types'][species_type][template_key][model]['samples']['inference_failed'].append(sample_data)
            elif knowledge_group:
                # Normalize knowledge group value for completed inferences
                normalized_knowledge = knowledge_group.lower().strip()
                if normalized_knowledge in ['limited', 'minimal', 'basic', 'low']:
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['limited'] += 1
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['samples']['limited'].append(sample_data)
                elif normalized_knowledge in ['moderate', 'medium', 'intermediate']:
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['moderate'] += 1
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['samples']['moderate'].append(sample_data)
                elif normalized_knowledge in ['extensive', 'comprehensive', 'detailed', 'high', 'full']:
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['extensive'] += 1
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['samples']['extensive'].append(sample_data)
                elif normalized_knowledge in ['na', 'n/a', 'n.a.', 'not available', 'not applicable', 'unknown']:
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['NA'] += 1
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['samples']['NA'].append(sample_data)
                else:
                    # Unrecognized response - treat as no result
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['no_result'] += 1
                    file_data[normalized_species_file]['types'][species_type][template_key][model]['samples']['no_result'].append(sample_data)
            else:
                # Null/empty knowledge_group for completed status - treat as no result (parsing failure)
                file_data[normalized_species_file]['types'][species_type][template_key][model]['no_result'] += 1
                file_data[normalized_species_file]['types'][species_type][template_key][model]['samples']['no_result'].append(sample_data)
            
            file_data[normalized_species_file]['types'][species_type][template_key][model]['total'] += 1
        
        # Format the data for the frontend with template display names
        formatted_data = {}
        
        for species_file, file_info in file_data.items():
            formatted_data[species_file] = {
                'has_type_column': file_info['has_type_column'],
                'types': {}
            }
            
            for species_type, templates in file_info['types'].items():
                formatted_data[species_file]['types'][species_type] = {}
                
                for template_key, models in templates.items():
                    system_template, user_template = template_key.split('|')
                    
                    # Get template display info
                    template_display = get_template_display_info(system_template, user_template)
                    display_name = template_display['display_name']
                    template_type = template_display['template_type']
                    
                    # Only include knowledge templates
                    if template_type == 'knowledge':
                        formatted_data[species_file]['types'][species_type][display_name] = models
        
        result = {
            'knowledge_analysis': formatted_data,
            'total_files': len(formatted_data),
            'file_list': list(formatted_data.keys()),
            'cache_info': {
                'cached': False,
                'calculated_at': current_time,
                'cache_duration': CACHE_DURATION
            }
        }
        
        # Update cache
        _update_knowledge_analysis_cache(result, current_time)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/template_field_definitions')
def get_template_field_definitions():
    """API endpoint to get field definitions from template validation files"""
    try:
        template_name = request.args.get('template', 'template1_phenotype')
        
        # Find the validation config file
        from microbellm.template_config import find_validation_config_for_template, TemplateValidator
        
        # Construct the user template path
        user_template_path = f"templates/user/{template_name}.txt"
        config_path = find_validation_config_for_template(user_template_path)
        
        if not config_path:
            return jsonify({'success': False, 'error': f'No validation config found for template {template_name}'}), 404
        
        # Load the template validator
        validator = TemplateValidator(config_path)
        if not validator.config:
            return jsonify({'success': False, 'error': f'Could not load validation config for template {template_name}'}), 500
        
        # Extract field definitions and template info
        field_definitions = validator.config.get('field_definitions', {})
        template_info = validator.get_template_info()
        
        # Process field definitions to extract allowed values and create legend mappings
        legend_data = {}
        value_orderings = {}
        color_mappings = {}
        
        for field_name, field_def in field_definitions.items():
            allowed_values = field_def.get('allowed_values', [])
            description = field_def.get('description', '')
            field_type_raw = field_def.get('type', 'string') # e.g. 'string', 'array'
            
            # Determine field category for UI filtering
            if field_type_raw == 'array':
                field_category = 'multi-select'
            elif field_type_raw == 'string' and len(allowed_values) > 2:
                field_category = 'categorical'
            else: # Defaults to binary for string with 2 or less values
                field_category = 'binary'
            
            visualization = field_def.get('visualization', {})
            color_mapping = visualization.get('color_mapping', {})
            
            # Store the ordering from the template
            value_orderings[field_name] = allowed_values
            
            # Store the color mapping from the template
            color_mappings[field_name] = color_mapping
            
            # Create legend items
            legend_items = []
            for value in allowed_values:
                color_info = color_mapping.get(value, {})
                legend_items.append({
                    'value': value,
                    'label': color_info.get('label', value.title()),
                    'canonical': value,
                    'background': color_info.get('background', '#f8f9fa'),
                    'color': color_info.get('color', '#495057')
                })
            
            # Always add NA as the last item
            legend_items.append({
                'value': 'NA',
                'label': 'NA/Failed',
                'canonical': 'NA',
                'background': '#e2e3e5',
                'color': '#6c757d'
            })
            
            legend_data[field_name] = {
                'name': field_name.replace('_', ' ').title(),
                'description': description,
                'type': field_type_raw,
                'field_type': field_category,
                'items': legend_items
            }
        
        return jsonify({
            'success': True,
            'template_name': template_name,
            'template_info': template_info,
            'field_definitions': field_definitions,
            'legend_data': legend_data,
            'value_orderings': value_orderings,
            'color_mappings': color_mappings
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search_count_data')
def get_search_count_data():
    """Get search count data for visualization"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get search count data
        cursor.execute("""
            SELECT binomial_name, search_count
            FROM search_count
            ORDER BY search_count DESC
        """)
        search_counts = [dict(row) for row in cursor.fetchall()]
        
        # Try to get accuracy data if available (optional)
        accuracy_data = {}
        try:
            # This is a placeholder - you can join with actual accuracy data if available
            cursor.execute("""
                SELECT DISTINCT binomial_name
                FROM ground_truth
            """)
            species_with_data = [row['binomial_name'] for row in cursor.fetchall()]
            
            # For demo purposes, we'll just mark which species have ground truth data
            for species in species_with_data:
                accuracy_data[species] = 0.75  # Placeholder accuracy value
                
        except:
            pass
        
        conn.close()
        
        return jsonify({
            'search_counts': search_counts,
            'accuracy_data': accuracy_data
        })
        
    except Exception as e:
        logger.error(f"Error fetching search count data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search_count_by_knowledge')
def get_search_count_by_knowledge():
    """Get search count data grouped by knowledge level"""
    try:
        species_file = request.args.get('species_file', 'wa_with_gcount.txt')
        model = request.args.get('model', '')
        
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # First, get available models if no model specified
        if not model:
            cursor.execute("""
                SELECT DISTINCT model
                FROM processing_results
                WHERE 
                    user_template LIKE '%template3_knowlege%'
                    AND species_file = ?
                    AND knowledge_group IS NOT NULL
                ORDER BY model
            """, (species_file,))
            
            models = [row['model'] for row in cursor.fetchall()]
            
            if not models:
                return jsonify({
                    'success': False,
                    'error': 'No knowledge group data found',
                    'available_models': []
                })
            
            # Use first model as default
            model = models[0] if not model else model
        else:
            models = []
        
        # Get knowledge groups for species with selected model
        cursor.execute("""
            SELECT DISTINCT 
                binomial_name,
                knowledge_group,
                model
            FROM processing_results
            WHERE 
                user_template LIKE '%template3_knowlege%'
                AND species_file = ?
                AND model = ?
                AND knowledge_group IS NOT NULL
        """, (species_file, model))
        
        knowledge_mapping = {}
        for row in cursor.fetchall():
            knowledge_mapping[row['binomial_name']] = row['knowledge_group']
        
        # Now get search counts and join with knowledge groups
        cursor.execute("""
            SELECT 
                sc.binomial_name,
                sc.search_count
            FROM search_count sc
        """)
        
        # Group search counts by knowledge level
        grouped_data = {
            'NA': [],
            'limited': [],
            'moderate': [],
            'extensive': []
        }
        
        species_with_both = 0
        species_without_knowledge = []
        
        for row in cursor.fetchall():
            species = row['binomial_name']
            search_count = row['search_count']
            
            if species in knowledge_mapping:
                knowledge_group = knowledge_mapping[species]
                if knowledge_group in grouped_data:
                    grouped_data[knowledge_group].append({
                        'species': species,
                        'search_count': search_count
                    })
                    species_with_both += 1
            else:
                species_without_knowledge.append(species)
        
        # Calculate statistics for each group
        stats_by_group = {}
        for group, species_list in grouped_data.items():
            if species_list:
                counts = [s['search_count'] for s in species_list]
                counts.sort()
                
                stats_by_group[group] = {
                    'count': len(counts),
                    'min': min(counts),
                    'q1': counts[len(counts)//4] if len(counts) > 3 else counts[0],
                    'median': counts[len(counts)//2],
                    'q3': counts[3*len(counts)//4] if len(counts) > 3 else counts[-1],
                    'max': max(counts),
                    'mean': sum(counts) / len(counts),
                    'values': counts  # All values for box plot
                }
            else:
                stats_by_group[group] = {
                    'count': 0,
                    'values': []
                }
        
        # Get all available models if not already fetched
        if not models:
            cursor.execute("""
                SELECT DISTINCT model
                FROM processing_results
                WHERE 
                    user_template LIKE '%template3_knowlege%'
                    AND species_file = ?
                    AND knowledge_group IS NOT NULL
                ORDER BY model
            """, (species_file,))
            models = [row['model'] for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats_by_group': stats_by_group,
            'raw_data': grouped_data,
            'metadata': {
                'species_file': species_file,
                'selected_model': model,
                'available_models': models,
                'total_species_with_both': species_with_both,
                'total_species_without_knowledge': len(species_without_knowledge)
            }
        })
        
    except Exception as e:
        logger.exception("Error fetching search count by knowledge")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/available_phenotype_templates')
def get_available_phenotype_templates():
    """API endpoint to get list of available phenotype templates with full content"""
    try:
        template_pairs = get_available_template_pairs()
        phenotype_templates = {}
        
        for template_name, paths in template_pairs.items():
            # Check if it's a phenotype template
            template_type = detect_template_type(paths['user'])
            if template_type == 'phenotype':
                # Read system template
                with open(paths['system'], 'r', encoding='utf-8') as f:
                    system_content = f.read()
                
                # Read user template  
                with open(paths['user'], 'r', encoding='utf-8') as f:
                    user_content = f.read()
                
                # Try to read validation config
                validation_content = None
                try:
                    from microbellm.template_config import find_validation_config_for_template
                    config_path = find_validation_config_for_template(paths['user'])
                    if config_path:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            validation_content = f.read()
                except Exception as e:
                    logger.warning("Could not load validation config for %s: %s", template_name, e)
                
                phenotype_templates[template_name] = {
                    'name': template_name,
                    'display_name': template_name.replace('_', ' ').title(),
                    'system': {
                        'path': paths['system'],
                        'content': system_content
                    },
                    'user': {
                        'path': paths['user'],
                        'content': user_content
                    },
                    'validation': {
                        'content': validation_content
                    } if validation_content else None
                }
        
        return jsonify({
            'success': True,
            'templates': phenotype_templates
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/available_knowledge_templates')
def get_available_knowledge_templates():
    """API endpoint to get list of available knowledge templates with full content"""
    try:
        template_pairs = get_available_template_pairs()
        knowledge_templates = {}
        
        for template_name, paths in template_pairs.items():
            # Check if it's a knowledge template
            template_type = detect_template_type(paths['user'])
            if template_type == 'knowledge':
                # Read system template
                with open(paths['system'], 'r', encoding='utf-8') as f:
                    system_content = f.read()
                
                # Read user template  
                with open(paths['user'], 'r', encoding='utf-8') as f:
                    user_content = f.read()
                
                # Try to read validation config
                validation_content = None
                try:
                    from microbellm.template_config import find_validation_config_for_template
                    config_path = find_validation_config_for_template(paths['user'])
                    if config_path:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            validation_content = f.read()
                except Exception as e:
                    logger.warning("Could not load validation config for %s: %s", template_name, e)
                
                knowledge_templates[template_name] = {
                    'name': template_name,
                    'display_name': template_name.replace('_', ' ').title(),
                    'system': {
                        'path': paths['system'],
                        'content': system_content
                    },
                    'user': {
                        'path': paths['user'],
                        'content': user_content
                    },
                    'validation': {
                        'content': validation_content
                    } if validation_content else None
                }
        
        return jsonify({
            'success': True,
            'templates': knowledge_templates
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/phenotype_analysis')
def get_phenotype_analysis():
    """Get phenotype prediction analysis data"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all phenotype results (excluding knowledge templates)
        # Use detect_template_type or check for phenotype fields instead of relying on template_metadata
        cursor.execute("""
            SELECT r.binomial_name, r.species_file, r.model, r.status,
                   r.system_template, r.user_template,
                   r.gram_staining, r.motility, r.aerophilicity, 
                   r.extreme_environment_tolerance, r.biofilm_formation,
                   r.animal_pathogenicity, r.biosafety_level, r.health_association,
                   r.host_association, r.plant_pathogenicity, r.spore_formation,
                   r.hemolysis, r.cell_shape
            FROM processing_results r
            WHERE r.user_template LIKE '%phenotype%'
                  AND (r.gram_staining IS NOT NULL OR r.motility IS NOT NULL 
                       OR r.aerophilicity IS NOT NULL OR r.biofilm_formation IS NOT NULL)
            ORDER BY r.species_file, r.model, r.binomial_name
        """)
        
        results = cursor.fetchall()
        
        # Get list of unique species files that have phenotype data
        cursor.execute("""
            SELECT DISTINCT r.species_file
            FROM processing_results r
            WHERE r.user_template LIKE '%phenotype%'
                  AND (r.gram_staining IS NOT NULL OR r.motility IS NOT NULL 
                       OR r.aerophilicity IS NOT NULL OR r.biofilm_formation IS NOT NULL)
            ORDER BY r.species_file
        """)
        
        files = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Process data by file
        file_data = {}
        
        for row in results:
            (binomial_name, species_file, model, status,
             system_template, user_template,
             gram_staining, motility, aerophilicity,
             extreme_environment_tolerance, biofilm_formation,
             animal_pathogenicity, biosafety_level, health_association,
             host_association, plant_pathogenicity, spore_formation,
             hemolysis, cell_shape) = row
            
            if species_file not in file_data:
                file_data[species_file] = []
            
            # Create phenotype data entry
            entry = {
                'binomial_name': binomial_name,
                'model': model,
                'status': status,
                'system_template': system_template,
                'user_template': user_template,
                'gram_staining': gram_staining,
                'motility': motility,
                'aerophilicity': aerophilicity,
                'extreme_environment_tolerance': extreme_environment_tolerance,
                'biofilm_formation': biofilm_formation,
                'animal_pathogenicity': animal_pathogenicity,
                'biosafety_level': biosafety_level,
                'health_association': health_association,
                'host_association': host_association,
                'plant_pathogenicity': plant_pathogenicity,
                'spore_formation': spore_formation,
                'hemolysis': hemolysis,
                'cell_shape': cell_shape
            }
            
            file_data[species_file].append(entry)
        
        return jsonify({
            'files': files,
            'data': file_data,
            'total_results': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/phenotype_analysis_filtered')
def get_phenotype_analysis_filtered():
    """Get filtered phenotype prediction data for model accuracy analysis"""
    try:
        # Get filter parameters - REQUIRE species_file to avoid loading all data
        species_file = request.args.get('species_file')
        
        if not species_file:
            # Return empty result if no species_file specified to avoid loading all data
            return jsonify({
                'data': [],
                'total_results': 0,
                'message': 'Please specify species_file parameter'
            })
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Only get data for the specified species file to reduce data transfer
        cursor.execute("""
            SELECT r.binomial_name, r.model,
                   r.gram_staining, r.motility, r.aerophilicity, 
                   r.extreme_environment_tolerance, r.biofilm_formation,
                   r.animal_pathogenicity, r.biosafety_level, r.health_association,
                   r.host_association, r.plant_pathogenicity, r.spore_formation,
                   r.hemolysis, r.cell_shape
            FROM processing_results r
            WHERE r.species_file = ?
                  AND r.user_template LIKE '%phenotype%'
                  AND (r.gram_staining IS NOT NULL OR r.motility IS NOT NULL 
                       OR r.aerophilicity IS NOT NULL OR r.biofilm_formation IS NOT NULL)
            ORDER BY r.model, r.binomial_name
        """, (species_file,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Process data into simple list format
        data = []
        for row in results:
            (binomial_name, model,
             gram_staining, motility, aerophilicity,
             extreme_environment_tolerance, biofilm_formation,
             animal_pathogenicity, biosafety_level, health_association,
             host_association, plant_pathogenicity, spore_formation,
             hemolysis, cell_shape) = row
            
            entry = {
                'binomial_name': binomial_name,
                'model': model,
                'gram_staining': gram_staining,
                'motility': motility,
                'aerophilicity': aerophilicity,
                'extreme_environment_tolerance': extreme_environment_tolerance,
                'biofilm_formation': biofilm_formation,
                'animal_pathogenicity': animal_pathogenicity,
                'biosafety_level': biosafety_level,
                'health_association': health_association,
                'host_association': host_association,
                'plant_pathogenicity': plant_pathogenicity,
                'spore_formation': spore_formation,
                'hemolysis': hemolysis,
                'cell_shape': cell_shape
            }
            data.append(entry)
        
        return jsonify({
            'data': data,
            'total_results': len(results),
            'species_file': species_file
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/phenotype_accuracy_by_knowledge')
def get_phenotype_accuracy_by_knowledge():
    """Get phenotype prediction accuracy grouped by knowledge level"""
    try:
        dataset_name = request.args.get('dataset', 'WA_Test_Dataset')
        species_file = request.args.get('species_file')

        def resolve_species_file(dataset: str) -> str:
            if not dataset:
                return 'wa_with_gcount.txt'
            lowered = dataset.lower()
            if 'la_test' in lowered:
                return 'la.txt'
            if 'artificial' in lowered:
                return 'artificial.txt'
            return 'wa_with_gcount.txt'

        if not species_file:
            species_file = resolve_species_file(dataset_name)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query to get phenotype predictions with corresponding knowledge groups
        cursor.execute("""
            SELECT 
                pheno.model,
                know.knowledge_group,
                pheno.binomial_name,
                pheno.gram_staining,
                pheno.motility,
                pheno.aerophilicity,
                pheno.extreme_environment_tolerance,
                pheno.biofilm_formation,
                pheno.animal_pathogenicity,
                pheno.biosafety_level,
                pheno.health_association,
                pheno.host_association,
                pheno.plant_pathogenicity,
                pheno.spore_formation,
                pheno.hemolysis,
                pheno.cell_shape
            FROM processing_results pheno
            INNER JOIN processing_results know 
                ON pheno.model = know.model 
                AND pheno.binomial_name = know.binomial_name 
                AND pheno.species_file = know.species_file
            WHERE 
                pheno.user_template LIKE '%template1_phenotype%'
                AND know.user_template LIKE '%template3_knowlege%'
                AND pheno.species_file = ?
                AND know.knowledge_group IS NOT NULL
            ORDER BY pheno.model, know.knowledge_group, pheno.binomial_name
        """, (species_file,))
        
        results = cursor.fetchall()
        
        # Also get ground truth data
        cursor.execute("""
            SELECT DISTINCT 
                LOWER(binomial_name) as binomial_name,
                gram_staining,
                motility,
                aerophilicity,
                extreme_environment_tolerance,
                biofilm_formation,
                animal_pathogenicity,
                biosafety_level,
                health_association,
                host_association,
                plant_pathogenicity,
                spore_formation,
                hemolysis,
                cell_shape
            FROM ground_truth
            WHERE dataset_name = ?
        """, (dataset_name,))
        
        ground_truth = {}
        for row in cursor.fetchall():
            ground_truth[row[0]] = {
                'gram_staining': row[1],
                'motility': row[2],
                'aerophilicity': row[3],
                'extreme_environment_tolerance': row[4],
                'biofilm_formation': row[5],
                'animal_pathogenicity': row[6],
                'biosafety_level': row[7],
                'health_association': row[8],
                'host_association': row[9],
                'plant_pathogenicity': row[10],
                'spore_formation': row[11],
                'hemolysis': row[12],
                'cell_shape': row[13]
            }
        
        conn.close()
        
        # Process results into structured format
        data = []
        for row in results:
            model = row[0]
            knowledge_group = row[1]
            binomial_name = row[2].lower() if row[2] else None
            
            if binomial_name and binomial_name in ground_truth:
                entry = {
                    'model': model,
                    'knowledge_group': knowledge_group,
                    'binomial_name': binomial_name,
                    'predictions': {
                        'gram_staining': row[3],
                        'motility': row[4],
                        'aerophilicity': row[5],
                        'extreme_environment_tolerance': row[6],
                        'biofilm_formation': row[7],
                        'animal_pathogenicity': row[8],
                        'biosafety_level': row[9],
                        'health_association': row[10],
                        'host_association': row[11],
                        'plant_pathogenicity': row[12],
                        'spore_formation': row[13],
                        'hemolysis': row[14],
                        'cell_shape': row[15]
                    },
                    'ground_truth': ground_truth[binomial_name]
                }
                data.append(entry)
        
        species_count = len(ground_truth)

        return jsonify({
            'success': True,
            'data': data,
            'total_results': len(data),
            'models': list(set([d['model'] for d in data])),
            'knowledge_groups': list(set([d['knowledge_group'] for d in data])),
            'dataset': dataset_name,
            'species_file': species_file,
            'species_count': species_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/phenotype_datasets')
def get_phenotype_datasets():
    """Get list of available phenotype datasets"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT species_file, COUNT(DISTINCT binomial_name) as count
            FROM processing_results
            WHERE user_template LIKE '%phenotype%'
            GROUP BY species_file
            ORDER BY count DESC
        """)
        
        datasets = []
        for row in cursor.fetchall():
            species_file, count = row
            if species_file:
                # Create display name from file name
                display_name = species_file.replace('.txt', '').replace('_', ' ').title()
                if 'wa_with_gcount' in species_file.lower():
                    display_name = 'WA Dataset'
                elif 'artificial' in species_file.lower():
                    display_name = 'Artificial Dataset'
                
                datasets.append({
                    'species_file': species_file,
                    'display_name': display_name,
                    'count': count
                })
        
        conn.close()
        return jsonify(datasets)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/phenotype_ground_truth')
def get_phenotype_ground_truth():
    """Get ground truth data for phenotype analysis"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT binomial_name, gram_staining, motility, aerophilicity,
                   extreme_environment_tolerance, biofilm_formation,
                   animal_pathogenicity, biosafety_level, health_association,
                   host_association, plant_pathogenicity, spore_formation,
                   hemolysis, cell_shape
            FROM ground_truth
        """)
        
        data = []
        for row in cursor.fetchall():
            data.append({
                'binomial_name': row[0],
                'gram_staining': row[1],
                'motility': row[2],
                'aerophilicity': row[3],
                'extreme_environment_tolerance': row[4],
                'biofilm_formation': row[5],
                'animal_pathogenicity': row[6],
                'biosafety_level': row[7],
                'health_association': row[8],
                'host_association': row[9],
                'plant_pathogenicity': row[10],
                'spore_formation': row[11],
                'hemolysis': row[12],
                'cell_shape': row[13]
            })
        
        conn.close()
        return jsonify({'data': data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_csv')
def export_csv_api():
    """Export results as CSV"""
    species_file = request.args.get('species_file')
    model = request.args.get('model')
    
    try:
        csv_content = processing_manager.export_results_to_csv(species_file, model)
        
        # Generate filename
        filename_parts = ['microbebench_results']
        if species_file:
            filename_parts.append(os.path.basename(species_file).replace('.txt', ''))
        if model:
            filename_parts.append(model.split('/')[-1])
        filename = '_'.join(filename_parts) + '.csv'
        
        # Return CSV as download
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/import_csv', methods=['POST'])
def import_csv_api():
    """Import results from CSV file"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be a CSV file'}), 400
        
        # Read CSV content
        content = file.read().decode('utf-8')
        
        # Process the import
        import_results = processing_manager.import_results_from_csv(content)
        
        return jsonify({
            'success': True,
            'results': import_results
        })
        
    except UnicodeDecodeError:
        return jsonify({'error': 'File encoding error. Please ensure the file is UTF-8 encoded.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_combination', methods=['POST'])
def create_combination_api():
    """API endpoint to create and start a single combination"""
    try:
        data = request.get_json()
        species_file = data.get('species_file')
        model = data.get('model')
        system_template = data.get('system_template')
        user_template = data.get('user_template')
        logger.debug(
            "create_combination_api payload: species_file=%s, model=%s, system_template=%s, user_template=%s",
            species_file,
            model,
            system_template,
            user_template
        )
        
        if not all([species_file, model, system_template, user_template]):
            logger.error("Missing required parameters when creating combination")
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Create the combination using existing logic
        template_pairs = {
            'custom': {
                'system': system_template,
                'user': user_template
            }
        }
        
        logger.debug("Creating combinations")
        created_combinations = processing_manager.create_combinations(species_file, [model], template_pairs)
        logger.debug("Created combinations: %s", created_combinations)
        
        if created_combinations:
            # Get the created combination ID
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM combinations 
                WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
            ''', (species_file, model, system_template, user_template))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                combination_id = result[0]
                logger.debug("Found combination ID: %s", combination_id)
                
                # Start the combination immediately
                logger.debug("Attempting to start combination %s", combination_id)
                start_result = processing_manager.start_combination(combination_id)
                logger.debug("Start result for combination %s: %s", combination_id, start_result)
                
                if start_result:
                    logger.info("Combination %s started successfully via API", combination_id)
                    return jsonify({'success': True, 'combination_id': combination_id})
                else:
                    logger.warning("Combination %s created but could not start", combination_id)
                    return jsonify({'success': True, 'combination_id': combination_id, 'note': 'Created but could not start immediately'})
            else:
                logger.error("Combination created but could not retrieve ID")
                return jsonify({'error': 'Combination created but could not retrieve ID'}), 500
        else:
            logger.error("Failed to create combination - may already exist")
            return jsonify({'error': 'Failed to create combination - may already exist'}), 400
            
    except Exception as e:
        logger.exception("Unexpected error creating combination")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_model', methods=['POST'])
def add_model_api():
    try:
        data = request.get_json()
        model = data.get('model')
        if not model:
            return jsonify({'error': 'Model name is required'}), 400
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO managed_models (model) VALUES (?)", (model,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'Model {model} added to dashboard view.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_model', methods=['POST'])
def delete_model_api():
    try:
        data = request.get_json()
        model = data.get('model')
        if not model:
            return jsonify({'error': 'Model name is required'}), 400
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM managed_models WHERE model = ?", (model,))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': f'Model {model} removed from dashboard view.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_species_file', methods=['POST'])
def add_species_file_api():
    try:
        data = request.get_json()
        species_file = data.get('species_file')
        if not species_file:
            return jsonify({'error': 'Species file is required'}), 400
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO managed_species_files (species_file) VALUES (?)", (species_file,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'Species file {species_file} added to dashboard view.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_species_file', methods=['POST'])
def delete_species_file_api():
    try:
        data = request.get_json()
        species_file = data.get('species_file')
        if not species_file:
            return jsonify({'error': 'Species file is required'}), 400
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM managed_species_files WHERE species_file = ?", (species_file,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'Species file {species_file} removed from dashboard view.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/combination_details/<int:combination_id>')
def get_combination_details(combination_id):
    """Get detailed results for a specific combination"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get combination info with progress
        cursor.execute('''
            SELECT species_file, model, system_template, user_template, status, 
                   total_species, submitted_species, received_species, successful_species, failed_species, timeout_species
            FROM combinations WHERE id = ?
        ''', (combination_id,))
        
        combo_info = cursor.fetchone()
        if not combo_info:
            conn.close()
            return jsonify({'error': 'Combination not found'}), 404
        
        combination_data = {
            'id': combination_id,
            'species_file': combo_info[0],
            'model': combo_info[1],
            'system_template': combo_info[2].split('/')[-1],
            'user_template': combo_info[3].split('/')[-1],
            'status': combo_info[4],
            'total_species': combo_info[5],
            'submitted_species': combo_info[6] or 0,
            'received_species': combo_info[7] or 0,
            'successful_species': combo_info[8] or 0,
            'failed_species': combo_info[9] or 0,
            'timeout_species': combo_info[10] or 0
        }
        
        # Get detailed results for each species
        cursor.execute("""
            SELECT binomial_name, status, result, error, timestamp,
                   gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                   biofilm_formation, animal_pathogenicity, biosafety_level, health_association,
                   host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape,
                   knowledge_group
            FROM results 
            WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
            ORDER BY timestamp
        """, (combo_info[0], combo_info[1], combo_info[2], combo_info[3]))
        
        # Detect if this is a knowledge template
        is_knowledge_template = detect_template_type(combo_info[3]) == 'knowledge'
        
        species_results = []
        for row in cursor.fetchall():
            result_data = {
                'binomial_name': row[0],
                'status': row[1],
                'result': row[2],
                'error': row[3],
                'timestamp': row[4],
                'is_knowledge_template': is_knowledge_template
            }
            
            if is_knowledge_template:
                # For knowledge templates, include knowledge_group
                result_data['knowledge_group'] = row[18]  # knowledge_group is the last column
            else:
                # For phenotype templates, include phenotypes
                result_data['phenotypes'] = {
                    'gram_staining': row[5],
                    'motility': row[6],
                    'aerophilicity': row[7],
                    'extreme_environment_tolerance': row[8],
                    'biofilm_formation': row[9],
                    'animal_pathogenicity': row[10],
                    'biosafety_level': row[11],
                    'health_association': row[12],
                    'host_association': row[13],
                    'plant_pathogenicity': row[14],
                    'spore_formation': row[15],
                    'hemolysis': row[16],
                    'cell_shape': row[17]
                }
            
            species_results.append(result_data)
        
        # Get list of all species that should have been processed
        species_file_path = os.path.join(config.SPECIES_DIR, combo_info[0])
        try:
            with open(species_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # Import the header filtering function
                from microbellm.utils import filter_species_list
                all_species = filter_species_list(lines)
        except:
            all_species = []
        
        # Find species that weren't processed yet
        processed_species = {result['binomial_name'] for result in species_results}
        unprocessed_species = [species for species in all_species if species not in processed_species]
        
        conn.close()
        
        return jsonify({
            'combination_info': combination_data,
            'species_results': species_results,
            'unprocessed_species': unprocessed_species,
            'total_species_in_file': len(all_species)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/api_key_status')
def api_key_status():
    """Check if OpenRouter API key is configured"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        return jsonify({
            'status': 'missing',
            'message': 'OPENROUTER_API_KEY environment variable is not set',
            'configured': False
        })
    elif len(api_key.strip()) == 0:
        return jsonify({
            'status': 'empty',
            'message': 'OPENROUTER_API_KEY is set but empty',
            'configured': False
        })
    elif len(api_key) < 10:  # Basic sanity check
        return jsonify({
            'status': 'invalid',
            'message': 'OPENROUTER_API_KEY appears to be invalid (too short)',
            'configured': False
        })
    else:
        # Mask the key for security - show first 4 and last 4 characters
        masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
        return jsonify({
            'status': 'configured',
            'message': f'API key configured: {masked_key}',
            'configured': True
        })

@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    """Set the OpenRouter API key"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'success': False, 'error': 'API key is required'}), 400
        
        if len(api_key) < 10:
            return jsonify({'success': False, 'error': 'API key appears to be too short'}), 400
        
        # For security and simplicity, we'll write the API key to a local .env file
        # and provide instructions for the user to restart the server
        env_file_path = os.path.join(os.getcwd(), '.env')
        
        # Read existing .env file if it exists
        env_lines = []
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                env_lines = f.readlines()
        
        # Remove any existing OPENROUTER_API_KEY lines
        env_lines = [line for line in env_lines if not line.strip().startswith('OPENROUTER_API_KEY=')]
        
        # Add the new API key
        env_lines.append(f'OPENROUTER_API_KEY={api_key}\n')
        
        # Write back to .env file
        with open(env_file_path, 'w') as f:
            f.writelines(env_lines)
        
        # Also set it in the current environment (though this requires a restart to be permanent)
        os.environ['OPENROUTER_API_KEY'] = api_key
        
        return jsonify({
            'success': True, 
            'message': 'API key updated successfully',
            'restart_required': True
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_openrouter_models')
def get_openrouter_models():
    """Fetch all available models from OpenRouter API"""
    try:
        api_key = os.getenv('OPENROUTER_API_KEY')
        
        # Make request to OpenRouter models endpoint
        import requests
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            models_data = response.json()["data"]
            
            # Extract and sort model information
            models = []
            for model in models_data:
                models.append({
                    'id': model['id'],
                    'name': model.get('name', model['id']),
                    'description': model.get('description', ''),
                    'context_length': model.get('context_length', 0),
                    'pricing': model.get('pricing', {}),
                    'top_provider': model.get('top_provider', {})
                })
            
            # Sort by popularity/name for better UX
            models.sort(key=lambda x: (
                x['id'].startswith('openai/') or x['id'].startswith('anthropic/'),
                x['name'].lower()
            ), reverse=True)
            
            return jsonify({
                'success': True,
                'models': models,
                'count': len(models)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'OpenRouter API error (status {response.status_code})',
                'models': []
            })
            
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'OpenRouter API timeout',
            'models': []
        })
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Network error: {str(e)}',
            'models': []
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error fetching models: {str(e)}',
            'models': []
        })

@app.route('/api/validate_model', methods=['POST'])
def validate_model_api():
    """Validate if a model is supported by OpenRouter (for manual text input)"""
    try:
        data = request.get_json()
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({'valid': False, 'error': 'Model name is required'})
        
        # Clean the model name
        model_name = model_name.strip()
        
        # First try to fetch from the models list API (faster)
        try:
            models_response = get_openrouter_models()
            models_data = models_response.get_json()
            
            if models_data.get('success'):
                model_ids = [model['id'] for model in models_data['models']]
                if model_name in model_ids:
                    return jsonify({'valid': True, 'message': f'Model "{model_name}" is supported by OpenRouter'})
                else:
                    return jsonify({'valid': False, 'error': f'Model "{model_name}" not found in OpenRouter catalog'})
        except:
            pass  # Fall back to test request method
        
        # Fallback: Check with OpenRouter API using test request
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            return jsonify({'valid': False, 'error': 'OPENROUTER_API_KEY not configured'})
        
        import requests
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use a simple test prompt to check if model exists
        test_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
            "temperature": 0
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify({'valid': True, 'message': f'Model "{model_name}" is supported by OpenRouter'})
        elif response.status_code == 400:
            # Check if it's a model-not-found error
            try:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', '')
                if 'model' in error_message.lower() and ('not found' in error_message.lower() or 'invalid' in error_message.lower()):
                    return jsonify({'valid': False, 'error': f'Model "{model_name}" not found on OpenRouter'})
                else:
                    # Other validation error, but model might exist
                    return jsonify({'valid': True, 'message': f'Model "{model_name}" exists (validation error: {error_message})'})
            except:
                return jsonify({'valid': False, 'error': f'Model "{model_name}" validation failed'})
        elif response.status_code == 401:
            return jsonify({'valid': False, 'error': 'Invalid API key'})
        else:
            return jsonify({'valid': False, 'error': f'OpenRouter API error (status {response.status_code})'})
            
    except requests.exceptions.Timeout:
        return jsonify({'valid': False, 'error': 'OpenRouter API timeout - please try again'})
    except requests.exceptions.RequestException as e:
        return jsonify({'valid': False, 'error': f'Network error: {str(e)}'})
    except Exception as e:
        return jsonify({'valid': False, 'error': f'Validation error: {str(e)}'})

@app.route('/view_template/<template_type>/<template_name>')
def view_template(template_type, template_name):
    """View template content in a read-only format"""
    try:
        if template_type not in ['system', 'user']:
            return "Invalid template type", 400
        
        if template_type == 'system':
            template_dir = Path(config.SYSTEM_TEMPLATES_DIR)
        else:
            template_dir = Path(config.USER_TEMPLATES_DIR)
        
        template_file = template_dir / f"{template_name}.txt"
        
        if not template_file.exists():
            return "Template not found", 404
        
        with open(template_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return render_template('view_template.html', 
                               template_name=template_name,
                               template_type=template_type,
                               content=content)
    
    except Exception as e:
        return f"Error reading template: {str(e)}", 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.debug('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.debug('Client disconnected')

def main():
    """Main entry point for the web application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MicrobeBench Web Interface")
    parser.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'), help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load .env file if it exists
    env_file_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_file_path):
        logger.info("Loading environment variables from .env file...")
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value
    
    # Check environment
    if not os.getenv('OPENROUTER_API_KEY'):
        logger.warning("OPENROUTER_API_KEY not set. Set it with environment variable or .env entry OPENROUTER_API_KEY=your-api-key")
    
    # Initialize the processing manager after loading environment
    global processing_manager
    reset_running_jobs_on_startup()
    processing_manager = ProcessingManager()
    
    logger.info("Starting MicrobeBench Web Interface on http://%s:%s", args.host, args.port)
    socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main()

# Global variable for processing manager - initialized in main()
processing_manager = None 

@app.route('/api/rerun_failed_species', methods=['POST'])
def rerun_failed_species_api():
    """Re-run a single failed species"""
    try:
        data = request.get_json()
        combination_id = data.get('combination_id')
        species_name = data.get('species_name')
        
        if not all([combination_id, species_name]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Get combination details
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT species_file, model, system_template, user_template
            FROM combinations WHERE id = ?
        """, (combination_id,))
        
        combo_info = cursor.fetchone()
        if not combo_info:
            conn.close()
            return jsonify({'error': 'Combination not found'}), 404
        
        species_file, model, system_template, user_template = combo_info
        
        # Check if this species has a failed result
        cursor.execute("""
            SELECT id FROM results 
            WHERE species_file = ? AND binomial_name = ? AND model = ? 
            AND system_template = ? AND user_template = ? AND status = 'failed'
        """, (species_file, species_name, model, system_template, user_template))
        
        failed_result = cursor.fetchone()
        if not failed_result:
            conn.close()
            return jsonify({'error': 'No failed result found for this species'}), 404
        
        # Delete the failed result
        cursor.execute("""
            DELETE FROM results 
            WHERE species_file = ? AND binomial_name = ? AND model = ? 
            AND system_template = ? AND user_template = ? AND status = 'failed'
        """, (species_file, species_name, model, system_template, user_template))
        
        conn.commit()
        conn.close()
        
        # Process this single species
        system_template_content = read_template_from_file(system_template)
        user_template_content = read_template_from_file(user_template)
        
        # Run in a separate thread
        thread = threading.Thread(
            target=processing_manager._rerun_single_species,
            args=(combination_id, species_file, species_name, model, 
                  system_template, user_template, system_template_content, user_template_content)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': f'Re-running {species_name}...'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reparse_phenotype_data/<int:combination_id>', methods=['POST'])
def reparse_phenotype_data_api(combination_id):
    """Re-parse phenotype data from raw responses for a combination"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get combination details
        cursor.execute("""
            SELECT species_file, model, system_template, user_template 
            FROM combinations WHERE id = ?
        """, (combination_id,))
        
        combo = cursor.fetchone()
        if not combo:
            conn.close()
            return jsonify({'error': 'Combination not found'}), 404
        
        species_file, model, system_template, user_template = combo
        
        # Check if this is a phenotype template
        if detect_template_type(user_template) != 'phenotype':
            conn.close()
            return jsonify({'error': 'This is not a phenotype template combination'}), 400
        
        # Get all results with raw responses for this combination
        cursor.execute("""
            SELECT id, binomial_name, result
            FROM results 
            WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
                  AND result IS NOT NULL AND result != ''
                  AND status = 'completed'
        """, (species_file, model, system_template, user_template))
        
        results = cursor.fetchall()
        
        if not results:
            conn.close()
            return jsonify({'success': False, 'message': 'No results with raw data found to re-parse'}), 404
        
        # Re-parse each result
        updated_count = 0
        failed_count = 0
        invalid_count = 0
        
        from microbellm.utils import parse_response
        
        for result_id, binomial_name, raw_response in results:
            try:
                # Parse the raw response using template validation
                parsed_result = parse_response(raw_response, user_template)
                
                if parsed_result:
                    # Check if there were any invalid fields
                    if 'invalid_fields' in parsed_result:
                        invalid_count += 1
                        logger.warning("%s has invalid fields: %s", binomial_name, parsed_result['invalid_fields'])
                    
                    # Handle aerophilicity as array - convert to string for database storage
                    aerophilicity = parsed_result.get('aerophilicity')
                    if isinstance(aerophilicity, list):
                        aerophilicity_str = str(aerophilicity)
                    else:
                        aerophilicity_str = aerophilicity
                    
                    # Update the phenotype fields in the database
                    cursor.execute("""
                        UPDATE results 
                        SET gram_staining = ?,
                            motility = ?,
                            aerophilicity = ?,
                            extreme_environment_tolerance = ?,
                            biofilm_formation = ?,
                            animal_pathogenicity = ?,
                            biosafety_level = ?,
                            health_association = ?,
                            host_association = ?,
                            plant_pathogenicity = ?,
                            spore_formation = ?,
                            hemolysis = ?,
                            cell_shape = ?
                        WHERE id = ?
                    """, (
                        parsed_result.get('gram_staining'),
                        parsed_result.get('motility'),
                        aerophilicity_str,
                        parsed_result.get('extreme_environment_tolerance'),
                        parsed_result.get('biofilm_formation'),
                        parsed_result.get('animal_pathogenicity'),
                        parsed_result.get('biosafety_level'),
                        parsed_result.get('health_association'),
                        parsed_result.get('host_association'),
                        parsed_result.get('plant_pathogenicity'),
                        parsed_result.get('spore_formation'),
                        parsed_result.get('hemolysis'),
                        parsed_result.get('cell_shape'),
                        result_id
                    ))
                    updated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.exception("Error re-parsing result for %s", binomial_name)
                failed_count += 1
        
        conn.commit()
        conn.close()
        
        # Invalidate caches
        _invalidate_all_caches()
        
        return jsonify({
            'success': True,
            'message': f'Re-parsed {updated_count} results successfully, {failed_count} failed, {invalid_count} had validation issues',
            'updated': updated_count,
            'failed': failed_count,
            'invalid': invalid_count,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rerun_all_failed/<int:combination_id>', methods=['POST'])
def rerun_all_failed_api(combination_id):
    """Re-run all failed species for a combination"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get combination details
        cursor.execute("""
            SELECT species_file, model, system_template, user_template, status
            FROM combinations WHERE id = ?
        """, (combination_id,))
        
        combo_info = cursor.fetchone()
        if not combo_info:
            conn.close()
            return jsonify({'error': 'Combination not found'}), 404
        
        species_file, model, system_template, user_template, status = combo_info
        
        # Get all failed species
        cursor.execute("""
            SELECT binomial_name FROM results 
            WHERE species_file = ? AND model = ? AND system_template = ? 
            AND user_template = ? AND status = 'failed'
        """, (species_file, model, system_template, user_template))
        
        failed_species = [row[0] for row in cursor.fetchall()]
        
        if not failed_species:
            conn.close()
            return jsonify({'message': 'No failed species to re-run'}), 200
        
        # Delete all failed results
        cursor.execute("""
            DELETE FROM results 
            WHERE species_file = ? AND model = ? AND system_template = ? 
            AND user_template = ? AND status = 'failed'
        """, (species_file, model, system_template, user_template))
        
        # Update combination statistics
        cursor.execute("""
            UPDATE combinations 
            SET failed_species = 0, status = 'pending'
            WHERE id = ?
        """, (combination_id,))
        
        conn.commit()
        conn.close()
        
        # Process the failed species
        processing_manager._rerun_failed_species(combination_id, failed_species)
        
        return jsonify({
            'success': True, 
            'message': f'Re-running {len(failed_species)} species...',
            'failed_count': len(failed_species)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ground Truth Data Routes
@app.route('/ground_truth')
def ground_truth_viewer():
    """Ground truth data viewer page"""
    return render_template('ground_truth.html')

@app.route('/api/ground_truth/datasets', methods=['GET'])
def api_get_ground_truth_datasets():
    """Get list of ground truth datasets"""
    try:
        datasets = get_ground_truth_datasets()
        return jsonify({
            'success': True,
            'datasets': datasets
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ground_truth/data', methods=['GET'])
def api_get_ground_truth_data():
    """Get ground truth data with pagination and filtering"""
    try:
        dataset_name = request.args.get('dataset')
        search_term = request.args.get('search')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        
        offset = (page - 1) * per_page
        
        # Get total count
        all_data = get_ground_truth_data(dataset_name=dataset_name, binomial_name=search_term)
        total_count = len(all_data)
        
        # Get paginated data
        data = get_ground_truth_data(
            dataset_name=dataset_name,
            binomial_name=search_term,
            limit=per_page,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'pages': (total_count + per_page - 1) // per_page
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ground_truth/import', methods=['POST'])
def api_import_ground_truth():
    """Import ground truth data from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        dataset_name = request.form.get('dataset_name')
        template_name = request.form.get('template_name')
        description = request.form.get('description')
        source = request.form.get('source')
        
        if not dataset_name or not template_name:
            return jsonify({'success': False, 'error': 'Dataset name and template are required'})
        
        # Save uploaded file temporarily
        temp_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        # Import the data
        result = import_ground_truth_csv(
            temp_path,
            dataset_name,
            template_name,
            description,
            source
        )

        # Clean up temp file
        os.remove(temp_path)

        if result.get('success'):
            _invalidate_ground_truth_stats_cache(dataset_name)
            _invalidate_model_accuracy_cache(dataset_name)
            _invalidate_knowledge_accuracy_cache(dataset_name)
            _invalidate_performance_year_cache(dataset_name)

        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ground_truth/species/<binomial_name>', methods=['GET'])
def api_get_species_ground_truth(binomial_name):
    """Get ground truth data for a specific species"""
    try:
        data = get_ground_truth_data(binomial_name=binomial_name)
        
        return jsonify({
            'success': True,
            'data': data[0] if data else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_accuracy')
def model_accuracy():
    return render_template('model_accuracy.html')

@app.route('/accuracy_by_knowledge')
def accuracy_by_knowledge():
    return render_template('accuracy_by_knowledge.html')

def get_all_predictions():
    """Helper function to get all predictions from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM results WHERE status = "completed"')
    rows = cursor.fetchall()
    conn.close()

    predictions = {}
    for row in rows:
        species_name = row['binomial_name'].lower()
        if species_name not in predictions:
            predictions[species_name] = []
        predictions[species_name].append(dict(row))
    return predictions

@app.route('/model_accuracy/details')
def model_accuracy_details():
    """Display detailed comparison for a model's phenotype predictions."""
    dataset_name = request.args.get('dataset')
    model_name = request.args.get('model')
    phenotype = request.args.get('phenotype')

    if not all([dataset_name, model_name, phenotype]):
        return "Error: Missing required parameters (dataset, model, phenotype)", 400

    # Load ground truth data
    gt_data_list = get_ground_truth_data(dataset_name)
    ground_truth_map = {item['binomial_name'].lower(): item for item in gt_data_list}
    
    # Load all predictions
    all_predictions = get_all_predictions()
    
    # Filter predictions for the specific model
    model_predictions = {}
    for species, preds in all_predictions.items():
        for pred in preds:
            if pred['model'] == model_name:
                model_predictions[species.lower()] = pred
                break

    # Categorize species
    results = {
        'correct': [],
        'incorrect': [],
        'missing_prediction': [],
        'missing_ground_truth': []
    }
    
    all_species = set(ground_truth_map.keys()) | set(model_predictions.keys())

    for species in sorted(list(all_species)):
        gt = ground_truth_map.get(species)
        pred = model_predictions.get(species)

        gt_value = normalize_value(gt.get(phenotype)) if gt else 'NA'
        pred_value = normalize_value(pred.get(phenotype)) if pred else 'NA'

        if gt_value != 'NA':
            if pred_value != 'NA':
                if gt_value.lower() == pred_value.lower():
                    results['correct'].append((gt['binomial_name'], gt_value))
                else:
                    results['incorrect'].append((gt['binomial_name'], pred_value, gt_value))
            else:
                results['missing_prediction'].append((gt['binomial_name'], gt_value))
        elif pred_value != 'NA':
            results['missing_ground_truth'].append((pred['binomial_name'], pred_value))

    summary = {key: len(value) for key, value in results.items()}

    return render_template('compare_details.html',
                           dataset_name=dataset_name,
                           model_name=model_name,
                           phenotype=phenotype,
                           data=results,
                           summary=summary)

@app.route('/ground_truth_alternative')
def ground_truth_alternative():
    return render_template('ground_truth_alternative.html')

@app.route('/api/model_accuracy/calculate', methods=['POST'])
def api_calculate_model_accuracy():
    """Calculate and return model accuracy metrics."""
    try:
        predictions_file = request.json.get('predictions_file')
        dataset_name = request.json.get('dataset_name')
        template_name = request.json.get('template_name')
        
        if not all([predictions_file, dataset_name, template_name]):
            return jsonify({'success': False, 'error': 'Missing required parameters'})
        
        # Calculate accuracy metrics
        metrics = calculate_model_accuracy(predictions_file, dataset_name, template_name)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/available_template_runs')
def api_available_template_runs():
    """Get available template runs from the results table."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get unique combinations of user_template, system_template with counts
        cursor.execute('''
            SELECT user_template, system_template, 
                   COUNT(DISTINCT model) as model_count,
                   COUNT(DISTINCT binomial_name) as species_count,
                   COUNT(*) as total_results
            FROM results 
            WHERE status = "completed"
            GROUP BY user_template, system_template
            ORDER BY user_template, system_template
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        all_templates = []
        
        for row in results:
            user_template, system_template, model_count, species_count, total_results = row
            
            # Extract template name for display
            template_name = user_template.split('/')[-1].replace('.txt', '') if '/' in user_template else user_template
            
            template_info = {
                'key': f"{system_template}|{user_template}",  # Unique key for backend lookup
                'display_name': f"{template_name} ({model_count} models, {species_count} species)",
                'template_name': template_name,
                'user_template': user_template,
                'system_template': system_template,
                'models': model_count,
                'species_count': species_count,
                'total_results': total_results
            }
            
            all_templates.append(template_info)
        
        return jsonify({
            'success': True,
            'all_templates': all_templates
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/accuracy_by_knowledge', methods=['POST'])
def api_accuracy_by_knowledge():
    """Calculate accuracy metrics stratified by knowledge group."""
    try:
        dataset_name = request.json.get('dataset_name')
        phenotype_template_key = request.json.get('phenotype_template_key')
        knowledge_template_key = request.json.get('knowledge_template_key')
        
        if not all([dataset_name, phenotype_template_key, knowledge_template_key]):
            return jsonify({'success': False, 'error': 'Missing required parameters'})
        
        # Parse template keys
        phenotype_system_template, phenotype_user_template = phenotype_template_key.split('|')
        knowledge_system_template, knowledge_user_template = knowledge_template_key.split('|')
        
        # Get ground truth data
        ground_truth_data = get_ground_truth_data(dataset_name=dataset_name)
        ground_truth_map = {item['binomial_name'].lower(): item for item in ground_truth_data}
        
        # Get prediction data from the selected phenotype template run
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM results 
            WHERE status = "completed" 
            AND system_template = ? 
            AND user_template = ?
        ''', (phenotype_system_template, phenotype_user_template))
        
        phenotype_rows = cursor.fetchall()
        
        # Get knowledge data from the selected knowledge template run
        cursor.execute('''
            SELECT model, binomial_name, knowledge_group FROM results 
            WHERE status = "completed" 
            AND system_template = ? 
            AND user_template = ?
            AND knowledge_group IS NOT NULL
        ''', (knowledge_system_template, knowledge_user_template))
        
        knowledge_rows = cursor.fetchall()
        conn.close()
        
        # Organize predictions by model
        predictions_by_model = {}
        for row in phenotype_rows:
            model = row['model']
            species = row['binomial_name'].lower()
            if model not in predictions_by_model:
                predictions_by_model[model] = {}
            predictions_by_model[model][species] = dict(row)
        
        # Organize knowledge data by model
        knowledge_by_model = {}
        for row in knowledge_rows:
            model = row['model']
            species = row['binomial_name'].lower()
            knowledge_group = row['knowledge_group']
            
            if model not in knowledge_by_model:
                knowledge_by_model[model] = {}
            knowledge_by_model[model][species] = knowledge_group
        
        # Get template name for field definitions
        phenotype_template_name = phenotype_user_template.split('/')[-1].replace('.txt', '') if '/' in phenotype_user_template else phenotype_user_template
        
        # Calculate metrics stratified by knowledge group
        metrics = calculate_accuracy_by_knowledge_groups(
            predictions_by_model, 
            ground_truth_map, 
            knowledge_by_model,
            phenotype_template_name
        )
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

def calculate_accuracy_by_knowledge_groups(predictions_by_model, ground_truth_map, knowledge_by_model, template_name):
    """Calculate accuracy metrics grouped by knowledge level."""
    # Get field definitions for the template
    try:
        from microbellm.template_config import find_validation_config_for_template, TemplateValidator
        
        # Construct the user template path
        user_template_path = f"templates/user/{template_name}.txt"
        config_path = find_validation_config_for_template(user_template_path)
        
        if config_path:
            validator = TemplateValidator(config_path)
            if validator.config:
                field_definitions = validator.config.get('field_definitions', {})
                phenotype_fields = [field for field, defn in field_definitions.items() if defn.get('type') != 'array']
            else:
                raise Exception("Could not load validator config")
        else:
            raise Exception("No config found")
    except:
        # Fallback to common phenotype fields
        phenotype_fields = [
            'gram_staining', 'motility', 'aerophilicity', 'extreme_environment_tolerance',
            'biofilm_formation', 'animal_pathogenicity', 'biosafety_level',
            'health_association', 'host_association', 'plant_pathogenicity',
            'spore_formation', 'hemolysis', 'cell_shape'
        ]
    
    model_metrics = {}
    knowledge_groups = ['limited', 'moderate', 'extensive', 'overall']
    
    for model, model_predictions in predictions_by_model.items():
        if model not in knowledge_by_model:
            continue
            
        model_metrics[model] = {}
        
        # Initialize metrics for each knowledge group and phenotype
        for group in knowledge_groups:
            model_metrics[model][group] = {}
            for field in phenotype_fields:
                model_metrics[model][group][field] = {
                    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                    'correct': 0, 'incorrect': 0, 'missing': 0, 'total': 0
                }
        
        # Process each species prediction
        for species, prediction in model_predictions.items():
            if species not in ground_truth_map or species not in knowledge_by_model[model]:
                continue
                
            ground_truth = ground_truth_map[species]
            knowledge_group = knowledge_by_model[model][species].lower()
            
            if knowledge_group not in ['limited', 'moderate', 'extensive']:
                continue
            
            # Compare each phenotype field
            for field in phenotype_fields:
                if field not in prediction or field not in ground_truth:
                    continue
                    
                pred_value = normalize_value(prediction[field])
                truth_value = normalize_value(ground_truth[field])
                
                # Update metrics for specific knowledge group and overall
                for group in [knowledge_group, 'overall']:
                    metrics = model_metrics[model][group][field]
                    metrics['total'] += 1
                    
                    if truth_value == 'NA' or truth_value == '':
                        continue  # Skip ground truth NA values
                    elif pred_value == 'NA' or pred_value == '':
                        metrics['missing'] += 1
                        metrics['fn'] += 1
                    elif pred_value == truth_value:
                        metrics['correct'] += 1
                        metrics['tp'] += 1
                    else:
                        metrics['incorrect'] += 1
                        metrics['fp'] += 1
                        metrics['fn'] += 1
    
    # Calculate derived metrics
    for model in model_metrics:
        for group in knowledge_groups:
            for field in phenotype_fields:
                metrics = model_metrics[model][group][field]
                
                # Calculate accuracy
                total = metrics['correct'] + metrics['incorrect'] + metrics['missing']
                metrics['accuracy'] = (metrics['correct'] / total * 100) if total > 0 else 0
                
                # Calculate precision, recall, F1
                tp_fp = metrics['tp'] + metrics['fp']
                tp_fn = metrics['tp'] + metrics['fn']
                
                metrics['precision'] = (metrics['tp'] / tp_fp) if tp_fp > 0 else 0
                metrics['recall'] = (metrics['tp'] / tp_fn) if tp_fn > 0 else 0
                
                precision_recall_sum = metrics['precision'] + metrics['recall']
                metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] / precision_recall_sum) if precision_recall_sum > 0 else 0
                
                # Calculate MCC
                tp, tn, fp, fn = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']
                numerator = (tp * tn) - (fp * fn)
                denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
                metrics['mcc'] = numerator / denominator if denominator > 0 else 0
    
    return model_metrics

@app.route('/api/ground_truth/datasets/<dataset_name>', methods=['DELETE'])
def api_delete_ground_truth_dataset(dataset_name):
    """Delete a ground truth dataset"""
    try:
        result = delete_ground_truth_dataset(dataset_name)
        if result.get('success'):
            _invalidate_ground_truth_stats_cache(dataset_name)
            _invalidate_model_accuracy_cache(dataset_name)
            _invalidate_knowledge_accuracy_cache(dataset_name)
            _invalidate_performance_year_cache(dataset_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ground_truth/distribution', methods=['GET'])
def api_get_ground_truth_distribution():
    dataset_name = request.args.get('dataset_name')

    if not dataset_name:
        return jsonify({'success': False, 'error': 'Dataset name is required'}), 400

    try:
        logger.debug("Distribution API called for dataset: %s", dataset_name)
        
        conn = sqlite3.connect(db_path)
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ground_truth'")
        if not cursor.fetchone():
            logger.debug("Ground truth table does not exist")
            conn.close()
            return jsonify({'success': True, 'distribution': {}})

        # Define phenotype columns to analyze
        phenotype_columns = [
            'gram_staining', 'motility', 'aerophilicity', 'extreme_environment_tolerance',
            'biofilm_formation', 'animal_pathogenicity', 'biosafety_level',
            'health_association', 'host_association', 'plant_pathogenicity',
            'spore_formation', 'hemolysis', 'cell_shape'
        ]

        # Use pandas to read data and get value counts
        query = f"SELECT {', '.join(phenotype_columns)} FROM ground_truth WHERE dataset_name = ?"
        logger.debug("Executing distribution query: %s with params: %s", query, dataset_name)
        
        df = pd.read_sql_query(query, conn, params=(dataset_name,))
        logger.debug("Distribution DataFrame shape: %s, empty: %s", df.shape, df.empty)
        
        distribution = {}
        if not df.empty:
            for col in phenotype_columns:
                if col in df.columns:
                    # Fill NA values with a consistent string 'NA'
                    counts = df[col].fillna('NA').value_counts().to_dict()
                    # Convert numpy types to native python types
                    distribution[col] = {str(k): int(v) for k, v in counts.items()}
                    logger.debug("%s: %d unique values", col, len(counts))
            
            logger.debug("Distribution created with %d phenotypes", len(distribution))
        else:
            logger.debug("Distribution DataFrame is empty - no data found for dataset %s", dataset_name)

        conn.close()
        
        return jsonify({'success': True, 'distribution': distribution})

    except Exception as e:
        logger.exception("Error in distribution API for dataset %s", dataset_name)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ground_truth/phenotype_statistics', methods=['GET'])
def api_get_ground_truth_phenotype_statistics():
    """Get detailed phenotype statistics for a dataset"""
    dataset_name = request.args.get('dataset_name')

    if not dataset_name:
        return jsonify({'success': False, 'error': 'Dataset name is required'}), 400

    try:
        result = _calculate_ground_truth_statistics(dataset_name)
        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        logger.exception("Error in phenotype statistics API for dataset %s", dataset_name)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ground_truth/phenotype_statistics_cached', methods=['GET'])
def api_get_ground_truth_phenotype_statistics_cached():
    """Return cached phenotype statistics if available, recomputing when needed."""
    dataset_name = request.args.get('dataset_name')
    refresh_requested = request.args.get('refresh', 'false').lower() in ('1', 'true', 'yes', 'force', 'refresh')

    if not dataset_name:
        return jsonify({'success': False, 'error': 'Dataset name is required'}), 400

    try:
        current_metadata = _get_ground_truth_dataset_import_metadata(dataset_name)

        if not refresh_requested:
            cached_entry = _get_ground_truth_stats_cache_entry(dataset_name)
            if (cached_entry and
                current_metadata['import_timestamp'] <= cached_entry.get('import_timestamp', 0)):
                cached_payload = cached_entry['data'].copy()
                computed_at_ts = cached_entry.get('computed_at', 0)
                computed_iso = datetime.utcfromtimestamp(computed_at_ts).isoformat() + 'Z' if computed_at_ts else None
                cached_payload['success'] = True
                cached_payload['cache_info'] = {
                    'cached': True,
                    'computed_at': computed_iso,
                    'age_seconds': max(0, time.time() - computed_at_ts) if computed_at_ts else None,
                    'source': 'cache'
                }
                return jsonify(cached_payload)

        computed_at = time.time()
        result = _calculate_ground_truth_statistics(dataset_name, metadata=current_metadata)
        _update_ground_truth_stats_cache(
            dataset_name,
            result,
            result['metadata'].get('import_timestamp', 0),
            computed_at=computed_at
        )

        computed_iso = datetime.utcfromtimestamp(computed_at).isoformat() + 'Z'
        payload = result.copy()
        payload['success'] = True
        payload['cache_info'] = {
            'cached': False,
            'computed_at': computed_iso,
            'age_seconds': 0,
            'source': 'fresh'
        }

        return jsonify(payload)

    except Exception as e:
        logger.exception("Error in cached phenotype statistics API for dataset %s", dataset_name)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model_accuracy_cached', methods=['GET'])
def api_get_model_accuracy_cached():
    """Return cached model accuracy metrics for a dataset."""
    dataset_name = request.args.get('dataset_name')
    refresh_requested = request.args.get('refresh', 'false').lower() in ('1', 'true', 'yes', 'force', 'refresh')

    if not dataset_name:
        return jsonify({'success': False, 'error': 'Dataset name is required'}), 400

    try:
        current_metadata = _get_ground_truth_dataset_import_metadata(dataset_name)

        if not refresh_requested:
            cached_entry = _get_model_accuracy_cache_entry(dataset_name)
            if (cached_entry and
                current_metadata['import_timestamp'] <= cached_entry.get('import_timestamp', 0)):
                cached_payload = cached_entry['data'].copy()
                computed_at_ts = cached_entry.get('computed_at', 0)
                computed_iso = datetime.utcfromtimestamp(computed_at_ts).isoformat() + 'Z' if computed_at_ts else None
                cached_payload['success'] = True
                cached_payload['cache_info'] = {
                    'cached': True,
                    'computed_at': computed_iso,
                    'age_seconds': max(0, time.time() - computed_at_ts) if computed_at_ts else None,
                    'source': 'cache'
                }
                return jsonify(cached_payload)

        computed_at = time.time()
        result = _calculate_model_accuracy_metrics(dataset_name, metadata=current_metadata)
        _update_model_accuracy_cache(
            dataset_name,
            result,
            result['metadata'].get('import_timestamp', 0),
            computed_at=computed_at
        )

        computed_iso = datetime.utcfromtimestamp(computed_at).isoformat() + 'Z'
        payload = result.copy()
        payload['success'] = True
        payload['cache_info'] = {
            'cached': False,
            'computed_at': computed_iso,
            'age_seconds': 0,
            'source': 'fresh'
        }

        return jsonify(payload)

    except Exception as e:
        logger.exception("Error in cached model accuracy API for dataset %s", dataset_name)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/knowledge_accuracy_cached', methods=['GET'])
def api_get_knowledge_accuracy_cached():
    """Return cached knowledge-stratified accuracy metrics for a dataset."""
    dataset_name = request.args.get('dataset_name')
    refresh_requested = request.args.get('refresh', 'false').lower() in ('1', 'true', 'yes', 'force', 'refresh')

    if not dataset_name:
        return jsonify({'success': False, 'error': 'Dataset name is required'}), 400

    try:
        current_metadata = _get_ground_truth_dataset_import_metadata(dataset_name)

        if not refresh_requested:
            cached_entry = _get_knowledge_accuracy_cache_entry(dataset_name)
            if (cached_entry and
                current_metadata['import_timestamp'] <= cached_entry.get('import_timestamp', 0)):
                cached_payload = cached_entry['data'].copy()
                computed_at_ts = cached_entry.get('computed_at', 0)
                computed_iso = datetime.utcfromtimestamp(computed_at_ts).isoformat() + 'Z' if computed_at_ts else None
                cached_payload['success'] = True
                cached_payload['cache_info'] = {
                    'cached': True,
                    'computed_at': computed_iso,
                    'age_seconds': max(0, time.time() - computed_at_ts) if computed_at_ts else None,
                    'source': 'cache'
                }
                return jsonify(cached_payload)

        computed_at = time.time()
        result = _calculate_knowledge_accuracy_metrics(dataset_name, metadata=current_metadata)
        _update_knowledge_accuracy_cache(
            dataset_name,
            result,
            result['metadata'].get('import_timestamp', 0),
            computed_at=computed_at
        )

        computed_iso = datetime.utcfromtimestamp(computed_at).isoformat() + 'Z'
        payload = result.copy()
        payload['success'] = True
        payload['cache_info'] = {
            'cached': False,
            'computed_at': computed_iso,
            'age_seconds': 0,
            'source': 'fresh'
        }

        return jsonify(payload)

    except Exception as e:
        logger.exception("Error in cached knowledge accuracy API for dataset %s", dataset_name)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/performance_by_year_cached', methods=['GET'])
def api_get_performance_by_year_cached():
    """Return cached model performance vs publication year data."""
    dataset_name = request.args.get('dataset_name')
    refresh_requested = request.args.get('refresh', 'false').lower() in ('1', 'true', 'yes', 'force', 'refresh')

    if not dataset_name:
        return jsonify({'success': False, 'error': 'Dataset name is required'}), 400

    try:
        current_metadata = _get_ground_truth_dataset_import_metadata(dataset_name)

        if not refresh_requested:
            cached_entry = _get_performance_year_cache_entry(dataset_name)
            if (cached_entry and
                current_metadata['import_timestamp'] <= cached_entry.get('import_timestamp', 0)):
                normalized = json.loads(json.dumps(cached_entry['data'], ensure_ascii=False, default=str))
                computed_at_ts = cached_entry.get('computed_at', 0)
                computed_iso = datetime.utcfromtimestamp(computed_at_ts).isoformat() + 'Z' if computed_at_ts else None
                normalized['success'] = True
                normalized['cache_info'] = {
                    'cached': True,
                    'computed_at': computed_iso,
                    'age_seconds': max(0, time.time() - computed_at_ts) if computed_at_ts else None,
                    'source': 'cache'
                }
                return jsonify(normalized)

        computed_at = time.time()
        result = _calculate_performance_by_year_metrics(dataset_name, metadata=current_metadata)
        _update_performance_year_cache(
            dataset_name,
            result,
            float(result.get('metadata', {}).get('import_timestamp', 0)),
            computed_at=computed_at
        )

        computed_iso = datetime.utcfromtimestamp(computed_at).isoformat() + 'Z'
        payload = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        payload['success'] = True
        payload['cache_info'] = {
            'cached': False,
            'computed_at': computed_iso,
            'age_seconds': 0,
            'source': 'fresh'
        }

        return jsonify(payload)

    except Exception as e:
        logger.exception("Error in cached performance-by-year API for dataset %s", dataset_name)
        return jsonify({'success': False, 'error': str(e)}), 500
