#!/usr/bin/env python
"""
Admin dashboard application for MicrobeLLM
This is for local administration only - not for public deployment
"""

import os
import io
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from markupsafe import escape
from flask_socketio import SocketIO, emit
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sqlite3
import requests

# Import from shared utilities
from microbellm.shared import (
    get_db_connection, reset_running_jobs_on_startup, init_database,
    DATABASE_PATH, JOBS_DB_PATH
)
from microbellm.utils import (
    read_template_from_file, detect_template_type,
    create_ground_truth_tables, import_ground_truth_csv, get_ground_truth_datasets,
    get_ground_truth_data, calculate_model_accuracy, delete_ground_truth_dataset,
    normalize_value
)
from microbellm.predict import predict_binomial_name
from microbellm import config
from microbellm.unified_db import UnifiedDB
from microbellm.unified_db import UnifiedDB

# Create Flask app for admin
app = Flask(__name__, 
    static_folder='static', 
    static_url_path='/static',
    template_folder='templates'
)
app.config['SECRET_KEY'] = 'microbellm-admin-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for job management
processing_manager = None
# Use the jobs database which has all the existing data
db_path = JOBS_DB_PATH

class ProcessingManager:
    def __init__(self):
        self.unified_db = UnifiedDB(db_path)
        self.requests_per_second = 30.0  # Default rate limit
        self.last_request_time = {}
        self.max_concurrent_requests = 10  # Default concurrent requests (optimal range: 5-10)
        self.executor = None
        self.job_queue = []
        self.stopped_combinations = set()
        self.paused_combinations = set()
        self.running_combinations = {}  # Track running combinations
        self.rate_limit_lock = threading.Lock()
        self.request_times = []  # Track request times for rate limiting
        init_database(db_path)
    
    def set_rate_limit(self, requests_per_second):
        """Set the rate limit for API requests"""
        self.requests_per_second = max(0.1, requests_per_second)
    
    def set_max_concurrent_requests(self, max_concurrent):
        """Set the maximum number of concurrent requests"""
        self.max_concurrent_requests = max(1, max_concurrent)
        # Recreate executor with new size
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
    
    def _get_or_create_executor(self):
        """Get or create thread pool executor"""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        return self.executor
    
    def process_queue(self):
        """Process queued jobs if there's capacity"""
        # This is a placeholder - implement if queue processing is needed
        pass

    def run_job(self, job_id):
        """Run a complete job with parallel processing"""
        print(f"\n=== Starting job {job_id} ===")
        
        # Update job status to running
        self.unified_db.update_job_status(job_id, 'running', 'job_started_at')
        
        # Get job details
        job_summary = self.unified_db.get_job_summary(job_id)
        if not job_summary:
            print(f"ERROR: Job {job_id} not found")
            return False
        
        print(f"Processing {job_summary['total']} species with {job_summary['model']}")
        print(f"Using {self.max_concurrent_requests} concurrent workers")
        
        # Get pending species
        pending_species = self.unified_db.get_pending_species(job_id, limit=1000)
        
        # Read templates
        try:
            system_prompt = read_template_from_file(job_summary['system_template'])
            user_prompt = read_template_from_file(job_summary['user_template'])
        except Exception as e:
            print(f"ERROR reading templates: {e}")
            self.unified_db.update_job_status(job_id, 'failed', 'job_completed_at')
            return False
        
        # Process species in parallel
        successful = 0
        failed = 0
        results_lock = threading.Lock()
        
        def process_single_species(species_data):
            """Process a single species"""
            species_name = species_data['binomial_name']
            model = species_data['model']
            
            try:
                print(f"  Processing: {species_name}")
                
                # Update species status to running
                self.unified_db.update_species_result(job_id, species_name, status='running')
                
                # Make prediction
                result = predict_binomial_name(
                    binomial_name=species_name,
                    system_template=system_prompt,
                    user_template=user_prompt,
                    model=model,
                    verbose=False
                )
                
                if result:
                    # Store raw result and parse phenotypes
                    raw_result = json.dumps(result)
                    phenotypes = self._parse_phenotypes(result)
                    
                    self.unified_db.update_species_result(
                        job_id, species_name, 
                        result=raw_result, 
                        status='completed',
                        phenotypes=phenotypes
                    )
                    print(f"  ✓ Completed: {species_name}")
                    return True, None
                else:
                    self.unified_db.update_species_result(
                        job_id, species_name, 
                        status='failed', 
                        error='No result returned'
                    )
                    print(f"  ✗ Failed: {species_name} - No result")
                    return False, 'No result returned'
                    
            except Exception as e:
                self.unified_db.update_species_result(
                    job_id, species_name, 
                    status='failed', 
                    error=str(e)
                )
                print(f"  ✗ Error: {species_name} - {e}")
                return False, str(e)
        
        # Get or create executor
        executor = self._get_or_create_executor()
        
        # Submit all tasks
        print(f"\nSubmitting {len(pending_species)} tasks to executor...")
        futures = []
        for species_data in pending_species:
            future = executor.submit(process_single_species, species_data)
            futures.append(future)
        
        # Process results as they complete
        print(f"Processing results...")
        for future in as_completed(futures):
            try:
                success, error = future.result()
                with results_lock:
                    if success:
                        successful += 1
                    else:
                        failed += 1
                    
                    # Emit progress update
                    if socketio:
                        progress = successful + failed
                        socketio.emit('progress_update', {
                            'combination_id': job_id,
                            'submitted': progress,
                            'total': len(pending_species),
                            'successful': successful,
                            'failed': failed,
                            'timeouts': 0
                        })
            except Exception as e:
                print(f"Future error: {e}")
                with results_lock:
                    failed += 1
        
        # Update job status
        if failed == 0:
            self.unified_db.update_job_status(job_id, 'completed', 'job_completed_at')
        else:
            self.unified_db.update_job_status(job_id, 'completed', 'job_completed_at')  # Still completed, but with some failures
        
        print(f"\nJob completed: {successful} successful, {failed} failed")
        return True
    
    def _parse_phenotypes(self, result):
        """Parse phenotype data from LLM result"""
        phenotypes = {}
        
        if not result:
            return phenotypes
        
        # Extract knowledge_group
        knowledge_group = result.get('knowledge_group') or result.get('knowledge_level')
        if knowledge_group:
            phenotypes['knowledge_group'] = knowledge_group
        
        # Extract phenotypes if present
        if 'phenotypes' in result:
            phenotype_data = result['phenotypes']
        else:
            phenotype_data = result
        
        # Map phenotype fields
        phenotype_fields = [
            'gram_staining', 'motility', 'aerophilicity', 
            'extreme_environment_tolerance', 'biofilm_formation',
            'animal_pathogenicity', 'biosafety_level', 
            'health_association', 'host_association',
            'plant_pathogenicity', 'spore_formation', 
            'hemolysis', 'cell_shape'
        ]
        
        for field in phenotype_fields:
            if field in phenotype_data:
                phenotypes[field] = phenotype_data[field]
        
        return phenotypes
    
    def _apply_rate_limit_old(self, model):
        """Old rate limiting - replaced by better version"""
        pass
    
    def start_job_async(self, job_id):
        """Start a job asynchronously in a background thread"""
        try:
            # Check if job exists
            job_summary = self.unified_db.get_job_summary(job_id)
            if not job_summary:
                print(f"ERROR: Job {job_id} not found")
                return False
                
            if job_summary['job_status'] == 'running':
                print(f"WARNING: Job {job_id} is already running")
                return False
            
            # Start the job in a background thread
            import threading
            thread = threading.Thread(target=self.run_job, args=(job_id,))
            thread.daemon = True
            thread.start()
            
            print(f"Started job {job_id} in background thread")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start job {job_id} async: {e}")
            return False
    
    def create_job(self, species_file, model, system_template, user_template):
        """Create a new job - simplified version"""
        print(f"Creating job: {species_file} + {model} + {system_template} + {user_template}")
        
        # Read species file to get species list
        try:
            # Resolve species file path
            from microbellm.shared import PROJECT_ROOT
            species_path = Path(species_file)
            if not species_path.is_absolute():
                species_dir = PROJECT_ROOT / config.SPECIES_DIR
                full_path = species_dir / species_file
                if full_path.exists():
                    species_file_path = str(full_path)
                    print(f"Resolved species file to: {species_file_path}")
                else:
                    raise FileNotFoundError(f"Species file not found: {species_file}")
            else:
                species_file_path = species_file
            
            # Read CSV file
            df = pd.read_csv(species_file_path)
            
            # Find binomial_name column
            binomial_col = None
            for col in df.columns:
                if col.lower().replace(' ', '_') in ['binomial_name', 'species', 'taxon_name', 'organism', 'species_name']:
                    binomial_col = col
                    break
            
            if not binomial_col:
                raise ValueError("No binomial_name column found in species file")
            
            species_list = df[binomial_col].dropna().tolist()
            print(f"Found {len(species_list)} species in file")
            
            # Create job in unified database
            job_id = self.unified_db.create_job(
                species_file=Path(species_file).name,  # Store basename
                model=model,
                system_template=system_template,
                user_template=user_template,
                species_list=species_list
            )
            
            print(f"Created job {job_id} with {len(species_list)} species")
            return job_id
            
        except Exception as e:
            print(f"Error creating job: {e}")
            raise e
    
    def get_all_jobs(self):
        """Get list of all jobs with summary info"""
        conn = sqlite3.connect(self.unified_db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                job_id,
                species_file,
                model,
                system_template,
                user_template,
                job_status,
                job_created_at,
                job_started_at,
                job_completed_at,
                COUNT(*) as total_species,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running
            FROM processing_results
            GROUP BY job_id
            ORDER BY job_created_at DESC
        """)
        
        jobs = []
        for row in cursor.fetchall():
            job = dict(row)
            jobs.append(job)
        
        conn.close()
        return jobs
    
    def get_job_results(self, job_id):
        """Get detailed results for a specific job"""
        conn = sqlite3.connect(self.unified_db.db_path)
        cursor = conn.cursor()
        
        # Get job summary
        cursor.execute("""
            SELECT 
                job_id,
                species_file,
                model,
                system_template,
                user_template,
                job_status,
                job_created_at,
                job_started_at,
                job_completed_at
            FROM processing_results
            WHERE job_id = ?
            LIMIT 1
        """, (job_id,))
        
        job_info = cursor.fetchone()
        if not job_info:
            return None
        
        # Get all species results
        cursor.execute("""
            SELECT 
                binomial_name,
                status,
                result,
                error,
                created_at,
                started_at,
                completed_at,
                knowledge_group,
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
            FROM processing_results
            WHERE job_id = ?
            ORDER BY binomial_name
        """, (job_id,))
        
        species_results = []
        for row in cursor.fetchall():
            species_result = dict(row)
            # Parse raw result if it exists
            if species_result['result']:
                try:
                    species_result['raw_result'] = json.loads(species_result['result'])
                except:
                    species_result['raw_result'] = species_result['result']
            species_results.append(species_result)
        
        conn.close()
        
        return {
            'job_info': dict(job_info),
            'species_results': species_results
        }
    
    def get_available_models(self):
        """Get list of available models from managed models and OpenRouter"""
        conn = sqlite3.connect(self.unified_db.db_path)
        cursor = conn.cursor()
        
        # Get models from managed_models table
        cursor.execute("SELECT model FROM managed_models ORDER BY model")
        managed_models = [row[0] for row in cursor.fetchall()]
        
        # Get popular OpenRouter models
        openrouter_models = self.get_openrouter_models()
        
        # Combine both lists, removing duplicates
        all_models = list(set(managed_models + openrouter_models))
        
        conn.close()
        return sorted(all_models)
    
    def get_openrouter_models(self):
        """Get list of popular OpenRouter models"""
        # Return a curated list of popular models
        # You can expand this list or make it dynamic by calling OpenRouter API
        popular_models = [
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemini-pro",
            "google/gemini-flash-1.5",
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "cohere/command-r",
            "cohere/command-r-plus",
            "moonshot/v1-8k",
            "moonshot/v1-32k",
            "moonshot/v1-128k"
        ]
        
        # Optional: Try to get live model list from OpenRouter API
        try:
            import requests
            from microbellm import config
            
            if hasattr(config, 'OPENROUTER_API_KEY') and config.OPENROUTER_API_KEY:
                headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    # Extract model IDs from the API response
                    api_models = [model['id'] for model in data.get('data', [])]
                    
                    # Filter to include only the popular ones or top models
                    filtered_models = [model for model in api_models if any(keyword in model.lower() for keyword in ['gpt', 'claude', 'llama', 'gemini', 'mistral', 'moonshot'])]
                    
                    return filtered_models[:20]  # Return top 20 models
                    
        except Exception as e:
            print(f"Could not fetch OpenRouter models: {e}")
            
        return popular_models
    
    def get_available_species_files(self):
        """Get list of available species files"""
        conn = sqlite3.connect(self.unified_db.db_path)
        cursor = conn.cursor()
        
        # Get species files from managed_species_files table
        cursor.execute("SELECT species_file FROM managed_species_files ORDER BY species_file")
        species_files = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return species_files
    
    def get_available_system_templates(self):
        """Get list of available system templates"""
        from microbellm.shared import PROJECT_ROOT
        templates_dir = PROJECT_ROOT / config.TEMPLATES_DIR
        
        if not templates_dir.exists():
            return []
        
        system_templates = []
        for file in templates_dir.glob("**/system_*.txt"):
            system_templates.append(str(file.relative_to(PROJECT_ROOT)))
        
        return sorted(system_templates)
    
    def get_available_user_templates(self):
        """Get list of available user templates"""
        from microbellm.shared import PROJECT_ROOT
        templates_dir = PROJECT_ROOT / config.TEMPLATES_DIR
        
        if not templates_dir.exists():
            return []
        
        user_templates = []
        for file in templates_dir.glob("**/user_*.txt"):
            user_templates.append(str(file.relative_to(PROJECT_ROOT)))
        
        return sorted(user_templates)
    
    def process_job_unified(self, job_id):
        """Process a single job with proper error handling and status updates"""
        try:
            # Update job status to running
            self.unified_db.update_job_status(job_id, 'running', 'job_started_at')
            
            # Get job summary
            job_summary = self.unified_db.get_job_summary(job_id)
            
            if not job_summary:
                print(f"ERROR: Job {job_id} not found in database")
                return
                
            species_file = job_summary['species_file']
            model = job_summary['model']
            system_template = job_summary['system_template']
            user_template = job_summary['user_template']
            
            print(f"Processing job {job_id}:")
            print(f"  Species file: {species_file}")
            print(f"  Model: {model}")
            print(f"  System template: {system_template}")
            print(f"  User template: {user_template}")
            
            # Read templates
            try:
                system_prompt = read_template_from_file(system_template)
                user_prompt = read_template_from_file(user_template)
            except Exception as e:
                print(f"ERROR reading templates: {e}")
                self.unified_db.update_job_status(job_id, 'failed', 'job_completed_at')
                return
            
            # Read species file and templates
            try:
                # Check if species file exists
                species_file_path = species_file  # Keep original for file operations
                if not Path(species_file).exists():
                    # Try to resolve relative to project root
                    from microbellm.shared import PROJECT_ROOT
                    species_path = PROJECT_ROOT / config.SPECIES_DIR / Path(species_file).name
                    if species_path.exists():
                        species_file_path = str(species_path)  # Use full path for file operations
                        print(f"  Resolved species file path to: {species_file_path}")
                    else:
                        raise FileNotFoundError(f"Species file not found: {species_file}")
                
                # Normalize species_file to basename for database storage
                species_file = Path(species_file).name
                
                # Try to read the species file with different separators
                # First, try to detect the separator by reading the first line
                with open(species_file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if '\t' in first_line:
                        sep = '\t'
                        sep_name = 'tab'
                    elif ',' in first_line:
                        sep = ','
                        sep_name = 'comma'
                    else:
                        sep = None
                        sep_name = 'auto'
                
                try:
                    if sep:
                        df = pd.read_csv(species_file_path, sep=sep)
                        print(f"  Read species file with {sep_name} separator")
                    else:
                        # Let pandas auto-detect
                        df = pd.read_csv(species_file_path)
                        print(f"  Read species file with auto-detected separator")
                except Exception as e:
                    raise ValueError(f"Failed to read species file: {e}")
                
                # Check for binomial_name column (case-insensitive)
                print(f"  Columns found in file: {list(df.columns)}")
                
                binomial_col = None
                # Try exact match first
                if 'binomial_name' in df.columns:
                    binomial_col = 'binomial_name'
                else:
                    # Try case-insensitive and space variations
                    for col in df.columns:
                        col_lower = col.lower().replace(' ', '_')
                        if col_lower == 'binomial_name' or col_lower == 'binomial_name':
                            binomial_col = col
                            break
                        elif col_lower in ['species', 'taxon_name', 'organism', 'species_name']:
                            binomial_col = col
                            break
                
                if binomial_col:
                    if binomial_col != 'binomial_name':
                        print(f"  Using column '{binomial_col}' as binomial_name")
                        df['binomial_name'] = df[binomial_col]
                else:
                    raise ValueError(f"Could not find binomial_name column in {species_file}. Found columns: {list(df.columns)}")
                
                species_list = df['binomial_name'].tolist()
                print(f"  Found {len(species_list)} species to process")
                
                system_prompt = read_template_from_file(system_template)
                user_prompt = read_template_from_file(user_template)
                print(f"  Successfully loaded templates")
                
                # Update total species count
                cursor.execute("UPDATE combinations SET total_species = ? WHERE id = ?", 
                             (len(species_list), combination_id))
                conn.commit()
                
            except Exception as e:
                error_msg = f"Error reading files for combination {combination_id}: {str(e)}"
                print(f"ERROR: {error_msg}")
                cursor.execute("UPDATE combinations SET status = 'failed' WHERE id = ?", (combination_id,))
                conn.commit()
                # Emit error to frontend
                socketio.emit('job_error', {
                    'combination_id': combination_id,
                    'error': error_msg
                })
                return
            
            # Process species with rate limiting and concurrent execution
            total_species = len(species_list)
            successful = 0
            failed = 0
            timeouts = 0
            
            # Use ThreadPoolExecutor for concurrent processing
            executor = self._get_or_create_executor()
            futures = {}
            species_index = 0
            
            # Submit initial batch of requests up to max_concurrent
            while species_index < min(len(species_list), self.max_concurrent_requests):
                species = species_list[species_index]
                future = executor.submit(self._process_single_species, 
                                       species, model, system_prompt, user_prompt, 
                                       job_id)
                futures[future] = species
                species_index += 1
                
            print(f"  Submitted initial batch of {len(futures)} species for parallel processing")
            
            # Process results as they complete and submit new ones
            while futures or species_index < len(species_list):
                # Check if stopped
                if job_id in self.stopped_combinations:
                    self.unified_db.update_job_status(job_id, 'interrupted', 'job_completed_at')
                    break
                    
                # Check if paused
                while job_id in self.paused_combinations:
                    time.sleep(1)
                
                # Process completed futures
                completed_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        completed_futures.append(future)
                        
                for future in completed_futures:
                    species_name = futures.pop(future)
                    try:
                        result, status, error = future.result(timeout=1)
                        
                        # Store result in both tables
                        # Parse the result if it's JSON
                        knowledge_group = None
                        phenotype_data = {}
                        
                        if result and status == 'completed':
                            try:
                                result_dict = json.loads(result)
                                
                                # Extract knowledge_group if present
                                knowledge_group = result_dict.get('knowledge_group') or result_dict.get('knowledge_level')
                                
                                # Extract phenotypes if present
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
                                print(f"    Warning: Could not parse JSON result for {species_name}")
                        
                        # Store directly in results table (single source of truth)
                        cursor.execute("""
                            INSERT INTO results (
                                species_file, binomial_name, model, system_template, user_template,
                                status, result, error, knowledge_group,
                                gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                                biofilm_formation, animal_pathogenicity, biosafety_level,
                                health_association, host_association, plant_pathogenicity,
                                spore_formation, hemolysis, cell_shape
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            species_file, species_name, model, system_template, user_template,
                            status, result, error, knowledge_group,
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
                        conn.commit()
                        
                        print(f"    Stored result for {species_name}: status={status}")
                        
                        if status == 'completed':
                            successful += 1
                        elif status == 'timeout':
                            timeouts += 1
                        else:
                            failed += 1
                            
                    except Exception as e:
                        failed += 1
                        print(f"Error processing {species_name}: {e}")
                
                        # Submit a new species if we have more to process
                        if species_index < len(species_list) and len(futures) < self.max_concurrent_requests:
                            species = species_list[species_index]
                            future = executor.submit(self._process_single_species, 
                                                   species, model, system_prompt, user_prompt, 
                                                   job_id)
                            futures[future] = species
                            species_index += 1
                
                # Emit progress update
                submitted = species_index
                socketio.emit('progress_update', {
                    'combination_id': combination_id,
                    'submitted': submitted,
                    'total': total_species,
                    'successful': successful,
                    'failed': failed,
                    'timeouts': timeouts
                })
                
                # Small sleep to prevent CPU spinning
                if futures:
                    time.sleep(0.01)
                                
            # Wait for remaining futures to complete
            for future, species_name in futures.items():
                try:
                    result, status, error = future.result(timeout=60)
                    
                    # Parse the result if it's JSON
                    knowledge_group = None
                    phenotype_data = {}
                    
                    if result and status == 'completed':
                        try:
                            result_dict = json.loads(result)
                            knowledge_group = result_dict.get('knowledge_group') or result_dict.get('knowledge_level')
                            if 'phenotypes' in result_dict:
                                phenotype_data = result_dict['phenotypes']
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
                            pass
                    
                    # Store directly in results table
                    cursor.execute("""
                        INSERT INTO results (
                            species_file, binomial_name, model, system_template, user_template,
                            status, result, error, knowledge_group,
                            gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                            biofilm_formation, animal_pathogenicity, biosafety_level,
                            health_association, host_association, plant_pathogenicity,
                            spore_formation, hemolysis, cell_shape
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        species_file, species_name, model, system_template, user_template,
                        status, result, error, knowledge_group,
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
                    conn.commit()
                    
                    if status == 'completed':
                        successful += 1
                    elif status == 'timeout':
                        timeouts += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    
            # Update combination status and final counts
            final_status = 'completed'
            if combination_id in self.stopped_combinations:
                final_status = 'interrupted'
            
            print(f"\nFinal results for combination {combination_id}:")
            print(f"  Total: {total_species}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Timeouts: {timeouts}")
            
            # Update with final counts
            cursor.execute("""
                UPDATE combinations 
                SET status = ?, 
                    completed_at = ?,
                    successful_species = ?,
                    failed_species = ?,
                    timeout_species = ?,
                    total_species = ?
                WHERE id = ?
            """, (final_status, datetime.now(), successful, failed, timeouts, total_species, combination_id))
            conn.commit()
            
            # Final update
            socketio.emit('job_update', {
                'combination_id': combination_id,
                'status': final_status,
                'successful': successful,
                'failed': failed,
                'timeouts': timeouts,
                'total': total_species
            })
            
        except Exception as e:
            print(f"Error in process_combination: {e}")
            if conn:
                cursor.execute("UPDATE combinations SET status = 'failed' WHERE id = ?", 
                             (combination_id,))
                conn.commit()
        finally:
            if conn:
                conn.close()
            # Remove from running combinations
            self.running_combinations.pop(combination_id, None)
            self.stopped_combinations.discard(combination_id)
            self.paused_combinations.discard(combination_id)
            # Process next in queue
            self.process_queue()
    
    def _process_single_species(self, species, model, system_prompt, user_prompt, combination_id):
        """Process a single species with rate limiting"""
        # Rate limiting
        self._apply_rate_limit(model)
        
        # print(f"  Processing species: {species} with model: {model}")  # Commented for performance
        
        try:
            # Make prediction
            result = predict_binomial_name(
                binomial_name=species,
                system_template=system_prompt,
                user_template=user_prompt,
                model=model,
                verbose=False  # Disable for better performance
            )
            
            if result:
                print(f"    ✓ Success: {species}")
                return json.dumps(result), 'completed', None
            else:
                print(f"    ✗ Failed: {species} - No result returned")
                return None, 'failed', 'No result returned'
                
        except requests.exceptions.Timeout:
            print(f"    ⏱ Timeout: {species}")
            return None, 'timeout', 'Request timed out'
        except Exception as e:
            print(f"    ✗ Error processing {species}: {str(e)}")
            return None, 'failed', str(e)
    
    
    def _apply_rate_limit(self, model):
        """Apply rate limiting using a sliding window approach"""
        # Temporarily disable rate limiting for better performance
        # The API provider (OpenRouter) has its own rate limiting
        return
        
        # Original implementation commented out:
        # if self.requests_per_second <= 0:
        #     return
        # 
        # with self.rate_limit_lock:
        #     current_time = time.time()
        #     
        #     # Remove old request times (older than 1 second)
        #     self.request_times = [t for t in self.request_times if current_time - t < 1.0]
        #     
        #     # Check if we've hit the rate limit
        #     if len(self.request_times) >= self.requests_per_second:
        #         # Calculate how long to wait
        #         oldest_request = self.request_times[0]
        #         sleep_time = 1.0 - (current_time - oldest_request)
        #         if sleep_time > 0:
        #             time.sleep(sleep_time)
        #             # Remove the oldest request time after sleeping
        #             self.request_times.pop(0)
        #     
        #     # Add current request time
        #     self.request_times.append(time.time())
    
    def set_rate_limit(self, requests_per_second):
        """Set the rate limit for API requests"""
        self.requests_per_second = max(0.1, requests_per_second)
    
    def start_combination(self, combination_id):
        """Start processing a combination - simplified synchronous version"""
        try:
            # Check if job exists
            job_summary = self.unified_db.get_job_summary(combination_id)
            if not job_summary:
                print(f"ERROR: Job {combination_id} not found")
                return False
                
            if job_summary['job_status'] == 'running':
                print(f"WARNING: Job {combination_id} is already running")
                return False
            
            # Run the job synchronously
            return self.run_job(combination_id)
            
        except Exception as e:
            print(f"ERROR: Failed to start job {combination_id}: {e}")
            return False
    
    def restart_combination(self, combination_id):
        """Restart a combination (retry failed species) using unified database"""
        try:
            # Check if job exists in unified database
            job_summary = self.unified_db.get_job_summary(combination_id)
            if not job_summary:
                return False, "Combination not found"
            
            # Reset all failed/timeout species back to pending in unified database
            conn = sqlite3.connect(self.unified_db.db_path)
            cursor = conn.cursor()
            
            # Reset failed/timeout species to pending
            cursor.execute("""
                UPDATE processing_results 
                SET status = 'pending', error = NULL, started_at = NULL, completed_at = NULL
                WHERE job_id = ? AND status IN ('failed', 'timeout')
            """, (combination_id,))
            
            # Reset job status to pending
            cursor.execute("""
                UPDATE processing_results 
                SET job_status = 'pending'
                WHERE job_id = ?
            """, (combination_id,))
            
            conn.commit()
            conn.close()
            
            # Add to processing queue
            self.add_combination(combination_id)
            return True, "Combination restarted successfully"
            
        except Exception as e:
            return False, str(e)
    
    def get_dashboard_data(self):
        """Get data for the dashboard matrix view"""
        # Use unified DB method
        return self.unified_db.get_dashboard_data()
    
    def get_combination_details(self, combination_id):
        """Get detailed information about a combination/job"""
        # Use job_id which is the combination_id
        job_summary = self.unified_db.get_job_summary(combination_id)
        if not job_summary:
            return jsonify({'error': 'Combination not found'}), 404
        
        # Get species results from unified table
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # First check if raw_response column exists
        cursor.execute("PRAGMA table_info(processing_results)")
        columns = [col[1] for col in cursor.fetchall()]
        has_raw_response = 'raw_response' in columns
        
        # Build query based on available columns
        select_fields = """binomial_name, status, error, knowledge_group, result,
                   gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                   biofilm_formation, animal_pathogenicity, biosafety_level,
                   health_association, host_association, plant_pathogenicity,
                   spore_formation, hemolysis, cell_shape"""
        
        if has_raw_response:
            select_fields += ", raw_response"
            
        cursor.execute(f"""
            SELECT {select_fields}
            FROM processing_results
            WHERE job_id = ?
            ORDER BY binomial_name
        """, (combination_id,))
        
        species_results = []
        for row in cursor.fetchall():
            result_data = {
                'binomial_name': row['binomial_name'],
                'status': row['status'],
                'error': row['error'],
                'knowledge_group': row['knowledge_group'],
                'result': row['result']
            }
            
            # Add raw_response if column exists
            if has_raw_response:
                result_data['raw_response'] = row['raw_response']
            elif row['result']:
                # Use result field as fallback for raw response
                result_data['raw_response'] = row['result']
            
            # Add individual phenotype fields to result_data
            phenotype_fields = ['gram_staining', 'motility', 'aerophilicity', 
                              'extreme_environment_tolerance', 'biofilm_formation',
                              'animal_pathogenicity', 'biosafety_level', 
                              'health_association', 'host_association',
                              'plant_pathogenicity', 'spore_formation', 
                              'hemolysis', 'cell_shape']
            
            # Add phenotypes both individually and as a group
            phenotypes = {}
            for field in phenotype_fields:
                result_data[field] = row[field]  # Add individual field
                if row[field]:
                    phenotypes[field] = row[field]
            
            if phenotypes:
                result_data['phenotypes'] = phenotypes
            
            species_results.append(result_data)
        
        conn.close()
        
        # Return in the format expected by the dashboard and job results page
        return jsonify({
            'combination_info': {
                'id': job_summary['job_id'],
                'species_file': job_summary['species_file'],
                'model': job_summary['model'],
                'system_template': job_summary['system_template'],
                'user_template': job_summary['user_template'],
                'status': job_summary['job_status'],
                'total_species': job_summary['total'],
                'successful_species': job_summary['successful'],
                'failed_species': job_summary['failed'],
                'timeout_species': job_summary['timeouts']
            },
            'species_results': species_results,
            # Add these for job_results.html compatibility
            'model': job_summary['model'],
            'species_file': job_summary['species_file'],
            'system_template': job_summary['system_template'],
            'user_template': job_summary['user_template'],
            'status': job_summary['job_status'],
            'created_at': job_summary.get('job_created_at', ''),
            'all_results': species_results,  # Include all results
            # Detect template type to help with column display
            'template_type': detect_template_type(job_summary['system_template'])
        })
    
    def create_combinations(self, species_file, models, template_pairs):
        """Create new combinations for processing using unified DB"""
        print(f"DEBUG: create_combinations called with species_file={species_file}, models={models}, template_pairs={template_pairs}")
        created = []
        
        # Resolve full path for species file
        from microbellm.shared import PROJECT_ROOT
        species_path = Path(species_file)
        species_file_path = species_file  # Keep original for file operations
        print(f"DEBUG: Original species_file_path: {species_file_path}")
        print(f"DEBUG: species_path.is_absolute(): {species_path.is_absolute()}")
        
        if not species_path.is_absolute():
            # Try to find the file in the species directory
            species_dir = PROJECT_ROOT / config.SPECIES_DIR
            full_path = species_dir / species_file
            print(f"DEBUG: Checking for file at: {full_path}")
            if full_path.exists():
                species_file_path = str(full_path)  # Use full path for file operations
                print(f"Resolved species file to: {species_file_path}")
            else:
                print(f"WARNING: Could not find species file: {species_file} in {species_dir}")
                print(f"DEBUG: Available files in {species_dir}:")
                if species_dir.exists():
                    for f in species_dir.iterdir():
                        if f.is_file():
                            print(f"  - {f.name}")
        
        # Normalize species_file to basename for database storage
        species_file = Path(species_file).name
        print(f"DEBUG: Normalized species_file for DB: {species_file}")
        
        # Read species file to get species list
        try:
            # Check if species file exists
            print(f"DEBUG: Checking if species file exists: {species_file_path}")
            if not Path(species_file_path).exists():
                raise FileNotFoundError(f"Species file not found: {species_file_path}")
            
            # Read species file
            print(f"DEBUG: Reading CSV file: {species_file_path}")
            df = pd.read_csv(species_file_path)
            print(f"DEBUG: CSV shape: {df.shape}")
            print(f"DEBUG: CSV columns: {list(df.columns)}")
            
            # Find binomial_name column
            binomial_col = None
            for col in df.columns:
                if col.lower().replace(' ', '_') in ['binomial_name', 'species', 'taxon_name', 'organism', 'species_name']:
                    binomial_col = col
                    break
            
            print(f"DEBUG: Found binomial column: {binomial_col}")
            if not binomial_col:
                raise ValueError("No binomial_name column found in species file")
            
            species_list = df[binomial_col].dropna().tolist()
            print(f"DEBUG: Found {len(species_list)} species in file")
            
        except Exception as e:
            print(f"DEBUG: Exception reading species file: {e}")
            raise ValueError(f"Failed to read species file: {e}")
        
        # Create jobs
        print(f"DEBUG: Creating jobs for {len(models)} models and {len(template_pairs)} template pairs")
        for model in models:
            for pair in template_pairs:
                print(f"DEBUG: Creating job for model={model}, system={pair['system']}, user={pair['user']}")
                try:
                    job_id = self.unified_db.create_job(
                        species_file=species_file,
                        model=model,
                        system_template=pair['system'],
                        user_template=pair['user'],
                        species_list=species_list
                    )
                    print(f"DEBUG: Created job with ID: {job_id}")
                    
                    created.append({
                        'id': job_id,
                        'species_file': species_file,
                        'model': model,
                        'system_template': pair['system'],
                        'user_template': pair['user']
                    })
                except Exception as e:
                    print(f"DEBUG: Failed to create job: {e}")
                    raise e
        
        print(f"DEBUG: Successfully created {len(created)} combinations")
        return created
    
    def export_results_to_csv(self, species_file=None, model=None, system_template=None, user_template=None):
        """Export results to CSV format with flexible filtering"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query with dynamic filters
        query = """
            SELECT r.binomial_name, r.result, r.status, r.error, 
                   r.species_file, r.model, r.system_template, r.user_template,
                   r.knowledge_group,
                   r.gram_staining, r.motility, r.aerophilicity, 
                   r.extreme_environment_tolerance, r.biofilm_formation, 
                   r.animal_pathogenicity, r.biosafety_level, r.health_association, 
                   r.host_association, r.plant_pathogenicity, r.spore_formation, 
                   r.hemolysis, r.cell_shape
            FROM results r
            WHERE 1=1
        """
        
        params = []
        if species_file:
            query += " AND r.species_file = ?"
            params.append(species_file)
        if model:
            query += " AND r.model = ?"
            params.append(model)
        if system_template:
            query += " AND r.system_template = ?"
            params.append(system_template)
        if user_template:
            query += " AND r.user_template = ?"
            params.append(user_template)
        
        cursor.execute(query, params)
        
        rows = []
        columns = [desc[0] for desc in cursor.description]
        
        for row in cursor.fetchall():
            row_data = dict(zip(columns, row))
            
            # Parse result if available and it's a JSON string
            if row_data.get('result'):
                try:
                    parsed = json.loads(row_data['result'])
                    # If result contains phenotypes, merge them
                    if isinstance(parsed, dict) and 'phenotypes' in parsed:
                        for key, value in parsed['phenotypes'].items():
                            if key not in row_data:
                                row_data[f'parsed_{key}'] = value
                except:
                    pass
            
            rows.append(row_data)
        
        conn.close()
        
        # Convert to CSV
        if rows:
            df = pd.DataFrame(rows)
            # Order columns nicely
            priority_cols = ['binomial_name', 'status', 'species_file', 'model', 
                           'system_template', 'user_template', 'knowledge_group']
            phenotype_cols = ['gram_staining', 'motility', 'aerophilicity', 
                            'extreme_environment_tolerance', 'biofilm_formation', 
                            'animal_pathogenicity', 'biosafety_level', 'health_association', 
                            'host_association', 'plant_pathogenicity', 'spore_formation', 
                            'hemolysis', 'cell_shape']
            
            # Arrange columns
            cols = []
            for col in priority_cols:
                if col in df.columns:
                    cols.append(col)
            for col in phenotype_cols:
                if col in df.columns:
                    cols.append(col)
            # Add any remaining columns
            for col in df.columns:
                if col not in cols:
                    cols.append(col)
            
            df = df[cols]
            return df.to_csv(index=False)
        else:
            return "No results found"
    
    def import_results_from_csv(self, csv_content):
        """Import results from CSV content"""
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            imported = 0
            skipped = 0
            errors = 0
            
            for _, row in df.iterrows():
                try:
                    # Required fields
                    species_file = row.get('species_file')
                    binomial_name = row.get('binomial_name')
                    model = row.get('model')
                    status = row.get('status', 'completed')
                    
                    if not all([species_file, binomial_name, model]):
                        errors += 1
                        continue
                    
                    # Get or create combination
                    system_template = row.get('system_template', '')
                    user_template = row.get('user_template', '')
                    
                    cursor.execute(
                        """SELECT id FROM combinations 
                           WHERE species_file = ? AND model = ?""",
                        (species_file, model)
                    )
                    
                    combo = cursor.fetchone()
                    if not combo:
                        # Create combination
                        cursor.execute(
                            """INSERT INTO combinations 
                               (species_file, model, system_template, user_template, status)
                               VALUES (?, ?, ?, ?, 'completed')""",
                            (species_file, model, system_template, user_template)
                        )
                        combination_id = cursor.lastrowid
                    else:
                        combination_id = combo[0]
                    
                    # Build result JSON
                    result_data = {}
                    phenotype_mapping = {
                        'gram_stain': 'gram_staining',
                        'shape': 'cell_shape',
                        'aerobic': 'aerophilicity',
                        'motility': 'motility',
                        'spore_forming': 'spore_formation',
                        'biofilm_forming': 'biofilm_formation'
                    }
                    
                    phenotypes = {}
                    for csv_field, db_field in phenotype_mapping.items():
                        if csv_field in row and pd.notna(row[csv_field]):
                            phenotypes[db_field] = row[csv_field]
                    
                    if phenotypes:
                        result_data['phenotypes'] = phenotypes
                    
                    knowledge_group = None
                    if 'knowledge_group' in row and pd.notna(row['knowledge_group']):
                        knowledge_group = row['knowledge_group']
                        result_data['knowledge_group'] = knowledge_group
                    
                    result_json = json.dumps(result_data) if result_data else None
                    
                    # Insert into results table
                    cursor.execute(
                        """INSERT OR REPLACE INTO results 
                           (species_file, binomial_name, model, system_template, user_template,
                            status, result, error, knowledge_group,
                            gram_staining, motility, aerophilicity, spore_formation,
                            biofilm_formation, cell_shape)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (species_file, binomial_name, model, system_template, user_template,
                         status, result_json, None, knowledge_group,
                         phenotypes.get('gram_staining'),
                         phenotypes.get('motility'),
                         phenotypes.get('aerophilicity'),
                         phenotypes.get('spore_formation'),
                         phenotypes.get('biofilm_formation'),
                         phenotypes.get('cell_shape'))
                    )
                    
                    imported += 1
                    
                except Exception as e:
                    print(f"Error importing row: {e}")
                    errors += 1
            
            conn.commit()
            conn.close()
            
            return {
                'total_rows': len(df),
                'imported': imported,
                'skipped': skipped,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'total_rows': 0,
                'imported': 0,
                'skipped': 0,
                'errors': 1,
                'error_message': str(e)
            }
    
    def import_results_validated(self, csv_content, template_name, template_type):
        """Import results from CSV with validation using template configs"""
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Get validation config if available
            validator = None
            if template_name:
                from microbellm.template_config import find_validation_config_for_template, TemplateValidator
                config_path = find_validation_config_for_template(f"templates/{template_name}")
                if config_path:
                    validator = TemplateValidator(config_path)
            
            # Track unique models and species files for auto-registration
            unique_models = set()
            unique_species_files = set()
            
            imported = 0
            skipped = 0
            errors = 0
            validation_errors = []
            
            for idx, row in df.iterrows():
                try:
                    # Required fields
                    binomial_name = row.get('binomial_name')
                    model = row.get('model')
                    status = row.get('status', 'completed')
                    
                    if not all([binomial_name, model]):
                        validation_errors.append(f"Row {idx+1}: Missing required fields (binomial_name, model)")
                        errors += 1
                        continue
                    
                    # Track models and species files
                    unique_models.add(model)
                    
                    # Get template paths
                    species_file = row.get('species_file', 'imported_species.txt')
                    system_template = row.get('system_template', 'templates/system_phenotype.txt')
                    user_template = row.get('user_template', f'templates/{template_name}' if template_name else 'unknown')
                    
                    unique_species_files.add(species_file)
                    
                    # Check if entry already exists
                    existing = self.unified_db.get_species_result_by_params(
                        binomial_name, model, system_template, user_template
                    )
                    
                    if existing:
                        skipped += 1
                        continue
                    
                    # Parse result field
                    result_json = row.get('result')
                    parsed_result = {}
                    
                    if result_json and pd.notna(result_json):
                        try:
                            parsed_result = json.loads(result_json)
                        except json.JSONDecodeError:
                            # Try to parse as string
                            parsed_result = {'raw_response': result_json}
                    
                    # Validate parsed result if validator available
                    if validator and parsed_result:
                        validated_data, val_errors = validator.validate_response(parsed_result)
                        if val_errors:
                            validation_errors.extend([f"Row {idx+1}: {err}" for err in val_errors])
                        parsed_result = validated_data
                    
                    # Extract fields from CSV or parsed result
                    knowledge_group = row.get('knowledge_group') or parsed_result.get('knowledge_group')
                    
                    # Get phenotype fields
                    phenotype_fields = [
                        'gram_staining', 'motility', 'aerophilicity', 
                        'extreme_environment_tolerance', 'biofilm_formation',
                        'animal_pathogenicity', 'biosafety_level', 
                        'health_association', 'host_association',
                        'plant_pathogenicity', 'spore_formation', 
                        'hemolysis', 'cell_shape'
                    ]
                    
                    phenotype_data = {}
                    for field in phenotype_fields:
                        # Check CSV first, then parsed result
                        value = row.get(field)
                        if pd.isna(value) or value is None:
                            value = parsed_result.get(field)
                        if value and pd.notna(value):
                            phenotype_data[field] = value
                    
                    # Create job for imported data
                    job_id = self.unified_db.create_import_job(
                        species_file=species_file,
                        model=model,
                        system_template=system_template,
                        user_template=user_template,
                        binomial_name=binomial_name,
                        status=status,
                        result=json.dumps(parsed_result) if parsed_result else result_json,
                        error=row.get('error'),
                        knowledge_group=knowledge_group,
                        **phenotype_data
                    )
                    
                    imported += 1
                    
                except Exception as e:
                    validation_errors.append(f"Row {idx+1}: {str(e)}")
                    errors += 1
            
            # Auto-register any new models and species files
            if unique_models or unique_species_files:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                try:
                    # Add new models to managed_models table
                    for model in unique_models:
                        cursor.execute("INSERT OR IGNORE INTO managed_models (model) VALUES (?)", (model,))
                    
                    # Add new species files to managed_species_files table
                    for species_file in unique_species_files:
                        cursor.execute("INSERT OR IGNORE INTO managed_species_files (species_file) VALUES (?)", (species_file,))
                    
                    conn.commit()
                except Exception as e:
                    # Log but don't fail the import
                    print(f"Warning: Failed to auto-register models/species files: {e}")
                finally:
                    conn.close()
            
            return {
                'total_rows': len(df),
                'imported': imported,
                'skipped': skipped,
                'errors': errors,
                'validation_errors': validation_errors,
                'new_models_added': list(unique_models),
                'new_species_files_added': list(unique_species_files)
            }
            
        except Exception as e:
            return {
                'total_rows': 0,
                'imported': 0,
                'skipped': 0,
                'errors': 1,
                'error_message': str(e),
                'validation_errors': [str(e)]
            }
    
    def _rerun_single_species(self, combination_id, species_name, model, system_template, user_template):
        """Re-run a single species in a separate thread"""
        thread = threading.Thread(
            target=self._process_single_species_rerun,
            args=(combination_id, species_name, model, system_template, user_template)
        )
        thread.daemon = True
        thread.start()
    
    def _process_single_species_rerun(self, combination_id, species_name, model, system_template, user_template):
        """Process a single species rerun"""
        try:
            # Read templates
            system_prompt = read_template_from_file(system_template)
            user_prompt = read_template_from_file(user_template)
            
            # Process species
            result, status, error = self._process_single_species(
                species_name, model, system_prompt, user_prompt, combination_id
            )
            
            # Store result
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get species file from combination
            cursor.execute("SELECT species_file FROM combinations WHERE id = ?", (combination_id,))
            species_file = cursor.fetchone()[0]
            
            # Parse result for knowledge_group and phenotypes
            knowledge_group = None
            phenotype_data = {}
            
            if result and status == 'completed':
                try:
                    result_dict = json.loads(result)
                    knowledge_group = result_dict.get('knowledge_group') or result_dict.get('knowledge_level')
                    if 'phenotypes' in result_dict:
                        phenotype_data = result_dict['phenotypes']
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
                    pass
            
            # Insert into results table
            cursor.execute("""
                INSERT INTO results (
                    species_file, binomial_name, model, system_template, user_template,
                    status, result, error, knowledge_group,
                    gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                    biofilm_formation, animal_pathogenicity, biosafety_level,
                    health_association, host_association, plant_pathogenicity,
                    spore_formation, hemolysis, cell_shape
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                species_file, species_name, model, system_template, user_template,
                status, result, error, knowledge_group,
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
            
            conn.commit()
            conn.close()
            
            # Emit update
            socketio.emit('species_rerun_complete', {
                'combination_id': combination_id,
                'species_name': species_name,
                'status': status
            })
            
        except Exception as e:
            print(f"Error in rerun: {e}")
    
    def reparse_phenotype_data(self, combination_id):
        """Re-parse phenotype data for all results in a combination"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Get combination details first
            cursor.execute("""
                SELECT species_file, model, system_template, user_template
                FROM combinations WHERE id = ?
            """, (combination_id,))
            combo = cursor.fetchone()
            
            if not combo:
                return 0
            
            # Get all results for this combination from results table
            cursor.execute(
                """SELECT binomial_name, result 
                   FROM results 
                   WHERE species_file = ? AND model = ? AND system_template = ? 
                   AND user_template = ? AND result IS NOT NULL""",
                (combo[0], combo[1], combo[2], combo[3])
            )
            
            results = cursor.fetchall()
            updated = 0
            
            for result_id, raw_result in results:
                if raw_result:
                    # Re-parse the result
                    # This is a placeholder - in reality you'd apply your parsing logic
                    try:
                        # Try to parse existing JSON
                        parsed = json.loads(raw_result)
                        # Result is already properly formatted
                        continue
                    except:
                        # Need to parse raw text response
                        # This would involve extracting phenotype fields from text
                        # For now, we'll skip this
                        pass
            
            conn.commit()
            return jsonify({
                'success': True,
                'updated': updated,
                'message': f'Re-parsed {updated} results'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 500
        finally:
            conn.close()
    
    def rerun_all_failed(self, combination_id):
        """Re-run all failed species for a combination"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Get combination details
            cursor.execute("""
                SELECT id, species_file, model, system_template, user_template
                FROM combinations WHERE id = ?
            """, (combination_id,))
            combo = cursor.fetchone()
            
            if not combo:
                return jsonify({'success': False, 'error': 'Combination not found'}), 404
            
            # Get failed species from results table
            cursor.execute(
                """SELECT DISTINCT binomial_name 
                   FROM results 
                   WHERE species_file = ? AND model = ? AND system_template = ? 
                   AND user_template = ? AND status != 'completed'""",
                (combo[1], combo[2], combo[3], combo[4])
            )
            
            failed_species = [row[0] for row in cursor.fetchall()]
            
            if not failed_species:
                return jsonify({
                    'success': False,
                    'error': 'No failed species found'
                })
            
            # Delete failed results from results table
            cursor.execute(
                """DELETE FROM results 
                   WHERE species_file = ? AND model = ? AND system_template = ? 
                   AND user_template = ? AND status != 'completed'""",
                (combo[1], combo[2], combo[3], combo[4])
            )
            
            # Reset combination to pending
            cursor.execute(
                "UPDATE combinations SET status = 'pending' WHERE id = ?",
                (combination_id,)
            )
            
            conn.commit()
            conn.close()
            
            # Add to processing queue
            self.add_combination(combination_id)
            
            return jsonify({
                'success': True,
                'failed_count': len(failed_species),
                'message': f'Re-running {len(failed_species)} failed species'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        finally:
            if conn:
                conn.close()

# Helper functions for dashboard
def get_available_species_files():
    """Get list of available species files"""
    # Get directory from config
    species_dir = Path(config.SPECIES_DIR)
    
    print(f"\n=== Looking for species files ===")
    print(f"Config SPECIES_DIR: {config.SPECIES_DIR}")
    
    # Try absolute path first
    if not species_dir.is_absolute():
        # Try relative to project root
        from microbellm.shared import PROJECT_ROOT
        species_dir = PROJECT_ROOT / species_dir
        print(f"Trying relative to PROJECT_ROOT: {species_dir}")
    
    if not species_dir.exists():
        # Try relative to the script location
        species_dir = Path(__file__).parent.parent / config.SPECIES_DIR
        print(f"Trying relative to script: {species_dir}")
    
    if not species_dir.exists():
        print(f"ERROR: Species directory not found at {species_dir}")
        return []
    
    # Look for .tsv and .txt files
    files = list(species_dir.glob('*.tsv')) + list(species_dir.glob('*.txt'))
    print(f"Found {len(files)} species files in {species_dir}")
    for f in files:
        print(f"  - {f.name}")
    return [f.name for f in files]

def get_popular_models():
    """Get list of popular models"""
    # This could be expanded to actually track popularity
    return [
        'anthropic/claude-3-haiku',
        'anthropic/claude-3-sonnet',
        'openai/gpt-4-turbo',
        'openai/gpt-3.5-turbo',
        'meta-llama/llama-3-70b-instruct'
    ]

def get_available_template_pairs():
    """Get available template pairs"""
    # Templates are in the root templates directory
    template_dir = Path('templates')
    if not template_dir.exists():
        # Try relative to the script location
        template_dir = Path(__file__).parent.parent / 'templates'
    if not template_dir.exists():
        return []
    
    # Get all system templates
    system_templates = [f for f in template_dir.glob('system/*.txt')]
    
    # Group by base name
    template_pairs = []
    seen_bases = set()
    
    for sys_template in system_templates:
        base_name = sys_template.stem
        if base_name not in seen_bases:
            user_template = template_dir / 'user' / f'{base_name}.txt'
            if user_template.exists():
                template_pairs.append({
                    'system': str(sys_template),
                    'user': str(user_template),
                    'name': base_name
                })
                seen_bases.add(base_name)
    
    return template_pairs

# Routes
@app.route('/')
def index():
    """Main dashboard page with table/matrix view"""
    dashboard_data = processing_manager.get_dashboard_data()
    
    # Add additional data needed for the modals
    dashboard_data['available_species_files'] = get_available_species_files()
    dashboard_data['popular_models'] = get_popular_models()
    dashboard_data['available_template_pairs'] = get_available_template_pairs()
    
    return render_template('dashboard.html', dashboard_data=dashboard_data)

@app.route('/dashboard')
def dashboard():
    """Technical dashboard for managing processing jobs"""
    dashboard_data = processing_manager.get_dashboard_data()
    
    # Add additional data needed for the modals
    dashboard_data['available_species_files'] = get_available_species_files()
    dashboard_data['popular_models'] = get_popular_models()
    dashboard_data['available_template_pairs'] = get_available_template_pairs()
    
    return render_template('dashboard.html', dashboard_data=dashboard_data)

@app.route('/api/start_combination/<combination_id>', methods=['POST'])
def start_combination_api(combination_id):
    if processing_manager.start_combination(combination_id):
        return jsonify({'success': True, 'message': f'Combination {combination_id} started successfully.'})
    else:
        return jsonify({'success': False, 'error': 'Failed to start combination'}), 500

@app.route('/api/restart_combination/<combination_id>', methods=['POST'])
def restart_combination_api(combination_id):
    success, message = processing_manager.restart_combination(combination_id)
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'error': message}), 500

@app.route('/api/pause_combination/<combination_id>', methods=['POST'])
def pause_combination_api(combination_id):
    success = processing_manager.pause_combination(combination_id)
    if success:
        return jsonify({'success': True, 'message': 'Combination paused'})
    else:
        return jsonify({'success': False, 'error': 'Failed to pause combination'}), 500

@app.route('/api/stop_combination/<combination_id>', methods=['POST'])
def stop_combination_api(combination_id):
    success = processing_manager.stop_combination(combination_id)
    if success:
        return jsonify({'success': True, 'message': 'Combination stopped'})
    else:
        return jsonify({'success': False, 'error': 'Failed to stop combination'}), 500

@app.route('/api/set_rate_limit', methods=['POST'])
def set_rate_limit():
    data = request.get_json()
    requests_per_second = data.get('requests_per_second', 30.0)
    max_concurrent_requests = data.get('max_concurrent_requests', 30)
    
    processing_manager.set_rate_limit(requests_per_second)
    if max_concurrent_requests:
        processing_manager.set_max_concurrent_requests(max_concurrent_requests)
    
    return jsonify({'message': 'Rate limit updated successfully'})

@app.route('/api/get_settings')
def get_settings():
    return jsonify({
        'rate_limit': processing_manager.requests_per_second,
        'max_concurrent_requests': processing_manager.max_concurrent_requests,
        'queue_length': len(processing_manager.job_queue)
    })

@app.route('/api/dashboard_data')
def get_dashboard_data_api():
    data = processing_manager.get_dashboard_data()
    return jsonify(data)

@app.route('/api/delete_combination/<combination_id>', methods=['DELETE'])
def delete_combination_api(combination_id):
    try:
        print(f"DEBUG: Deleting combination/job: {combination_id}")
        
        # Check if job exists first
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processing_results WHERE job_id = ?", (combination_id,))
        count_before = cursor.fetchone()[0]
        print(f"DEBUG: Found {count_before} entries for job {combination_id}")
        conn.close()
        
        # Use unified database to delete the job
        from microbellm.unified_db import UnifiedDB
        unified_db = UnifiedDB(db_path)
        
        # Delete all entries for this job_id
        unified_db.delete_job(combination_id)
        
        # Verify deletion worked
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processing_results WHERE job_id = ?", (combination_id,))
        count_after = cursor.fetchone()[0]
        print(f"DEBUG: After deletion, found {count_after} entries for job {combination_id}")
        
        # Also clean up from legacy tables if they exist
        cursor.execute("DELETE FROM combinations WHERE id = ?", (combination_id,))
        cursor.execute("DELETE FROM results WHERE id = ?", (combination_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'Job {combination_id} deleted. Removed {count_before} entries.'})
    except Exception as e:
        print(f"DEBUG: Error deleting job: {e}")
        return jsonify({'success': False, 'message': f'Error deleting job: {e}'}), 500

@app.route('/api/database_info')
def database_info_api():
    """Get database structure information"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        db_info = {
            'database_path': db_path,
            'tables': {}
        }
        
        for table in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [{'name': row[1], 'type': row[2], 'not_null': bool(row[3]), 'primary_key': bool(row[5])} 
                      for row in cursor.fetchall()]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            db_info['tables'][table] = {
                'columns': columns,
                'row_count': row_count
            }
        
        conn.close()
        return jsonify(db_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/table_data/<table_name>')
def table_data_api(table_name):
    """Get sample data from a table"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Validate table name exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            return jsonify({'error': 'Table not found'}), 404
        
        # Get sample data (first 100 rows)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
        rows = cursor.fetchall()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'table_name': table_name,
            'columns': columns,
            'rows': rows,
            'sample_size': len(rows)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup_orphaned/<species_file>/<model>/<system_template>/<user_template>', methods=['DELETE'])
def cleanup_orphaned_results_api(species_file, model, system_template, user_template):
    """Delete orphaned results that don't have a combination"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Delete orphaned results
        cursor.execute("""
            DELETE FROM results 
            WHERE species_file = ? AND model = ? 
            AND system_template = ? AND user_template = ?
            AND NOT EXISTS (
                SELECT 1 FROM combinations c 
                WHERE c.species_file = results.species_file 
                AND c.model = results.model 
                AND c.system_template = results.system_template 
                AND c.user_template = results.user_template
            )
        """, (species_file, model, system_template, user_template))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        return jsonify({
            'success': True, 
            'message': f'Deleted {deleted_count} orphaned results',
            'deleted_count': deleted_count
        })
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/export')
def export_page():
    """Export page for downloading results"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get unique values from results table (not combinations)
    cursor.execute("SELECT DISTINCT species_file FROM results ORDER BY species_file")
    species_files = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT model FROM results ORDER BY model")
    models = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT system_template, user_template FROM results ORDER BY system_template, user_template")
    templates = [{'system': row[0], 'user': row[1]} for row in cursor.fetchall()]
    
    # Get result count for summary
    cursor.execute("SELECT COUNT(*) FROM results")
    total_results = cursor.fetchone()[0]
    
    conn.close()
    
    return render_template('export.html', 
                         species_files=species_files, 
                         models=models, 
                         templates=templates,
                         total_results=total_results)

@app.route('/import')
def import_page():
    return render_template('import.html')

@app.route('/api/import_csv', methods=['POST'])
def import_csv():
    """Import results from uploaded CSV file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        # Read CSV content
        content = file.read().decode('utf-8')
        import_results = processing_manager.import_results_from_csv(content)
        
        return jsonify({
            'success': True,
            'total_rows': import_results['total_rows'],
            'imported': import_results['imported'],
            'skipped': import_results['skipped'],
            'errors': import_results['errors']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/import_csv_validated', methods=['POST'])
def import_csv_validated():
    """Import results from uploaded CSV file with validation"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    template_name = request.form.get('template')
    template_type = request.form.get('template_type')
    
    try:
        # Read CSV content
        content = file.read().decode('utf-8')
        import_results = processing_manager.import_results_validated(content, template_name, template_type)
        
        return jsonify({
            'success': True,
            'total_rows': import_results['total_rows'],
            'imported': import_results['imported'],
            'skipped': import_results['skipped'],
            'errors': import_results['errors'],
            'validation_errors': import_results.get('validation_errors', [])
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e),
            'validation_errors': [str(e)]
        }), 500

@app.route('/api/export_csv')
def export_csv_api():
    """Export results as CSV with flexible filtering"""
    species_file = request.args.get('species_file')
    model = request.args.get('model')
    system_template = request.args.get('system_template')
    user_template = request.args.get('user_template')
    
    try:
        # Pass filters to export function (None values will be ignored)
        csv_content = processing_manager.export_results_to_csv(
            species_file=species_file if species_file else None,
            model=model if model else None,
            system_template=system_template if system_template else None,
            user_template=user_template if user_template else None
        )
        
        if csv_content == "No results found":
            return jsonify({'error': 'No results found for the selected filters'}), 404
        
        # Create response with CSV
        output = io.StringIO()
        output.write(csv_content)
        output.seek(0)
        
        # Generate filename based on filters
        filename_parts = ['microbebench_export']
        if species_file:
            filename_parts.append(Path(species_file).stem)
        if model:
            filename_parts.append(model.replace('/', '_'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_parts.append(timestamp)
        filename = f"{'_'.join(filename_parts)}.csv"
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/settings')
def settings():
    return render_template('settings.html')

# Removed duplicate /templates route - using templates_page() instead

@app.route('/api/create_job', methods=['POST'])
def create_job_api():
    data = request.get_json()
    species_file = data.get('species_file')
    model = data.get('model')
    system_template = data.get('system_template')
    user_template = data.get('user_template')
    
    print(f"Creating job: {model} + {system_template} + {user_template} + {species_file}")
    
    if not all([species_file, model, system_template, user_template]):
        missing = [k for k, v in {'species_file': species_file, 'model': model, 'system_template': system_template, 'user_template': user_template}.items() if not v]
        return jsonify({'success': False, 'error': f'Missing required fields: {missing}'}), 400
    
    try:
        # Create job using simplified method
        job_id = processing_manager.create_job(species_file, model, system_template, user_template)
        
        if job_id:
            return jsonify({
                'success': True,
                'job_id': job_id,
                'combination_id': job_id,  # For backward compatibility
                'message': 'Job created successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create job'}), 500
            
    except Exception as e:
        print(f"Error creating job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Alias for backward compatibility
@app.route('/api/create_and_run_job', methods=['POST'])
def create_and_run_job_api():
    """Create and immediately start a job - async workflow"""
    data = request.get_json()
    species_file = data.get('species_file')
    model = data.get('model')
    system_template = data.get('system_template')
    user_template = data.get('user_template')
    
    print(f"Creating and starting job: {model} + {system_template} + {user_template} + {species_file}")
    
    if not all([species_file, model, system_template, user_template]):
        missing = [k for k, v in {'species_file': species_file, 'model': model, 'system_template': system_template, 'user_template': user_template}.items() if not v]
        return jsonify({'success': False, 'error': f'Missing required fields: {missing}'}), 400
    
    try:
        # Create the job
        job_id = processing_manager.create_job(species_file, model, system_template, user_template)
        
        if job_id:
            # Start the job asynchronously
            success = processing_manager.start_job_async(job_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'job_id': job_id,
                    'combination_id': job_id,  # For backward compatibility
                    'message': 'Job created and started successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'job_id': job_id,
                    'error': 'Job created but failed to start'
                }), 500
        else:
            return jsonify({'success': False, 'error': 'Failed to create job'}), 500
            
    except Exception as e:
        print(f"Error creating and starting job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/create_combination', methods=['POST'])
def create_combination_api():
    """Backward compatibility alias for create_and_run_job_api"""
    return create_and_run_job_api()

@app.route('/api/run_job/<job_id>', methods=['POST'])
def run_job_api(job_id):
    try:
        success = processing_manager.run_job(job_id)
        if success:
            return jsonify({'success': True, 'message': 'Job completed successfully'})
        else:
            return jsonify({'success': False, 'error': 'Job failed'}), 500
    except Exception as e:
        print(f"Error running job: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobs')
def get_jobs():
    """Get list of all jobs"""
    try:
        jobs = processing_manager.get_all_jobs()
        return jsonify({'success': True, 'jobs': jobs})
    except Exception as e:
        print(f"Error getting jobs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/job/<job_id>/results')
def get_job_results(job_id):
    """Get detailed results for a job"""
    try:
        results = processing_manager.get_job_results(job_id)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"Error getting job results: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/job/<job_id>/results')
def job_results_page(job_id):
    """Job results page"""
    return render_template('job_results.html', job_id=job_id)

@app.route('/job_results/<combination_id>')
def job_results_detail_page(combination_id):
    """Detailed job results page showing all species"""
    # Use simpler template for now
    return render_template('job_results_simple.html', combination_id=combination_id)

@app.route('/database')
def database_browser():
    """Database browser page"""
    return render_template('database_browser.html')

@app.route('/api/add_model', methods=['POST'])
def add_model_api():
    data = request.get_json()
    model = data.get('model')
    
    if not model:
        return jsonify({'success': False, 'error': 'Model name required'}), 400
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT OR IGNORE INTO managed_models (model) VALUES (?)", (model,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Model added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/delete_model', methods=['POST'])
def delete_model_api():
    data = request.get_json()
    model = data.get('model')
    
    if not model:
        return jsonify({'success': False, 'error': 'Model name required'}), 400
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM managed_models WHERE model = ?", (model,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Model removed successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/add_species_file', methods=['POST'])
def add_species_file_api():
    data = request.get_json()
    species_file = data.get('species_file')
    
    if not species_file:
        return jsonify({'success': False, 'error': 'Species file required'}), 400
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT OR IGNORE INTO managed_species_files (species_file) VALUES (?)", (species_file,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Species file added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/delete_species_file', methods=['POST'])
def delete_species_file_api():
    data = request.get_json()
    species_file = data.get('species_file')
    
    if not species_file:
        return jsonify({'success': False, 'error': 'Species file required'}), 400
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM managed_species_files WHERE species_file = ?", (species_file,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Species file removed successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/combination_details/<combination_id>')
def get_combination_details(combination_id):
    """Get detailed information about a combination and its results"""
    return processing_manager.get_combination_details(combination_id)

@app.route('/api/api_key_status')
def api_key_status():
    """Check if API key is configured"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        # Mask the API key for display (show first 6 and last 4 characters)
        if len(api_key) > 10:
            masked_key = api_key[:6] + '*' * (len(api_key) - 10) + api_key[-4:]
        else:
            masked_key = '*' * len(api_key)
        
        return jsonify({
            'configured': True,
            'status': 'configured',
            'message': 'API key is set',
            'masked_key': masked_key
        })
    else:
        return jsonify({
            'configured': False,
            'status': 'missing',
            'message': 'No API key found',
            'masked_key': None
        })

@app.route('/api/set_api_key', methods=['POST'])
def set_api_key():
    """Set the API key (stores in .env file)"""
    data = request.get_json()
    api_key = data.get('api_key')
    
    if not api_key:
        return jsonify({'success': False, 'status': 'error', 'message': 'API key required'}), 400
    
    try:
        # Update .env file
        env_file = Path('.env')
        lines = []
        key_found = False
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('OPENROUTER_API_KEY='):
                        lines.append(f'OPENROUTER_API_KEY={api_key}\n')
                        key_found = True
                    else:
                        lines.append(line)
        
        if not key_found:
            lines.append(f'OPENROUTER_API_KEY={api_key}\n')
        
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        # Update environment variable for current session
        os.environ['OPENROUTER_API_KEY'] = api_key
        
        return jsonify({'success': True, 'status': 'success', 'message': 'API key updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_openrouter_models')
def get_openrouter_models():
    """Fetch available models from OpenRouter API"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        # Return common models as fallback when no API key
        common_models = [
            {'id': 'anthropic/claude-3-haiku', 'name': 'Claude 3 Haiku', 'context_length': 200000, 'pricing': {'prompt': 0.25, 'completion': 1.25}},
            {'id': 'anthropic/claude-3-sonnet', 'name': 'Claude 3 Sonnet', 'context_length': 200000, 'pricing': {'prompt': 3, 'completion': 15}},
            {'id': 'anthropic/claude-3-opus', 'name': 'Claude 3 Opus', 'context_length': 200000, 'pricing': {'prompt': 15, 'completion': 75}},
            {'id': 'openai/gpt-4-turbo', 'name': 'GPT-4 Turbo', 'context_length': 128000, 'pricing': {'prompt': 10, 'completion': 30}},
            {'id': 'openai/gpt-3.5-turbo', 'name': 'GPT-3.5 Turbo', 'context_length': 16385, 'pricing': {'prompt': 0.5, 'completion': 1.5}},
            {'id': 'meta-llama/llama-3.2-3b-instruct', 'name': 'Llama 3.2 3B Instruct', 'context_length': 131072, 'pricing': {'prompt': 0.06, 'completion': 0.06}},
            {'id': 'mistralai/mistral-7b-instruct-v0.3', 'name': 'Mistral 7B Instruct v0.3', 'context_length': 32768, 'pricing': {'prompt': 0.06, 'completion': 0.06}},
            {'id': 'google/gemini-flash-1.5', 'name': 'Gemini Flash 1.5', 'context_length': 1048576, 'pricing': {'prompt': 0.075, 'completion': 0.3}}
        ]
        return jsonify({'models': common_models, 'warning': 'Using fallback model list. Set OPENROUTER_API_KEY for full list.'})
    
    try:
        response = requests.get(
            'https://openrouter.ai/api/v1/models',
            headers={'Authorization': f'Bearer {api_key}'}
        )
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            
            # Filter and format models
            formatted_models = []
            for model in models:
                formatted_models.append({
                    'id': model['id'],
                    'name': model.get('name', model['id']),
                    'context_length': model.get('context_length', 'N/A'),
                    'pricing': model.get('pricing', {})
                })
            
            # Sort by name
            formatted_models.sort(key=lambda x: x['name'].lower())
            
            return jsonify({'models': formatted_models})
        else:
            return jsonify({'error': 'Failed to fetch models'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_model', methods=['POST'])
def validate_model():
    """Validate if a model exists on OpenRouter"""
    data = request.get_json()
    model = data.get('model')
    
    if not model:
        return jsonify({'valid': False, 'error': 'Model name required'}), 400
    
    # For now, just check format
    if '/' in model and len(model.split('/')) == 2:
        return jsonify({'valid': True})
    else:
        return jsonify({'valid': False, 'error': 'Invalid format. Use: provider/model-name'})

@app.route('/templates')
def templates_page():
    """View all template configurations"""
    # Templates are in the root project templates directory
    # Get the project root (parent of microbellm package)
    project_root = Path(__file__).parent.parent
    template_dir = project_root / 'templates'
    
    template_data = {}
    
    # Find all system templates
    system_dir = template_dir / 'system'
    if system_dir.exists():
        for system_file in system_dir.glob('*.txt'):
            template_name = system_file.stem
            
            # Look for matching user template
            user_file = template_dir / 'user' / f'{template_name}.txt'
            if user_file.exists():
                template_data[template_name] = {
                    'system': {
                        'path': str(system_file),
                        'content': system_file.read_text()
                    },
                    'user': {
                        'path': str(user_file),
                        'content': user_file.read_text()
                    }
                }
                
                # Look for validation config
                validation_file = template_dir / 'validation' / f'{template_name}.json'
                if validation_file.exists():
                    try:
                        import json
                        validation_content = validation_file.read_text()
                        validation_json = json.loads(validation_content)
                        template_data[template_name]['validation'] = {
                            'path': str(validation_file),
                            'content': validation_content,
                            'info': validation_json.get('template_info', {})
                        }
                    except:
                        pass
    
    return render_template('view_template.html', template_data=template_data)

@app.route('/api/templates')
def get_templates_api():
    """Get all available templates with validation configs"""
    try:
        # Get templates from disk
        project_root = Path(__file__).parent.parent
        template_dir = project_root / 'templates'
        
        templates = []
        
        # Find all user templates (these are what users select)
        user_dir = template_dir / 'user'
        if user_dir.exists():
            for template_file in user_dir.glob('*.txt'):
                template_name = template_file.name
                templates.append({
                    'name': template_name,
                    'path': str(template_file),
                    'type': detect_template_type(str(template_file))
                })
            
            # Also check for markdown templates
            for template_file in user_dir.glob('*.md'):
                template_name = template_file.name
                templates.append({
                    'name': template_name,
                    'path': str(template_file),
                    'type': detect_template_type(str(template_file))
                })
        
        # Get validation configs
        from microbellm.template_config import get_all_template_validation_configs
        validation_configs = get_all_template_validation_configs()
        
        return jsonify({
            'success': True,
            'templates': templates,
            'validation_configs': validation_configs
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/view_template/<template_type>/<template_name>')
def view_template_single(template_type, template_name):
    """View a single template content - for direct links"""
    # Redirect to the templates page with this template highlighted
    return redirect(f'/templates#{template_name}')

@app.route('/ground_truth')
def ground_truth_viewer():
    """Ground truth data viewer page"""
    # Initialize ground truth tables if needed
    create_ground_truth_tables()
    return render_template('ground_truth.html')

# Ground Truth API Routes
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
        
        if not dataset_name:
            return jsonify({'success': False, 'error': 'Dataset name is required'})
        
        # Save the uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
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
        if data:
            return jsonify({
                'success': True,
                'data': data[0]  # Return first match
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Species not found in ground truth data'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ground_truth/dataset/<dataset_name>', methods=['DELETE'])
def api_delete_ground_truth_dataset(dataset_name):
    """Delete a ground truth dataset"""
    try:
        success = delete_ground_truth_dataset(dataset_name)
        if success:
            return jsonify({
                'success': True,
                'message': f'Dataset "{dataset_name}" deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete dataset'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ground_truth/distribution', methods=['GET'])
def api_get_ground_truth_distribution():
    """Get phenotype distribution for all ground truth data"""
    try:
        dataset_name = request.args.get('dataset')
        
        # Get all data for the dataset
        data = get_ground_truth_data(dataset_name=dataset_name)
        
        # Calculate distribution
        phenotypes = [
            'gram_staining', 'motility', 'aerophilicity', 'extreme_environment_tolerance',
            'biofilm_formation', 'animal_pathogenicity', 'biosafety_level',
            'health_association', 'host_association', 'plant_pathogenicity',
            'spore_formation', 'hemolysis', 'cell_shape'
        ]
        
        distribution = {}
        for phenotype in phenotypes:
            distribution[phenotype] = {
                'total': len(data),
                'annotated': 0,
                'values': {}
            }
            
            for record in data:
                value = record.get(phenotype, '')
                if value and value.lower() not in ['na', 'n/a', 'unknown', '']:
                    distribution[phenotype]['annotated'] += 1
                    if value not in distribution[phenotype]['values']:
                        distribution[phenotype]['values'][value] = 0
                    distribution[phenotype]['values'][value] += 1
        
        return jsonify({
            'success': True,
            'distribution': distribution,
            'total_species': len(data),
            'dataset_name': dataset_name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/rerun_failed_species', methods=['POST'])
def rerun_failed_species():
    """Re-run a specific failed species"""
    data = request.get_json()
    combination_id = data.get('combination_id')
    species_name = data.get('species_name')
    
    if not combination_id or not species_name:
        return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
    
    try:
        # Get combination details
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, species_file, model, system_template, user_template
            FROM combinations WHERE id = ?
        """, (combination_id,))
        combination = cursor.fetchone()
        
        if not combination:
            return jsonify({'success': False, 'error': 'Combination not found'}), 404
        
        # Delete the old failed result from results table
        cursor.execute(
            """DELETE FROM results 
               WHERE species_file = ? AND model = ? AND system_template = ? 
               AND user_template = ? AND binomial_name = ? AND status != 'completed'""",
            (combination[1], combination[2], combination[3], combination[4], species_name)
        )
        conn.commit()
        conn.close()
        
        # Queue for reprocessing
        processing_manager._rerun_single_species(combination_id, species_name, 
                                               combination[2], combination[3], 
                                               combination[4])
        
        return jsonify({
            'success': True,
            'message': f'Re-running {species_name}...'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reparse_phenotype_data/<combination_id>', methods=['POST'])
def reparse_phenotype_data(combination_id):
    """Re-parse phenotype data for a combination"""
    return processing_manager.reparse_phenotype_data(combination_id)

@app.route('/api/rerun_all_failed/<combination_id>', methods=['POST'])
def rerun_all_failed(combination_id):
    """Re-run all failed species for a combination"""
    return processing_manager.rerun_all_failed(combination_id)

def main():
    """Main entry point for the admin application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MicrobeLLM Admin Dashboard")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5050, help='Port to bind to (default: 5050)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize processing manager
    global processing_manager
    processing_manager = ProcessingManager()
    
    # Reset running jobs
    reset_running_jobs_on_startup(db_path)
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("=" * 50)
        print("WARNING: OPENROUTER_API_KEY not set!")
        print("=" * 50)
        print("You need to set your OpenRouter API key to use the prediction features.")
        print("Get your key from: https://openrouter.ai/keys")
        print("Then set it with: export OPENROUTER_API_KEY='your-api-key'")
        print("=" * 50)
        print()
    
    print(f"Starting MicrobeLLM Admin Dashboard...")
    print(f"Access the dashboard at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Run with SocketIO
    socketio.run(app, host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main()