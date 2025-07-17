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
        self.running_combinations = {}
        self.job_queue = []
        self.paused_combinations = set()
        self.stopped_combinations = set()
        self.requests_per_second = 30.0  # Default rate limit
        self.last_request_time = {}
        self.max_concurrent_requests = 30  # Default concurrent requests
        self.executor = None  # Thread pool executor
        init_database(db_path)
    
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
    
    def add_combination(self, combination_id):
        """Add a combination to the processing queue"""
        if combination_id not in self.running_combinations and combination_id not in self.job_queue:
            self.job_queue.append(combination_id)
            self.process_queue()
    
    def pause_combination(self, combination_id):
        """Pause a running combination"""
        self.paused_combinations.add(combination_id)
        
    def resume_combination(self, combination_id):
        """Resume a paused combination"""
        self.paused_combinations.discard(combination_id)
        
    def stop_combination(self, combination_id):
        """Stop a running combination"""
        self.stopped_combinations.add(combination_id)
        if combination_id in self.running_combinations:
            # The processing thread will check this and stop
            pass
            
    def process_queue(self):
        """Process combinations in the queue"""
        if not self.job_queue or len(self.running_combinations) >= 1:  # Process one at a time for now
            return
            
        combination_id = self.job_queue.pop(0)
        thread = threading.Thread(target=self.process_combination, args=(combination_id,))
        thread.daemon = True
        thread.start()
        self.running_combinations[combination_id] = thread

    def process_combination(self, combination_id):
        """Process a single combination with proper error handling and status updates"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Update status to running
            cursor.execute("UPDATE combinations SET status = 'running', started_at = ? WHERE id = ?", 
                         (datetime.now(), combination_id))
            conn.commit()
            
            # Get combination details
            cursor.execute("SELECT * FROM combinations WHERE id = ?", (combination_id,))
            combination = cursor.fetchone()
            
            if not combination:
                print(f"ERROR: Combination {combination_id} not found in database")
                return
                
            species_file = combination[1]
            model = combination[2]
            system_template = combination[3]
            user_template = combination[4]
            
            print(f"Processing combination {combination_id}:")
            print(f"  Species file: {species_file}")
            print(f"  Model: {model}")
            print(f"  System template: {system_template}")
            print(f"  User template: {user_template}")
            
            # Read species file and templates
            try:
                # Check if species file exists
                if not Path(species_file).exists():
                    # Try to resolve relative to project root
                    from microbellm.shared import PROJECT_ROOT
                    species_path = PROJECT_ROOT / config.SPECIES_DIR / Path(species_file).name
                    if species_path.exists():
                        species_file = str(species_path)
                        print(f"  Resolved species file path to: {species_file}")
                    else:
                        raise FileNotFoundError(f"Species file not found: {species_file}")
                
                # Try to read the species file with different separators
                # First, try to detect the separator by reading the first line
                with open(species_file, 'r') as f:
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
                        df = pd.read_csv(species_file, sep=sep)
                        print(f"  Read species file with {sep_name} separator")
                    else:
                        # Let pandas auto-detect
                        df = pd.read_csv(species_file)
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
            
            for i, species in enumerate(species_list):
                # Check if stopped
                if combination_id in self.stopped_combinations:
                    cursor.execute("UPDATE combinations SET status = 'interrupted' WHERE id = ?", 
                                 (combination_id,))
                    conn.commit()
                    break
                    
                # Check if paused
                while combination_id in self.paused_combinations:
                    time.sleep(1)
                    
                # Submit job to executor
                future = executor.submit(self._process_single_species, 
                                       species, model, system_prompt, user_prompt, 
                                       combination_id)
                futures[future] = species
                
                # Process completed futures
                completed_futures = []
                for future in futures:
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
                
                # Emit progress update
                submitted = len(futures) + successful + failed + timeouts
                socketio.emit('progress_update', {
                    'combination_id': combination_id,
                    'submitted': submitted,
                    'total': total_species,
                    'successful': successful,
                    'failed': failed,
                    'timeouts': timeouts
                })
                
                # Limit number of concurrent futures
                while len(futures) >= self.max_concurrent_requests:
                    time.sleep(0.1)
                    # Check for completed futures again
                    for future in list(futures.keys()):
                        if future.done():
                            species_name = futures.pop(future)
                            try:
                                result, status, error = future.result(timeout=1)
                                
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
                                
                                print(f"    Stored result for {species_name}: status={status}")
                                
                                if status == 'completed':
                                    successful += 1
                                elif status == 'timeout':
                                    timeouts += 1
                                else:
                                    failed += 1
                            except Exception as e:
                                failed += 1
                                
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
        
        print(f"  Processing species: {species} with model: {model}")
        
        try:
            # Make prediction
            result = predict_binomial_name(
                binomial_name=species,
                system_template=system_prompt,
                user_template=user_prompt,
                model=model,
                verbose=True  # Enable verbose logging
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
        """Apply rate limiting per model"""
        if self.requests_per_second <= 0:
            return
            
        min_interval = 1.0 / self.requests_per_second
        
        # Thread-safe rate limiting
        with threading.Lock():
            current_time = time.time()
            last_time = self.last_request_time.get(model, 0)
            elapsed = current_time - last_time
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
                
            self.last_request_time[model] = time.time()
    
    def set_rate_limit(self, requests_per_second):
        """Set the rate limit for API requests"""
        self.requests_per_second = max(0.1, requests_per_second)
    
    def start_combination(self, combination_id):
        """Start processing a combination"""
        print(f"\n=== Starting combination {combination_id} ===")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if combination exists and is not already running
        cursor.execute("SELECT status, species_file, model, system_template, user_template FROM combinations WHERE id = ?", (combination_id,))
        result = cursor.fetchone()
        
        if not result:
            print(f"ERROR: Combination {combination_id} not found")
            conn.close()
            return False
            
        status, species_file, model, sys_tmpl, usr_tmpl = result
        print(f"Current status: {status}")
        print(f"Species file: {species_file}")
        print(f"Model: {model}")
        
        if status == 'running':
            print(f"WARNING: Combination {combination_id} is already running")
            conn.close()
            return False
        
        # Reset status to pending and add to queue
        cursor.execute("UPDATE combinations SET status = 'pending' WHERE id = ?", (combination_id,))
        conn.commit()
        conn.close()
        
        print(f"Added combination {combination_id} to processing queue")
        # Add to processing queue
        self.add_combination(combination_id)
        return True
    
    def restart_combination(self, combination_id):
        """Restart a combination (retry failed species)"""
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
                return False, "Combination not found"
            
            # Delete failed results from results table
            cursor.execute(
                """DELETE FROM results 
                   WHERE species_file = ? AND model = ? AND system_template = ? 
                   AND user_template = ? AND status != 'completed'""",
                (combo[0], combo[1], combo[2], combo[3])
            )
            
            # Reset combination status
            cursor.execute(
                "UPDATE combinations SET status = 'pending', completed_at = NULL WHERE id = ?",
                (combination_id,)
            )
            conn.commit()
            
            # Add to processing queue
            self.add_combination(combination_id)
            return True, "Combination restarted successfully"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()
    
    def get_dashboard_data(self):
        """Get data for the dashboard matrix view"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all combinations - using the existing columns in the combinations table
        cursor.execute('''
            SELECT 
                id, species_file, model, system_template, user_template, status,
                total_species, successful_species, failed_species, timeout_species,
                submitted_species, received_species
            FROM combinations
        ''')
        
        combinations = []
        for row in cursor.fetchall():
            combinations.append({
                'id': row[0],
                'species_file': row[1],
                'model': row[2],
                'system_template': row[3],
                'user_template': row[4],
                'status': row[5],
                'total': row[6] or 0,
                'successful': row[7] or 0,
                'failed': row[8] or 0,
                'timeouts': row[9] or 0,
                'submitted': row[10] or 0,
                'received': row[11] or 0
            })
        
        # Get unique models from both combinations and managed_models
        cursor.execute("""
            SELECT DISTINCT model FROM (
                SELECT DISTINCT model FROM combinations
                UNION
                SELECT model FROM managed_models
            ) ORDER BY model
        """)
        models = [row[0] for row in cursor.fetchall()]
        
        # Get unique species files from both combinations and managed_species_files
        # Also include files from results table for historical data
        cursor.execute("""
            SELECT DISTINCT species_file FROM (
                SELECT DISTINCT species_file FROM combinations
                UNION
                SELECT species_file FROM managed_species_files
                UNION
                SELECT DISTINCT species_file FROM results
            ) ORDER BY species_file
        """)
        
        # Normalize paths to avoid duplicates
        seen_files = set()
        species_files = []
        for row in cursor.fetchall():
            file_path = row[0]
            # Get just the filename for comparison
            file_name = Path(file_path).name
            if file_name not in seen_files:
                seen_files.add(file_name)
                species_files.append(file_path)
        
        # Get template display info from all available templates, not just used ones
        template_info = []
        available_templates = get_available_template_pairs()
        
        # First add all available templates
        for template_pair in available_templates:
            template_info.append({
                'system_template': template_pair['system'],
                'user_template': template_pair['user'],
                'display_name': template_pair['name'],
                'description': f'Template: {template_pair["name"]}',
                'template_type': detect_template_type(template_pair['user']) if template_pair['user'] else 'unknown'
            })
        
        # Also check if there are any templates in combinations that aren't in the filesystem
        cursor.execute("SELECT DISTINCT system_template, user_template FROM combinations")
        for row in cursor.fetchall():
            # Check if this template is already in our list
            found = False
            for t in template_info:
                if t['system_template'] == row[0] and t['user_template'] == row[1]:
                    found = True
                    break
            if not found:
                template_info.append({
                    'system_template': row[0],
                    'user_template': row[1],
                    'display_name': Path(row[0]).stem,
                    'description': f'Template: {Path(row[0]).stem} (from DB)',
                    'template_type': detect_template_type(row[1]) if row[1] else 'unknown'
                })
        
        # Create matrix structure
        matrix = {}
        
        # Debug: Print what we have
        
        # Helper function to normalize paths for consistent keys
        def normalize_key_path(species_file, sys_tmpl, usr_tmpl):
            # Normalize species file to just the filename
            species_name = Path(species_file).name
            # Normalize template paths to just the filename
            sys_name = Path(sys_tmpl).name if sys_tmpl else sys_tmpl
            usr_name = Path(usr_tmpl).name if usr_tmpl else usr_tmpl
            return f"{species_name}|{sys_name}|{usr_name}"
        
        # First, add data from combinations table
        for combo in combinations:
            # Use normalized key
            key = normalize_key_path(combo['species_file'], combo['system_template'], combo['user_template'])
            if key not in matrix:
                matrix[key] = {'models': {}}
            matrix[key]['models'][combo['model']] = {
                'id': combo['id'],
                'status': combo['status'],
                'total': combo['total'],
                'successful': combo['successful'],
                'failed': combo['failed'],
                'timeouts': combo['timeouts'],
                'submitted': combo['submitted'] if combo['submitted'] else (combo['successful'] + combo['failed'] + combo['timeouts'])
            }
        
        # Also check the results table for historical data
        # Only add results that don't already have a combination entry
        cursor.execute("""
            SELECT r.species_file, r.model, r.system_template, r.user_template, 
                   COUNT(*) as total,
                   SUM(CASE WHEN r.status = 'completed' OR r.status IS NULL THEN 1 ELSE 0 END) as successful,
                   SUM(CASE WHEN r.status = 'failed' THEN 1 ELSE 0 END) as failed,
                   SUM(CASE WHEN r.status = 'timeout' THEN 1 ELSE 0 END) as timeouts
            FROM results r
            WHERE NOT EXISTS (
                SELECT 1 FROM combinations c 
                WHERE c.species_file = r.species_file 
                AND c.model = r.model 
                AND c.system_template = r.system_template 
                AND c.user_template = r.user_template
            )
            GROUP BY r.species_file, r.model, r.system_template, r.user_template
        """)
        
        # Add historical results to matrix if not already present
        for row in cursor.fetchall():
            species_file, model, sys_tmpl, usr_tmpl, total, successful, failed, timeouts = row
            key = f"{species_file}|{sys_tmpl}|{usr_tmpl}"
            
            if key not in matrix:
                matrix[key] = {'models': {}}
            
            # Only add if this model isn't already in the matrix for this key
            if model not in matrix[key]['models']:
                matrix[key]['models'][model] = {
                    'id': None,  # No combination ID for historical data
                    'status': 'completed',  # Historical data is completed
                    'total': total,
                    'successful': successful or 0,
                    'failed': failed or 0,
                    'timeouts': timeouts or 0,
                    'submitted': total,
                    'historical': True,  # Mark as historical data
                    'orphaned': True  # Mark as orphaned (no combination)
                }
            
            # Also make sure this model is in our models list
            if model not in models:
                models.append(model)
        
        # Re-sort models
        models.sort()
        
        conn.close()
        
        return {
            'combinations': combinations,
            'models': models,
            'species_files': species_files,
            'template_display_info': template_info,
            'matrix': matrix
        }
    
    def get_combination_details(self, combination_id):
        """Get detailed information about a combination"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get combination info with all columns
        cursor.execute("""
            SELECT id, species_file, model, system_template, user_template, status,
                   total_species, successful_species, failed_species, timeout_species
            FROM combinations WHERE id = ?
        """, (combination_id,))
        combo = cursor.fetchone()
        
        if not combo:
            conn.close()
            return jsonify({'error': 'Combination not found'}), 404
        
        # Get species results from the results table (single source of truth)
        cursor.execute("""
            SELECT binomial_name, result, status, error, knowledge_group,
                   gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                   biofilm_formation, animal_pathogenicity, biosafety_level,
                   health_association, host_association, plant_pathogenicity,
                   spore_formation, hemolysis, cell_shape
            FROM results 
            WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
        """, (combo[1], combo[2], combo[3], combo[4]))
        species_results_raw = cursor.fetchall()
        from_results_table = True
        
        species_results = []
        for row in species_results_raw:
            if from_results_table:
                # Results table has more columns
                result_data = {
                    'binomial_name': row[0],
                    'result': row[1],
                    'status': row[2] or 'completed',  # Default to completed if no status
                    'error': row[3],
                    'knowledge_group': row[4] if len(row) > 4 else None
                }
                
                # Collect phenotypes from individual columns if available
                if len(row) > 5:
                    phenotypes = {}
                    phenotype_columns = [
                        ('gram_staining', row[5]),
                        ('motility', row[6]),
                        ('aerophilicity', row[7]),
                        ('extreme_environment_tolerance', row[8]),
                        ('biofilm_formation', row[9]),
                        ('animal_pathogenicity', row[10]),
                        ('biosafety_level', row[11]),
                        ('health_association', row[12]),
                        ('host_association', row[13]),
                        ('plant_pathogenicity', row[14]),
                        ('spore_formation', row[15]),
                        ('hemolysis', row[16]),
                        ('cell_shape', row[17])
                    ]
                    
                    for col_name, value in phenotype_columns:
                        if value and value.strip():
                            phenotypes[col_name] = value
                    
                    if phenotypes:
                        result_data['phenotypes'] = phenotypes
            else:
                # species_results table has fewer columns
                result_data = {
                    'binomial_name': row[0],
                    'result': row[1],
                    'status': row[2] or 'completed',
                    'error': row[3]
                }
                
                # Try to parse result JSON if available
                if row[1]:
                    try:
                        parsed_result = json.loads(row[1])
                        if isinstance(parsed_result, dict):
                            # Extract any fields from the parsed result
                            if 'knowledge_group' in parsed_result:
                                result_data['knowledge_group'] = parsed_result['knowledge_group']
                            if 'phenotypes' in parsed_result:
                                result_data['phenotypes'] = parsed_result['phenotypes']
                    except json.JSONDecodeError:
                        pass
            
            species_results.append(result_data)
        
        conn.close()
        
        # Use the stored counts from the combination table
        return jsonify({
            'combination_info': {
                'id': combo[0],
                'species_file': combo[1],
                'model': combo[2],
                'system_template': combo[3],
                'user_template': combo[4],
                'status': combo[5],
                'total_species': combo[6] or 0,
                'successful_species': combo[7] or 0,
                'failed_species': combo[8] or 0,
                'timeout_species': combo[9] or 0
            },
            'species_results': species_results
        })
    
    def create_combinations(self, species_file, models, template_pairs):
        """Create new combinations for processing"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        created = []
        
        # Resolve full path for species file
        from microbellm.shared import PROJECT_ROOT
        species_path = Path(species_file)
        if not species_path.is_absolute():
            # Try to find the file in the species directory
            species_dir = PROJECT_ROOT / config.SPECIES_DIR
            full_path = species_dir / species_file
            if full_path.exists():
                species_file = str(full_path)
                print(f"Resolved species file to: {species_file}")
            else:
                print(f"WARNING: Could not find species file: {species_file} in {species_dir}")
        
        try:
            for model in models:
                for pair in template_pairs:
                    # Check if combination already exists
                    cursor.execute(
                        """SELECT id FROM combinations 
                           WHERE species_file = ? AND model = ? 
                           AND system_template = ? AND user_template = ?""",
                        (species_file, model, pair['system'], pair['user'])
                    )
                    
                    existing = cursor.fetchone()
                    if not existing:
                        # Create new combination
                        cursor.execute(
                            """INSERT INTO combinations 
                               (species_file, model, system_template, user_template, status)
                               VALUES (?, ?, ?, ?, 'pending')""",
                            (species_file, model, pair['system'], pair['user'])
                        )
                        
                        created.append({
                            'id': cursor.lastrowid,
                            'species_file': species_file,
                            'model': model,
                            'system_template': pair['system'],
                            'user_template': pair['user']
                        })
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        
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
        return jsonify({'success': True, 'message': f'Combination {combination_id} started successfully.'})
    else:
        return jsonify({'success': False, 'error': 'Failed to start combination'}), 500

@app.route('/api/restart_combination/<int:combination_id>', methods=['POST'])
def restart_combination_api(combination_id):
    success, message = processing_manager.restart_combination(combination_id)
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'error': message}), 500

@app.route('/api/pause_combination/<int:combination_id>', methods=['POST'])
def pause_combination_api(combination_id):
    success = processing_manager.pause_combination(combination_id)
    if success:
        return jsonify({'success': True, 'message': 'Combination paused'})
    else:
        return jsonify({'success': False, 'error': 'Failed to pause combination'}), 500

@app.route('/api/stop_combination/<int:combination_id>', methods=['POST'])
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

@app.route('/api/delete_combination/<int:combination_id>', methods=['DELETE'])
def delete_combination_api(combination_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get combination details first
        cursor.execute("""
            SELECT species_file, model, system_template, user_template
            FROM combinations WHERE id = ?
        """, (combination_id,))
        combo = cursor.fetchone()
        
        if combo:
            # Delete from results table
            cursor.execute("""
                DELETE FROM results 
                WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
            """, (combo[0], combo[1], combo[2], combo[3]))
        
        # Delete combination
        cursor.execute("DELETE FROM combinations WHERE id = ?", (combination_id,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Combination deleted successfully'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()

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

@app.route('/api/create_combination', methods=['POST'])
def create_combination_api():
    data = request.get_json()
    species_file = data.get('species_file')
    model = data.get('model')
    system_template = data.get('system_template')
    user_template = data.get('user_template')
    
    if not all([species_file, model, system_template, user_template]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    try:
        # Create template pair
        template_pairs = [{
            'system': system_template,
            'user': user_template
        }]
        
        # Create combination
        created_combinations = processing_manager.create_combinations(species_file, [model], template_pairs)
        
        if created_combinations:
            combination_id = created_combinations[0]['id']
            return jsonify({
                'success': True,
                'combination_id': combination_id,
                'message': 'Combination created successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create combination'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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

@app.route('/api/combination_details/<int:combination_id>')
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

@app.route('/api/reparse_phenotype_data/<int:combination_id>', methods=['POST'])
def reparse_phenotype_data(combination_id):
    """Re-parse phenotype data for a combination"""
    return processing_manager.reparse_phenotype_data(combination_id)

@app.route('/api/rerun_all_failed/<int:combination_id>', methods=['POST'])
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