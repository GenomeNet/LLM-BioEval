#!/usr/bin/env python

import os
import io
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from flask_socketio import SocketIO, emit
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from microbellm.utils import read_template_from_file
from microbellm.predict import predict_binomial_name
import sqlite3
from pathlib import Path
from microbellm import config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'microbellm-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for job management
job_manager = None
db_path = config.DATABASE_PATH

def reset_running_jobs_on_startup():
    """Set status of 'running' jobs to 'interrupted' on application start."""
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

class ProcessingManager:
    def __init__(self):
        self.running_combinations = {}
        self.executor = None
        self.futures = {}
        self.paused_combinations = set()  # Track paused jobs
        self.stopped_combinations = set()  # Track stopped jobs
        self.job_queue = []  # Queue for jobs waiting to start
        self.max_concurrent_jobs = 1  # Default: only one job at a time
        self.requests_per_second = 2  # Default rate limit: 2 requests/second
        self.last_request_time = {}  # Track last request time per combination
        self.init_database()
    
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
                cell_shape TEXT
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
            ('cell_shape', 'TEXT')
        ]
        
        for column_name, column_type in phenotype_columns:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE results ADD COLUMN {column_name} {column_type}')
                print(f"Added column {column_name} to results table")
        
        # Migration: Add new progress tracking columns if they don't exist
        cursor.execute("PRAGMA table_info(combinations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        progress_columns = [
            ('submitted_species', 'INTEGER DEFAULT 0'),
            ('received_species', 'INTEGER DEFAULT 0'),
            ('successful_species', 'INTEGER DEFAULT 0'),
            ('retried_species', 'INTEGER DEFAULT 0')
        ]
        
        for column_name, column_type in progress_columns:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE combinations ADD COLUMN {column_name} {column_type}')
                print(f"Added column {column_name} to combinations table")
        
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
        """Read species list from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                species = [line.strip() for line in file.readlines() if line.strip()]
                return species
        except Exception as e:
            print(f"Error reading species file {file_path}: {e}")
            return []
    
    def start_combination(self, combination_id):
        """Start processing a specific combination"""
        if combination_id in self.running_combinations:
            return False  # Already running
        
        # Remove from paused/stopped sets if it was there
        self.paused_combinations.discard(combination_id)
        self.stopped_combinations.discard(combination_id)
        
        # Check if we can start immediately or need to queue
        if not self.can_start_new_job():
            if combination_id not in self.job_queue:
                self.job_queue.append(combination_id)
                self._emit_log(combination_id, f"Job queued - waiting for slot. Queue position: {len(self.job_queue)}")
                socketio.emit('job_update', {'combination_id': combination_id, 'status': 'queued'})
            return True
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get combination details
        cursor.execute('''
            SELECT species_file, model, system_template, user_template 
            FROM combinations WHERE id = ?
        ''', (combination_id,))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False
        
        species_file, model, system_template, user_template = result
        
        # Update status to running
        cursor.execute('''
            UPDATE combinations SET status = ?, started_at = ? WHERE id = ?
        ''', ('running', datetime.now(), combination_id))
        conn.commit()
        conn.close()
        
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
        """The actual processing logic for a combination with smart resume and rate limiting."""
        
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

            system_template_content = read_template_from_file(system_template)
            user_template_content = read_template_from_file(user_template)

            for i, species_name in enumerate(remaining_species):
                # Check if job was paused or stopped
                if combination_id in self.paused_combinations:
                    self._emit_log(combination_id, "Job paused.")
                    break
                
                if combination_id in self.stopped_combinations:
                    self._emit_log(combination_id, "Job stopped.")
                    break
                
                self._emit_log(combination_id, f"Processing species {len(already_processed) + i + 1}/{len(species_list)}: {species_name}")

                # Apply rate limiting
                self._wait_for_rate_limit(combination_id)

                # Track that we're submitting this species to the LLM
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("UPDATE combinations SET submitted_species = submitted_species + 1 WHERE id = ?", (combination_id,))
                conn.commit()
                conn.close()

                try:
                    result = predict_binomial_name(
                        species_name,
                        system_template_content,
                        user_template_content,
                        model,
                        temperature=0.0,
                        verbose=False
                    )

                    self._process_species_result(combination_id, species_file, species_name, model, 
                                               system_template, user_template, result)

                except Exception as e:
                    self._process_species_error(combination_id, species_file, species_name, model, 
                                              system_template, user_template, str(e))

                # Emit detailed progress update
                self._emit_progress_update(combination_id)
                time.sleep(0.1)

            # Final status update
            self._finalize_combination(combination_id)
            
        except Exception as e:
            self._emit_log(combination_id, f"Critical error in processing: {str(e)}")
            self._mark_combination_failed(combination_id)
        finally:
            # Clean up and start next job in queue
            self._cleanup_completed_job(combination_id)

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
        """Emit a log message to the client."""
        socketio.emit('log_message', {
            'combination_id': combination_id,
            'log': f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        })

    def _process_species_result(self, combination_id, species_file, species_name, model, system_template, user_template, result):
        """Process the result from a species prediction"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if result:
            # We received a response from the LLM
            cursor.execute("UPDATE combinations SET received_species = received_species + 1 WHERE id = ?", (combination_id,))
            
            # Check if the result contains valid phenotype data
            has_phenotype_data = any([
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

            if has_phenotype_data:
                # Successfully parsed result with actual phenotype data
                self._emit_log(combination_id, f"✓ Successfully processed {species_name} with phenotype data")
                cursor.execute("UPDATE combinations SET successful_species = successful_species + 1, completed_species = completed_species + 1 WHERE id = ?", (combination_id,))
                result_status = 'completed'
            else:
                # Got a response but no valid phenotype data
                self._emit_log(combination_id, f"⚠ Received response for {species_name} but no valid phenotype data extracted")
                cursor.execute("UPDATE combinations SET failed_species = failed_species + 1 WHERE id = ?", (combination_id,))
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
                'timestamp': datetime.now(),
                'gram_staining': result.get('gram_staining'),
                'motility': result.get('motility'),
                'aerophilicity': result.get('aerophilicity'),
                'extreme_environment_tolerance': result.get('extreme_environment_tolerance'),
                'biofilm_formation': result.get('biofilm_formation'),
                'animal_pathogenicity': result.get('animal_pathogenicity'),
                'biosafety_level': result.get('biosafety_level'),
                'health_association': result.get('health_association'),
                'host_association': result.get('host_association'),
                'plant_pathogenicity': result.get('plant_pathogenicity'),
                'spore_formation': result.get('spore_formation'),
                'hemolysis': result.get('hemolysis'),
                'cell_shape': result.get('cell_shape')
            }
            
            cursor.execute('''
                INSERT INTO results (species_file, binomial_name, model, system_template, user_template, status, result, error, timestamp, gram_staining, motility, aerophilicity, extreme_environment_tolerance, biofilm_formation, animal_pathogenicity, biosafety_level, health_association, host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape)
                VALUES (:species_file, :binomial_name, :model, :system_template, :user_template, :status, :result, :error, :timestamp, :gram_staining, :motility, :aerophilicity, :extreme_environment_tolerance, :biofilm_formation, :animal_pathogenicity, :biosafety_level, :health_association, :host_association, :plant_pathogenicity, :spore_formation, :hemolysis, :cell_shape)
            ''', result_data)

        else:
            # No response from API at all
            self._emit_log(combination_id, f"✗ Failed to get any response for {species_name}")
            cursor.execute('''
                INSERT INTO results (species_file, binomial_name, model, system_template, user_template, status, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (species_file, species_name, model, system_template, user_template, 'failed', 'No response from API', datetime.now()))
            cursor.execute("UPDATE combinations SET failed_species = failed_species + 1 WHERE id = ?", (combination_id,))

        conn.commit()
        conn.close()

    def _process_species_error(self, combination_id, species_file, species_name, model, system_template, user_template, error_message):
        """Process an error during species prediction"""
        self._emit_log(combination_id, f"✗ Exception processing {species_name}: {error_message}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results (species_file, binomial_name, model, system_template, user_template, status, error, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (species_file, species_name, model, system_template, user_template, 'failed', error_message, datetime.now()))
        cursor.execute("UPDATE combinations SET failed_species = failed_species + 1 WHERE id = ?", (combination_id,))
        conn.commit()
        conn.close()

    def _emit_progress_update(self, combination_id):
        """Emit detailed progress update"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT total_species, submitted_species, received_species, successful_species, failed_species, completed_species 
            FROM combinations WHERE id = ?
        """, (combination_id,))
        progress = cursor.fetchone()
        conn.close()

        if progress:
            socketio.emit('job_update', {
                'combination_id': combination_id, 
                'status': 'running',
                'total': progress[0],
                'submitted': progress[1],
                'received': progress[2],
                'successful': progress[3],
                'failed': progress[4],
                'completed': progress[5]
            })

    def _finalize_combination(self, combination_id):
        """Finalize combination status based on results"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT successful_species, total_species FROM combinations WHERE id = ?", (combination_id,))
        final_counts = cursor.fetchone()
        
        if combination_id in self.paused_combinations:
            final_status = 'interrupted'
            self._emit_log(combination_id, f"Job paused - processed {final_counts[0]} out of {final_counts[1]} species successfully")
        elif combination_id in self.stopped_combinations:
            final_status = 'interrupted'
            self._emit_log(combination_id, f"Job stopped - processed {final_counts[0]} out of {final_counts[1]} species successfully")
        elif final_counts[0] > 0:
            final_status = 'completed'
            self._emit_log(combination_id, f"Job completed - processed {final_counts[0]} out of {final_counts[1]} species successfully")
        else:
            final_status = 'failed'
            self._emit_log(combination_id, f"Job failed - no species were processed successfully out of {final_counts[1]} total")
        
        cursor.execute("UPDATE combinations SET status = ?, completed_at = ? WHERE id = ?", (final_status, datetime.now(), combination_id))
        conn.commit()
        conn.close()
        
        socketio.emit('job_update', {'combination_id': combination_id, 'status': final_status})

    def get_dashboard_data(self):
        """Get data for the dashboard matrix view"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all combinations
        cursor.execute('''
            SELECT id, species_file, model, system_template, user_template, 
                   status, total_species, completed_species, failed_species,
                   submitted_species, received_species, successful_species, retried_species
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
                'total_species': row[6],
                'completed_species': row[7],
                'failed_species': row[8],
                'submitted_species': row[9] or 0,
                'received_species': row[10] or 0,
                'successful_species': row[11] or 0,
                'retried_species': row[12] or 0
            })
        
        conn.close()
        
        # Get managed models and species files to ensure they appear in the UI
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT model FROM managed_models")
        managed_models = set([row[0] for row in cursor.fetchall()])
        
        cursor.execute("SELECT species_file FROM managed_species_files")
        managed_species = set([row[0] for row in cursor.fetchall()])
        
        conn.close()

        if not combinations and not managed_models and not managed_species:
            return {
                'matrix': {},
                'species_files': [],
                'models': [],
                'template_combinations': []
            }
        
        # Organize data for template-based matrix view
        species_from_combinations = set([c['species_file'] for c in combinations])
        all_species_files = sorted(list(species_from_combinations.union(managed_species)))

        models_from_combinations = set([c['model'] for c in combinations])
        all_models = sorted(list(models_from_combinations.union(managed_models)))
        
        # Get unique template combinations
        template_combinations = list(set([(c['system_template'], c['user_template']) for c in combinations]))
        
        # Create matrix organized by template combinations
        matrix = {}
        
        for species_file in all_species_files:
            for sys_template, user_template in template_combinations:
                row_key = f"{species_file}|{sys_template}|{user_template}"
                template_label = f"{sys_template.split('/')[-1].replace('.txt', '')} + {user_template.split('/')[-1].replace('.txt', '')}"
                
                matrix[row_key] = {
                    'species_file': species_file,
                    'template_label': template_label,
                    'models': {}
                }
                
                for model in all_models:
                    # Find specific combination
                    combo = next((c for c in combinations 
                                if c['species_file'] == species_file and c['model'] == model 
                                and c['system_template'] == sys_template and c['user_template'] == user_template), None)
                    
                    if combo:
                        # Check if this combination was imported (no submitted_species means it was imported)
                        was_imported = combo['submitted_species'] == 0 and combo['successful_species'] > 0
                        
                        matrix[row_key]['models'][model] = {
                            'id': combo['id'],
                            'completed': combo['completed_species'],
                            'total': combo['total_species'],
                            'status': combo['status'],
                            'submitted': combo['submitted_species'],
                            'received': combo['received_species'],
                            'successful': combo['successful_species'],
                            'failed': combo['failed_species'],
                            'retried': combo['retried_species'],
                            'imported': was_imported
                        }
                    else:
                        matrix[row_key]['models'][model] = None
        
        return {
            'matrix': matrix,
            'species_files': all_species_files,
            'models': all_models,
            'template_combinations': sorted(template_combinations)
        }
    
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
                        'cell_shape': row.get('cell_shape', '').strip() or None
                    }
                    
                    # Derive user_template from system_template (assuming same filename pattern)
                    user_template = system_template.replace('/system/', '/user/')
                    
                    # Check if entry already exists
                    cursor.execute('''
                        SELECT id, gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                               biofilm_formation, animal_pathogenicity, biosafety_level, health_association,
                               host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape
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
                            'cell_shape': existing[13]
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
                                hemolysis = ?, cell_shape = ?, timestamp = ?, status = "completed"
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
                                               spore_formation, hemolysis, cell_shape)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            phenotype_data['cell_shape']
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
        self.requests_per_second = max(0.1, min(10.0, requests_per_second))  # Clamp between 0.1 and 10 RPS
        print(f"Rate limit set to {self.requests_per_second} requests per second")
    
    def can_start_new_job(self):
        """Check if we can start a new job based on concurrent job limit"""
        return len(self.running_combinations) < self.max_concurrent_jobs
    
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
    """Main dashboard"""
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
    try:
        data = request.get_json()
        rate_limit = float(data.get('requests_per_second', 2.0))
        processing_manager.set_rate_limit(rate_limit)
        return jsonify({'message': f'Rate limit set to {processing_manager.requests_per_second} requests per second'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_settings')
def get_settings_api():
    return jsonify({
        'rate_limit': processing_manager.requests_per_second,
        'max_concurrent_jobs': processing_manager.max_concurrent_jobs,
        'queue_length': len(processing_manager.job_queue)
    })

@app.route('/api/dashboard_data')
def dashboard_data_api():
    """Get dashboard data via API"""
    data = processing_manager.get_dashboard_data()
    return jsonify(data)

@app.route('/api/delete_combination/<int:combination_id>', methods=['DELETE'])
def delete_combination_api(combination_id):
    """API endpoint to delete a combination and its results."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get combination details to find associated results
        cursor.execute('''
            SELECT species_file, model, system_template, user_template 
            FROM combinations WHERE id = ?
        ''', (combination_id,))
        combo = cursor.fetchone()

        # Delete from combinations table
        cursor.execute("DELETE FROM combinations WHERE id = ?", (combination_id,))
        
        if combo:
            # Delete associated results
            cursor.execute('''
                DELETE FROM results 
                WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
            ''', (combo[0], combo[1], combo[2], combo[3]))

        conn.commit()
        conn.close()
        
        socketio.emit('job_update', {'combination_id': combination_id, 'status': 'deleted'})
        return jsonify({'message': f'Combination {combination_id} and its results have been deleted.'})
    except Exception as e:
        return jsonify({'message': f'Error deleting combination: {e}'}), 500

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

@app.route('/templates')
def templates_page():
    """Template viewing page to display user and system templates side by side"""
    template_pairs = get_available_template_pairs()
    
    # Get template contents for each pair
    template_data = {}
    for template_name, paths in template_pairs.items():
        try:
            with open(paths['system'], 'r', encoding='utf-8') as f:
                system_content = f.read()
            with open(paths['user'], 'r', encoding='utf-8') as f:
                user_content = f.read()
            
            template_data[template_name] = {
                'system': {
                    'path': paths['system'],
                    'content': system_content
                },
                'user': {
                    'path': paths['user'],
                    'content': user_content
                }
            }
        except Exception as e:
            print(f"Error reading template {template_name}: {e}")
            continue
    
    return render_template('view_template.html', template_data=template_data)

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

@app.route('/api/export_csv')
def export_csv_api():
    """Export results as CSV"""
    species_file = request.args.get('species_file')
    model = request.args.get('model')
    
    try:
        csv_content = processing_manager.export_results_to_csv(species_file, model)
        
        # Generate filename
        filename_parts = ['microbellm_results']
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
        
        if not all([species_file, model, system_template, user_template]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Create the combination using existing logic
        template_pairs = {
            'custom': {
                'system': system_template,
                'user': user_template
            }
        }
        
        created_combinations = processing_manager.create_combinations(species_file, [model], template_pairs)
        
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
                # Start the combination immediately
                if processing_manager.start_combination(combination_id):
                    return jsonify({'success': True, 'combination_id': combination_id})
                else:
                    return jsonify({'success': True, 'combination_id': combination_id, 'note': 'Created but could not start immediately'})
            else:
                return jsonify({'error': 'Combination created but could not retrieve ID'}), 500
        else:
            return jsonify({'error': 'Failed to create combination - may already exist'}), 400
            
    except Exception as e:
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
        
        # Get combination info
        cursor.execute("""
            SELECT species_file, model, system_template, user_template, status, 
                   total_species, submitted_species, received_species, successful_species, failed_species
            FROM combinations WHERE id = ?
        """, (combination_id,))
        
        combo_info = cursor.fetchone()
        if not combo_info:
            return jsonify({'error': 'Combination not found'}), 404
            
        # Get detailed results for each species
        cursor.execute("""
            SELECT binomial_name, status, result, error, timestamp,
                   gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                   biofilm_formation, animal_pathogenicity, biosafety_level, health_association,
                   host_association, plant_pathogenicity, spore_formation, hemolysis, cell_shape
            FROM results 
            WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
            ORDER BY timestamp
        """, (combo_info[0], combo_info[1], combo_info[2], combo_info[3]))
        
        species_results = []
        for row in cursor.fetchall():
            species_results.append({
                'binomial_name': row[0],
                'status': row[1],
                'result': row[2],
                'error': row[3],
                'timestamp': row[4],
                'phenotypes': {
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
            })
        
        # Get list of all species that should have been processed
        species_file_path = os.path.join(config.SPECIES_DIR, combo_info[0])
        try:
            with open(species_file_path, 'r', encoding='utf-8') as file:
                all_species = [line.strip() for line in file.readlines() if line.strip()]
        except:
            all_species = []
        
        # Find species that weren't processed yet
        processed_species = {result['binomial_name'] for result in species_results}
        unprocessed_species = [species for species in all_species if species not in processed_species]
        
        conn.close()
        
        return jsonify({
            'combination_info': {
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
                'failed_species': combo_info[9] or 0
            },
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

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def main():
    """Main entry point for the web application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MicrobeLLM Web Interface")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check environment
    if not os.getenv('OPENROUTER_API_KEY'):
        print("Warning: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-api-key'")
    
    print(f"Starting MicrobeLLM Web Interface on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()

# Reset job statuses on startup
reset_running_jobs_on_startup()
job_manager = ProcessingManager() 