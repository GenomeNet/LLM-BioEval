"""
Unified database interface for single-table architecture.
This module provides all database operations for the unified processing_results table.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple

from microbellm.shared import get_db_connection, DATABASE_PATH


class UnifiedDB:
    """Database interface for unified processing_results table"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_PATH
    
    def create_job(self, species_file: str, model: str, system_template: str, 
                   user_template: str, species_list: List[str]) -> str:
        """
        Create a new processing job with all species entries.
        Returns the job_id.
        """
        job_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert all species for this job
            for species in species_list:
                cursor.execute("""
                    INSERT INTO processing_results 
                    (job_id, species_file, model, system_template, user_template, 
                     binomial_name, status, job_status)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', 'pending')
                """, (job_id, species_file, model, system_template, user_template, species))
            
            conn.commit()
            return job_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_job_summary(self, job_id: str) -> Optional[Dict]:
        """Get summary statistics for a job"""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                job_id, species_file, model, system_template, user_template,
                job_status, job_created_at, job_started_at, job_completed_at,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) as timeouts
            FROM processing_results
            WHERE job_id = ?
            GROUP BY job_id
        """, (job_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_species_result_by_params(self, binomial_name: str, model: str, 
                                   system_template: str, user_template: str) -> Optional[Dict]:
        """Check if a species result already exists with these parameters"""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM processing_results 
            WHERE binomial_name = ? AND model = ? 
            AND system_template = ? AND user_template = ?
            AND status = 'completed'
            LIMIT 1
        """, (binomial_name, model, system_template, user_template))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def create_import_entry(self, job_id: str, species_file: str, model: str, system_template: str,
                         user_template: str, binomial_name: str, status: str = 'completed',
                         result: str = None, error: str = None, knowledge_group: str = None,
                         **phenotype_data):
        """Create a single entry for imported data with provided job_id"""
        print(f"[DEBUG unified_db] create_import_entry called with job_id: {job_id}, binomial_name: {binomial_name}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build dynamic SQL for phenotype fields
            phenotype_fields = []
            phenotype_values = []
            
            for field, value in phenotype_data.items():
                if value is not None:
                    phenotype_fields.append(field)
                    phenotype_values.append(value)
            
            # Build SQL query
            fields = ['job_id', 'species_file', 'model', 'system_template', 'user_template',
                     'binomial_name', 'status', 'result', 'error', 'knowledge_group',
                     'job_status', 'completed_at'] + phenotype_fields
            
            placeholders = ','.join(['?' for _ in fields])
            fields_str = ','.join(fields)
            
            values = [job_id, species_file, model, system_template, user_template,
                     binomial_name, status, result, error, knowledge_group,
                     'completed', datetime.now()] + phenotype_values
            
            cursor.execute(f"""
                INSERT INTO processing_results ({fields_str})
                VALUES ({placeholders})
            """, values)
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def update_import_result(self, job_id: str, binomial_name: str, result: str = None,
                           knowledge_group: str = None, **phenotype_data):
        """Update an existing result with new data from import"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build update query dynamically
            update_fields = []
            values = []
            
            if result is not None:
                update_fields.append('result = ?')
                values.append(result)
            
            if knowledge_group is not None:
                update_fields.append('knowledge_group = ?')
                values.append(knowledge_group)
            
            # Add phenotype fields
            for field, value in phenotype_data.items():
                if value is not None:
                    update_fields.append(f'{field} = ?')
                    values.append(value)
            
            if update_fields:
                # Add timestamp and where clause
                update_fields.append('completed_at = ?')
                values.append(datetime.now())
                
                values.extend([job_id, binomial_name])
                
                query = f"""
                    UPDATE processing_results 
                    SET {', '.join(update_fields)}
                    WHERE job_id = ? AND binomial_name = ?
                """
                
                cursor.execute(query, values)
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def update_job_status(self, job_id: str, status: str, timestamp_field: str = None):
        """Update job status and optionally set timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Update job status
            cursor.execute("""
                UPDATE processing_results 
                SET job_status = ?
                WHERE job_id = ?
            """, (status, job_id))
            
            # Update timestamp if specified
            if timestamp_field:
                cursor.execute(f"""
                    UPDATE processing_results 
                    SET {timestamp_field} = ?
                    WHERE job_id = ?
                """, (datetime.now(), job_id))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def update_species_result(self, job_id: str, binomial_name: str, 
                            result: str = None, status: str = None, 
                            error: str = None, phenotypes: Dict = None):
        """Update result for a specific species"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Build update query dynamically
            updates = []
            params = []
            
            if result is not None:
                updates.append("result = ?")
                params.append(result)
            
            if status is not None:
                updates.append("status = ?")
                params.append(status)
                
                # Set timestamps based on status
                if status == 'running':
                    updates.append("started_at = ?")
                    params.append(datetime.now())
                elif status in ['completed', 'failed', 'timeout']:
                    updates.append("completed_at = ?")
                    params.append(datetime.now())
            
            if error is not None:
                updates.append("error = ?")
                params.append(error)
            
            # Handle phenotypes
            if phenotypes:
                for field, value in phenotypes.items():
                    if field in ['knowledge_group', 'gram_staining', 'motility', 
                               'aerophilicity', 'extreme_environment_tolerance',
                               'biofilm_formation', 'animal_pathogenicity',
                               'biosafety_level', 'health_association',
                               'host_association', 'plant_pathogenicity',
                               'spore_formation', 'hemolysis', 'cell_shape']:
                        updates.append(f"{field} = ?")
                        params.append(value)
            
            if updates:
                params.extend([job_id, binomial_name])
                query = f"""
                    UPDATE processing_results 
                    SET {', '.join(updates)}
                    WHERE job_id = ? AND binomial_name = ?
                """
                cursor.execute(query, params)
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_pending_species(self, job_id: str, limit: int = None) -> List[Dict]:
        """Get pending species for processing"""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        if limit:
            cursor.execute("""
                SELECT binomial_name, species_file, model, system_template, user_template
                FROM processing_results
                WHERE job_id = ? AND status = 'pending'
                LIMIT ?
            """, (job_id, limit))
        else:
            cursor.execute("""
                SELECT binomial_name, species_file, model, system_template, user_template
                FROM processing_results
                WHERE job_id = ? AND status = 'pending'
            """, (job_id,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def delete_job(self, job_id: str):
        """Delete all entries for a job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM processing_results WHERE job_id = ?", (job_id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_dashboard_data(self) -> Dict:
        """Get data for admin dashboard matrix view"""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # Get all unique jobs
        cursor.execute("""
            SELECT DISTINCT 
                job_id, species_file, model, system_template, user_template,
                job_status, job_created_at
            FROM processing_results
            ORDER BY job_created_at DESC
        """)
        
        jobs = []
        job_map = {}  # Map for quick lookup
        
        for row in cursor.fetchall():
            job_data = dict(row)
            jobs.append(job_data)
            
            # Create key for matrix
            key = f"{job_data['species_file']}|{job_data['system_template']}|{job_data['user_template']}"
            if key not in job_map:
                job_map[key] = {}
            job_map[key][job_data['model']] = job_data['job_id']
        
        # Get statistics for each job-model combination
        cursor.execute("""
            SELECT 
                job_id,
                model,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) as timeouts,
                SUM(CASE WHEN status IN ('running', 'completed', 'failed', 'timeout') THEN 1 ELSE 0 END) as submitted
            FROM processing_results
            GROUP BY job_id, model
        """)
        
        # Store stats by job_id and model
        stats = {}
        for row in cursor.fetchall():
            job_id = row['job_id']
            model = row['model']
            if job_id not in stats:
                stats[job_id] = {}
            stats[job_id][model] = dict(row)
        
        # Get unique models from both processing_results and managed_models
        cursor.execute("SELECT DISTINCT model FROM processing_results ORDER BY model")
        models_from_results = set([row['model'] for row in cursor.fetchall()])
        
        cursor.execute("SELECT model FROM managed_models ORDER BY model")
        models_from_managed = set([row['model'] for row in cursor.fetchall()])
        
        # Combine both sets and sort
        models = sorted(list(models_from_results.union(models_from_managed)))
        
        # Get unique species files from both processing_results and managed_species_files
        cursor.execute("SELECT DISTINCT species_file FROM processing_results ORDER BY species_file")
        species_from_results = set([row['species_file'] for row in cursor.fetchall()])
        
        cursor.execute("SELECT species_file FROM managed_species_files ORDER BY species_file")
        species_from_managed = set([row['species_file'] for row in cursor.fetchall()])
        
        # Combine both sets and sort
        species_files = sorted(list(species_from_results.union(species_from_managed)))
        
        conn.close()
        
        # Build matrix data structure
        matrix = {}
        for job in jobs:
            # Use only filenames for the key to match dashboard template expectations
            species_name = Path(job['species_file']).name
            sys_name = Path(job['system_template']).name if job['system_template'] else job['system_template']
            usr_name = Path(job['user_template']).name if job['user_template'] else job['user_template']
            key = f"{species_name}|{sys_name}|{usr_name}"
            if key not in matrix:
                matrix[key] = {'models': {}}
            
            # Get stats for this specific job-model combination
            job_model_stats = stats.get(job['job_id'], {}).get(job['model'], {})
            matrix[key]['models'][job['model']] = {
                'id': job['job_id'],
                'status': job['job_status'],
                'total': job_model_stats.get('total', 0),
                'successful': job_model_stats.get('successful', 0),
                'failed': job_model_stats.get('failed', 0),
                'timeouts': job_model_stats.get('timeouts', 0),
                'submitted': job_model_stats.get('submitted', 0)
            }
        
        # Convert job data to combination format for compatibility
        combinations = []
        for job in jobs:
            # Get stats for this specific job-model combination
            job_model_stat = stats.get(job['job_id'], {}).get(job['model'], {})
            combinations.append({
                'id': job['job_id'],  # Dashboard expects 'id' field
                'species_file': job['species_file'],
                'model': job['model'],
                'system_template': job['system_template'],
                'user_template': job['user_template'],
                'status': job['job_status'],
                'total': job_model_stat.get('total', 0),
                'successful': job_model_stat.get('successful', 0),
                'failed': job_model_stat.get('failed', 0),
                'timeouts': job_model_stat.get('timeouts', 0),
                'submitted': job_model_stat.get('submitted', 0),
                'received': job_model_stat.get('submitted', 0)  # For compatibility
            })
        
        # Build template display info from the data
        template_display_info = []
        seen_templates = set()
        
        for job in jobs:
            template_key = (job['system_template'], job['user_template'])
            if template_key not in seen_templates:
                seen_templates.add(template_key)
                # Extract template name from path
                template_name = Path(job['system_template']).stem if job['system_template'] else 'unknown'
                template_display_info.append({
                    'system_template': job['system_template'],
                    'user_template': job['user_template'],
                    'display_name': template_name,
                    'description': f'Template: {template_name}',
                    'template_type': 'knowledge' if 'knowlege' in template_name else 'phenotype'
                })
        
        # Sort by display name
        template_display_info.sort(key=lambda x: x['display_name'])
        
        return {
            'combinations': combinations,  # Dashboard expects 'combinations'
            'models': models,
            'species_files': species_files,
            'matrix': matrix,
            'template_display_info': template_display_info
        }
    
    def get_results_for_analysis(self) -> List[Dict]:
        """Get all completed results for knowledge analysis"""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM processing_results
            WHERE status = 'completed'
            ORDER BY species_file, binomial_name, model
        """)
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results