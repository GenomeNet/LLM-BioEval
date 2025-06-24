import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime
from microbellm import config

def import_results_from_csv(csv_path, progress_callback=None):
    """
    Imports results from a CSV file into the database.

    Args:
        csv_path (str): The path to the CSV file to import.
        progress_callback (function, optional): A function to call with progress updates.
    """
    
    def log_and_callback(message, level='info', **kwargs):
        """Helper to send log messages to callback or console."""
        if progress_callback:
            data = {'type': 'log', 'level': level, 'message': message}
            data.update(kwargs)
            progress_callback(data)
        else:
            if level == 'error':
                logging.error(message)
            elif level == 'warning':
                logging.warning(message)
            else:
                logging.info(message)

    log_and_callback(f"Starting import from {os.path.basename(csv_path)}...")

    if not os.path.exists(csv_path):
        log_and_callback(f"Error: CSV file not found at {csv_path}", level='error')
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log_and_callback(f"Error reading CSV file: {e}", level='error')
        return

    expected_columns = [
        'species_file', 'binomial_name', 'model', 'system_template', 'user_template', 
        'status', 'result', 'error', 'timestamp'
    ]
    
    optional_phenotype_columns = [
        'gram_staining', 'motility', 'aerophilicity', 'extreme_environment_tolerance', 
        'biofilm_formation', 'animal_pathogenicity', 'biosafety_level', 'health_association', 
        'host_association', 'plant_pathogenicity', 'spore_formation', 'hemolysis', 'cell_shape'
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        log_and_callback(f"CSV file is missing required columns: {', '.join(missing_columns)}", level='error')
        return

    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    imported_count = 0
    skipped_count = 0
    error_count = 0
    total_rows = len(df)

    for index, row in df.iterrows():
        try:
            row = row.where(pd.notnull(row), None)

            cursor.execute('''
                SELECT id FROM combinations 
                WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
            ''', (row['species_file'], row['model'], row['system_template'], row['user_template']))
            
            combination = cursor.fetchone()
            
            if not combination:
                cursor.execute('''
                    INSERT INTO combinations (species_file, model, system_template, user_template, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (row['species_file'], row['model'], row['system_template'], row['user_template'], 'completed', datetime.now()))

            cursor.execute('''
                SELECT id FROM results 
                WHERE species_file = ? AND binomial_name = ? AND model = ? AND system_template = ? AND user_template = ?
            ''', (row['species_file'], row['binomial_name'], row['model'], row['system_template'], row['user_template']))

            if cursor.fetchone():
                skipped_count += 1
                continue
            
            # Build insert query dynamically for optional columns
            result_data = {col: row.get(col) for col in expected_columns}
            for col in optional_phenotype_columns:
                if col in row:
                    result_data[col] = row.get(col)

            columns = ', '.join(result_data.keys())
            placeholders = ', '.join('?' * len(result_data))
            
            cursor.execute(f'INSERT INTO results ({columns}) VALUES ({placeholders})', list(result_data.values()))

            imported_count += 1

        except sqlite3.Error as e:
            error_count += 1
            log_and_callback(f"Skipping row {index + 2} due to database error: {e}", level='warning')
        except Exception as e:
            error_count += 1
            log_and_callback(f"Skipping row {index + 2} due to unexpected error: {e}", level='warning')
        
        if progress_callback and (index + 1) % 10 == 0 or (index + 1) == total_rows:
            progress_callback({
                'type': 'progress',
                'current': index + 1,
                'total': total_rows,
                'imported': imported_count,
                'skipped': skipped_count,
                'errors': error_count
            })

    conn.commit()
    conn.close()

    summary_data = {
        'imported': imported_count,
        'skipped': skipped_count,
        'errors': error_count,
        'total': total_rows
    }
    log_and_callback("Import complete.", type='summary', summary=summary_data) 