# utils.py

import csv
import json
import re
import os
import time
from datetime import datetime
from string import Template
import requests
from colorama import Fore, Style
from microbellm import config
from tenacity import retry, stop_after_attempt, wait_exponential
import sqlite3
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path

# --- LLM Provider Abstraction ---

class LLMProvider:
    """Base class for LLM providers."""
    def query(self, messages, model, temperature, verbose=False):
        raise NotImplementedError("Subclasses must implement the query method.")

class OpenRouterProvider(LLMProvider):
    """LLM Provider for OpenRouter.ai."""
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key=None):
        self.api_key = api_key or config.OPENROUTER_API_KEY

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def query(self, messages, model, temperature, verbose=False):
        if not self.api_key:
            raise ValueError("OpenRouter API key is not set.")

        if verbose:
            print("Raw query to OpenRouter API:")
            print(json.dumps(messages, indent=2))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.API_URL, headers=headers, json=data)
            response.raise_for_status()
            
            completion = response.json()
            
            if verbose:
                print("\nRaw response from OpenRouter API:")
                print(json.dumps(completion, indent=2))
            
            result = completion['choices'][0]['message']['content']
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying OpenRouter API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None
        except KeyError as e:
            print(f"Error parsing OpenRouter API response: {e}")
            print(f"Response: {completion}")
            return None

# --- Existing Utility Functions ---

def read_template_from_file(template_path):
    """
    Reads a template from a file.
    
    Args:
        template_path (str): Path to the template file.
    
    Returns:
        str: The template content.
    """
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: Template file not found: {template_path}")
        return None
    except Exception as e:
        print(f"Error reading template file {template_path}: {e}")
        return None

def read_csv(file_path, delimiter=';'):
    """
    Reads a CSV file and returns the headers and rows.
    
    Args:
        file_path (str): Path to the CSV file.
        delimiter (str): Delimiter used in the CSV file.
    
    Returns:
        tuple: Headers and rows of the CSV file.
    """
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = next(reader)  # Skip the header row
        return headers, list(reader)

def extract_and_validate_json(data):
    """
    Extracts and validates JSON data from a string.
    
    Args:
        data (str): String containing JSON data.
    
    Returns:
        dict: Parsed JSON data.
    """
    if isinstance(data, dict):
        return data  # Already a dictionary, return as is.
    try:
        # Assuming data is a string that needs to be parsed
        json_str = re.search(r'\{.*\}', data, re.DOTALL).group()
        return json.loads(json_str)
    except (re.error, AttributeError, json.JSONDecodeError) as e:
        #print(f"Error extracting or decoding JSON: {e}")
        return None

def load_query_template(template_path, binomial_name):
    """
    Loads a query template and substitutes the binomial name.
    
    Args:
        template_path (str): Path to the template file.
        binomial_name (str): Binomial name to substitute in the template.
    
    Returns:
        str: The query message with the binomial name substituted.
    """
    with open(template_path, 'r') as file:
        template = file.read()
    # Replace the placeholder with the actual binomial name
    query_message = template.format(binomial_name=binomial_name)
    return query_message

def write_prediction(output_file, prediction, model_used, template_path):
    """
    Writes a prediction to a CSV file.
    
    Args:
        output_file (str): Path to the output CSV file.
        prediction (dict): Prediction data to write.
        model_used (str): Model used for the prediction.
        template_path (str): Path to the query template used.
    """
    write_header = False
    if not os.path.exists(output_file):
        write_header = True

    with open(output_file, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        
        # Define headers for the CSV file
        headers = list(prediction.keys()) + ['Model Used', 'Query Template', 'Date']
        
        # Check if the file is empty and write headers if it is
        if write_header:
            writer.writerow(headers)
        
        # Prepare data for writing
        raw_json = json.dumps(prediction, ensure_ascii=False).replace('"', "'")
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [prediction.get(header, 'N/A') for header in headers[:-3]] + [model_used, template_path, current_date]
        
        # Write the prediction row to the CSV file
        writer.writerow(row)

@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Wait 2^x * 1 seconds between retries
    reraise=True  # Reraise the exception if all retries fail
)
def query_openrouter_api(messages, model, temperature, verbose=False):
    """
    Queries the OpenRouter API with the given messages and model.
    Includes exponential backoff for retries.
    
    Args:
        messages (list): List of messages to send to the API.
        model (str): Model to use for the API call.
        temperature (float): Temperature to use for the API call.
        verbose (bool): Whether to print verbose output.
    
    Returns:
        str: The content of the API response.
    """
    provider = OpenRouterProvider()
    return provider.query(messages, model, temperature, verbose)

def summarize_predictions(predictions):
    """
    Summarizes predictions by calculating the majority vote and identifying disagreements.
    
    Args:
        predictions (list): List of prediction dictionaries.
    
    Returns:
        tuple: Summary of predictions and disagreements.
    """
    from collections import Counter
    summary = {}
    disagreements = {}

    # Initialize dictionaries to store counts of each category
    for key in predictions[0].keys():
        if key not in ['Model Used', 'Query Template', 'Date', 'Type']:
            summary[key] = []
            disagreements[key] = Counter()

    # Collect all predictions for each category
    for prediction in predictions:
        for key, value in prediction.items():
            if key in summary:
                # Convert lists to tuples for hashability
                if isinstance(value, list):
                    value = tuple(value)
                summary[key].append(value)

    # Calculate majority vote and identify disagreements
    results = {}
    for key, values in summary.items():
        count = Counter(values)
        most_common, num_most_common = count.most_common(1)[0]
        results[key] = most_common
        # Check if there's a disagreement
        if num_most_common < len(values):
            disagreements[key] = count

    # Ensure 'Binomial name' is included in results if it's a key in the predictions
    if 'Binomial name' in predictions[0]:
        results['Binomial name'] = predictions[0]['Binomial name']

    return results, disagreements


def pretty_print_prediction(prediction, model):
    """
    Pretty prints a prediction dictionary to the console, including the model used.
    
    Args:
        prediction (dict): The prediction dictionary to print.
        model (str): The model used for the prediction.
    """
    print("\nPrediction Results:")
    print("=" * 40)
    print(f"Model used: {model}")
    print("-" * 40)
    for key, value in prediction.items():
        if key not in ['Model Used', 'Query Template', 'Date']:
            if isinstance(value, (list, dict)):
                print(f"{key}:")
                print(json.dumps(value, indent=2))
            else:
                print(f"{key}: {value}")
    print("=" * 40)

def write_batch_jsonl(data, file_path):
    """
    Writes a list of dictionaries to a JSONL file.
    
    Args:
        data (list): List of dictionaries to write.
        file_path (str): Path to the output file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')
        print(f"Data written to {file_path}")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def parse_response(response):
    """
    Parses the response from the LLM and extracts the relevant information.
    
    Args:
        response (str): The response from the LLM.
    
    Returns:
        dict: A dictionary containing the parsed information.
    """
    if not response:
        return None
    
    # First try to extract JSON from the response
    json_data = extract_and_validate_json(response)
    if json_data:
        # Clean the JSON data using our cleaning function
        cleaned_data = {}
        for key, value in json_data.items():
            cleaned_key = key.strip().lower().replace(' ', '_').replace('-', '_')
            cleaned_data[cleaned_key] = clean_phenotype_value(value)
        return cleaned_data
        
    # Initialize the result dictionary with default values
    result = {
        'oxygen_requirements': '',
        'gram_staining': '',
        'spore_formation': '',
        'motility': '',
        'glucose_fermentation': '',
        'catalase_positive': '',
        'oxidase_positive': '',
        'biosafety_level': '',
        'health_association': '',
        'animal_pathogenicity': '',
        'plant_pathogenicity': '',
        'phenotypic_category': '',
        'hemolysis': '',
        'cell_shape': '',
        'biofilm_formation': '',
        'extreme_environment_tolerance': '',
        'host_association': ''
    }
    
    # Split response into lines and process each line
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_').replace('-', '_')
            value = clean_phenotype_value(value)
            
            # Map response keys to our standard keys with broader matching
            if 'oxygen' in key or 'aerobic' in key or 'aerophilic' in key:
                result['oxygen_requirements'] = value
            elif 'gram' in key:
                result['gram_staining'] = value
            elif 'spore' in key:
                result['spore_formation'] = value
            elif 'motil' in key:
                result['motility'] = value
            elif 'glucose' in key or 'ferment' in key:
                result['glucose_fermentation'] = value
            elif 'catalase' in key:
                result['catalase_positive'] = value
            elif 'oxidase' in key:
                result['oxidase_positive'] = value
            elif 'biosafety' in key or 'safety' in key:
                result['biosafety_level'] = value
            elif 'health' in key or ('human' in key and 'pathogen' in key):
                result['health_association'] = value
            elif 'animal' in key and 'pathogen' in key:
                result['animal_pathogenicity'] = value
            elif 'plant' in key and 'pathogen' in key:
                result['plant_pathogenicity'] = value
            elif 'phenotypic' in key or 'category' in key:
                result['phenotypic_category'] = value
            elif 'hemolytic' in key or 'hemolysis' in key:
                result['hemolysis'] = value
            elif 'shape' in key or 'morphology' in key:
                result['cell_shape'] = value
            elif 'biofilm' in key:
                result['biofilm_formation'] = value
            elif 'extreme' in key or 'environment' in key:
                result['extreme_environment_tolerance'] = value
            elif 'host' in key:
                result['host_association'] = value
    
    return result

def colorize_text(text, color):
    """
    Colorizes text using colorama.
    
    Args:
        text (str): Text to colorize.
        color (str): Color to use (from colorama.Fore).
    
    Returns:
        str: Colorized text.
    """
    return f"{color}{text}{Style.RESET_ALL}"

def clean_phenotype_value(value):
    """
    Clean and normalize a phenotype value from LLM response.
    
    Args:
        value: Raw value from LLM response (could be string, list, dict, etc.)
    
    Returns:
        str: Cleaned and normalized string value
    """
    if value is None:
        return ''
    
    # Convert to string first
    if isinstance(value, (list, tuple)):
        # Handle lists like ['anaerobic'] -> 'anaerobic'
        if len(value) == 1:
            value = str(value[0])
        elif len(value) > 1:
            value = ', '.join(str(v) for v in value)
        else:
            value = ''
    elif isinstance(value, dict):
        # Handle dict responses by converting to string representation
        value = str(value)
    else:
        value = str(value)
    
    # Remove surrounding quotes
    value = value.strip().strip('"').strip("'")
    
    # Remove brackets if they exist
    value = re.sub(r'^[\[\(](.+)[\]\)]$', r'\1', value)
    
    # Remove extra quotes inside the string
    value = value.replace('"', '').replace("'", '')
    
    # Clean up whitespace
    value = ' '.join(value.split())
    
    # Handle boolean-like values
    if value.lower() in ['true', 'yes', '1']:
        return 'TRUE'
    elif value.lower() in ['false', 'no', '0']:
        return 'FALSE'
    
    # Handle special cases and normalize common values
    value_lower = value.lower()
    
    # Normalize gram staining
    if 'gram' in value_lower:
        if 'positive' in value_lower or 'gram+' in value_lower:
            return 'gram stain positive'
        elif 'negative' in value_lower or 'gram-' in value_lower:
            return 'gram stain negative'
        elif 'variable' in value_lower:
            return 'gram stain variable'
    
    # Normalize motility
    if value_lower in ['motile', 'mobile', 'yes', 'true']:
        return 'TRUE'
    elif value_lower in ['non-motile', 'nonmotile', 'immobile', 'no', 'false']:
        return 'FALSE'
    
    # Normalize aerophilicity 
    if value_lower in ['aerobic', 'obligate aerobic', 'strict aerobic']:
        return 'aerobic'
    elif value_lower in ['anaerobic', 'obligate anaerobic', 'strict anaerobic']:
        return 'anaerobic'
    elif value_lower in ['facultative', 'facultative anaerobic', 'facultatively anaerobic']:
        return 'facultative'
    elif value_lower in ['microaerophilic', 'microaerophile']:
        return 'microaerophilic'
    
    # Normalize biosafety levels
    if 'biosafety' in value_lower or 'bsl' in value_lower:
        numbers = re.findall(r'\d+', value)
        if numbers:
            return f'biosafety level {numbers[0]}'
    
    # Normalize hemolysis
    if 'hemolysis' in value_lower or 'hemolytic' in value_lower:
        if 'alpha' in value_lower or 'α' in value_lower:
            return 'alpha-hemolytic'
        elif 'beta' in value_lower or 'β' in value_lower:
            return 'beta-hemolytic'
        elif 'gamma' in value_lower or 'γ' in value_lower or 'non' in value_lower:
            return 'non-hemolytic'
    
    # Return cleaned value
    return value.strip()

def clean_csv_field(field):
    """
    Clean a CSV field to prevent CSV formatting issues.
    
    Args:
        field (str): Field value to clean
    
    Returns:
        str: Cleaned field value safe for CSV export
    """
    if field is None:
        return ''
    
    field = str(field).strip()
    
    # Remove any existing quotes
    field = field.strip('"').strip("'")
    
    # Clean up any problematic characters for CSV
    field = field.replace('\n', ' ').replace('\r', ' ')
    field = field.replace(';', ',')  # Replace semicolons with commas since we use ; as delimiter
    
    return field

def get_all_jobs_from_db():
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, species_file, model, system_template, user_template, status, total_species, completed_species, failed_species FROM combinations ORDER BY created_at DESC")
    jobs = cursor.fetchall()
    conn.close()
    return jobs

def import_results_from_csv(csv_path):
    """
    Imports results from a CSV file into the database.

    Args:
        csv_path (str): The path to the CSV file to import.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting import from {csv_path}...")
    
    if not os.path.exists(csv_path):
        logging.error(f"Error: CSV file not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return

    # Define expected columns
    expected_columns = [
        'species_file', 'binomial_name', 'model', 'system_template', 'user_template', 
        'status', 'result', 'error', 'timestamp', 'gram_staining', 'motility', 'aerophilicity',
        'extreme_environment_tolerance', 'biofilm_formation', 'animal_pathogenicity',
        'biosafety_level', 'health_association', 'host_association', 'plant_pathogenicity',
        'spore_formation', 'hemolysis', 'cell_shape'
    ]

    # Validate columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"CSV file is missing required columns: {', '.join(missing_columns)}")
        return

    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    imported_count = 0
    skipped_count = 0
    error_count = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Importing Results"):
        try:
            # Sanitize data
            row = row.where(pd.notnull(row), None)

            # Find or create the combination ID
            cursor.execute('''
                SELECT id, total_species FROM combinations 
                WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
            ''', (row['species_file'], row['model'], row['system_template'], row['user_template']))
            
            combination = cursor.fetchone()
            
            if combination:
                combination_id = combination[0]
            else:
                # Create a new combination if it doesn't exist
                cursor.execute('''
                    INSERT INTO combinations (species_file, model, system_template, user_template, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (row['species_file'], row['model'], row['system_template'], row['user_template'], 'completed', datetime.now()))
                combination_id = cursor.lastrowid
            
            # Check if this specific result already exists
            cursor.execute('''
                SELECT id FROM results 
                WHERE species_file = ? AND binomial_name = ? AND model = ? AND system_template = ? AND user_template = ?
            ''', (row['species_file'], row['binomial_name'], row['model'], row['system_template'], row['user_template']))

            if cursor.fetchone():
                skipped_count += 1
                continue

            # Insert the new result
            cursor.execute('''
                INSERT INTO results (
                    species_file, binomial_name, model, system_template, user_template, status, 
                    result, error, timestamp, gram_staining, motility, aerophilicity, 
                    extreme_environment_tolerance, biofilm_formation, animal_pathogenicity,
                    biosafety_level, health_association, host_association, plant_pathogenicity,
                    spore_formation, hemolysis, cell_shape
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['species_file'], row['binomial_name'], row['model'], row['system_template'], 
                row['user_template'], row['status'], row['result'], row['error'], row['timestamp'],
                row['gram_staining'], row['motility'], row['aerophilicity'], 
                row['extreme_environment_tolerance'], row['biofilm_formation'], row['animal_pathogenicity'],
                row['biosafety_level'], row['health_association'], row['host_association'], 
                row['plant_pathogenicity'], row['spore_formation'], row['hemolysis'], row['cell_shape']
            ))

            # Update combination progress
            if row['status'] == 'success':
                cursor.execute("UPDATE combinations SET successful_species = successful_species + 1 WHERE id = ?", (combination_id,))
            elif row['status'] == 'failed':
                cursor.execute("UPDATE combinations SET failed_species = failed_species + 1 WHERE id = ?", (combination_id,))
            
            cursor.execute("UPDATE combinations SET received_species = received_species + 1, completed_species = received_species, submitted_species = received_species WHERE id = ?", (combination_id,))

            imported_count += 1

        except sqlite3.Error as e:
            logging.warning(f"Skipping row {index + 1} due to database error: {e}")
            error_count += 1
        except Exception as e:
            logging.warning(f"Skipping row {index + 1} due to unexpected error: {e}")
            error_count += 1

    conn.commit()
    conn.close()

    logging.info("--- Import Summary ---")
    logging.info(f"Successfully imported: {imported_count} records")
    logging.info(f"Skipped (already exist): {skipped_count} records")
    logging.info(f"Failed (errors): {error_count} records")
    logging.info("----------------------")