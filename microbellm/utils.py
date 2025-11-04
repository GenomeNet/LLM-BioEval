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

# Try to import tenacity for retry logic, fallback if not available
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    TENACITY_AVAILABLE = True
except ImportError:
    print("Warning: tenacity not available, retries disabled")
    TENACITY_AVAILABLE = False
    # Create dummy decorators
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def stop_after_attempt(*args):
        pass
    def wait_exponential(*args, **kwargs):
        pass

import sqlite3

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

    def _query_with_retry(self, messages, model, temperature, verbose=False):
        """Internal method with retry logic"""
        if not self.api_key:
            raise ValueError("OpenRouter API key is not set.")

        if verbose:
            print(f"\n=== OpenRouter API Request ===")
            print(f"Model: {model}")
            print(f"Temperature: {temperature}")
            print(f"API Key: {'Set' if self.api_key else 'NOT SET'}")
            print("Messages:")
            print(json.dumps(messages, indent=2))
            print("============================")

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
            if verbose:
                print(f"Sending request to: {self.API_URL}")
                
            response = requests.post(self.API_URL, headers=headers, json=data, timeout=30)
            
            if verbose:
                print(f"Response status: {response.status_code}")
                
            response.raise_for_status()
            
            completion = response.json()
            
            if verbose:
                print("\nRaw response from OpenRouter API:")
                print(json.dumps(completion, indent=2))
            
            result = completion['choices'][0]['message']['content']
            return result
            
        except requests.exceptions.Timeout:
            print(f"Timeout querying OpenRouter API for model {model}")
            raise TimeoutError(f"OpenRouter API timeout after 30 seconds for model {model}")
        except requests.exceptions.RequestException as e:
            print(f"\nERROR: OpenRouter API request failed")
            print(f"  Model: {model}")
            print(f"  Error Type: {type(e).__name__}")
            print(f"  Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Status Code: {e.response.status_code}")
                try:
                    error_data = e.response.json()
                    print(f"  API Error Details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"  Response Text: {e.response.text[:500]}..." if len(e.response.text) > 500 else f"  Response Text: {e.response.text}")
            return None
        except KeyError as e:
            print(f"Error parsing OpenRouter API response: {e}")
            print(f"Response: {completion}")
            return None

    if TENACITY_AVAILABLE:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True
        )
        def query(self, messages, model, temperature, verbose=False):
            return self._query_with_retry(messages, model, temperature, verbose)
    else:
        def query(self, messages, model, temperature, verbose=False):
            return self._query_with_retry(messages, model, temperature, verbose)

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

def clean_malformed_json(json_str):
    """
    Clean malformed JSON that contains extra quotes around keys/values.
    
    Args:
        json_str (str): Raw JSON string that may have extra quotes
        
    Returns:
        str: Cleaned JSON string
    """
    if not json_str:
        return json_str
    
    # Pattern to match keys with extra quotes: ""key"" or """key"""
    # Replace with proper single quotes: "key"
    json_str = re.sub(r'"{2,}([^"]+)"{2,}', r'"\1"', json_str)
    
    # Pattern to match values with extra quotes around strings
    # This is more complex as we need to handle: "key": ""value""
    json_str = re.sub(r'(\w+"):\s*"{2,}([^"]+)"{2,}', r'\1: "\2"', json_str)
    json_str = re.sub(r'("[\w_]+"):\s*"{2,}([^"]+)"{2,}', r'\1: "\2"', json_str)
    
    # Handle standalone quoted values with extra quotes
    json_str = re.sub(r':\s*"{3,}([^"]+)"{3,}', r': "\1"', json_str)
    json_str = re.sub(r':\s*"{2}([^"]+)"{2}', r': "\1"', json_str)
    
    return json_str

def extract_and_validate_json(data):
    """
    Extracts and validates JSON data from a string with robust malformed JSON handling.
    
    Args:
        data (str): String containing JSON data.
    
    Returns:
        dict: Parsed JSON data.
    """
    if isinstance(data, dict):
        return data  # Already a dictionary, return as is.
    
    if not isinstance(data, str):
        return None
        
    try:
        # Extract JSON pattern from the response
        json_match = re.search(r'\{.*\}', data, re.DOTALL)
        if not json_match:
            return None
            
        json_str = json_match.group()
        
        # First try parsing as-is
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Clean malformed JSON and try again
        cleaned_json = clean_malformed_json(json_str)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            pass
        
        # Additional cleaning attempts for common LLM errors
        # Remove trailing commas
        cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
        
        # Fix missing quotes around keys
        cleaned_json = re.sub(r'(\w+):', r'"\1":', cleaned_json)
        
        # Fix single quotes to double quotes
        cleaned_json = cleaned_json.replace("'", '"')
        
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            print(f"Error extracting or decoding JSON after cleaning attempts: {e}")
            print(f"Original JSON: {json_str[:200]}...")
            print(f"Cleaned JSON: {cleaned_json[:200]}...")
            return None
            
    except (re.error, AttributeError) as e:
        print(f"Error in JSON extraction: {e}")
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

if TENACITY_AVAILABLE:
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
else:
    def query_openrouter_api(messages, model, temperature, verbose=False):
        """
        Queries the OpenRouter API with the given messages and model.
        No retries when tenacity is not available.
        
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

def parse_response_with_template_config(response, user_template_path):
    """
    Parse LLM response using JSON-based template configuration system
    
    Args:
        response (str): Raw response from LLM
        user_template_path (str): Path to user template file
        
    Returns:
        dict: Parsed and validated response data
    """
    try:
        from microbellm.template_config import validate_template_response_from_file
        
        # Store raw response
        result = {'raw_response': response}
        
        # Try to extract JSON from response
        response_cleaned = response.strip()
        
        # Use improved JSON extraction with malformed JSON handling
        response_data = extract_and_validate_json(response_cleaned)
        if response_data:
            try:
                # Validate using JSON configuration file
                validated_data, errors, validator = validate_template_response_from_file(user_template_path, response_data)
                
                if errors:
                    print(f"Template validation errors: {errors}")
                    # Still include validation errors for tracking
                    result['validation_errors'] = errors
                
                # Process validated data - only include properly validated values
                cleaned_data = {}
                invalid_fields = []
                
                for key, value in validated_data.items():
                    if value is None:
                        # Skip None values
                        continue
                    elif isinstance(value, str) and value.startswith('INVALID:'):
                        # Track invalid values
                        original_value = value[8:]  # Remove "INVALID:" prefix
                        invalid_fields.append({'field': key, 'value': original_value})
                        # Don't include in cleaned data
                    else:
                        # Include valid values
                        cleaned_data[key] = value
                
                # Add tracking for invalid fields
                if invalid_fields:
                    result['invalid_fields'] = invalid_fields
                
                # Merge cleaned data into result
                result.update(cleaned_data)
                
                return result
                
            except Exception as e:
                print(f"Template validation error: {e}")
                result['validation_error'] = str(e)
        else:
            print(f"Failed to extract JSON from response: {response_cleaned[:200]}...")
            result['json_extraction_failed'] = True
        
        # If JSON parsing fails, try fallback parsing
        fallback_result = parse_response_fallback(response)
        if fallback_result:
            result.update(fallback_result)
        
        return result
        
    except Exception as e:
        print(f"Error in template-based parsing: {e}")
        # Fallback to original parsing
        return parse_response_fallback(response)

def parse_response(response, user_template_path=None):
    """
    Parse LLM response using template configuration system when available.
    
    Args:
        response (str): Raw response from LLM
        user_template_path (str, optional): Path to user template file for validation
        
    Returns:
        dict: Parsed and validated response data
    """
    if user_template_path:
        return parse_response_with_template_config(response, user_template_path)
    else:
        return parse_response_fallback(response)

def parse_response_fallback(response):
    """
    Fallback parsing method for LLM responses (original implementation).
    
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
    
    # Initial cleanup
    value = value.strip()
    
    # Remove trailing commas and other JSON punctuation that might be included
    value = value.rstrip(',.;:')
    
    # Remove surrounding quotes (including multiple quotes)
    # Handle multiple surrounding quotes like ""value"" or """value"""
    value = re.sub(r'^"{2,}(.+?)"{2,}$', r'\1', value)
    value = re.sub(r"^'{2,}(.+?)'{2,}$", r'\1', value)
    # Handle single surrounding quotes
    value = value.strip('"').strip("'")
    
    # Remove trailing commas again after quote removal
    value = value.rstrip(',.;:')
    
    # Remove brackets if they exist
    value = re.sub(r'^[\[\(](.+)[\]\)]$', r'\1', value)
    
    # Remove extra quotes inside the string (but preserve intentional quotes)
    # Only remove if there are clearly extra quotes
    if value.count('"') > 2 or value.count("'") > 2:
        value = value.replace('"', '').replace("'", '')
    
    # Final cleanup of whitespace and trailing punctuation
    value = ' '.join(value.split())
    value = value.rstrip(',.;:')
    
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

def normalize_value(value):
    """Normalize a value to a consistent format."""
    if value is None:
        return 'NA'
    
    val_str = str(value).strip().lower()
    
    if val_str in ['', 'na', 'n/a', 'null', 'none', 'unknown']:
        return 'NA'
        
    if isinstance(value, list):
        return ', '.join(sorted([str(v) for v in value])).lower()

    return str(value).strip()

def detect_template_type(user_template_path):
    """
    Detects if a template is for phenotype prediction or knowledge level assessment.
    
    Args:
        user_template_path (str): Path to the user template file
    
    Returns:
        str: 'phenotype' or 'knowledge'
    """
    try:
        # First try to detect from JSON config file
        from microbellm.template_config import detect_template_type_from_config
        template_type = detect_template_type_from_config(user_template_path)
        
        if template_type != 'unknown':
            return template_type
            
    except Exception:
        pass
    
    # Fallback to content-based detection
    try:
        with open(user_template_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        knowledge_indicators = [
            'knowledge_group',
            'knowledge_level', 
            'knowledge level',
            '<limited|moderate|extensive',
            '"limited"',
            '"moderate"', 
            '"extensive"'
        ]
        
        phenotype_indicators = [
            'gram_staining',
            'motility',
            'biosafety_level',
            'pathogenicity',
            'aerophilicity',
            'biofilm_formation'
        ]
        
        # Check for knowledge indicators
        if any(indicator in content for indicator in knowledge_indicators):
            return 'knowledge'
        
        # Check for phenotype indicators
        if any(indicator in content for indicator in phenotype_indicators):
            return 'phenotype'
        
        # Default to phenotype for backward compatibility
        return 'phenotype'
    except:
        return 'phenotype'

def normalize_knowledge_level(knowledge_level):
    """Normalize knowledge level values to standard categories"""
    if not knowledge_level:
        return None
    
    knowledge_str = str(knowledge_level).lower().strip()
    
    if knowledge_str in ['limited', 'minimal', 'basic', 'low']:
        return 'limited'
    elif knowledge_str in ['moderate', 'medium', 'intermediate']:
        return 'moderate'  
    elif knowledge_str in ['extensive', 'comprehensive', 'detailed', 'high', 'full']:
        return 'extensive'
    elif knowledge_str in ['na', 'n/a', 'n.a.', 'not available', 'not applicable', 'unknown']:
        return 'NA'
    else:
        return None  # Changed: Don't default to 'limited', return None for unrecognized values

def is_header_line(line):
    """
    Detect if a line appears to be a CSV header rather than a species name.
    
    Args:
        line (str): The line to check
    
    Returns:
        bool: True if the line appears to be a header, False otherwise
    """
    if not line or not line.strip():
        return False
    
    line_lower = line.strip().lower()
    
    # Common header patterns
    header_indicators = [
        'binomial name',
        'binomial_name', 
        'species name',
        'species_name',
        'taxon name',
        'taxon_name',
        'organism',
        'name,type',
        'name,species',
        'binomial,type'
    ]
    
    # Check for exact matches or if line starts with these patterns
    for indicator in header_indicators:
        if line_lower == indicator or line_lower.startswith(indicator + ','):
            return True
    
    # Check for patterns that suggest this is a header row
    # Headers often contain comma-separated field names
    if ',' in line_lower and any(keyword in line_lower for keyword in ['name', 'type', 'species', 'binomial', 'taxon']):
        return True
    
    return False

def filter_species_list(lines):
    """
    Filter a list of lines to remove headers and return only valid species names.
    
    Args:
        lines (list): List of lines from a species file
    
    Returns:
        list: Filtered list containing only species names (no headers)
    """
    if not lines:
        return []
    
    filtered_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Skip empty lines
        if not stripped_line:
            continue
            
        # Skip header lines
        if is_header_line(stripped_line):
            continue

        delimiter = None
        if '\t' in stripped_line:
            delimiter = '\t'
        elif ',' in stripped_line:
            delimiter = ','
            
        # For CSV/TSV format, extract just the first column (binomial name)
        if delimiter:
            # Split by delimiter and take the first part as the binomial name
            binomial_name = stripped_line.split(delimiter)[0].strip()
            if binomial_name and not is_header_line(binomial_name):
                filtered_lines.append(binomial_name)
        else:
            # Plain text format - add the line as is
            filtered_lines.append(stripped_line)
    
    return filtered_lines

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
    
    # Remove extra whitespace
    field = ' '.join(field.split())
    
    return field

# Ground Truth Data Management
def create_ground_truth_tables():
    """Create tables for storing ground truth data"""
    from .config import DATABASE_PATH
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Main ground truth table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ground_truth (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            template_name TEXT NOT NULL,
            binomial_name TEXT NOT NULL,
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
            import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(dataset_name, binomial_name)
        )
    ''')
    
    # Ground truth metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ground_truth_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT UNIQUE NOT NULL,
            description TEXT,
            source TEXT,
            template_name TEXT NOT NULL,
            species_count INTEGER,
            import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            validation_summary TEXT
        )
    ''')

    # Cached ground truth statistics snapshots
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ground_truth_statistics_cache (
            dataset_name TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            import_timestamp REAL DEFAULT 0,
            computed_at REAL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Cached model accuracy snapshots
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_accuracy_cache (
            dataset_name TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            import_timestamp REAL DEFAULT 0,
            computed_at REAL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Cached knowledge accuracy snapshots
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_accuracy_cache (
            dataset_name TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            import_timestamp REAL DEFAULT 0,
            computed_at REAL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Cached model performance by publication year snapshots
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance_year_cache (
            dataset_name TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            import_timestamp REAL DEFAULT 0,
            computed_at REAL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def validate_ground_truth_value(field_name, value, template_data):
    """Validate a ground truth value against template specifications"""
    if not template_data or 'field_definitions' not in template_data:
        return {'valid': True, 'normalized': value}
    
    field_def = template_data['field_definitions'].get(field_name)
    if not field_def:
        return {'valid': True, 'normalized': value}
    
    # Handle null/empty values
    if value is None or value == '' or value.upper() in ['NA', 'N/A', 'NULL', 'NONE']:
        return {'valid': True, 'normalized': None}
    
    # Get validation rules
    validation_rules = field_def.get('validation_rules', {})
    allowed_values = field_def.get('allowed_values', [])
    
    # Normalize the value
    normalized_value = value
    if validation_rules.get('trim_whitespace', True):
        normalized_value = normalized_value.strip()
    
    if not validation_rules.get('case_sensitive', True):
        normalized_value = normalized_value.lower()
    
    # Check normalization mapping
    normalize_mapping = validation_rules.get('normalize_mapping', {})
    for canonical, variations in normalize_mapping.items():
        if normalized_value.lower() in [v.lower() for v in variations]:
            normalized_value = canonical
            break
    
    # Check if value is allowed
    if allowed_values:
        allowed_lower = [v.lower() for v in allowed_values]
        if normalized_value.lower() not in allowed_lower:
            return {
                'valid': False,
                'normalized': normalized_value,
                'error': f"Value '{value}' not in allowed values: {allowed_values}"
            }
    
    return {'valid': True, 'normalized': normalized_value}

def import_ground_truth_csv(csv_path, dataset_name, template_name, description=None, source=None):
    """Import ground truth data from CSV file"""
    import csv
    
    # Load template data for validation
    template_path = os.path.join('templates', 'validation', f'{template_name}.json')
    with open(template_path, 'r') as f:
        template_data = json.load(f)
    
    # Create tables if they don't exist
    create_ground_truth_tables()
    
    from .config import DATABASE_PATH
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Read CSV
    imported_count = 0
    validation_errors = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Detect delimiter
        first_line = f.readline()
        f.seek(0)
        delimiter = ';' if ';' in first_line else ','
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Normalize header fields to handle different casing or spacing
        if reader.fieldnames:
            reader.fieldnames = [field.strip().lower().replace(" ", "_") for field in reader.fieldnames]

        for row in reader:
            binomial_name = row.get('binomial_name', '').strip()
            if not binomial_name:
                continue
            
            # Validate and normalize each field
            validated_row = {'dataset_name': dataset_name, 'template_name': template_name, 'binomial_name': binomial_name}
            row_errors = []
            
            for field_name in ['gram_staining', 'motility', 'aerophilicity', 'extreme_environment_tolerance',
                              'biofilm_formation', 'animal_pathogenicity', 'biosafety_level',
                              'health_association', 'host_association', 'plant_pathogenicity',
                              'spore_formation', 'hemolysis', 'cell_shape']:
                
                if field_name in row:
                    validation_result = validate_ground_truth_value(field_name, row[field_name], template_data)
                    
                    if validation_result['valid']:
                        validated_row[field_name] = validation_result['normalized']
                    else:
                        row_errors.append(f"{field_name}: {validation_result['error']}")
                        validated_row[field_name] = row[field_name]  # Store original value
                else:
                    validated_row[field_name] = None
            
            if row_errors:
                validation_errors.append({
                    'binomial_name': binomial_name,
                    'errors': row_errors
                })
            
            # Insert or update the record
            placeholders = ', '.join(['?' for _ in validated_row])
            columns = ', '.join(validated_row.keys())
            
            cursor.execute(f'''
                INSERT OR REPLACE INTO ground_truth ({columns})
                VALUES ({placeholders})
            ''', list(validated_row.values()))
            
            imported_count += 1
    
    # Update dataset metadata
    cursor.execute('''
        INSERT OR REPLACE INTO ground_truth_datasets 
        (dataset_name, description, source, template_name, species_count, validation_summary)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (dataset_name, description, source, template_name, imported_count, json.dumps(validation_errors)))
    
    conn.commit()
    conn.close()
    
    return {
        'success': True,
        'imported_count': imported_count,
        'validation_errors': validation_errors
    }

def get_ground_truth_datasets():
    """Get list of imported ground truth datasets"""
    from .config import DATABASE_PATH
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT dataset_name, description, source, template_name, species_count, import_date
        FROM ground_truth_datasets
        ORDER BY import_date DESC
    ''')
    
    datasets = []
    for row in cursor.fetchall():
        datasets.append({
            'dataset_name': row[0],
            'description': row[1],
            'source': row[2],
            'template_name': row[3],
            'species_count': row[4],
            'import_date': row[5]
        })
    
    conn.close()
    return datasets

def get_ground_truth_data(dataset_name=None, binomial_name=None, limit=None, offset=0):
    """Retrieve ground truth data with optional filtering"""
    from .config import DATABASE_PATH
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT * FROM ground_truth WHERE 1=1'
    params = []
    
    if dataset_name:
        query += ' AND dataset_name = ?'
        params.append(dataset_name)
    
    if binomial_name:
        query += ' AND binomial_name LIKE ?'
        params.append(f'%{binomial_name}%')
    
    query += ' ORDER BY binomial_name'
    
    if limit:
        query += f' LIMIT {limit} OFFSET {offset}'
    
    cursor.execute(query, params)
    
    columns = [description[0] for description in cursor.description]
    results = []
    
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))
    
    conn.close()
    return results

def delete_ground_truth_dataset(dataset_name):
    """Deletes a ground truth dataset and its associated data."""
    from .config import DATABASE_PATH
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Delete data from the main table
        cursor.execute("DELETE FROM ground_truth WHERE dataset_name = ?", (dataset_name,))
        
        # Delete metadata from the datasets table
        cursor.execute("DELETE FROM ground_truth_datasets WHERE dataset_name = ?", (dataset_name,))
        
        conn.commit()
        return {'success': True}
    except Exception as e:
        if conn:
            conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        if conn:
            conn.close()

def calculate_model_accuracy(predictions_file, dataset_name, template_name):
    """Calculate accuracy metrics for model predictions against ground truth"""
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Get ground truth data
    ground_truth = get_ground_truth_data(dataset_name=dataset_name)
    gt_dict = {gt['binomial_name']: gt for gt in ground_truth}
    
    # Calculate metrics for each field
    metrics = {}
    phenotype_fields = ['gram_staining', 'motility', 'aerophilicity', 'extreme_environment_tolerance',
                       'biofilm_formation', 'animal_pathogenicity', 'biosafety_level',
                       'health_association', 'host_association', 'plant_pathogenicity',
                       'spore_formation', 'hemolysis', 'cell_shape']
    
    for field in phenotype_fields:
        metrics[field] = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'missing_prediction': 0,
            'missing_ground_truth': 0,
            'confusion_matrix': {}
        }
    
    # Process each prediction
    for pred in predictions:
        binomial_name = pred.get('prompt')
        if not binomial_name or binomial_name not in gt_dict:
            continue
        
        gt = gt_dict[binomial_name]
        
        for field in phenotype_fields:
            pred_value = pred.get(field)
            gt_value = gt.get(field)
            
            # Skip if ground truth is missing
            if gt_value is None or gt_value == '':
                metrics[field]['missing_ground_truth'] += 1
                continue
            
            # Count total
            metrics[field]['total'] += 1
            
            # Check if prediction is missing
            if pred_value is None or pred_value == '' or pred_value == 'NA':
                metrics[field]['missing_prediction'] += 1
                continue
            
            # Normalize values for comparison
            pred_normalized = str(pred_value).lower().strip()
            gt_normalized = str(gt_value).lower().strip()
            
            # Check correctness
            if pred_normalized == gt_normalized:
                metrics[field]['correct'] += 1
            else:
                metrics[field]['incorrect'] += 1
            
            # Update confusion matrix
            if gt_normalized not in metrics[field]['confusion_matrix']:
                metrics[field]['confusion_matrix'][gt_normalized] = {}
            
            if pred_normalized not in metrics[field]['confusion_matrix'][gt_normalized]:
                metrics[field]['confusion_matrix'][gt_normalized][pred_normalized] = 0
            
            metrics[field]['confusion_matrix'][gt_normalized][pred_normalized] += 1
    
    # Calculate accuracy percentages
    for field in phenotype_fields:
        if metrics[field]['total'] > 0:
            metrics[field]['accuracy'] = (metrics[field]['correct'] / metrics[field]['total']) * 100
            metrics[field]['missing_rate'] = (metrics[field]['missing_prediction'] / metrics[field]['total']) * 100
        else:
            metrics[field]['accuracy'] = 0
            metrics[field]['missing_rate'] = 0
    
    return metrics
