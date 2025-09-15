#!/usr/bin/env python
"""
Prediction Validation Module for MicrobeLLM

This module validates and normalizes LLM predictions against template specifications
defined in templates/validation/*.json files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sqlite3
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PredictionValidator:
    """Validates and normalizes predictions based on template specifications."""
    
    def __init__(self):
        self.validation_configs = {}
        self.load_validation_configs()
        self.validation_stats = defaultdict(int)
        self.validation_log = []
        
    def load_validation_configs(self):
        """Load all validation configurations from templates/validation/"""
        validation_dir = Path(__file__).parent.parent / 'templates' / 'validation'
        
        if not validation_dir.exists():
            logger.error(f"Validation directory not found: {validation_dir}")
            return
            
        for config_file in validation_dir.glob('*.json'):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    template_name = config_file.stem
                    self.validation_configs[template_name] = config
                    logger.info(f"Loaded validation config: {template_name}")
            except Exception as e:
                logger.error(f"Error loading config {config_file}: {e}")
    
    def get_template_config(self, system_template: str, user_template: str) -> Optional[Dict]:
        """Get the appropriate validation config for a template pair."""
        # Extract template name from path
        if '/' in user_template:
            template_name = Path(user_template).stem
        else:
            template_name = user_template.replace('.txt', '')
            
        logger.info(f"Looking for validation config for template: {template_name}")
        
        # Determine template type based on template name
        template_type = None
        if 'knowlege' in template_name.lower() or 'knowledge' in template_name.lower():
            template_type = 'knowlege'  # Use the actual filename spelling
        elif 'phenotype' in template_name.lower():
            template_type = 'phenotype'
        
        # Try exact match first
        if template_name in self.validation_configs:
            logger.info(f"Found exact match: {template_name}")
            return self.validation_configs[template_name]
            
        # Try with different naming patterns
        for suffix in ['_phenotype', '_knowlege', '_knowledge']:
            candidate = template_name + suffix
            if candidate in self.validation_configs:
                logger.info(f"Found config with suffix: {candidate}")
                return self.validation_configs[candidate]
        
        # Try to match base template name (e.g., template1 -> template1_phenotype or template1_knowlege)
        base_name = template_name.replace('_phenotype', '').replace('_knowlege', '').replace('_knowledge', '')
        
        # Determine which type to use based on context
        if template_type == 'knowlege':
            candidates = [f"{base_name}_knowlege", f"{base_name}_knowledge"]
        else:
            candidates = [f"{base_name}_phenotype"]
        
        for candidate in candidates:
            if candidate in self.validation_configs:
                logger.info(f"Found config for base template: {candidate}")
                return self.validation_configs[candidate]
                
        # Try matching by template type in key
        for key, config in self.validation_configs.items():
            if base_name in key:
                logger.info(f"Found partial match: {key}")
                return config
        
        # Use appropriate default based on template type
        if template_type == 'knowlege':
            default = 'template1_knowlege'
        else:
            default = 'template1_phenotype'
        
        if default in self.validation_configs:
            logger.warning(f"No specific config found for {template_name}, using default: {default}")
            return self.validation_configs[default]
                
        logger.error(f"No validation config found for template: {template_name}")
        return None
    
    def normalize_value(self, value: Any, field_config: Dict) -> Tuple[Any, bool, str]:
        """
        Normalize a value according to field configuration rules.
        
        Returns:
            Tuple of (normalized_value, is_valid, action_taken)
        """
        if value is None or value == '' or value == 'NA':
            return 'NA', True, 'kept_na'
            
        # Convert to string for processing
        str_value = str(value).strip()
        
        # Get validation rules
        rules = field_config.get('validation_rules', {})
        
        # Apply case sensitivity
        if not rules.get('case_sensitive', True):
            str_value_lower = str_value.lower()
        else:
            str_value_lower = str_value
            
        # Check if it's an allowed value
        allowed_values = field_config.get('allowed_values', [])
        
        # First check exact match
        if str_value in allowed_values:
            return str_value, True, 'kept_valid'
            
        # Check normalization mappings
        normalize_mapping = rules.get('normalize_mapping', {})
        for correct_value, variations in normalize_mapping.items():
            # Convert variations to lowercase if case insensitive
            if not rules.get('case_sensitive', True):
                variations_to_check = [v.lower() for v in variations]
                if str_value_lower in variations_to_check:
                    self.validation_stats['normalized'] += 1
                    return correct_value, True, f'normalized_{str_value}_to_{correct_value}'
            else:
                if str_value in variations:
                    self.validation_stats['normalized'] += 1
                    return correct_value, True, f'normalized_{str_value}_to_{correct_value}'
        
        # Handle array fields
        if field_config.get('type') == 'array':
            # Try to parse as JSON array
            try:
                if str_value.startswith('['):
                    array_values = json.loads(str_value)
                else:
                    # Handle comma-separated values
                    array_values = [v.strip() for v in str_value.split(',')]
                
                normalized_array = []
                all_valid = True
                
                for item in array_values:
                    item_str = str(item).strip()
                    item_normalized = False
                    
                    # Check if item is in allowed values
                    if item_str in allowed_values:
                        normalized_array.append(item_str)
                        item_normalized = True
                    else:
                        # Check normalization mapping for this item
                        for correct_value, variations in normalize_mapping.items():
                            if not rules.get('case_sensitive', True):
                                variations_to_check = [v.lower() for v in variations]
                                if item_str.lower() in variations_to_check:
                                    normalized_array.append(correct_value)
                                    item_normalized = True
                                    break
                            else:
                                if item_str in variations:
                                    normalized_array.append(correct_value)
                                    item_normalized = True
                                    break
                    
                    if not item_normalized:
                        all_valid = False
                        logger.warning(f"Invalid array item '{item_str}' for field {field_config.get('description')}")
                
                if normalized_array and all_valid:
                    self.validation_stats['normalized'] += 1
                    return normalized_array, True, f'normalized_array_{str_value}'
                elif normalized_array:
                    self.validation_stats['partially_normalized'] += 1
                    return normalized_array, True, f'partially_normalized_array_{str_value}'
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Could not parse array value '{str_value}': {e}")
        
        # Value is invalid
        self.validation_stats['invalid'] += 1
        logger.warning(f"Invalid value '{str_value}' for field {field_config.get('description')}. Allowed values: {allowed_values}")
        return 'NA', False, f'invalid_{str_value}_replaced_with_NA'
    
    def validate_prediction(self, prediction_data: Dict, template_config: Dict) -> Tuple[Dict, Dict]:
        """
        Validate and normalize a prediction according to template configuration.
        
        Returns:
            Tuple of (validated_data, validation_log)
        """
        validated = {}
        log = {
            'timestamp': datetime.now().isoformat(),
            'changes': [],
            'stats': {'total_fields': 0, 'valid': 0, 'normalized': 0, 'invalid': 0}
        }
        
        field_definitions = template_config.get('field_definitions', {})
        
        for field_name, field_config in field_definitions.items():
            log['stats']['total_fields'] += 1
            
            # Get the value from prediction data
            original_value = prediction_data.get(field_name)
            
            # Normalize the value
            normalized_value, is_valid, action = self.normalize_value(original_value, field_config)
            
            # Store the normalized value
            validated[field_name] = normalized_value
            
            # Log the change if any
            if action != 'kept_valid' and action != 'kept_na':
                log['changes'].append({
                    'field': field_name,
                    'original': original_value,
                    'normalized': normalized_value,
                    'action': action
                })
                
                if 'normalized' in action:
                    log['stats']['normalized'] += 1
                elif 'invalid' in action:
                    log['stats']['invalid'] += 1
            else:
                log['stats']['valid'] += 1
        
        return validated, log
    
    def validate_job_predictions(self, job_id: str, db_path: str) -> Dict:
        """
        Validate all predictions for a specific job.
        
        Returns:
            Summary statistics of the validation process
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get all predictions for this job
            cursor.execute("""
                SELECT id, binomial_name, result, system_template, user_template,
                       gram_staining, motility, aerophilicity, extreme_environment_tolerance,
                       biofilm_formation, animal_pathogenicity, biosafety_level,
                       health_association, host_association, plant_pathogenicity,
                       spore_formation, hemolysis, cell_shape, knowledge_group,
                       raw_gram_staining, raw_motility, raw_aerophilicity, 
                       raw_extreme_environment_tolerance, raw_biofilm_formation,
                       raw_animal_pathogenicity, raw_biosafety_level,
                       raw_health_association, raw_host_association, raw_plant_pathogenicity,
                       raw_spore_formation, raw_hemolysis, raw_cell_shape, raw_knowledge_group,
                       raw_data_preserved
                FROM processing_results
                WHERE job_id = ? AND status = 'completed'
            """, (job_id,))
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.info(f"No completed predictions found for job {job_id}")
                return {'status': 'no_predictions', 'count': 0}
            
            # Get template config
            first_row = rows[0]
            template_config = self.get_template_config(
                first_row['system_template'], 
                first_row['user_template']
            )
            
            if not template_config:
                logger.error(f"No validation config found for job {job_id}")
                return {'status': 'no_config', 'count': 0}
            
            validation_summary = {
                'job_id': job_id,
                'total_predictions': len(rows),
                'validated': 0,
                'changes_made': 0,
                'fields_normalized': 0,
                'fields_invalidated': 0,
                'start_time': datetime.now().isoformat()
            }
            
            field_definitions = template_config.get('field_definitions', {})
            
            for row in rows:
                row_dict = dict(row)
                prediction_id = row_dict['id']
                changes_made = False
                row_log = []
                
                # Validate each phenotype field
                update_fields = []
                update_values = []
                raw_update_fields = []
                raw_update_values = []
                
                for field_name in field_definitions.keys():
                    if field_name in row_dict:
                        # First preserve the raw value if not already preserved
                        raw_field_name = f"raw_{field_name}"
                        if raw_field_name not in row_dict or row_dict.get('raw_data_preserved') != 1:
                            raw_update_fields.append(f"{raw_field_name} = ?")
                            raw_update_values.append(row_dict[field_name])
                        
                        # Get the original value (use raw if available, otherwise current)
                        original_value = row_dict.get(raw_field_name, row_dict[field_name])
                        
                        # Normalize based on the raw/original value
                        normalized_value, is_valid, action = self.normalize_value(
                            original_value, 
                            field_definitions[field_name]
                        )
                        
                        # Convert array to JSON string for storage
                        if isinstance(normalized_value, list):
                            normalized_value = json.dumps(normalized_value)
                        
                        # Always update the normalized field (even if unchanged, to ensure consistency)
                        update_fields.append(f"{field_name} = ?")
                        update_values.append(normalized_value)
                        
                        if original_value != normalized_value:
                            changes_made = True
                            
                            row_log.append({
                                'field': field_name,
                                'original': original_value,
                                'normalized': normalized_value,
                                'action': action
                            })
                            
                            if 'normalized' in action:
                                validation_summary['fields_normalized'] += 1
                            elif 'invalid' in action:
                                validation_summary['fields_invalidated'] += 1
                            
                            logger.info(f"Prediction {prediction_id} - {row_dict['binomial_name']}: "
                                      f"{field_name} '{original_value}' -> '{normalized_value}' ({action})")
                
                # Update the database
                # First preserve raw data if needed
                if raw_update_fields:
                    raw_update_values.append(prediction_id)
                    cursor.execute(f"""
                        UPDATE processing_results
                        SET {', '.join(raw_update_fields)},
                            raw_data_preserved = 1
                        WHERE id = ?
                    """, raw_update_values)
                
                # Then update normalized values and validation status
                if update_fields:
                    update_values.extend([
                        'validated',  # validation_status
                        datetime.now(),  # validation_date
                        json.dumps(row_log) if row_log else None,  # validation_log
                        prediction_id
                    ])
                    
                    cursor.execute(f"""
                        UPDATE processing_results
                        SET {', '.join(update_fields)},
                            validation_status = ?,
                            validation_date = ?,
                            validation_log = ?
                        WHERE id = ?
                    """, update_values)
                    
                    if changes_made:
                        validation_summary['changes_made'] += 1
                else:
                    # Mark as validated even if no changes needed
                    cursor.execute("""
                        UPDATE processing_results
                        SET validation_status = 'validated',
                            validation_date = ?
                        WHERE id = ?
                    """, (datetime.now(), prediction_id))
                
                validation_summary['validated'] += 1
            
            conn.commit()
            validation_summary['end_time'] = datetime.now().isoformat()
            validation_summary['status'] = 'success'
            
            logger.info(f"Validation completed for job {job_id}: {validation_summary}")
            return validation_summary
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error validating job {job_id}: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            conn.close()
    
    def validate_all_unvalidated(self, db_path: str) -> Dict:
        """
        Validate all unvalidated predictions in the database.
        
        Returns:
            Summary statistics of the validation process
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Get all unique jobs with unvalidated predictions
            cursor.execute("""
                SELECT DISTINCT job_id
                FROM processing_results
                WHERE (validation_status IS NULL OR validation_status = 'unvalidated')
                AND status = 'completed'
            """)
            
            job_ids = [row[0] for row in cursor.fetchall()]
            
            if not job_ids:
                logger.info("No unvalidated predictions found")
                return {'status': 'no_unvalidated', 'jobs_processed': 0}
            
            overall_summary = {
                'status': 'success',
                'jobs_processed': 0,
                'total_predictions_validated': 0,
                'total_changes_made': 0,
                'job_summaries': []
            }
            
            for job_id in job_ids:
                logger.info(f"Validating job: {job_id}")
                job_summary = self.validate_job_predictions(job_id, db_path)
                
                if job_summary.get('status') == 'success':
                    overall_summary['jobs_processed'] += 1
                    overall_summary['total_predictions_validated'] += job_summary.get('validated', 0)
                    overall_summary['total_changes_made'] += job_summary.get('changes_made', 0)
                    overall_summary['job_summaries'].append(job_summary)
            
            logger.info(f"Validation completed: {overall_summary}")
            return overall_summary
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            conn.close()
    
    def get_validation_stats(self, db_path: str) -> Dict:
        """Get statistics about validation status in the database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN validation_status = 'validated' THEN 1 ELSE 0 END) as validated,
                    SUM(CASE WHEN validation_status IS NULL OR validation_status = 'unvalidated' THEN 1 ELSE 0 END) as unvalidated,
                    SUM(CASE WHEN validation_status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM processing_results
                WHERE status = 'completed'
            """)
            
            row = cursor.fetchone()
            
            return {
                'total': row[0] or 0,
                'validated': row[1] or 0,
                'unvalidated': row[2] or 0,
                'failed': row[3] or 0
            }
            
        except Exception as e:
            logger.error(f"Error getting validation stats: {e}")
            return {'error': str(e)}
        finally:
            conn.close()


# Main execution for testing
if __name__ == "__main__":
    validator = PredictionValidator()
    
    # Test validation configs loaded
    print(f"Loaded {len(validator.validation_configs)} validation configs")
    
    # Example of validating a single prediction
    test_prediction = {
        'gram_staining': 'positive',  # Should normalize to 'gram stain positive'
        'motility': 'true',  # Should normalize to 'TRUE'
        'aerophilicity': 'facultatively anaerobic',  # Should stay as is (valid)
        'biosafety_level': 'BSL-2',  # Should normalize to 'biosafety level 2'
        'invalid_field': 'some_value'  # Should be ignored
    }
    
    # Get a phenotype template config for testing
    for key, config in validator.validation_configs.items():
        if 'phenotype' in key:
            validated, log = validator.validate_prediction(test_prediction, config)
            print(f"\nTest validation result:")
            print(f"Original: {test_prediction}")
            print(f"Validated: {validated}")
            print(f"Log: {json.dumps(log, indent=2)}")
            break