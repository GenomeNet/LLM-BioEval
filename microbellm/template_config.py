#!/usr/bin/env python
"""
Template Configuration System for MicrobeLLM

This module loads template validation configurations from JSON files and validates
LLM responses according to the specifications defined in those files.
"""

import json
import re
from pathlib import Path
import os

class TemplateValidator:
    """Validator class that loads configuration from JSON files"""
    
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load validation configuration from JSON file"""
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading template config from {self.config_file_path}: {e}")
            return None
    
    def validate_response(self, response_data):
        """
        Validate and normalize a response according to the JSON configuration
        
        Args:
            response_data (dict): Parsed JSON response from LLM
            
        Returns:
            tuple: (validated_data, errors)
        """
        if not self.config:
            return {}, ["Configuration file not loaded"]
        
        result = {}
        errors = []
        
        field_definitions = self.config.get('field_definitions', {})
        required_fields = self.config.get('expected_response', {}).get('required_fields', [])
        
        # Check required fields
        for field_name in required_fields:
            if field_name not in response_data:
                field_def = field_definitions.get(field_name, {})
                error_msg = field_def.get('validation_error_messages', {}).get('missing', f"Required field '{field_name}' is missing")
                errors.append(error_msg)
                continue
        
        # Validate and normalize each field
        for field_name, field_def in field_definitions.items():
            value = response_data.get(field_name)
            
            if value is None:
                if field_def.get('required', False):
                    continue  # Already handled above
                else:
                    result[field_name] = None
                    continue
            
            try:
                normalized_value = self._validate_field(field_name, value, field_def)
                result[field_name] = normalized_value
            except ValueError as e:
                errors.append(str(e))
        
        return result, errors
    
    def _validate_field(self, field_name, value, field_def):
        """Validate and normalize a single field according to its definition"""
        
        # Check type
        expected_type = field_def.get('type', 'string')
        if expected_type == 'string' and not isinstance(value, str):
            error_msg = field_def.get('validation_error_messages', {}).get('wrong_type', f"Field '{field_name}' must be a string")
            raise ValueError(error_msg)
        
        # Convert to string and apply basic transformations
        str_value = str(value)
        validation_rules = field_def.get('validation_rules', {})
        
        if validation_rules.get('trim_whitespace', True):
            str_value = str_value.strip()
        
        if not validation_rules.get('case_sensitive', True):
            str_value = str_value.lower()
        
        # Apply normalization mapping
        normalize_mapping = validation_rules.get('normalize_mapping', {})
        normalized_value = None
        
        for canonical_value, variants in normalize_mapping.items():
            if str_value in [v.lower() if not validation_rules.get('case_sensitive', True) else v for v in variants]:
                normalized_value = canonical_value
                break
        
        # If no mapping found, check if it's in allowed values directly
        if normalized_value is None:
            allowed_values = field_def.get('allowed_values', [])
            if allowed_values:
                # Check against allowed values (case insensitive if specified)
                for allowed in allowed_values:
                    if validation_rules.get('case_sensitive', True):
                        if str_value == allowed:
                            normalized_value = allowed
                            break
                    else:
                        if str_value.lower() == allowed.lower():
                            normalized_value = allowed
                            break
        
        # If still no match and we have allowed values, it's an error
        if normalized_value is None and field_def.get('allowed_values'):
            error_msg = field_def.get('validation_error_messages', {}).get('invalid_value', 
                f"Invalid value for '{field_name}'. Expected one of: {', '.join(field_def.get('allowed_values', []))}")
            raise ValueError(error_msg)
        
        # If no allowed values restriction, use the cleaned value
        if normalized_value is None:
            normalized_value = str_value
            
        return normalized_value
    
    def get_template_info(self):
        """Get template information from config"""
        if not self.config:
            return {}
        return self.config.get('template_info', {})


def find_validation_config_for_template(user_template_path):
    """
    Find the corresponding validation config file for a template
    
    Args:
        user_template_path (str): Path to the user template file
        
    Returns:
        str or None: Path to validation config file if found
    """
    template_path = Path(user_template_path)
    template_name = template_path.stem  # Get filename without extension
    
    # Look for validation config in templates/validation/ directory
    config_dir = template_path.parent.parent / 'validation'
    config_file = config_dir / f"{template_name}.json"
    
    if config_file.exists():
        return str(config_file)
    
    # Fallback: look in same directory as template
    fallback_config = template_path.parent / f"{template_name}.json"
    if fallback_config.exists():
        return str(fallback_config)
    
    return None


def validate_template_response_from_file(user_template_path, response_data):
    """
    Validate a response using JSON configuration file for the template
    
    Args:
        user_template_path (str): Path to the user template file
        response_data (dict): Parsed JSON response from LLM
        
    Returns:
        tuple: (validated_data, errors, validator)
    """
    config_path = find_validation_config_for_template(user_template_path)
    
    if not config_path:
        # Fallback to old hardcoded validation
        from microbellm.utils import normalize_knowledge_level
        
        # Basic knowledge template detection
        try:
            with open(user_template_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            if 'knowledge_group' in content:
                # Apply basic knowledge validation
                validated_data = {}
                errors = []
                
                if 'knowledge_group' in response_data:
                    try:
                        normalized = normalize_knowledge_level(response_data['knowledge_group'])
                        validated_data['knowledge_group'] = normalized
                    except Exception as e:
                        errors.append(f"Knowledge level validation error: {str(e)}")
                
                return validated_data, errors, None
            
        except Exception:
            pass
        
        # Return original data if no validation available
        return response_data, [], None
    
    # Use JSON-based validation
    validator = TemplateValidator(config_path)
    validated_data, errors = validator.validate_response(response_data)
    
    return validated_data, errors, validator


def get_all_template_validation_configs():
    """
    Get all available template validation configurations
    
    Returns:
        dict: Mapping of template names to their validation config paths
    """
    configs = {}
    
    # Look in templates/validation/ directory
    validation_dir = Path('templates/validation')
    if validation_dir.exists():
        for config_file in validation_dir.glob('*.json'):
            template_name = config_file.stem
            configs[template_name] = str(config_file)
    
    return configs


def detect_template_type_from_config(user_template_path):
    """
    Detect template type from validation config file
    
    Args:
        user_template_path (str): Path to user template file
        
    Returns:
        str: Template type ('knowledge', 'phenotype', etc.) or 'unknown'
    """
    config_path = find_validation_config_for_template(user_template_path)
    
    if config_path:
        try:
            validator = TemplateValidator(config_path)
            template_info = validator.get_template_info()
            return template_info.get('type', 'unknown')
        except Exception:
            pass
    
    # Fallback to content-based detection
    try:
        with open(user_template_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        if any(indicator in content for indicator in ['knowledge_group', 'knowledge_level', 'knowledge level']):
            return 'knowledge'
        elif any(indicator in content for indicator in ['gram_staining', 'motility', 'biosafety']):
            return 'phenotype'
    except Exception:
        pass
    
    return 'unknown' 