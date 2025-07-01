#!/usr/bin/env python3
"""
Phenotype CSV Validation Script

This script validates a CSV file containing phenotype data against a JSON template
that defines allowed values and validation rules for each phenotype field.
"""

import json
import csv
import sys
import re
from typing import Dict, List, Any, Tuple, Optional

def load_template(template_path: str) -> Dict[str, Any]:
    """Load and parse the JSON template file."""
    try:
        with open(template_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading template: {e}")
        sys.exit(1)

def normalize_value(value: str, field_config: Dict[str, Any]) -> str:
    """Normalize a value according to the field's validation rules."""
    if not value or value.strip() == "":
        return ""
    
    value = value.strip()
    
    # Handle case sensitivity
    validation_rules = field_config.get('validation_rules', {})
    if not validation_rules.get('case_sensitive', True):
        value = value.lower()
    
    # Apply normalization mappings
    normalize_mapping = validation_rules.get('normalize_mapping', {})
    for canonical_value, aliases in normalize_mapping.items():
        for alias in aliases:
            if not validation_rules.get('case_sensitive', True):
                alias = alias.lower()
            if value == alias:
                return canonical_value
    
    return value

def validate_field_value(value: str, field_name: str, field_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single field value against its configuration.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Handle empty/NA values
    if not value or value.strip() == "" or value.strip().upper() == "NA":
        # Check if field is required
        if field_config.get('required', False):
            errors.append(f"Required field '{field_name}' is empty or NA")
            return False, errors
        else:
            return True, []  # Optional field can be empty/NA
    
    # Normalize the value
    normalized_value = normalize_value(value, field_config)
    allowed_values = field_config.get('allowed_values', [])
    
    # Handle array type fields (like aerophilicity)
    if field_config.get('type') == 'array':
        # Check if single value is allowed for array fields
        validation_rules = field_config.get('validation_rules', {})
        if validation_rules.get('allow_single_value', False):
            # Single value case
            if normalized_value not in allowed_values:
                errors.append(f"Value '{value}' not in allowed values: {allowed_values}")
                return False, errors
        else:
            # Multiple values case - would need to parse comma-separated or similar
            # For now, treat as single value
            if normalized_value not in allowed_values:
                errors.append(f"Value '{value}' not in allowed values: {allowed_values}")
                return False, errors
    else:
        # Regular single-value field
        if normalized_value not in allowed_values:
            errors.append(f"Value '{value}' not in allowed values: {allowed_values}")
            return False, errors
    
    return True, []

def validate_csv(csv_path: str, template: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate the entire CSV file against the template.
    
    Returns:
        List of validation errors with location information
    """
    field_definitions = template.get('field_definitions', {})
    validation_errors = []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 because header is row 1
                binomial_name = row.get('binomial_name', f'Row {row_num}')
                
                # Validate each phenotype field
                for field_name, field_config in field_definitions.items():
                    if field_name in row:
                        value = row[field_name]
                        is_valid, errors = validate_field_value(value, field_name, field_config)
                        
                        if not is_valid:
                            for error in errors:
                                validation_errors.append({
                                    'row': row_num,
                                    'binomial_name': binomial_name,
                                    'field': field_name,
                                    'value': value,
                                    'error': error,
                                    'allowed_values': field_config.get('allowed_values', [])
                                })
    
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    return validation_errors

def print_validation_report(errors: List[Dict[str, Any]], csv_path: str, template_path: str):
    """Print a comprehensive validation report."""
    print(f"Phenotype Validation Report")
    print(f"=" * 50)
    print(f"CSV File: {csv_path}")
    print(f"Template: {template_path}")
    print(f"Total Validation Errors: {len(errors)}")
    print()
    
    if not errors:
        print("✅ All phenotype values are valid according to the template!")
        return
    
    # Group errors by field for summary
    errors_by_field = {}
    for error in errors:
        field = error['field']
        if field not in errors_by_field:
            errors_by_field[field] = []
        errors_by_field[field].append(error)
    
    # Print summary by field
    print("Summary by Field:")
    print("-" * 30)
    for field, field_errors in sorted(errors_by_field.items()):
        print(f"{field}: {len(field_errors)} errors")
    print()
    
    # Print detailed errors
    print("Detailed Errors:")
    print("-" * 30)
    for i, error in enumerate(errors, 1):
        print(f"{i}. Row {error['row']} ({error['binomial_name']})")
        print(f"   Field: {error['field']}")
        print(f"   Value: '{error['value']}'")
        print(f"   Error: {error['error']}")
        print(f"   Allowed: {error['allowed_values']}")
        print()

def main():
    """Main function to run the validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate phenotype CSV against JSON template')
    parser.add_argument('csv_file', help='Path to the CSV file to validate')
    parser.add_argument('template_file', help='Path to the JSON template file')
    parser.add_argument('--output', '-o', help='Output file for validation report (optional)')
    
    args = parser.parse_args()
    
    # Load template
    template = load_template(args.template_file)
    
    # Validate CSV
    errors = validate_csv(args.csv_file, template)
    
    # Print report
    if args.output:
        original_stdout = sys.stdout
        with open(args.output, 'w') as f:
            sys.stdout = f
            print_validation_report(errors, args.csv_file, args.template_file)
        sys.stdout = original_stdout
        print(f"Validation report written to: {args.output}")
        print(f"Total errors found: {len(errors)}")
    else:
        print_validation_report(errors, args.csv_file, args.template_file)
    
    # Exit with error code if validation failed
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 