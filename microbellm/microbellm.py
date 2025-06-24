#!/usr/bin/env python

# Import necessary libraries
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pandas as pd
from tqdm import tqdm
from microbellm.utils import read_template_from_file, write_batch_jsonl, import_results_from_csv
from microbellm.predict import predict_binomial_name
from microbellm import config
import sqlite3
from datetime import datetime
from pathlib import Path

def read_genes_from_file(file_path):
    """
    Reads gene names from a file and returns them as a list.
    
    Args:
        file_path (str): Path to the file containing gene names.
    
    Returns:
        list: List of gene names.
    """
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def check_api_key():
    """Check if the OpenRouter API key is available."""
    if not config.OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY is not set.")
        print("Please set it as an environment variable or in microbellm/config.py")
        print("Example: export OPENROUTER_API_KEY='your-api-key'")
        return False
    return True

def batch_prediction(args):
    """
    Adds a batch prediction job to the database queue.
    """
    if not check_api_key():
        return

    # Read species names from the input CSV
    try:
        df = pd.read_csv(args.input_csv)
        if 'Binomial.name' not in df.columns:
            print("Error: Input CSV must have a 'Binomial.name' column.")
            return
        # Ensure species file path is stored for the database
        species_file_path = Path(args.input_csv).name
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    # Prepare for database interaction
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    # Get template paths
    system_template_path = args.system_template
    user_template_path = args.user_template
    
    # Check if combination already exists
    cursor.execute('''
        SELECT id FROM combinations 
        WHERE species_file = ? AND model = ? AND system_template = ? AND user_template = ?
    ''', (species_file_path, args.model, system_template_path, user_template_path))

    if cursor.fetchone():
        print(f"Combination already exists for {species_file_path}, {args.model}, and specified templates. Skipping.")
    else:
        # Create new combination job
        total_species = len(df['Binomial.name'].unique())
        cursor.execute('''
            INSERT INTO combinations 
            (species_file, model, system_template, user_template, total_species, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (species_file_path, args.model, system_template_path, user_template_path, total_species, datetime.now(), 'pending'))
        conn.commit()
        print(f"Successfully added job for {species_file_path} with model {args.model}.")
        print("You can start and monitor this job from the web interface.")

    conn.close()

def single_prediction(args):
    """
    Runs a single prediction on a binomial name.
    """
    if not check_api_key():
        return

    # Read system and user templates
    system_template = read_template_from_file(args.system_template)
    user_template = read_template_from_file(args.user_template)

    print(f"Processing: {args.binomial_name}")
    print(f"Model: {args.model}")

    # Run prediction
    result = predict_binomial_name(
        args.binomial_name,
        system_template,
        user_template,
        args.model,
        args.temperature,
        args.verbose
    )

    if result:
        # Save to CSV
        df = pd.DataFrame([result])
        df.to_csv(args.output_csv, index=False)
        print(f"Result saved to: {args.output_csv}")
        
        # Also print to console
        print("\nPrediction Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to get prediction.")

def main():
    """Main function to parse command-line arguments and execute the tool."""
    parser = argparse.ArgumentParser(description="MicrobeLLM: Evaluate LLMs on microbial phenotype prediction.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Batch prediction command
    parser_batch = subparsers.add_parser('batch', help='Run batch predictions on multiple species')
    parser_batch.add_argument('--input_csv', type=str, required=True,
                            help='Input CSV file with Binomial.name column')
    parser_batch.add_argument('--output_csv', type=str, required=True,
                            help='Output CSV file for results')
    parser_batch.add_argument('--system_template', type=str, required=True,
                            help='Path to system message template file')
    parser_batch.add_argument('--user_template', type=str, required=True,
                            help='Path to user message template file')
    parser_batch.add_argument('--model', type=str, default=config.POPULAR_MODELS[0],
                             help=f'Model to use for predictions (default: {config.POPULAR_MODELS[0]})')
    parser_batch.add_argument('--temperature', type=float, default=0.0,
                            help='Temperature for model predictions (default: 0.0)')
    parser_batch.add_argument('--threads', type=int, default=4, help='Number of threads for parallel processing.')
    parser_batch.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    
    # Single prediction command
    parser_single = subparsers.add_parser('single', help='Predict phenotype for a single bacterial species.')
    parser_single.add_argument('--binomial_name', type=str, required=True,
                             help='Binomial name of the species to predict')
    parser_single.add_argument('--output_csv', type=str, required=True,
                             help='Output CSV file for the result')
    parser_single.add_argument('--system_template', type=str, required=True,
                             help='Path to system message template file')
    parser_single.add_argument('--user_template', type=str, required=True,
                             help='Path to user message template file')
    parser_single.add_argument('--model', type=str, default=config.POPULAR_MODELS[0],
                             help=f'Model to use for predictions (default: {config.POPULAR_MODELS[0]})')
    parser_single.add_argument('--temperature', type=float, default=0.0,
                             help='Temperature for model predictions (default: 0.0)')
    parser_single.add_argument('--verbose', action='store_true',
                             help='Enable verbose output')
    
    # Import results command
    parser_import = subparsers.add_parser('import', help='Import results from a CSV file into the database.')
    parser_import.add_argument('--csv', required=True, help='Path to the CSV file to import.')
    parser_import.add_argument('--verbose', action='store_true', help='Enable verbose logging.')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'batch':
        batch_prediction(args)
    elif args.command == 'single':
        single_prediction(args)
    elif args.command == 'import':
        from microbellm.importer import import_results_from_csv
        import_results_from_csv(args.csv)

if __name__ == '__main__':
    main()