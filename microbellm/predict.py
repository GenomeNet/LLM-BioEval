# predict.py

# Import necessary libraries
import argparse
import json
import pandas as pd
from colorama import Fore, Style
from microbellm.utils import parse_response, OpenRouterProvider
import sys
from tqdm import tqdm
from datetime import datetime

def predict_binomial_name(binomial_name, system_template, user_template, model, temperature=0.0, verbose=False):
    """
    Predicts various characteristics of a bacterial species using an LLM.
    
    Args:
        binomial_name (str): The binomial name of the bacterial species.
        system_template (str): The system message template.
        user_template (str): The user message template.
        model (str): The model to use for prediction.
        temperature (float): The temperature for the model.
        verbose (bool): Whether to print verbose output.
    
    Returns:
        dict: A dictionary containing the prediction results.
    """
    
    # Format the user message with the binomial name
    user_message = user_template.replace("{binomial_name}", binomial_name)
    
    # Prepare messages for the API
    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user_message}
    ]
    
    if verbose:
        print(f"Predicting for: {binomial_name}")
        print(f"Using model: {model}")
    
    # Query the API using the provider
    provider = OpenRouterProvider()
    response = provider.query(messages, model, temperature, verbose)
    
    if not response:
        if verbose:
            print(f"Failed to get response for {binomial_name}")
        return None
    
    # Parse the response
    parsed_result = parse_response(response)
    
    if not parsed_result:
        if verbose:
            print(f"Failed to parse response for {binomial_name}")
        return None
    
    # Add metadata to the result
    result = {
        'binomial_name': binomial_name,
        'model': model,
        'temperature': temperature,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'raw_response': response
    }
    
    # Add parsed fields
    result.update(parsed_result)
    
    if verbose:
        print(f"Successfully processed {binomial_name}")
    
    return result

def main():
    """
    Main function to parse arguments and execute the prediction command.
    """
    parser = argparse.ArgumentParser(description="Microbe LLM Prediction Tool")
    parser.add_argument('--binomial_name', type=str, help='Binomial name or path to file containing binomial names')
    parser.add_argument('--column_name', type=str, default='Taxon_name', help='Column name in the file for binomial names')
    parser.add_argument('--output', type=str, help='Output file path to save the predictions as CSV')
    parser.add_argument('--is_file', action='store_true', help='Indicate if the binomial_name argument points to a file')
    parser.add_argument('--use_genes', action='store_true', help='Indicate if gene list should be included in the query')
    parser.add_argument('--gene_list', type=str, nargs='+', help='List of genes to include in the query')
    parser.add_argument('--model_host', type=str, choices=['openrouter', 'openai'], default='openrouter', help="Select the model host (default: openrouter)")
    parser.add_argument('--batchoutput', action='store_true', help='Generate batch output file for OpenAI processing')
    args = parser.parse_args()

    if args.is_file:
        # Read binomial names from the provided file
        data = pd.read_csv(args.binomial_name, delimiter=';')
        binomial_names = data[args.column_name].dropna().unique()
        for name in binomial_names:
            if args.use_genes:
                gene_list = args.gene_list
            else:
                gene_list = None
            predict_binomial_name(name, args.system_template, args.user_template, args.model[0], args.temperature, args.verbose)

if __name__ == "__main__":
    main()