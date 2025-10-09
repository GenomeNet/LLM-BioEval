#!/usr/bin/env python3
"""
List all tail morphology species and their classification by different models.
"""

import urllib.request
import json
from typing import Dict, List


def fetch_api_data(endpoint: str, api_url: str = 'http://localhost:5050') -> Dict:
    """Fetch data from API endpoint."""
    full_url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urllib.request.urlopen(full_url, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def normalize_value(value) -> str:
    """Normalize values for comparison."""
    if value is None or value == '' or str(value).lower() in ['n/a', 'na', 'null', 'none', 'nan', 'unknown']:
        return None
    return str(value).strip().lower()


def main():
    print("=" * 100)
    print("COMPLETE LIST: TAIL MORPHOLOGY SPECIES AND MODEL PREDICTIONS")
    print("=" * 100)

    # Fetch data
    print("\nFetching data...")
    pred_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt')
    predictions = pred_data.get('data', [])

    # Fetch ground truth
    gt_data = fetch_api_data('/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000')

    # Get all tail species from ground truth
    tail_species = []
    for item in gt_data.get('data', []):
        cell_shape = normalize_value(item.get('cell_shape'))
        if cell_shape == 'tail':
            tail_species.append(item['binomial_name'])

    tail_species.sort()  # Sort alphabetically

    # Models to check
    models_to_check = [
        'openai/gpt-4.1-nano',
        'openai/gpt-5',
        'openai/gpt-4o',
        'google/gemini-2.5-pro',
        'anthropic/claude-3.5-sonnet',
        'deepseek/deepseek-r1'
    ]

    # Get predictions for each species and model
    species_predictions = {}

    for species_name in tail_species:
        species_lower = species_name.lower()
        species_predictions[species_name] = {}

        for model in models_to_check:
            # Find prediction for this species by this model
            model_preds = [p for p in predictions if p['model'] == model]

            for pred in model_preds:
                if pred.get('binomial_name', '').lower() == species_lower:
                    pred_val = normalize_value(pred.get('cell_shape'))
                    species_predictions[species_name][model] = pred_val
                    break

    # Print header
    print("\n" + "=" * 100)
    print("SPECIES-BY-SPECIES BREAKDOWN")
    print("=" * 100)
    print("\nLegend: ✓ = correct (tail), ✗ = wrong (bacillus/other), - = no prediction\n")

    # Create summary table header
    header = "Species".ljust(45) + " | "
    for model in models_to_check:
        model_short = model.split('/')[-1][:12]  # Shorten model names
        header += model_short.center(12) + " | "
    print(header)
    print("-" * len(header))

    # Track statistics
    model_correct = {m: 0 for m in models_to_check}
    model_wrong = {m: 0 for m in models_to_check}

    # Print each species
    for species_name in tail_species:
        row = species_name[:45].ljust(45) + " | "

        for model in models_to_check:
            pred = species_predictions[species_name].get(model, '-')

            if pred == 'tail':
                symbol = '✓'
                model_correct[model] += 1
            elif pred == '-':
                symbol = '-'
            else:
                symbol = f'✗({pred[:3]})'  # Show what it was misclassified as
                model_wrong[model] += 1

            row += symbol.center(12) + " | "

        print(row)

    # Print summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    print("\n" + "Model Performance Summary".center(100))
    print("-" * 100)

    header2 = "Model".ljust(30) + " | " + "Correct".center(15) + " | " + "Wrong".center(15) + " | " + "Accuracy".center(15)
    print(header2)
    print("-" * len(header2))

    for model in models_to_check:
        model_short = model.split('/')[-1]
        correct = model_correct[model]
        wrong = model_wrong[model]
        total = correct + wrong

        if total > 0:
            accuracy = correct / total * 100
            row = model_short.ljust(30) + " | "
            row += str(correct).center(15) + " | "
            row += str(wrong).center(15) + " | "
            row += f"{accuracy:.1f}%".center(15)
            print(row)

    # List species that ONLY GPT-4.1-nano gets right
    print("\n" + "=" * 100)
    print("SPECIES CORRECTLY CLASSIFIED ONLY BY GPT-4.1-NANO")
    print("=" * 100)

    nano_only_correct = []

    for species_name in tail_species:
        # Check if nano got it right
        if species_predictions[species_name].get('openai/gpt-4.1-nano') == 'tail':
            # Check if ALL other models got it wrong
            others_all_wrong = True
            for model in models_to_check:
                if model == 'openai/gpt-4.1-nano':
                    continue
                if species_predictions[species_name].get(model) == 'tail':
                    others_all_wrong = False
                    break

            if others_all_wrong:
                nano_only_correct.append(species_name)

    if nano_only_correct:
        print(f"\n{len(nano_only_correct)} species correctly identified as 'tail' ONLY by GPT-4.1-nano:\n")

        # Print in columns for readability
        cols = 2
        rows_per_col = (len(nano_only_correct) + cols - 1) // cols

        for i in range(rows_per_col):
            row = ""
            for j in range(cols):
                idx = i + j * rows_per_col
                if idx < len(nano_only_correct):
                    row += f"{idx+1:3}. {nano_only_correct[idx]:<45}"
            print(row)

    # Find the species that GPT-4.1-nano misses
    print("\n" + "=" * 100)
    print("SPECIES MISCLASSIFIED BY GPT-4.1-NANO")
    print("=" * 100)

    nano_wrong = []
    for species_name in tail_species:
        pred = species_predictions[species_name].get('openai/gpt-4.1-nano')
        if pred and pred != 'tail':
            nano_wrong.append((species_name, pred))

    if nano_wrong:
        print(f"\n{len(nano_wrong)} species misclassified by GPT-4.1-nano:\n")
        for species, pred in nano_wrong:
            print(f"  - {species} → predicted as '{pred}'")
    else:
        print("\nNo misclassifications by GPT-4.1-nano!")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()