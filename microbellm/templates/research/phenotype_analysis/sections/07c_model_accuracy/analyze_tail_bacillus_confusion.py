#!/usr/bin/env python3
"""
Analyze species with tail morphology that are misclassified as bacillus across models.
"""

import urllib.request
import json
import numpy as np
from typing import Dict, List, Set
from collections import defaultdict


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
    print("=" * 80)
    print("ANALYSIS: TAIL → BACILLUS MISCLASSIFICATION")
    print("=" * 80)

    # Fetch data
    print("\nFetching data...")
    pred_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt')
    predictions = pred_data.get('data', [])

    # Fetch ground truth
    gt_data = fetch_api_data('/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000')

    # Create ground truth mapping
    gt_map = {}
    tail_species = []  # List of species that have tail morphology

    for item in gt_data.get('data', []):
        species_name = item['binomial_name'].lower()
        gt_map[species_name] = item

        # Check if this species has tail morphology
        cell_shape = normalize_value(item.get('cell_shape'))
        if cell_shape == 'tail':
            tail_species.append(species_name)

    print(f"Found {len(tail_species)} species with tail morphology in ground truth")

    # Analyze misclassifications for each model
    models_to_analyze = [
        'openai/gpt-5',
        'openai/gpt-4.1-nano',
        'openai/gpt-4o',
        'openai/gpt-oss-120b',
        'google/gemini-2.5-pro',
        'anthropic/claude-3.5-sonnet',
        'deepseek/deepseek-r1'
    ]

    model_misclassifications = {}

    for model in models_to_analyze:
        model_preds = [p for p in predictions if p['model'] == model]

        # Track misclassifications for this model
        misclassified_species = []
        correctly_classified = []
        other_misclassifications = defaultdict(list)

        for pred in model_preds:
            species = pred.get('binomial_name', '').lower()

            # Check if this is a tail species
            if species in tail_species:
                pred_val = normalize_value(pred.get('cell_shape'))

                if pred_val == 'bacillus':
                    misclassified_species.append(species)
                elif pred_val == 'tail':
                    correctly_classified.append(species)
                elif pred_val:
                    other_misclassifications[pred_val].append(species)

        model_misclassifications[model] = {
            'tail_to_bacillus': misclassified_species,
            'correct': correctly_classified,
            'other': other_misclassifications,
            'total_tail': len(misclassified_species) + len(correctly_classified) +
                         sum(len(v) for v in other_misclassifications.values())
        }

    # Print model comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: TAIL CLASSIFICATION")
    print("=" * 80)
    print(f"\n{'Model':<30} {'Correct':<12} {'→Bacillus':<12} {'→Other':<12} {'Accuracy':<10}")
    print("-" * 80)

    for model in models_to_analyze:
        stats = model_misclassifications[model]
        total = stats['total_tail']
        if total > 0:
            correct = len(stats['correct'])
            to_bacillus = len(stats['tail_to_bacillus'])
            to_other = sum(len(v) for v in stats['other'].values())
            accuracy = correct / total

            model_short = model.split('/')[-1]
            print(f"{model_short:<30} {correct:<12} {to_bacillus:<12} {to_other:<12} {accuracy:.1%}")

    # Detailed analysis for GPT-5
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: GPT-5 TAIL→BACILLUS MISCLASSIFICATIONS")
    print("=" * 80)

    gpt5_misclassified = model_misclassifications['openai/gpt-5']['tail_to_bacillus']

    if gpt5_misclassified:
        print(f"\nGPT-5 misclassified {len(gpt5_misclassified)} tail species as bacillus:")
        print("-" * 80)

        # Get additional information about these species
        species_info = []
        for species in gpt5_misclassified[:20]:  # Show first 20
            if species in gt_map:
                info = gt_map[species]
                species_info.append({
                    'name': info['binomial_name'],
                    'gram': normalize_value(info.get('gram_staining')),
                    'motility': normalize_value(info.get('motility')),
                    'spore': normalize_value(info.get('spore_formation'))
                })

        # Sort by name for readability
        species_info.sort(key=lambda x: x['name'])

        print(f"\n{'Species':<40} {'Gram':<15} {'Motility':<15} {'Spore':<10}")
        print("-" * 80)

        for info in species_info:
            gram = info['gram'] or 'unknown'
            motility = info['motility'] or 'unknown'
            spore = info['spore'] or 'unknown'
            print(f"{info['name']:<40} {gram:<15} {motility:<15} {spore:<10}")

        if len(gpt5_misclassified) > 20:
            print(f"\n... and {len(gpt5_misclassified) - 20} more species")

    # Compare with GPT-4.1-nano
    print("\n" + "=" * 80)
    print("COMPARISON: GPT-5 vs GPT-4.1-NANO")
    print("=" * 80)

    gpt5_miss = set(model_misclassifications['openai/gpt-5']['tail_to_bacillus'])
    nano_miss = set(model_misclassifications['openai/gpt-4.1-nano']['tail_to_bacillus'])

    both_miss = gpt5_miss & nano_miss
    only_gpt5 = gpt5_miss - nano_miss
    only_nano = nano_miss - gpt5_miss

    print(f"\nBoth models misclassify: {len(both_miss)} species")
    print(f"Only GPT-5 misclassifies: {len(only_gpt5)} species")
    print(f"Only GPT-4.1-nano misclassifies: {len(only_nano)} species")

    if only_gpt5:
        print("\n" + "-" * 80)
        print("Species that GPT-5 misclassifies but GPT-4.1-nano gets correct (top 10):")
        print("-" * 80)

        for species in sorted(only_gpt5)[:10]:
            if species in gt_map:
                print(f"  - {gt_map[species]['binomial_name']}")

    # Check if there's a pattern in the misclassified species
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    # Analyze characteristics of commonly misclassified species
    commonly_misclassified = []
    for species in tail_species:
        # Count how many models misclassify this species
        misclassify_count = 0
        for model in models_to_analyze:
            if species in model_misclassifications[model]['tail_to_bacillus']:
                misclassify_count += 1

        if misclassify_count >= 4:  # Misclassified by at least 4 models
            commonly_misclassified.append((species, misclassify_count))

    commonly_misclassified.sort(key=lambda x: x[1], reverse=True)

    if commonly_misclassified:
        print(f"\nSpecies commonly misclassified as bacillus (by 4+ models):")
        print("-" * 80)
        print(f"{'Species':<40} {'# Models Missing':<20} {'Other Properties':<30}")
        print("-" * 80)

        for species, count in commonly_misclassified[:15]:
            if species in gt_map:
                info = gt_map[species]
                gram = normalize_value(info.get('gram_staining'))
                other_props = []
                if gram:
                    other_props.append(f"Gram: {gram}")
                if normalize_value(info.get('motility')) == 'true':
                    other_props.append("Motile")
                props_str = ", ".join(other_props) if other_props else "No additional info"

                print(f"{info['binomial_name']:<40} {count:<20} {props_str:<30}")

    # Find species that ALL models get wrong
    all_wrong = None
    for species in tail_species:
        wrong_in_all = True
        for model in models_to_analyze:
            if species not in model_misclassifications[model]['tail_to_bacillus']:
                wrong_in_all = False
                break

        if wrong_in_all:
            if all_wrong is None:
                all_wrong = []
            all_wrong.append(species)

    if all_wrong:
        print(f"\n" + "=" * 80)
        print(f"UNIVERSALLY MISCLASSIFIED: {len(all_wrong)} species misclassified by ALL models")
        print("=" * 80)

        for species in all_wrong[:10]:
            if species in gt_map:
                print(f"  - {gt_map[species]['binomial_name']}")

        if len(all_wrong) > 10:
            print(f"  ... and {len(all_wrong) - 10} more")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()