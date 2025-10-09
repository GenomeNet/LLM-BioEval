#!/usr/bin/env python3
"""
Generate a table showing the best model for each phenotype based on accuracy.
"""

import urllib.request
import json
from typing import Dict, List
from collections import defaultdict

def fetch_api_data(endpoint: str, api_url: str = 'http://localhost:5050') -> Dict:
    """Fetch data from API endpoint."""
    full_url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urllib.request.urlopen(full_url, timeout=60) as resp:
        return json.loads(resp.read().decode('utf-8'))


def normalize_value(value, phenotype) -> str:
    """Normalize phenotype values for comparison."""
    if value is None or value == '' or str(value).lower() in ['n/a', 'na', 'null', 'none', 'nan']:
        return None

    val_lower = str(value).strip().lower()

    # Binary phenotypes
    binary_phenotypes = ['motility', 'spore_formation', 'biofilm_formation',
                        'animal_pathogenicity', 'plant_pathogenicity', 'host_association']

    if phenotype in binary_phenotypes:
        if val_lower in ['yes', 'true', '1', 'positive', 'motile', 'forms spores',
                         'spore-forming', 'pathogenic', 'pathogen', 'forms biofilm',
                         'biofilm-forming', 'associated']:
            return 'yes'
        elif val_lower in ['no', 'false', '0', 'negative', 'non-motile', 'immotile',
                          'non-spore-forming', 'does not form spores', 'non-pathogenic',
                          'not pathogenic', 'does not form biofilm', 'not associated']:
            return 'no'

    # Return normalized value for categorical phenotypes
    return val_lower


def calculate_accuracies():
    """Calculate accuracy for each model-phenotype combination."""

    print("Fetching data from API...")

    # Fetch prediction data
    pred_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt')
    predictions = pred_data.get('data', [])

    # Fetch ground truth data
    gt_data = fetch_api_data('/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000')

    # Create ground truth mapping
    gt_map = {}
    for item in gt_data.get('data', []):
        gt_map[item['binomial_name'].lower()] = item

    print(f"Found {len(predictions)} predictions")
    print(f"Found {len(gt_map)} ground truth entries")

    # Phenotypes to analyze
    phenotypes = [
        'gram_staining',
        'motility',
        'aerophilicity',
        'extreme_environment_tolerance',
        'biofilm_formation',
        'animal_pathogenicity',
        'biosafety_level',
        'health_association',
        'host_association',
        'plant_pathogenicity',
        'spore_formation',
        'hemolysis',
        'cell_shape'
    ]

    # Calculate accuracy for each model-phenotype combination
    model_phenotype_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    for pred in predictions:
        species = pred.get('binomial_name', '').lower()
        model = pred.get('model', '')

        if species in gt_map:
            for phenotype in phenotypes:
                true_val = normalize_value(gt_map[species].get(phenotype), phenotype)
                pred_val = normalize_value(pred.get(phenotype), phenotype)

                if true_val and pred_val:
                    model_phenotype_stats[model][phenotype]['total'] += 1
                    if true_val == pred_val:
                        model_phenotype_stats[model][phenotype]['correct'] += 1

    # Find best model for each phenotype
    best_models = {}
    for phenotype in phenotypes:
        best_accuracy = 0
        best_model = None
        best_samples = 0

        for model in model_phenotype_stats:
            stats = model_phenotype_stats[model][phenotype]
            if stats['total'] >= 50:  # Minimum sample size
                accuracy = stats['correct'] / stats['total']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_samples = stats['total']

        if best_model:
            best_models[phenotype] = {
                'model': best_model,
                'accuracy': best_accuracy,
                'samples': best_samples
            }

    return best_models


def generate_markdown_table(best_models):
    """Generate markdown table of results."""

    # Sort by accuracy (descending)
    sorted_phenotypes = sorted(best_models.keys(),
                              key=lambda x: best_models[x]['accuracy'],
                              reverse=True)

    # Create markdown table
    md_lines = ["# Best Model-Phenotype Combinations\n"]
    md_lines.append("Generated from MicrobeLLM API data\n")
    md_lines.append("\n| Phenotype | Best Model | Accuracy | Sample Size |")
    md_lines.append("|-----------|------------|----------|-------------|")

    for phenotype in sorted_phenotypes:
        info = best_models[phenotype]
        phenotype_display = phenotype.replace('_', ' ').title()
        model_display = info['model']
        accuracy_pct = f"{info['accuracy']*100:.1f}%"
        samples = f"{info['samples']:,}"

        md_lines.append(f"| {phenotype_display} | {model_display} | {accuracy_pct} | {samples} |")

    # Add summary statistics
    md_lines.append("\n## Summary Statistics\n")
    md_lines.append(f"- Total phenotypes analyzed: {len(best_models)}")

    if best_models:
        avg_accuracy = sum(v['accuracy'] for v in best_models.values()) / len(best_models)
        md_lines.append(f"- Average best accuracy: {avg_accuracy*100:.1f}%")

        # Count models
        model_counts = defaultdict(int)
        for info in best_models.values():
            model_counts[info['model']] += 1

        md_lines.append("\n### Best Models Distribution:\n")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"- {model}: {count} phenotypes")

    return '\n'.join(md_lines)


def main():
    print("="*60)
    print("ANALYZING BEST MODEL-PHENOTYPE COMBINATIONS")
    print("="*60)

    # Calculate accuracies
    best_models = calculate_accuracies()

    if not best_models:
        print("No valid model-phenotype combinations found!")
        return

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Sort by accuracy for display
    sorted_phenotypes = sorted(best_models.keys(),
                              key=lambda x: best_models[x]['accuracy'],
                              reverse=True)

    print(f"\n{'Phenotype':<30} {'Best Model':<30} {'Accuracy':<10} {'Samples':<10}")
    print("-"*80)

    for phenotype in sorted_phenotypes:
        info = best_models[phenotype]
        phenotype_display = phenotype.replace('_', ' ').title()
        print(f"{phenotype_display:<30} {info['model']:<30} "
              f"{info['accuracy']*100:>6.1f}% {info['samples']:>8,}")

    # Generate markdown table
    md_content = generate_markdown_table(best_models)

    # Save to file
    output_file = 'best_model_phenotype_combinations.md'
    with open(output_file, 'w') as f:
        f.write(md_content)

    print(f"\nMarkdown table saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Phenotypes analyzed: {len(best_models)}")

    if best_models:
        avg_accuracy = sum(v['accuracy'] for v in best_models.values()) / len(best_models)
        print(f"Average best accuracy: {avg_accuracy*100:.1f}%")

        # Show model distribution
        model_counts = defaultdict(int)
        for info in best_models.values():
            model_counts[info['model']] += 1

        print("\nModels appearing as best:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} phenotype(s)")


if __name__ == '__main__':
    main()