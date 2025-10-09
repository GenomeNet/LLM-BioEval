#!/usr/bin/env python3
"""
Generate a table showing the best model for each phenotype based on BALANCED accuracy.
This matches the methodology used in the heatmap generation.
"""

import urllib.request
import json
from typing import Dict, List, Optional
from collections import defaultdict

def fetch_api_data(endpoint: str, api_url: str = 'http://localhost:5050') -> Dict:
    """Fetch data from API endpoint."""
    full_url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urllib.request.urlopen(full_url, timeout=60) as resp:
        return json.loads(resp.read().decode('utf-8'))


def normalize_value(value) -> Optional[str]:
    """Normalize values with deterministic handling - matching heatmap script."""
    missing_tokens = ['n/a', 'na', 'null', 'none', 'nan', 'undefined', '-', 'unknown', 'missing']

    if value is None or value == '':
        return None

    # Convert to string and normalize
    str_value = str(value).strip().lower()

    # Check if it's a missing token
    if str_value in missing_tokens:
        return None

    # For multi-value fields, parse and sort
    if ',' in str_value or ';' in str_value:
        parts = [s.strip() for s in str_value.replace(';', ',').split(',') if s.strip()]
        return ','.join(sorted(parts))

    return str_value


def to_boolean(v) -> Optional[bool]:
    """Convert to boolean with consistent handling - matching heatmap script."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ['true', '1', 'yes', 't', 'y']:
        return True
    if s in ['false', '0', 'no', 'f', 'n']:
        return False
    return None


def compute_balanced_accuracy(preds: List, truths: List) -> float:
    """Compute balanced accuracy - matching the heatmap methodology."""
    # Try binary first
    mapped = [(to_boolean(p), to_boolean(t)) for p, t in zip(preds, truths)]
    mapped = [(p, t) for p, t in mapped if p is not None and t is not None]

    if len(mapped) > 0 and len(mapped) == len(preds):
        # Binary classification
        tp = tn = fp = fn = 0
        for p, t in mapped:
            if t and p:
                tp += 1
            elif not t and not p:
                tn += 1
            elif not t and p:
                fp += 1
            else:
                fn += 1

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        return (sens + spec) / 2

    # Multiclass - calculate average recall
    labels = sorted(list(set([str(v) for v in truths + preds])))
    conf = {r: {c: 0 for c in labels} for r in labels}

    for t, p in zip(truths, preds):
        conf[str(t)][str(p)] += 1

    recall_sum = 0
    for lab in labels:
        tp = conf[lab][lab]
        fn = sum(conf[lab][l2] for l2 in labels if l2 != lab)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_sum += rec

    return recall_sum / len(labels)


def calculate_accuracies():
    """Calculate balanced accuracy for each model-phenotype combination."""

    print("Fetching data from API (using balanced accuracy methodology)...")

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

    # Phenotypes to analyze (excluding the ones filtered in heatmap)
    excluded_phenotypes = ['aerophilicity', 'health_association', 'hemolysis']

    all_phenotypes = [
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

    # Filter out excluded phenotypes
    phenotypes = [p for p in all_phenotypes if p not in excluded_phenotypes]

    # Get unique models
    models = sorted(list(set(p['model'] for p in predictions)))

    # Calculate balanced accuracy for each model-phenotype combination
    best_models = {}

    for phenotype in phenotypes:
        best_accuracy = 0
        best_model = None
        best_samples = 0

        for model in models:
            model_preds = [p for p in predictions if p['model'] == model]

            true_vals = []
            pred_vals = []

            for pred in model_preds:
                species = pred.get('binomial_name', '').lower()
                if species and species in gt_map:
                    t = normalize_value(gt_map[species].get(phenotype))
                    y = normalize_value(pred.get(phenotype))

                    # Only include if both values are non-null
                    if t is not None and y is not None:
                        true_vals.append(t)
                        pred_vals.append(y)

            if len(true_vals) > 0:
                balanced_acc = compute_balanced_accuracy(pred_vals, true_vals)

                if balanced_acc > best_accuracy:
                    best_accuracy = balanced_acc
                    best_model = model
                    best_samples = len(true_vals)

        if best_model:
            # Shorten model name like in heatmap
            model_short = best_model.split('/')[-1] if '/' in best_model else best_model
            best_models[phenotype] = {
                'model': best_model,
                'model_short': model_short,
                'balanced_accuracy': best_accuracy,
                'samples': best_samples
            }

    return best_models


def generate_markdown_table(best_models):
    """Generate markdown table of results."""

    # Sort by balanced accuracy (descending)
    sorted_phenotypes = sorted(best_models.keys(),
                              key=lambda x: best_models[x]['balanced_accuracy'],
                              reverse=True)

    # Create markdown table
    md_lines = ["# Best Model-Phenotype Combinations (Balanced Accuracy)\n"]
    md_lines.append("Generated from MicrobeLLM API data using balanced accuracy methodology\n")
    md_lines.append("*Note: Excludes aerophilicity, health_association, and hemolysis phenotypes*\n")
    md_lines.append("\n| Phenotype | Best Model | Model (Short) | Balanced Accuracy | Sample Size |")
    md_lines.append("|-----------|------------|---------------|-------------------|-------------|")

    for phenotype in sorted_phenotypes:
        info = best_models[phenotype]
        phenotype_display = phenotype.replace('_', ' ').title()
        model_display = info['model']
        model_short = info['model_short']
        accuracy_pct = f"{info['balanced_accuracy']*100:.1f}%"
        samples = f"{info['samples']:,}"

        md_lines.append(f"| {phenotype_display} | {model_display} | {model_short} | {accuracy_pct} | {samples} |")

    # Add summary statistics
    md_lines.append("\n## Summary Statistics\n")
    md_lines.append(f"- Total phenotypes analyzed: {len(best_models)}")

    if best_models:
        avg_accuracy = sum(v['balanced_accuracy'] for v in best_models.values()) / len(best_models)
        md_lines.append(f"- Average best balanced accuracy: {avg_accuracy*100:.1f}%")

        # Count models
        model_counts = defaultdict(int)
        for info in best_models.values():
            model_counts[info['model_short']] += 1

        md_lines.append("\n### Best Models Distribution (by short name):\n")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            md_lines.append(f"- {model}: {count} phenotypes")

    return '\n'.join(md_lines)


def main():
    print("="*60)
    print("ANALYZING BEST MODEL-PHENOTYPE COMBINATIONS (BALANCED ACCURACY)")
    print("="*60)

    # Calculate accuracies
    best_models = calculate_accuracies()

    if not best_models:
        print("No valid model-phenotype combinations found!")
        return

    # Print results
    print("\n" + "="*60)
    print("RESULTS (Balanced Accuracy)")
    print("="*60)

    # Sort by balanced accuracy for display
    sorted_phenotypes = sorted(best_models.keys(),
                              key=lambda x: best_models[x]['balanced_accuracy'],
                              reverse=True)

    print(f"\n{'Phenotype':<30} {'Best Model (short)':<25} {'Balanced Acc':<15} {'Samples':<10}")
    print("-"*80)

    for phenotype in sorted_phenotypes:
        info = best_models[phenotype]
        phenotype_display = phenotype.replace('_', ' ').title()
        print(f"{phenotype_display:<30} {info['model_short']:<25} "
              f"{info['balanced_accuracy']*100:>6.1f}% {info['samples']:>12,}")

    # Generate markdown table
    md_content = generate_markdown_table(best_models)

    # Save to file
    output_file = 'best_model_phenotype_combinations_balanced.md'
    with open(output_file, 'w') as f:
        f.write(md_content)

    print(f"\nMarkdown table saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Phenotypes analyzed: {len(best_models)}")

    if best_models:
        avg_accuracy = sum(v['balanced_accuracy'] for v in best_models.values()) / len(best_models)
        print(f"Average best balanced accuracy: {avg_accuracy*100:.1f}%")

        # Show model distribution
        model_counts = defaultdict(int)
        for info in best_models.values():
            model_counts[info['model_short']] += 1

        print("\nModels appearing as best (short names):")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} phenotype(s)")

    # Highlight Cell Shape specifically
    print("\n" + "="*60)
    print("CELL SHAPE SPECIFIC RESULT:")
    print("="*60)
    if 'cell_shape' in best_models:
        cs_info = best_models['cell_shape']
        print(f"Best model for Cell Shape: {cs_info['model_short']}")
        print(f"Balanced Accuracy: {cs_info['balanced_accuracy']*100:.2f}%")
        print(f"Sample Size: {cs_info['samples']:,}")
        print("\nThis should match the heatmap visualization!")


if __name__ == '__main__':
    main()