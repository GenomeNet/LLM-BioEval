#!/usr/bin/env python3
"""
Analyze Cell Shape phenotype performance across models, with focus on GPT-4.1-nano.
"""

import urllib.request
import json
import numpy as np
from typing import List, Dict, Tuple


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


def calculate_balanced_accuracy(true_vals: List[str], pred_vals: List[str]) -> float:
    """Calculate balanced accuracy for multiclass classification."""
    labels = sorted(list(set(true_vals + pred_vals)))

    # Calculate confusion matrix
    conf = {r: {c: 0 for c in labels} for r in labels}
    for t, p in zip(true_vals, pred_vals):
        conf[t][p] += 1

    # Calculate macro-average recall (balanced accuracy)
    recall_sum = 0
    for label in labels:
        tp = conf[label][label]
        fn = sum(conf[label][l2] for l2 in labels if l2 != label)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_sum += recall

    return recall_sum / len(labels)


def main():
    print("=" * 70)
    print("CELL SHAPE PHENOTYPE ANALYSIS")
    print("=" * 70)

    # Fetch prediction data
    print("\nFetching prediction data...")
    pred_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt')
    predictions = pred_data.get('data', [])

    # Fetch ground truth data
    print("Fetching ground truth data...")
    gt_data = fetch_api_data('/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000')

    # Create ground truth mapping
    gt_map = {}
    for item in gt_data.get('data', []):
        gt_map[item['binomial_name'].lower()] = item

    # Get unique cell shape values from ground truth
    cell_shape_values = set()
    for item in gt_data.get('data', []):
        val = normalize_value(item.get('cell_shape'))
        if val:
            cell_shape_values.add(val)

    print(f"\nCell shape categories in ground truth: {sorted(cell_shape_values)}")

    # Calculate balanced accuracy for each model
    print("\n" + "=" * 70)
    print("BALANCED ACCURACY ANALYSIS FOR CELL SHAPE")
    print("=" * 70)

    model_results = []

    for model in sorted(set(p['model'] for p in predictions)):
        model_preds = [p for p in predictions if p['model'] == model]

        true_vals = []
        pred_vals = []

        # Collect cell shape predictions
        for pred in model_preds:
            species = pred.get('binomial_name', '').lower()
            if species in gt_map:
                true_val = normalize_value(gt_map[species].get('cell_shape'))
                pred_val = normalize_value(pred.get('cell_shape'))

                if true_val and pred_val:
                    true_vals.append(true_val)
                    pred_vals.append(pred_val)

        if len(true_vals) > 0:
            balanced_acc = calculate_balanced_accuracy(true_vals, pred_vals)

            # Calculate simple accuracy for comparison
            simple_acc = sum(1 for t, p in zip(true_vals, pred_vals) if t == p) / len(true_vals)

            # Calculate per-class accuracy
            class_accuracies = {}
            for shape in cell_shape_values:
                shape_true = [t for t, p in zip(true_vals, pred_vals) if t == shape]
                shape_correct = [1 for t, p in zip(true_vals, pred_vals) if t == shape and t == p]
                if shape_true:
                    class_accuracies[shape] = len(shape_correct) / len(shape_true)

            model_results.append({
                'model': model,
                'balanced_accuracy': balanced_acc,
                'simple_accuracy': simple_acc,
                'sample_size': len(true_vals),
                'class_accuracies': class_accuracies
            })

    # Sort by balanced accuracy
    model_results.sort(key=lambda x: x['balanced_accuracy'], reverse=True)

    # Display top performers
    print("\nTOP 10 MODELS FOR CELL SHAPE PREDICTION:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Model':<35} {'Balanced Acc':<15} {'Simple Acc':<12} {'N':<8}")
    print("-" * 70)

    for i, result in enumerate(model_results[:10], 1):
        model_short = result['model'].split('/')[-1]
        print(f"{i:<6} {model_short:<35} {result['balanced_accuracy']:.3f}          "
              f"{result['simple_accuracy']:.3f}        {result['sample_size']:<8}")

    # Find GPT-4.1-nano specifically
    print("\n" + "=" * 70)
    print("GPT-4.1-NANO DETAILED ANALYSIS")
    print("=" * 70)

    gpt_nano_result = None
    gpt_nano_rank = None

    for i, result in enumerate(model_results, 1):
        if 'gpt-4.1-nano' in result['model'].lower():
            gpt_nano_result = result
            gpt_nano_rank = i
            break

    if gpt_nano_result:
        print(f"\nRank: #{gpt_nano_rank} out of {len(model_results)} models")
        print(f"Balanced Accuracy: {gpt_nano_result['balanced_accuracy']:.3f}")
        print(f"Simple Accuracy: {gpt_nano_result['simple_accuracy']:.3f}")
        print(f"Sample Size: {gpt_nano_result['sample_size']}")

        print("\nPer-class performance:")
        for shape, acc in sorted(gpt_nano_result['class_accuracies'].items()):
            print(f"  {shape:20s}: {acc:.3f}")
    else:
        print("\nGPT-4.1-nano not found in results!")

    # Calculate statistics across all models
    all_balanced = [r['balanced_accuracy'] for r in model_results]
    mean_balanced = np.mean(all_balanced)
    std_balanced = np.std(all_balanced)

    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"\nMean balanced accuracy across all models: {mean_balanced:.3f} (±{std_balanced:.3f})")

    if gpt_nano_result:
        z_score = (gpt_nano_result['balanced_accuracy'] - mean_balanced) / std_balanced
        print(f"GPT-4.1-nano z-score: {z_score:.2f}")

        if gpt_nano_result['balanced_accuracy'] > mean_balanced:
            print(f"GPT-4.1-nano performs {(gpt_nano_result['balanced_accuracy'] - mean_balanced):.3f} above average")
        else:
            print(f"GPT-4.1-nano performs {(mean_balanced - gpt_nano_result['balanced_accuracy']):.3f} below average")

    # Analyze all phenotypes for GPT-4.1-nano
    print("\n" + "=" * 70)
    print("GPT-4.1-NANO PERFORMANCE ACROSS ALL PHENOTYPES")
    print("=" * 70)

    if gpt_nano_result:
        # Get all phenotype performance for GPT-4.1-nano
        phenotypes = ['animal_pathogenicity', 'biofilm_formation', 'biosafety_level',
                     'cell_shape', 'extreme_environment_tolerance', 'gram_staining',
                     'host_association', 'motility', 'plant_pathogenicity', 'spore_formation']

        print(f"\n{'Phenotype':<30} {'Balanced Accuracy':<20}")
        print("-" * 50)

        for phenotype in phenotypes:
            # Calculate balanced accuracy for this phenotype
            model_preds = [p for p in predictions if 'gpt-4.1-nano' in p['model'].lower()]

            true_vals = []
            pred_vals = []

            for pred in model_preds:
                species = pred.get('binomial_name', '').lower()
                if species in gt_map:
                    true_val = normalize_value(gt_map[species].get(phenotype))
                    pred_val = normalize_value(pred.get(phenotype))

                    if true_val and pred_val:
                        true_vals.append(true_val)
                        pred_vals.append(pred_val)

            if len(true_vals) > 0:
                balanced_acc = calculate_balanced_accuracy(true_vals, pred_vals)
                marker = " ***" if phenotype == 'cell_shape' else ""
                print(f"{phenotype.replace('_', ' ').title():<30} {balanced_acc:.3f}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()