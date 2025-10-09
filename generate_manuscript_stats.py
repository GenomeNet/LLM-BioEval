#!/usr/bin/env python3
"""
Generate aggregate statistics for the manuscript.
Reports on LLMs tested, phenotypes analyzed, and performance metrics.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional
import numpy as np
import argparse


def fetch_api_data(endpoint: str, api_url: str = 'http://localhost:5050') -> Optional[Dict]:
    """Fetch data from API endpoint."""
    full_url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"

    try:
        with urllib.request.urlopen(full_url, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
        return None


def get_phenotype_statistics(api_url: str) -> Dict:
    """Get statistics about phenotype predictions."""

    # Fetch ground truth datasets
    datasets_data = fetch_api_data('/api/ground_truth/datasets', api_url)
    datasets = datasets_data.get('datasets', []) if datasets_data else []

    # Use WA_Test_Dataset as the main dataset
    main_dataset = None
    for ds in datasets:
        if ds['dataset_name'] == 'WA_Test_Dataset':
            main_dataset = ds
            break

    if not main_dataset:
        print("Warning: WA_Test_Dataset not found")
        return {}

    # Fetch prediction data
    predictions_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt', api_url)

    if not predictions_data or predictions_data.get('error'):
        print("Error fetching prediction data")
        return {}

    predictions = predictions_data.get('data', [])

    # Get template definitions
    template_data = fetch_api_data(f'/api/template_field_definitions?template={main_dataset["template_name"]}', api_url)
    phenotype_fields = list(template_data.get('field_definitions', {}).keys()) if template_data else []

    # Remove fields that are typically excluded from analysis
    excluded_fields = ['aerophilicity', 'health_association', 'hemolysis']
    analyzed_phenotypes = [p for p in phenotype_fields if p not in excluded_fields]

    # Get unique models
    models = list(set(p['model'] for p in predictions))

    # Get unique species
    species = list(set(p['binomial_name'] for p in predictions))

    # Calculate performance metrics
    ground_truth_data = fetch_api_data(f'/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000', api_url)

    if not ground_truth_data or not ground_truth_data.get('success'):
        print("Error fetching ground truth data")
        return {}

    # Create ground truth map
    gt_map = {}
    for item in ground_truth_data.get('data', []):
        gt_map[item['binomial_name'].lower()] = item

    # Calculate metrics for each model and phenotype
    model_metrics = {}

    for model in models:
        model_preds = [p for p in predictions if p['model'] == model]
        balanced_accuracies = []
        phenotype_accuracies = {}

        for phenotype in analyzed_phenotypes:
            true_vals = []
            pred_vals = []

            for pred in model_preds:
                species_name = pred.get('binomial_name', '').lower()
                if species_name in gt_map:
                    true_val = normalize_value(gt_map[species_name].get(phenotype))
                    pred_val = normalize_value(pred.get(phenotype))

                    if true_val is not None and pred_val is not None:
                        true_vals.append(true_val)
                        pred_vals.append(pred_val)

            if len(true_vals) > 0:
                # Calculate balanced accuracy properly
                balanced_acc = calculate_balanced_accuracy(true_vals, pred_vals)
                phenotype_accuracies[phenotype] = {
                    'balanced_accuracy': balanced_acc,
                    'sample_size': len(true_vals)
                }
                if balanced_acc is not None:
                    balanced_accuracies.append(balanced_acc)

        if balanced_accuracies:
            model_metrics[model] = {
                'mean_balanced_accuracy': np.mean(balanced_accuracies),
                'phenotype_accuracies': phenotype_accuracies,
                'predictions_count': len(model_preds)
            }

    # Find best model
    best_model = None
    best_accuracy = 0

    for model, metrics in model_metrics.items():
        if metrics['mean_balanced_accuracy'] > best_accuracy:
            best_accuracy = metrics['mean_balanced_accuracy']
            best_model = model

    return {
        'total_models': len(models),
        'total_phenotypes': len(analyzed_phenotypes),
        'total_predictions': len(predictions),
        'total_species': len(species),
        'models': models,
        'phenotypes': analyzed_phenotypes,
        'best_model': best_model,
        'best_accuracy': best_accuracy,
        'model_metrics': model_metrics,
        'dataset': main_dataset['dataset_name'],
        'species_count': main_dataset['species_count']
    }


def get_knowledge_calibration_statistics(api_url: str) -> Dict:
    """Get statistics about knowledge calibration."""

    # Fetch knowledge analysis data
    knowledge_data = fetch_api_data('/api/knowledge_analysis_data', api_url)

    if not knowledge_data:
        print("Error fetching knowledge analysis data")
        return {}

    # Count models tested on knowledge templates
    models_tested = set()
    total_assessments = 0
    template_counts = {}

    for file_name, file_data in knowledge_data.get('knowledge_analysis', {}).items():
        if file_data.get('has_type_column'):
            for input_type, type_data in file_data.get('types', {}).items():
                for template_name, template_data in type_data.items():
                    if 'knowlege' in template_name:  # Note: typo in original data
                        template_counts[template_name] = template_counts.get(template_name, 0)

                        for model_name, model_data in template_data.items():
                            models_tested.add(model_name)
                            # Count all responses
                            for key in ['NA', 'limited', 'moderate', 'extensive', 'no_result', 'inference_failed']:
                                total_assessments += model_data.get(key, 0)
                            template_counts[template_name] += model_data.get('total', 0)

    return {
        'total_models_tested': len(models_tested),
        'total_assessments': total_assessments,
        'templates_used': list(template_counts.keys()),
        'template_counts': template_counts,
        'models_tested': list(models_tested)
    }


def normalize_value(value) -> Optional[str]:
    """Normalize values for comparison."""
    missing_tokens = ['n/a', 'na', 'null', 'none', 'nan', 'undefined', '-', 'unknown', 'missing']

    if value is None or value == '':
        return None

    str_value = str(value).strip().lower()

    if str_value in missing_tokens:
        return None

    return str_value


def to_boolean(v) -> Optional[bool]:
    """Convert value to boolean."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ['true', '1', 'yes', 't', 'y']:
        return True
    if s in ['false', '0', 'no', 'f', 'n']:
        return False
    return None


def calculate_balanced_accuracy(true_vals: List[str], pred_vals: List[str]) -> float:
    """Calculate balanced accuracy for predictions."""
    # Try binary classification first
    mapped = []
    for t, p in zip(true_vals, pred_vals):
        t_bool = to_boolean(t)
        p_bool = to_boolean(p)
        if t_bool is not None and p_bool is not None:
            mapped.append((t_bool, p_bool))

    if len(mapped) == len(true_vals):
        # Binary classification
        tp = tn = fp = fn = 0
        for t, p in mapped:
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

    # Multiclass classification
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


def format_model_name(model_name: str) -> str:
    """Format model name for display."""
    if '/' in model_name:
        return model_name.split('/')[-1]
    return model_name


def main():
    """Generate manuscript statistics."""
    parser = argparse.ArgumentParser(
        description='Generate aggregate statistics for manuscript'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='Base URL for the API')
    parser.add_argument('--output', default='manuscript_stats.txt',
                       help='Output file for statistics')

    args = parser.parse_args()

    print("Generating manuscript statistics...")
    print("=" * 60)

    # Get phenotype analysis statistics
    print("\nFetching phenotype analysis data...")
    phenotype_stats = get_phenotype_statistics(args.api_url)

    # Get knowledge calibration statistics
    print("Fetching knowledge calibration data...")
    knowledge_stats = get_knowledge_calibration_statistics(args.api_url)

    # Format output
    output = []
    output.append("MANUSCRIPT STATISTICS")
    output.append("=" * 60)

    # Phenotype Analysis Section
    output.append("\nPHENOTYPE ANALYSIS")
    output.append("-" * 40)

    if phenotype_stats:
        output.append(f"Dataset: {phenotype_stats['dataset']}")
        output.append(f"Species in dataset: {phenotype_stats['species_count']}")
        output.append(f"Total LLMs tested: {phenotype_stats['total_models']}")
        output.append(f"Phenotypes analyzed: {phenotype_stats['total_phenotypes']}")
        output.append(f"Total predictions generated: {phenotype_stats['total_predictions']:,}")
        output.append(f"Unique species predicted: {phenotype_stats['total_species']}")

        output.append("\nPhenotypes analyzed:")
        for phenotype in sorted(phenotype_stats['phenotypes']):
            formatted_name = phenotype.replace('_', ' ').title()
            output.append(f"  - {formatted_name}")

        if phenotype_stats['best_model']:
            output.append(f"\nBest performing model:")
            output.append(f"  Model: {format_model_name(phenotype_stats['best_model'])}")
            output.append(f"  Mean balanced accuracy: {phenotype_stats['best_accuracy']:.3f}")

        # Top 5 models
        output.append("\nTop 5 models by balanced accuracy:")
        sorted_models = sorted(
            phenotype_stats['model_metrics'].items(),
            key=lambda x: x[1]['mean_balanced_accuracy'],
            reverse=True
        )[:5]

        for i, (model, metrics) in enumerate(sorted_models, 1):
            output.append(f"  {i}. {format_model_name(model)}: {metrics['mean_balanced_accuracy']:.3f}")

    # Knowledge Calibration Section
    output.append("\n\nKNOWLEDGE CALIBRATION")
    output.append("-" * 40)

    if knowledge_stats:
        output.append(f"Total LLMs tested: {knowledge_stats['total_models_tested']}")
        output.append(f"Total knowledge assessments: {knowledge_stats['total_assessments']:,}")
        output.append(f"Templates used: {len(knowledge_stats['templates_used'])}")

        output.append("\nTemplates and assessment counts:")
        for template, count in knowledge_stats['template_counts'].items():
            output.append(f"  - {template}: {count:,} assessments")

    # Summary for manuscript
    output.append("\n\nMANUSCRIPT TEXT SUGGESTIONS")
    output.append("-" * 40)

    if phenotype_stats and knowledge_stats:
        # Combine unique models from both tasks
        all_models = set(phenotype_stats['models']) | set(knowledge_stats['models_tested'])

        output.append(f"\nWe evaluated {len(all_models)} different large language models across two major tasks:")
        output.append(f"")
        output.append(f"1. Phenotype Prediction: {phenotype_stats['total_models']} LLMs were tested on "
                     f"{phenotype_stats['total_phenotypes']} bacterial phenotypes, generating "
                     f"{phenotype_stats['total_predictions']:,} individual predictions across "
                     f"{phenotype_stats['total_species']} unique bacterial species.")

        if phenotype_stats['best_model']:
            output.append(f"   The best performing model was {format_model_name(phenotype_stats['best_model'])} "
                         f"with a mean balanced accuracy of {phenotype_stats['best_accuracy']:.1%}.")

        output.append(f"")
        output.append(f"2. Knowledge Calibration: {knowledge_stats['total_models_tested']} LLMs were evaluated "
                     f"on their ability to assess their own knowledge levels, "
                     f"generating {knowledge_stats['total_assessments']:,} knowledge assessments.")

    # Write to file
    output_text = '\n'.join(output)

    with open(args.output, 'w') as f:
        f.write(output_text)

    # Also print to console
    print("\n" + output_text)

    print(f"\n\nStatistics saved to: {args.output}")


if __name__ == '__main__':
    main()