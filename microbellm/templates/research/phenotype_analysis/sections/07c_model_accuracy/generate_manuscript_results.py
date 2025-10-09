#!/usr/bin/env python3
"""
Generate manuscript results text with detailed statistics for phenotype prediction accuracy.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
import numpy as np
import argparse
from collections import defaultdict
import csv
import os


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


def normalize_value(value) -> Optional[str]:
    """Normalize values for comparison."""
    missing_tokens = ['n/a', 'na', 'null', 'none', 'nan', 'undefined', '-', 'unknown', 'missing']

    if value is None or value == '':
        return None

    str_value = str(value).strip().lower()

    if str_value in missing_tokens:
        return None

    # For multi-value fields, parse and sort
    if ',' in str_value or ';' in str_value:
        parts = [s.strip() for s in str_value.replace(';', ',').split(',') if s.strip()]
        return ','.join(sorted(parts))

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


def calculate_balanced_accuracy(true_vals: List[str], pred_vals: List[str]) -> Tuple[float, Dict]:
    """Calculate balanced accuracy and additional metrics for predictions."""
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
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        return (sens + spec) / 2, {
            'sensitivity': sens,
            'specificity': spec,
            'precision': precision,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'type': 'binary'
        }

    # Multiclass classification
    labels = sorted(list(set(true_vals + pred_vals)))

    # Calculate confusion matrix
    conf = {r: {c: 0 for c in labels} for r in labels}
    for t, p in zip(true_vals, pred_vals):
        conf[t][p] += 1

    # Calculate macro-average recall (balanced accuracy) and precision
    recall_sum = 0
    precision_sum = 0

    for label in labels:
        tp = conf[label][label]
        fn = sum(conf[label][l2] for l2 in labels if l2 != label)
        fp = sum(conf[l2][label] for l2 in labels if l2 != label)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        recall_sum += recall
        precision_sum += precision

    return recall_sum / len(labels), {
        'precision': precision_sum / len(labels),
        'n_classes': len(labels),
        'type': 'multiclass'
    }


def get_detailed_phenotype_analysis(api_url: str) -> Dict:
    """Get detailed statistics about phenotype predictions."""

    # Fetch prediction data
    predictions_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt', api_url)

    if not predictions_data or predictions_data.get('error'):
        print("Error fetching prediction data")
        return {}

    predictions = predictions_data.get('data', [])

    # Get ground truth
    ground_truth_data = fetch_api_data('/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000', api_url)

    if not ground_truth_data or not ground_truth_data.get('success'):
        print("Error fetching ground truth data")
        return {}

    # Get template definitions
    template_data = fetch_api_data('/api/template_field_definitions?template=template1_phenotype', api_url)
    phenotype_fields = list(template_data.get('field_definitions', {}).keys()) if template_data else []

    # Remove typically excluded fields
    excluded_fields = ['aerophilicity', 'health_association', 'hemolysis']
    analyzed_phenotypes = [p for p in phenotype_fields if p not in excluded_fields]

    # Create ground truth map
    gt_map = {}
    for item in ground_truth_data.get('data', []):
        gt_map[item['binomial_name'].lower()] = item

    # Get unique models and species
    models = sorted(list(set(p['model'] for p in predictions)))
    species = list(set(p['binomial_name'] for p in predictions))

    # Calculate detailed metrics
    model_metrics = {}
    phenotype_details = defaultdict(lambda: defaultdict(dict))

    for model in models:
        model_preds = [p for p in predictions if p['model'] == model]
        balanced_accuracies = []
        precisions = []

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
                balanced_acc, metrics = calculate_balanced_accuracy(true_vals, pred_vals)

                phenotype_details[phenotype][model] = {
                    'balanced_accuracy': balanced_acc,
                    'precision': metrics.get('precision', 0),
                    'sample_size': len(true_vals),
                    'type': metrics['type']
                }

                balanced_accuracies.append(balanced_acc)
                precisions.append(metrics.get('precision', 0))

        if balanced_accuracies:
            model_metrics[model] = {
                'mean_balanced_accuracy': np.mean(balanced_accuracies),
                'std_balanced_accuracy': np.std(balanced_accuracies),
                'mean_precision': np.mean(precisions),
                'std_precision': np.std(precisions),
                'n_phenotypes': len(balanced_accuracies),
                'total_predictions': len(model_preds)
            }

    # Find best performers for each phenotype
    phenotype_best = {}
    for phenotype in analyzed_phenotypes:
        best_model = None
        best_acc = 0

        for model, metrics in phenotype_details[phenotype].items():
            if metrics['balanced_accuracy'] > best_acc:
                best_acc = metrics['balanced_accuracy']
                best_model = model

        phenotype_best[phenotype] = {
            'model': best_model,
            'accuracy': best_acc,
            'sample_size': phenotype_details[phenotype][best_model]['sample_size'] if best_model else 0
        }

    # Calculate how many times each model is best
    model_best_counts = defaultdict(int)
    for pheno_info in phenotype_best.values():
        if pheno_info['model']:
            model_best_counts[pheno_info['model']] += 1

    return {
        'models': models,
        'species': species,
        'phenotypes': analyzed_phenotypes,
        'model_metrics': model_metrics,
        'phenotype_details': dict(phenotype_details),
        'phenotype_best': phenotype_best,
        'model_best_counts': dict(model_best_counts),
        'total_predictions': len(predictions)
    }


def load_organization_mapping(tsv_path: str = 'data/year_size.tsv') -> Dict[str, str]:
    """Load model to organization mapping from TSV file."""
    mapping = {}

    # Try from current directory first, then from parent directories
    search_paths = [
        tsv_path,
        os.path.join('..', '..', '..', '..', '..', tsv_path),
        '/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/data/year_size.tsv'
    ]

    for path in search_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    model_name = row['Model'].lower()
                    organization = row['Organization']
                    if organization:  # Only add if organization is not empty
                        # Store various possible model name formats
                        mapping[model_name] = organization
                        # Also store without version numbers
                        base_name = model_name.split('-')[0]
                        if base_name not in mapping:
                            mapping[base_name] = organization
            break

    return mapping


def get_organization_performance(model_metrics: Dict) -> Dict[str, Dict]:
    """Group models by organization and calculate performance."""
    org_mapping = load_organization_mapping()

    # Group models by organization
    org_groups = defaultdict(list)
    unmapped_models = []

    for model_name, metrics in model_metrics.items():
        # Try to find organization for this model
        org = None

        # Try exact match first
        if model_name.lower() in org_mapping:
            org = org_mapping[model_name.lower()]
        else:
            # Try partial matches
            for tsv_model, tsv_org in org_mapping.items():
                # More flexible matching
                model_clean = model_name.lower().replace('-', '').replace('_', '').replace('.', '')
                tsv_clean = tsv_model.replace('-', '').replace('_', '').replace('.', '')
                if tsv_clean in model_clean or model_clean in tsv_clean:
                    org = tsv_org
                    break

        if org:
            # Handle multiple organizations (separated by comma)
            orgs = [o.strip() for o in org.split(',')]
            primary_org = orgs[0]  # Use first organization as primary

            # Clean up organization name
            if primary_org == 'OpenAI':
                primary_org = 'OpenAI'
            elif primary_org == 'Anthropic':
                primary_org = 'Anthropic'
            elif primary_org == 'Google' or primary_org == 'Google DeepMind':
                primary_org = 'Google'
            elif primary_org == 'Meta AI' or primary_org == 'Meta':
                primary_org = 'Meta'

            org_groups[primary_org].append(metrics['mean_balanced_accuracy'])
        else:
            unmapped_models.append(model_name)

    # Calculate statistics per organization
    org_performance = {}
    for org, accuracies in org_groups.items():
        if accuracies:
            org_performance[org] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'n': len(accuracies)
            }

    # Log unmapped models for debugging
    if unmapped_models:
        print(f"Warning: Could not find organization for models: {unmapped_models}")

    return org_performance


def format_model_name(model_name: str) -> str:
    """Format model name for manuscript."""
    if '/' in model_name:
        name = model_name.split('/')[-1]
    else:
        name = model_name

    # Clean up common patterns
    name = name.replace('-', ' ').replace('_', ' ')

    # Special formatting for known models
    if 'claude' in name.lower():
        if '3.5' in name and 'sonnet' in name:
            return 'Claude-3.5-Sonnet'
        elif 'sonnet' in name and '4' in name:
            return 'Claude-Sonnet-4'
    elif 'gpt' in name.lower():
        if 'gpt 5' in name or 'gpt5' in name:
            return 'GPT-5'
        elif '4o' in name:
            return 'GPT-4o'
    elif 'gemini' in name.lower():
        if '2.5' in name and 'pro' in name:
            return 'Gemini-2.5-Pro'
        elif 'flash' in name:
            return 'Gemini-Flash-1.5'
    elif 'deepseek' in name.lower():
        if 'r1' in name:
            return 'DeepSeek-R1'

    # Default: capitalize words
    return ' '.join(word.capitalize() for word in name.split())


def generate_results_text(data: Dict) -> str:
    """Generate formatted results text for manuscript."""

    results = []

    # Sort models by mean balanced accuracy
    sorted_models = sorted(
        data['model_metrics'].items(),
        key=lambda x: x[1]['mean_balanced_accuracy'],
        reverse=True
    )

    # Get top performers
    top_model = sorted_models[0] if sorted_models else None
    top_5_models = sorted_models[:5]

    # Introduction paragraph
    results.append("RESULTS\n")
    results.append("=" * 60)
    results.append("\nPhenotype Prediction Performance\n")
    results.append("-" * 40)

    # Overall performance summary
    results.append(f"\nWe evaluated {len(data['models'])} large language models on their ability to predict "
                  f"{len(data['phenotypes'])} bacterial phenotypes across {len(data['species']):,} species from "
                  f"the WA Test Dataset, generating a total of {data['total_predictions']:,} individual predictions.")

    # Best overall model
    if top_model:
        model_name = format_model_name(top_model[0])
        metrics = top_model[1]

        results.append(f"\n{model_name} achieved the highest overall performance with a mean balanced accuracy "
                      f"of {metrics['mean_balanced_accuracy']:.1%} (SD = {metrics['std_balanced_accuracy']:.1%}) "
                      f"and mean precision of {metrics['mean_precision']:.1%} (SD = {metrics['std_precision']:.1%}) "
                      f"across all phenotypes.")

    # Top 5 comparison
    results.append(f"\nThe top five models demonstrated strong performance across phenotypes:")
    for i, (model, metrics) in enumerate(top_5_models, 1):
        model_name = format_model_name(model)
        best_count = data['model_best_counts'].get(model, 0)

        results.append(f"  {i}. {model_name}: {metrics['mean_balanced_accuracy']:.1%} balanced accuracy, "
                      f"best performer for {best_count} phenotype{'s' if best_count != 1 else ''}")

    # Performance range analysis
    all_accuracies = [m['mean_balanced_accuracy'] for m in data['model_metrics'].values()]
    acc_range = max(all_accuracies) - min(all_accuracies)

    results.append(f"\nModel performance varied substantially, with balanced accuracies ranging from "
                  f"{min(all_accuracies):.1%} to {max(all_accuracies):.1%} (range: {acc_range:.1%}), "
                  f"indicating significant differences in model capabilities for bacterial phenotype prediction.")

    # Phenotype-specific analysis
    results.append("\n\nPhenotype-Specific Performance\n")
    results.append("-" * 40)

    # Find easiest and hardest phenotypes
    phenotype_avg_acc = {}
    for phenotype in data['phenotypes']:
        accs = []
        for model_data in data['phenotype_details'][phenotype].values():
            accs.append(model_data['balanced_accuracy'])
        if accs:
            phenotype_avg_acc[phenotype] = np.mean(accs)

    sorted_phenotypes = sorted(phenotype_avg_acc.items(), key=lambda x: x[1], reverse=True)

    if sorted_phenotypes:
        easiest = sorted_phenotypes[0]
        hardest = sorted_phenotypes[-1]

        results.append(f"\nAcross all models, {easiest[0].replace('_', ' ').title()} was the most accurately "
                      f"predicted phenotype (mean balanced accuracy: {easiest[1]:.1%}), while "
                      f"{hardest[0].replace('_', ' ').title()} proved most challenging "
                      f"(mean balanced accuracy: {hardest[1]:.1%}).")

    # Best performers per phenotype
    dominant_models = defaultdict(list)
    for phenotype, best_info in data['phenotype_best'].items():
        if best_info['model']:
            dominant_models[best_info['model']].append(phenotype)

    # Report models that dominate multiple phenotypes
    multi_best = [(model, phenos) for model, phenos in dominant_models.items() if len(phenos) > 1]
    multi_best.sort(key=lambda x: len(x[1]), reverse=True)

    if multi_best:
        results.append(f"\nNotably, {format_model_name(multi_best[0][0])} achieved best-in-class performance for "
                      f"{len(multi_best[0][1])} phenotypes ({', '.join(p.replace('_', ' ').title() for p in multi_best[0][1][:3])}"
                      f"{', and others' if len(multi_best[0][1]) > 3 else ''}), demonstrating broad capability across diverse bacterial traits.")

    # Check for GPT-4.1-nano's cell shape performance
    if 'cell_shape' in data['phenotype_best'] and data['phenotype_best']['cell_shape']['model']:
        if 'gpt-4.1-nano' in data['phenotype_best']['cell_shape']['model'].lower():
            cell_shape_acc = data['phenotype_best']['cell_shape']['accuracy']
            results.append(f"\nInterestingly, GPT-4.1-Nano was the top performer for Cell Shape prediction "
                          f"(balanced accuracy: {cell_shape_acc:.1%}), significantly outperforming larger models on this morphological phenotype.")

    # Sample size information
    total_samples = []
    for phenotype_models in data['phenotype_details'].values():
        for model_data in phenotype_models.values():
            total_samples.append(model_data['sample_size'])

    if total_samples:
        results.append(f"\n\nSample sizes varied across phenotype-model combinations, ranging from "
                      f"{min(total_samples):,} to {max(total_samples):,} predictions per evaluation "
                      f"(mean: {np.mean(total_samples):.0f}, SD: {np.std(total_samples):.0f}).")

    # Statistical comparisons
    results.append("\n\nStatistical Analysis\n")
    results.append("-" * 40)

    # Compare top models
    if len(top_5_models) >= 2:
        top1_acc = top_5_models[0][1]['mean_balanced_accuracy']
        top2_acc = top_5_models[1][1]['mean_balanced_accuracy']
        diff = top1_acc - top2_acc

        results.append(f"\nThe performance difference between the top two models "
                      f"({format_model_name(top_5_models[0][0])} vs {format_model_name(top_5_models[1][0])}) "
                      f"was {diff:.1%} in terms of mean balanced accuracy.")

    # Organization-based comparison using TSV file
    org_performance = get_organization_performance(data['model_metrics'])

    if len(org_performance) > 1:
        # Get top performing organizations with 2+ models
        sorted_orgs = sorted(org_performance.items(), key=lambda x: x[1]['mean'], reverse=True)
        top_orgs = [(org, perf) for org, perf in sorted_orgs if perf['n'] >= 2][:3]

        if top_orgs:
            # Create concise summary sentence
            org_summary = []
            for org, perf in top_orgs:
                org_summary.append(f"{org} ({perf['mean']:.1%}, n={perf['n']})")

            results.append(f"\n\nAcross organizations with multiple models tested, {top_orgs[0][0]} achieved "
                          f"the highest mean accuracy ({top_orgs[0][1]['mean']:.1%}, n={top_orgs[0][1]['n']}), "
                          f"followed by {top_orgs[1][0]} ({top_orgs[1][1]['mean']:.1%}, n={top_orgs[1][1]['n']}) "
                          f"and {top_orgs[2][0]} ({top_orgs[2][1]['mean']:.1%}, n={top_orgs[2][1]['n']}).")

    return '\n'.join(results)


def main():
    """Generate manuscript results text."""
    parser = argparse.ArgumentParser(
        description='Generate manuscript results text for phenotype predictions'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='Base URL for the API')
    parser.add_argument('--output', default='manuscript_results.txt',
                       help='Output file for results text')

    args = parser.parse_args()

    print("Analyzing phenotype prediction data...")
    print("=" * 60)

    # Get detailed analysis
    data = get_detailed_phenotype_analysis(args.api_url)

    if not data:
        print("Error: Could not fetch or analyze data")
        return

    # Generate results text
    results_text = generate_results_text(data)

    # Write to file
    with open(args.output, 'w') as f:
        f.write(results_text)

    # Also print to console
    print("\n" + results_text)

    print(f"\n\nResults text saved to: {args.output}")


if __name__ == '__main__':
    main()