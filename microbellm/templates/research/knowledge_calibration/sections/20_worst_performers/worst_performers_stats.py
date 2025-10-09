#!/usr/bin/env python3
"""
Generate statistical summary for the Worst Performing Models in Hallucination Benchmark.
Outputs key metrics for inclusion in manuscript text.
"""

import json
import urllib.request
import urllib.error
import os
import sqlite3
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

def fetch_from_api(api_url: str = None) -> Dict:
    """Fetch knowledge analysis data from the API."""
    base_url = api_url or os.environ.get('MICROBELLM_API_URL') or 'http://localhost:5050'
    endpoint = base_url.rstrip('/') + '/api/knowledge_analysis_data'

    try:
        with urllib.request.urlopen(endpoint, timeout=10) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"Error fetching from API: {e}")
        return None

def calculate_worst_performer_stats(data: Dict) -> Dict:
    """Calculate comprehensive statistics for worst performing models in hallucination benchmark."""

    if not data or 'knowledge_analysis' not in data:
        return {}

    knowledge_data = data['knowledge_analysis']

    # Aggregate statistics across all models
    model_stats = {}

    # Process each file/model/template combination
    for file_name, file_data in knowledge_data.items():
        if not file_data.get('has_type_column') or not file_data.get('types'):
            continue

        for input_type, template_data in file_data['types'].items():
            if input_type in ['UNCLASSIFIED', 'WA_WITH_GCOUNT']:
                continue

            for template_name, models in template_data.items():
                # Filter to only include Template 3 (Query 3) which allows NA responses
                # Note: template name has typo "knowlege" in the database
                if 'template3' not in template_name.lower():
                    continue

                for model_name, stats in models.items():
                    if model_name not in model_stats:
                        model_stats[model_name] = {
                            'na': 0,
                            'limited': 0,
                            'moderate': 0,
                            'extensive': 0,
                            'no_result': 0,
                            'inference_failed': 0,
                            'total': 0,
                            'correct_rejections': 0,  # NA + failed
                            'hallucinations': 0,  # limited + moderate + extensive
                            'accuracy': 0.0
                        }

                    # Update counts
                    for key in ['NA', 'limited', 'moderate', 'extensive', 'no_result', 'inference_failed']:
                        count = stats.get(key, 0)
                        model_stats[model_name][key.lower()] = model_stats[model_name].get(key.lower(), 0) + count

                    model_stats[model_name]['total'] += stats.get('total', 0)

    # Calculate derived metrics for each model
    for model_name, stats in model_stats.items():
        stats['correct_rejections'] = stats['na'] + stats['no_result'] + stats['inference_failed']
        stats['hallucinations'] = stats['limited'] + stats['moderate'] + stats['extensive']

        if stats['total'] > 0:
            stats['accuracy'] = (stats['correct_rejections'] / stats['total']) * 100
            stats['hallucination_rate'] = (stats['hallucinations'] / stats['total']) * 100

    # Sort models by accuracy (correct rejection rate) - ascending for worst performers
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['accuracy'])

    # Get worst performers
    worst_5 = sorted_models[:5] if len(sorted_models) >= 5 else sorted_models

    # Calculate statistics for worst performers group
    if worst_5:
        worst_stats = {
            'total_tests': sum(m[1]['total'] for m in worst_5),
            'total_na': sum(m[1]['na'] for m in worst_5),
            'total_failed': sum(m[1]['no_result'] + m[1]['inference_failed'] for m in worst_5),
            'total_hallucinated': sum(m[1]['hallucinations'] for m in worst_5),
            'avg_accuracy': np.mean([m[1]['accuracy'] for m in worst_5]),
            'avg_hallucination_rate': np.mean([m[1]['hallucination_rate'] for m in worst_5])
        }
    else:
        worst_stats = {}

    return {
        'all_models': dict(sorted_models),
        'worst_5': worst_5,
        'worst_stats': worst_stats,
        'total_models': len(model_stats)
    }

def print_manuscript_results(stats: Dict):
    """Print formatted results for manuscript inclusion."""
    if not stats or not stats.get('worst_5'):
        print("No statistics available")
        return

    print("\n" + "="*80)
    print("WORST PERFORMING MODELS - HALLUCINATION BENCHMARK")
    print("Statistics for Manuscript - Template 3 (Query 3) Only")
    print("="*80 + "\n")

    worst_stats = stats['worst_stats']
    total_models = stats['total_models']

    print("WORST 5 MODELS STATISTICS:")
    print("-" * 40)
    print(f"Number of worst models analyzed: 5")
    print(f"Total test queries (5 worst models): {worst_stats['total_tests']:,}")
    print(f"Average correct rejection rate: {worst_stats['avg_accuracy']:.1f}%")
    print(f"Average hallucination rate: {worst_stats['avg_hallucination_rate']:.1f}%")

    print("\n" + "="*80)
    print("INDIVIDUAL MODEL PERFORMANCE (5 Worst):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Model':<35} {'Accuracy':<12} {'NA Rate':<12} {'Halluc. Rate'}")
    print("-" * 80)

    for i, (model_name, model_stats) in enumerate(stats['worst_5'], 1):
        display_name = model_name.split('/')[-1][:33]  # Truncate long names
        na_rate = (model_stats['na']/model_stats['total']*100) if model_stats['total'] > 0 else 0
        print(f"{i:<6} {display_name:<35} {model_stats['accuracy']:>10.1f}% {na_rate:>10.1f}% {model_stats['hallucination_rate']:>12.1f}%")

    print("\n" + "="*80)
    print("BREAKDOWN BY HALLUCINATION SEVERITY (5 Worst Models):")
    print("-" * 80)

    # Calculate averages across worst 5 models
    if stats['worst_5']:
        avg_limited = np.mean([m[1]['limited']/m[1]['total']*100 for m in stats['worst_5'] if m[1]['total'] > 0])
        avg_moderate = np.mean([m[1]['moderate']/m[1]['total']*100 for m in stats['worst_5'] if m[1]['total'] > 0])
        avg_extensive = np.mean([m[1]['extensive']/m[1]['total']*100 for m in stats['worst_5'] if m[1]['total'] > 0])

        print(f"Average hallucination distribution across worst 5 models:")
        print(f"  - Limited hallucination: {avg_limited:.1f}%")
        print(f"  - Moderate hallucination: {avg_moderate:.1f}%")
        print(f"  - Extensive hallucination: {avg_extensive:.1f}%")
        print(f"  - Total hallucination rate: {avg_limited + avg_moderate + avg_extensive:.1f}%")

    print("\n" + "="*80)
    print("SUGGESTED MANUSCRIPT TEXT:")
    print("="*80)

    # Get worst performer details
    if stats['worst_5']:
        worst1 = stats['worst_5'][0]
        worst2 = stats['worst_5'][1] if len(stats['worst_5']) > 1 else None
        worst3 = stats['worst_5'][2] if len(stats['worst_5']) > 2 else None

        print(f"""
[SUGGESTED TEXT FOR YOUR MANUSCRIPT - Focus on worst performers:]

At the opposite end of the performance spectrum, we identified models with 
significantly higher hallucination rates when confronted with artificial taxa names. 
The five worst-performing models exhibited an average correct rejection rate of only 
{worst_stats['avg_accuracy']:.1f}%, hallucinating phenotypic information for non-existent 
species in {worst_stats['avg_hallucination_rate']:.1f}% of cases.

The worst performer, {worst1[0].split('/')[-1]}, correctly identified artificial taxa 
as unknown in only {worst1[1]['accuracy']:.1f}% of cases, with a hallucination rate of 
{worst1[1]['hallucination_rate']:.1f}%. """)

        if worst2 and worst3:
            print(f"""Similarly poor performance was observed in 
{worst2[0].split('/')[-1]} ({worst2[1]['hallucination_rate']:.1f}% hallucination rate) and 
{worst3[0].split('/')[-1]} ({worst3[1]['hallucination_rate']:.1f}% hallucination rate).""")

        print(f"""
Among these worst performers, hallucination severity varied, with an average of 
{avg_limited:.1f}% providing limited incorrect information, {avg_moderate:.1f}% generating 
moderate hallucinations, and {avg_extensive:.1f}% producing extensive fabricated details 
about non-existent microbial species.

These results highlight substantial variability in model reliability when handling 
unknown or fabricated microbial taxa, with the worst-performing models showing 
hallucination rates exceeding {worst1[1]['hallucination_rate']:.0f}%, compared to the 
best performers which maintained hallucination rates below 10%. This {int(worst1[1]['hallucination_rate']/10):.0f}-fold 
difference in hallucination susceptibility underscores the importance of careful model 
selection for microbiological applications where accurate identification of unknown 
species is critical.""")

    print("\n" + "="*80)
    print("COMPARISON WITH TOP PERFORMERS:")
    print("-" * 80)

    # Note: These values would need to be updated based on actual top performer data
    print(f"""Key contrasts:
- Worst 5 models average accuracy: {worst_stats['avg_accuracy']:.1f}%
- Best models achieve >90% accuracy (see top_performers analysis)
- Hallucination rate range: {worst_stats['avg_hallucination_rate']:.1f}% (worst) vs <10% (best)
- This represents a significant reliability gap in handling unknown taxa""")

    print("\n" + "="*80 + "\n")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate worst performer statistics for hallucination benchmark')
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='API URL for fetching data')
    parser.add_argument('--output', '-o', help='Output file for results (optional)')

    args = parser.parse_args()

    # Fetch data from API
    print(f"Fetching data from {args.api_url}...")
    data = fetch_from_api(args.api_url)

    if not data:
        print("Failed to fetch data from API")
        return

    # Calculate statistics
    stats = calculate_worst_performer_stats(data)

    # Print results
    if args.output:
        import sys
        orig_stdout = sys.stdout
        with open(args.output, 'w') as f:
            sys.stdout = f
            print_manuscript_results(stats)
        sys.stdout = orig_stdout
        print(f"Results saved to {args.output}")
    else:
        print_manuscript_results(stats)

if __name__ == '__main__':
    main()