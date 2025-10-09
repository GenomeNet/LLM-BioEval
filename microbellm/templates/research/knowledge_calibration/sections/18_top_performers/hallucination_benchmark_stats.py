#!/usr/bin/env python3
"""
Generate statistical summary for the Hallucination Benchmark results.
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

def calculate_hallucination_stats(data: Dict) -> Dict:
    """Calculate comprehensive statistics for hallucination benchmark."""

    if not data or 'knowledge_analysis' not in data:
        return {}

    knowledge_data = data['knowledge_analysis']

    # Aggregate statistics across all models
    model_stats = {}
    total_tests = 0
    total_na_responses = 0
    total_failed = 0
    total_hallucinated = 0  # Limited + Moderate + Extensive

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

        # Add to totals
        total_tests += stats['total']
        total_na_responses += stats['na']
        total_failed += stats['no_result'] + stats['inference_failed']
        total_hallucinated += stats['hallucinations']

    # Sort models by accuracy (correct rejection rate)
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    # Calculate overall statistics
    overall_stats = {
        'total_models': len(model_stats),
        'total_tests': total_tests,
        'total_queries': total_tests,  # Assuming 150 artificial names per model
        'overall_na_rate': (total_na_responses / total_tests * 100) if total_tests > 0 else 0,
        'overall_failed_rate': (total_failed / total_tests * 100) if total_tests > 0 else 0,
        'overall_correct_rejection_rate': ((total_na_responses + total_failed) / total_tests * 100) if total_tests > 0 else 0,
        'overall_hallucination_rate': (total_hallucinated / total_tests * 100) if total_tests > 0 else 0
    }

    return {
        'overall': overall_stats,
        'models': dict(sorted_models),
        'top_5': sorted_models[:5] if len(sorted_models) >= 5 else sorted_models,
        'bottom_5': sorted_models[-5:] if len(sorted_models) >= 5 else []
    }

def print_manuscript_results(stats: Dict):
    """Print formatted results for manuscript inclusion."""
    if not stats:
        print("No statistics available")
        return

    print("\n" + "="*80)
    print("HALLUCINATION BENCHMARK RESULTS - Template 3 (Query 3) Only")
    print("Statistics for Manuscript - Focusing on queries that allow NA responses")
    print("="*80 + "\n")

    overall = stats['overall']

    print("OVERALL STATISTICS (Template 3 Only):")
    print("-" * 40)
    print(f"Total models evaluated: {overall['total_models']}")
    print(f"Total test queries (Template 3): {overall['total_queries']:,}")
    print(f"Overall correct rejection rate: {overall['overall_correct_rejection_rate']:.1f}%")
    print(f"  - NA responses: {overall['overall_na_rate']:.1f}%")
    print(f"  - Failed inferences: {overall['overall_failed_rate']:.1f}%")
    print(f"Overall hallucination rate: {overall['overall_hallucination_rate']:.1f}%")

    print("\n" + "="*80)
    print("TOP PERFORMING MODELS (Best at Detecting Artificial Names):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Model':<30} {'Accuracy':<12} {'NA Rate':<12} {'Halluc. Rate'}")
    print("-" * 80)

    for i, (model_name, stats) in enumerate(stats['top_5'], 1):
        display_name = model_name.split('/')[-1][:28]  # Truncate long names
        print(f"{i:<6} {display_name:<30} {stats['accuracy']:>10.1f}% {stats['na']/stats['total']*100 if stats['total'] > 0 else 0:>10.1f}% {stats['hallucination_rate']:>12.1f}%")

    if stats.get('bottom_5'):
        print("\n" + "="*80)
        print("WORST PERFORMING MODELS (Most Prone to Hallucination):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<30} {'Accuracy':<12} {'NA Rate':<12} {'Halluc. Rate'}")
        print("-" * 80)

        bottom_models = list(reversed(stats['bottom_5']))
        for i, (model_name, model_stats) in enumerate(bottom_models, 1):
            display_name = model_name.split('/')[-1][:28]
            rank = overall['total_models'] - len(bottom_models) + i
            print(f"{rank:<6} {display_name:<30} {model_stats['accuracy']:>10.1f}% {model_stats['na']/model_stats['total']*100 if model_stats['total'] > 0 else 0:>10.1f}% {model_stats['hallucination_rate']:>12.1f}%")

    print("\n" + "="*80)
    print("SUGGESTED MANUSCRIPT TEXT:")
    print("="*80)

    # Get top and bottom performers
    if stats.get('top_5'):
        top1 = stats['top_5'][0]
        top2 = stats['top_5'][1] if len(stats['top_5']) > 1 else None
        top3 = stats['top_5'][2] if len(stats['top_5']) > 2 else None

        bottom1 = stats['bottom_5'][-1] if stats.get('bottom_5') else None

        print(f"""
[SUGGESTED TEXT FOR YOUR MANUSCRIPT - Add after your existing paragraph:]

Using Template 3 (Query 3), which explicitly allows models to respond with "NA" for unknown species,
we evaluated {overall['total_models']} models on our hallucination benchmark of 150 artificial binomial names.
The overall correct rejection rate across all models averaged {overall['overall_correct_rejection_rate']:.1f}%,
with models correctly identifying artificial taxa as unknown (NA) in {overall['overall_na_rate']:.1f}% of cases
and experiencing inference failures in {overall['overall_failed_rate']:.1f}% of cases. The remaining
{overall['overall_hallucination_rate']:.1f}% of responses constituted hallucinations, where models incorrectly
provided phenotypic information for non-existent species.

We observed substantial variation in hallucination resistance, with correct rejection rates ranging from
{min(m[1]['accuracy'] for m in stats['models'].items()) if stats['models'] else 0:.1f}% to
{max(m[1]['accuracy'] for m in stats['models'].items()) if stats['models'] else 0:.1f}%. The top-performing model,
{top1[0].split('/')[-1]}, achieved a {top1[1]['accuracy']:.1f}% accuracy in correctly identifying artificial taxa as
unknown (NA response rate: {top1[1]['na']/top1[1]['total']*100:.1f}%), while only hallucinating phenotypic
information in {top1[1]['hallucination_rate']:.1f}% of cases.""")

        if top2 and top3:
            print(f"""Other high-performing models included {top2[0].split('/')[-1]}
({top2[1]['accuracy']:.1f}% accuracy) and {top3[0].split('/')[-1]} ({top3[1]['accuracy']:.1f}% accuracy). """)

        if bottom1:
            print(f"""In contrast, the poorest-performing model, {bottom1[0].split('/')[-1]},
exhibited a hallucination rate of {bottom1[1]['hallucination_rate']:.1f}%, correctly
rejecting only {bottom1[1]['accuracy']:.1f}% of artificial names. Overall, models
demonstrated a mean correct rejection rate of {overall['overall_correct_rejection_rate']:.1f}%
(±SD), highlighting significant inter-model variability in hallucination susceptibility
when confronted with fabricated microbial taxa.""")

    print("\n" + "="*80)
    print("BREAKDOWN BY RESPONSE TYPE:")
    print("-" * 80)

    # Calculate averages across all models
    all_models = stats.get('models', {})
    if all_models:
        avg_na = np.mean([m['na']/m['total']*100 for m in all_models.values() if m['total'] > 0])
        avg_limited = np.mean([m['limited']/m['total']*100 for m in all_models.values() if m['total'] > 0])
        avg_moderate = np.mean([m['moderate']/m['total']*100 for m in all_models.values() if m['total'] > 0])
        avg_extensive = np.mean([m['extensive']/m['total']*100 for m in all_models.values() if m['total'] > 0])

        print(f"Average response distribution across all models:")
        print(f"  - NA (correct): {avg_na:.1f}%")
        print(f"  - Limited hallucination: {avg_limited:.1f}%")
        print(f"  - Moderate hallucination: {avg_moderate:.1f}%")
        print(f"  - Extensive hallucination: {avg_extensive:.1f}%")

    print("\n" + "="*80 + "\n")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate hallucination benchmark statistics')
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
    stats = calculate_hallucination_stats(data)

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