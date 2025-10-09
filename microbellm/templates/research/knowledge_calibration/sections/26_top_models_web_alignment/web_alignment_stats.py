#!/usr/bin/env python3
"""
Generate statistical summary for Web Alignment (Knowledge-Web Correlation) analysis.
Outputs key metrics for inclusion in manuscript text.
"""

import json
import urllib.request
import urllib.error
import os
import numpy as np
from typing import Dict, List, Tuple
import statistics

def fetch_correlation_data(api_url: str = None) -> Dict:
    """Fetch correlation data from the API."""
    base_url = api_url or os.environ.get('MICROBELLM_API_URL') or 'http://localhost:5050'
    endpoint = base_url.rstrip('/') + '/api/search_count_correlation'
    
    try:
        with urllib.request.urlopen(endpoint, timeout=10) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"Error fetching from API: {e}")
        return None

def calculate_web_alignment_stats(data: Dict) -> Dict:
    """Calculate comprehensive statistics for web alignment analysis."""
    
    if not data or 'correlation_data' not in data:
        return {}
    
    model_correlations = {}
    all_correlations = []
    total_data_points = 0
    
    # Process correlation data from all files
    for file_name, file_data in data['correlation_data'].items():
        for template_name, models in file_data.items():
            for model_name, stats in models.items():
                if stats.get('species_count', 0) < 2:  # Skip if too few data points
                    continue
                
                correlation = stats.get('correlation_coefficient', 0)
                species_count = stats.get('species_count', 0)
                
                if model_name not in model_correlations:
                    model_correlations[model_name] = {
                        'correlations': [],
                        'total_species': 0,
                        'knowledge_distribution': stats.get('knowledge_distribution', {}),
                        'templates': []
                    }
                
                model_correlations[model_name]['correlations'].append(correlation)
                model_correlations[model_name]['total_species'] += species_count
                model_correlations[model_name]['templates'].append(template_name)
                
                all_correlations.append(correlation)
                total_data_points += species_count
    
    # Calculate average correlations per model
    model_results = []
    for model_name, data in model_correlations.items():
        avg_correlation = np.mean(data['correlations']) if data['correlations'] else 0
        model_results.append({
            'model_name': model_name,
            'display_name': model_name.split('/')[-1],
            'avg_correlation': avg_correlation,
            'num_templates': len(set(data['templates'])),
            'total_species': data['total_species'],
            'knowledge_distribution': data['knowledge_distribution']
        })
    
    # Sort by average correlation (descending)
    model_results.sort(key=lambda x: x['avg_correlation'], reverse=True)
    
    # Calculate overall statistics
    overall_stats = {}
    if all_correlations:
        overall_stats = {
            'total_models': len(model_correlations),
            'total_data_points': total_data_points,
            'mean_correlation': np.mean(all_correlations),
            'median_correlation': np.median(all_correlations),
            'std_correlation': np.std(all_correlations),
            'min_correlation': min(all_correlations),
            'max_correlation': max(all_correlations),
            'correlation_range': max(all_correlations) - min(all_correlations)
        }
    
    return {
        'overall': overall_stats,
        'models': model_results,
        'top_5': model_results[:5] if len(model_results) >= 5 else model_results,
        'bottom_5': model_results[-5:] if len(model_results) >= 5 else [],
        'files_analyzed': list(data.get('files_with_search_counts', []))
    }

def print_manuscript_results(stats: Dict):
    """Print formatted results for manuscript inclusion."""
    if not stats or not stats.get('overall'):
        print("No statistics available")
        return
    
    print("\n" + "="*80)
    print("WEB ALIGNMENT (KNOWLEDGE-WEB CORRELATION) ANALYSIS")
    print("Statistics for Manuscript")
    print("="*80 + "\n")
    
    overall = stats['overall']
    
    print("OVERALL STATISTICS:")
    print("-" * 40)
    print(f"Total models evaluated: {overall['total_models']}")
    print(f"Total data points analyzed: {overall['total_data_points']:,}")
    print(f"Mean correlation coefficient: {overall['mean_correlation']:.3f}")
    print(f"Median correlation coefficient: {overall['median_correlation']:.3f}")
    print(f"Standard deviation: {overall['std_correlation']:.3f}")
    print(f"Correlation range: {overall['min_correlation']:.3f} to {overall['max_correlation']:.3f}")
    
    print("\n" + "="*80)
    print("TOP PERFORMING MODELS (Highest Knowledge-Web Alignment):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Model':<35} {'Avg Correlation':<18} {'Data Points'}")
    print("-" * 80)
    
    for i, model in enumerate(stats['top_5'], 1):
        print(f"{i:<6} {model['display_name'][:33]:<35} {model['avg_correlation']:>15.3f} {model['total_species']:>15}")
    
    if stats.get('bottom_5'):
        print("\n" + "="*80)
        print("WORST PERFORMING MODELS (Lowest Knowledge-Web Alignment):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<35} {'Avg Correlation':<18} {'Data Points'}")
        print("-" * 80)
        
        total = len(stats['models'])
        for i, model in enumerate(reversed(stats['bottom_5']), 1):
            rank = total - len(stats['bottom_5']) + i
            print(f"{rank:<6} {model['display_name'][:33]:<35} {model['avg_correlation']:>15.3f} {model['total_species']:>15}")
    
    print("\n" + "="*80)
    print("DATASETS ANALYZED:")
    print("-" * 40)
    for file_name in stats['files_analyzed']:
        print(f"  - {file_name}")
    
    print("\n" + "="*80)
    print("SUGGESTED MANUSCRIPT TEXT:")
    print("="*80)
    
    # Get top performers
    if stats['top_5']:
        top1 = stats['top_5'][0]
        top2 = stats['top_5'][1] if len(stats['top_5']) > 1 else None
        top3 = stats['top_5'][2] if len(stats['top_5']) > 2 else None
        
        print(f"""
[SUGGESTED TEXT FOR YOUR MANUSCRIPT:]

To assess the alignment between model knowledge and real-world microbial prevalence,
we analyzed the correlation between assigned knowledge levels and web search frequencies
for {overall['total_data_points']:,} species-model pairs across {overall['total_models']} models.

The analysis revealed substantial variation in knowledge-web alignment, with correlation
coefficients ranging from {overall['min_correlation']:.3f} to {overall['max_correlation']:.3f} 
(mean = {overall['mean_correlation']:.3f} ± {overall['std_correlation']:.3f}). This indicates
that models differ significantly in their ability to appropriately calibrate their knowledge
confidence with real-world species prevalence.

The top-performing model, {top1['display_name']}, achieved an average correlation of
{top1['avg_correlation']:.3f} between its knowledge assessments and web presence metrics,
suggesting strong alignment with real-world information availability.""")
        
        if top2 and top3:
            print(f""" Other high-performing models
included {top2['display_name']} (r = {top2['avg_correlation']:.3f}) and
{top3['display_name']} (r = {top3['avg_correlation']:.3f}).""")
        
        if stats.get('bottom_5'):
            bottom1 = stats['bottom_5'][-1]
            print(f"""
In contrast, the poorest-performing models showed weak or negative correlations,
with {bottom1['display_name']} achieving only r = {bottom1['avg_correlation']:.3f},
indicating a misalignment between model confidence and actual species prevalence
in scientific literature and web resources.""")
        
        # Calculate fold difference if meaningful
        if stats['top_5'] and stats.get('bottom_5') and top1['avg_correlation'] > 0 and stats['bottom_5'][-1]['avg_correlation'] > 0:
            fold_diff = top1['avg_correlation'] / abs(stats['bottom_5'][-1]['avg_correlation'])
            if fold_diff > 1.5:
                print(f"""
This represents a {fold_diff:.1f}-fold difference in alignment quality between
the best and worst performers, highlighting the importance of model selection
for applications requiring accurate assessment of species prevalence and importance.""")
    
    print("\n" + "="*80)
    print("INTERPRETATION NOTES:")
    print("-" * 40)
    print("""
- Higher correlation indicates better alignment between model knowledge and web presence
- Correlation > 0.7: Strong alignment
- Correlation 0.4-0.7: Moderate alignment  
- Correlation < 0.4: Weak alignment
- Negative correlation: Inverse relationship (problematic)""")
    
    print("\n" + "="*80 + "\n")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate web alignment statistics')
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='API URL for fetching data')
    parser.add_argument('--output', '-o', help='Output file for results (optional)')
    
    args = parser.parse_args()
    
    # Fetch data from API
    print(f"Fetching correlation data from {args.api_url}...")
    data = fetch_correlation_data(args.api_url)
    
    if not data:
        print("Failed to fetch correlation data from API")
        return
    
    # Calculate statistics
    stats = calculate_web_alignment_stats(data)
    
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