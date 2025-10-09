#!/usr/bin/env python3
"""
Generate stacked bar plot for Hallucination Benchmark Top Performers.
Shows all top models in a single stacked bar chart visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from typing import Dict, List
import os
import urllib.request
import urllib.error

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

def create_stacked_bar_plot(api_url: str, output_path: str = 'top_performers_stacked.pdf'):
    """Create a stacked bar plot of top performers in hallucination benchmark."""

    # Fetch data
    data = fetch_from_api(api_url)
    if not data or 'knowledge_analysis' not in data:
        print("No knowledge analysis data available")
        return

    knowledge_data = data['knowledge_analysis']

    # Aggregate statistics across all models (Template 3 only)
    model_stats = {}

    for file_name, file_data in knowledge_data.items():
        if not file_data.get('has_type_column') or not file_data.get('types'):
            continue

        for input_type, template_data in file_data['types'].items():
            if input_type in ['UNCLASSIFIED', 'WA_WITH_GCOUNT']:
                continue

            for template_name, models in template_data.items():
                # Filter to only Template 3 (allows NA responses)
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
                            'total': 0
                        }

                    # Update counts
                    for key in ['NA', 'limited', 'moderate', 'extensive', 'no_result', 'inference_failed']:
                        count = stats.get(key, 0)
                        model_stats[model_name][key.lower()] = model_stats[model_name].get(key.lower(), 0) + count

                    model_stats[model_name]['total'] += stats.get('total', 0)

    # Calculate metrics and prepare data
    model_results = []
    for model_name, stats in model_stats.items():
        if stats['total'] > 0:
            # Combine failed categories with NA for correct rejections
            correct_rejections = stats['na'] + stats['no_result'] + stats['inference_failed']
            hallucinations = stats['limited'] + stats['moderate'] + stats['extensive']
            accuracy = (correct_rejections / stats['total']) * 100

            model_results.append({
                'name': model_name.split('/')[-1][:25],  # Truncate long names
                'full_name': model_name,
                'accuracy': accuracy,
                'hallucination_rate': (hallucinations / stats['total']) * 100,
                'na_combined_pct': (correct_rejections / stats['total']) * 100,
                'na_combined_count': correct_rejections,
                'limited_pct': (stats['limited'] / stats['total']) * 100,
                'limited_count': stats['limited'],
                'moderate_pct': (stats['moderate'] / stats['total']) * 100,
                'moderate_count': stats['moderate'],
                'extensive_pct': (stats['extensive'] / stats['total']) * 100,
                'extensive_count': stats['extensive'],
                'total': stats['total']
            })

    # Sort by accuracy (descending - higher is better for hallucination detection)
    model_results.sort(key=lambda x: x['accuracy'], reverse=True)

    # Take top 10 models
    top_models = model_results[:10]

    if not top_models:
        print("No models to plot")
        return

    # Prepare data for plotting
    model_names = [m['name'] for m in top_models]
    accuracies = [m['accuracy'] for m in top_models]

    na_values = [m['na_combined_pct'] for m in top_models]
    limited_values = [m['limited_pct'] for m in top_models]
    moderate_values = [m['moderate_pct'] for m in top_models]
    extensive_values = [m['extensive_pct'] for m in top_models]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={'height_ratios': [1, 3]})

    # Top subplot: Accuracy (correct rejection rate)
    bars = ax1.barh(range(len(model_names)), accuracies, color='#7e57c2', alpha=0.7)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names, fontsize=10)
    ax1.set_xlabel('Correct Rejection Rate (%)', fontsize=11)
    ax1.set_title('Hallucination Benchmark - Correct Rejection of Artificial Taxa', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 105)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    # Bottom subplot: Stacked bar chart
    bar_height = 0.6
    y_positions = np.arange(len(model_names))

    # Define colors
    colors = {
        'na_failed': '#8B8B8B',  # Darker gray for NA/Failed (correct rejections)
        'limited': '#FFD700',    # Gold/yellow for limited hallucination
        'moderate': '#87CEEB',   # Sky blue for moderate hallucination
        'extensive': '#FF6B6B'   # Light red for extensive hallucination
    }

    # Create stacked bars
    p1 = ax2.barh(y_positions, na_values, bar_height,
                  label='NA/Failed (Correct)', color=colors['na_failed'],
                  edgecolor='black', linewidth=0.5)
    p2 = ax2.barh(y_positions, limited_values, bar_height, left=na_values,
                  label='Limited Hallucination', color=colors['limited'],
                  edgecolor='black', linewidth=0.5)

    # Calculate cumulative positions
    left_moderate = [na + lim for na, lim in zip(na_values, limited_values)]
    p3 = ax2.barh(y_positions, moderate_values, bar_height, left=left_moderate,
                  label='Moderate Hallucination', color=colors['moderate'],
                  edgecolor='black', linewidth=0.5)

    left_extensive = [na + lim + mod for na, lim, mod in
                      zip(na_values, limited_values, moderate_values)]
    p4 = ax2.barh(y_positions, extensive_values, bar_height, left=left_extensive,
                  label='Extensive Hallucination', color=colors['extensive'],
                  edgecolor='black', linewidth=0.5)

    # Add percentage and count labels on segments (only if > 5%)
    for i in range(len(model_names)):
        cumulative = 0
        segments = [
            (na_values[i], top_models[i]['na_combined_count'], colors['na_failed'], 'NA'),
            (limited_values[i], top_models[i]['limited_count'], colors['limited'], 'Lim'),
            (moderate_values[i], top_models[i]['moderate_count'], colors['moderate'], 'Mod'),
            (extensive_values[i], top_models[i]['extensive_count'], colors['extensive'], 'Ext')
        ]

        for pct_value, count, color, label in segments:
            if pct_value > 5:  # Only show if > 5%
                # Show both percentage and count
                text_label = f'{pct_value:.0f}%\n({count})'
                ax2.text(cumulative + pct_value/2, i, text_label,
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color='white' if label == 'NA' else 'black')
            cumulative += pct_value

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(model_names, fontsize=10)
    ax2.set_xlabel('Response Distribution (%)', fontsize=11)
    ax2.set_title('Knowledge Level Distribution - Template 3 Only', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)

    # Add legend
    ax2.legend(loc='lower right', frameon=True, fontsize=10,
               title='Response Type', title_fontsize=10)

    # Add grid
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Overall title
    fig.suptitle('Top Performing Models - Hallucination Benchmark',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Stacked bar plot saved to {output_path}")

    # Also create top 5 focused plot
    fig2, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(model_names[:5]))  # Top 5 only
    accuracies_top5 = accuracies[:5]
    model_names_top5 = model_names[:5]

    bars = ax.barh(y_pos, accuracies_top5, color='#7e57c2', alpha=0.8, height=0.6)

    # Add value labels
    for bar, val in zip(bars, accuracies_top5):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names_top5, fontsize=12)
    ax.set_xlabel('Correct Rejection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Models - Hallucination Benchmark',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    output_accuracy = output_path.replace('.pdf', '_accuracy_only.pdf')
    plt.tight_layout()
    plt.savefig(output_accuracy, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Accuracy-only plot saved to {output_accuracy}")

    plt.close('all')

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate stacked bar plot for Hallucination Benchmark Top Performers'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='Base URL for the running web API')
    parser.add_argument('--output', '-o', default='top_performers_stacked.pdf',
                       help='Output file path')

    args = parser.parse_args()

    create_stacked_bar_plot(args.api_url, args.output)

if __name__ == '__main__':
    main()