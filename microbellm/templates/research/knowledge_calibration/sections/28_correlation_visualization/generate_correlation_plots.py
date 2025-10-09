#!/usr/bin/env python3
"""
Generate publication-quality correlation visualization plots.
Creates a grid of plots showing the relationship between knowledge levels
and web search counts for different models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import argparse
from typing import Dict, List, Tuple
import os
import urllib.request
import urllib.error
from collections import defaultdict
import math

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

    return None

def process_correlation_data(data: Dict) -> List[Dict]:
    """Process and aggregate correlation data by model and knowledge level."""

    if not data or 'correlation_data' not in data:
        return []

    # Get the first dataset with search counts
    if not data.get('files_with_search_counts'):
        return []

    selected_dataset = data['files_with_search_counts'][0]
    file_data = data['correlation_data'][selected_dataset]

    # Aggregate data by model
    aggregated_data = {}

    for template_name, models in file_data.items():
        for model_name, model_info in models.items():
            if model_info.get('data_points') and len(model_info['data_points']) > 0:
                short_name = model_name.split('/')[-1][:20]  # Truncate long names

                if short_name not in aggregated_data:
                    aggregated_data[short_name] = {
                        'correlation': model_info['correlation_coefficient'],
                        'by_level': {1: [], 2: [], 3: []},
                        'raw_points': model_info['data_points']
                    }

                # Group by knowledge level (including NA/0)
                for point in model_info['data_points']:
                    level = point.get('knowledge_score', 0)
                    search_count = point.get('search_count', 1)
                    if level in [0, 1, 2, 3] and search_count > 0:
                        # Use log10 of search count
                        if level not in aggregated_data[short_name]['by_level']:
                            aggregated_data[short_name]['by_level'][level] = []
                        aggregated_data[short_name]['by_level'][level].append(math.log10(search_count))

    # Calculate statistics for each model and level
    model_stats = []
    for model_name, data in aggregated_data.items():
        stats = {}
        for level in [0, 1, 2, 3]:
            values = data['by_level'].get(level, [])
            if values:
                mean = np.mean(values)
                std = np.std(values)
                stats[level] = {
                    'mean': mean,
                    'std': std,
                    'count': len(values),
                    'values': values
                }

        if stats:  # Only include models with data
            model_stats.append({
                'model_name': model_name,
                'correlation': data['correlation'],
                'stats': stats,
                'raw_points': data['raw_points']
            })

    # Sort by correlation (descending)
    model_stats.sort(key=lambda x: x['correlation'], reverse=True)

    return model_stats

def create_correlation_grid_plot(model_stats: List[Dict], output_path: str = 'correlation_grid.pdf'):
    """Create a grid of correlation plots for multiple models."""

    if not model_stats:
        print("No model data available for plotting")
        return

    # Select top 16 models (or fewer if not available)
    models_to_plot = model_stats[:min(16, len(model_stats))]
    n_models = len(models_to_plot)

    # Create grid layout
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))

    # Ensure axes is always 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Calculate global bounds for consistent scaling
    all_x_values = []
    for model in models_to_plot:
        for stat in model['stats'].values():
            all_x_values.extend(stat['values'])

    if all_x_values:
        x_min = min(all_x_values) - 0.3
        x_max = max(all_x_values) + 0.3
    else:
        x_min, x_max = 0, 6

    # Calculate average statistics across all models for reference
    avg_stats = {}
    for level in [1, 2, 3]:
        all_means = []
        all_stds = []
        for model in model_stats:  # Use all models for average
            if level in model['stats']:
                all_means.append(model['stats'][level]['mean'])
                all_stds.append(model['stats'][level]['std'])

        if all_means:
            avg_stats[level] = {
                'mean': np.mean(all_means),
                'std': np.mean(all_stds)
            }

    # Create individual plots
    for idx, model in enumerate(models_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Set title with model name and correlation
        ax.set_title(f"{model['model_name']}\n(r = {model['correlation']:.3f})",
                     fontsize=10, fontweight='bold', pad=8)

        # Plot average reference lines (gray dashed)
        for level in [1, 2, 3]:
            if level in avg_stats:
                ax.axvline(avg_stats[level]['mean'],
                          ymin=(level-1)/3 + 0.05, ymax=level/3 - 0.05,
                          color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Plot model data
        for level in [1, 2, 3]:
            if level in model['stats']:
                stat = model['stats'][level]
                y = level

                # Plot error bar (horizontal)
                ax.errorbar(stat['mean'], y, xerr=stat['std'],
                           fmt='o', markersize=8,
                           color='#6366f1', markeredgecolor='#4f46e5',
                           markeredgewidth=1.5,
                           capsize=5, capthick=2,
                           elinewidth=2, alpha=0.9)

                # Add count annotation
                ax.text(stat['mean'], y + 0.15, f'n={stat["count"]}',
                       fontsize=8, ha='center', color='gray')

        # Customize axes
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.5, 3.5)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Limited', 'Moderate', 'Extensive'], fontsize=9)
        ax.set_xlabel('log₁₀(Search Count)', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add vertical line at x=0 for reference
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Add overall title
    fig.suptitle('Knowledge-Web Alignment: Correlation Visualization',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6366f1',
                   markersize=8, label='Model mean ± SD'),
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1,
                   label='All models average')
    ]
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Correlation grid plot saved to {output_path}")

    plt.close()

def create_top_models_detailed_plot(model_stats: List[Dict], output_path: str = 'correlation_top_detailed.pdf'):
    """Create a detailed plot for the top 4 models with individual data points."""

    if not model_stats:
        print("No model data available for plotting")
        return

    # Select top 4 models
    top_models = model_stats[:min(4, len(model_stats))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, model in enumerate(top_models):
        ax = axes[idx]

        # Plot individual data points
        if model.get('raw_points'):
            for point in model['raw_points']:
                x = math.log10(point.get('search_count', 1)) if point.get('search_count', 0) > 0 else 0
                y = point.get('knowledge_score', 0)

                # Add jitter to y for better visibility
                y_jittered = y + np.random.uniform(-0.15, 0.15)

                ax.scatter(x, y_jittered, alpha=0.3, s=20,
                          color='lightblue', edgecolor='none')

        # Overlay mean ± SD for each level
        for level in [1, 2, 3]:
            if level in model['stats']:
                stat = model['stats'][level]

                # Plot mean with error bar
                ax.errorbar(stat['mean'], level, xerr=stat['std'],
                           fmt='o', markersize=10,
                           color='#6366f1', markeredgecolor='#4f46e5',
                           markeredgewidth=2,
                           capsize=8, capthick=2.5,
                           elinewidth=2.5, alpha=1, zorder=10)

        # Add trend line if enough points
        if model.get('raw_points') and len(model['raw_points']) > 3:
            x_vals = []
            y_vals = []
            for point in model['raw_points']:
                if point.get('search_count', 0) > 0:
                    x_vals.append(math.log10(point['search_count']))
                    y_vals.append(point.get('knowledge_score', 0))

            if len(x_vals) > 1:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=1.5,
                       label=f'Trend (slope={z[0]:.3f})')

        # Customize axes
        ax.set_title(f"{model['model_name']}\nCorrelation: r = {model['correlation']:.3f}",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('log₁₀(Google Search Count)', fontsize=10)
        ax.set_ylabel('Knowledge Level', fontsize=10)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Limited', 'Moderate', 'Extensive'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Top Models - Detailed Knowledge-Web Correlation',
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Detailed correlation plot saved to {output_path}")

    plt.close()

def create_specific_models_plot(model_stats: List[Dict], model_names: List[str],
                               output_path: str = 'specific_models.pdf', include_na: bool = False):
    """Create a detailed plot for specific models by name."""

    # Find the specified models
    selected_models = []
    for name in model_names:
        for model in model_stats:
            if name.lower() in model['model_name'].lower():
                selected_models.append(model)
                break

    if not selected_models:
        print(f"No models found matching {model_names}")
        return

    n_models = len(selected_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, model in enumerate(selected_models):
        ax = axes[idx]

        # Plot individual data points with jitter
        if model.get('raw_points'):
            for point in model['raw_points']:
                x = math.log10(point.get('search_count', 1)) if point.get('search_count', 0) > 0 else 0
                y = point.get('knowledge_score', 0)

                # Add jitter to y for better visibility
                y_jittered = y + np.random.uniform(-0.15, 0.15)

                # Use different color for NA points
                color = 'gray' if y == 0 else 'lightblue'
                ax.scatter(x, y_jittered, alpha=0.3, s=30,
                          color=color, edgecolor='none')

        # Overlay mean ± SD for each level
        levels_to_plot = [0, 1, 2, 3] if include_na else [1, 2, 3]
        for level in levels_to_plot:
            if level in model['stats']:
                stat = model['stats'][level]

                # Use different color for NA level
                if level == 0:
                    color = '#8B8B8B'  # Gray for NA
                    edge_color = '#696969'
                else:
                    color = '#6366f1'  # Blue for knowledge levels
                    edge_color = '#4f46e5'

                # Plot mean with error bar
                ax.errorbar(stat['mean'], level, xerr=stat['std'],
                           fmt='o', markersize=12,
                           color=color, markeredgecolor=edge_color,
                           markeredgewidth=2,
                           capsize=10, capthick=3,
                           elinewidth=3, alpha=1, zorder=10)

                # Add count annotation
                ax.text(stat['mean'], level + 0.2, f'n={stat["count"]}',
                       fontsize=10, ha='center', color='gray', fontweight='bold')

        # Add trend line if enough points (optionally including NA)
        if model.get('raw_points') and len(model['raw_points']) > 3:
            x_vals = []
            y_vals = []
            for point in model['raw_points']:
                if point.get('search_count', 0) > 0:
                    y = point.get('knowledge_score', 0)
                    if not include_na and y == 0:
                        continue  # Skip NA if not including
                    x_vals.append(math.log10(point['search_count']))
                    y_vals.append(y)

            if len(x_vals) > 1:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2,
                       label=f'Trend (slope={z[0]:.3f})')

        # Customize axes
        ax.set_title(f"{model['model_name']}\nCorrelation: r = {model['correlation']:.3f}",
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('log₁₀(Google Search Count)', fontsize=11)
        ax.set_ylabel('Knowledge Level', fontsize=11)

        if include_na:
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(['NA/Unknown', 'Limited', 'Moderate', 'Extensive'], fontsize=10)
            ax.set_ylim(-0.5, 3.5)
        else:
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(['Limited', 'Moderate', 'Extensive'], fontsize=10)
            ax.set_ylim(0.5, 3.5)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=10)

        # Set consistent x-axis limits
        if model.get('raw_points'):
            x_vals = []
            for point in model['raw_points']:
                if point.get('search_count', 0) > 0:
                    x_vals.append(math.log10(point['search_count']))
            if x_vals:
                ax.set_xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)

    # Add overall title
    models_str = ' vs '.join([m['model_name'] for m in selected_models])
    fig.suptitle(f'Knowledge-Web Correlation: {models_str}',
                 fontsize=14, fontweight='bold', y=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Specific models plot saved to {output_path}")

    plt.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate correlation visualization plots'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='Base URL for the running web API')
    parser.add_argument('--output', '-o', default='correlation_grid.pdf',
                       help='Output file path for grid plot')
    parser.add_argument('--detailed', action='store_true',
                       help='Also generate detailed plot for top models')
    parser.add_argument('--models', nargs='+',
                       help='Specific model names to plot (e.g., gemini-2.5-pro gpt-4o-mini)')
    parser.add_argument('--include-na', action='store_true',
                       help='Include NA/Unknown (level 0) in the plots')

    args = parser.parse_args()

    # Fetch data
    print(f"Fetching correlation data from {args.api_url}...")
    data = fetch_correlation_data(args.api_url)

    if not data:
        print("Failed to fetch correlation data")
        return

    # Process data
    model_stats = process_correlation_data(data)

    if not model_stats:
        print("No model statistics to plot")
        return

    print(f"Found {len(model_stats)} models with correlation data")

    # Generate plots
    if args.models:
        # Generate plot for specific models
        suffix = '_with_na' if args.include_na else '_specific'
        specific_path = args.output.replace('.pdf', f'{suffix}.pdf')
        create_specific_models_plot(model_stats, args.models, specific_path, args.include_na)
    else:
        # Generate standard plots
        create_correlation_grid_plot(model_stats, args.output)

        if args.detailed:
            detailed_path = args.output.replace('.pdf', '_detailed.pdf')
            create_top_models_detailed_plot(model_stats, detailed_path)

if __name__ == '__main__':
    main()