#!/usr/bin/env python3
"""
Generate publication-quality plots for Worst Models - Web Alignment.
Filters out incomplete runs and specific models to show genuine worst performers.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from typing import Dict, List
import os
import urllib.request
import urllib.error

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

def create_worst_models_stacked_plot(api_url: str, output_path: str = 'worst_web_alignment_stacked.pdf',
                                     min_sample_size: int = 100,
                                     exclude_models: List[str] = None):
    """Create a stacked bar plot of worst performing models in web alignment.

    Args:
        api_url: API URL for fetching data
        output_path: Output file path
        min_sample_size: Minimum number of data points required to include a model
        exclude_models: List of model names to exclude (e.g., ['gemini-2.5-pro'])
    """

    if exclude_models is None:
        exclude_models = ['gemini-2.5-pro']  # Default exclusion

    # Fetch data
    data = fetch_correlation_data(api_url)
    if not data or 'correlation_data' not in data:
        print("No correlation data available")
        return

    # Process correlation data from all files
    model_results = []

    for file_name, file_data in data['correlation_data'].items():
        for template_name, models in file_data.items():
            for model_name, stats in models.items():
                # Get short name for display
                short_name = model_name.split('/')[-1]

                # Skip excluded models
                if any(excluded.lower() in short_name.lower() for excluded in exclude_models):
                    print(f"Excluding {short_name} from worst performers list")
                    continue

                species_count = stats.get('species_count', 0)

                # Filter by minimum sample size
                if species_count < min_sample_size:
                    print(f"Skipping {short_name}: insufficient data (n={species_count}, min={min_sample_size})")
                    continue

                dist = stats.get('knowledge_distribution', {})
                total = species_count

                if total > 0:
                    model_results.append({
                        'name': short_name[:25],  # Truncate long names
                        'correlation': stats.get('correlation_coefficient', 0),
                        'na_pct': (dist.get('NA', 0) / total) * 100,
                        'na_count': dist.get('NA', 0),
                        'limited_pct': (dist.get('limited', 0) / total) * 100,
                        'limited_count': dist.get('limited', 0),
                        'moderate_pct': (dist.get('moderate', 0) / total) * 100,
                        'moderate_count': dist.get('moderate', 0),
                        'extensive_pct': (dist.get('extensive', 0) / total) * 100,
                        'extensive_count': dist.get('extensive', 0),
                        'total': total
                    })

    # Sort by correlation (ascending - worst performers first)
    model_results.sort(key=lambda x: x['correlation'])

    # Take worst 10 models
    worst_models = model_results[:10]

    if not worst_models:
        print("No models to plot after filtering")
        return

    print(f"\nWorst performing models (lowest correlation):")
    for i, model in enumerate(worst_models[:5], 1):
        print(f"{i}. {model['name']}: r={model['correlation']:.3f}, n={model['total']}")

    # Prepare data for plotting
    model_names = [m['name'] for m in worst_models]
    correlations = [m['correlation'] for m in worst_models]

    na_values = [m['na_pct'] for m in worst_models]
    limited_values = [m['limited_pct'] for m in worst_models]
    moderate_values = [m['moderate_pct'] for m in worst_models]
    extensive_values = [m['extensive_pct'] for m in worst_models]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={'height_ratios': [1, 3]})

    # Top subplot: Correlation coefficients (red for poor alignment)
    bars = ax1.barh(range(len(model_names)), correlations, color='#dc2626', alpha=0.7)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names, fontsize=10)
    ax1.set_xlabel('Correlation Coefficient (lower = worse alignment)', fontsize=11)
    ax1.set_title('Worst Models - Knowledge-Web Alignment', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Set x-axis limits to show negative and positive range
    max_abs = max(abs(min(correlations)), abs(max(correlations))) * 1.2
    ax1.set_xlim(-max_abs, max_abs)
    ax1.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, correlations)):
        x_pos = bar.get_width() + (0.01 if val >= 0 else -0.01)
        ha = 'left' if val >= 0 else 'right'
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, ha=ha)

    # Bottom subplot: Stacked bar chart
    bar_height = 0.6
    y_positions = np.arange(len(model_names))

    # Define colors
    colors = {
        'na': '#8B8B8B',      # Darker gray for better visibility
        'limited': '#FFD700',  # Gold/yellow
        'moderate': '#87CEEB', # Sky blue
        'extensive': '#90EE90' # Light green
    }

    # Create stacked bars
    p1 = ax2.barh(y_positions, na_values, bar_height,
                  label='NA/Unknown', color=colors['na'], edgecolor='black', linewidth=0.5)
    p2 = ax2.barh(y_positions, limited_values, bar_height, left=na_values,
                  label='Limited', color=colors['limited'], edgecolor='black', linewidth=0.5)

    # Calculate cumulative positions
    left_moderate = [na + lim for na, lim in zip(na_values, limited_values)]
    p3 = ax2.barh(y_positions, moderate_values, bar_height, left=left_moderate,
                  label='Moderate', color=colors['moderate'], edgecolor='black', linewidth=0.5)

    left_extensive = [na + lim + mod for na, lim, mod in
                      zip(na_values, limited_values, moderate_values)]
    p4 = ax2.barh(y_positions, extensive_values, bar_height, left=left_extensive,
                  label='Extensive', color=colors['extensive'], edgecolor='black', linewidth=0.5)

    # Add percentage and count labels on segments (only if > 5%)
    for i in range(len(model_names)):
        cumulative = 0
        segments = [
            (na_values[i], worst_models[i]['na_count'], colors['na'], 'NA'),
            (limited_values[i], worst_models[i]['limited_count'], colors['limited'], 'Lim'),
            (moderate_values[i], worst_models[i]['moderate_count'], colors['moderate'], 'Mod'),
            (extensive_values[i], worst_models[i]['extensive_count'], colors['extensive'], 'Ext')
        ]

        for pct_value, count, color, label in segments:
            if pct_value > 5:  # Only show if > 5%
                # Show both percentage and count
                text_label = f'{pct_value:.0f}%\n({count})'
                ax2.text(cumulative + pct_value/2, i, text_label,
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color='white' if label == 'NA' else 'black')
            cumulative += pct_value

    # Add sample size annotations on the right
    for i, model in enumerate(worst_models):
        ax2.text(101, i, f'n={model["total"]}', fontsize=8,
                va='center', ha='left', color='gray')

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(model_names, fontsize=10)
    ax2.set_xlabel('Knowledge Distribution (%)', fontsize=11)
    ax2.set_title('Knowledge Level Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 105)  # Extended to show sample size

    # Add legend
    ax2.legend(loc='lower right', frameon=True, fontsize=10,
               title='Knowledge Level', title_fontsize=10)

    # Add grid
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Overall title
    fig.suptitle(f'Worst Performing Models - Web Alignment\n(Minimum n={min_sample_size}, excluding: {", ".join(exclude_models)})',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nStacked bar plot saved to {output_path}")

    # Also create correlation-only plot for worst 5
    fig2, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(model_names[:5]))  # Worst 5 only
    correlations_worst5 = correlations[:5]
    model_names_worst5 = model_names[:5]

    bars = ax.barh(y_pos, correlations_worst5, color='#dc2626', alpha=0.8, height=0.6)

    # Add value labels
    for bar, val in zip(bars, correlations_worst5):
        x_pos = bar.get_width() + (0.005 if val >= 0 else -0.005)
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=11, fontweight='bold', ha=ha)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names_worst5, fontsize=12)
    ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Worst 5 Models - Knowledge-Web Alignment',
                 fontsize=14, fontweight='bold', pad=20)

    # Set x-axis to show negative values
    max_abs = max(abs(min(correlations_worst5)), abs(max(correlations_worst5))) * 1.2
    ax.set_xlim(-max_abs, max_abs)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    output_correlation = output_path.replace('.pdf', '_correlation_only.pdf')
    plt.tight_layout()
    plt.savefig(output_correlation, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Correlation-only plot saved to {output_correlation}")

    plt.close('all')

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate worst performers plot for Web Alignment'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='Base URL for the running web API')
    parser.add_argument('--output', '-o', default='worst_web_alignment_stacked.pdf',
                       help='Output file path')
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum sample size to include a model (default: 100)')
    parser.add_argument('--exclude', nargs='+', default=['gemini-2.5-pro'],
                       help='Models to exclude from worst performers list')

    args = parser.parse_args()

    print(f"Generating worst performers plot with filters:")
    print(f"- Minimum sample size: {args.min_samples}")
    print(f"- Excluding models: {args.exclude}")

    create_worst_models_stacked_plot(args.api_url, args.output,
                                    args.min_samples, args.exclude)

if __name__ == '__main__':
    main()