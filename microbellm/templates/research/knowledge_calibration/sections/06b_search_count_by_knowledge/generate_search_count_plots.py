#!/usr/bin/env python3
"""
Generate publication-quality plots for Search Count by Knowledge Group visualization.
Creates violin plots, bar charts, and scatter plots showing the distribution of
Google Scholar search counts across different model-reported knowledge levels.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import json
import argparse
import urllib.request
import urllib.error
from typing import Dict, List, Tuple
import os

def fetch_search_knowledge_data(api_url: str = None, model: str = None,
                               species_file: str = 'wa_with_gcount.txt') -> Dict:
    """Fetch search count by knowledge data from the API."""
    base_url = api_url or os.environ.get('MICROBELLM_API_URL') or 'http://localhost:5050'

    # Build URL with parameters
    endpoint = f"{base_url.rstrip('/')}/api/search_count_by_knowledge"
    params = f"?species_file={species_file}"
    if model:
        params += f"&model={urllib.parse.quote(model)}"

    full_url = endpoint + params
    print(f"Fetching data from: {full_url}")

    try:
        with urllib.request.urlopen(full_url, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"Error fetching from API: {e}")
        return None

    return None

def create_violin_plot(data: Dict, output_path: str = 'search_count_violin.pdf',
                      y_scale: str = 'log', figsize: Tuple[float, float] = (3, 2),
                      jitter_style: bool = True):
    """Create a violin-style plot showing distribution of search counts by knowledge level."""

    if not data or not data.get('success'):
        print("No valid data to plot")
        return

    stats = data['stats_by_group']
    raw_data = data['raw_data']
    metadata = data['metadata']

    # Define colors for each knowledge level
    colors = {
        'NA': '#9CA3AF',      # Gray
        'limited': '#FBBF24',  # Yellow/Orange
        'moderate': '#3B82F6', # Blue
        'extensive': '#22C55E' # Green
    }

    # Groups to plot (excluding NA as in the original)
    groups = ['limited', 'moderate', 'extensive']
    group_labels = ['Limited', 'Moderate', 'Extensive']

    fig, ax = plt.subplots(figsize=figsize)

    if jitter_style:
        # Create jitter-style violin plot similar to web version
        np.random.seed(42)  # For reproducibility

        # Reduce spacing between groups for compact layout
        x_positions = [0, 0.8, 1.6]  # Closer spacing instead of [0, 1, 2]

        for idx, group in enumerate(groups):
            i = x_positions[idx]
            if group in raw_data and len(raw_data[group]) > 0:
                # Extract data
                group_data = raw_data[group]
                search_counts = np.array([item['search_count'] for item in group_data])

                # Log transform for density calculation if using log scale
                if y_scale == 'log':
                    log_counts = np.log10(np.maximum(search_counts, 1))
                    # Try to use KDE if scipy is available
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(log_counts, bw_method=0.3)
                        # Evaluate density at each point
                        densities = kde(log_counts)
                        densities = densities / np.max(densities)  # Normalize
                    except ImportError:
                        # Fallback to percentile-based density
                        sorted_log = np.sort(log_counts)
                        densities = []
                        for lc in log_counts:
                            percentile = np.searchsorted(sorted_log, lc) / len(sorted_log)
                            densities.append(np.sin(percentile * np.pi))
                        densities = np.array(densities)
                else:
                    # For linear scale
                    sorted_counts = np.sort(search_counts)
                    densities = []
                    for sc in search_counts:
                        percentile = np.searchsorted(sorted_counts, sc) / len(sorted_counts)
                        densities.append(np.sin(percentile * np.pi))
                    densities = np.array(densities)

                # Scale width based on density - reduced for compact layout
                max_width = 0.25  # Reduced from 0.35

                # Create jittered points
                for j, (item, density) in enumerate(zip(group_data, densities)):
                    value = item['search_count']

                    # Width based on density
                    width = density * max_width

                    # Random jitter within the width
                    jitter = (np.random.random() - 0.5) * width * 2

                    # Plot the point
                    ax.scatter(i + jitter, value,
                              color=colors[group], alpha=0.5, s=2,  # Smaller points
                              edgecolor='none', zorder=2)

                # Add median line only
                if group in stats and stats[group]['count'] > 0:
                    median = stats[group]['median']
                    ax.plot([i - 0.2, i + 0.2], [median, median],
                           color='black', linewidth=2, alpha=0.9, zorder=5)
    else:
        # Original violin plot code
        plot_data = []
        positions = []
        group_colors = []

        for i, group in enumerate(groups):
            if group in raw_data and len(raw_data[group]) > 0:
                search_counts = [item['search_count'] for item in raw_data[group]]
                plot_data.append(search_counts)
                positions.append(i)
                group_colors.append(colors[group])

        if not plot_data:
            print("No data available for plotting")
            return

        parts = ax.violinplot(plot_data, positions=positions, widths=0.7,
                              showmeans=False, showmedians=False, showextrema=False)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(group_colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)

        bp = ax.boxplot(plot_data, positions=positions, widths=0.15,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='white', alpha=0.8),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1),
                        capprops=dict(color='black', linewidth=1))

        for i, (group, pos) in enumerate(zip(groups, positions)):
            if group in stats and stats[group]['count'] > 0:
                median = stats[group]['median']
                ax.text(pos, ax.get_ylim()[1] * 0.95, f'n={stats[group]["count"]}',
                       ha='center', va='top', fontsize=10, fontweight='bold')
                ax.hlines(median, pos - 0.35, pos + 0.35, colors='red',
                         linewidth=2, alpha=0.8, linestyles='solid')

    # Customize axes
    if jitter_style:
        ax.set_xticks(x_positions)
    else:
        ax.set_xticks(range(len(groups)))

    # Use abbreviated labels for compact figure
    compact_labels = ['Lim', 'Mod', 'Ext'] if figsize[0] <= 3 else group_labels
    ax.set_xticklabels(compact_labels, fontsize=8)
    ax.set_xlabel('Knowledge Level', fontsize=9)
    ax.set_ylabel('Search Count', fontsize=9)

    # Set y-axis scale
    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(1, 1e6)
        # Set custom y-ticks for cleaner appearance
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 1000000])
        ax.set_yticklabels(['1', '10', '100', '1K', '10K', '100K', '1M'], fontsize=7)
        # Remove minor ticks
        ax.minorticks_off()

    # Add sample sizes below x-axis labels
    positions_to_use = x_positions if jitter_style else range(len(groups))
    for i, group in zip(positions_to_use, groups):
        if group in stats and stats[group]['count'] > 0:
            ax.text(i, ax.get_ylim()[0] * 0.4, f'n={stats[group]["count"]}',
                   ha='center', va='top', fontsize=6, style='italic', color='gray')

    # Add title
    model_name = metadata.get('selected_model', 'All Models')
    if '/' in model_name:
        model_name = model_name.split('/')[-1][:20]  # Shorten long names

    ax.set_title(f'{model_name}',
                fontsize=9, fontweight='bold', pad=3)

    # Add grid
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Smaller tick labels
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=8)

    # Skip legend for very compact figures
    if figsize[0] > 3:
        legend_patches = [mpatches.Patch(color=colors[g], alpha=0.6, label=l)
                         for g, l in zip(groups, compact_labels if figsize[0] <= 3 else group_labels)]
        ax.legend(handles=legend_patches, loc='upper left', frameon=True,
                 fontsize=7, title=None)

    # Add very compact summary statistics
    if figsize[0] > 3:
        stats_text = []
        for group, label in zip(groups, ['L', 'M', 'E']):
            if group in stats and stats[group]['count'] > 0:
                s = stats[group]
                stats_text.append(f"{s['median']:.0f}")

        if stats_text:
            ax.text(0.98, 0.98, 'Med: ' + '/'.join(stats_text), transform=ax.transAxes,
                   fontsize=7, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.2))

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Jitter violin plot saved to {output_path}")
    plt.close()

def create_bar_chart(data: Dict, output_path: str = 'search_count_bar.pdf',
                    y_scale: str = 'log', figsize: Tuple[float, float] = (10, 6)):
    """Create a bar chart showing median search counts by knowledge level."""

    if not data or not data.get('success'):
        print("No valid data to plot")
        return

    stats = data['stats_by_group']
    metadata = data['metadata']

    # Define colors
    colors = {
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    groups = ['limited', 'moderate', 'extensive']
    group_labels = ['Limited', 'Moderate', 'Extensive']

    # Extract median values
    medians = []
    q1_values = []
    q3_values = []
    counts = []
    bar_colors = []

    for group in groups:
        if group in stats and stats[group]['count'] > 0:
            medians.append(stats[group]['median'])
            q1_values.append(stats[group]['q1'])
            q3_values.append(stats[group]['q3'])
            counts.append(stats[group]['count'])
            bar_colors.append(colors[group])
        else:
            medians.append(0)
            q1_values.append(0)
            q3_values.append(0)
            counts.append(0)
            bar_colors.append(colors[group])

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    x_pos = np.arange(len(group_labels))
    bars = ax.bar(x_pos, medians, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add error bars for Q1-Q3 range
    errors = [[medians[i] - q1_values[i] for i in range(len(medians))],
              [q3_values[i] - medians[i] for i in range(len(medians))]]

    ax.errorbar(x_pos, medians, yerr=errors, fmt='none',
               ecolor='black', capsize=5, capthick=2, alpha=0.6)

    # Add value labels on bars
    for i, (bar, median, count) in enumerate(zip(bars, medians, counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                   f'{median:,.0f}\n(n={count})', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    # Customize axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.set_xlabel('Model-Reported Knowledge Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Median Google Scholar Search Count', fontsize=14, fontweight='bold')

    # Set y-axis scale
    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)

    # Add title
    model_name = metadata.get('selected_model', 'All Models')
    if '/' in model_name:
        model_name = model_name.split('/')[-1][:30]

    ax.set_title(f'Median Search Counts by Knowledge Level\n{model_name}',
                fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to {output_path}")
    plt.close()

def create_scatter_plot(data: Dict, output_path: str = 'search_count_scatter.pdf',
                       y_scale: str = 'log', figsize: Tuple[float, float] = (12, 8),
                       show_species_labels: bool = False):
    """Create a scatter plot showing individual species by knowledge level."""

    if not data or not data.get('success'):
        print("No valid data to plot")
        return

    raw_data = data['raw_data']
    stats = data['stats_by_group']
    metadata = data['metadata']

    # Define colors
    colors = {
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    groups = ['limited', 'moderate', 'extensive']
    group_labels = ['Limited', 'Moderate', 'Extensive']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each group
    for i, group in enumerate(groups):
        if group in raw_data and len(raw_data[group]) > 0:
            # Extract data
            species_data = raw_data[group]
            search_counts = [item['search_count'] for item in species_data]

            # Add jitter for x-axis
            x_positions = np.random.normal(i, 0.15, len(search_counts))

            # Plot scatter
            ax.scatter(x_positions, search_counts,
                      color=colors[group], alpha=0.6, s=30,
                      edgecolor='black', linewidth=0.5,
                      label=f'{group_labels[i]} (n={len(search_counts)})')

            # Add median line
            if group in stats and stats[group]['count'] > 0:
                median = stats[group]['median']
                ax.hlines(median, i - 0.3, i + 0.3,
                         colors='red', linewidth=2.5, alpha=0.8)

                # Add quartile box
                q1 = stats[group]['q1']
                q3 = stats[group]['q3']
                rect = Rectangle((i - 0.15, q1), 0.3, q3 - q1,
                               facecolor='none', edgecolor='black',
                               linewidth=1.5, linestyle='--', alpha=0.5)
                ax.add_patch(rect)

    # Customize axes
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.set_xlabel('Model-Reported Knowledge Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Google Scholar Search Count', fontsize=14, fontweight='bold')

    # Set y-axis scale
    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)

    # Add title
    model_name = metadata.get('selected_model', 'All Models')
    if '/' in model_name:
        model_name = model_name.split('/')[-1][:30]

    ax.set_title(f'Individual Species Search Counts by Knowledge Level\n{model_name}',
                fontsize=16, fontweight='bold', pad=20)

    # Add legend
    ax.legend(loc='upper left', frameon=True, fontsize=10)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {output_path}")
    plt.close()

def create_combined_plot(data: Dict, output_path: str = 'search_count_combined.pdf',
                        y_scale: str = 'log'):
    """Create a combined figure with all three plot types."""

    if not data or not data.get('success'):
        print("No valid data to plot")
        return

    fig = plt.figure(figsize=(18, 6))

    # Get model name for title
    metadata = data['metadata']
    model_name = metadata.get('selected_model', 'All Models')
    if '/' in model_name:
        model_name = model_name.split('/')[-1][:30]

    # Main title
    fig.suptitle(f'Google Scholar Search Counts by Model-Reported Knowledge Level\n{model_name}',
                fontsize=18, fontweight='bold', y=1.02)

    # Create three subplots
    # Violin plot
    ax1 = plt.subplot(131)
    plt.sca(ax1)
    create_violin_subplot(data, ax1, y_scale)
    ax1.set_title('Distribution', fontsize=14, fontweight='bold')

    # Bar chart
    ax2 = plt.subplot(132)
    plt.sca(ax2)
    create_bar_subplot(data, ax2, y_scale)
    ax2.set_title('Median Values', fontsize=14, fontweight='bold')

    # Scatter plot
    ax3 = plt.subplot(133)
    plt.sca(ax3)
    create_scatter_subplot(data, ax3, y_scale)
    ax3.set_title('Individual Species', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_path}")
    plt.close()

def create_violin_subplot(data, ax, y_scale='log'):
    """Helper function to create violin plot in subplot."""
    stats = data['stats_by_group']
    raw_data = data['raw_data']

    colors = {
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    groups = ['limited', 'moderate', 'extensive']
    group_labels = ['Limited', 'Moderate', 'Extensive']

    plot_data = []
    positions = []
    group_colors = []

    for i, group in enumerate(groups):
        if group in raw_data and len(raw_data[group]) > 0:
            search_counts = [item['search_count'] for item in raw_data[group]]
            plot_data.append(search_counts)
            positions.append(i)
            group_colors.append(colors[group])

    if plot_data:
        parts = ax.violinplot(plot_data, positions=positions, widths=0.6,
                             showmeans=False, showmedians=True, showextrema=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(group_colors[i])
            pc.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel('Search Count')

    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

def create_bar_subplot(data, ax, y_scale='log'):
    """Helper function to create bar chart in subplot."""
    stats = data['stats_by_group']

    colors = {
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    groups = ['limited', 'moderate', 'extensive']
    group_labels = ['Limited', 'Moderate', 'Extensive']

    medians = []
    bar_colors = []

    for group in groups:
        if group in stats and stats[group]['count'] > 0:
            medians.append(stats[group]['median'])
            bar_colors.append(colors[group])
        else:
            medians.append(0)
            bar_colors.append(colors[group])

    x_pos = np.arange(len(group_labels))
    ax.bar(x_pos, medians, color=bar_colors, alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel('Median Search Count')

    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

def create_scatter_subplot(data, ax, y_scale='log'):
    """Helper function to create scatter plot in subplot."""
    raw_data = data['raw_data']

    colors = {
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    groups = ['limited', 'moderate', 'extensive']
    group_labels = ['Limited', 'Moderate', 'Extensive']

    for i, group in enumerate(groups):
        if group in raw_data and len(raw_data[group]) > 0:
            species_data = raw_data[group]
            search_counts = [item['search_count'] for item in species_data]
            x_positions = np.random.normal(i, 0.1, len(search_counts))

            ax.scatter(x_positions, search_counts,
                      color=colors[group], alpha=0.5, s=20,
                      edgecolor='black', linewidth=0.3)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(group_labels)
    ax.set_ylabel('Search Count')

    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate search count by knowledge level plots'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='Base URL for the running web API')
    parser.add_argument('--model', default=None,
                       help='Specific model to analyze (e.g., "anthropic/claude-3-sonnet")')
    parser.add_argument('--species-file', default='wa_with_gcount.txt',
                       help='Species file to use')
    parser.add_argument('--plot-type', choices=['violin', 'bar', 'scatter', 'combined', 'all'],
                       default='all', help='Type of plot to generate')
    parser.add_argument('--y-scale', choices=['linear', 'log'],
                       default='log', help='Y-axis scale')
    parser.add_argument('--output-prefix', default='search_count',
                       help='Prefix for output files')

    args = parser.parse_args()

    print(f"Fetching search count by knowledge data...")
    print(f"- API URL: {args.api_url}")
    print(f"- Model: {args.model or 'Default (first available)'}")
    print(f"- Species file: {args.species_file}")
    print(f"- Y-axis scale: {args.y_scale}")

    # Fetch data
    data = fetch_search_knowledge_data(args.api_url, args.model, args.species_file)

    if not data or not data.get('success'):
        print("Failed to fetch data or no data available")
        return

    print(f"Data loaded successfully for model: {data['metadata']['selected_model']}")
    print(f"Total species with both search and knowledge data: {data['metadata']['total_species_with_both']}")

    # Generate plots based on type
    if args.plot_type == 'all':
        create_violin_plot(data, f'{args.output_prefix}_violin.pdf', args.y_scale)
        create_bar_chart(data, f'{args.output_prefix}_bar.pdf', args.y_scale)
        create_scatter_plot(data, f'{args.output_prefix}_scatter.pdf', args.y_scale)
        create_combined_plot(data, f'{args.output_prefix}_combined.pdf', args.y_scale)
    elif args.plot_type == 'violin':
        create_violin_plot(data, f'{args.output_prefix}_violin.pdf', args.y_scale)
    elif args.plot_type == 'bar':
        create_bar_chart(data, f'{args.output_prefix}_bar.pdf', args.y_scale)
    elif args.plot_type == 'scatter':
        create_scatter_plot(data, f'{args.output_prefix}_scatter.pdf', args.y_scale)
    elif args.plot_type == 'combined':
        create_combined_plot(data, f'{args.output_prefix}_combined.pdf', args.y_scale)

if __name__ == '__main__':
    main()