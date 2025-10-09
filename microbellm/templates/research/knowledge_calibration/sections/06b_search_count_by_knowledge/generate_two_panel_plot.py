#!/usr/bin/env python3
"""
Generate a two-panel publication-ready jitter violin plot for search count by knowledge.
Shows two models stacked vertically for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import urllib.request
import urllib.error
import urllib.parse
import os

def fetch_data(model):
    """Fetch search count by knowledge data for a specific model."""
    base_url = 'http://localhost:5050'
    url = f'{base_url}/api/search_count_by_knowledge?species_file=wa_with_gcount.txt&model={urllib.parse.quote(model)}'

    print(f"Fetching data for {model}...")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching data for {model}: {e}")
        return None

def create_violin_subplot(ax, data, x_positions, model_label):
    """Create a jitter violin plot in a subplot."""

    if not data or not data.get('success'):
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return

    stats = data['stats_by_group']
    raw_data = data['raw_data']

    # Colors - using matplotlib colors to ensure compatibility
    import matplotlib.colors as mcolors
    colors = {
        'limited': mcolors.to_rgba('#FBBF24', alpha=0.6),   # Yellow/Orange
        'moderate': mcolors.to_rgba('#3B82F6', alpha=0.6),  # Blue
        'extensive': mcolors.to_rgba('#22C55E', alpha=0.6)  # Green
    }

    groups = ['limited', 'moderate', 'extensive']
    labels = ['Lim', 'Mod', 'Ext']

    np.random.seed(42)

    # Plot each group
    for idx, group in enumerate(groups):
        i = x_positions[idx]

        if group in raw_data and len(raw_data[group]) > 0:
            group_data = raw_data[group]
            search_counts = np.array([item['search_count'] for item in group_data])

            # Log transform for density calculation
            log_counts = np.log10(np.maximum(search_counts, 1))
            sorted_log = np.sort(log_counts)

            # Calculate densities
            densities = []
            for lc in log_counts:
                percentile = np.searchsorted(sorted_log, lc) / len(sorted_log)
                densities.append(np.sin(percentile * np.pi))
            densities = np.array(densities)

            # Scale width - reduced for tighter spacing
            max_width = 0.18  # Reduced from 0.25

            # Create arrays for all points in this group
            x_positions_group = []
            y_positions_group = []

            for j, (item, density) in enumerate(zip(group_data, densities)):
                value = item['search_count']
                width = density * max_width
                jitter = (np.random.random() - 0.5) * width * 2

                x_positions_group.append(i + jitter)
                y_positions_group.append(value)

            # Plot all points for this group at once
            ax.scatter(x_positions_group, y_positions_group,
                      c=[colors[group]] * len(x_positions_group),
                      s=1.5, edgecolor='none', zorder=2, rasterized=False)

            # Add median line only - shorter for compact layout
            if group in stats and stats[group]['count'] > 0:
                median = stats[group]['median']
                ax.plot([i - 0.15, i + 0.15], [median, median],  # Shorter lines
                       color='black', linewidth=1.8, alpha=0.9, zorder=5)

                # Add sample size
                ax.text(i, 0.5, f'{stats[group]["count"]}',
                       ha='center', va='center', fontsize=6,
                       color='gray', style='italic',
                       transform=ax.get_xaxis_transform())

    # Formatting
    ax.set_yscale('log')

    # Find the actual maximum value across all groups for this subplot's data
    max_val = 1
    for group in groups:
        if group in raw_data and len(raw_data[group]) > 0:
            group_max = max([d['search_count'] for d in raw_data[group]])
            max_val = max(max_val, group_max)

    # Set y-limits to include all data (will be shared across both panels)
    # We'll set this after creating both subplots

    # Set custom y-ticks based on the data range
    if max_val > 10000000:  # More than 10M
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])
        ax.set_yticklabels(['1', '10', '100', '1K', '10K', '100K', '1M', '10M', '100M'], fontsize=7)
    elif max_val > 1000000:  # More than 1M
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
        ax.set_yticklabels(['1', '10', '100', '1K', '10K', '100K', '1M', '10M'], fontsize=7)
    else:
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 1000000])
        ax.set_yticklabels(['1', '10', '100', '1K', '10K', '100K', '1M'], fontsize=7)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=7)  # Smaller font

    # Y-axis label - smaller font
    ax.set_ylabel('Search Count', fontsize=8)

    # Title - smaller and closer
    ax.set_title(model_label, fontsize=8, fontweight='bold', pad=2)

    # Grid - only horizontal lines at major ticks
    ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.3, which='major')
    ax.set_axisbelow(True)

    # Remove minor ticks
    ax.minorticks_off()

    # Tick sizes
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=8)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_two_panel_plot(model1='openai/gpt-4', model2='openai/gpt-4o-mini',
                         output='two_panel_violin.pdf'):
    """Create a two-panel figure with two models for comparison."""

    # Fetch data for both models
    data1 = fetch_data(model1)
    data2 = fetch_data(model2)

    # Find the maximum value across both datasets
    max_val = 1
    groups = ['limited', 'moderate', 'extensive']

    for data in [data1, data2]:
        if data and data.get('success'):
            raw_data = data.get('raw_data', {})
            for group in groups:
                if group in raw_data:
                    for item in raw_data[group]:
                        max_val = max(max_val, item['search_count'])

    # Create figure with 2 subplots
    # Reduced width by 30% (from 3 to 2.1 inches)
    # Height remains 4 inches to accommodate two plots
    # Don't share y-axis so we can set limits after
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.1, 4), sharex=True)

    # Set y-limits for both to include all data
    ax1.set_ylim(1, max_val * 1.5)
    ax2.set_ylim(1, max_val * 1.5)

    # Even more reduced spacing between groups for very compact layout
    x_positions = [0, 0.5, 1.0]  # Reduced from [0, 0.8, 1.6]

    # Create first subplot
    model1_label = model1.split('/')[-1][:20]
    create_violin_subplot(ax1, data1, x_positions, model1_label)

    # Create second subplot
    model2_label = model2.split('/')[-1][:20]
    create_violin_subplot(ax2, data2, x_positions, model2_label)

    # Common x-axis label - smaller
    ax2.set_xlabel('Knowledge Level', fontsize=8)

    # Remove x-axis labels from top plot
    ax1.set_xlabel('')

    # Adjust spacing between subplots - tighter
    plt.subplots_adjust(hspace=0.2)

    # Tight layout with minimal padding
    plt.tight_layout(pad=0.3)

    # Save
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Two-panel plot saved to {output}")
    plt.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate two-panel search count violin plots'
    )
    parser.add_argument('--model1', default='openai/gpt-4',
                       help='First model to plot')
    parser.add_argument('--model2', default='openai/gpt-4o-mini',
                       help='Second model to plot')
    parser.add_argument('--output', default='two_panel_violin.pdf',
                       help='Output file path')

    args = parser.parse_args()

    print(f"Creating two-panel plot:")
    print(f"  Top panel: {args.model1}")
    print(f"  Bottom panel: {args.model2}")

    create_two_panel_plot(args.model1, args.model2, args.output)

if __name__ == '__main__':
    main()