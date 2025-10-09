#!/usr/bin/env python3
"""
Generate a clean single violin plot specifically for GPT-4 with better handling of outliers.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import urllib.request
import urllib.parse

def fetch_data(model='openai/gpt-4'):
    """Fetch search count by knowledge data."""
    url = f'http://localhost:5050/api/search_count_by_knowledge?species_file=wa_with_gcount.txt&model={urllib.parse.quote(model)}'

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error: {e}")
        return None

def create_gpt4_plot(output='gpt4_violin_clean.pdf'):
    """Create a clean violin plot for GPT-4."""

    data = fetch_data('openai/gpt-4')
    if not data or not data.get('success'):
        print("No data available")
        return

    stats = data['stats_by_group']
    raw_data = data['raw_data']

    # Create figure - single panel
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Colors
    colors = {
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    groups = ['limited', 'moderate', 'extensive']
    labels = ['Limited', 'Moderate', 'Extensive']

    # Wider spacing for single plot
    x_positions = [0, 1, 2]

    np.random.seed(42)

    # Plot each group
    for idx, group in enumerate(groups):
        i = x_positions[idx]

        if group in raw_data and len(raw_data[group]) > 0:
            group_data = raw_data[group]
            search_counts = np.array([item['search_count'] for item in group_data])

            # Log transform for density calculation (no capping)
            log_counts = np.log10(np.maximum(search_counts, 1))
            sorted_log = np.sort(log_counts)

            # Calculate densities with better distribution
            densities = []
            for lc in log_counts:
                # Use percentile for density
                percentile = np.searchsorted(sorted_log, lc) / len(sorted_log)
                # Create violin shape
                density = np.sin(percentile * np.pi)
                densities.append(density)
            densities = np.array(densities)

            # Normalize densities
            if np.max(densities) > 0:
                densities = densities / np.max(densities)

            # Scale width
            max_width = 0.35

            # Plot points with varying sizes based on density
            for j, (sc, density) in enumerate(zip(search_counts, densities)):
                width = density * max_width
                jitter = (np.random.random() - 0.5) * width * 2

                # Plot all values without capping
                ax.scatter(i + jitter, sc,
                          color=colors[group], alpha=0.4, s=1.5,
                          edgecolor='none', zorder=2, rasterized=True)

            # Add median line
            if group in stats and stats[group]['count'] > 0:
                median = stats[group]['median']
                ax.plot([i - 0.3, i + 0.3], [median, median],
                       color='black', linewidth=2.5, alpha=0.9, zorder=5)

                # Add sample size below
                ax.text(i, 0.3, f'n={stats[group]["count"]}',
                       ha='center', va='center', fontsize=8,
                       color='#666666', style='italic',
                       transform=ax.get_xaxis_transform())

                # Add median value above
                if median < 1000:
                    med_label = f'{median:.0f}'
                elif median < 1000000:
                    med_label = f'{median/1000:.0f}K'
                else:
                    med_label = f'{median/1000000:.1f}M'

                ax.text(i, median * 1.5, med_label,
                       ha='center', va='bottom', fontsize=7,
                       color='black', fontweight='bold')

    # Formatting
    ax.set_yscale('log')

    # Find the actual maximum value across all groups
    max_val = 1
    for group in groups:
        if group in raw_data and len(raw_data[group]) > 0:
            group_max = max([d['search_count'] for d in raw_data[group]])
            max_val = max(max_val, group_max)

    # Set y-limits to include all data
    ax.set_ylim(1, max_val * 1.5)

    # Set custom y-ticks based on the data range
    if max_val > 10000000:  # More than 10M
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])
        ax.set_yticklabels(['1', '10', '100', '1K', '10K', '100K', '1M', '10M', '100M'], fontsize=8)
    elif max_val > 1000000:  # More than 1M
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
        ax.set_yticklabels(['1', '10', '100', '1K', '10K', '100K', '1M', '10M'], fontsize=8)
    else:
        ax.set_yticks([1, 10, 100, 1000, 10000, 100000, 1000000])
        ax.set_yticklabels(['1', '10', '100', '1K', '10K', '100K', '1M'], fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_xlabel('Model-Reported Knowledge Level', fontsize=10)
    ax.set_ylabel('Google Scholar Search Count', fontsize=10)

    # Title
    ax.set_title('GPT-4', fontsize=11, fontweight='bold', pad=5)

    # Grid - subtle
    ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.3, which='major')
    ax.set_axisbelow(True)

    # Remove minor ticks
    ax.minorticks_off()

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)


    plt.tight_layout(pad=0.5)
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"GPT-4 plot saved to {output}")
    plt.close()

if __name__ == '__main__':
    create_gpt4_plot()