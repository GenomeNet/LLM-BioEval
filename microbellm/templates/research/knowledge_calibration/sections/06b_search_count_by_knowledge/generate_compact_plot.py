#!/usr/bin/env python3
"""
Generate a compact publication-ready jitter violin plot for search count by knowledge.
Optimized for small figure sizes (3.5 x 2.5 inches - single column width).
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import urllib.request
import urllib.error
import os

def fetch_data(model='openai/gpt-5-nano'):
    """Fetch search count by knowledge data."""
    url = f'http://localhost:5050/api/search_count_by_knowledge?species_file=wa_with_gcount.txt&model={urllib.parse.quote(model)}'

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error: {e}")
        return None

def create_compact_plot(model='openai/gpt-5-nano', output='compact_violin.pdf'):
    """Create a very compact violin plot suitable for single column publication."""

    data = fetch_data(model)
    if not data or not data.get('success'):
        print("No data available")
        return

    # Extract data
    stats = data['stats_by_group']
    raw_data = data['raw_data']

    # Set up compact figure - single column width
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Colors
    colors = {
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    groups = ['limited', 'moderate', 'extensive']
    labels = ['L', 'M', 'E']  # Abbreviated labels

    np.random.seed(42)

    # Plot each group
    for i, group in enumerate(groups):
        if group in raw_data and len(raw_data[group]) > 0:
            group_data = raw_data[group]
            search_counts = np.array([item['search_count'] for item in group_data])

            # Log transform for better visualization
            log_counts = np.log10(np.maximum(search_counts, 1))
            sorted_log = np.sort(log_counts)

            # Calculate densities
            densities = []
            for lc in log_counts:
                percentile = np.searchsorted(sorted_log, lc) / len(sorted_log)
                densities.append(np.sin(percentile * np.pi))
            densities = np.array(densities)

            # Plot points
            for j, (sc, density) in enumerate(zip(search_counts, densities)):
                width = density * 0.3  # Narrower for compact plot
                jitter = (np.random.random() - 0.5) * width * 2

                ax.scatter(i + jitter, sc,
                          color=colors[group], alpha=0.3, s=1,  # Very small points
                          edgecolor='none', rasterized=True)  # Rasterize for smaller file

            # Add median line
            if group in stats:
                median = stats[group]['median']
                ax.plot([i - 0.2, i + 0.2], [median, median],
                       color='black', linewidth=1.5, alpha=0.9)

                # Add n below
                ax.text(i, 0.5, f'{stats[group]["count"]}',
                       ha='center', va='top', fontsize=6,
                       color='gray', transform=ax.get_xaxis_transform())

    # Formatting
    ax.set_yscale('log')
    ax.set_ylim(1, 1e6)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel('Knowledge Level', fontsize=8)
    ax.set_ylabel('Search Count', fontsize=8)

    # Minimal title
    model_short = model.split('/')[-1].replace('openai/', '').replace('anthropic/', '')[:20]
    ax.set_title(model_short, fontsize=9, pad=5)

    # Tick formatting
    ax.tick_params(axis='both', labelsize=7)

    # Grid
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Very tight layout
    plt.tight_layout(pad=0.5)

    # Save with minimal padding
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Compact plot saved to {output}")
    plt.close()

if __name__ == '__main__':
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else 'openai/gpt-5-nano'
    output = sys.argv[2] if len(sys.argv) > 2 else 'compact_violin.pdf'

    create_compact_plot(model, output)