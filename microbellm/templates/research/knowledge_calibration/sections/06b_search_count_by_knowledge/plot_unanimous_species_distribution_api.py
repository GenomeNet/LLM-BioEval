#!/usr/bin/env python3
"""
Create a figure showing the distribution of knowledge groups for species where all models agree.
These are the 91 species with entropy = 0 (unanimous agreement).
Uses API data instead of direct database access.
"""

import json
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import math

def fetch_species_predictions(api_url='http://localhost:5050'):
    """Fetch all species predictions from the API."""
    endpoint = f"{api_url}/api/search_count_data"

    try:
        with urllib.request.urlopen(endpoint, timeout=30) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                return data.get('species_data', {})
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {}

def calculate_entropy(predictions):
    """Calculate Shannon entropy for a set of predictions."""
    if not predictions:
        return 0

    # Count occurrences of each prediction
    counts = Counter(predictions)
    total = sum(counts.values())

    # Calculate entropy
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return entropy

def find_unanimous_species():
    """Find species where all models agree (entropy = 0)."""

    print("Fetching species predictions from API...")
    species_data = fetch_species_predictions()

    if not species_data:
        print("No species data found")
        return []

    unanimous_species = []

    for species_name, predictions in species_data.items():
        if not predictions:
            continue

        # Calculate entropy
        entropy = calculate_entropy(predictions)

        # If entropy is 0 (or very close), all models agree
        if entropy < 0.001:  # Using small threshold for floating point comparison
            # Get the unanimous prediction
            unique_preds = list(set(predictions))
            if len(unique_preds) == 1:
                unanimous_species.append({
                    'species': species_name,
                    'knowledge_group': unique_preds[0].lower(),
                    'model_count': len(predictions)
                })

    return unanimous_species

def create_unanimous_distribution_plot(output_path='unanimous_species_distribution.pdf',
                                      tiny=False):
    """Create a bar plot showing distribution of unanimous species across knowledge groups."""

    unanimous_species = find_unanimous_species()

    if not unanimous_species:
        print("No unanimous species found")
        return []

    print(f"Found {len(unanimous_species)} species with unanimous agreement")

    # Count distribution across knowledge groups
    knowledge_counts = Counter([s['knowledge_group'] for s in unanimous_species])

    # Define order and colors
    groups = ['na', 'limited', 'moderate', 'extensive']
    colors = {
        'na': '#9CA3AF',        # Grey
        'limited': '#FBBF24',   # Yellow
        'moderate': '#3B82F6',  # Blue
        'extensive': '#22C55E'  # Green
    }

    # Prepare data
    counts = [knowledge_counts.get(g, 0) for g in groups]
    total = sum(counts)
    percentages = [(c/total)*100 if total > 0 else 0 for c in counts]

    # Create figure
    if tiny:
        fig, ax = plt.subplots(figsize=(2.5, 2))
        fontsize_labels = 6
        fontsize_title = 7
        fontsize_values = 5
        bar_width = 0.6
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        fontsize_labels = 10
        fontsize_title = 12
        fontsize_values = 9
        bar_width = 0.7

    # Create bars
    x_pos = np.arange(len(groups))
    bars = ax.bar(x_pos, counts, bar_width,
                  color=[colors[g] for g in groups],
                  edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        if count > 0:
            # Position text at top of bar for better visibility
            y_pos = count + max(counts) * 0.02  # Slightly above bar
            ax.text(i, y_pos, f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom',
                   fontsize=fontsize_values, fontweight='bold')

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['NA', 'Limited', 'Moderate', 'Extensive'],
                       fontsize=fontsize_labels)
    ax.set_ylabel('Number of Species', fontsize=fontsize_labels)

    if tiny:
        ax.set_title(f'Unanimous Agreement\n(n={total})',
                    fontsize=fontsize_title, fontweight='bold')
    else:
        ax.set_title(f'Knowledge Groups for Species with Unanimous Agreement\n(n={total} species)',
                    fontsize=fontsize_title, fontweight='bold')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y-axis limits
    if max(counts) > 0:
        ax.set_ylim(0, max(counts) * 1.15)  # Add 15% space at top for labels

    plt.tight_layout()

    # Save
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Print statistics
    print("\nStatistics for unanimous species:")
    for group in groups:
        count = knowledge_counts.get(group, 0)
        pct = (count/total)*100 if total > 0 else 0
        if count > 0:
            print(f"  {group.capitalize():10s}: {count:3d} species ({pct:5.1f}%)")

    plt.close()

    return unanimous_species

def create_pie_chart(output_path='unanimous_species_pie.pdf'):
    """Create a pie chart showing the distribution."""

    unanimous_species = find_unanimous_species()

    if not unanimous_species:
        print("No unanimous species found")
        return

    # Count distribution
    knowledge_counts = Counter([s['knowledge_group'] for s in unanimous_species])

    # Filter out zero counts
    labels = []
    sizes = []
    colors_list = []
    colors_map = {
        'na': '#9CA3AF',
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    for group in ['na', 'limited', 'moderate', 'extensive']:
        if knowledge_counts.get(group, 0) > 0:
            labels.append(group.capitalize())
            sizes.append(knowledge_counts[group])
            colors_list.append(colors_map[group])

    if not sizes:
        print("No data to plot")
        return

    # Create pie chart
    fig, ax = plt.subplots(figsize=(6, 6))

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list,
                                       autopct='%1.1f%%', startangle=90,
                                       pctdistance=0.85)

    # Enhance text
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    # Add counts to labels
    for i, (label, size) in enumerate(zip(labels, sizes)):
        texts[i].set_text(f'{label}\n(n={size})')

    ax.set_title(f'Distribution of {len(unanimous_species)} Species with Unanimous Agreement',
                fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Pie chart saved to {output_path}")
    plt.close()

def main():
    """Generate all versions of the unanimous species distribution plot."""

    print("="*60)
    print("Generating unanimous species distribution plots")
    print("="*60)

    # Regular bar chart
    unanimous_species = create_unanimous_distribution_plot(
        'unanimous_species_distribution.pdf',
        tiny=False
    )

    # Tiny version
    create_unanimous_distribution_plot(
        'unanimous_species_distribution_tiny.pdf',
        tiny=True
    )

    # Pie chart version
    create_pie_chart('unanimous_species_pie.pdf')

    print("\n" + "="*60)
    print("All plots generated successfully!")

if __name__ == '__main__':
    main()