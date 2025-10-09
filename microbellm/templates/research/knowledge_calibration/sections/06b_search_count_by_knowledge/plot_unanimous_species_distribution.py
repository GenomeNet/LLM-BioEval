#!/usr/bin/env python3
"""
Create a figure showing the distribution of knowledge groups for species where all models agree.
These are the 91 species with entropy = 0 (unanimous agreement).
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def get_unanimous_species():
    """Get species where all models agree (entropy = 0)."""

    # Connect to database
    db_path = '/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm.db'
    conn = sqlite3.connect(db_path)

    # Query to get all results for real species
    query = """
    SELECT binomial_name, knowledge_group, model
    FROM results
    WHERE species_file = 'wa_with_gcount.txt'
    AND system_template = 'template3_knowlege'
    AND knowledge_group IS NOT NULL
    AND knowledge_group != ''
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Group by species and check for unanimous agreement
    unanimous_species = []

    for species, group in df.groupby('binomial_name'):
        knowledge_groups = group['knowledge_group'].unique()

        # If all models agree (only one unique knowledge group)
        if len(knowledge_groups) == 1:
            unanimous_species.append({
                'species': species,
                'knowledge_group': knowledge_groups[0],
                'model_count': len(group)
            })

    return unanimous_species

def create_unanimous_distribution_plot(output_path='unanimous_species_distribution.pdf',
                                      tiny=False):
    """Create a bar plot showing distribution of unanimous species across knowledge groups."""

    print("Fetching unanimous species from database...")
    unanimous_species = get_unanimous_species()

    if not unanimous_species:
        print("No unanimous species found")
        return

    print(f"Found {len(unanimous_species)} species with unanimous agreement")

    # Count distribution across knowledge groups
    knowledge_counts = Counter([s['knowledge_group'].lower() for s in unanimous_species])

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
    percentages = [(c/sum(counts))*100 if sum(counts) > 0 else 0 for c in counts]

    # Create figure
    if tiny:
        fig, ax = plt.subplots(figsize=(2, 2))
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
            ax.text(i, count/2, f'{count}\n({pct:.1f}%)',
                   ha='center', va='center',
                   fontsize=fontsize_values, fontweight='bold',
                   color='white' if groups[i] in ['moderate', 'extensive'] else 'black')

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['NA', 'Limited', 'Moderate', 'Extensive'],
                       fontsize=fontsize_labels)
    ax.set_ylabel('Number of Species', fontsize=fontsize_labels)
    ax.set_title(f'Knowledge Groups for Species with Unanimous Agreement\n(n={len(unanimous_species)} species)',
                fontsize=fontsize_title, fontweight='bold')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Print statistics
    print("\nStatistics for unanimous species:")
    total = len(unanimous_species)
    for group in groups:
        count = knowledge_counts.get(group, 0)
        pct = (count/total)*100 if total > 0 else 0
        print(f"  {group.capitalize():10s}: {count:3d} species ({pct:5.1f}%)")

    plt.close()

    return unanimous_species

def create_stacked_bar_version(output_path='unanimous_species_stacked.pdf'):
    """Create a horizontal stacked bar showing the distribution."""

    print("\nCreating stacked bar version...")
    unanimous_species = get_unanimous_species()

    if not unanimous_species:
        print("No unanimous species found")
        return

    # Count distribution
    knowledge_counts = Counter([s['knowledge_group'].lower() for s in unanimous_species])
    total = len(unanimous_species)

    # Define order and colors
    groups = ['na', 'limited', 'moderate', 'extensive']
    colors = ['#9CA3AF', '#FBBF24', '#3B82F6', '#22C55E']

    # Calculate percentages
    percentages = [knowledge_counts.get(g, 0)/total*100 for g in groups]
    counts = [knowledge_counts.get(g, 0) for g in groups]

    # Create figure - horizontal bar
    fig, ax = plt.subplots(figsize=(8, 1.5))

    # Create horizontal stacked bar
    left = 0
    for i, (group, pct, count, color) in enumerate(zip(groups, percentages, counts, colors)):
        if pct > 0:
            ax.barh(0, pct, height=0.5, left=left, color=color,
                   edgecolor='white', linewidth=1)

            # Add label if segment is large enough
            if pct > 10:
                label_text = f'{group.capitalize()}\n{count} ({pct:.1f}%)'
                ax.text(left + pct/2, 0, label_text,
                       ha='center', va='center', fontsize=9,
                       fontweight='bold',
                       color='white' if group in ['moderate', 'extensive'] else 'black')
            elif pct > 5:
                ax.text(left + pct/2, 0, f'{count}',
                       ha='center', va='center', fontsize=8,
                       fontweight='bold',
                       color='white' if group in ['moderate', 'extensive'] else 'black')

            left += pct

    # Formatting
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Percentage of Unanimous Species (%)', fontsize=10)
    ax.set_title(f'Distribution of {total} Species with Unanimous Model Agreement',
                fontsize=11, fontweight='bold')

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Stacked bar saved to {output_path}")
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

    # Horizontal stacked bar
    create_stacked_bar_version('unanimous_species_stacked.pdf')

    print("\n" + "="*60)
    print("All plots generated successfully!")

if __name__ == '__main__':
    main()