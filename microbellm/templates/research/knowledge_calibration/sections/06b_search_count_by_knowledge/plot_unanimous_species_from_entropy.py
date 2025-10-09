#!/usr/bin/env python3
"""
Create a figure showing the distribution of knowledge groups for species where all models agree.
These are the 91 species with entropy = 0 (unanimous agreement).
Uses the same data source as plot_all_species_entropy.py
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import math

def load_all_species_predictions():
    """Load all predictions for real species from database."""
    print("Loading species predictions from database...")

    # Connect to database
    db_path = '/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query all predictions for real species
    query = """
    SELECT s.binomial_name, s.model, s.result
    FROM search_count s
    WHERE EXISTS (
        SELECT 1
        FROM ground_truth g
        WHERE g.binomial_name = s.binomial_name
        AND g.dataset = 'wa_with_gcount.txt'
    )
    AND s.result IN ('NA', 'limited', 'moderate', 'extensive')
    """

    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()

    # Organize by species
    species_predictions = defaultdict(list)
    for species_name, model, result in results:
        species_predictions[species_name].append(result.lower())

    return species_predictions

def calculate_entropy(predictions):
    """Calculate Shannon entropy for a set of predictions."""
    if not predictions:
        return 0

    # Count occurrences
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

    species_predictions = load_all_species_predictions()

    unanimous_species = []

    for species_name, predictions in species_predictions.items():
        if len(predictions) < 10:  # Skip species with too few predictions
            continue

        # Calculate entropy
        entropy = calculate_entropy(predictions)

        # If entropy is 0, all models agree
        if entropy < 0.001:  # Small threshold for floating point
            unique_preds = list(set(predictions))
            if len(unique_preds) == 1:
                unanimous_species.append({
                    'species': species_name,
                    'knowledge_group': unique_preds[0],
                    'model_count': len(predictions),
                    'entropy': entropy
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
        fig, ax = plt.subplots(figsize=(3, 2.5))
        fontsize_labels = 7
        fontsize_title = 8
        fontsize_values = 6
        bar_width = 0.7
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        fontsize_labels = 11
        fontsize_title = 13
        fontsize_values = 10
        bar_width = 0.75

    # Create bars
    x_pos = np.arange(len(groups))
    bars = ax.bar(x_pos, counts, bar_width,
                  color=[colors[g] for g in groups],
                  edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        if count > 0:
            # Position text at top of bar
            y_pos = count + max(counts) * 0.02
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
        ax.set_title(f'Knowledge Group Distribution for Species with Unanimous Agreement\n(n={total} species, all models agree)',
                    fontsize=fontsize_title, fontweight='bold')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y-axis limits
    if max(counts) > 0:
        ax.set_ylim(0, max(counts) * 1.2)

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

    # Print some example species for each group
    print("\nExample species for each group:")
    for group in groups:
        examples = [s['species'] for s in unanimous_species if s['knowledge_group'] == group][:3]
        if examples:
            print(f"  {group.capitalize()}:")
            for ex in examples:
                print(f"    - {ex}")

    plt.close()
    return unanimous_species

def create_stacked_horizontal_bar(output_path='unanimous_species_horizontal.pdf'):
    """Create a horizontal stacked bar for manuscript inclusion."""

    unanimous_species = find_unanimous_species()

    if not unanimous_species:
        return

    # Count distribution
    knowledge_counts = Counter([s['knowledge_group'] for s in unanimous_species])
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
            ax.barh(0, pct, height=0.6, left=left, color=color,
                   edgecolor='white', linewidth=1)

            # Add label if segment is large enough
            if pct > 15:
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
    ax.set_xlabel('Percentage (%)', fontsize=10)
    ax.set_title(f'Knowledge Groups for {total} Species with Unanimous Model Agreement',
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
    print(f"Horizontal bar saved to {output_path}")
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
    create_stacked_horizontal_bar('unanimous_species_horizontal.pdf')

    print("\n" + "="*60)
    print("All plots generated successfully!")

if __name__ == '__main__':
    main()