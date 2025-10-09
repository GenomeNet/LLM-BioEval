#!/usr/bin/env python3
"""
Create bar plots showing species entropy and knowledge distribution.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def load_agreement_data(json_file: str = 'species_agreement_results.json'):
    """Load species agreement data from JSON."""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_entropy_bar_plot(output: str = 'species_entropy_plot.pdf'):
    """Create a bar plot showing entropy and distribution for disagreement species."""

    # Load data
    data = load_agreement_data()
    disagreement_species = data.get('high_disagreement_species', [])[:15]  # Top 15

    if not disagreement_species:
        print("No disagreement data found")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])

    # Colors for knowledge groups
    colors = {
        'na': '#9CA3AF',
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    # Prepare data
    species_names = []
    entropies = []

    for species in disagreement_species:
        name = species['species']
        # Truncate long names
        if len(name) > 30:
            name = name[:27] + '...'
        species_names.append(name)
        entropies.append(species.get('entropy', 0))

    # Top subplot: Entropy values
    x_pos = np.arange(len(species_names))
    bars = ax1.bar(x_pos, entropies, color='#374151', alpha=0.8)

    # Color bars by entropy value (gradient from low to high)
    norm = plt.Normalize(vmin=min(entropies), vmax=max(entropies))
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)

    for bar, entropy in zip(bars, entropies):
        bar.set_color(sm.to_rgba(entropy))

    ax1.set_ylabel('Entropy (H)', fontsize=10, fontweight='bold')
    ax1.set_title('Model Disagreement: Species Ranked by Entropy', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks([])  # No x-labels on top plot
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add entropy values on bars
    for i, (bar, entropy) in enumerate(zip(bars, entropies)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{entropy:.2f}', ha='center', va='bottom', fontsize=7)

    # Bottom subplot: Stacked distribution bars
    bottom_na = np.zeros(len(disagreement_species))
    bottom_limited = np.zeros(len(disagreement_species))
    bottom_moderate = np.zeros(len(disagreement_species))
    bottom_extensive = np.zeros(len(disagreement_species))

    # Extract percentages for each category
    na_vals = []
    limited_vals = []
    moderate_vals = []
    extensive_vals = []

    for species in disagreement_species:
        dist = species['distribution']
        total = species['num_models']

        na_vals.append(dist.get('na', 0) / total * 100)
        limited_vals.append(dist.get('limited', 0) / total * 100)
        moderate_vals.append(dist.get('moderate', 0) / total * 100)
        extensive_vals.append(dist.get('extensive', 0) / total * 100)

    # Create stacked bars
    bars_na = ax2.bar(x_pos, na_vals, color=colors['na'], label='NA')
    bars_limited = ax2.bar(x_pos, limited_vals, bottom=na_vals,
                           color=colors['limited'], label='Limited')

    bottom_mod = [na + lim for na, lim in zip(na_vals, limited_vals)]
    bars_moderate = ax2.bar(x_pos, moderate_vals, bottom=bottom_mod,
                            color=colors['moderate'], label='Moderate')

    bottom_ext = [bm + mod for bm, mod in zip(bottom_mod, moderate_vals)]
    bars_extensive = ax2.bar(x_pos, extensive_vals, bottom=bottom_ext,
                             color=colors['extensive'], label='Extensive')

    # Add model counts in the middle of each bar
    for i, species in enumerate(disagreement_species):
        ax2.text(i, 50, f"n={species['num_models']}",
                ha='center', va='center', fontsize=7,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax2.set_ylabel('Knowledge Classification (%)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Species', fontsize=10, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(species_names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add legend
    ax2.legend(loc='upper left', frameon=True, fontsize=9, ncol=4,
              bbox_to_anchor=(0, 1), borderaxespad=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Entropy plot saved to {output}")
    plt.close()

def create_simple_entropy_ranking(output: str = 'species_entropy_ranking.pdf'):
    """Create a simple horizontal bar chart showing entropy ranking."""

    # Load data
    data = load_agreement_data()
    disagreement_species = data.get('high_disagreement_species', [])[:20]  # Top 20

    if not disagreement_species:
        print("No disagreement data found")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))

    # Prepare data (reverse for horizontal bars to show highest at top)
    species_names = []
    entropies = []

    for species in reversed(disagreement_species):
        name = species['species']
        # Truncate long names
        if len(name) > 40:
            name = name[:37] + '...'
        species_names.append(name)
        entropies.append(species.get('entropy', 0))

    # Create horizontal bars
    y_pos = np.arange(len(species_names))

    # Color gradient
    norm = plt.Normalize(vmin=min(entropies), vmax=max(entropies))
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)  # Red (high) to Blue (low)
    colors_list = [sm.to_rgba(e) for e in entropies]

    bars = ax.barh(y_pos, entropies, color=colors_list, alpha=0.8)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(species_names, fontsize=9)
    ax.set_xlabel('Entropy (H)', fontsize=11, fontweight='bold')
    ax.set_title('Species Ranked by Model Disagreement (Entropy)',
                fontsize=12, fontweight='bold', pad=15)

    # Add value labels
    for i, (bar, entropy) in enumerate(zip(bars, entropies)):
        # Get the corresponding species data for model count
        species_idx = len(disagreement_species) - 1 - i
        n_models = disagreement_species[species_idx]['num_models']

        # Add entropy value at end of bar
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{entropy:.3f}', ha='left', va='center', fontsize=8)

        # Add model count inside bar
        ax.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height()/2,
               f'n={n_models}', ha='center', va='center', fontsize=7,
               color='white', fontweight='bold')

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set x-axis limits
    ax.set_xlim(0, max(entropies) * 1.1)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Simple entropy ranking saved to {output}")
    plt.close()

def create_distribution_matrix(output: str = 'species_distribution_matrix.pdf'):
    """Create a compact matrix/heatmap showing distribution across species."""

    # Load data
    data = load_agreement_data()
    disagreement_species = data.get('high_disagreement_species', [])[:7]  # Reduced to 7 for ultra-compact height

    if not disagreement_species:
        print("No disagreement data found")
        return

    # Create figure - smaller for compact display
    fig, ax = plt.subplots(figsize=(2.0, 2.0))

    # Prepare matrix data
    species_names = []
    matrix_data = []

    groups = ['na', 'limited', 'moderate', 'extensive']

    for species in disagreement_species:
        name = species['species']
        # Keep full species name - no truncation
        species_names.append(name)

        dist = species['distribution']
        total = species['num_models']

        row = []
        for group in groups:
            row.append(dist.get(group, 0) / total * 100)
        matrix_data.append(row)

    # Create heatmap with square cells (aspect='equal' makes cells square)
    # Use interpolation='nearest' to ensure proper vector rendering
    im = ax.imshow(matrix_data, cmap='YlOrRd', aspect='equal', vmin=0, vmax=40,
                   interpolation='nearest', rasterized=False)

    # Set ticks
    ax.set_xticks(np.arange(len(groups)))
    ax.set_yticks(np.arange(len(species_names)))
    ax.set_xticklabels(['NA', 'Lim', 'Mod', 'Ext'], fontsize=5, fontweight='bold')
    ax.set_yticklabels(species_names, fontsize=3.5)

    # Add text annotations with smaller font
    for i in range(len(species_names)):
        for j in range(len(groups)):
            value = matrix_data[i][j]
            if value > 0:
                text = ax.text(j, i, f'{value:.0f}',
                             ha="center", va="center",
                             color="white" if value > 25 else "black",
                             fontsize=4, fontweight='bold')

    # Add compact colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('%', fontsize=5)
    cbar.ax.tick_params(labelsize=4)

    ax.set_title('Model Disagreement',
                fontsize=6, fontweight='bold', pad=1)
    ax.set_xlabel('Knowledge Level', fontsize=5, fontweight='bold', labelpad=1)
    ax.set_ylabel('Species', fontsize=5, fontweight='bold', labelpad=1)

    # No grid for cleaner look
    # Remove all ticks
    ax.tick_params(which='both', size=0)

    plt.tight_layout(pad=0.1)

    # Save - ensure vector format for Illustrator compatibility
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.01,
                backend='pdf', metadata={'Creator': 'matplotlib'})
    print(f"Distribution matrix saved to {output}")
    plt.close()

def main():
    print("Creating entropy and distribution visualizations...")

    # Create different visualization styles
    create_entropy_bar_plot('species_entropy_plot.pdf')
    create_simple_entropy_ranking('species_entropy_ranking.pdf')
    create_distribution_matrix('species_distribution_matrix.pdf')

    print("\nAll plots created successfully!")
    print("Generated files:")
    print("  - species_entropy_plot.pdf (stacked bar with entropy)")
    print("  - species_entropy_ranking.pdf (simple horizontal ranking)")
    print("  - species_distribution_matrix.pdf (heatmap matrix)")

if __name__ == '__main__':
    main()