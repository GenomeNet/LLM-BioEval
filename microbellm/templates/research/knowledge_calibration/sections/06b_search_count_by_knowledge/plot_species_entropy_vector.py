#!/usr/bin/env python3
"""
Create vector-compatible distribution matrix for Adobe Illustrator.
Uses explicit rectangles instead of imshow for better compatibility.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

def load_agreement_data(json_file: str = 'species_agreement_results.json'):
    """Load species agreement data from JSON."""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_distribution_matrix_vector(output: str = 'species_distribution_matrix_vector.pdf'):
    """Create a vector-compatible matrix using rectangles."""

    # Load data
    data = load_agreement_data()
    disagreement_species = data.get('high_disagreement_species', [])[:7]

    if not disagreement_species:
        print("No disagreement data found")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(2.0, 2.0))

    # Prepare data
    species_names = []
    matrix_data = []
    groups = ['na', 'limited', 'moderate', 'extensive']

    for species in disagreement_species:
        species_names.append(species['species'])
        dist = species['distribution']
        total = species['num_models']
        row = []
        for group in groups:
            row.append(dist.get(group, 0) / total * 100)
        matrix_data.append(row)

    # Create colormap
    cmap = cm.get_cmap('YlOrRd')
    norm = plt.Normalize(vmin=0, vmax=40)

    # Draw rectangles for each cell
    for i in range(len(species_names)):
        for j in range(len(groups)):
            value = matrix_data[i][j]
            color = cmap(norm(value))

            # Create rectangle with explicit color
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                facecolor=color,
                                edgecolor='none',
                                linewidth=0)
            ax.add_patch(rect)

            # Add text
            if value > 0:
                text_color = "white" if value > 25 else "black"
                ax.text(j, i, f'{value:.0f}',
                       ha="center", va="center",
                       color=text_color,
                       fontsize=4, fontweight='bold')

    # Set limits
    ax.set_xlim(-0.5, len(groups)-0.5)
    ax.set_ylim(len(species_names)-0.5, -0.5)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(groups)))
    ax.set_yticks(np.arange(len(species_names)))
    ax.set_xticklabels(['NA', 'Lim', 'Mod', 'Ext'], fontsize=5, fontweight='bold')
    ax.set_yticklabels(species_names, fontsize=3.5)

    # Labels
    ax.set_title('Model Disagreement', fontsize=6, fontweight='bold', pad=1)
    ax.set_xlabel('Knowledge Level', fontsize=5, fontweight='bold', labelpad=1)
    ax.set_ylabel('Species', fontsize=5, fontweight='bold', labelpad=1)

    # Remove ticks and spines
    ax.tick_params(which='both', size=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add colorbar manually with explicit colors
    # Create a separate axis for colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])

    # Create colorbar using explicit patches
    n_colors = 20
    for i in range(n_colors):
        val = i * 40 / n_colors
        color = cmap(norm(val))
        rect = plt.Rectangle((0, i/n_colors), 1, 1/n_colors,
                            facecolor=color, edgecolor='none')
        cbar_ax.add_patch(rect)

    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 1)
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([0, 0.5, 1])
    cbar_ax.set_yticklabels(['0', '20', '40'], fontsize=4)
    cbar_ax.set_ylabel('%', fontsize=5)
    cbar_ax.yaxis.set_label_position('right')
    cbar_ax.yaxis.tick_right()

    for spine in cbar_ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.1)

    # Save as vector PDF
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Vector-compatible matrix saved to {output}")
    plt.close()

def main():
    print("Creating vector-compatible distribution matrix...")
    create_distribution_matrix_vector('species_distribution_matrix_vector.pdf')
    print("Done!")

if __name__ == '__main__':
    main()