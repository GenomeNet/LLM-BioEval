#!/usr/bin/env python3
"""
Generate a stacked bar plot showing the distribution of species by model agreement.
Shows how many species fall into each entropy-based category.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def create_entropy_stacked_bar(output_path: str = 'species_entropy_distribution.pdf',
                               tiny: bool = False):
    """Create a stacked bar plot showing species distribution by model agreement."""

    # Data from the entropy analysis
    total_species = 3884

    # Species counts by agreement level
    data = {
        '1 class\n(unanimous)': 91,      # H=0, all models agree
        '2 classes': 967,                 # 0<H<0.69
        '3 classes': 2336,                # 0.69≤H<1.10
        '4 classes': 490                  # 1.10≤H<1.38
    }

    # Calculate percentages
    percentages = {k: (v/total_species)*100 for k, v in data.items()}

    # Colors matching the entropy plot regions
    colors = {
        '1 class\n(unanimous)': '#1E40AF',  # Dark blue - perfect agreement
        '2 classes': '#3B82F6',              # Light blue - good agreement
        '3 classes': '#FBBF24',              # Yellow - moderate disagreement
        '4 classes': '#FB923C'               # Orange - high disagreement
    }

    # Create figure
    if tiny:
        fig, ax = plt.subplots(figsize=(1.2, 1.2))  # 3cm x 3cm
        fontsize_label = 4
        fontsize_title = 5
        fontsize_pct = 4
        bar_width = 0.6
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        fontsize_label = 9
        fontsize_title = 11
        fontsize_pct = 8
        bar_width = 0.4

    # Create stacked bar
    bottom = 0
    for label, count in data.items():
        pct = percentages[label]
        ax.bar(0, pct, bar_width, bottom=bottom,
               color=colors[label], edgecolor='white', linewidth=0.5)

        # Add percentage labels for significant segments
        if pct > 5:  # Only label if > 5%
            y_pos = bottom + pct/2
            if tiny:
                ax.text(0, y_pos, f'{pct:.1f}%',
                       ha='center', va='center', fontsize=fontsize_pct,
                       fontweight='bold', color='white' if label == '1 class\n(unanimous)' else 'black')
            else:
                ax.text(0, y_pos, f'{label}\n{count:,} ({pct:.1f}%)',
                       ha='center', va='center', fontsize=fontsize_pct,
                       fontweight='bold', color='white' if label == '1 class\n(unanimous)' else 'black')

        bottom += pct

    # Formatting
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_ylabel('Percentage of Species', fontsize=fontsize_label)

    if tiny:
        ax.set_title(f'n={total_species:,}', fontsize=fontsize_title, pad=3)
        # Reduce tick label size
        ax.tick_params(axis='y', labelsize=fontsize_label-1)
    else:
        ax.set_title(f'Species Distribution by Model Agreement\n(n={total_species:,} species)',
                    fontsize=fontsize_title, fontweight='bold', pad=10)
        ax.tick_params(axis='y', labelsize=fontsize_label)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add legend if not tiny
    if not tiny:
        legend_elements = [
            mpatches.Patch(facecolor=colors['1 class\n(unanimous)'],
                          label='1 class (unanimous)', edgecolor='white'),
            mpatches.Patch(facecolor=colors['2 classes'],
                          label='2 classes', edgecolor='white'),
            mpatches.Patch(facecolor=colors['3 classes'],
                          label='3 classes', edgecolor='white'),
            mpatches.Patch(facecolor=colors['4 classes'],
                          label='4 classes', edgecolor='white')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                 frameon=False, fontsize=fontsize_label)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Stacked bar plot saved to {output_path}")

    # Print statistics
    print("\nStatistics:")
    print(f"Total species: {total_species:,}")
    for label, count in data.items():
        print(f"  {label}: {count:,} ({percentages[label]:.1f}%)")

    plt.close()

def create_horizontal_stacked_bar(output_path: str = 'species_entropy_distribution_horizontal.pdf'):
    """Create a horizontal stacked bar showing species distribution."""

    # Data
    total_species = 3884
    data = {
        '1 class': 91,
        '2 classes': 967,
        '3 classes': 2336,
        '4 classes': 490
    }

    # Calculate percentages
    percentages = {k: (v/total_species)*100 for k, v in data.items()}

    # Colors
    colors = ['#1E40AF', '#3B82F6', '#FBBF24', '#FB923C']

    # Create figure - wide and short for horizontal bar
    fig, ax = plt.subplots(figsize=(8, 2))

    # Create horizontal stacked bar
    left = 0
    for i, (label, count) in enumerate(data.items()):
        pct = percentages[label]
        ax.barh(0, pct, height=0.5, left=left,
                color=colors[i], edgecolor='white', linewidth=1)

        # Add labels
        if pct > 5:
            x_pos = left + pct/2
            ax.text(x_pos, 0, f'{label}\n{count:,}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=9,
                   fontweight='bold',
                   color='white' if i == 0 else 'black')

        left += pct

    # Formatting
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Percentage of Species', fontsize=10)
    ax.set_title(f'Distribution of {total_species:,} Species by Model Agreement Level',
                fontsize=11, fontweight='bold', pad=10)

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
    print(f"Horizontal stacked bar saved to {output_path}")
    plt.close()

def main():
    """Generate all versions of the entropy distribution plot."""

    print("Generating entropy distribution stacked bar plots...")

    # Regular vertical version
    create_entropy_stacked_bar('species_entropy_distribution.pdf', tiny=False)

    # Tiny version (3cm x 3cm)
    create_entropy_stacked_bar('species_entropy_distribution_tiny.pdf', tiny=True)

    # Horizontal version
    create_horizontal_stacked_bar('species_entropy_distribution_horizontal.pdf')

    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()