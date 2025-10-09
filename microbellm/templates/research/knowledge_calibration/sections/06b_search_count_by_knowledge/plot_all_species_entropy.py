#!/usr/bin/env python3
"""
Create entropy distribution plot for ALL species with disagreement.
Shows how entropy is distributed across thousands of species.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sqlite3

def get_all_species_entropy(db_path: str = '/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm.db') -> list:
    """Calculate entropy for all species with sufficient model predictions."""

    # First, load valid species from wa_with_gcount.txt
    valid_species = set()
    species_file = '/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/data/wa_with_gcount.txt'

    with open(species_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            if ',' in line:
                species_name = line.split(',')[0].strip()
                valid_species.add(species_name)

    print(f"Loaded {len(valid_species)} valid species from wa_with_gcount.txt")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all predictions
    query = """
    SELECT binomial_name, model, knowledge_group
    FROM processing_results
    WHERE user_template LIKE '%template3_knowlege%'
    AND knowledge_group IS NOT NULL
    AND knowledge_group != ''
    AND status = 'completed'
    """

    cursor.execute(query)
    results = cursor.fetchall()

    # Organize by species - ONLY include valid species
    from collections import defaultdict
    species_predictions = defaultdict(list)

    for species, model, knowledge_group in results:
        if species and model and knowledge_group and species in valid_species:
            species_predictions[species].append(knowledge_group.lower())

    conn.close()

    print(f"Found predictions for {len(species_predictions)} species (filtered to wa_with_gcount.txt)")

    # Calculate entropy for each species
    entropy_data = []

    for species, predictions in species_predictions.items():
        if len(predictions) < 5:  # Need at least 5 models
            continue

        # Count classifications
        counts = Counter(predictions)
        total = len(predictions)

        # Calculate entropy
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p)

        entropy_data.append({
            'species': species,
            'entropy': entropy,
            'num_models': total,
            'num_classes': len(counts)
        })

    return entropy_data

def create_entropy_distribution_plot(entropy_data: list, output: str = 'all_species_entropy_distribution.pdf'):
    """Create a plot showing entropy distribution across all species."""

    # Sort by entropy
    entropy_data.sort(key=lambda x: x['entropy'], reverse=True)

    # Extract entropy values
    entropies = [d['entropy'] for d in entropy_data]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # Top subplot: Bar plot of all species ranked by entropy
    x = range(len(entropies))

    # Create color gradient based on entropy value
    norm = plt.Normalize(vmin=0, vmax=max(entropies))
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    colors = [sm.to_rgba(e) for e in entropies]

    ax1.bar(x, entropies, color=colors, width=1.0, edgecolor='none')

    ax1.set_xlabel('Species Ranked by Entropy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Entropy (H)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Model Disagreement Across {len(entropy_data)} Species',
                  fontsize=13, fontweight='bold', pad=15)

    # Add grid
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add reference lines
    ax1.axhline(y=np.log(2), color='red', linestyle='--', alpha=0.5,
                label='2 classes (max H=0.69)')
    ax1.axhline(y=np.log(3), color='orange', linestyle='--', alpha=0.5,
                label='3 classes (max H=1.10)')
    ax1.axhline(y=np.log(4), color='green', linestyle='--', alpha=0.5,
                label='4 classes (max H=1.39)')

    ax1.legend(loc='upper right', fontsize=9)

    # Set x-axis limits
    ax1.set_xlim(-10, len(entropies) + 10)

    # Add annotations for top species
    for i in range(min(5, len(entropy_data))):
        species_name = entropy_data[i]['species']
        if len(species_name) > 25:
            species_name = species_name[:22] + '...'
        ax1.annotate(species_name,
                    xy=(i, entropies[i]),
                    xytext=(i + 50, entropies[i] + 0.05),
                    fontsize=7,
                    arrowprops=dict(arrowstyle='->', alpha=0.5))

    # Bottom subplot: Histogram of entropy distribution
    ax2.hist(entropies, bins=50, color='#374151', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Entropy (H)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Species', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Entropy Values', fontsize=11, fontweight='bold')

    # Add grid
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add statistics text
    stats_text = f"Total: {len(entropies)} species\n"
    stats_text += f"Mean H: {np.mean(entropies):.3f}\n"
    stats_text += f"Median H: {np.median(entropies):.3f}\n"
    stats_text += f"Max H: {max(entropies):.3f}"

    ax2.text(0.98, 0.95, stats_text,
             transform=ax2.transAxes,
             fontsize=9,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Entropy distribution plot saved to {output}")
    plt.close()

    return entropy_data

def create_simple_ranking_plot(entropy_data: list, output: str = 'species_entropy_ranking_simple.pdf'):
    """Create a vertical plot with entropy on x-axis and species rank on y-axis."""

    # Sort by entropy
    entropy_data.sort(key=lambda x: x['entropy'], reverse=True)

    # Extract entropy values
    entropies = [d['entropy'] for d in entropy_data]

    # Create figure - tall and narrow
    fig, ax = plt.subplots(figsize=(3, 4.7))

    # Create line plot with swapped axes
    y = range(1, len(entropies) + 1)

    # Define entropy thresholds
    max_entropy_2 = np.log(2)
    max_entropy_3 = np.log(3)
    max_entropy_4 = np.log(4)

    # Find where the entropy curve crosses each threshold
    # This tells us the rank at which species transition between categories

    # Find transition points
    rank_0class = len(entropies)  # Where H=0 ends (at bottom)
    rank_2class = len(entropies)  # Default to bottom if no crossing
    rank_3class = len(entropies)
    rank_4class = len(entropies)

    # Find where H>0 starts (working from bottom up)
    for i in range(len(entropies)-1, -1, -1):
        if entropies[i] > 0:
            rank_0class = i + 1
            break

    for i, entropy_val in enumerate(entropies):
        if entropy_val < max_entropy_4 and rank_4class == len(entropies):
            rank_4class = i + 1
        if entropy_val < max_entropy_3 and rank_3class == len(entropies):
            rank_3class = i + 1
        if entropy_val < max_entropy_2 and rank_2class == len(entropies):
            rank_2class = i + 1

    # Create horizontal bands based on where the curve crosses thresholds
    # Red band: species with near-maximum disagreement (4 classes) - at top
    if rank_4class > 1:
        ax.axhspan(1, rank_4class, alpha=0.15, color='#EF4444', zorder=0)

    # Orange band: species with 3-4 class disagreement
    if rank_3class > rank_4class:
        ax.axhspan(rank_4class, rank_3class, alpha=0.15, color='#FB923C', zorder=0)

    # Yellow band: species with 2-3 class disagreement
    if rank_2class > rank_3class:
        ax.axhspan(rank_3class, rank_2class, alpha=0.15, color='#FBBF24', zorder=0)

    # Light blue band: species with 1-2 classes (0 < H < 0.69)
    if rank_0class > rank_2class:
        ax.axhspan(rank_2class, rank_0class, alpha=0.15, color='#3B82F6', zorder=0)

    # Dark blue band: species with perfect agreement (1 class, H=0) - at bottom
    if rank_0class < len(entropies):
        ax.axhspan(rank_0class, len(entropies), alpha=0.25, color='#1E40AF', zorder=0)

    # Draw the main line on top
    ax.plot(entropies, y, color='#1F2937', linewidth=2, alpha=0.9, zorder=10)

    # Add subtle horizontal lines at transitions
    if rank_0class > 1:
        ax.axhline(y=rank_0class, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    ax.axhline(y=rank_2class, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    ax.axhline(y=rank_3class, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    if rank_4class > rank_0class:
        ax.axhline(y=rank_4class, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    ax.set_ylabel('Species Rank', fontsize=12, fontweight='bold')
    ax.set_xlabel('Entropy (H)', fontsize=12, fontweight='bold')
    ax.set_title(f'Entropy Distribution\n{len(entropy_data)} Species',
                 fontsize=13, fontweight='bold', pad=15)

    # No grid for cleaner look
    ax.grid(False)

    # Set y-axis to show actual numbers (inverted so rank 1 is at top)
    ax.set_ylim(len(entropies), 1)
    ax.set_xlim(0, max(entropies) * 1.05)

    # Count species with perfect agreement (entropy = 0)
    n_1class = sum(1 for e in entropies if e == 0)

    # Calculate sample sizes for each band
    n_4class = rank_4class - 1 if rank_4class > 1 else 0
    n_3to4class = rank_3class - rank_4class if rank_3class > rank_4class else 0
    n_2to3class = rank_2class - rank_3class if rank_2class > rank_3class else 0
    n_2class_total = len(entropies) - rank_2class + 1  # Total ≤2 classes
    n_1to2class = n_2class_total - n_1class  # Species with 1-2 classes (excluding perfect agreement)

    # Print for verification
    print(f"\nSpecies distribution by entropy classes:")
    print(f"  1 class only (H=0, perfect agreement): {n_1class} species")
    print(f"  1-2 classes (0<H<{max_entropy_2:.2f}): {n_1to2class:,} species")
    print(f"  2-3 classes ({max_entropy_2:.2f}≤H<{max_entropy_3:.2f}): {n_2to3class:,} species")
    print(f"  3-4 classes ({max_entropy_3:.2f}≤H<{max_entropy_4:.2f}): {n_3to4class} species")
    print(f"  4 classes (H≥{max_entropy_4:.2f}): {n_4class} species")
    print(f"  Total ≤2 classes: {n_2class_total:,} species ({n_1class} perfect + {n_1to2class:,} partial agreement)")

    # Add labels on the right side for each band with sample sizes
    if n_4class > 0:
        ax.text(max(entropies) * 1.02, (1 + rank_4class)/2, f'4 classes\n(n={n_4class})',
                fontsize=8, ha='left', va='center', color='#EF4444', fontweight='bold')

    ax.text(max(entropies) * 1.02, (rank_4class + rank_3class)/2, f'3-4 classes\n(n={n_3to4class})',
            fontsize=8, ha='left', va='center', color='#FB923C', fontweight='bold')
    ax.text(max(entropies) * 1.02, (rank_3class + rank_2class)/2, f'2-3 classes\n(n={n_2to3class:,})',
            fontsize=8, ha='left', va='center', color='#FBBF24', fontweight='bold')
    ax.text(max(entropies) * 1.02, (rank_2class + rank_0class)/2, f'1-2 classes\n(n={n_1to2class})',
            fontsize=8, ha='left', va='center', color='#3B82F6', fontweight='bold')

    if n_1class > 0:
        ax.text(max(entropies) * 1.02, (rank_0class + len(entropies))/2, f'1 class\n(n={n_1class})',
                fontsize=8, ha='left', va='center', color='#1E40AF', fontweight='bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    plt.savefig(output, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Simple entropy ranking saved to {output}")
    plt.close()

def save_entropy_summary(entropy_data: list, output: str = 'all_species_entropy_summary.txt'):
    """Save summary statistics about entropy distribution."""

    entropies = [d['entropy'] for d in entropy_data]

    with open(output, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ENTROPY DISTRIBUTION SUMMARY FOR ALL SPECIES\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total species analyzed: {len(entropy_data):,}\n")
        f.write(f"Mean entropy: {np.mean(entropies):.4f}\n")
        f.write(f"Median entropy: {np.median(entropies):.4f}\n")
        f.write(f"Std deviation: {np.std(entropies):.4f}\n")
        f.write(f"Min entropy: {min(entropies):.4f}\n")
        f.write(f"Max entropy: {max(entropies):.4f}\n\n")

        f.write("Distribution breakdown:\n")
        f.write(f"  H = 0 (perfect agreement): {sum(1 for e in entropies if e == 0):,} species\n")
        f.write(f"  H < 0.5 (high agreement): {sum(1 for e in entropies if e < 0.5):,} species\n")
        f.write(f"  0.5 ≤ H < 1.0 (moderate disagreement): {sum(1 for e in entropies if 0.5 <= e < 1.0):,} species\n")
        f.write(f"  1.0 ≤ H < 1.3 (high disagreement): {sum(1 for e in entropies if 1.0 <= e < 1.3):,} species\n")
        f.write(f"  H ≥ 1.3 (very high disagreement): {sum(1 for e in entropies if e >= 1.3):,} species\n\n")

        # Percentiles
        f.write("Percentiles:\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            f.write(f"  {p}th percentile: {np.percentile(entropies, p):.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("TOP 20 SPECIES BY ENTROPY\n")
        f.write("-"*40 + "\n\n")

        # Sort by entropy
        entropy_data.sort(key=lambda x: x['entropy'], reverse=True)

        for i, data in enumerate(entropy_data[:20], 1):
            f.write(f"{i:2d}. {data['species'][:50]:<50} H={data['entropy']:.4f} (n={data['num_models']})\n")

    print(f"Summary saved to {output}")

def main():
    print("Loading all species predictions from database...")
    print("This may take a moment...")

    # Get entropy for all species
    entropy_data = get_all_species_entropy()

    print(f"Calculated entropy for {len(entropy_data)} species")

    if not entropy_data:
        print("No data found. Make sure the database path is correct.")
        return

    print("Creating visualizations...")

    # Create distribution plot
    create_entropy_distribution_plot(entropy_data, 'all_species_entropy_distribution.pdf')

    # Create simple ranking plot
    create_simple_ranking_plot(entropy_data, 'species_entropy_ranking_simple.pdf')

    # Save summary
    save_entropy_summary(entropy_data, 'all_species_entropy_summary.txt')

    print("\nAll visualizations created successfully!")
    print("Generated files:")
    print("  - all_species_entropy_distribution.pdf (distribution and histogram)")
    print("  - species_entropy_ranking_simple.pdf (simple ranking curve)")
    print("  - all_species_entropy_summary.txt (statistics summary)")

if __name__ == '__main__':
    main()