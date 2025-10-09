#!/usr/bin/env python3
"""
Analyze model agreement patterns and create visualizations.
Shows which models tend to classify species into each knowledge group.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches

def load_species_predictions(json_path: str = 'species_agreement_results.json') -> Dict:
    """Load species predictions from the JSON results file."""

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Reconstruct species predictions from the saved data
    species_predictions = {}

    # Add high agreement species
    for species_data in data.get('high_agreement_species', []):
        # These have very high agreement, so we'll skip them for diversity
        pass

    # Add high disagreement species
    for species_data in data.get('high_disagreement_species', []):
        # These are the interesting ones with disagreement
        species = species_data['species']
        # We need to reconstruct the model predictions
        # The JSON doesn't have individual model predictions, so we'll generate from distribution
        pass

    # Since the JSON doesn't have complete model predictions, we'll need to load from the original script
    # Let's use the analyze_species_agreement.py script's logic
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import and use the original function
    from analyze_species_agreement import get_species_predictions

    # Get the actual predictions from database
    import os
    db_path = os.path.join(os.path.dirname(__file__), '../../../../../microbellm.db')
    species_predictions = get_species_predictions(db_path)

    return species_predictions

def analyze_model_preferences(species_predictions: Dict) -> Dict:
    """Analyze which models prefer which knowledge groups."""

    model_group_counts = defaultdict(lambda: defaultdict(int))
    model_total_counts = defaultdict(int)

    for species, model_predictions in species_predictions.items():
        for model, knowledge_group in model_predictions.items():
            model_group_counts[model][knowledge_group] += 1
            model_total_counts[model] += 1

    # Calculate percentages
    model_group_percentages = defaultdict(dict)
    for model, group_counts in model_group_counts.items():
        total = model_total_counts[model]
        for group, count in group_counts.items():
            model_group_percentages[model][group] = (count / total) * 100

    return model_group_percentages

def find_top_models_per_group(model_group_percentages: Dict, top_n: int = 5) -> Dict:
    """Find top N models for each knowledge group."""

    groups = ['na', 'limited', 'moderate', 'extensive']
    top_models = {}

    for group in groups:
        # Get all models with their percentage for this group
        model_scores = []
        for model, percentages in model_group_percentages.items():
            if group in percentages:
                model_scores.append((model, percentages[group]))

        # Sort by percentage
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models[group] = model_scores[:top_n]

    return top_models

def find_species_with_uneven_agreement(species_predictions: Dict, n: int = 10) -> List:
    """Find species with most uneven agreement (high entropy but not uniform)."""

    uneven_species = []

    for species, model_predictions in species_predictions.items():
        if len(model_predictions) < 10:  # Need enough models
            continue

        # Count classifications
        classification_counts = Counter(model_predictions.values())
        total_models = len(model_predictions)

        # Skip if only one or two classes
        if len(classification_counts) < 3:
            continue

        # Calculate proportions
        proportions = [count/total_models for count in classification_counts.values()]

        # Calculate entropy
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in proportions)

        # Calculate standard deviation of proportions (for unevenness)
        std_dev = np.std(proportions)

        # We want high entropy but also some unevenness (not uniform)
        # Uniform distribution would have low std_dev
        if entropy > 1.0 and std_dev > 0.05:
            uneven_species.append({
                'species': species,
                'entropy': entropy,
                'std_dev': std_dev,
                'num_models': total_models,
                'distribution': dict(classification_counts),
                'model_predictions': model_predictions
            })

    # Sort by entropy * std_dev (combination of disagreement and unevenness)
    uneven_species.sort(key=lambda x: x['entropy'] * x['std_dev'], reverse=True)

    return uneven_species[:n]

def create_disagreement_visualization(uneven_species: List, output_path: str = 'species_disagreement_viz.pdf'):
    """Create a visualization showing species with uneven agreement."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    colors = {
        'na': '#9CA3AF',
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    for idx, species_data in enumerate(uneven_species[:10]):
        ax = axes[idx]

        # Prepare data for stacked bar
        distribution = species_data['distribution']
        groups = ['na', 'limited', 'moderate', 'extensive']

        values = []
        group_colors = []
        for group in groups:
            if group in distribution:
                values.append(distribution[group])
                group_colors.append(colors[group])
            else:
                values.append(0)
                group_colors.append(colors[group])

        # Create bar
        bottom = 0
        for i, (value, color) in enumerate(zip(values, group_colors)):
            if value > 0:
                ax.bar(0, value, bottom=bottom, color=color, width=0.8)
                # Add text if significant
                if value >= 3:
                    ax.text(0, bottom + value/2, str(value),
                           ha='center', va='center', fontsize=8, fontweight='bold')
                bottom += value

        # Format
        species_name = species_data['species']
        # Truncate long species names
        if len(species_name) > 25:
            species_name = species_name[:22] + '...'

        ax.set_title(species_name, fontsize=8, fontweight='bold', pad=3)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, species_data['num_models'] * 1.05)
        ax.set_xticks([])

        # Only show y-axis on leftmost plots
        if idx % 5 == 0:
            ax.set_ylabel('Number of Models', fontsize=8)
            ax.tick_params(axis='y', labelsize=7)
        else:
            ax.set_yticks([])

        # Add entropy value
        ax.text(0.5, 0.02, f'H={species_data["entropy"]:.2f}',
               transform=ax.transAxes, fontsize=6,
               ha='center', style='italic', color='gray')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['na'], label='NA'),
        mpatches.Patch(facecolor=colors['limited'], label='Limited'),
        mpatches.Patch(facecolor=colors['moderate'], label='Moderate'),
        mpatches.Patch(facecolor=colors['extensive'], label='Extensive')
    ]

    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=9)

    plt.suptitle('Species with Most Uneven Model Agreement', fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()

    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()

def save_summary_report(top_models: Dict, uneven_species: List,
                        output_path: str = 'agreement_analysis_summary.txt'):
    """Save a detailed summary report."""

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL AGREEMENT ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("TOP 5 MODELS FOR EACH KNOWLEDGE GROUP\n")
        f.write("-" * 40 + "\n\n")

        groups_labels = {
            'na': 'NA (No Information)',
            'limited': 'Limited Knowledge',
            'moderate': 'Moderate Knowledge',
            'extensive': 'Extensive Knowledge'
        }

        for group, label in groups_labels.items():
            f.write(f"{label}:\n")
            if group in top_models:
                for i, (model, percentage) in enumerate(top_models[group], 1):
                    f.write(f"  {i}. {model:30s} {percentage:5.1f}%\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 10 SPECIES WITH MOST UNEVEN AGREEMENT\n")
        f.write("-" * 40 + "\n\n")

        for i, species_data in enumerate(uneven_species[:10], 1):
            f.write(f"{i}. {species_data['species']}\n")
            f.write(f"   Entropy: {species_data['entropy']:.3f}\n")
            f.write(f"   Models tested: {species_data['num_models']}\n")
            f.write(f"   Distribution:\n")

            # Sort distribution by count
            sorted_dist = sorted(species_data['distribution'].items(),
                               key=lambda x: x[1], reverse=True)
            for group, count in sorted_dist:
                percentage = (count / species_data['num_models']) * 100
                f.write(f"     - {group.capitalize():10s}: {count:2d} models ({percentage:4.1f}%)\n")

            # Show which models chose what
            f.write(f"   Model breakdown:\n")
            group_models = defaultdict(list)
            for model, classification in species_data['model_predictions'].items():
                group_models[classification].append(model)

            for group in ['extensive', 'moderate', 'limited', 'na']:
                if group in group_models:
                    models = group_models[group]
                    f.write(f"     {group.upper()} ({len(models)} models):\n")
                    # Show first 3 models
                    for model in models[:3]:
                        f.write(f"       - {model}\n")
                    if len(models) > 3:
                        f.write(f"       ... and {len(models) - 3} more\n")

            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION NOTES\n")
        f.write("-" * 40 + "\n\n")

        f.write("1. Model Preferences:\n")
        f.write("   - Some models consistently classify more species as 'NA' (uncertain)\n")
        f.write("   - Others tend toward 'extensive' knowledge claims\n")
        f.write("   - This reveals different calibration strategies across models\n\n")

        f.write("2. Species with Uneven Agreement:\n")
        f.write("   - High entropy indicates disagreement among models\n")
        f.write("   - These species represent edge cases in knowledge classification\n")
        f.write("   - Often involves less common or recently discovered species\n\n")

        f.write("3. Implications:\n")
        f.write("   - Model ensemble approaches could improve reliability\n")
        f.write("   - Species with high disagreement need expert validation\n")
        f.write("   - Model calibration varies significantly across providers\n")

def main():
    print("Loading predictions from database...")
    species_predictions = load_species_predictions()

    print("Analyzing model preferences...")
    model_percentages = analyze_model_preferences(species_predictions)

    print("Finding top models per knowledge group...")
    top_models = find_top_models_per_group(model_percentages)

    print("Finding species with uneven agreement...")
    uneven_species = find_species_with_uneven_agreement(species_predictions)

    print("Creating visualization...")
    create_disagreement_visualization(uneven_species,
                                     'species_disagreement_viz.pdf')

    print("Saving summary report...")
    save_summary_report(top_models, uneven_species,
                       'agreement_analysis_summary.txt')

    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - species_disagreement_viz.pdf")
    print("  - agreement_analysis_summary.txt")

if __name__ == '__main__':
    main()