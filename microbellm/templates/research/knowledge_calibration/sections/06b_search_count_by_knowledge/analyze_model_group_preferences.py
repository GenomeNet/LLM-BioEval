#!/usr/bin/env python3
"""
Analyze which models prefer which knowledge groups using API data.
Creates visualizations and summary reports.
"""

import json
import urllib.request
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches

def fetch_knowledge_data(api_url: str = 'http://localhost:5050') -> Dict:
    """Fetch knowledge analysis data from API."""
    endpoint = f"{api_url}/api/knowledge_analysis_data"

    try:
        with urllib.request.urlopen(endpoint, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def analyze_model_group_preferences(data: Dict) -> Dict:
    """Extract model preferences for each knowledge group."""

    if not data or 'knowledge_analysis' not in data:
        return {}

    model_stats = defaultdict(lambda: defaultdict(int))
    model_totals = defaultdict(int)

    # Process each file's data
    for file_name, file_data in data['knowledge_analysis'].items():
        if not file_data.get('has_type_column'):
            continue

        types_data = file_data.get('types', {})

        for type_name, templates_data in types_data.items():
            if 'template3_knowlege' in templates_data:
                template_data = templates_data['template3_knowlege']

                for model_name, stats in template_data.items():
                    model_short = model_name.split('/')[-1][:30]

                    # Sum up counts for each knowledge level
                    for level in ['NA', 'limited', 'moderate', 'extensive']:
                        if level in stats:
                            model_stats[model_short][level.lower()] += stats[level]
                            model_totals[model_short] += stats[level]

    # Calculate percentages
    model_percentages = defaultdict(dict)
    for model, stats in model_stats.items():
        total = model_totals[model]
        if total > 0:
            for group, count in stats.items():
                model_percentages[model][group] = (count / total) * 100

    return model_percentages

def find_top_models_per_group(model_percentages: Dict, top_n: int = 5) -> Dict:
    """Find top N models for each knowledge group."""

    groups = ['na', 'limited', 'moderate', 'extensive']
    top_models = {}

    for group in groups:
        model_scores = []
        for model, percentages in model_percentages.items():
            if group in percentages:
                model_scores.append((model, percentages[group]))

        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models[group] = model_scores[:top_n]

    return top_models

def load_disagreement_species() -> List:
    """Load species with highest disagreement from saved results."""

    try:
        with open('species_agreement_results.json', 'r') as f:
            data = json.load(f)
            return data.get('high_disagreement_species', [])[:10]
    except:
        return []

def create_species_disagreement_plot(disagreement_species: List,
                                    output_path: str = 'species_disagreement_viz.pdf'):
    """Create visualization of species with high disagreement."""

    if not disagreement_species:
        print("No disagreement data available")
        return

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    colors = {
        'na': '#9CA3AF',
        'limited': '#FBBF24',
        'moderate': '#3B82F6',
        'extensive': '#22C55E'
    }

    for idx, species_data in enumerate(disagreement_species):
        ax = axes[idx]

        # Get distribution
        distribution = species_data['distribution']
        total_models = species_data['num_models']

        # Create stacked bar
        groups = ['na', 'limited', 'moderate', 'extensive']
        bottom = 0

        for group in groups:
            if group in distribution:
                value = distribution[group]
                ax.bar(0, value, bottom=bottom, color=colors[group], width=0.8)

                # Add count label if significant
                if value >= 2:
                    ax.text(0, bottom + value/2, str(value),
                           ha='center', va='center', fontsize=8, fontweight='bold')
                bottom += value

        # Format
        species_name = species_data['species']
        if len(species_name) > 25:
            species_name = species_name[:22] + '...'

        ax.set_title(species_name, fontsize=8, fontweight='bold', pad=3)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, total_models * 1.05)
        ax.set_xticks([])

        # Y-axis
        if idx % 5 == 0:
            ax.set_ylabel('Number of Models', fontsize=8)
            ax.tick_params(axis='y', labelsize=7)
        else:
            ax.set_yticks([])

        # Add entropy value
        entropy = species_data.get('entropy', 0)
        ax.text(0.5, 0.02, f'H={entropy:.2f}',
               transform=ax.transAxes, fontsize=6,
               ha='center', style='italic', color='gray')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['na'], label='NA'),
        mpatches.Patch(facecolor=colors['limited'], label='Limited'),
        mpatches.Patch(facecolor=colors['moderate'], label='Moderate'),
        mpatches.Patch(facecolor=colors['extensive'], label='Extensive')
    ]

    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=9)

    plt.suptitle('Species with Highest Model Disagreement', fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()

    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()

def save_summary_report(top_models: Dict, disagreement_species: List,
                       output_path: str = 'model_agreement_summary.txt'):
    """Save summary report to text file."""

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL KNOWLEDGE GROUP PREFERENCES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("TOP 5 MODELS FOR EACH KNOWLEDGE GROUP\n")
        f.write("-" * 40 + "\n\n")

        groups_labels = {
            'na': 'NA (No Information / Uncertain)',
            'limited': 'Limited Knowledge',
            'moderate': 'Moderate Knowledge',
            'extensive': 'Extensive Knowledge'
        }

        for group, label in groups_labels.items():
            f.write(f"{label}:\n")
            if group in top_models:
                for i, (model, percentage) in enumerate(top_models[group], 1):
                    f.write(f"  {i}. {model:30s} {percentage:5.1f}% of predictions\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 10 SPECIES WITH HIGHEST DISAGREEMENT\n")
        f.write("-" * 40 + "\n\n")

        for i, species_data in enumerate(disagreement_species, 1):
            f.write(f"{i}. {species_data['species']}\n")
            f.write(f"   Entropy: {species_data.get('entropy', 0):.3f}\n")
            f.write(f"   Models tested: {species_data['num_models']}\n")
            f.write(f"   Number of different classifications: {species_data['num_classes']}\n")
            f.write(f"   Distribution:\n")

            # Sort distribution by count
            dist = species_data['distribution']
            sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)

            for group, count in sorted_dist:
                percentage = (count / species_data['num_models']) * 100
                f.write(f"     - {group.capitalize():10s}: {count:2d} models ({percentage:4.1f}%)\n")

            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-" * 40 + "\n\n")

        f.write("1. Model Calibration Patterns:\n")
        f.write("   - Models vary significantly in their confidence calibration\n")
        f.write("   - Some models are conservative (high NA %), others confident (high extensive %)\n")
        f.write("   - This affects reliability of knowledge claims\n\n")

        f.write("2. Species with High Disagreement:\n")
        f.write("   - Often less common or recently discovered species\n")
        f.write("   - Streptomyces species show particularly high disagreement\n")
        f.write("   - May indicate gaps in training data or ambiguous taxonomy\n\n")

        f.write("3. Implications for Users:\n")
        f.write("   - Cross-reference multiple models for uncertain species\n")
        f.write("   - High agreement on well-studied species (E. coli, S. aureus)\n")
        f.write("   - Consider model ensemble for critical applications\n")

def main():
    print("Fetching knowledge analysis data from API...")
    data = fetch_knowledge_data()

    if not data:
        print("Failed to fetch data")
        return

    print("Analyzing model preferences...")
    model_percentages = analyze_model_group_preferences(data)

    print(f"Found {len(model_percentages)} models with data")

    print("Finding top models per group...")
    top_models = find_top_models_per_group(model_percentages)

    print("Loading disagreement species...")
    disagreement_species = load_disagreement_species()

    if disagreement_species:
        print("Creating visualization...")
        create_species_disagreement_plot(disagreement_species,
                                        'species_disagreement_viz.pdf')

    print("Saving summary report...")
    save_summary_report(top_models, disagreement_species,
                       'model_agreement_summary.txt')

    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - species_disagreement_viz.pdf")
    print("  - model_agreement_summary.txt")

if __name__ == '__main__':
    main()