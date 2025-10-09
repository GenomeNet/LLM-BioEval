#!/usr/bin/env python3
"""
Generate stacked bar plot showing how each model classifies the 3,884 real species.
Each bar represents a model, showing the distribution of NA, Limited, Moderate, Extensive.
"""

import json
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def fetch_knowledge_data(api_url: str = 'http://localhost:5050') -> Dict:
    """Fetch knowledge analysis data from the API."""
    endpoint = f"{api_url}/api/knowledge_analysis_data"

    try:
        with urllib.request.urlopen(endpoint, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def process_model_classifications(data: Dict) -> Dict:
    """Process data to get each model's classification distribution for real species."""

    if not data or 'knowledge_analysis' not in data:
        return {}

    model_stats = {}

    # Process wa_with_gcount.txt data (real species)
    for file_name, file_data in data['knowledge_analysis'].items():
        # Look for the real species file
        if 'wa_with_gcount' not in file_name.lower():
            continue

        # Check if it has types structure
        if file_data.get('types'):
            types_data = file_data.get('types', {})

            # Look for data in both unclassified and wa_with_gcount
            for type_name, templates in types_data.items():
                # Process template3_knowlege (the one that allows NA)
                if 'template3_knowlege' in templates:
                    for model_name, stats in templates['template3_knowlege'].items():
                        # Get counts for each category
                        na_count = stats.get('NA', 0)
                        limited = stats.get('limited', 0)
                        moderate = stats.get('moderate', 0)
                        extensive = stats.get('extensive', 0)
                        no_result = stats.get('no_result', 0)
                        inference_failed = stats.get('inference_failed', 0)

                        total = stats.get('total', 0)

                        # Skip if no data
                        if total == 0:
                            continue

                        # Aggregate if model already exists (from different type)
                        if model_name in model_stats:
                            model_stats[model_name]['NA'] += na_count
                            model_stats[model_name]['limited'] += limited
                            model_stats[model_name]['moderate'] += moderate
                            model_stats[model_name]['extensive'] += extensive
                            model_stats[model_name]['no_result'] += no_result
                            model_stats[model_name]['inference_failed'] += inference_failed
                            model_stats[model_name]['total'] += total
                        else:
                            # Store the stats
                            model_stats[model_name] = {
                            'NA': na_count,
                            'limited': limited,
                            'moderate': moderate,
                            'extensive': extensive,
                            'no_result': no_result,
                            'inference_failed': inference_failed,
                            'total': total,
                            # Calculate percentages
                            'NA_pct': (na_count / total * 100) if total > 0 else 0,
                            'limited_pct': (limited / total * 100) if total > 0 else 0,
                            'moderate_pct': (moderate / total * 100) if total > 0 else 0,
                            'extensive_pct': (extensive / total * 100) if total > 0 else 0,
                            'no_result_pct': (no_result / total * 100) if total > 0 else 0,
                            'inference_failed_pct': (inference_failed / total * 100) if total > 0 else 0
                        }

    # Recalculate percentages after aggregation
    for model_name, stats in model_stats.items():
        total = stats['total']
        if total > 0:
            stats['NA_pct'] = (stats['NA'] / total * 100)
            stats['limited_pct'] = (stats['limited'] / total * 100)
            stats['moderate_pct'] = (stats['moderate'] / total * 100)
            stats['extensive_pct'] = (stats['extensive'] / total * 100)
            stats['no_result_pct'] = (stats['no_result'] / total * 100)
            stats['inference_failed_pct'] = (stats['inference_failed'] / total * 100)

    return model_stats

def create_model_classification_stacked_bar(api_url: str = 'http://localhost:5050',
                                           output_path: str = 'model_classifications_real_species.pdf',
                                           top_n: int = 30,
                                           sort_by: str = 'NA',
                                           tiny: bool = False):
    """Create stacked bar plot showing how models classify real species."""

    # Fetch and process data
    print("Fetching data from API...")
    data = fetch_knowledge_data(api_url)

    if not data:
        print("Failed to fetch data")
        return

    print("Processing model classifications...")
    model_stats = process_model_classifications(data)

    if not model_stats:
        print("No model statistics found for real species")
        return

    print(f"Found {len(model_stats)} models with classification data")

    # Sort models based on criteria
    if sort_by == 'NA':
        sorted_models = sorted(model_stats.items(),
                             key=lambda x: x[1]['NA_pct'],
                             reverse=True)
    elif sort_by == 'extensive':
        sorted_models = sorted(model_stats.items(),
                             key=lambda x: x[1]['extensive_pct'],
                             reverse=True)
    elif sort_by == 'moderate':
        sorted_models = sorted(model_stats.items(),
                             key=lambda x: x[1]['moderate_pct'],
                             reverse=True)
    else:
        sorted_models = list(model_stats.items())

    # Take top N models
    if top_n and top_n < len(sorted_models):
        sorted_models = sorted_models[:top_n]

    # Prepare data for plotting
    model_names = []
    na_pcts = []  # This will now include NA + failed
    limited_pcts = []
    moderate_pcts = []
    extensive_pcts = []

    for model_name, stats in sorted_models:
        # Shorten model name for display
        display_name = model_name.split('/')[-1][:25]
        model_names.append(display_name)

        # Combine NA with failed responses (no_result + inference_failed)
        combined_na_pct = stats['NA_pct'] + stats['no_result_pct'] + stats['inference_failed_pct']
        na_pcts.append(combined_na_pct)
        limited_pcts.append(stats['limited_pct'])
        moderate_pcts.append(stats['moderate_pct'])
        extensive_pcts.append(stats['extensive_pct'])

    # Create figure
    if tiny:
        # Adjust height based on number of models for tiny version
        height = min(6, max(3, 0.15 * len(model_names)))
        fig, ax = plt.subplots(figsize=(3, height))
        fontsize_labels = 4
        fontsize_title = 6
        fontsize_values = 3
    else:
        height = min(12, 0.3 * len(model_names))
        fig, ax = plt.subplots(figsize=(10, max(6, height)))
        fontsize_labels = 8
        fontsize_title = 10
        fontsize_values = 6

    # Create horizontal stacked bars
    y_positions = np.arange(len(model_names))
    bar_height = 1.0  # Full height with no gaps

    # Stack the bars
    p1 = ax.barh(y_positions, na_pcts, height=bar_height,
                color='#9CA3AF', label='NA/Failed')

    p2 = ax.barh(y_positions, limited_pcts, height=bar_height, left=na_pcts,
                color='#FBBF24', label='Limited')

    left2 = [na + lim for na, lim in zip(na_pcts, limited_pcts)]
    p3 = ax.barh(y_positions, moderate_pcts, height=bar_height, left=left2,
                color='#3B82F6', label='Moderate')

    left3 = [l2 + mod for l2, mod in zip(left2, moderate_pcts)]
    p4 = ax.barh(y_positions, extensive_pcts, height=bar_height, left=left3,
                color='#22C55E', label='Extensive')

    # Add percentage values on bars (only for segments > 5%)
    if not tiny:
        for i, (na, lim, mod, ext) in enumerate(zip(na_pcts, limited_pcts,
                                                     moderate_pcts, extensive_pcts)):
            # NA percentage
            if na > 5:
                ax.text(na/2, i, f'{na:.0f}%', ha='center', va='center',
                       fontsize=fontsize_values, fontweight='bold')

            # Limited percentage
            if lim > 5:
                ax.text(na + lim/2, i, f'{lim:.0f}%', ha='center', va='center',
                       fontsize=fontsize_values, fontweight='bold')

            # Moderate percentage
            if mod > 5:
                ax.text(na + lim + mod/2, i, f'{mod:.0f}%', ha='center', va='center',
                       fontsize=fontsize_values, fontweight='bold', color='white')

            # Extensive percentage
            if ext > 5:
                ax.text(na + lim + mod + ext/2, i, f'{ext:.0f}%', ha='center', va='center',
                       fontsize=fontsize_values, fontweight='bold', color='white')

    # Add sample size annotation
    if not tiny:
        # Add sample size text in bottom right
        ax.text(95, len(model_names)-0.5, 'n=3,884 species',
               fontsize=fontsize_labels, ha='right', va='bottom',
               style='italic', alpha=0.7)

    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names, fontsize=fontsize_labels)
    ax.set_xlabel('Percentage of Species', fontsize=fontsize_labels)
    ax.set_xlim(0, 100)

    # Title
    if tiny:
        title = f'Model Classifications (n=3,884)'
        if sort_by == 'extensive':
            title += f'\nSorted by Extensive %'
        elif sort_by == 'NA':
            title += f'\nSorted by NA/Failed %'
    else:
        title = f'Model Knowledge Classifications for Real Species'
        title += f'\n(n=3,884 species, sorted by {sort_by.upper()} percentage)'
    ax.set_title(title, fontsize=fontsize_title, fontweight='bold', pad=10)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    if not tiny:
        ax.legend(loc='upper right', frameon=True, fontsize=fontsize_labels)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Invert y-axis to have highest at top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print statistics
    print(f"\nDisplaying top {len(sorted_models)} models (sorted by {sort_by})")
    print("\nTop 5 models by NA percentage:")
    for i, (model, stats) in enumerate(sorted_models[:5]):
        print(f"  {i+1}. {model.split('/')[-1][:30]:30s} - NA: {stats['NA_pct']:.1f}%")

    plt.close()

def main():
    """Generate various versions of the model classification plot."""

    api_url = 'http://localhost:5050'

    print("="*60)
    print("Generating model classification stacked bar plots")
    print("="*60)

    # Version sorted by NA percentage (highest NA at top)
    create_model_classification_stacked_bar(
        api_url=api_url,
        output_path='model_classifications_sorted_NA.pdf',
        top_n=30,
        sort_by='NA',
        tiny=False
    )

    # Version sorted by extensive percentage
    create_model_classification_stacked_bar(
        api_url=api_url,
        output_path='model_classifications_sorted_extensive.pdf',
        top_n=30,
        sort_by='extensive',
        tiny=False
    )

    # Tiny version with all models sorted by NA (same as _all.pdf but tiny style)
    create_model_classification_stacked_bar(
        api_url=api_url,
        output_path='model_classifications_tiny.pdf',
        top_n=None,  # Show all models
        sort_by='NA',  # Same sorting as model_classifications_all.pdf
        tiny=True
    )

    # All models version
    create_model_classification_stacked_bar(
        api_url=api_url,
        output_path='model_classifications_all.pdf',
        top_n=None,
        sort_by='NA',
        tiny=False
    )

    print("\n" + "="*60)
    print("All plots generated successfully!")

if __name__ == '__main__':
    main()