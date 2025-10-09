#!/usr/bin/env python3
"""
Generate confusion matrices for all best model-phenotype combinations in a single PDF.
"""

import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib
from typing import List, Dict, Optional
import seaborn as sns
from collections import defaultdict

# Set font type for PDF editing
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans',
                                          'Bitstream Vera Sans', 'sans-serif']


def fetch_api_data(endpoint: str, api_url: str = 'http://localhost:5050') -> Dict:
    """Fetch data from API endpoint."""
    full_url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urllib.request.urlopen(full_url, timeout=60) as resp:
        return json.loads(resp.read().decode('utf-8'))


def normalize_value(value) -> Optional[str]:
    """Normalize values with deterministic handling."""
    missing_tokens = ['n/a', 'na', 'null', 'none', 'nan', 'undefined', '-', 'unknown', 'missing']

    if value is None or value == '':
        return None

    str_value = str(value).strip().lower()

    if str_value in missing_tokens:
        return None

    if ',' in str_value or ';' in str_value:
        parts = [s.strip() for s in str_value.replace(';', ',').split(',') if s.strip()]
        return ','.join(sorted(parts))

    return str_value


def create_confusion_matrix_plot(true_vals, pred_vals, phenotype, model, ax, vmax=None):
    """Create a single confusion matrix plot with consistent color scale."""

    # Get unique labels
    labels = sorted(list(set(true_vals + pred_vals)))

    # Create display labels based on phenotype
    label_display = {}

    if phenotype == 'spore_formation':
        label_display = {
            'yes': 'Spore-forming',
            'no': 'Non-spore-forming',
            'true': 'Spore-forming',
            'false': 'Non-spore-forming',
            '1': 'Spore-forming',
            '0': 'Non-spore-forming'
        }
    elif phenotype == 'cell_shape':
        label_display = {
            'bacillus': 'Bacillus',
            'coccus': 'Coccus',
            'spirillum': 'Spirillum',
            'coccobacillus': 'Coccobacillus',
            'filamentous': 'Filamentous',
            'pleomorphic': 'Pleomorphic',
            'spiral': 'Spiral',
            'vibrio': 'Vibrio',
            'rod': 'Rod/Bacillus'
        }
    elif phenotype == 'biosafety_level':
        label_display = {
            '1': 'BSL-1',
            '2': 'BSL-2',
            '3': 'BSL-3',
            '4': 'BSL-4',
            'bsl-1': 'BSL-1',
            'bsl-2': 'BSL-2',
            'bsl-3': 'BSL-3',
            'bsl-4': 'BSL-4'
        }
    elif phenotype == 'gram_staining':
        label_display = {
            'positive': 'Gram-positive',
            'negative': 'Gram-negative',
            'variable': 'Variable',
            'gram-positive': 'Gram-positive',
            'gram-negative': 'Gram-negative',
            '+': 'Gram-positive',
            '-': 'Gram-negative'
        }
    elif phenotype in ['motility', 'animal_pathogenicity', 'plant_pathogenicity',
                       'host_association', 'biofilm_formation']:
        label_display = {
            'yes': 'Yes',
            'no': 'No',
            'true': 'Yes',
            'false': 'No',
            '1': 'Yes',
            '0': 'No',
            'positive': 'Yes',
            'negative': 'No',
            'motile': 'Motile',
            'non-motile': 'Non-motile',
            'immotile': 'Non-motile',
            'pathogenic': 'Pathogenic',
            'non-pathogenic': 'Non-pathogenic'
        }
    elif phenotype == 'extreme_environment_tolerance':
        label_display = {
            'mesophile': 'Mesophile',
            'thermophile': 'Thermophile',
            'psychrophile': 'Psychrophile',
            'halophile': 'Halophile',
            'acidophile': 'Acidophile',
            'alkaliphile': 'Alkaliphile',
            'yes': 'Tolerant',
            'no': 'Non-tolerant'
        }

    # Create confusion matrix
    n_classes = len(labels)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(true_vals, pred_vals):
        true_idx = labels.index(true)
        pred_idx = labels.index(pred)
        conf_matrix[true_idx, pred_idx] += 1

    # Calculate metrics
    total_correct = np.trace(conf_matrix)
    total_samples = conf_matrix.sum()
    overall_acc = total_correct / total_samples if total_samples > 0 else 0

    # Calculate balanced accuracy
    class_accuracies = []
    for i in range(n_classes):
        row_sum = conf_matrix[i, :].sum()
        if row_sum > 0:
            class_accuracies.append(conf_matrix[i, i] / row_sum)
    balanced_acc = np.mean(class_accuracies) if class_accuracies else 0

    # Create heatmap
    display_labels = [label_display.get(l, l.title()) for l in labels]

    # Use grey to green colormap with consistent scale
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='Greens',  # Grey to green gradient
                vmin=0,  # Always start at 0
                vmax=vmax,  # Use consistent maximum across all plots
                xticklabels=display_labels,
                yticklabels=display_labels,
                cbar_kws={'shrink': 0.8},
                linewidths=0.5,
                linecolor='white',  # White lines for better contrast
                square=True,
                ax=ax,
                annot_kws={'fontsize': 8})

    # Format title and labels
    phenotype_title = phenotype.replace('_', ' ').title()
    model_short = model.split('/')[-1] if '/' in model else model

    ax.set_title(f'{phenotype_title}\nModel: {model_short}',
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Add metrics text
    metrics_text = f'Acc: {overall_acc:.1%}\nBal: {balanced_acc:.1%}\nN={total_samples}'
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    return overall_acc, balanced_acc, total_samples


def main():
    """Generate confusion matrices for all best model-phenotype pairs."""

    print("Fetching data from API...")

    # Fetch prediction data
    pred_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt')
    predictions = pred_data.get('data', [])

    # Fetch ground truth data
    gt_data = fetch_api_data('/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000')

    # Create ground truth mapping
    gt_map = {}
    for item in gt_data.get('data', []):
        gt_map[item['binomial_name'].lower()] = item

    print(f"Found {len(predictions)} predictions")
    print(f"Found {len(gt_map)} ground truth entries")

    # Best model-phenotype pairs from the balanced accuracy analysis
    best_pairs = [
        ('spore_formation', 'google/gemini-2.5-pro'),
        ('cell_shape', 'openai/gpt-4.1-nano'),
        ('biosafety_level', 'anthropic/claude-3.5-sonnet'),
        ('motility', 'openai/gpt-5'),
        ('animal_pathogenicity', 'google/gemini-flash-1.5'),
        ('host_association', 'openai/gpt-4o'),
        ('plant_pathogenicity', 'google/gemini-pro-1.5'),
        ('extreme_environment_tolerance', 'deepseek/deepseek-r1'),
        ('gram_staining', 'google/gemini-2.5-pro'),
        ('biofilm_formation', 'openai/gpt-4')
    ]

    # First pass: collect all data and find maximum value for consistent color scale
    print("Calculating consistent color scale...")
    all_confusion_data = []
    max_value = 0

    for phenotype, model in best_pairs:
        model_preds = [p for p in predictions if p['model'] == model]
        true_vals = []
        pred_vals = []

        for pred in model_preds:
            species = pred.get('binomial_name', '').lower()
            if species and species in gt_map:
                t = normalize_value(gt_map[species].get(phenotype))
                y = normalize_value(pred.get(phenotype))

                if t is not None and y is not None:
                    true_vals.append(t)
                    pred_vals.append(y)

        if len(true_vals) > 0:
            # Calculate confusion matrix to find max value
            labels = sorted(list(set(true_vals + pred_vals)))
            n_classes = len(labels)
            conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

            for true, pred in zip(true_vals, pred_vals):
                true_idx = labels.index(true)
                pred_idx = labels.index(pred)
                conf_matrix[true_idx, pred_idx] += 1

            max_value = max(max_value, conf_matrix.max())
            all_confusion_data.append((phenotype, model, true_vals, pred_vals))

    print(f"Maximum value across all confusion matrices: {max_value}")

    # Create PDF with all confusion matrices
    pdf_filename = 'all_best_model_confusion_matrices.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

    # Create figure with subplots (2x2 grid per page)
    plots_per_page = 4
    n_pages = (len(all_confusion_data) + plots_per_page - 1) // plots_per_page

    plot_idx = 0
    summary_data = []

    for page in range(n_pages):
        fig = plt.figure(figsize=(12, 12))

        for subplot_idx in range(plots_per_page):
            if plot_idx >= len(all_confusion_data):
                break

            phenotype, model, true_vals, pred_vals = all_confusion_data[plot_idx]

            print(f"Processing {phenotype} with {model}...")

            if len(true_vals) > 0:
                ax = plt.subplot(2, 2, subplot_idx + 1)
                overall_acc, balanced_acc, n_samples = create_confusion_matrix_plot(
                    true_vals, pred_vals, phenotype, model, ax, vmax=max_value
                )

                summary_data.append({
                    'phenotype': phenotype.replace('_', ' ').title(),
                    'model': model.split('/')[-1] if '/' in model else model,
                    'overall_acc': overall_acc,
                    'balanced_acc': balanced_acc,
                    'n_samples': n_samples
                })

            plot_idx += 1

        plt.suptitle('Best Model Confusion Matrices for Each Phenotype',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # Add summary page
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')

    # Create summary table
    headers = ['Phenotype', 'Best Model', 'Accuracy', 'Balanced\nAccuracy', 'Sample Size']
    table_data = []

    for item in summary_data:
        table_data.append([
            item['phenotype'],
            item['model'],
            f"{item['overall_acc']:.1%}",
            f"{item['balanced_acc']:.1%}",
            f"{item['n_samples']:,}"
        ])

    # Sort by balanced accuracy
    table_data.sort(key=lambda x: float(x[3].strip('%')), reverse=True)

    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.25, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header with green theme
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2d5a2d')  # Dark green header
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors with light green tones
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#e8f5e8')  # Very light green

    ax.set_title('Summary: Best Model Performance by Phenotype',
                fontsize=14, fontweight='bold', pad=20)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Close PDF
    pdf.close()

    print(f"\n{'='*60}")
    print(f"PDF created: {pdf_filename}")
    print(f"{'='*60}")
    print(f"Total phenotypes analyzed: {len(summary_data)}")

    # Print summary
    if summary_data:
        avg_acc = np.mean([d['overall_acc'] for d in summary_data])
        avg_bal = np.mean([d['balanced_acc'] for d in summary_data])
        print(f"Average accuracy: {avg_acc:.1%}")
        print(f"Average balanced accuracy: {avg_bal:.1%}")


if __name__ == '__main__':
    main()