#!/usr/bin/env python3
"""
Generate confusion matrices for all best model-phenotype combinations in a single panel.
All matrices have uniform cell sizes for better comparability.
"""

import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from typing import List, Dict, Optional
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def get_label_display(phenotype, label):
    """Get display label for a given phenotype and value."""

    if phenotype == 'spore_formation':
        mapping = {
            'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No',
            '1': 'Yes', '0': 'No'
        }
    elif phenotype == 'cell_shape':
        mapping = {
            'bacillus': 'Bacillus', 'coccus': 'Coccus', 'spirillum': 'Spirillum',
            'coccobacillus': 'Coccobac.', 'filamentous': 'Filament.',
            'pleomorphic': 'Pleomorph.', 'spiral': 'Spiral', 'vibrio': 'Vibrio',
            'rod': 'Rod'
        }
    elif phenotype == 'biosafety_level':
        mapping = {
            '1': 'BSL-1', '2': 'BSL-2', '3': 'BSL-3', '4': 'BSL-4',
            'bsl-1': 'BSL-1', 'bsl-2': 'BSL-2', 'bsl-3': 'BSL-3', 'bsl-4': 'BSL-4'
        }
    elif phenotype == 'gram_staining':
        mapping = {
            'positive': 'Pos', 'negative': 'Neg', 'variable': 'Var',
            'gram-positive': 'Pos', 'gram-negative': 'Neg',
            '+': 'Pos', '-': 'Neg'
        }
    elif phenotype in ['motility', 'animal_pathogenicity', 'plant_pathogenicity',
                       'host_association', 'biofilm_formation']:
        mapping = {
            'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No',
            '1': 'Yes', '0': 'No', 'positive': 'Yes', 'negative': 'No',
            'motile': 'Yes', 'non-motile': 'No', 'immotile': 'No',
            'pathogenic': 'Yes', 'non-pathogenic': 'No'
        }
    elif phenotype == 'extreme_environment_tolerance':
        mapping = {
            'mesophile': 'Meso', 'thermophile': 'Thermo',
            'psychrophile': 'Psychro', 'halophile': 'Halo',
            'acidophile': 'Acido', 'alkaliphile': 'Alkali',
            'yes': 'Tolerant', 'no': 'Non-tol'
        }
    else:
        mapping = {}

    return mapping.get(label, label[:8] if len(label) > 8 else label)  # Truncate long labels


def create_uniform_confusion_matrix(ax, conf_matrix, labels, phenotype, model, vmax,
                                   cell_size=0.5, show_colorbar=False):
    """Create a confusion matrix with uniform cell sizes."""

    n_classes = len(labels)

    # Clear the axis
    ax.clear()

    # Calculate figure size based on number of classes
    matrix_size = n_classes * cell_size

    # Create the matrix visualization manually
    for i in range(n_classes):
        for j in range(n_classes):
            value = conf_matrix[i, j]
            # Normalize color based on global vmax
            color_intensity = value / vmax if vmax > 0 else 0

            # Add rectangle with color
            rect = plt.Rectangle((j * cell_size, (n_classes - i - 1) * cell_size),
                                cell_size, cell_size,
                                facecolor=plt.cm.Greens(color_intensity),
                                edgecolor='white', linewidth=1)
            ax.add_patch(rect)

            # Add text annotation
            text_color = 'white' if color_intensity > 0.6 else 'black'
            ax.text(j * cell_size + cell_size/2,
                   (n_classes - i - 1) * cell_size + cell_size/2,
                   str(value), ha='center', va='center',
                   fontsize=7, color=text_color, fontweight='bold')

    # Set axis properties
    ax.set_xlim(0, matrix_size)
    ax.set_ylim(0, matrix_size)
    ax.set_aspect('equal')

    # Add labels
    display_labels = [get_label_display(phenotype, l) for l in labels]

    # X-axis (predicted) labels at top
    ax.set_xticks([i * cell_size + cell_size/2 for i in range(n_classes)])
    ax.set_xticklabels(display_labels, rotation=45, ha='left', fontsize=6)
    ax.xaxis.tick_top()

    # Y-axis (actual) labels
    ax.set_yticks([i * cell_size + cell_size/2 for i in range(n_classes)])
    ax.set_yticklabels(display_labels[::-1], rotation=0, fontsize=6)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add title
    phenotype_title = phenotype.replace('_', ' ').title()
    model_short = model.split('/')[-1] if '/' in model else model

    # Shorter phenotype names for compact display
    short_names = {
        'Extreme Environment Tolerance': 'Extreme Env.',
        'Animal Pathogenicity': 'Animal Path.',
        'Plant Pathogenicity': 'Plant Path.',
        'Biofilm Formation': 'Biofilm',
        'Host Association': 'Host Assoc.',
        'Spore Formation': 'Spore Form.',
        'Biosafety Level': 'Biosafety',
        'Gram Staining': 'Gram Stain'
    }
    phenotype_title = short_names.get(phenotype_title, phenotype_title)

    # Calculate metrics
    total_samples = conf_matrix.sum()
    correct = np.trace(conf_matrix)
    accuracy = correct / total_samples if total_samples > 0 else 0

    # Calculate balanced accuracy
    class_accuracies = []
    for i in range(n_classes):
        row_sum = conf_matrix[i, :].sum()
        if row_sum > 0:
            class_accuracies.append(conf_matrix[i, i] / row_sum)
    balanced_acc = np.mean(class_accuracies) if class_accuracies else 0

    # Title with metrics
    ax.set_title(f'{phenotype_title}\n{model_short}\nAcc: {accuracy:.1%} | Bal: {balanced_acc:.1%}',
                fontsize=7, fontweight='bold', pad=10)

    # Add axis labels
    ax.set_xlabel('Predicted', fontsize=6, labelpad=5)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Actual', fontsize=6, labelpad=5)

    return accuracy, balanced_acc, total_samples


def main():
    """Generate confusion matrices with uniform cell sizes."""

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

    # Best model-phenotype pairs
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

    # First pass: collect all confusion matrices and find max value
    print("Calculating confusion matrices...")
    all_matrices = []
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
            labels = sorted(list(set(true_vals + pred_vals)))
            n_classes = len(labels)
            conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

            for true, pred in zip(true_vals, pred_vals):
                true_idx = labels.index(true)
                pred_idx = labels.index(pred)
                conf_matrix[true_idx, pred_idx] += 1

            max_value = max(max_value, conf_matrix.max())
            all_matrices.append((phenotype, model, conf_matrix, labels))

    print(f"Maximum value: {max_value}")

    # Create single figure with all matrices
    fig = plt.figure(figsize=(16, 10))

    # Create grid layout (2 rows x 5 columns)
    n_cols = 5
    n_rows = 2

    # Add main title
    fig.suptitle('Best Model Confusion Matrices - Uniform Cell Size Comparison',
                 fontsize=12, fontweight='bold', y=0.98)

    summary_data = []

    for idx, (phenotype, model, conf_matrix, labels) in enumerate(all_matrices):
        row = idx // n_cols
        col = idx % n_cols

        # Create subplot with custom position for uniform sizing
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        print(f"Processing {phenotype} with {model} ({len(labels)} classes)...")

        accuracy, balanced_acc, n_samples = create_uniform_confusion_matrix(
            ax, conf_matrix, labels, phenotype, model, max_value,
            cell_size=0.4, show_colorbar=(idx == len(all_matrices) - 1)
        )

        summary_data.append({
            'phenotype': phenotype.replace('_', ' ').title(),
            'model': model.split('/')[-1] if '/' in model else model,
            'accuracy': accuracy,
            'balanced_acc': balanced_acc,
            'n_samples': n_samples,
            'n_classes': len(labels)
        })

    # Add a single colorbar for all plots
    cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='Greens',
                                              norm=plt.Normalize(vmin=0, vmax=max_value)),
                       cax=cbar_ax)
    cbar.set_label('Count', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])

    # Save the main figure
    output_file = 'all_confusion_matrices_uniform.pdf'
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nUniform confusion matrices saved to: {output_file}")

    # Create summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"{'Phenotype':<30} {'Classes':<8} {'Accuracy':<10} {'Balanced':<10}")
    print("-"*60)

    for item in sorted(summary_data, key=lambda x: x['balanced_acc'], reverse=True):
        print(f"{item['phenotype']:<30} {item['n_classes']:<8} "
              f"{item['accuracy']:>8.1%} {item['balanced_acc']:>8.1%}")

    avg_acc = np.mean([d['accuracy'] for d in summary_data])
    avg_bal = np.mean([d['balanced_acc'] for d in summary_data])
    print("-"*60)
    print(f"{'AVERAGE':<30} {'':<8} {avg_acc:>8.1%} {avg_bal:>8.1%}")

    plt.close('all')


if __name__ == '__main__':
    main()