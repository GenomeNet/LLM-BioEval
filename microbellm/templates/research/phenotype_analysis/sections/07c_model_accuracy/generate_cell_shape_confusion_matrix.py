#!/usr/bin/env python3
"""
Generate confusion matrix for GPT-4.1-nano's Cell Shape predictions.
"""

import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict
import seaborn as sns

# Set font type for PDF editing
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans',
                                          'Bitstream Vera Sans', 'sans-serif']


def fetch_api_data(endpoint: str, api_url: str = 'http://localhost:5050') -> Dict:
    """Fetch data from API endpoint."""
    full_url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urllib.request.urlopen(full_url, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def normalize_value(value) -> str:
    """Normalize values for comparison."""
    if value is None or value == '' or str(value).lower() in ['n/a', 'na', 'null', 'none', 'nan', 'unknown']:
        return None
    return str(value).strip().lower()


def create_confusion_matrix():
    """Create confusion matrix for GPT-4.1-nano Cell Shape predictions."""

    print("Fetching data...")

    # Fetch prediction data
    pred_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt')
    predictions = pred_data.get('data', [])

    # Fetch ground truth data
    gt_data = fetch_api_data('/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000')

    # Create ground truth mapping
    gt_map = {}
    for item in gt_data.get('data', []):
        gt_map[item['binomial_name'].lower()] = item

    # Get GPT-4.1-nano predictions for cell shape
    model_preds = [p for p in predictions if 'gpt-4.1-nano' in p['model'].lower()]

    true_vals = []
    pred_vals = []

    for pred in model_preds:
        species = pred.get('binomial_name', '').lower()
        if species in gt_map:
            true_val = normalize_value(gt_map[species].get('cell_shape'))
            pred_val = normalize_value(pred.get('cell_shape'))

            if true_val and pred_val:
                true_vals.append(true_val)
                pred_vals.append(pred_val)

    # Get unique labels
    labels = sorted(list(set(true_vals + pred_vals)))
    label_display = {
        'bacillus': 'Bacillus',
        'coccus': 'Coccus',
        'spirillum': 'Spirillum',
        'tail': 'Tail'
    }

    # Create confusion matrix
    n_classes = len(labels)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(true_vals, pred_vals):
        true_idx = labels.index(true)
        pred_idx = labels.index(pred)
        conf_matrix[true_idx, pred_idx] += 1

    # Calculate per-class metrics
    print("\nCalculating metrics...")
    accuracies = []
    for i, label in enumerate(labels):
        tp = conf_matrix[i, i]
        total = conf_matrix[i, :].sum()
        if total > 0:
            acc = tp / total
            accuracies.append(acc)
            print(f"{label_display.get(label, label):12s}: {acc:.3f} ({tp}/{total})")

    # Create figure with compact size
    fig, ax = plt.subplots(figsize=(5, 4))

    # Create heatmap with annotations
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[label_display.get(l, l) for l in labels],
                yticklabels=[label_display.get(l, l) for l in labels],
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='gray',
                square=True,
                ax=ax,
                annot_kws={'fontsize': 10})

    # Set labels and title
    ax.set_xlabel('Predicted Cell Shape', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Cell Shape', fontsize=11, fontweight='bold')
    ax.set_title('GPT-4.1-Nano Cell Shape Confusion Matrix', fontsize=12, fontweight='bold', pad=15)

    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add accuracy annotations on diagonal
    for i in range(n_classes):
        tp = conf_matrix[i, i]
        total = conf_matrix[i, :].sum()
        if total > 0:
            acc = tp / total
            # Add accuracy percentage below count
            current_text = ax.texts[i * n_classes + i]
            current_text.set_text(f'{tp}\n({acc:.1%})')
            current_text.set_fontsize(9)
            current_text.set_fontweight('bold')

    # Calculate overall metrics
    total_correct = np.trace(conf_matrix)
    total_samples = conf_matrix.sum()
    overall_acc = total_correct / total_samples

    # Calculate balanced accuracy
    balanced_acc = np.mean(accuracies)

    # Add text box with overall metrics
    metrics_text = (f'Overall Accuracy: {overall_acc:.1%}\n'
                   f'Balanced Accuracy: {balanced_acc:.1%}\n'
                   f'Total Samples: {total_samples:,}')

    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()

    # Save as PDF
    output_path = 'gpt_4.1_nano_cell_shape_confusion_matrix.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nConfusion matrix saved to: {output_path}")

    # Also create a normalized version
    fig2, ax2 = plt.subplots(figsize=(5, 4))

    # Normalize confusion matrix by row (true label)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Create normalized heatmap
    sns.heatmap(conf_matrix_norm,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                xticklabels=[label_display.get(l, l) for l in labels],
                yticklabels=[label_display.get(l, l) for l in labels],
                cbar_kws={'label': 'Proportion', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='gray',
                square=True,
                ax=ax2,
                annot_kws={'fontsize': 10})

    # Set labels and title
    ax2.set_xlabel('Predicted Cell Shape', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Cell Shape', fontsize=11, fontweight='bold')
    ax2.set_title('GPT-4.1-Nano Cell Shape Confusion Matrix (Normalized)', fontsize=12, fontweight='bold', pad=15)

    # Rotate labels
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    # Add text box with overall metrics
    ax2.text(0.02, 0.98, metrics_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()

    # Save normalized version
    output_path_norm = 'gpt_4.1_nano_cell_shape_confusion_matrix_normalized.pdf'
    plt.savefig(output_path_norm, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Normalized confusion matrix saved to: {output_path_norm}")

    plt.close('all')

    # Print summary statistics
    print("\n" + "="*50)
    print("CONFUSION MATRIX SUMMARY")
    print("="*50)
    print(f"Model: GPT-4.1-Nano")
    print(f"Phenotype: Cell Shape")
    print(f"Total predictions: {total_samples:,}")
    print(f"Overall accuracy: {overall_acc:.1%}")
    print(f"Balanced accuracy: {balanced_acc:.1%}")
    print("\nPer-class recall (sensitivity):")
    for i, label in enumerate(labels):
        tp = conf_matrix[i, i]
        total = conf_matrix[i, :].sum()
        if total > 0:
            recall = tp / total
            print(f"  {label_display.get(label, label):12s}: {recall:.3f}")

    print("\nPer-class precision:")
    for j, label in enumerate(labels):
        tp = conf_matrix[j, j]
        total = conf_matrix[:, j].sum()
        if total > 0:
            precision = tp / total
            print(f"  {label_display.get(label, label):12s}: {precision:.3f}")

    print("\nMost common confusions:")
    # Find off-diagonal elements with highest counts
    conf_copy = conf_matrix.copy()
    np.fill_diagonal(conf_copy, 0)

    confusions = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and conf_copy[i, j] > 0:
                confusions.append((labels[i], labels[j], conf_copy[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)

    for true, pred, count in confusions[:5]:
        total_true = conf_matrix[labels.index(true), :].sum()
        pct = (count / total_true) * 100
        print(f"  {label_display.get(true, true):12s} → {label_display.get(pred, pred):12s}: {count:3d} ({pct:.1f}% of {label_display.get(true, true)})")


if __name__ == '__main__':
    create_confusion_matrix()