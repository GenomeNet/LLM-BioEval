#!/usr/bin/env python3
"""
Generate confusion matrix for Google Gemini-2.5-Pro's Spore Formation predictions.
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
        return 'unknown'

    val_lower = str(value).strip().lower()

    # Map common variations for spore formation
    if val_lower in ['yes', 'true', '1', 'positive', 'spore-forming', 'spore forming',
                     'forms spores', 'spore producer', 'spore-producer', 'endospore-forming']:
        return 'yes'
    elif val_lower in ['no', 'false', '0', 'negative', 'non-spore-forming',
                       'non spore forming', 'non-spore forming', 'does not form spores',
                       'no spore formation', 'non-endospore-forming']:
        return 'no'
    elif val_lower in ['unknown', 'not known', 'unclear', 'variable']:
        return 'unknown'

    # Return as-is if doesn't match common patterns
    return val_lower


def create_confusion_matrix():
    """Create confusion matrix for Gemini-2.5-Pro Spore Formation predictions."""

    print("Fetching data...")

    # Fetch prediction data - adjust the species file as needed
    pred_data = fetch_api_data('/api/phenotype_analysis_filtered?species_file=wa_with_gcount.txt')
    predictions = pred_data.get('data', [])

    # Fetch ground truth data
    gt_data = fetch_api_data('/api/ground_truth/data?dataset=WA_Test_Dataset&per_page=20000')

    # Create ground truth mapping
    gt_map = {}
    for item in gt_data.get('data', []):
        gt_map[item['binomial_name'].lower()] = item

    # Get Gemini-2.5-pro predictions for spore formation
    model_preds = [p for p in predictions if 'gemini-2.5-pro' in p['model'].lower() or 'gemini-2.5-pro' in p['model']]

    if not model_preds:
        # Try alternative model name format
        model_preds = [p for p in predictions if p['model'] == 'google/gemini-2.5-pro']

    print(f"Found {len(model_preds)} predictions for Gemini-2.5-Pro")

    true_vals = []
    pred_vals = []

    for pred in model_preds:
        species = pred.get('binomial_name', '').lower()
        if species in gt_map:
            true_val = normalize_value(gt_map[species].get('spore_formation'))
            pred_val = normalize_value(pred.get('spore_formation'))

            if true_val and pred_val:
                true_vals.append(true_val)
                pred_vals.append(pred_val)

    print(f"Matched {len(true_vals)} predictions with ground truth")

    if len(true_vals) == 0:
        print("No matching data found! Check model name and data availability.")
        return

    # Get unique labels
    labels = sorted(list(set(true_vals + pred_vals)))

    # Display labels for better readability
    label_display = {
        'yes': 'Spore-forming',
        'no': 'Non-spore-forming',
        'unknown': 'Unknown'
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
            print(f"{label_display.get(label, label):20s}: {acc:.3f} ({tp}/{total})")

    # Create figure with compact size
    fig, ax = plt.subplots(figsize=(6, 5))

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
    ax.set_xlabel('Predicted Spore Formation', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Spore Formation', fontsize=11, fontweight='bold')
    ax.set_title('Google Gemini-2.5-Pro Spore Formation Confusion Matrix',
                 fontsize=12, fontweight='bold', pad=15)

    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add accuracy annotations on diagonal
    for i in range(n_classes):
        tp = conf_matrix[i, i]
        total = conf_matrix[i, :].sum()
        if total > 0:
            acc = tp / total
            # Find the text annotation for this cell
            for text in ax.texts:
                if text.get_position() == (i + 0.5, i + 0.5):
                    text.set_text(f'{tp}\n({acc:.1%})')
                    text.set_fontsize(9)
                    text.set_fontweight('bold')
                    break

    # Calculate overall metrics
    total_correct = np.trace(conf_matrix)
    total_samples = conf_matrix.sum()
    overall_acc = total_correct / total_samples

    # Calculate balanced accuracy (excluding unknown if present)
    if 'unknown' in labels:
        # Calculate balanced accuracy only for yes/no
        binary_accuracies = []
        for i, label in enumerate(labels):
            if label != 'unknown':
                tp = conf_matrix[i, i]
                total = conf_matrix[i, :].sum()
                if total > 0:
                    binary_accuracies.append(tp / total)
        balanced_acc = np.mean(binary_accuracies) if binary_accuracies else 0
    else:
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
    output_path = 'gemini_2.5_pro_spore_formation_confusion_matrix.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nConfusion matrix saved to: {output_path}")

    # Also create a normalized version
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    # Normalize confusion matrix by row (true label)
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)

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
    ax2.set_xlabel('Predicted Spore Formation', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Spore Formation', fontsize=11, fontweight='bold')
    ax2.set_title('Google Gemini-2.5-Pro Spore Formation Confusion Matrix (Normalized)',
                  fontsize=12, fontweight='bold', pad=15)

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
    output_path_norm = 'gemini_2.5_pro_spore_formation_confusion_matrix_normalized.pdf'
    plt.savefig(output_path_norm, format='pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Normalized confusion matrix saved to: {output_path_norm}")

    plt.close('all')

    # Print summary statistics
    print("\n" + "="*50)
    print("CONFUSION MATRIX SUMMARY")
    print("="*50)
    print(f"Model: Google Gemini-2.5-Pro")
    print(f"Phenotype: Spore Formation")
    print(f"Total predictions: {total_samples:,}")
    print(f"Overall accuracy: {overall_acc:.1%}")
    print(f"Balanced accuracy: {balanced_acc:.1%}")

    print("\nPer-class recall (sensitivity):")
    for i, label in enumerate(labels):
        tp = conf_matrix[i, i]
        total = conf_matrix[i, :].sum()
        if total > 0:
            recall = tp / total
            print(f"  {label_display.get(label, label):20s}: {recall:.3f}")

    print("\nPer-class precision:")
    for j, label in enumerate(labels):
        tp = conf_matrix[j, j]
        total = conf_matrix[:, j].sum()
        if total > 0:
            precision = tp / total
            print(f"  {label_display.get(label, label):20s}: {precision:.3f}")

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
        true_disp = label_display.get(true, true)
        pred_disp = label_display.get(pred, pred)
        print(f"  {true_disp:20s} → {pred_disp:20s}: {count:3d} ({pct:.1f}% of {true_disp})")


if __name__ == '__main__':
    create_confusion_matrix()