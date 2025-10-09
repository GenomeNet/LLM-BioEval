#!/usr/bin/env python3
"""Correlate grok-3-mini balanced accuracies between WA and LA datasets."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Database and model configuration
DB_PATH = Path('microbellm.db')
MODEL_NAME = 'x-ai/grok-3-mini'
SYSTEM_TEMPLATE = 'templates/system/template1_phenotype.txt'

WA_DATASET = 'WA_Test_Dataset'
WA_SPECIES_FILE = 'wa_with_gcount.txt'

LA_DATASET = 'LA_Test_Dataset'
LA_SPECIES_FILE = 'la.txt'

EXCLUDED_FIELDS = {
    'id', 'dataset_name', 'template_name', 'binomial_name', 'import_date',
    'aerophilicity', 'health_association', 'hemolysis'
}

MISSING_TOKENS = {
    'n/a', 'na', 'null', 'none', 'nan', 'undefined', '-', 'unknown', 'missing', ''
}


def normalize_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in MISSING_TOKENS:
        return None
    if ',' in s or ';' in s:
        parts = [p.strip() for p in s.replace(';', ',').split(',') if p.strip()]
        if not parts:
            return None
        return ','.join(sorted(parts))
    return s


def to_boolean(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {'true', '1', 'yes', 't', 'y'}:
        return True
    if s in {'false', '0', 'no', 'f', 'n'}:
        return False
    return None


def compute_balanced_accuracy(truths: Iterable[str], preds: Iterable[str]) -> float:
    truth_list = list(truths)
    pred_list = list(preds)
    binary_pairs = []
    for t, p in zip(truth_list, pred_list):
        tb = to_boolean(t)
        pb = to_boolean(p)
        if tb is not None and pb is not None:
            binary_pairs.append((tb, pb))

    if binary_pairs and len(binary_pairs) == len(truth_list) == len(pred_list):
        tp = tn = fp = fn = 0
        for t, p in binary_pairs:
            if t and p:
                tp += 1
            elif not t and not p:
                tn += 1
            elif not t and p:
                fp += 1
            else:
                fn += 1
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return (sens + spec) / 2.0

    labels = sorted({str(v) for v in truth_list + pred_list})
    if not labels:
        return float('nan')

    conf = {label: {label2: 0 for label2 in labels} for label in labels}
    for t, p in zip(truth_list, pred_list):
        conf[str(t)][str(p)] += 1

    recall_sum = 0.0
    for label in labels:
        tp = conf[label][label]
        fn = sum(conf[label][other] for other in labels if other != label)
        denom = tp + fn
        recall_sum += (tp / denom) if denom else 0.0

    return recall_sum / len(labels) if labels else float('nan')


def load_ground_truth(conn: sqlite3.Connection, dataset_name: str, phenotypes: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    query = """
        SELECT binomial_name, {}
        FROM ground_truth
        WHERE dataset_name = ?
    """.format(', '.join(phenotypes))
    result: Dict[str, Dict[str, Optional[str]]] = {}
    for row in conn.execute(query, (dataset_name,)):
        species = row[0].lower()
        result[species] = {
            phenotype: normalize_value(value)
            for phenotype, value in zip(phenotypes, row[1:])
        }
    return result


def load_predictions(conn: sqlite3.Connection, species_file: str, phenotypes: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    query = """
    WITH ranked AS (
        SELECT
            lower(binomial_name) AS species_lower,
            model,
            status,
            created_at,
            {} ,
            ROW_NUMBER() OVER (
                PARTITION BY binomial_name, model
                ORDER BY
                    CASE status
                        WHEN 'completed' THEN 0
                        WHEN 'failed' THEN 1
                        ELSE 2
                    END,
                    datetime(created_at) DESC
            ) AS rn
        FROM processing_results
        WHERE species_file = ?
          AND system_template = ?
          AND model = ?
    )
    SELECT * FROM ranked
    WHERE rn = 1 AND status = 'completed'
    """.format(', '.join(phenotypes))

    predictions: Dict[str, Dict[str, Optional[str]]] = {}
    for row in conn.execute(query, (species_file, SYSTEM_TEMPLATE, MODEL_NAME)):
        species = row[0]
        field_values = [normalize_value(value) for value in row[4:]]
        predictions[species] = dict(zip(phenotypes, field_values))
    return predictions


def calculate_metrics(ground_truth: Dict[str, Dict[str, Optional[str]]],
                      predictions: Dict[str, Dict[str, Optional[str]]],
                      phenotypes: List[str]) -> pd.DataFrame:
    rows = []
    for phenotype in phenotypes:
        truth_vals: List[str] = []
        pred_vals: List[str] = []
        for species, pred_fields in predictions.items():
            gt_fields = ground_truth.get(species)
            if not gt_fields:
                continue
            gt_val = gt_fields.get(phenotype)
            pred_val = pred_fields.get(phenotype)
            if gt_val is None or pred_val is None:
                continue
            truth_vals.append(gt_val)
            pred_vals.append(pred_val)

        if not truth_vals:
            continue

        score = compute_balanced_accuracy(truth_vals, pred_vals)
        if math.isnan(score):
            continue
        rows.append({
            'phenotype': phenotype,
            'balanced_acc': score,
            'sample_size': len(truth_vals),
        })
    return pd.DataFrame(rows)


def correlate_grok() -> Tuple[pd.DataFrame, float]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    columns = [row['name'] for row in conn.execute("PRAGMA table_info(ground_truth)")]
    phenotypes = [col for col in columns if col not in EXCLUDED_FIELDS]

    wa_truth = load_ground_truth(conn, WA_DATASET, phenotypes)
    la_truth = load_ground_truth(conn, LA_DATASET, phenotypes)

    wa_preds = load_predictions(conn, WA_SPECIES_FILE, phenotypes)
    la_preds = load_predictions(conn, LA_SPECIES_FILE, phenotypes)

    conn.close()

    wa_metrics = calculate_metrics(wa_truth, wa_preds, phenotypes)
    la_metrics = calculate_metrics(la_truth, la_preds, phenotypes)

    merged = pd.merge(
        wa_metrics,
        la_metrics,
        on='phenotype',
        suffixes=('_wa', '_la')
    )

    if merged.empty:
        raise RuntimeError('No overlapping phenotypes to correlate.')

    correlation = float(np.corrcoef(merged['balanced_acc_wa'], merged['balanced_acc_la'])[0, 1])

    merged['phenotype_label'] = merged['phenotype'].str.replace('_', ' ').str.title()

    return merged, correlation


def plot_correlation(df: pd.DataFrame, correlation: float, output_path: Path) -> None:
    plt.figure(figsize=(4.0, 3.2))
    x = df['balanced_acc_wa'].values
    y = df['balanced_acc_la'].values

    plt.scatter(x, y, color='#1d4ed8', label='_nolegend_')

    # Best-fit trend line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='#f97316', linewidth=1.5,
             label=f'Trend: y = {slope:.2f}x + {intercept:.2f}')

    # Reference line y=x to show parity between datasets
    parity_min = min(x.min(), y.min())
    parity_max = max(x.max(), y.max())
    plt.plot([parity_min, parity_max], [parity_min, parity_max],
             color='0.6', linestyle='--', linewidth=1, label='y = x')

    for _, row in df.iterrows():
        plt.annotate(row['phenotype_label'],
                     (row['balanced_acc_wa'], row['balanced_acc_la']),
                     textcoords='offset points', xytext=(4, 4), fontsize=7)

    plt.xlabel('WA Balanced Accuracy')
    plt.ylabel('LA Balanced Accuracy')
    plt.title(f'grok-3-mini: WA vs LA balanced accuracy\nPearson r = {correlation:.3f}')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.legend(frameon=False, fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path)
    plt.close()


def main() -> None:
    df, corr = correlate_grok()

    output_csv = Path('microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/grok_wa_la_accuracy.csv')
    df.to_csv(output_csv, index=False)
    print('Per-phenotype balanced accuracy saved to', output_csv)

    output_plot = Path('microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/grok_wa_la_accuracy_correlation.png')
    plot_correlation(df, corr, output_plot)
    print('Correlation plot saved to', output_plot)

    display_df = df[['phenotype', 'balanced_acc_wa', 'sample_size_wa', 'balanced_acc_la', 'sample_size_la']]
    display_df = display_df.sort_values('balanced_acc_wa', ascending=False)

    print('\nGrok-3-mini balanced accuracy comparison (WA vs LA):')
    print(display_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nPearson correlation (WA vs LA): {corr:.3f}")


if __name__ == '__main__':
    main()
# Ensure text is preserved in vector exports (Type 42 fonts)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Use a clean sans-serif font
plt.rcParams['font.family'] = 'DejaVu Sans'
