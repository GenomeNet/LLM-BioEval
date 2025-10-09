#!/usr/bin/env python3
"""Export per-model phenotype accuracy metrics for supplementary tables."""
from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DB_PATH = Path('microbellm.db')
DATASET_NAME = 'WA_Test_Dataset'
SPECIES_FILE = 'wa_with_gcount.txt'
SYSTEM_TEMPLATE = 'templates/system/template1_phenotype.txt'

OUTPUT_LONG = Path('microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/model_accuracy_metrics_long.csv')
OUTPUT_PIVOT = Path('microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/model_accuracy_metrics_pivot.csv')
OUTPUT_PREDICTIONS = Path('microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/model_predictions_long.csv')

MISSING_TOKENS = {'n/a', 'na', 'null', 'none', 'nan', 'undefined', '-', 'unknown', 'missing', ''}
EXCLUDED_FIELDS = {'id', 'dataset_name', 'template_name', 'binomial_name', 'import_date',
                   'aerophilicity', 'health_association', 'hemolysis'}


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


def balanced_accuracy(true_vals: List[str], pred_vals: List[str]) -> Tuple[float, Dict[str, float]]:
    binary_pairs = []
    for t, p in zip(true_vals, pred_vals):
        tb = to_boolean(t)
        pb = to_boolean(p)
        if tb is not None and pb is not None:
            binary_pairs.append((tb, pb))

    if len(binary_pairs) == len(true_vals) == len(pred_vals) and binary_pairs:
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
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        return (sens + spec) / 2.0, {
            'metric_type': 'binary',
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec
        }

    labels = sorted({str(v) for v in true_vals + pred_vals})
    if not labels:
        return float('nan'), {'metric_type': 'multiclass'}

    conf = {label: {label2: 0 for label2 in labels} for label in labels}
    for t, p in zip(true_vals, pred_vals):
        conf[str(t)][str(p)] += 1

    recalls = []
    precisions = []
    for label in labels:
        tp = conf[label][label]
        fn = sum(conf[label][other] for other in labels if other != label)
        fp = sum(conf[other][label] for other in labels if other != label)
        denom_rec = tp + fn
        denom_prec = tp + fp
        recalls.append(tp / denom_rec if denom_rec else 0.0)
        precisions.append(tp / denom_prec if denom_prec else 0.0)

    macro_recall = (sum(recalls) / len(recalls)) if recalls else float('nan')
    macro_precision = (sum(precisions) / len(precisions)) if precisions else float('nan')
    return macro_recall, {
        'metric_type': 'multiclass',
        'n_classes': len(labels),
        'precision': macro_precision
    }


def load_ground_truth(conn: sqlite3.Connection, phenotypes: List[str]) -> Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, str]]:
    gt_map: Dict[str, Dict[str, Optional[str]]] = {}
    species_names: Dict[str, str] = {}
    query = """
        SELECT *
        FROM ground_truth
        WHERE dataset_name = ? AND template_name = 'template1_phenotype'
    """
    for row in conn.execute(query, (DATASET_NAME,)):
        species_original = row['binomial_name']
        species = species_original.lower()
        species_names[species] = species_original
        gt_map[species] = {field: normalize_value(row[field]) for field in phenotypes}
    return gt_map, species_names


def load_predictions(conn: sqlite3.Connection, phenotypes: List[str]) -> Tuple[Dict[str, Dict[str, Dict[str, Optional[str]]]], Dict[str, str]]:
    query = """
    WITH ranked AS (
        SELECT
            binomial_name,
            lower(binomial_name) AS species_lower,
            model,
            status,
            created_at,
            """ + ", ".join(phenotypes) + """
        ,
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
        WHERE species_file = ? AND system_template = ?
    )
    SELECT * FROM ranked
    WHERE rn = 1 AND status = 'completed'
    """

    predictions: Dict[str, Dict[str, Dict[str, Optional[str]]]] = defaultdict(dict)
    species_names: Dict[str, str] = {}
    for row in conn.execute(query, (SPECIES_FILE, SYSTEM_TEMPLATE)):
        species = row['species_lower']
        model = row['model']
        species_names.setdefault(species, row['binomial_name'])
        predictions[model][species] = {field: normalize_value(row[field]) for field in phenotypes}
    return predictions, species_names


def export_tables():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    columns = [row['name'] for row in conn.execute("PRAGMA table_info(ground_truth)")]
    phenotypes = [col for col in columns if col not in EXCLUDED_FIELDS]

    ground_truth, gt_species_names = load_ground_truth(conn, phenotypes)
    predictions, pred_species_names = load_predictions(conn, phenotypes)
    conn.close()

    records = []
    pivot_data: Dict[str, Dict[str, float]] = defaultdict(dict)
    prediction_records: List[Dict[str, str]] = []

    species_name_lookup = gt_species_names.copy()
    species_name_lookup.update(pred_species_names)

    for model, species_preds in predictions.items():
        model_display = model.split('/')[-1] if '/' in model else model

        for species, pred_fields in species_preds.items():
            species_name = species_name_lookup.get(species, species)
            truth_fields = ground_truth.get(species, {})

            for phenotype in phenotypes:
                pred = pred_fields.get(phenotype)
                truth = truth_fields.get(phenotype)
                if pred is None and truth is None:
                    continue
                prediction_records.append({
                    'Model': model,
                    'Model_Display': model_display,
                    'Species': species_name,
                    'Phenotype': phenotype,
                    'Prediction': pred if pred is not None else '',
                    'Ground_Truth': truth if truth is not None else ''
                })

        for phenotype in phenotypes:
            true_vals: List[str] = []
            pred_vals: List[str] = []

            for species, pred_fields in species_preds.items():
                truth = ground_truth.get(species, {}).get(phenotype)
                pred = pred_fields.get(phenotype)
                if truth is not None and pred is not None:
                    true_vals.append(truth)
                    pred_vals.append(pred)

            if not true_vals:
                continue

            score, details = balanced_accuracy(true_vals, pred_vals)
            if math.isnan(score):
                continue

            precision_val = details.get('precision')
            records.append({
                'Model': model,
                'Model_Display': model_display,
                'Phenotype': phenotype,
                'Balanced_Accuracy': round(score, 4),
                'Precision': round(precision_val, 4) if isinstance(precision_val, (int, float)) and not math.isnan(precision_val) else '',
                'Sample_Size': len(true_vals),
                'Metric_Type': details.get('metric_type', 'unknown')
            })
            pivot_data[model_display][phenotype] = round(score, 4)

    records.sort(key=lambda r: (r['Model_Display'], r['Phenotype']))

    OUTPUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_LONG.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Model_Display', 'Phenotype',
                                               'Balanced_Accuracy', 'Precision', 'Sample_Size', 'Metric_Type'])
        writer.writeheader()
        writer.writerows(records)

    # Build pivot table
    phenotypes_sorted = sorted({row['Phenotype'] for row in records})
    model_rows = sorted(pivot_data.keys(), key=lambda m: m.lower())

    with OUTPUT_PIVOT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model'] + phenotypes_sorted)
        for model_display in model_rows:
            row = [model_display]
            for phenotype in phenotypes_sorted:
                val = pivot_data[model_display].get(phenotype)
                row.append(f"{val:.4f}" if val is not None else '')
            writer.writerow(row)

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PREDICTIONS.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Model_Display', 'Species', 'Phenotype', 'Prediction', 'Ground_Truth'])
        writer.writeheader()
        writer.writerows(prediction_records)

    print(f"Saved long-format metrics to {OUTPUT_LONG}")
    print(f"Saved pivot table to {OUTPUT_PIVOT}")
    print(f"Saved prediction records to {OUTPUT_PREDICTIONS}")
    print(f"Models processed: {len(pivot_data)}; Phenotypes: {len(phenotypes_sorted)}")


if __name__ == '__main__':
    export_tables()
