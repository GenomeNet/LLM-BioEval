#!/usr/bin/env python3
"""Generate a per-model accuracy table PDF for the LA test dataset.

This replicates the style of `model_accuracy_separate.pdf` but filters down to
models that have predictions for the LA species list. Data are pulled via the
running microbeLLM API so the script can be reused against other instances by
changing the dataset name or species file.
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use editable fonts in the generated PDF
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Mapping from phenotype field → short display label (keeps published ordering)
PHENOTYPE_LABELS = OrderedDict([
    ('spore_formation', 'Spore Form.'),
    ('motility', 'Motility'),
    ('biosafety_level', 'Biosafety'),
    ('animal_pathogenicity', 'Animal Path.'),
    ('plant_pathogenicity', 'Plant Path.'),
    ('host_association', 'Host Assoc.'),
    ('extreme_environment_tolerance', 'Extreme Env.'),
    ('cell_shape', 'Cell Shape'),
    ('gram_staining', 'Gram Stain'),
    ('biofilm_formation', 'Biofilm'),
])

# Phenotypes that stay hidden regardless of API response
EXCLUDED_PHENOTYPES = {
    'aerophilicity',
    'health_association',
    'hemolysis',
}

DATASET_SPECIES_FILE_MAP = {
    'WA_Test_Dataset': 'wa_with_gcount.txt',
    'LA_Test_Dataset': 'la.txt',
}


def fetch_json(endpoint: str) -> Optional[dict]:
    """Fetch JSON payload from the API, returning None on failure."""
    try:
        with urllib.request.urlopen(endpoint, timeout=60) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except urllib.error.URLError as exc:  # pragma: no cover - informational logging
        print(f"Request failed for {endpoint}: {exc}")
    return None


def fetch_ground_truth_map(api_url: str, dataset: str) -> Dict[str, Dict[str, str]]:
    endpoint = f"{api_url.rstrip('/')}/api/ground_truth/data?dataset={dataset}&per_page=20000"
    payload = fetch_json(endpoint) or {}
    if not payload.get('success'):
        raise RuntimeError(f"Unable to fetch ground truth data for {dataset}")

    gt_map: Dict[str, Dict[str, str]] = {}
    for item in payload.get('data', []):
        gt_map[item['binomial_name'].lower()] = item
    return gt_map


def fetch_predictions(api_url: str, species_file: str) -> List[Dict[str, str]]:
    endpoint = f"{api_url.rstrip('/')}/api/phenotype_analysis_filtered?species_file={species_file}"
    payload = fetch_json(endpoint) or {}
    if payload.get('error'):
        raise RuntimeError(f"Prediction API returned an error: {payload['error']}")
    return payload.get('data', [])


def fetch_template_fields(api_url: str, dataset: str) -> Dict[str, Dict[str, str]]:
    datasets_resp = fetch_json(f"{api_url.rstrip('/')}/api/ground_truth/datasets") or {}
    template_name = None
    for entry in datasets_resp.get('datasets', []):
        if entry.get('dataset_name') == dataset:
            template_name = entry.get('template_name')
            break
    if not template_name:
        raise RuntimeError(f"Template name for dataset {dataset} was not found")

    template_resp = fetch_json(
        f"{api_url.rstrip('/')}/api/template_field_definitions?template={template_name}"
    ) or {}
    if not template_resp.get('success'):
        raise RuntimeError(f"Unable to fetch field definitions for {template_name}")
    return template_resp.get('field_definitions', {})


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


def build_metrics_dataframe(predictions: List[Dict[str, str]],
                            ground_truth: Dict[str, Dict[str, str]],
                            field_definitions: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    phenotypes = [p for p in field_definitions.keys()
                  if p not in EXCLUDED_PHENOTYPES]

    rows = []
    models = sorted({pred['model'] for pred in predictions})

    for model in models:
        model_preds = [p for p in predictions if p['model'] == model]
        for phenotype in phenotypes:
            truth_vals: List[str] = []
            pred_vals: List[str] = []
            for pred in model_preds:
                species = pred.get('binomial_name', '').lower()
                gt = ground_truth.get(species)
                if not gt:
                    continue
                truth_val = normalize_value(gt.get(phenotype))
                pred_val = normalize_value(pred.get(phenotype))
                if truth_val is None or pred_val is None:
                    continue
                truth_vals.append(truth_val)
                pred_vals.append(pred_val)

            if not truth_vals:
                continue
            score = compute_balanced_accuracy(truth_vals, pred_vals)
            if math.isnan(score):
                continue
            short_model = model.split('/')[-1] if '/' in model else model
            rows.append({
                'model': short_model,
                'phenotype': phenotype,
                'balanced_acc': score,
                'sample_size': len(truth_vals),
            })

    return pd.DataFrame(rows)


def create_table_figure(df: pd.DataFrame, output_path: str) -> None:
    if df.empty:
        raise RuntimeError('No metrics available for table output')

    # Restrict to the configured phenotype order
    available_cols = [p for p in PHENOTYPE_LABELS.keys()
                      if p in df['phenotype'].unique()]
    display_headers = ['Model'] + [PHENOTYPE_LABELS[p] for p in available_cols]

    pivot = df.pivot(index='model', columns='phenotype', values='balanced_acc')
    pivot = pivot.reindex(columns=available_cols)

    # Order models by average balanced accuracy (desc)
    model_order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[model_order]

    cell_text = []
    for model in pivot.index:
        row_values = [model]
        for phenotype in pivot.columns:
            val = pivot.loc[model, phenotype]
            if pd.isna(val):
                row_values.append('\u2014')  # em dash for missing values
            else:
                row_values.append(f"{val:.2f}")
        cell_text.append(row_values)

    fig, ax = plt.subplots(figsize=(8.5, 2 + 0.4 * len(cell_text)))
    ax.axis('off')

    table = ax.table(cellText=cell_text,
                     colLabels=display_headers,
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Header styling
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor('#374151')
            cell.set_text_props(color='white', fontweight='bold')
        elif col_idx == 0:
            cell.set_text_props(fontweight='bold', ha='left')
        cell.set_edgecolor('#D1D5DB')

    plt.tight_layout()
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def resolve_species_file(dataset: str, species_file: Optional[str]) -> str:
    if species_file:
        return species_file
    if dataset in DATASET_SPECIES_FILE_MAP:
        return DATASET_SPECIES_FILE_MAP[dataset]
    dataset_lower = dataset.lower()
    if 'la_test' in dataset_lower:
        return 'la.txt'
    if 'wa_test' in dataset_lower or 'washington' in dataset_lower:
        return 'wa_with_gcount.txt'
    if 'artificial' in dataset_lower:
        return 'artificial.txt'
    raise RuntimeError('Species file could not be inferred; provide --species-file explicitly')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate a per-model accuracy table PDF for a ground-truth dataset.'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                        help='Base URL for the running microbeLLM web app')
    parser.add_argument('--dataset', default='LA_Test_Dataset',
                        help='Ground truth dataset name to analyze')
    parser.add_argument('--species-file', default=None,
                        help='Override the species file used for predictions')
    parser.add_argument('--output', default='model_accuracy_separate_la.pdf',
                        help='Output PDF path')

    args = parser.parse_args()

    plt.style.use('seaborn-v0_8-whitegrid')

    species_file = resolve_species_file(args.dataset, args.species_file)
    print(f"Using dataset={args.dataset}, species_file={species_file}")

    ground_truth = fetch_ground_truth_map(args.api_url, args.dataset)
    print(f"Loaded {len(ground_truth)} ground truth entries")

    predictions = fetch_predictions(args.api_url, species_file)
    print(f"Loaded {len(predictions)} predictions")

    fields = fetch_template_fields(args.api_url, args.dataset)
    print(f"Template exposes {len(fields)} phenotype fields")

    metrics_df = build_metrics_dataframe(predictions, ground_truth, fields)
    print(f"Computed metrics for {metrics_df['model'].nunique()} models")

    create_table_figure(metrics_df, args.output)
    print(f"Table saved to {args.output}")


if __name__ == '__main__':
    main()
