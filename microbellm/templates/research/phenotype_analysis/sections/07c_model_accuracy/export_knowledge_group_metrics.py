#!/usr/bin/env python3
"""Generate per-phenotype accuracy metrics stratified by knowledge group from exported predictions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


PHENOTYPES: Iterable[str] = (
    'gram_staining',
    'motility',
    'extreme_environment_tolerance',
    'biofilm_formation',
    'animal_pathogenicity',
    'biosafety_level',
    'host_association',
    'plant_pathogenicity',
    'spore_formation',
    'cell_shape',
)

VALID_GROUPS = {'limited', 'moderate', 'extensive'}
MIN_SAMPLES = 30


def dataset_csv_path(dataset: str) -> Path:
    base = Path('microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy')
    mapping = {
        'WA_Test_Dataset': base / 'knowledge_accuracy_wa.csv',
        'LA_Test_Dataset': base / 'knowledge_accuracy_la.csv',
    }
    if dataset not in mapping:
        raise ValueError(f'Unsupported dataset {dataset}')
    return mapping[dataset]


def compute_metrics_from_counts(counts: Dict[Tuple[str, str], int]) -> Tuple[int, float, float, float, float, int]:
    sample_size = sum(counts.values())
    if sample_size == 0:
        return sample_size, float('nan'), float('nan'), float('nan'), float('nan'), 0

    labels = sorted({truth for truth, _ in counts} | {pred for _, pred in counts})
    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    matrix = [[0] * n_labels for _ in range(n_labels)]
    for (truth, pred), count in counts.items():
        i = label_to_idx[truth]
        j = label_to_idx[pred]
        matrix[i][j] += count

    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0

    for idx in range(n_labels):
        tp = matrix[idx][idx]
        fn = sum(matrix[idx][j] for j in range(n_labels) if j != idx)
        fp = sum(matrix[i][idx] for i in range(n_labels) if i != idx)

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        recall_total += recall
        precision_total += precision
        f1_total += f1

    balanced_accuracy = recall_total / n_labels if n_labels else float('nan')
    precision_macro = precision_total / n_labels if n_labels else float('nan')
    recall_macro = recall_total / n_labels if n_labels else float('nan')
    f1_macro = f1_total / n_labels if n_labels else float('nan')

    return sample_size, balanced_accuracy, precision_macro, recall_macro, f1_macro, n_labels


def export_metrics(dataset: str, output: Path, model_filter: str | None) -> None:
    source_path = dataset_csv_path(dataset)
    if not source_path.exists():
        raise FileNotFoundError(
            f'{source_path} not found. Generate the per-observation export first with '
            'export_knowledge_grouped_predictions.py.'
        )

    # Aggregated counts: model -> knowledge group -> phenotype -> {(truth, pred): count}
    counts: Dict[str, Dict[str, Dict[str, Dict[Tuple[str, str], int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )

    with source_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            knowledge_group = (row['knowledge_group'] or '').lower()
            if knowledge_group not in VALID_GROUPS:
                continue

            model = row['model']
            if model_filter and model != model_filter:
                continue

            phenotype = row['phenotype']
            if phenotype not in PHENOTYPES:
                continue

            truth = (row['ground_truth'] or '').strip().lower()
            pred = (row['prediction'] or '').strip().lower()
            if not truth or not pred:
                continue

            counts[model][knowledge_group][phenotype][(truth, pred)] += 1

    with output.open('w', newline='', encoding='utf-8') as fh:
        fieldnames = [
            'dataset', 'model', 'knowledge_group', 'phenotype',
            'sample_size', 'balanced_accuracy', 'precision', 'recall', 'f1', 'num_classes'
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for model in sorted(counts.keys()):
            for knowledge_group in VALID_GROUPS:
                phenotype_map = counts[model].get(knowledge_group, {})
                for phenotype in PHENOTYPES:
                    pair_counts = phenotype_map.get(phenotype)
                    if not pair_counts:
                        continue
                    sample_size, bal_acc, precision, recall, f1, n_classes = compute_metrics_from_counts(pair_counts)
                    if sample_size < MIN_SAMPLES:
                        continue
                    writer.writerow({
                        'dataset': dataset,
                        'model': model,
                        'knowledge_group': knowledge_group,
                        'phenotype': phenotype,
                        'sample_size': sample_size,
                        'balanced_accuracy': round(bal_acc, 4),
                        'precision': round(precision, 4),
                        'recall': round(recall, 4),
                        'f1': round(f1, 4),
                        'num_classes': n_classes,
                    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Summarise phenotype accuracy metrics by knowledge group from exported CSV data.'
    )
    parser.add_argument(
        '--dataset',
        choices=['WA_Test_Dataset', 'LA_Test_Dataset'],
        default='LA_Test_Dataset',
        help='Dataset to analyse.'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV path (default: knowledge_accuracy_metrics_<dataset>.csv).'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Optional model identifier to filter (e.g., x-ai/grok-3-mini).'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else Path(
        f'microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/knowledge_accuracy_metrics_{args.dataset.lower()}.csv'
    )
    export_metrics(args.dataset, output_path, model_filter=args.model)
    print(f'Exported knowledge-group metrics to {output_path}')


if __name__ == '__main__':
    main()
