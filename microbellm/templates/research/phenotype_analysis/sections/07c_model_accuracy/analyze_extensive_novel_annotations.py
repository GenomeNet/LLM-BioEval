#!/usr/bin/env python3
"""Summarize and visualize LA predictions with extensive knowledge but missing ground truth."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt

PHENOTYPE_ORDER: Iterable[str] = (
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

DISPLAY_NAMES = {
    'gram_staining': 'Gram Staining',
    'motility': 'Motility',
    'extreme_environment_tolerance': 'Extreme Env. Tol.',
    'biofilm_formation': 'Biofilm Formation',
    'animal_pathogenicity': 'Animal Path.',
    'biosafety_level': 'Biosafety Level',
    'host_association': 'Host Association',
    'plant_pathogenicity': 'Plant Path.',
    'spore_formation': 'Spore Formation',
    'cell_shape': 'Cell Shape',
}


def load_records(csv_path: Path):
    with csv_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def analyse(csv_path: Path, model_filter: str | None = None):
    total = 0
    species_set = set()
    counts_by_pheno: Dict[str, int] = defaultdict(int)
    counts_by_model: Dict[str, int] = defaultdict(int)
    counts_model_pheno: Dict[tuple[str, str], int] = defaultdict(int)
    counts_by_pheno_label: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in load_records(csv_path):
        if (row.get('knowledge_group', '').lower() != 'extensive'):
            continue

        truth = (row.get('ground_truth') or '').strip()
        pred = (row.get('prediction') or '').strip()
        if truth:
            continue
        if not pred:
            continue

        phenotype = row.get('phenotype', '')
        model = row.get('model', '')
        if not phenotype:
            continue
        if model_filter and model != model_filter:
            continue

        total += 1
        species = (row.get('binomial_name') or '').strip()
        if species:
            species_set.add(species)

        counts_by_pheno[phenotype] += 1
        counts_by_model[model] += 1
        counts_model_pheno[(model, phenotype)] += 1
        counts_by_pheno_label[phenotype][pred.lower()] += 1

    return {
        'total': total,
        'species': species_set,
        'counts_by_pheno': counts_by_pheno,
        'counts_by_model': counts_by_model,
        'counts_model_pheno': counts_model_pheno,
        'counts_by_pheno_label': counts_by_pheno_label,
    }


def canonical_label(label: str) -> str:
    key = label.lower()
    mapping = {
        'true': 'True',
        'false': 'False',
        'positive': 'Positive',
        'negative': 'Negative',
        'variable': 'Variable',
        'level_1': 'Level 1',
        'level_2': 'Level 2',
        'level_3': 'Level 3',
    }
    if key in mapping:
        return mapping[key]
    return label.title()


def plot_distribution(results: dict, output_path: Path, pdf_path: Path | None = None):
    counts_by_pheno = results['counts_by_pheno']
    counts_by_pheno_label = results['counts_by_pheno_label']

    phenos = sorted(
        (p for p in PHENOTYPE_ORDER if counts_by_pheno.get(p)),
        key=lambda p: counts_by_pheno[p],
        reverse=True
    )
    if not phenos:
        print('No extensive-without-truth records to plot.')
        return

    label_set = set()
    for pheno in phenos:
        label_set.update(counts_by_pheno_label[pheno].keys())

    labels = sorted(label_set)
    if not labels:
        print('No predicted labels to plot.')
        return

    plt.figure(figsize=(8, max(3, 0.5 * len(phenos))))
    bottom = [0] * len(phenos)

    for label in labels:
        values = [counts_by_pheno_label[pheno].get(label, 0) for pheno in phenos]
        if sum(values) == 0:
            continue
        plt.barh(phenos, values, left=bottom, label=canonical_label(label))
        bottom = [b + v for b, v in zip(bottom, values)]

    plt.xlabel('Predictions (knowledge=extensive, ground truth missing)')
    plt.ylabel('Phenotype')
    plt.yticks(list(phenos), [DISPLAY_NAMES.get(p, p.replace('_', ' ').title()) for p in phenos])
    plt.title('Extensive-knowledge predictions lacking ground truth (LA_Test_Dataset)')
    if len(labels) <= 12:
        plt.legend(loc='best', fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if pdf_path:
        plt.savefig(pdf_path)
    plt.close()


def write_summary(results: dict, output_path: Path, model_filter: str | None = None):
    total = results['total']
    species_count = len(results['species'])
    counts_by_pheno = results['counts_by_pheno']
    counts_by_model = results['counts_by_model']
    counts_by_pheno_label = results['counts_by_pheno_label']

    with output_path.open('w', encoding='utf-8') as fh:
        if model_filter:
            fh.write(f'Extensive knowledge predictions without ground truth (LA_Test_Dataset, model={model_filter})\n')
        else:
            fh.write('Extensive knowledge predictions without ground truth (LA_Test_Dataset)\n')
        fh.write(f'Total predictions: {total}\n')
        fh.write(f'Unique species: {species_count}\n')
        fh.write('\nCounts by phenotype:\n')
        for phenotype, count in sorted(counts_by_pheno.items(), key=lambda x: (-x[1], x[0])):
            if not count:
                continue
            label = DISPLAY_NAMES.get(phenotype, phenotype.replace('_', ' ').title())
            fh.write(f'  {label}: {count}\n')
            label_counts = counts_by_pheno_label.get(phenotype, {})
            for pred_label, pred_count in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
                display = canonical_label(pred_label)
                fh.write(f'    - {display}: {pred_count}\n')
        fh.write('\nCounts by model:\n')
        for model, count in sorted(counts_by_model.items(), key=lambda x: (-x[1], x[0])):
            fh.write(f'  {model}: {count}\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyse LA predictions with extensive knowledge group and missing ground-truth labels.'
    )
    parser.add_argument(
        '--input',
        default='microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/knowledge_accuracy_la.csv',
        help='Path to the per-observation export CSV.'
    )
    parser.add_argument(
        '--plot',
        default='microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/extensive_novel_predictions_la.png',
        help='Output path for the bar-chart PNG.'
    )
    parser.add_argument(
        '--plot-pdf',
        default=None,
        help='Optional output path for a PDF copy of the plot.'
    )
    parser.add_argument(
        '--summary',
        default='microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/extensive_novel_predictions_la.txt',
        help='Output path for the text summary.'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Optional model filter (e.g., x-ai/grok-3-mini).'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'{input_path} does not exist. Export knowledge_accuracy_la.csv first.')

    results = analyse(input_path, model_filter=args.model)
    print(f"Found {results['total']} predictions across {len(results['species'])} species.")

    pdf_path = Path(args.plot_pdf) if args.plot_pdf else None
    plot_distribution(results, Path(args.plot), pdf_path=pdf_path)
    write_summary(results, Path(args.summary), model_filter=args.model)
    print('Outputs written:')
    print('  Plot ->', args.plot)
    print('  Summary ->', args.summary)


if __name__ == '__main__':
    main()
