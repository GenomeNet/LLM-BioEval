#!/usr/bin/env python3
"""Create a combined plot of extensive-knowledge fill-ins for WA and LA datasets."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['svg.fonttype'] = 'none'


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

BOOLEAN_TRUE = {'true', 't', '1', 'yes', 'y'}
BOOLEAN_FALSE = {'false', 'f', '0', 'no', 'n'}


def canonical_prediction(text: str) -> str:
    value = text.strip().lower()
    if value in BOOLEAN_TRUE:
        return 'Positive'
    if value in BOOLEAN_FALSE:
        return 'Negative'
    if value.startswith('gram stain '):
        return value.replace('gram stain ', 'Gram ').title()
    if value.startswith('gram '):
        return value.replace('gram ', 'Gram ').title()
    if value.startswith('biosafety level'):
        return value.replace('biosafety ', '').upper()
    if value.startswith('level '):
        return value.replace('level ', 'Level ').title()
    if value.startswith('level_'):
        return value.replace('level_', 'Level ').title()
    if not value:
        return 'Unknown'
    return text.strip().title()


def load_wa_counts(path: Path) -> Dict[str, Dict[str, Counter[str]]]:
    totals: Counter[str] = Counter()
    by_label: Dict[str, Counter[str]] = defaultdict(Counter)

    with path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get('knowledge_group', '').lower() != 'extensive'):
                continue
            phenotype = row.get('phenotype')
            if phenotype not in PHENOTYPE_ORDER:
                continue
            pred_label = canonical_prediction(row.get('prediction', ''))
            totals[phenotype] += 1
            by_label[phenotype][pred_label] += 1

    return {'totals': totals, 'by_label': by_label}


def load_la_counts(path: Path, model_filter: str) -> Dict[str, Dict[str, Counter[str]]]:
    totals: Counter[str] = Counter()
    by_label: Dict[str, Counter[str]] = defaultdict(Counter)

    with path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get('model') != model_filter:
                continue
            if (row.get('knowledge_group', '').lower() != 'extensive'):
                continue
            if row.get('ground_truth', '').strip():
                continue
            prediction = row.get('prediction', '').strip()
            if not prediction:
                continue
            phenotype = row.get('phenotype')
            if phenotype not in PHENOTYPE_ORDER:
                continue

            pred_label = canonical_prediction(prediction)
            totals[phenotype] += 1
            by_label[phenotype][pred_label] += 1

    return {'totals': totals, 'by_label': by_label}


def determine_order(data_by_dataset: Dict[str, dict]) -> list[str]:
    if 'LA' in data_by_dataset:
        totals = data_by_dataset['LA']['totals']
    else:
        totals = Counter()
        for dataset in data_by_dataset.values():
            for phenotype, count in dataset['totals'].items():
                totals[phenotype] += count

    ordered = sorted(
        (p for p in PHENOTYPE_ORDER if totals.get(p)),
        key=lambda p: totals.get(p, 0),
        reverse=True
    )

    return ordered


def plot_combined(
    data_by_dataset: Dict[str, dict],
    phenotypes: list[str],
    labels: list[str],
    colors: Dict[str, tuple[float, float, float, float]],
    output_png: Path,
    output_pdf: Path | None,
    output_svg: Path | None,
) -> None:
    dataset_order = [key for key in ('LA', 'WA') if key in data_by_dataset]
    if not dataset_order:
        raise RuntimeError('No dataset provided for plotting.')

    base_width = 8.0 * 1.1
    base_panel_height = max(3.5, 0.6 * len(phenotypes))

    phenos_per_dataset: Dict[str, list[str]] = {}
    for key in dataset_order:
        dataset = data_by_dataset[key]
        phenos_ds = [p for p in phenotypes if dataset['totals'].get(p)]
        phenos_per_dataset[key] = phenos_ds

    fig, axes = plt.subplots(
        nrows=len(dataset_order),
        ncols=1,
        figsize=(base_width, base_panel_height * len(dataset_order)),
        sharex=False,
        sharey=False
    )

    if len(dataset_order) == 1:
        axes = [axes]

    for ax, dataset_key in zip(axes, dataset_order):
        dataset = data_by_dataset[dataset_key]
        totals = dataset['totals']
        by_label = dataset['by_label']
        phenos_ds = phenos_per_dataset.get(dataset_key, [])
        positions = [phenotypes.index(p) for p in phenos_ds]
        if not phenos_ds:
            ax.set_visible(False)
            continue

        bottom = [0] * len(phenos_ds)
        for label in labels:
            values = [by_label.get(ph, {}).get(label, 0) for ph in phenos_ds]
            if sum(values) == 0:
                continue
            bars = ax.barh(
                positions,
                values,
                left=bottom,
                color=colors[label],
                edgecolor='none',
                height=0.44,
                label=label
            )
            for patch in bars:
                patch.set_clip_on(False)
                patch.set_clip_path(None)
            bottom = [b + v for b, v in zip(bottom, values)]

        title = 'LA: Extensive predictions without ground truth' if dataset_key == 'LA' else 'WA: Extensive fill-in annotations'
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Number of annotations')
        ax.grid(False)
        ax.invert_yaxis()

    for ax, dataset_key in zip(axes, dataset_order):
        phenos_ds = phenos_per_dataset.get(dataset_key, [])
        if not phenos_ds or not ax.get_visible():
            continue
        positions = [phenotypes.index(p) for p in phenos_ds]
        positions_set = set(positions)
        ax.set_yticks(range(len(phenotypes)))
        ax.set_yticklabels([
            DISPLAY_NAMES.get(p, p.replace('_', ' ').title()) if idx in positions_set else ''
            for idx, p in enumerate(phenotypes)
        ])
        ax.set_ylim(-0.5, len(phenotypes) - 0.5)
        ax.tick_params(axis='y', labelsize=9)
        ax.set_ylabel('Phenotype', labelpad=12)
        # Avoid axis patch clip masks in vector export
        ax.set_facecolor('white')
        ax.patch.set_alpha(0)
        ax.set_clip_on(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    labels_clean = labels

    fig.subplots_adjust(left=0.24, right=0.74, top=0.96, bottom=0.04, hspace=0.35)
    fig.legend(
        handles,
        labels_clean,
        title='Predicted label',
        fontsize=8,
        title_fontsize=9,
        loc='center left',
        bbox_to_anchor=(0.82, 0.5)
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    if output_pdf:
        fig.savefig(output_pdf)
    if output_svg:
        fig.savefig(output_svg)
    plt.close(fig)


def write_summary(data_by_dataset: Dict[str, dict], phenotypes: list[str], output_path: Path) -> None:
    with output_path.open('w', encoding='utf-8') as fh:
        for key in ('LA', 'WA'):
            if key not in data_by_dataset:
                continue
            dataset = data_by_dataset[key]
            totals = dataset['totals']
            by_label = dataset['by_label']
            label = 'LA_Test_Dataset (grok-3-mini)' if key == 'LA' else 'WA_Test_Dataset'
            fh.write(f'{label}\n')
            fh.write(f'  Total annotations: {sum(totals.values())}\n\n')
            for phenotype in phenotypes:
                count = totals.get(phenotype)
                if not count:
                    continue
                display = DISPLAY_NAMES.get(phenotype, phenotype.replace('_', ' ').title())
                fh.write(f'  {display}: {count}\n')
                for label_name, label_count in sorted(by_label.get(phenotype, {}).items(), key=lambda x: (-x[1], x[0])):
                    fh.write(f'    - {label_name}: {label_count}\n')
            fh.write('\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--wa-input',
        default='microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/extensive_fill_annotations.csv',
        help='CSV of WA extensive fill-in annotations.'
    )
    parser.add_argument(
        '--la-input',
        default='microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/knowledge_accuracy_la.csv',
        help='CSV of LA per-annotation predictions.'
    )
    parser.add_argument(
        '--la-model',
        default='x-ai/grok-3-mini',
        help='Model name to filter in the LA dataset (default: x-ai/grok-3-mini).'
    )
    parser.add_argument(
        '--output-png',
        default='microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/extensive_fill_distribution_combined.png',
        help='Output PNG path.'
    )
    parser.add_argument(
        '--output-pdf',
        default='microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/extensive_fill_distribution_combined.pdf',
        help='Output PDF path.'
    )
    parser.add_argument(
        '--output-svg',
        default='microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/extensive_fill_distribution_combined.svg',
        help='Output SVG path (set to empty string to skip).'
    )
    parser.add_argument(
        '--summary',
        default='microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/extensive_fill_distribution_summary.txt',
        help='Text summary output path.'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_by_dataset: Dict[str, dict] = {}

    wa_path = Path(args.wa_input)
    if wa_path.exists():
        data_by_dataset['WA'] = load_wa_counts(wa_path)
    else:
        print(f'Warning: WA input not found at {wa_path}, skipping.')

    la_path = Path(args.la_input)
    if la_path.exists():
        la_data = load_la_counts(la_path, args.la_model)
        if la_data['totals']:
            data_by_dataset['LA'] = la_data
    else:
        print(f'Warning: LA input not found at {la_path}, skipping.')

    if not data_by_dataset:
        raise RuntimeError('No datasets available to plot.')

    phenotypes = determine_order(data_by_dataset)
    label_set = set()
    for dataset in data_by_dataset.values():
        for label_counts in dataset['by_label'].values():
            label_set.update(label_counts.keys())
    labels = sorted(label_set)

    cmap = plt.get_cmap('tab20')
    colors = {label: cmap(i % cmap.N) for i, label in enumerate(labels)}

    plot_combined(
        data_by_dataset,
        phenotypes,
        labels,
        colors,
        output_png=Path(args.output_png),
        output_pdf=Path(args.output_pdf) if args.output_pdf else None,
        output_svg=Path(args.output_svg) if args.output_svg else None,
    )

    write_summary(data_by_dataset, phenotypes, Path(args.summary))

    print('Combined plot generated:')
    print('  PNG ->', args.output_png)
    print('  PDF ->', args.output_pdf)
    if args.output_svg:
        print('  SVG ->', args.output_svg)
    print('  Summary ->', args.summary)


if __name__ == '__main__':
    main()
