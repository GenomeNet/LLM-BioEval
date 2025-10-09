#!/usr/bin/env python3
"""Export phenotype predictions with knowledge-group metadata for supplementary material."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DB_PATH = Path('microbellm.db')

DATASET_SPECIES_FILE = {
    'WA_Test_Dataset': 'wa_with_gcount.txt',
    'LA_Test_Dataset': 'la.txt',
}

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

MISSING_TOKENS = {
    'n/a', 'na', 'null', 'none', 'nan', 'undefined', '-', 'unknown', 'missing', ''
}


def resolve_species_file(dataset_name: str) -> str:
    if dataset_name in DATASET_SPECIES_FILE:
        return DATASET_SPECIES_FILE[dataset_name]
    lowered = dataset_name.lower()
    if 'la_test' in lowered:
        return 'la.txt'
    if 'wa_test' in lowered:
        return 'wa_with_gcount.txt'
    if 'artificial' in lowered:
        return 'artificial.txt'
    raise ValueError(f'Unknown dataset name: {dataset_name}')


def normalize_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    lowered = s.lower()
    if lowered in MISSING_TOKENS:
        return None
    return s


def fetch_ground_truth(conn: sqlite3.Connection, dataset_name: str) -> Dict[str, Dict[str, Optional[str]]]:
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT LOWER(binomial_name) AS species,
               {', '.join(PHENOTYPES)}
        FROM ground_truth
        WHERE dataset_name = ?
        """,
        (dataset_name,),
    )

    truth: Dict[str, Dict[str, Optional[str]]] = {}
    for row in cursor.fetchall():
        species = row[0]
        truth[species] = {
            phenotype: normalize_value(row[idx + 1])
            for idx, phenotype in enumerate(PHENOTYPES)
        }
    return truth


def fetch_predictions_with_knowledge(
    conn: sqlite3.Connection,
    species_file: str,
    model: Optional[str] = None,
) -> List[sqlite3.Row]:
    cursor = conn.cursor()

    query = f"""
        SELECT
            pheno.model,
            pheno.binomial_name,
            know.knowledge_group,
            {', '.join(f'pheno.{p}' for p in PHENOTYPES)}
        FROM processing_results pheno
        JOIN processing_results know
          ON pheno.model = know.model
         AND pheno.binomial_name = know.binomial_name
         AND pheno.species_file = know.species_file
        WHERE pheno.species_file = ?
          AND pheno.user_template LIKE '%template1_phenotype%'
          AND know.user_template LIKE '%template3_knowlege%'
          AND know.knowledge_group IS NOT NULL
          {"AND pheno.model = ?" if model else ''}
        ORDER BY pheno.model, know.knowledge_group, pheno.binomial_name
    """

    params: List[str] = [species_file]
    if model:
        params.append(model)

    cursor.execute(query, params)
    return cursor.fetchall()


def export_dataset(
    dataset_name: str,
    output_path: Path,
    model: Optional[str] = None,
) -> None:
    species_file = resolve_species_file(dataset_name)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    truth = fetch_ground_truth(conn, dataset_name)
    rows = fetch_predictions_with_knowledge(conn, species_file, model=model)

    conn.close()

    with output_path.open('w', newline='', encoding='utf-8') as fh:
        fieldnames = [
            'dataset',
            'species_file',
            'model',
            'knowledge_group',
            'binomial_name',
            'phenotype',
            'ground_truth',
            'prediction',
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            model_name = row['model']
            binomial_name = row['binomial_name']
            knowledge_group = (row['knowledge_group'] or '').lower()
            species_key = (binomial_name or '').lower()
            truth_row = truth.get(species_key)
            if not truth_row:
                continue

            for phenotype in PHENOTYPES:
                pred_val = normalize_value(row[phenotype])
                truth_val = truth_row.get(phenotype)
                if pred_val is None and truth_val is None:
                    continue
                writer.writerow({
                    'dataset': dataset_name,
                    'species_file': species_file,
                    'model': model_name,
                    'knowledge_group': knowledge_group,
                    'binomial_name': binomial_name,
                    'phenotype': phenotype,
                    'ground_truth': truth_val if truth_val is not None else '',
                    'prediction': pred_val if pred_val is not None else '',
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export phenotype predictions grouped by knowledge level for supplementary data.'
    )
    parser.add_argument(
        '--dataset',
        choices=sorted(DATASET_SPECIES_FILE.keys()),
        default='WA_Test_Dataset',
        help='Ground-truth dataset to export.'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV filename (default: knowledge_accuracy_<dataset>.csv)'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Optional model identifier to filter (default: export all models).'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else Path(
        f'microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy/knowledge_accuracy_{args.dataset.lower()}.csv'
    )
    export_dataset(args.dataset, output, model=args.model)
    print(f'Exported knowledge-grouped phenotype predictions to {output}')


if __name__ == '__main__':
    main()

