#!/usr/bin/env python3
"""Summarize annotation completeness for key phenotypes in the LA test dataset."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable

PHENOTYPES: Iterable[str] = (
    'spore_formation',
    'cell_shape',
    'biosafety_level',
    'motility',
    'animal_pathogenicity',
    'host_association',
    'plant_pathogenicity',
    'extreme_environment_tolerance',
    'gram_staining',
    'biofilm_formation',
)

DATASET_NAME = 'LA_Test_Dataset'
DB_PATH = Path('microbellm.db')


def fetch_missing_stats() -> Dict[str, Dict[str, float]]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM ground_truth WHERE dataset_name = ?",
            (DATASET_NAME,),
        )
        (total_species,) = cursor.fetchone()

        results: Dict[str, Dict[str, float]] = {}
        for phenotype in PHENOTYPES:
            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM ground_truth
                WHERE dataset_name = ?
                  AND {phenotype} IS NOT NULL
                  AND TRIM({phenotype}) != ''
                """,
                (DATASET_NAME,),
            )
            (non_missing,) = cursor.fetchone()
            missing = total_species - non_missing
            results[phenotype] = {
                'total_species': total_species,
                'non_missing': non_missing,
                'missing': missing,
                'missing_pct': (missing / total_species) * 100,
            }

    return results


def main() -> None:
    stats = fetch_missing_stats()

    header = (
        f"Annotation completeness for {DATASET_NAME} (n={next(iter(stats.values()))['total_species']:,} species)"
    )
    print(header)
    print('-' * len(header))
    print(f"{'Phenotype':30s} {'Labeled':>8s} {'Missing':>8s} {'Missing %':>10s}")

    for phenotype in PHENOTYPES:
        values = stats[phenotype]
        label = phenotype.replace('_', ' ').title()
        print(
            f"{label:30s} "
            f"{int(values['non_missing']):8d} "
            f"{int(values['missing']):8d} "
            f"{values['missing_pct']:10.2f}"
        )


if __name__ == '__main__':
    main()
