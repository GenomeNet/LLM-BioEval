#!/usr/bin/env python3
"""Compute overall hallucination benchmark summary stats from model summary CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

def compute_totals(csv_path: Path) -> Tuple[int, int, int, int, int, float, float, float]:
    total_samples = 0
    total_na = 0
    limited = 0
    moderate = 0
    extensive = 0
    no_result = 0
    inf_failed = 0

    with csv_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples = int(row['Samples_Evaluated'])
            total_samples += samples
            total_na += int(row['NA_Count'])
            limited += int(row['Limited_Count'])
            moderate += int(row['Moderate_Count'])
            extensive += int(row['Extensive_Count'])
            no_result += int(row['NoResult_Count'])
            inf_failed += int(row['InferenceFailed_Count'])

    exposures = total_samples  # each row covers 200 samples
    correct = total_na
    failures = inf_failed + no_result
    hallucinations = limited + moderate + extensive

    correct_rate = correct / exposures * 100 if exposures else 0.0
    failure_rate = failures / exposures * 100 if exposures else 0.0
    halluc_rate = hallucinations / exposures * 100 if exposures else 0.0

    return exposures, correct, failures, hallucinations, limited + moderate + extensive, correct_rate, failure_rate, halluc_rate

def main():
    parser = argparse.ArgumentParser(description='Compute aggregate hallucination stats')
    parser.add_argument('--csv', default='microbellm/templates/research/knowledge_calibration/sections/06b_search_count_by_knowledge/model_quality_scores_table.csv',
                        help='Path to per-model summary CSV')
    args = parser.parse_args()

    exposures, correct, failures, hallucinations, _, correct_rate, failure_rate, halluc_rate = compute_totals(Path(args.csv))

    print(f"Total model-sample predictions: {exposures:,}")
    print(f"Correct rejections (NA): {correct:,} ({correct_rate:.1f}%)")
    print(f"Inference failures / no result: {failures:,} ({failure_rate:.1f}%)")
    print(f"Hallucinations (limited/moderate/extensive): {hallucinations:,} ({halluc_rate:.1f}%)")

if __name__ == '__main__':
    main()
