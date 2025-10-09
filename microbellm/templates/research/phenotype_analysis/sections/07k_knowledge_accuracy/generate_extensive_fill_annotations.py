#!/usr/bin/env python3
"""Generate fill-in annotations for WA species using extensive-knowledge models."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


API_URL_DEFAULT = "http://localhost:5050"
DATASET_DEFAULT = "WA_Test_Dataset"
PAIRINGS_PATH_DEFAULT = (
    "microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/"
    "attic/extensive_fill_candidates.json"
)
MISSING_PATH_DEFAULT = (
    "microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/"
    "attic/wa_missing_by_phenotype.json"
)
MISSING_TOKENS = {"n/a", "na", "null", "none", "nan", "undefined", "-", "unknown", "missing", ""}


OUTPUT_CSV_DEFAULT = (
    "microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/"
    "attic/extensive_fill_annotations.csv"
)
MISSING_REPORT_DEFAULT = (
    "microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/"
    "attic/extensive_fill_missing_predictions.json"
)


def load_trends_module():
    script_path = Path(__file__).with_name("generate_knowledge_accuracy_trends.py")
    spec = importlib.util.spec_from_file_location("knowledge_trends", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["knowledge_trends"] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def build_lookup_by_model(knowledge_data: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    lookup: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for item in knowledge_data:
        group = (item.get("knowledge_group") or "").lower()
        if group != "extensive":
            continue
        model = item.get("model")
        species = (item.get("binomial_name") or "").strip().lower()
        if not model or not species:
            continue
        lookup[model][species] = item
    return lookup


def load_json(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "species",
                "phenotype",
                "phenotype_label",
                "model_id",
                "model_display",
                "knowledge_group",
                "prediction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_missing_report(missing: Dict[str, List[str]], path: Path) -> None:
    if not missing:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # sort for stability
    serialisable = {key: sorted(values) for key, values in sorted(missing.items())}
    path.write_text(json.dumps(serialisable, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL_DEFAULT)
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--pairings", default=PAIRINGS_PATH_DEFAULT)
    parser.add_argument("--missing-species", default=MISSING_PATH_DEFAULT)
    parser.add_argument("--output-csv", default=OUTPUT_CSV_DEFAULT)
    parser.add_argument("--missing-report", default=MISSING_REPORT_DEFAULT)
    args = parser.parse_args()

    trends = load_trends_module()

    print("Fetching knowledge-stratified predictions...")
    knowledge_data = trends.load_knowledge_data(args.api_url, args.dataset)
    lookup = build_lookup_by_model(knowledge_data)
    print(f"  Loaded {len(knowledge_data):,} knowledge-tagged predictions")

    pairings = load_json(Path(args.pairings))
    missing_species = load_json(Path(args.missing_species))

    rows: List[Dict[str, str]] = []
    missing_predictions: Dict[str, List[str]] = defaultdict(list)

    for entry in pairings:
        phenotype = entry["phenotype"]
        label = entry["label"]
        model_id = entry["model_id"]
        species_list: List[str] = missing_species.get(phenotype, [])
        model_lookup = lookup.get(model_id, {})

        for species in species_list:
            species_key = species.lower()
            record = model_lookup.get(species_key)
            if record is None:
                missing_predictions[phenotype].append(species)
                continue
            prediction = record.get("predictions", {}).get(phenotype)
            pred_text = (str(prediction).strip().lower() if prediction is not None else "")
            if pred_text in MISSING_TOKENS:
                missing_predictions[phenotype].append(species)
                continue
            rows.append(
                {
                    "species": species,
                    "phenotype": phenotype,
                    "phenotype_label": label,
                    "model_id": model_id,
                    "model_display": entry["model_display"],
                    "knowledge_group": record.get("knowledge_group"),
                    "prediction": prediction,
                }
            )

    write_csv(rows, Path(args.output_csv))
    write_missing_report(missing_predictions, Path(args.missing_report))

    species_count = len({row["species"] for row in rows})
    count_by_field = Counter(row["phenotype"] for row in rows)

    print(
        f"Wrote {len(rows)} annotations covering {species_count} unique species to {args.output_csv}."
    )
    if missing_predictions:
        missing_total = sum(len(v) for v in missing_predictions.values())
        print(
            f"Missing extensive predictions for {missing_total} species/phenotype pairs; "
            f"see {args.missing_report}."
        )
    else:
        print("No missing predictions for selected pairings.")

    print("Breakdown by phenotype (filled assignments):")
    for phenotype, count in sorted(count_by_field.items()):
        print(f"  {phenotype}: {count}")


if __name__ == "__main__":
    main()
