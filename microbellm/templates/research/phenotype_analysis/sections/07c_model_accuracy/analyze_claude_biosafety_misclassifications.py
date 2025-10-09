#!/usr/bin/env python3
"""Report biosafety predictions vs ground truth for Claude-3.5-Sonnet."""

import json
import urllib.request
from collections import Counter
from typing import Dict, Iterable, List, Tuple

API_URL = "http://localhost:5050"
TARGET_MODEL = "anthropic/claude-3.5-sonnet"
SPECIES_FILE = "wa_with_gcount.txt"
GROUND_TRUTH_DATASET = "WA_Test_Dataset"


def fetch_api_data(endpoint: str, api_url: str = API_URL) -> Dict:
    """Fetch JSON payload from the local MicrobeLLM API."""
    full_url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urllib.request.urlopen(full_url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_biosafety(value: str) -> str | None:
    """Normalize biosafety values such as 'BSL-2' to 'biosafety level 2'."""
    if not value:
        return None

    value_lower = str(value).strip().lower()

    mapping = {
        "bsl-1": "biosafety level 1",
        "bsl1": "biosafety level 1",
        "level 1": "biosafety level 1",
        "biosafety 1": "biosafety level 1",
        "biosafety level 1": "biosafety level 1",
        "biosafety-level-1": "biosafety level 1",
        "bsl-2": "biosafety level 2",
        "bsl2": "biosafety level 2",
        "level 2": "biosafety level 2",
        "biosafety 2": "biosafety level 2",
        "biosafety level 2": "biosafety level 2",
        "biosafety-level-2": "biosafety level 2",
        "bsl-3": "biosafety level 3",
        "bsl3": "biosafety level 3",
        "level 3": "biosafety level 3",
        "biosafety 3": "biosafety level 3",
        "biosafety level 3": "biosafety level 3",
        "biosafety-level-3": "biosafety level 3",
    }

    return mapping.get(value_lower)


def build_ground_truth_map(records: Iterable[Dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (value map, canonical-name map) keyed by lowercase species name."""
    value_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}

    for item in records:
        name = item.get("binomial_name")
        value = normalize_biosafety(item.get("biosafety_level"))
        if not name or not value:
            continue
        key = name.lower()
        value_map[key] = value
        name_map[key] = name

    return value_map, name_map


def to_short_label(level: str | None) -> str:
    mapping = {
        "biosafety level 1": "BSL-1",
        "biosafety level 2": "BSL-2",
        "biosafety level 3": "BSL-3",
    }
    return mapping.get(level, "-")


def main() -> None:
    print("=" * 100)
    print("CLAUDE-3.5-SONNET BIOSAFETY: FULL COMPARISON")
    print("=" * 100)
    print()
    print(f"Fetching predictions for {TARGET_MODEL!r}...")
    pred_payload = fetch_api_data(
        f"/api/phenotype_analysis_filtered?species_file={SPECIES_FILE}"
    )
    predictions = pred_payload.get("data", [])

    print("Fetching ground truth biosafety levels...")
    gt_payload = fetch_api_data(
        f"/api/ground_truth/data?dataset={GROUND_TRUTH_DATASET}&per_page=20000"
    )
    ground_truth = gt_payload.get("data", [])
    gt_map, name_map = build_ground_truth_map(ground_truth)
    print(f"Loaded {len(gt_map)} ground-truth species with biosafety labels.")

    claude_predictions: Dict[str, str | None] = {}
    for record in predictions:
        if record.get("model") != TARGET_MODEL:
            continue
        species = record.get("binomial_name")
        if not species:
            continue
        claude_predictions[species.lower()] = normalize_biosafety(
            record.get("biosafety_level")
        )

    print()
    print("Building comparison table...")

    rows: List[Tuple[str, str | None, str | None, str]] = []
    mismatch_counter = Counter()
    status_counter = Counter()

    for species_lower, gt_level in gt_map.items():
        predicted = claude_predictions.get(species_lower)
        status: str
        if predicted is None:
            status = "missing"
        elif predicted == gt_level:
            status = "match"
        else:
            status = "mismatch"
            mismatch_counter[(gt_level, predicted)] += 1

        status_counter[status] += 1
        rows.append((name_map[species_lower], gt_level, predicted, status))

    rows.sort(key=lambda x: x[0].lower())

    print(f"Total ground-truth species compared: {len(rows)}")
    print("Status counts:")
    for status in ("match", "mismatch", "missing"):
        print(f"  {status}: {status_counter.get(status, 0)}")

    if mismatch_counter:
        print("\nMismatches by ground-truth → predicted:")
        for (gt_level, predicted), count in sorted(
            mismatch_counter.items(), key=lambda item: item[1], reverse=True
        ):
            print(
                f"  {to_short_label(gt_level)} → {to_short_label(predicted)}: {count}"
            )

    print("\nFULL TABLE (species, GT, Claude, status):")
    header = f"{'Species':<40} | {'GT':^7} | {'Claude':^7} | {'Status':^9}"
    print(header)
    print("-" * len(header))

    for species_name, gt_level, predicted, status in rows:
        print(
            f"{species_name:<40} | {to_short_label(gt_level):^7} | {to_short_label(predicted):^7} | {status:^9}"
        )

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
