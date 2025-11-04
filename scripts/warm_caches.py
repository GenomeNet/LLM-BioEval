#!/usr/bin/env python
"""Precompute ground-truth and model accuracy caches.

Run this on a machine that has write access to the SQLite database.  It will
populate cache tables so production (possibly read-only) deployments can serve
cached data immediately.
"""

import argparse
import sys
import time

from microbellm.utils import get_ground_truth_datasets
from microbellm.web_app import (
    _get_ground_truth_dataset_import_metadata,
    _calculate_ground_truth_statistics,
    _update_ground_truth_stats_cache,
    _calculate_model_accuracy_metrics,
    _update_model_accuracy_cache,
    _calculate_performance_by_year_metrics,
    _update_performance_year_cache,
    _calculate_knowledge_accuracy_metrics,
    _update_knowledge_accuracy_cache,
)


def _timestamp_from_metadata(result: dict) -> float:
    metadata = (result or {}).get("metadata", {})
    return float(metadata.get("import_timestamp", 0))


def warm_dataset(dataset_name: str) -> None:
    print(f"Warming caches for dataset: {dataset_name}")

    metadata = _get_ground_truth_dataset_import_metadata(dataset_name)

    stats = _calculate_ground_truth_statistics(dataset_name, metadata=metadata)
    _update_ground_truth_stats_cache(
        dataset_name,
        stats,
        _timestamp_from_metadata(stats),
        computed_at=time.time(),
    )
    print("  ✓ Ground truth statistics cached")

    model_metrics = _calculate_model_accuracy_metrics(dataset_name, metadata=metadata)
    _update_model_accuracy_cache(
        dataset_name,
        model_metrics,
        _timestamp_from_metadata(model_metrics),
        computed_at=time.time(),
    )
    print("  ✓ Model accuracy cached")

    perf_by_year = _calculate_performance_by_year_metrics(dataset_name, metadata=metadata)
    _update_performance_year_cache(
        dataset_name,
        perf_by_year,
        _timestamp_from_metadata(perf_by_year),
        computed_at=time.time(),
    )
    print("  ✓ Performance-by-year cached")

    knowledge_metrics = _calculate_knowledge_accuracy_metrics(dataset_name, metadata=metadata)
    _update_knowledge_accuracy_cache(
        dataset_name,
        knowledge_metrics,
        _timestamp_from_metadata(knowledge_metrics),
        computed_at=time.time(),
    )
    print("  ✓ Knowledge metrics cached")


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm cached analytics tables")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Specific dataset names to warm (default: all available)",
    )
    args = parser.parse_args()

    available = [entry["dataset_name"] for entry in get_ground_truth_datasets()]
    targets = args.datasets or available

    missing = sorted(set(targets) - set(available))
    if missing:
        print(f"Error: unknown dataset(s): {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    for dataset in targets:
        warm_dataset(dataset)


if __name__ == "__main__":
    main()
