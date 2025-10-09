#!/usr/bin/env python3
"""Recreate the baseline vs. extensive knowledge accuracy comparison plot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"


API_URL_DEFAULT = "http://localhost:5050"
DATASET_DEFAULT = "WA_Test_Dataset"
SPECIES_FILE_DEFAULT = "wa_with_gcount.txt"
OUTPUT_DEFAULT = (
    "microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/"
    "knowledge_accuracy_baseline_vs_extensive.pdf"
)


@dataclass
class AccuracySummary:
    phenotype: str
    label: str
    baseline_accuracy: float
    baseline_model: str
    baseline_display: str
    baseline_samples: int
    extensive_accuracy: float | None
    extensive_model: str | None
    extensive_display: str | None
    extensive_samples: int

    @property
    def delta(self) -> float | None:
        if self.extensive_accuracy is None:
            return None
        return self.extensive_accuracy - self.baseline_accuracy


def fetch_json(endpoint: str, base_url: str) -> Dict:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_trends_module():
    script_path = Path(__file__).with_name("generate_knowledge_accuracy_trends.py")
    spec = importlib.util.spec_from_file_location("knowledge_trends", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["knowledge_trends"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def compute_baseline_best(
    trends_module,
    api_url: str,
    dataset: str,
    species_file: str,
    min_samples: int,
) -> Dict[str, Dict[str, float | int | str]]:
    predictions = fetch_json(
        f"/api/phenotype_analysis_filtered?species_file={species_file}",
        api_url,
    ).get("data", [])

    ground_truth_records = fetch_json(
        f"/api/ground_truth/data?dataset={dataset}&per_page=50000",
        api_url,
    ).get("data", [])

    truth_map = {
        item.get("binomial_name", "").lower(): item for item in ground_truth_records
    }

    per_model_field: Dict[str, Dict[str, List[Dict[str, object]]]] = {}

    for record in predictions:
        model = record.get("model")
        species = str(record.get("binomial_name", "")).lower()
        if not model or species not in truth_map:
            continue

        model_fields = per_model_field.setdefault(model, {})
        truth_entry = truth_map[species]

        for field in trends_module.PHENOTYPES:
            truth_value = trends_module.normalize_value(field, truth_entry.get(field))
            pred_value = trends_module.normalize_value(field, record.get(field))
            if truth_value is None or pred_value is None:
                continue
            model_fields.setdefault(field, []).append(
                {"truth": truth_value, "prediction": pred_value}
            )

    best_by_field: Dict[str, Dict[str, float | int | str]] = {}

    for field in trends_module.PHENOTYPES:
        best_accuracy = float("nan")
        best_model: str | None = None
        best_samples = 0
        observations_count = 0

        for model, fields in per_model_field.items():
            observations = fields.get(field, [])
            sample_count = len(observations)
            if sample_count < min_samples:
                continue
            accuracy = trends_module.balanced_accuracy(field, observations)
            if not np.isfinite(accuracy):
                continue
            accuracy *= 100.0
            if best_model is None or accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_samples = sample_count
                observations_count = sample_count

        if best_model is not None:
            best_by_field[field] = {
                "accuracy": best_accuracy,
                "model": best_model,
                "display": trends_module.format_model_name(best_model),
                "samples": observations_count,
            }

    return best_by_field


def compute_extensive_best(trends_module, api_url: str, dataset: str) -> Dict[str, Dict[str, float | int | str]]:
    knowledge_data = trends_module.load_knowledge_data(api_url, dataset)
    metadata = trends_module.load_model_metadata(
        Path("microbellm/static/data/year_size.tsv")
    )

    best_by_field: Dict[str, Dict[str, float | int | str]] = {}

    for field in trends_module.PHENOTYPES:
        points = trends_module.compute_points(
            knowledge_data,
            metadata,
            top_n=0,
            phenotypes=[field],
        )

        best_accuracy = float("nan")
        best_point = None

        for point in points:
            accuracy = point.accuracy.get("extensive", float("nan"))
            samples = point.sample_sizes.get("extensive", 0)
            if samples <= 0 or not np.isfinite(accuracy):
                continue
            if best_point is None or accuracy > best_accuracy:
                best_point = point
                best_accuracy = accuracy

        if best_point is not None:
            best_by_field[field] = {
                "accuracy": best_accuracy,
                "model": best_point.model,
                "display": best_point.display_name,
                "samples": best_point.sample_sizes.get("extensive", 0),
            }

    return best_by_field


def assemble_summaries(
    trends_module,
    baseline: Dict[str, Dict[str, float | int | str]],
    extensive: Dict[str, Dict[str, float | int | str]],
) -> List[AccuracySummary]:
    summaries: List[AccuracySummary] = []
    for field in trends_module.PHENOTYPES:
        if field == "biofilm_formation":
            continue
        label = trends_module.PHENOTYPE_LABELS.get(
            field, field.replace("_", " ").title()
        )
        baseline_info = baseline.get(field)
        if not baseline_info:
            continue
        extensive_info = extensive.get(field)
        summaries.append(
            AccuracySummary(
                phenotype=field,
                label=label,
                baseline_accuracy=float(baseline_info["accuracy"]),
                baseline_model=str(baseline_info["model"]),
                baseline_display=str(baseline_info["display"]),
                baseline_samples=int(baseline_info["samples"]),
                extensive_accuracy=None
                if extensive_info is None
                else float(extensive_info["accuracy"]),
                extensive_model=None
                if extensive_info is None
                else str(extensive_info["model"]),
                extensive_display=None
                if extensive_info is None
                else str(extensive_info["display"]),
                extensive_samples=0
                if extensive_info is None
                else int(extensive_info["samples"]),
            )
        )
    return summaries


def plot_summaries(summaries: List[AccuracySummary], output_path: Path) -> None:
    if not summaries:
        raise RuntimeError("No accuracy summaries available for plotting")

    ordered = sorted(
        summaries,
        key=lambda item: item.delta if item.delta is not None else float("-inf"),
    )

    y_positions = np.arange(len(ordered))
    bar_height = 0.34

    baseline_values = [item.baseline_accuracy for item in ordered]
    extensive_values = [
        item.extensive_accuracy if item.extensive_accuracy is not None else 0.0
        for item in ordered
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    baseline_bars = ax.barh(
        y_positions - bar_height / 2,
        baseline_values,
        height=bar_height,
        color="#B3BDC9",
        label="Baseline (all knowledge)",
    )

    has_extensive = False
    for idx, item in enumerate(ordered):
        if item.extensive_accuracy is None:
            continue
        has_extensive = True
        ax.barh(
            y_positions[idx] + bar_height / 2,
            item.extensive_accuracy,
            height=bar_height,
            color="#4C7EA9",
            label="Extensive knowledge" if not idx else None,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([item.label for item in ordered], fontsize=9)
    ax.set_xlabel("Balanced Accuracy (%)", fontsize=10)
    ax.set_xlim(0, max(baseline_values + extensive_values) + 10)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    if has_extensive:
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.legend([baseline_bars], ["Baseline (all knowledge)"] , loc="lower right", fontsize=8)

    for idx, item in enumerate(ordered):
        y = y_positions[idx]
        ax.text(
            item.baseline_accuracy - 1.5,
            y - bar_height / 2,
            f"{item.baseline_display}\n(n={item.baseline_samples:,})",
            ha="right",
            va="center",
            fontsize=6.5,
            color="#444444",
        )

        if item.extensive_accuracy is None:
            ax.text(
                item.baseline_accuracy + 1.0,
                y + bar_height / 2,
                "n/a",
                ha="left",
                va="center",
                fontsize=7,
                color="#777777",
            )
        else:
            ax.text(
                item.extensive_accuracy + 1.0,
                y + bar_height / 2,
                f"{item.extensive_display}\n(n={item.extensive_samples:,})",
                ha="left",
                va="center",
                fontsize=6.5,
                color="#1F3B57",
            )

        delta = item.delta
        if delta is None:
            continue
        color = "#1B9E77" if delta >= 0 else "#D95F02"
        anchor = max(item.baseline_accuracy, item.extensive_accuracy or 0.0) + 2.0
        ax.text(
            anchor,
            y,
            f"Δ = {delta:+.2f} pts",
            ha="left",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )

    ax.set_title("Balanced Accuracy Improvement: Baseline vs Extensive", fontsize=12)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL_DEFAULT)
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--species-file", default=SPECIES_FILE_DEFAULT)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    try:
        trends_module = load_trends_module()

        print("Computing baseline best-performing models...")
        baseline_best = compute_baseline_best(
            trends_module,
            args.api_url,
            args.dataset,
            args.species_file,
            args.min_samples,
        )

        print("Computing extensive-knowledge best-performing models...")
        extensive_best = compute_extensive_best(trends_module, args.api_url, args.dataset)

        summaries = assemble_summaries(trends_module, baseline_best, extensive_best)
        if not summaries:
            raise RuntimeError("No accuracy summaries produced")

        print("Summary of baseline vs extensive performance:")
        for item in summaries:
            delta_text = "n/a" if item.delta is None else f"{item.delta:+.2f}"
            if item.extensive_accuracy is None:
                extensive_text = "n/a"
            else:
                extensive_text = (
                    f"{item.extensive_accuracy:.2f}% "
                    f"({item.extensive_display}, n={item.extensive_samples:,})"
                )
            print(
                f"  {item.label}: baseline {item.baseline_accuracy:.2f}% "
                f"({item.baseline_display}, n={item.baseline_samples:,}); "
                f"extensive {extensive_text} → Δ {delta_text}"
            )

        print(f"Saving plot to {args.output}...")
        plot_summaries(summaries, Path(args.output))
        print("Done.")

    except URLError as exc:
        raise SystemExit(f"Failed to reach API: {exc}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
