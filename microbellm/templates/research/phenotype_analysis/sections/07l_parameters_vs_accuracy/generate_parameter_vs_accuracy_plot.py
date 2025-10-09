#!/usr/bin/env python3
"""Generate parameter-count vs. average performance scatter plot.

This script mirrors the logic in the 07l parameters vs. accuracy component:
- Loads model metadata from `microbellm/static/data/year_size.tsv`
- Pulls predictions and ground-truth labels via the local MicrobeLLM API
- Computes balanced accuracy per phenotype per model (sample size ≥ 100)
- Aggregates an "all phenotypes (average)" score per model
- Produces a PDF scatter plot with model parameter count on log scale
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import numpy as np

API_URL = "http://localhost:5050"
DATASET_NAME = "WA_Test_Dataset"
SPECIES_FILE = "wa_with_gcount.txt"
MODEL_METADATA_PATH = Path("microbellm/static/data/year_size.tsv")
OUTPUT_PATH = Path(
    "microbellm/templates/research/phenotype_analysis/sections/07l_parameters_vs_accuracy/parameter_vs_accuracy_average.pdf"
)

AVERAGE_PHENOTYPES = (
    "gram_staining",
    "motility",
    "extreme_environment_tolerance",
    "biofilm_formation",
    "animal_pathogenicity",
    "biosafety_level",
    "host_association",
    "plant_pathogenicity",
    "spore_formation",
    "cell_shape",
)

CATEGORICAL_PHENOTYPES = {"cell_shape", "biosafety_level", "gram_staining"}
SAMPLE_SIZE_THRESHOLD = 100
MISSING_TOKENS = {"n/a", "na", "null", "none", "nan", "undefined", "-", "unknown", "missing"}


def fetch_json(endpoint: str) -> Dict:
    """Fetch JSON payload from the local API."""
    url = f"{API_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    with urlopen(url, timeout=60) as resp:
        return json_loads(resp.read())


def json_loads(data: bytes) -> Dict:
    import json

    return json.loads(data.decode("utf-8"))


def normalize_value(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s_lower = s.lower()
    if s_lower in MISSING_TOKENS:
        return None
    if "," in s or ";" in s:
        parts = [part.strip().lower() for part in s.replace(";", ",").split(",") if part.strip()]
        return ",".join(sorted(parts)) if parts else None
    return s_lower


def normalize_boolean(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "t", "y"}:
        return True
    if s in {"false", "0", "no", "f", "n"}:
        return False
    return None


def normalize_categorical(phenotype: str, value) -> str | None:
    normalized = normalize_value(value)
    if normalized is None:
        return None

    if phenotype == "gram_staining":
        if "positive" in normalized:
            return "gram stain positive"
        if "negative" in normalized:
            return "gram stain negative"
        if "variable" in normalized:
            return "gram stain variable"
    if phenotype == "biosafety_level":
        if "1" in normalized:
            return "biosafety level 1"
        if "2" in normalized:
            return "biosafety level 2"
        if "3" in normalized:
            return "biosafety level 3"
    return normalized


def compute_metrics_boolean(preds: List[bool], truths: List[bool]) -> Tuple[float, int]:
    tp = tn = fp = fn = 0
    for p, t in zip(preds, truths):
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif not p and not t:
            tn += 1
        else:
            fn += 1

    sample_size = tp + tn + fp + fn
    if sample_size == 0:
        return float("nan"), 0

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    return balanced_accuracy, sample_size


def compute_metrics_categorical(preds: List[str], truths: List[str]) -> Tuple[float, int]:
    labels = sorted({*preds, *truths})
    if not labels:
        return float("nan"), 0

    confusion = {label: Counter() for label in labels}
    for p, t in zip(preds, truths):
        confusion[t][p] += 1

    recalls = []
    for label in labels:
        tp = confusion[label][label]
        fn = sum(confusion[label][other] for other in labels if other != label)
        denom = tp + fn
        recalls.append(tp / denom if denom else 0.0)

    balanced_accuracy = sum(recalls) / len(labels)
    return balanced_accuracy, len(truths)


def calculate_metrics(
    predictions: Iterable[dict],
    ground_truth_map: Dict[str, dict],
    phenotypes: Iterable[str],
    sample_threshold: int = SAMPLE_SIZE_THRESHOLD,
) -> Dict[str, Dict[str, Tuple[float, int]]]:
    by_model: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(dict)
    by_model_predictions: Dict[str, List[dict]] = defaultdict(list)
    for pred in predictions:
        by_model_predictions[pred["model"]].append(pred)

    for model, rows in by_model_predictions.items():
        for phenotype in phenotypes:
            truths = []
            preds = []
            for row in rows:
                species = row.get("binomial_name")
                if not species:
                    continue
                gt = ground_truth_map.get(species.lower())
                if not gt:
                    continue

                if phenotype in CATEGORICAL_PHENOTYPES:
                    truth_val = normalize_categorical(phenotype, gt.get(phenotype))
                    pred_val = normalize_categorical(phenotype, row.get(phenotype))
                else:
                    truth_val = normalize_boolean(gt.get(phenotype))
                    pred_val = normalize_boolean(row.get(phenotype))

                if truth_val is None or pred_val is None:
                    continue

                truths.append(truth_val)
                preds.append(pred_val)

            if not truths:
                continue

            if phenotype in CATEGORICAL_PHENOTYPES:
                score, sample = compute_metrics_categorical(preds, truths)
            else:
                score, sample = compute_metrics_boolean(preds, truths)

            if sample >= sample_threshold and math.isfinite(score):
                by_model[model][phenotype] = (score, sample)

    return by_model


def load_model_metadata(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")

    metadata: Dict[str, dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    for row in rows:
        model_name = row.get("Model")
        if not model_name:
            continue
        variants = {
            model_name,
            model_name.lower(),
            model_name.lower().replace(" ", ""),
            model_name.lower().replace("-", "").replace("_", ""),
            model_name.replace("-", ""),
            model_name.replace("/", "_"),
        }
        for key in variants:
            metadata[key] = row
    return metadata


def find_metadata(model: str, metadata: Dict[str, dict]) -> dict | None:
    candidates = [model, model.lower(), model.split("/")[-1], model.split("/")[-1].lower()]
    normalized = model.lower().replace("/", "_").replace("-", "").replace("_", "")
    candidates.extend([
        normalized,
        model.lower().replace("/", "_"),
        model.lower().replace("/", "").replace("-", "").replace("_", ""),
    ])

    for key in candidates:
        if key in metadata:
            return metadata[key]

    for key, row in metadata.items():
        key_norm = key.lower().replace("-", "").replace("_", "")
        if key_norm == normalized or key_norm in normalized or normalized in key_norm:
            return row
    return None


def parse_parameter_count(value: str | None) -> float | None:
    if not value or value.strip() in {"", "Unknown"}:
        return None
    clean = value.replace(",", "")
    try:
        return float(clean)
    except ValueError:
        return None


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_scatter(points: List[dict], output_path: Path) -> None:
    if not points:
        raise RuntimeError("No data points available for plotting.")

    ensure_output_directory(output_path)

    # Prepare arrays for regression / correlation
    accuracies = np.array([p["accuracy"] for p in points])
    log_params = np.log10([p["parameters"] for p in points])

    # Linear regression: log10(parameters) = m * accuracy + b
    slope, intercept = np.polyfit(accuracies, log_params, 1)
    correlation = np.corrcoef(accuracies, log_params)[0, 1]

    x_line = np.linspace(max(0.0, accuracies.min() - 0.02), min(1.0, accuracies.max() + 0.02), 200)
    y_line = 10 ** (slope * x_line + intercept)

    # Colors by organization
    palette = {
        'DeepSeek': '#1F77B4',
        'DeepSeek AI': '#1F77B4',
        'Google': '#FF7F0F',
        'google': '#FF7F0F',
        'Meta AI': '#2BA02B',
        'Meta': '#2BA02B',
        'Microsoft': '#9467BD',
        'Mistral': '#8C564C',
        'Mistral AI': '#8C564C',
        'OpenAI': '#7F7F7F',
        'Perplexity': '#BCBD21',
        'Tsinghua University': '#15BECF',
        'Zhipu AI': '#15BECF',
        'x-ai': '#E377C2',
        'xAI': '#E377C2',
        'Anthropic': '#E377C2',
        'Alibaba': '#D62728',
        'Nous Research': '#17BECF',
        'Moonshot': '#F7B6D2',
    }
    def map_color(org: str) -> str:
        for key, color in palette.items():
            if key.lower() in org.lower():
                return color
        return '#bbbbbb'
    organizations = sorted({p["organization"] for p in points})
    color_map = {org: map_color(org) for org in organizations}

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for point in points:
        ax.scatter(
            point["accuracy"],
            point["parameters"],
            color=color_map[point["organization"]],
            s=28,
            edgecolors="k",
            linewidths=0.35,
            alpha=0.85,
        )

    fit_label = "Linear fit"
    ax.plot(x_line, y_line, color="#444", linewidth=1.1, linestyle="--", label=fit_label)

    for point in points:
        ax.annotate(
            point["label"],
            (point["accuracy"], point["parameters"]),
            textcoords="offset points",
            xytext=(0, 2.5),
            ha="center",
            fontsize=4.8,
        )

    ax.set_xlabel("Balanced Accuracy (All Phenotypes Average)", fontsize=8)
    ax.set_ylabel("Model Parameters", fontsize=8)
    ax.set_title("Model Size vs. Average Phenotype Performance", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")

    ax.tick_params(axis="both", length=0, labelsize=7)
    ax.grid(False, which="both")
    ax.grid(False, which="minor")

    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    legend_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=color_map[org], markeredgecolor="k", label=org)
        for org in organizations
    ]
    legend_handles.append(plt.Line2D([], [], color="#444", linestyle="--", label=fit_label))
    legend = ax.legend(
        handles=legend_handles,
        title="Organization",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6,
        title_fontsize=6,
        frameon=False,
    )

    ax.text(
        0.02,
        0.95,
        f"Pearson r = {correlation:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, linewidth=0.0),
    )

    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)



def main() -> None:
    global API_URL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL, help="Base URL of the running MicrobeLLM API")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Ground-truth dataset name")
    parser.add_argument("--species-file", default=SPECIES_FILE, help="Species file for prediction endpoint")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Path to the PDF scatter plot")
    args = parser.parse_args()

    output_path = Path(args.output)

    API_URL = args.api_url

    try:
        print("Fetching predictions...")
        pred_payload = fetch_json(f"/api/phenotype_analysis_filtered?species_file={args.species_file}")
        predictions = pred_payload.get("data", [])
        if not predictions:
            raise RuntimeError("No prediction data returned from API.")

        print("Fetching ground truth...")
        gt_payload = fetch_json(f"/api/ground_truth/data?dataset={args.dataset}&per_page=20000")
        ground_truth_records = gt_payload.get("data", [])
        if not ground_truth_records:
            raise RuntimeError("No ground-truth records returned from API.")

        ground_truth_map = {
            record["binomial_name"].lower(): record
            for record in ground_truth_records
            if record.get("binomial_name")
        }

        metadata = load_model_metadata(MODEL_METADATA_PATH)
        metrics = calculate_metrics(predictions, ground_truth_map, AVERAGE_PHENOTYPES)

        points = []
        for model, phenotype_scores in metrics.items():
            # Select phenotypes present after threshold filtering
            selected = [score for pheno, score in phenotype_scores.items() if pheno in AVERAGE_PHENOTYPES]
            if not selected:
                continue
            accuracy = sum(score for score, _ in selected) / len(selected)

            meta = find_metadata(model, metadata)
            param_count = parse_parameter_count(meta.get("Parameters") if meta else None)
            if not param_count:
                continue

            organization = meta.get("Organization", "Unknown") if meta else "Unknown"
            points.append(
                {
                    "model": model,
                    "label": model.split("/")[-1],
                    "organization": organization,
                    "parameters": param_count,
                    "accuracy": accuracy,
                }
            )

        if not points:
            raise RuntimeError("No models with both metadata and sufficient metrics.")

        print(f"Plotting {len(points)} models...")
        plot_scatter(points, output_path)
        print(f"Saved plot to {output_path}")

    except URLError as exc:
        print(f"Failed to contact API at {args.api_url}: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
