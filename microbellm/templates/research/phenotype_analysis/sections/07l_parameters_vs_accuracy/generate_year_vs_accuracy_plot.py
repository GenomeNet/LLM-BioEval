#!/usr/bin/env python3
"""Generate publication date vs. average phenotype performance scatter plot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

API_URL = "http://localhost:5050"
DATASET_NAME = "WA_Test_Dataset"
SPECIES_FILE = "wa_with_gcount.txt"
MODEL_METADATA_PATH = Path("microbellm/static/data/year_size.tsv")
OUTPUT_PATH = Path(
    "microbellm/templates/research/phenotype_analysis/sections/07l_parameters_vs_accuracy/year_vs_accuracy_average.pdf"
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


def fetch_json(endpoint: str, api_url: str) -> Dict:
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_value(value) -> str | None:
    if value is None:
        return None
    string = str(value).strip()
    if not string:
        return None
    lower = string.lower()
    if lower in MISSING_TOKENS:
        return None
    if "," in string or ";" in string:
        parts = [part.strip().lower() for part in string.replace(";", ",").split(",") if part.strip()]
        return ",".join(sorted(parts)) if parts else None
    return lower


def normalize_boolean(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lower = str(value).strip().lower()
    if lower in {"true", "1", "yes", "t", "y"}:
        return True
    if lower in {"false", "0", "no", "f", "n"}:
        return False
    return None


def normalize_categorical(phenotype: str, value) -> str | None:
    base = normalize_value(value)
    if base is None:
        return None
    if phenotype == "gram_staining":
        if "positive" in base:
            return "gram stain positive"
        if "negative" in base:
            return "gram stain negative"
        if "variable" in base:
            return "gram stain variable"
    if phenotype == "biosafety_level":
        if "1" in base:
            return "biosafety level 1"
        if "2" in base:
            return "biosafety level 2"
        if "3" in base:
            return "biosafety level 3"
    return base


def compute_metrics(
    predictions: Iterable[dict],
    ground_truth_map: Dict[str, dict],
    phenotypes: Iterable[str],
    sample_threshold: int = SAMPLE_SIZE_THRESHOLD,
) -> Dict[str, Dict[str, Tuple[float, int]]]:
    metrics: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(dict)
    by_model: Dict[str, List[dict]] = defaultdict(list)

    for record in predictions:
        by_model[record.get("model")].append(record)

    for model, rows in by_model.items():
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
                labels = sorted(set(truths) | set(preds))
                confusion = {lab: Counter() for lab in labels}
                for t_val, p_val in zip(truths, preds):
                    confusion[t_val][p_val] += 1
                recalls = []
                for label in labels:
                    tp = confusion[label][label]
                    fn = sum(confusion[label][other] for other in labels if other != label)
                    denom = tp + fn
                    recalls.append(tp / denom if denom else 0.0)
                balanced_acc = sum(recalls) / len(recalls)
                sample_size = len(truths)
            else:
                tp = tn = fp = fn = 0
                for t_val, p_val in zip(truths, preds):
                    if p_val and t_val:
                        tp += 1
                    elif p_val and not t_val:
                        fp += 1
                    elif not p_val and not t_val:
                        tn += 1
                    else:
                        fn += 1
                sample_size = tp + tn + fp + fn
                if sample_size == 0:
                    continue
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                specificity = tn / (tn + fp) if (tn + fp) else 0.0
                balanced_acc = (recall + specificity) / 2

            if sample_size >= sample_threshold and np.isfinite(balanced_acc):
                metrics[model][phenotype] = (balanced_acc, sample_size)

    return metrics


def load_model_metadata(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")

    metadata: Dict[str, dict] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    for row in rows:
        name = row.get("Model")
        if not name:
            continue
        variants = {
            name,
            name.lower(),
            name.lower().replace(" ", ""),
            name.lower().replace("-", "").replace("_", ""),
            name.replace("-", ""),
            name.replace("/", "_"),
        }
        for key in variants:
            metadata[key] = row
    return metadata


def parse_publication_date(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip().split()[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_scatter(points: List[dict], output_path: Path) -> None:
    if not points:
        raise RuntimeError("No data points available for plotting.")

    ensure_output_directory(output_path)

    dates = [p["date"] for p in points]
    accuracies = np.array([p["accuracy"] for p in points])

    ordinals = np.array([mdates.date2num(d) for d in dates])
    slope, intercept = np.polyfit(ordinals, accuracies, 1)
    correlation = np.corrcoef(ordinals, accuracies)[0, 1]

    x_line = np.linspace(ordinals.min(), ordinals.max(), 200)
    y_line = slope * x_line + intercept

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

    # Decide which models to highlight (top/bottom performers)
    sorted_points = sorted(points, key=lambda p: p["accuracy"], reverse=True)
    highlight_top = sorted_points[:3]
    highlight_bottom = sorted_points[-3:] if len(sorted_points) > 3 else []
    highlight_set = {id(p): p for p in highlight_top + highlight_bottom}

    fig, ax = plt.subplots(figsize=(4.5, 3))

    for point in points:
        is_highlighted = id(point) in highlight_set
        ax.scatter(
            point["date"],
            point["accuracy"],
            color=color_map[point["organization"]],
            s=34,
            edgecolors="k",
            linewidths=0.5 if is_highlighted else 0.4,
            alpha=0.9,
            zorder=3 if is_highlighted else 2,
        )

    fit_label = "Linear fit"
    ax.plot(mdates.num2date(x_line), y_line, color="#444", linewidth=1.1, linestyle="--", label=fit_label)

    for point in highlight_set.values():
        ax.annotate(
            point["label"],
            (point["date"], point["accuracy"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=6,
            fontweight='bold',
            arrowprops=dict(arrowstyle='-', color='black', linewidth=0.6, shrinkA=0, shrinkB=4),
        )

    ax.set_xlabel("Publication Date", fontsize=8)
    ax.set_ylabel("Balanced Accuracy (All Phenotypes Average)", fontsize=8)
    ax.set_title("Model Vintage vs. Phenotype Accuracy", fontsize=9)

    min_date = min(dates)
    max_date = max(dates)
    pad = timedelta(days=180)
    ax.set_xlim(min_date - pad, max_date + pad)
    ax.set_ylim(0.45, 0.9)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=7)

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
    ax.legend(
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--species-file", default=SPECIES_FILE)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    output_path = Path(args.output)

    try:
        pred_payload = fetch_json(f"/api/phenotype_analysis_filtered?species_file={args.species_file}", args.api_url)
        predictions = pred_payload.get("data", [])
        if not predictions:
            raise RuntimeError("No prediction data returned from API.")

        gt_payload = fetch_json(f"/api/ground_truth/data?dataset={args.dataset}&per_page=20000", args.api_url)
        ground_truth_records = gt_payload.get("data", [])
        if not ground_truth_records:
            raise RuntimeError("No ground-truth records returned from API.")

        ground_truth_map = {
            record["binomial_name"].lower(): record
            for record in ground_truth_records
            if record.get("binomial_name")
        }

        metadata = load_model_metadata(MODEL_METADATA_PATH)
        metrics = compute_metrics(predictions, ground_truth_map, AVERAGE_PHENOTYPES)

        points: List[dict] = []
        for model, phenotype_scores in metrics.items():
            valid_scores = [score for ph, score in phenotype_scores.items() if ph in AVERAGE_PHENOTYPES]
            if not valid_scores:
                continue
            accuracy = float(np.mean([score for score, _ in valid_scores]))

            meta = metadata.get(model) or metadata.get(model.lower()) or metadata.get(model.split('/')[-1])
            pub_date = parse_publication_date(meta.get("Publication date") if meta else None)
            organization = meta.get("Organization", "Unknown") if meta else "Unknown"
            if pub_date is None:
                continue

            points.append(
                {
                    "model": model,
                    "label": model.split("/")[-1],
                    "organization": organization,
                    "date": pub_date,
                    "accuracy": accuracy,
                }
            )

        if not points:
            raise RuntimeError("No models with both metadata and sufficient metrics.")

        plot_scatter(points, output_path)
        print(f"Saved plot to {output_path}")

    except URLError as exc:
        raise SystemExit(f"Failed to contact API at {args.api_url}: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
