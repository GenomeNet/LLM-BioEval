#!/usr/bin/env python3
"""Generate an organization-level phenotype performance overview with range markers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib
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
    "microbellm/templates/research/phenotype_analysis/sections/07l_parameters_vs_accuracy/organization_accuracy_overview.pdf"
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

PALETTE = {
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


def map_color(org: str) -> str:
    for key, color in PALETTE.items():
        if key.lower() in org.lower():
            return color
    return '#bbbbbb'


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_range(points: List[dict], output_path: Path) -> None:
    if not points:
        raise RuntimeError("No data points available for plotting.")

    ensure_output_directory(output_path)

    order = np.argsort([p["avg_accuracy"] for p in points])
    points = [points[i] for i in order]

    organizations = [p["organization"] for p in points]
    averages = np.array([p["avg_accuracy"] for p in points])
    mins = np.array([p["min_accuracy"] for p in points])
    maxs = np.array([p["max_accuracy"] for p in points])
    counts = np.array([p["model_count"] for p in points])
    models = [p["models"] for p in points]
    best_models = [p["best_model"] for p in points]

    y_positions = np.arange(len(organizations))

    fig, ax = plt.subplots(figsize=(6, 1.95))

    for y, org, mn, mx, avg, models_list, best, count in zip(
        y_positions, organizations, mins, maxs, averages, models, best_models, counts
    ):
        color = map_color(org)
        ax.hlines(y, mn, mx, color=color, linewidth=1.8, alpha=0.9)

        accs = np.array([m["accuracy"] for m in models_list])
        ax.scatter(
            accs,
            np.full_like(accs, y, dtype=float),
            color=color,
            edgecolors='k',
            linewidths=0.45,
            s=34,
            alpha=0.9,
            zorder=3,
        )

        ax.text(
            0.91,
            y,
            f"{best['label']} ({best['accuracy']:.3f}) – n={count}",
            va='center',
            ha='left',
            fontsize=7,
            fontweight='bold',
            color=color,
            transform=ax.get_yaxis_transform(),
        )

    ax.set_xlabel("Balanced Accuracy", fontsize=9)
    ax.set_title("Organization Performance Overview", fontsize=10)
    ax.set_xlim(0.45, 0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(organizations, fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis='x', labelsize=8)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

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

        org_scores: Dict[str, List[float]] = defaultdict(list)
        org_models: Dict[str, List[dict]] = defaultdict(list)
        org_counts: Counter = Counter()

        for model, phenotype_scores in metrics.items():
            valid = [score for ph, score in phenotype_scores.items() if ph in AVERAGE_PHENOTYPES]
            if not valid:
                continue
            avg_accuracy = float(np.mean([score for score, _ in valid]))
            meta = metadata.get(model) or metadata.get(model.lower()) or metadata.get(model.split('/')[-1])
            org = meta.get("Organization", "Unknown") if meta else "Unknown"
            if org == "Unknown":
                model_lower = model.lower()
                if 'gemini' in model_lower:
                    org = 'Google'
                elif model_lower.startswith('openai') or 'gpt' in model_lower:
                    org = 'OpenAI'
                elif 'deepseek' in model_lower:
                    org = 'DeepSeek'
                elif 'anthropic' in model_lower or 'claude' in model_lower:
                    org = 'Anthropic'
                elif 'mistral' in model_lower or 'mixtral' in model_lower:
                    org = 'Mistral'
                elif 'llama' in model_lower or 'meta' in model_lower:
                    org = 'Meta AI'
                elif 'wizardlm' in model_lower or 'phi-' in model_lower:
                    org = 'Microsoft'
                elif 'perplexity' in model_lower or 'sonar' in model_lower:
                    org = 'Perplexity'
                elif 'grok' in model_lower or 'x-ai' in model_lower or 'xai' in model_lower:
                    org = 'xAI'
            org_scores[org].append(avg_accuracy)
            org_models[org].append({"label": model.split('/')[-1], "accuracy": avg_accuracy})
            org_counts[org] += 1

        points: List[dict] = []
        for org, scores in org_scores.items():
            if org.lower() == 'unknown':
                continue
            acc_array = np.array(scores)
            models_list = org_models[org]
            best_model = max(models_list, key=lambda m: m["accuracy"])
            points.append(
                {
                    "organization": org,
                    "avg_accuracy": float(acc_array.mean()),
                    "min_accuracy": float(acc_array.min()),
                    "max_accuracy": float(acc_array.max()),
                    "model_count": org_counts[org],
                    "models": models_list,
                    "best_model": best_model,
                }
            )

        if not points:
            raise RuntimeError("No organizations with sufficient metrics.")

        plot_range(points, output_path)
        print(f"Saved plot to {output_path}")

    except URLError as exc:
        raise SystemExit(f"Failed to contact API at {args.api_url}: {exc}")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
