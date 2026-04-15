#!/usr/bin/env python3
"""Phenotype balanced-accuracy stratified by open-weight vs. closed-source models.

Supplementary figure addressing reviewer Q6. Model accessibility is pulled
from ``microbellm/static/data/year_size.tsv``; anything starting with
"Open weights" is treated as open-weight, "API access" and
"Hosted access (no API)" as closed-source. Models tagged "Unreleased" or
with an empty accessibility field are dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
    "microbellm/templates/research/phenotype_analysis/sections/07m_open_vs_closed/open_vs_closed_accuracy.pdf"
)

PHENOTYPES = (
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

PHENOTYPE_LABELS = {
    "gram_staining": "Gram staining",
    "motility": "Motility",
    "extreme_environment_tolerance": "Extreme envt.",
    "biofilm_formation": "Biofilm",
    "animal_pathogenicity": "Animal path.",
    "biosafety_level": "Biosafety",
    "host_association": "Host assoc.",
    "plant_pathogenicity": "Plant path.",
    "spore_formation": "Spore form.",
    "cell_shape": "Cell shape",
}

CATEGORICAL_PHENOTYPES = {"cell_shape", "biosafety_level", "gram_staining"}
SAMPLE_SIZE_THRESHOLD = 100
MISSING_TOKENS = {"n/a", "na", "null", "none", "nan", "undefined", "-", "unknown", "missing"}

OPEN_COLOR = "#45b75f"
CLOSED_COLOR = "#517abd"


def fetch_json(endpoint: str, api_url: str) -> Dict:
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_value(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    low = s.lower()
    if low in MISSING_TOKENS:
        return None
    if "," in s or ";" in s:
        parts = [p.strip().lower() for p in s.replace(";", ",").split(",") if p.strip()]
        return ",".join(sorted(parts)) if parts else None
    return low


def normalize_boolean(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    low = str(value).strip().lower()
    if low in {"true", "1", "yes", "t", "y"}:
        return True
    if low in {"false", "0", "no", "f", "n"}:
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
    gt_map: Dict[str, dict],
    phenotypes: Iterable[str],
) -> Dict[str, Dict[str, Tuple[float, int]]]:
    metrics: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(dict)
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for rec in predictions:
        by_model[rec.get("model")].append(rec)

    for model, rows in by_model.items():
        for phenotype in phenotypes:
            truths: List = []
            preds: List = []
            for row in rows:
                species = row.get("binomial_name")
                if not species:
                    continue
                gt = gt_map.get(species.lower())
                if not gt:
                    continue
                if phenotype in CATEGORICAL_PHENOTYPES:
                    t = normalize_categorical(phenotype, gt.get(phenotype))
                    p = normalize_categorical(phenotype, row.get(phenotype))
                else:
                    t = normalize_boolean(gt.get(phenotype))
                    p = normalize_boolean(row.get(phenotype))
                if t is None or p is None:
                    continue
                truths.append(t)
                preds.append(p)

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
                score = sum(recalls) / len(recalls)
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
                score = (recall + specificity) / 2

            if sample_size >= SAMPLE_SIZE_THRESHOLD and math.isfinite(score):
                metrics[model][phenotype] = (score, sample_size)

    return metrics


def load_metadata(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    metadata: Dict[str, dict] = {}
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


def find_meta(model: str, metadata: Dict[str, dict]) -> dict | None:
    candidates = [model, model.lower(), model.split("/")[-1], model.split("/")[-1].lower()]
    norm = model.lower().replace("/", "_").replace("-", "").replace("_", "")
    candidates.extend([norm, model.lower().replace("/", "_")])
    for key in candidates:
        if key in metadata:
            return metadata[key]
    for key, row in metadata.items():
        k = key.lower().replace("-", "").replace("_", "")
        if k == norm or k in norm or norm in k:
            return row
    return None


def classify_accessibility(meta: dict | None) -> str | None:
    if not meta:
        return None
    raw = (meta.get("Model accessibility") or "").strip()
    if not raw or raw.lower() == "unreleased":
        return None
    lower = raw.lower()
    if lower.startswith("open weights"):
        return "open"
    if lower in {"api access", "hosted access (no api)"}:
        return "closed"
    return None


def plot_open_vs_closed(
    per_model: Dict[str, Dict[str, float]],
    categories: Dict[str, str],
    output_path: Path,
) -> None:
    # Per-phenotype collection
    pheno_scores: Dict[str, Dict[str, List[float]]] = {
        p: {"open": [], "closed": []} for p in PHENOTYPES
    }
    for model, scores in per_model.items():
        cat = categories.get(model)
        if cat is None:
            continue
        for phenotype, value in scores.items():
            if phenotype in pheno_scores:
                pheno_scores[phenotype][cat].append(value)

    # Overall (per-model average across phenotypes)
    overall: Dict[str, List[float]] = {"open": [], "closed": []}
    for model, scores in per_model.items():
        cat = categories.get(model)
        if cat is None or not scores:
            continue
        overall[cat].append(float(np.mean(list(scores.values()))))

    n_open = len(overall["open"])
    n_closed = len(overall["closed"])

    fig, (ax_strip, ax_bars) = plt.subplots(
        1, 2, figsize=(8.5, 3.2),
        gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.28},
    )

    # Left: per-model average accuracy (strip plot)
    rng = np.random.default_rng(0)
    x_positions = {"open": 0, "closed": 1}
    for cat in ("open", "closed"):
        values = np.array(overall[cat])
        if values.size:
            jitter = rng.uniform(-0.15, 0.15, size=values.size)
            ax_strip.scatter(
                np.full_like(values, x_positions[cat]) + jitter,
                values,
                s=26, alpha=0.8,
                color=OPEN_COLOR if cat == "open" else CLOSED_COLOR,
                edgecolors="k", linewidths=0.3, zorder=3,
            )
            mean_val = float(np.mean(values))
            ax_strip.hlines(
                mean_val, x_positions[cat] - 0.28, x_positions[cat] + 0.28,
                color="k", linewidth=1.4, zorder=4,
            )
    ax_strip.set_xticks([0, 1])
    ax_strip.set_xticklabels([f"Open-weight\n(n={n_open})", f"Closed\n(n={n_closed})"], fontsize=8)
    ax_strip.set_xlim(-0.5, 1.5)
    ax_strip.set_ylim(0.5, 0.8)
    ax_strip.set_ylabel("Balanced accuracy (avg. across phenotypes)", fontsize=8)
    ax_strip.set_title("Per-model average accuracy", fontsize=9)
    ax_strip.tick_params(axis="y", labelsize=7)
    ax_strip.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#E5E5E5", zorder=0)
    ax_strip.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_strip.spines[spine].set_visible(False)

    # Right: per-phenotype grouped bars with SEM
    phenotypes = list(PHENOTYPES)
    x = np.arange(len(phenotypes))
    bar_width = 0.36

    # Anchor bars to the visible y-floor so the underlying vector paths don't
    # extend below the axis (which makes the artboard bounds wrong in
    # Illustrator). Heights are computed relative to the floor.
    floor = 0.5
    for offset, cat, color in ((-bar_width / 2, "open", OPEN_COLOR),
                               (bar_width / 2, "closed", CLOSED_COLOR)):
        means = []
        sems = []
        for phenotype in phenotypes:
            vals = np.array(pheno_scores[phenotype][cat])
            if vals.size:
                means.append(float(vals.mean()))
                sems.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
            else:
                means.append(np.nan)
                sems.append(0.0)
        heights = [0.0 if np.isnan(m) else (m - floor) for m in means]
        ax_bars.bar(
            x + offset, heights, bar_width,
            bottom=floor,
            yerr=sems, capsize=2.5,
            color=color, edgecolor="white", linewidth=0.4,
            label=f"Open-weight (n={n_open})" if cat == "open" else f"Closed (n={n_closed})",
            error_kw={"linewidth": 0.6, "ecolor": "#333"},
        )

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([PHENOTYPE_LABELS[p] for p in phenotypes],
                            rotation=30, ha="right", fontsize=7)
    ax_bars.set_ylim(0.5, 0.9)
    ax_bars.set_ylabel("Balanced accuracy", fontsize=8)
    ax_bars.set_title("Per-phenotype mean accuracy (± SEM)", fontsize=9)
    ax_bars.tick_params(axis="y", labelsize=7)
    ax_bars.grid(True, axis="y", linestyle="-", linewidth=0.4, color="#E5E5E5", zorder=0)
    ax_bars.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_bars.spines[spine].set_visible(False)
    ax_bars.legend(loc="upper right", fontsize=7, frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--species-file", default=SPECIES_FILE)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    try:
        pred_payload = fetch_json(
            f"/api/phenotype_analysis_filtered?species_file={args.species_file}", args.api_url
        )
        predictions = pred_payload.get("data", [])
        if not predictions:
            raise RuntimeError("No predictions returned from API.")
        gt_payload = fetch_json(
            f"/api/ground_truth/data?dataset={args.dataset}&per_page=20000", args.api_url
        )
        gt_records = gt_payload.get("data", [])
        if not gt_records:
            raise RuntimeError("No ground truth returned from API.")
    except URLError as exc:
        sys.exit(f"Failed to reach API at {args.api_url}: {exc}")

    gt_map = {
        rec["binomial_name"].lower(): rec
        for rec in gt_records if rec.get("binomial_name")
    }

    metadata = load_metadata(MODEL_METADATA_PATH)
    metrics = compute_metrics(predictions, gt_map, PHENOTYPES)

    per_model: Dict[str, Dict[str, float]] = {}
    categories: Dict[str, str] = {}
    for model, scores in metrics.items():
        meta = find_meta(model, metadata)
        cat = classify_accessibility(meta)
        if cat is None:
            continue
        per_model[model] = {p: s for p, (s, _n) in scores.items()}
        categories[model] = cat

    if not per_model:
        sys.exit("No models with usable accessibility metadata.")

    by_cat = Counter(categories.values())
    print(f"Classified models: open-weight={by_cat['open']}, closed={by_cat['closed']}")

    # Also list any models that were dropped so the user can sanity-check them.
    unclassified = []
    for model in metrics.keys():
        if model in categories:
            continue
        meta = find_meta(model, metadata)
        raw = (meta.get("Model accessibility", "") if meta else "").strip()
        unclassified.append((model, raw or "no metadata"))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classification_tsv = output_path.with_name("model_classification.tsv")
    with classification_tsv.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["model", "category", "accessibility_raw", "organization", "avg_balanced_accuracy"])
        # Classified models
        for model in sorted(per_model.keys(), key=lambda m: (categories[m], m.lower())):
            meta = find_meta(model, metadata)
            raw = (meta.get("Model accessibility", "") if meta else "").strip()
            org = (meta.get("Organization", "") if meta else "").strip()
            avg_acc = float(np.mean(list(per_model[model].values())))
            writer.writerow([model, categories[model], raw, org, f"{avg_acc:.4f}"])
        # Dropped models — useful for audit
        for model, raw in sorted(unclassified):
            meta = find_meta(model, metadata)
            org = (meta.get("Organization", "") if meta else "").strip()
            writer.writerow([model, "excluded", raw, org, ""])
    print(f"Wrote classification to {classification_tsv}")

    plot_open_vs_closed(per_model, categories, output_path)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
