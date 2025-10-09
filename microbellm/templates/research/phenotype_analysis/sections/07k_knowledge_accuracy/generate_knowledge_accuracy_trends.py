#!/usr/bin/env python3
"""Generate knowledge-group accuracy trends from the MicrobeLLM API."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

API_URL_DEFAULT = "http://localhost:5050"
DATASET_DEFAULT = "WA_Test_Dataset"
SPECIES_FILE_DEFAULT = "wa_with_gcount.txt"
OUTPUT_DEFAULT = (
    "microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/"
    "knowledge_accuracy_trends.pdf"
)

KNOWLEDGE_GROUPS = ["limited", "moderate", "extensive"]
PHENOTYPES = [
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
]

PHENOTYPE_LABELS = {
    "gram_staining": "Gram Staining",
    "motility": "Motility",
    "extreme_environment_tolerance": "Extreme Environment Tolerance",
    "biofilm_formation": "Biofilm Formation",
    "animal_pathogenicity": "Animal Pathogenicity",
    "biosafety_level": "Biosafety Level",
    "host_association": "Host Association",
    "plant_pathogenicity": "Plant Pathogenicity",
    "spore_formation": "Spore Formation",
    "cell_shape": "Cell Shape",
}

BOOLEAN_FIELDS = {
    "motility",
    "biofilm_formation",
    "animal_pathogenicity",
    "plant_pathogenicity",
    "host_association",
}
MIN_SAMPLES_PER_FIELD = 30
MIN_SAMPLES_PER_GROUP = 250
MISSING_TOKENS = {"n/a", "na", "null", "none", "nan", "undefined", "-", "unknown", "missing", ""}

PALETTE = {
    "DeepSeek": "#1B9E77",
    "DeepSeek AI": "#1B9E77",
    "Google": "#D95F02",
    "google": "#D95F02",
    "Meta AI": "#7570B3",
    "Meta": "#7570B3",
    "Microsoft": "#E7298A",
    "Mistral": "#66A61E",
    "Mistral AI": "#66A61E",
    "OpenAI": "#E6AB02",
    "Perplexity": "#A6761D",
    "Tsinghua University": "#1F78B4",
    "Zhipu AI": "#1F78B4",
    "x-ai": "#B2DF8A",
    "xAI": "#B2DF8A",
    "Anthropic": "#CAB2D6",
    "Alibaba": "#FB9A99",
    "Nous Research": "#FDBF6F",
    "Moonshot": "#FF7F00",
}


@dataclass
class KnowledgePoint:
    model: str
    display_name: str
    organization: str
    color: str
    accuracy: Dict[str, float]
    sample_sizes: Dict[str, int]
    total_samples: int


@dataclass
class JonckheereTerpstraResult:
    statistic: float
    mean: float
    variance: float
    z_score: float
    p_value: float
    sample_sizes: Dict[str, int]
    adjusted_p: float = float("nan")


def fetch_json(endpoint: str) -> Dict:
    with urlopen(endpoint, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_model_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return metadata

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            model_name = row.get("Model")
            if not model_name:
                continue
            variants = {
                model_name,
                model_name.lower(),
                model_name.split("/")[-1],
                model_name.lower().replace("-", "").replace("_", ""),
            }
            for key in variants:
                metadata[key] = row
    return metadata


def map_organization(model: str, metadata: Dict[str, Dict[str, str]]) -> str:
    meta = metadata.get(model) or metadata.get(model.lower()) or metadata.get(model.split("/")[-1])
    if meta and meta.get("Organization") not in (None, "", "Unknown"):
        org = meta["Organization"]
    else:
        lower = model.lower()
        if "gemini" in lower:
            org = "Google"
        elif lower.startswith("openai") or "gpt" in lower:
            org = "OpenAI"
        elif "deepseek" in lower:
            org = "DeepSeek"
        elif "anthropic" in lower or "claude" in lower:
            org = "Anthropic"
        elif "mistral" in lower or "mixtral" in lower:
            org = "Mistral"
        elif "llama" in lower or "meta" in lower:
            org = "Meta AI"
        elif "phi-" in lower or "wizardlm" in lower:
            org = "Microsoft"
        elif "perplexity" in lower or "sonar" in lower:
            org = "Perplexity"
        elif "grok" in lower or "x-ai" in lower or "xai" in lower:
            org = "xAI"
        else:
            org = "Other"
    return org


def map_color(org: str) -> str:
    for key, color in PALETTE.items():
        if key.lower() in org.lower():
            return color
    return "#888888"


def collect_group_samples(points: List[KnowledgePoint]) -> Dict[str, List[float]]:
    samples: Dict[str, List[float]] = {group: [] for group in KNOWLEDGE_GROUPS}
    for point in points:
        for group in KNOWLEDGE_GROUPS:
            value = point.accuracy.get(group, float("nan"))
            if np.isfinite(value):
                samples[group].append(float(value))
    return samples


def summarize_group_accuracy(points: List[KnowledgePoint]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, float] = {group: 0.0 for group in KNOWLEDGE_GROUPS}
    weights: Dict[str, int] = {group: 0 for group in KNOWLEDGE_GROUPS}
    for point in points:
        for group in KNOWLEDGE_GROUPS:
            accuracy = point.accuracy.get(group, float("nan"))
            count = point.sample_sizes.get(group, 0)
            if np.isfinite(accuracy) and count > 0:
                totals[group] += accuracy * count
                weights[group] += count

    summary: Dict[str, Dict[str, float]] = {}
    for group in KNOWLEDGE_GROUPS:
        count = weights[group]
        mean = totals[group] / count if count else float("nan")
        summary[group] = {"mean": mean, "count": float(count)}
    return summary


def benjamini_hochberg(p_values: Dict[str, float]) -> Dict[str, float]:
    valid = [(field, p) for field, p in p_values.items() if np.isfinite(p)]
    if not valid:
        return {}
    valid.sort(key=lambda item: item[1])
    n = len(valid)
    intermediates = []
    for rank, (field, p_value) in enumerate(valid, start=1):
        adj = min(p_value * n / rank, 1.0)
        intermediates.append((field, adj))

    adjusted: Dict[str, float] = {}
    min_adjusted = 1.0
    for field, adj in reversed(intermediates):
        min_adjusted = min(min_adjusted, adj)
        adjusted[field] = min_adjusted
    return adjusted


def jonckheere_terpstra_test(samples: Dict[str, List[float]]) -> JonckheereTerpstraResult:
    groups = []
    sample_sizes: Dict[str, int] = {}
    for group in KNOWLEDGE_GROUPS:
        values = np.asarray(samples.get(group, []), dtype=float)
        values = values[np.isfinite(values)]
        sample_sizes[group] = int(values.size)
        if values.size:
            groups.append(values)

    if len(groups) < 2:
        return JonckheereTerpstraResult(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), sample_sizes)

    statistic = 0.0
    pair_sum = 0
    variance_term = 0.0

    for i in range(len(groups) - 1):
        lower = groups[i]
        for j in range(i + 1, len(groups)):
            higher = groups[j]
            diffs = higher[:, None] - lower[None, :]
            statistic += float(np.sum(diffs > 0))
            statistic += 0.5 * float(np.sum(diffs == 0))
            pairs = lower.size * higher.size
            pair_sum += pairs
            variance_term += pairs * (lower.size + higher.size + 1)

    if pair_sum == 0:
        return JonckheereTerpstraResult(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), sample_sizes)

    mean = 0.5 * pair_sum
    variance = variance_term / 12.0
    # Normal approximation without tie correction (accuracies seldom tie in practice).
    if variance <= 0:
        z_score = float("nan")
        p_value = float("nan")
    else:
        z_score = (statistic - mean) / float(np.sqrt(variance))
        p_value = float(norm.sf(z_score))

    return JonckheereTerpstraResult(statistic, mean, variance, z_score, p_value, sample_sizes)


def normalize_value(field: str, value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in MISSING_TOKENS:
        return None

    if field == "gram_staining":
        if "positive" in lower:
            return "positive"
        if "negative" in lower:
            return "negative"
        if "variable" in lower:
            return "variable"
        return lower

    if field == "biosafety_level":
        if "1" in lower:
            return "level_1"
        if "2" in lower:
            return "level_2"
        if "3" in lower:
            return "level_3"
        return lower

    if field == "cell_shape":
        if "bacill" in lower:
            return "bacillus"
        if "cocci" in lower or "coccus" in lower:
            return "coccus"
        if "spir" in lower:
            return "spirillum"
        if "tail" in lower:
            return "tail"
        if "filament" in lower:
            return "filamentous"
        return lower

    if field in BOOLEAN_FIELDS:
        if lower in {"true", "1", "yes", "y", "t"}:
            return True
        if lower in {"false", "0", "no", "n", "f"}:
            return False

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) > 1:
        return ",".join(sorted(part.lower() for part in parts))
    return lower


def balanced_accuracy(field: str, observations: List[Dict[str, object]]) -> float:
    truths = [obs["truth"] for obs in observations]
    preds = [obs["prediction"] for obs in observations]

    is_boolean = field in BOOLEAN_FIELDS or isinstance(truths[0], bool) or isinstance(preds[0], bool)

    if is_boolean:
        tp = tn = fp = fn = 0
        for truth, pred in zip(truths, preds):
            if pred is True and truth is True:
                tp += 1
            elif pred is True and truth is False:
                fp += 1
            elif pred is False and truth is False:
                tn += 1
            elif pred is False and truth is True:
                fn += 1
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        return (sensitivity + specificity) / 2

    labels = sorted(set(truths) | set(preds))
    if not labels:
        return float("nan")

    tp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}

    for truth, pred in zip(truths, preds):
        if truth == pred:
            tp[truth] += 1
        else:
            fn[truth] += 1

    recalls = []
    for label in labels:
        denom = tp[label] + fn[label]
        if denom == 0:
            continue
        recalls.append(tp[label] / denom)

    if not recalls:
        return float("nan")

    return float(np.mean(recalls))


def compute_points(
    data: List[Dict],
    metadata: Dict[str, Dict[str, str]],
    top_n: int,
    phenotypes: Iterable[str] | None = None,
) -> List[KnowledgePoint]:
    selected_fields = list(phenotypes) if phenotypes is not None else list(PHENOTYPES)
    grouped: Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for item in data:
        model = item.get("model")
        group_raw = (item.get("knowledge_group") or "").lower()
        if group_raw not in KNOWLEDGE_GROUPS:
            continue

        for field in selected_fields:
            truth = normalize_value(field, item.get("ground_truth", {}).get(field))
            pred = normalize_value(field, item.get("predictions", {}).get(field))
            if truth is None or pred is None:
                continue
            grouped[model][group_raw][field].append({"truth": truth, "prediction": pred})

    points: List[KnowledgePoint] = []

    for model, group_data in grouped.items():
        accuracies_per_group = {}
        samples_per_group = {}

        for group in KNOWLEDGE_GROUPS:
            field_data = group_data.get(group, {})
            per_field_scores = []
            total_samples = 0
            for field in selected_fields:
                observations = field_data.get(field, [])
                if len(observations) < MIN_SAMPLES_PER_FIELD:
                    continue
                score = balanced_accuracy(field, observations)
                if not np.isfinite(score):
                    continue
                per_field_scores.append(score * 100)
                total_samples += len(observations)

            if per_field_scores and total_samples >= MIN_SAMPLES_PER_GROUP:
                accuracies_per_group[group] = float(np.mean(per_field_scores))
                samples_per_group[group] = total_samples
            else:
                accuracies_per_group[group] = float("nan")
                samples_per_group[group] = 0

        if not any(np.isfinite(val) for val in accuracies_per_group.values()):
            continue

        organization = map_organization(model, metadata)
        color = map_color(organization)
        display_name = format_model_name(model)
        total_samples = sum(samples_per_group.values())

        points.append(
            KnowledgePoint(
                model=model,
                display_name=display_name,
                organization=organization,
                color=color,
                accuracy=accuracies_per_group,
                sample_sizes=samples_per_group,
                total_samples=total_samples,
            )
        )

    points.sort(key=lambda p: p.total_samples, reverse=True)

    if top_n > 0:
        points = points[:top_n]

    return points


def compute_correlation_stats(
    data: List[Dict],
    metadata: Dict[str, Dict[str, str]],
) -> Dict[str, float]:
    correlations: Dict[str, float] = {}
    for phenotype in PHENOTYPES:
        points = compute_points(
            data,
            metadata,
            top_n=0,
            phenotypes=[phenotype],
        )
        group_indices: List[int] = []
        accuracies: List[float] = []
        for point in points:
            for idx, group in enumerate(KNOWLEDGE_GROUPS):
                value = point.accuracy.get(group, float("nan"))
                if np.isfinite(value):
                    group_indices.append(idx)
                    accuracies.append(value)

        if len(group_indices) >= 3 and len(set(group_indices)) > 1:
            corr = float(np.corrcoef(group_indices, accuracies)[0, 1])
        else:
            corr = float("nan")
        correlations[phenotype] = corr
    return correlations


def format_model_name(model: str) -> str:
    if "/" not in model:
        return model
    provider, name = model.split("/", 1)
    provider_map = {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "google": "Google",
        "meta-llama": "Meta",
        "mistralai": "Mistral",
        "deepseek": "DeepSeek",
        "x-ai": "xAI",
    }
    display_provider = provider_map.get(provider, provider.capitalize())
    return f"{display_provider} {name}"


def build_trend_figure(points: List[KnowledgePoint], title: str) -> matplotlib.figure.Figure:
    fig_height = 3.0 + 0.12 * max(0, len(points) - 8)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    x_positions = np.arange(len(KNOWLEDGE_GROUPS))

    def marker_area(count: int) -> float:
        if count <= 0:
            return 0.0
        return min(180.0, 18.0 + 4.0 * np.sqrt(count))

    for point in points:
        x_vals: List[int] = []
        y_vals: List[float] = []
        sizes: List[float] = []
        for idx, group in enumerate(KNOWLEDGE_GROUPS):
            accuracy = point.accuracy.get(group, float("nan"))
            samples = point.sample_sizes.get(group, 0)
            if np.isfinite(accuracy):
                x_vals.append(x_positions[idx])
                y_vals.append(accuracy / 100.0)
                sizes.append(marker_area(samples))

        if not x_vals:
            continue

        line_width = 1.3 if len(x_vals) > 1 else 0.0
        label = f"{point.display_name} ({point.organization})"
        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            markersize=3,
            linewidth=line_width,
            label=label,
            color=point.color,
            alpha=0.65,
        )
        ax.scatter(
            x_vals,
            y_vals,
            s=sizes,
            color=point.color,
            alpha=0.85,
            edgecolors="none",
            label="_nolegend_",
        )

    avg_scores: Dict[str, float] = {}
    avg_counts: Dict[str, int] = {}
    for group in KNOWLEDGE_GROUPS:
        group_scores = []
        group_samples = 0
        for point in points:
            value = point.accuracy.get(group, float("nan"))
            if np.isfinite(value):
                group_scores.append(value)
                group_samples += point.sample_sizes.get(group, 0)
        if group_scores:
            avg_scores[group] = float(np.mean(group_scores)) / 100.0
            avg_counts[group] = group_samples

    if avg_scores:
        avg_line = [avg_scores.get(g, np.nan) for g in KNOWLEDGE_GROUPS]
        avg_x = [x_positions[idx] for idx, value in enumerate(avg_line) if np.isfinite(value)]
        avg_y = [value for value in avg_line if np.isfinite(value)]
        if avg_x:
            ax.plot(
                avg_x,
                avg_y,
                color="#222222",
                linewidth=2.0,
                linestyle="--",
                marker="s",
                markersize=4,
                label="Average",
            )
            avg_sizes = [
                marker_area(avg_counts.get(KNOWLEDGE_GROUPS[idx], 0))
                for idx, value in enumerate(avg_line)
                if np.isfinite(value)
            ]
            ax.scatter(
                avg_x,
                avg_y,
                s=avg_sizes,
                facecolors="none",
                edgecolors="#222222",
                linewidths=0.9,
                label="_nolegend_",
            )

            for idx, value in enumerate(avg_line):
                if not np.isfinite(value):
                    continue
                text_y = min(value + 0.03, 0.995)
                ax.text(
                    x_positions[idx],
                    text_y,
                    f"{value*100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#222222",
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([grp.capitalize() for grp in KNOWLEDGE_GROUPS], fontsize=9)
    ax.set_ylabel("Balanced Accuracy", fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(-0.1, len(KNOWLEDGE_GROUPS) - 0.9)
    ax.set_yticks(np.linspace(0.5, 1.0, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    ax.set_title(title, fontsize=11, pad=12)
    ax.text(
        0.98,
        0.06,
        "Marker size ∝ sample count",
        transform=ax.transAxes,
        fontsize=7,
        color="#444444",
        ha="right",
        va="bottom",
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=False,
        )
    right_margin = 0.78 if handles else 0.96
    fig.tight_layout(rect=(0, 0, right_margin, 1))
    return fig


def build_overview_figure(
    points: List[KnowledgePoint],
    title: str,
    overall_stats: JonckheereTerpstraResult | None = None,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    x_positions = np.arange(len(KNOWLEDGE_GROUPS))

    override_colors = {
        "OpenAI": "#939497",
        "Anthropic": "#DC88B9",
        "DeepSeek": "#217AB7",
        "xAI": "#DC88B9",
        "Google": "#F5812A",
    }

    global_x: List[float] = []
    global_y: List[float] = []

    for point in points:
        y_values = []
        x_vals = []
        for idx, group in enumerate(KNOWLEDGE_GROUPS):
            accuracy = point.accuracy.get(group, float("nan"))
            if np.isfinite(accuracy):
                x_vals.append(x_positions[idx])
                y_values.append(accuracy)
        if not x_vals:
            continue
        global_x.extend(x_vals)
        global_y.extend(y_values)
        color = override_colors.get(point.organization, point.color)
        corr_value = float("nan")
        if len(x_vals) >= 2:
            x_arr = np.array(x_vals, dtype=float)
            y_arr = np.array(y_values, dtype=float)
            if y_arr.std(ddof=0) > 0:
                corr_value = float(np.corrcoef(x_arr, y_arr)[0, 1])
        ax.plot(
            x_vals,
            y_values,
            color=color,
            linewidth=0.8,
            alpha=0.45,
            zorder=2,
        )
        ax.scatter(
            x_vals,
            y_values,
            s=36,
            color=color,
            edgecolors="k",
            linewidths=0.35,
            alpha=0.9,
            label=f"{point.display_name} ({point.organization})",
            zorder=3,
        )

        # Annotate extensive group point outside the plot area
        extensive_val = point.accuracy.get("extensive", float("nan"))
        if np.isfinite(extensive_val):
            if np.isfinite(corr_value):
                annotation = f"{point.display_name} (r={corr_value:.2f})"
            else:
                annotation = point.display_name
            ax.annotate(
                annotation,
                (x_positions[-1], extensive_val),
                textcoords="offset points",
                xytext=(18, 0),
                ha="left",
                va="center",
                fontsize=7,
                color=color,
                clip_on=False,
            )

    avg_scores: Dict[str, float] = {}
    for group in KNOWLEDGE_GROUPS:
        group_scores = [
            point.accuracy.get(group, float("nan"))
            for point in points
            if np.isfinite(point.accuracy.get(group, float("nan")))
        ]
        if group_scores:
            avg_scores[group] = float(np.mean(group_scores))

    overall_corr = float("nan")
    if len(global_x) >= 2:
        x_arr = np.array(global_x, dtype=float)
        y_arr = np.array(global_y, dtype=float)
        if y_arr.std(ddof=0) > 0:
            overall_corr = float(np.corrcoef(x_arr, y_arr)[0, 1])

    if avg_scores:
        avg_line = [avg_scores.get(g, np.nan) for g in KNOWLEDGE_GROUPS]
        ax.plot(
            x_positions,
            avg_line,
            color="#222222",
            linewidth=2.2,
            linestyle="--",
            label="Average",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([grp.capitalize() for grp in KNOWLEDGE_GROUPS], fontsize=10)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=10)
    ax.set_ylim(65, 85)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_title(title, fontsize=14)

    text_lines = []
    if np.isfinite(overall_corr):
        text_lines.append(f"Overall r = {overall_corr:.2f}")
    if overall_stats and np.isfinite(overall_stats.p_value):
        text_lines.append(f"JT p = {overall_stats.p_value:.3g}")
    if text_lines:
        ax.text(
            0.02,
            0.95,
            " | ".join(text_lines),
            transform=ax.transAxes,
            fontsize=9,
            color="#222222",
            ha="left",
        )

    fig.tight_layout(rect=(0, 0, 0.8, 1))
    return fig


def build_correlation_figure(correlation_stats: Dict[str, float], orientation: str = "vertical") -> matplotlib.figure.Figure:
    pairs = [
        (
            correlation_stats.get(field, float("nan")),
            PHENOTYPE_LABELS.get(field, field.replace("_", " ").title()),
        )
        for field in PHENOTYPES
    ]
    pairs = [item for item in pairs if np.isfinite(item[0])]
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No phenotypes met correlation criteria", ha="center", va="center")
        ax.axis("off")
        return fig

    values = [value for value, _ in pairs]
    labels = [label for _, label in pairs]

    if orientation == "horizontal":
        fig, ax = plt.subplots(figsize=(5.6, 2.268))
        y_positions = np.arange(len(labels))
        ax.barh(y_positions, values, color="#3f8fb5", alpha=0.8)
        ax.axvline(0.0, color="#222222", linewidth=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Pearson r (accuracy vs. knowledge level)", fontsize=10)
        ax.set_xlim(-1.0, 1.0)
        ax.set_title("Correlation Between Knowledge Level and Accuracy by Phenotype", fontsize=12)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(7.2, 3.78))
        x_positions = np.arange(len(labels))
        ax.bar(x_positions, values, color="#3f8fb5", alpha=0.8)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Pearson r (accuracy vs. knowledge level)", fontsize=10)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title("Correlation Between Knowledge Level and Accuracy by Phenotype", fontsize=12)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()

    return fig


def plot(
    overall_points: List[KnowledgePoint],
    per_field_points: Dict[str, List[KnowledgePoint]],
    correlation_stats: Dict[str, float],
    output_path: Path,
) -> None:
    if not overall_points and not any(per_field_points.values()):
        raise RuntimeError("No knowledge points available for plotting.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        if overall_points:
            overall_fig = build_trend_figure(overall_points, "Model Accuracy Across Knowledge Groups")
            pdf.savefig(overall_fig, bbox_inches="tight")
            plt.close(overall_fig)

        for field in PHENOTYPES:
            points = per_field_points.get(field, [])
            if not points:
                continue
            title = f"{PHENOTYPE_LABELS.get(field, field.replace('_', ' ').title())} Accuracy Across Knowledge Groups"
            fig = build_trend_figure(points, title)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if correlation_stats:
            corr_fig = build_correlation_figure(correlation_stats, orientation="horizontal")
            pdf.savefig(corr_fig, bbox_inches="tight")
            plt.close(corr_fig)


def load_knowledge_data(api_url: str, dataset: str) -> List[Dict]:
    endpoint = f"{api_url.rstrip('/')}/api/phenotype_accuracy_by_knowledge?dataset={dataset}"
    payload = fetch_json(endpoint)
    if not payload.get("success"):
        raise RuntimeError("API returned success=false")
    return payload.get("data", [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL_DEFAULT, help="Base URL of the MicrobeLLM API")
    parser.add_argument("--dataset", default=DATASET_DEFAULT, help="Ground-truth dataset name")
    parser.add_argument(
        "--metadata",
        default="microbellm/static/data/year_size.tsv",
        help="Path to TSV containing model metadata",
    )
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output PDF path")
    parser.add_argument(
        "--overview-output",
        default=None,
        help="Optional path for saving only the aggregate overview plot",
    )
    parser.add_argument(
        "--correlation-output",
        default=None,
        help="Optional path for saving only the phenotype correlation plot",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of models to display (sorted by total sample size)",
    )
    args = parser.parse_args()

    try:
        print("Fetching knowledge data...")
        knowledge_data = load_knowledge_data(args.api_url, args.dataset)
        print(f"Loaded {len(knowledge_data):,} knowledge-tagged predictions")

        print("Loading model metadata...")
        metadata = load_model_metadata(Path(args.metadata))

        print("Computing knowledge accuracy points...")
        points = compute_points(knowledge_data, metadata, top_n=args.top_n)
        if not points:
            raise RuntimeError("No models met the inclusion criteria")

        print("Computing per-phenotype trends...")
        per_field_points: Dict[str, List[KnowledgePoint]] = {}
        for field in PHENOTYPES:
            field_points = compute_points(
                knowledge_data,
                metadata,
                top_n=args.top_n,
                phenotypes=[field],
            )
            if field_points:
                per_field_points[field] = field_points
                print(f"  {PHENOTYPE_LABELS.get(field, field)}: {len(field_points)} models")

        print("Calculating knowledge-group correlations...")
        correlation_stats = compute_correlation_stats(knowledge_data, metadata)

        print("Running Jonckheere-Terpstra trend tests...")
        overall_samples = collect_group_samples(points)
        overall_jt = jonckheere_terpstra_test(overall_samples)
        if np.isfinite(overall_jt.z_score):
            print(
                "  Overall (all phenotypes aggregated): "
                f"z={overall_jt.z_score:.2f}, p={overall_jt.p_value:.3g}"
            )
        phenotype_results: Dict[str, JonckheereTerpstraResult] = {}
        for field, field_points in per_field_points.items():
            samples = collect_group_samples(field_points)
            result = jonckheere_terpstra_test(samples)
            phenotype_results[field] = result
            if np.isfinite(result.z_score):
                label = PHENOTYPE_LABELS.get(field, field)
                print(f"  {label}: z={result.z_score:.2f}, p={result.p_value:.3g}")

        if phenotype_results:
            bh_adjusted = benjamini_hochberg({field: res.p_value for field, res in phenotype_results.items()})
            if bh_adjusted:
                print("Applying Benjamini-Hochberg FDR correction (alpha=0.05)...")
                for field, res in phenotype_results.items():
                    res.adjusted_p = bh_adjusted.get(field, float("nan"))
                significant = [
                    (field, res)
                    for field, res in phenotype_results.items()
                    if np.isfinite(res.adjusted_p) and res.adjusted_p <= 0.05
                ]
                if significant:
                    significant.sort(key=lambda item: item[1].adjusted_p)
                    for field, res in significant:
                        label = PHENOTYPE_LABELS.get(field, field)
                        direction = "increasing" if res.z_score > 0 else "decreasing"
                        print(
                            f"    {label}: z={res.z_score:.2f}, raw p={res.p_value:.3g}, "
                            f"BH-adjusted p={res.adjusted_p:.3g} ({direction})"
                        )

        group_summary = summarize_group_accuracy(points)
        if group_summary:
            print("Weighted mean balanced accuracy by knowledge group (all models):")
            for group in KNOWLEDGE_GROUPS:
                metrics = group_summary.get(group, {})
                mean = metrics.get("mean", float("nan"))
                count = metrics.get("count", 0.0)
                if np.isfinite(mean):
                    print(f"  {group.capitalize():>9}: {mean:.2f}% (n={int(count):,} samples)")

        if args.overview_output:
            print(f"Saving overview-only plot to {args.overview_output}...")
            overview_fig = build_overview_figure(
                points,
                "Model Accuracy Across Knowledge Groups",
                overall_stats=overall_jt,
            )
            Path(args.overview_output).parent.mkdir(parents=True, exist_ok=True)
            overview_fig.savefig(args.overview_output, bbox_inches="tight")
            plt.close(overview_fig)

        if args.correlation_output and correlation_stats:
            print(f"Saving correlation plot to {args.correlation_output}...")
            corr_only_fig = build_correlation_figure(correlation_stats, orientation="horizontal")
            Path(args.correlation_output).parent.mkdir(parents=True, exist_ok=True)
            corr_only_fig.savefig(args.correlation_output, bbox_inches="tight")
            plt.close(corr_only_fig)

        print(f"Plotting to {args.output}...")
        plot(points, per_field_points, correlation_stats, Path(args.output))
        print(f"Saved plot to {args.output}")

    except URLError as exc:
        raise SystemExit(f"Failed to reach API: {exc}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
