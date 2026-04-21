#!/usr/bin/env python3
"""Phylogenetic stratification of LLM phenotype prediction accuracy.

Supplementary figure addressing reviewer Q2 ("show how LLM extraction
varies across broad clades"). Joins the WA evaluation predictions with
NCBI phylum annotations from ``supp_table_annot_taxon.tsv``, computes
per-(phylum, phenotype, model) balanced accuracy, and renders a 2-panel
figure: phylum × phenotype heatmap (Panel A) and per-phylum overall
accuracy bars (Panel B) sharing a y-axis. Outlier cells are tabulated
in ``phylum_outliers.txt`` rather than plotted.

Rows (phyla) are sorted by overall balanced accuracy (best on top);
columns (phenotypes) are sorted by global mean accuracy across phyla
(easiest on the left), giving a diagonal gradient in which aberrant
cells stand out.
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
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

API_URL = "http://localhost:5050"
DATASET_NAME = "WA_Test_Dataset"
SPECIES_FILE = "wa_with_gcount.txt"
TAXONOMY_PATH = Path("microbellm/static/data/supp_table_annot_taxon.tsv")
OUTPUT_PATH = Path(
    "microbellm/templates/research/phenotype_analysis/sections/07n_phylogenetic_stratification/phylogenetic_stratification.pdf"
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
    "gram_staining": "Gram",
    "motility": "Motility",
    "extreme_environment_tolerance": "Extreme env.",
    "biofilm_formation": "Biofilm",
    "animal_pathogenicity": "Animal path.",
    "biosafety_level": "Biosafety",
    "host_association": "Host assoc.",
    "plant_pathogenicity": "Plant path.",
    "spore_formation": "Spore form.",
    "cell_shape": "Cell shape",
}

CATEGORICAL_PHENOTYPES = {"cell_shape", "biosafety_level", "gram_staining"}
# Phylum-stratified sample-size floor per (phylum, phenotype, model).
# The unstratified analyses in 07l use 100, but many phyla have only 30–70
# species in the WA benchmark, so that cutoff would erase them entirely.
# 25 keeps all eight named phyla plus "Other" in every panel while still
# requiring each balanced-accuracy estimate to rest on a reasonable sample.
SAMPLE_SIZE_THRESHOLD = 25
PHYLUM_MIN_SPECIES = 30
MISSING_TOKENS = {"n/a", "na", "null", "none", "nan", "undefined", "-", "unknown", "missing"}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_json(endpoint: str, api_url: str) -> Dict:
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Phenotype value normalization (matches 07l/07m generators)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Taxonomy loading + per-(phylum, phenotype, model) balanced accuracy
# ---------------------------------------------------------------------------

def load_phylum_lookup(path: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Return (binomial_lower -> phylum, phylum -> species_count)."""
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy table not found: {path}")
    species_phylum: Dict[str, str] = {}
    counts: Counter = Counter()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if (row.get("Member of WA subset") or "").strip().upper() != "TRUE":
                continue
            name = (row.get("Binomial name") or "").strip()
            phylum = (row.get("Phylum") or "").strip()
            if not name:
                continue
            if not phylum or phylum.upper() == "NA":
                continue
            species_phylum[name.lower()] = phylum
            counts[phylum] += 1
    return species_phylum, counts


def assign_phylum_or_other(phylum: str, kept_phyla: set) -> str:
    return phylum if phylum in kept_phyla else "Other"


def compute_phylum_metrics(
    predictions: Iterable[dict],
    gt_map: Dict[str, dict],
    species_phylum: Dict[str, str],
    kept_phyla: set,
) -> Dict[Tuple[str, str], Dict[str, Tuple[float, int]]]:
    """Return {(phylum, phenotype): {model: (score, sample_size)}}."""

    by_model: Dict[str, List[dict]] = defaultdict(list)
    for rec in predictions:
        by_model[rec.get("model")].append(rec)

    result: Dict[Tuple[str, str], Dict[str, Tuple[float, int]]] = defaultdict(dict)

    for model, rows in by_model.items():
        # Pre-bucket the rows by phylum once per model
        rows_by_phylum: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            species = row.get("binomial_name")
            if not species:
                continue
            phylum = species_phylum.get(species.lower())
            if not phylum:
                continue
            rows_by_phylum[assign_phylum_or_other(phylum, kept_phyla)].append(row)

        for phylum, phylum_rows in rows_by_phylum.items():
            for phenotype in PHENOTYPES:
                truths: List = []
                preds: List = []
                for row in phylum_rows:
                    species = row.get("binomial_name")
                    gt = gt_map.get(species.lower()) if species else None
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
                    for tv, pv in zip(truths, preds):
                        confusion[tv][pv] += 1
                    recalls = []
                    for label in labels:
                        tp = confusion[label][label]
                        fn = sum(confusion[label][o] for o in labels if o != label)
                        denom = tp + fn
                        recalls.append(tp / denom if denom else 0.0)
                    score = sum(recalls) / len(recalls)
                    sample_size = len(truths)
                else:
                    tp = tn = fp = fn = 0
                    for tv, pv in zip(truths, preds):
                        if pv and tv:
                            tp += 1
                        elif pv and not tv:
                            fp += 1
                        elif not pv and not tv:
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
                    result[(phylum, phenotype)][model] = (score, sample_size)

    return result


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------

def aggregate_cell(scores: Dict[str, Tuple[float, int]]) -> Tuple[float, float, int, int]:
    """Mean accuracy across models, SEM, n_models, total samples."""
    values = np.array([v for v, _ in scores.values()])
    samples = sum(n for _, n in scores.values())
    n_models = values.size
    if n_models == 0:
        return float("nan"), 0.0, 0, 0
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n_models)) if n_models > 1 else 0.0
    return mean, sem, n_models, samples


def write_outputs(
    cell_metrics: Dict[Tuple[str, str], Tuple[float, float, int, int]],
    phyla_order: List[str],
    phylum_species_counts: Dict[str, int],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # phylum_phenotype_scores.tsv
    long_path = output_dir / "phylum_phenotype_scores.tsv"
    with long_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["phylum", "phenotype", "mean_balanced_accuracy",
                    "sem", "n_models", "n_samples", "n_species_in_phylum"])
        for phylum in phyla_order:
            for phenotype in PHENOTYPES:
                mean, sem, n_models, samples = cell_metrics.get(
                    (phylum, phenotype), (float("nan"), 0.0, 0, 0)
                )
                w.writerow([
                    phylum, phenotype,
                    f"{mean:.4f}" if math.isfinite(mean) else "",
                    f"{sem:.4f}",
                    n_models, samples,
                    phylum_species_counts.get(phylum, 0),
                ])

    # phylum_counts.tsv
    counts_path = output_dir / "phylum_counts.tsv"
    with counts_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["phylum", "n_species", "kept_or_other"])
        for phylum in phyla_order:
            tag = "Other" if phylum == "Other" else "kept"
            w.writerow([phylum, phylum_species_counts.get(phylum, 0), tag])


def write_outliers(
    cell_metrics: Dict[Tuple[str, str], Tuple[float, float, int, int]],
    phyla_order: List[str],
    output_dir: Path,
    overall_mean: float,
    phylum_overall: Dict[str, float],
) -> None:
    """Human-readable summary for the rebuttal letter."""
    lines: List[str] = []
    lines.append("Phylogenetic stratification — outlier summary")
    lines.append(f"Overall mean balanced accuracy across phyla & phenotypes: {overall_mean:.3f}")
    lines.append("")

    # Phylum-level deviations (>0.05 from overall mean)
    lines.append("Per-phylum mean accuracy (deviations from overall):")
    for phylum in sorted(phylum_overall, key=lambda p: -phylum_overall[p]):
        delta = phylum_overall[phylum] - overall_mean
        flag = " *" if abs(delta) >= 0.05 else ""
        lines.append(f"  {phylum:25s}  mean={phylum_overall[phylum]:.3f}  Δ={delta:+.3f}{flag}")
    lines.append("")

    # Per-phenotype z-score per phylum
    pheno_means: Dict[str, np.ndarray] = {}
    for phenotype in PHENOTYPES:
        vals = []
        for phylum in phyla_order:
            mean, *_ = cell_metrics.get((phylum, phenotype), (float("nan"), 0.0, 0, 0))
            if math.isfinite(mean):
                vals.append((phylum, mean))
        pheno_means[phenotype] = vals  # type: ignore[assignment]

    rows: List[Tuple[float, str, str, float]] = []
    for phenotype, vals in pheno_means.items():
        if len(vals) < 3:
            continue
        arr = np.array([v for _, v in vals])
        mu, sd = arr.mean(), arr.std(ddof=1) if arr.size > 1 else 0.0
        if sd == 0:
            continue
        for phylum, value in vals:
            z = (value - mu) / sd
            rows.append((z, phylum, phenotype, value))

    rows.sort(key=lambda r: r[0])
    lines.append("Worst (phylum, phenotype) outliers (z < -1.5):")
    for z, phylum, phenotype, value in rows[:8]:
        if z >= -1.5:
            break
        lines.append(f"  z={z:+.2f}  {phylum:22s}  {phenotype:30s}  acc={value:.3f}")
    lines.append("")
    lines.append("Best (phylum, phenotype) outliers (z > +1.5):")
    for z, phylum, phenotype, value in reversed(rows[-8:]):
        if z <= 1.5:
            break
        lines.append(f"  z={z:+.2f}  {phylum:22s}  {phenotype:30s}  acc={value:.3f}")
    lines.append("")

    out = output_dir / "phylum_outliers.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_figure(
    cell_metrics: Dict[Tuple[str, str], Tuple[float, float, int, int]],
    phyla_order: List[str],
    phylum_species_counts: Dict[str, int],
    output_path: Path,
    overall_mean: float,
    phylum_overall: Dict[str, float],
    phylum_overall_sem: Dict[str, float],
) -> None:
    # ------------------------------------------------------------------
    # Re-order rows and columns for readability
    # ------------------------------------------------------------------
    # Rows: phyla sorted by overall balanced accuracy (best first). This
    # matches Panel B and produces a top-to-bottom red→green gradient.
    # Drop phyla with no valid phenotype cells so we don't render an empty row.
    ordered_phyla = [
        p for p in sorted(phyla_order, key=lambda p: -phylum_overall.get(p, float("nan")))
        if math.isfinite(phylum_overall.get(p, float("nan")))
    ]

    # Columns: phenotypes sorted by their cross-phylum mean (easiest first).
    pheno_phylum_means: Dict[str, List[float]] = {ph: [] for ph in PHENOTYPES}
    for phylum in ordered_phyla:
        for phenotype in PHENOTYPES:
            mean, _sem, n_models, _samples = cell_metrics.get(
                (phylum, phenotype), (float("nan"), 0.0, 0, 0)
            )
            if n_models >= 3 and math.isfinite(mean):
                pheno_phylum_means[phenotype].append(mean)
    pheno_global_mean = {
        ph: (float(np.mean(v)) if v else float("nan"))
        for ph, v in pheno_phylum_means.items()
    }
    ordered_phenotypes = sorted(
        PHENOTYPES,
        key=lambda ph: -pheno_global_mean.get(ph, float("nan")),
    )

    n_phyla = len(ordered_phyla)
    n_phen = len(ordered_phenotypes)

    # Matrix of mean accuracies (NaN for empty cells).
    matrix = np.full((n_phyla, n_phen), np.nan)
    for i, phylum in enumerate(ordered_phyla):
        for j, phenotype in enumerate(ordered_phenotypes):
            mean, _sem, n_models, _samples = cell_metrics.get(
                (phylum, phenotype), (float("nan"), 0.0, 0, 0)
            )
            if n_models >= 3 and math.isfinite(mean):
                matrix[i, j] = mean

    # ------------------------------------------------------------------
    # Layout: heatmap + per-phylum bar chart side-by-side, shared y-axis
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(9.2, 4.6))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[3.0, 0.08, 1.0],
        wspace=0.04,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2], sharey=ax_heat)

    # ----- Panel A: heatmap -----
    norm = TwoSlopeNorm(vmin=0.4, vcenter=0.65, vmax=0.9)
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("#e6e6e6")
    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    im = ax_heat.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

    ax_heat.set_xticks(np.arange(n_phen))
    ax_heat.set_xticklabels(
        [PHENOTYPE_LABELS[p] for p in ordered_phenotypes],
        rotation=30, ha="right", fontsize=8,
    )
    ax_heat.set_yticks(np.arange(n_phyla))
    ax_heat.set_yticklabels(
        [f"{p}  (n={phylum_species_counts.get(p, 0)})" for p in ordered_phyla],
        fontsize=8,
    )
    ax_heat.set_title(
        "A · Mean balanced accuracy by phylum × phenotype",
        loc="left", fontsize=10, fontweight="bold",
    )
    ax_heat.tick_params(axis="both", length=0)

    for i in range(n_phyla):
        for j in range(n_phen):
            if np.isnan(matrix[i, j]):
                continue
            txt_color = "black" if 0.55 <= matrix[i, j] <= 0.78 else "white"
            ax_heat.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=7, color=txt_color,
            )

    # Thin vertical colorbar in its own gridspec column between heatmap and
    # bar panel — keeps the heatmap title clear and avoids overlapping data.
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Balanced accuracy", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ----- Panel B: per-phylum mean ± SEM (sharing y-axis with heatmap) -----
    # Floor anchors bars at y=0.4 — below the worst phylum but above 0 so the
    # Illustrator artboard stays clean (no paths leaking to the axis origin).
    floor = 0.4
    y = np.arange(n_phyla)
    means = np.array([phylum_overall[p] for p in ordered_phyla])
    sems = np.array([phylum_overall_sem[p] for p in ordered_phyla])
    species_n = np.array([phylum_species_counts.get(p, 0) for p in ordered_phyla])

    ax_bar.barh(
        y,
        means - floor,
        left=floor,
        height=0.7,
        color="#45b75f",
        edgecolor="white",
        linewidth=0.4,
        xerr=sems,
        error_kw={"linewidth": 0.6, "ecolor": "#333"},
    )
    ax_bar.axvline(
        overall_mean, linestyle="--", color="#444", linewidth=0.8,
        label=f"Overall mean = {overall_mean:.2f}",
    )
    ax_bar.set_xlim(0.4, 0.85)
    ax_bar.set_xlabel("Overall balanced accuracy", fontsize=8)
    ax_bar.set_title(
        "B · Per-phylum mean",
        loc="left", fontsize=10, fontweight="bold",
    )
    ax_bar.tick_params(axis="x", labelsize=7)
    ax_bar.tick_params(axis="y", length=0, labelleft=False)
    ax_bar.grid(axis="x", linestyle="-", linewidth=0.4, color="#E5E5E5", zorder=0)
    ax_bar.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    # Annotate n beside each bar, clipped into the axis range
    for yi, mean, n in zip(y, means, species_n):
        ax_bar.text(
            min(mean + 0.005, 0.83), yi, f" n={n}",
            va="center", ha="left", fontsize=6.5, color="#333",
        )
    ax_bar.legend(loc="lower right", fontsize=6.5, frameon=False)

    fig.suptitle(
        "Phylogenetic stratification of LLM phenotype accuracy",
        fontsize=11, y=0.995,
    )
    fig.savefig(
        output_path, facecolor="white", edgecolor="none",
        bbox_inches="tight", pad_inches=0.04,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--species-file", default=SPECIES_FILE)
    parser.add_argument("--taxonomy", default=str(TAXONOMY_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    species_phylum, all_counts = load_phylum_lookup(Path(args.taxonomy))
    print(f"Loaded taxonomy for {len(species_phylum)} WA species across "
          f"{len(all_counts)} phyla.")

    kept_phyla = {p for p, n in all_counts.items() if n >= PHYLUM_MIN_SPECIES}
    other_count = sum(n for p, n in all_counts.items() if p not in kept_phyla)
    print(f"Keeping {len(kept_phyla)} phyla with >= {PHYLUM_MIN_SPECIES} species; "
          f"merging {len(all_counts) - len(kept_phyla)} smaller phyla into "
          f"'Other' (n={other_count}).")

    # Effective species counts (after Other-merging)
    species_counts = {p: all_counts[p] for p in kept_phyla}
    if other_count:
        species_counts["Other"] = other_count

    # Stable display order: by descending species count
    phyla_order = sorted(species_counts.keys(), key=lambda p: -species_counts[p])

    # API fetch
    try:
        pred_payload = fetch_json(
            f"/api/phenotype_analysis_filtered?species_file={args.species_file}",
            args.api_url,
        )
        gt_payload = fetch_json(
            f"/api/ground_truth/data?dataset={args.dataset}&per_page=20000",
            args.api_url,
        )
    except URLError as exc:
        sys.exit(f"Failed to contact API at {args.api_url}: {exc}")

    predictions = pred_payload.get("data", [])
    gt_records = gt_payload.get("data", [])
    if not predictions or not gt_records:
        sys.exit("Empty predictions or ground-truth payload from API.")

    gt_map = {
        rec["binomial_name"].lower(): rec
        for rec in gt_records if rec.get("binomial_name")
    }

    print(f"Predictions: {len(predictions)} rows; "
          f"ground truth: {len(gt_map)} species.")

    cell_scores = compute_phylum_metrics(predictions, gt_map, species_phylum, kept_phyla)
    cell_metrics = {
        key: aggregate_cell(scores) for key, scores in cell_scores.items()
    }

    # Per-phylum overall (mean across phenotype cells)
    phylum_overall: Dict[str, float] = {}
    phylum_overall_sem: Dict[str, float] = {}
    for phylum in phyla_order:
        vals = []
        for phenotype in PHENOTYPES:
            mean, *_ = cell_metrics.get((phylum, phenotype), (float("nan"), 0.0, 0, 0))
            if math.isfinite(mean):
                vals.append(mean)
        if vals:
            arr = np.array(vals)
            phylum_overall[phylum] = float(arr.mean())
            phylum_overall_sem[phylum] = (
                float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
            )
        else:
            phylum_overall[phylum] = float("nan")
            phylum_overall_sem[phylum] = 0.0

    overall_mean = float(np.nanmean(list(phylum_overall.values())))

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    write_outputs(cell_metrics, phyla_order, species_counts, output_dir)
    write_outliers(cell_metrics, phyla_order, output_dir, overall_mean, phylum_overall)

    plot_figure(cell_metrics, phyla_order, species_counts, output_path,
                overall_mean, phylum_overall, phylum_overall_sem)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
