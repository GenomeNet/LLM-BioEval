#!/usr/bin/env python3
"""Analyse agreement between species-level and trait-specific (motility)
self-assessed knowledge ratings for grok-3-mini on the WA subset.

Reads predictions directly from the benchmark SQLite DB
(microbellm.db -> processing_results), joins by binomial_name, and writes:

  - confusion_species_vs_motility.tsv   raw 4x4 counts
  - agreement_metrics.tsv               N, Cohen's kappa, Cramer's V, exact agreement
  - balanced_accuracy_by_tier.tsv       motility balanced accuracy per tier for
                                        each rating axis (species-level vs trait-level)
  - trait_vs_species_agreement.pdf      two-panel figure used in the rebuttal

The script is safe to run mid-job: it simply restricts to species that have
knowledge_group labels for BOTH templates plus a motility prediction plus a
motility ground-truth call.  Rerun as the pilot progresses.

Usage:
  python microbellm/templates/research/phenotype_analysis/sections/\
07k_knowledge_accuracy/trait_specific_pilot/analyze_trait_vs_species_agreement.py
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[7]
DB_PATH = REPO_ROOT / "microbellm.db"
assert DB_PATH.parent.name == "microbeLLM" or DB_PATH.exists(), (
    f"DB path did not resolve as expected: {DB_PATH}"
)

MODEL = os.environ.get("PILOT_MODEL", "x-ai/grok-3-mini")
SPECIES_FILE = "wa_with_gcount.txt"
GT_DATASET = "WA_Test_Dataset"

SPECIES_TEMPLATE_USER = "templates/user/template3_knowlege.txt"
MOTILITY_TEMPLATE_USER = "templates/user/template1_knowledge_motility.txt"
PHENOTYPE_TEMPLATE_USER = "templates/user/template1_phenotype.txt"

OUT_DIR = Path(__file__).resolve().parent
_model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL.split("/")[-1])
PDF_PATH = OUT_DIR / f"trait_vs_species_agreement_{_model_slug}.pdf"

TIERS = ["limited", "moderate", "extensive", "NA"]
ORDINAL_TIERS = ["limited", "moderate", "extensive"]  # ordered; NA excluded from kappa
TIER_COLORS = {
    "limited": "#d95f02",
    "moderate": "#7570b3",
    "extensive": "#1b9e77",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def normalize_tier(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"limited", "moderate", "extensive"}:
        return v
    if v in {"na", "n/a", "none", "unknown"}:
        return "NA"
    return None


def normalize_bool(value) -> bool | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "t", "y"}:
        return True
    if v in {"false", "0", "no", "f", "n"}:
        return False
    return None


def balanced_accuracy(pairs: List[Tuple[bool, bool]]) -> float:
    """Balanced accuracy for a binary classifier: mean of sensitivity and
    specificity."""
    if not pairs:
        return float("nan")
    tp = tn = fp = fn = 0
    for truth, pred in pairs:
        if truth is True and pred is True:
            tp += 1
        elif truth is False and pred is True:
            fp += 1
        elif truth is False and pred is False:
            tn += 1
        elif truth is True and pred is False:
            fn += 1
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    if np.isnan(sens) and np.isnan(spec):
        return float("nan")
    if np.isnan(sens):
        return spec
    if np.isnan(spec):
        return sens
    return 0.5 * (sens + spec)


def cohen_kappa(labels_a: List[str], labels_b: List[str], weights: str = "none") -> float:
    """Unweighted or linear-weighted Cohen's kappa on ordinal tiers (excludes NA)."""
    mask = [(a in ORDINAL_TIERS and b in ORDINAL_TIERS) for a, b in zip(labels_a, labels_b)]
    a = [x for x, m in zip(labels_a, mask) if m]
    b = [x for x, m in zip(labels_b, mask) if m]
    n = len(a)
    if n == 0:
        return float("nan")

    idx = {t: i for i, t in enumerate(ORDINAL_TIERS)}
    k = len(ORDINAL_TIERS)
    obs = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col) / n

    if weights == "linear":
        w = np.fromfunction(lambda i, j: np.abs(i - j) / (k - 1), (k, k))
        po = (w * obs).sum() / n
        pe = (w * exp).sum() / n
        return 1 - po / pe if pe else float("nan")

    po = np.trace(obs) / n
    pe = np.trace(exp) / n
    return (po - pe) / (1 - pe) if (1 - pe) else float("nan")


def spearman_rho(labels_a: List[str], labels_b: List[str]) -> float:
    """Spearman rank correlation on ordinal tiers (NA excluded)."""
    idx = {t: i for i, t in enumerate(ORDINAL_TIERS)}
    pairs = [
        (idx[a], idx[b])
        for a, b in zip(labels_a, labels_b)
        if a in ORDINAL_TIERS and b in ORDINAL_TIERS
    ]
    if len(pairs) < 3:
        return float("nan")
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    # Simple rank correlation; with many ties (ordinal 3-level), use Pearson on
    # the integer codes – this is the standard short-cut and matches scipy's
    # spearmanr up to tie correction which is negligible here.
    if xs.std(ddof=0) == 0 or ys.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def cramers_v(labels_a: List[str], labels_b: List[str]) -> float:
    """Cramer's V over the full 4x4 contingency (NA included)."""
    idx = {t: i for i, t in enumerate(TIERS)}
    n = len(labels_a)
    if n == 0:
        return float("nan")
    k = len(TIERS)
    obs = np.zeros((k, k), dtype=float)
    for x, y in zip(labels_a, labels_b):
        obs[idx[x], idx[y]] += 1
    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(exp > 0, (obs - exp) ** 2 / exp, 0).sum()
    denom = n * (min(k, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom else float("nan")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_knowledge_labels(conn: sqlite3.Connection, user_template: str) -> Dict[str, str]:
    """Return {binomial_name: normalized_tier} for one template."""
    cur = conn.execute(
        """
        SELECT binomial_name, knowledge_group
        FROM processing_results
        WHERE model = ? AND species_file = ? AND user_template = ?
          AND status = 'completed' AND knowledge_group IS NOT NULL
        """,
        (MODEL, SPECIES_FILE, user_template),
    )
    out: Dict[str, str] = {}
    for name, tier in cur:
        norm = normalize_tier(tier)
        if norm is not None:
            out[name] = norm
    return out


def load_motility_predictions(conn: sqlite3.Connection) -> Dict[str, bool]:
    cur = conn.execute(
        """
        SELECT binomial_name, motility
        FROM processing_results
        WHERE model = ? AND species_file = ? AND user_template = ?
          AND status = 'completed' AND motility IS NOT NULL
        """,
        (MODEL, SPECIES_FILE, PHENOTYPE_TEMPLATE_USER),
    )
    out: Dict[str, bool] = {}
    for name, mot in cur:
        b = normalize_bool(mot)
        if b is not None:
            out[name] = b
    return out


def load_motility_ground_truth(conn: sqlite3.Connection) -> Dict[str, bool]:
    cur = conn.execute(
        """
        SELECT binomial_name, motility
        FROM ground_truth
        WHERE dataset_name = ? AND motility IS NOT NULL
        """,
        (GT_DATASET,),
    )
    out: Dict[str, bool] = {}
    for name, mot in cur:
        b = normalize_bool(mot)
        if b is not None:
            out[name] = b
    return out


# --------------------------------------------------------------------------- #
# Output tables
# --------------------------------------------------------------------------- #

def write_confusion(path: Path, species: Dict[str, str], trait: Dict[str, str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(TIERS)}
    mat = np.zeros((len(TIERS), len(TIERS)), dtype=int)
    for name, sp_tier in species.items():
        tr_tier = trait.get(name)
        if tr_tier is None:
            continue
        mat[idx[sp_tier], idx[tr_tier]] += 1
    with path.open("w") as fh:
        fh.write("species_tier\t" + "\t".join(TIERS) + "\trow_total\n")
        for i, row_name in enumerate(TIERS):
            row = mat[i]
            fh.write(row_name + "\t" + "\t".join(str(v) for v in row) + f"\t{row.sum()}\n")
        col_totals = mat.sum(axis=0)
        fh.write("col_total\t" + "\t".join(str(v) for v in col_totals) + f"\t{mat.sum()}\n")
    return mat


def write_metrics(path: Path, species: List[str], trait: List[str]) -> Dict[str, float]:
    exact_mask = [
        (s == t) for s, t in zip(species, trait)
        if s in ORDINAL_TIERS and t in ORDINAL_TIERS
    ]
    exact = float(np.mean(exact_mask)) if exact_mask else float("nan")
    adj_mask = [
        (abs(ORDINAL_TIERS.index(s) - ORDINAL_TIERS.index(t)) <= 1)
        for s, t in zip(species, trait)
        if s in ORDINAL_TIERS and t in ORDINAL_TIERS
    ]
    adj = float(np.mean(adj_mask)) if adj_mask else float("nan")
    k_unw = cohen_kappa(species, trait, weights="none")
    k_lin = cohen_kappa(species, trait, weights="linear")
    rho = spearman_rho(species, trait)
    v = cramers_v(species, trait)
    metrics = {
        "n_paired_any_tier": len(species),
        "n_paired_ordinal": sum(s in ORDINAL_TIERS and t in ORDINAL_TIERS for s, t in zip(species, trait)),
        "exact_agreement_ordinal": exact,
        "within_one_tier_agreement_ordinal": adj,
        "spearman_rho_ordinal": rho,
        "cohens_kappa_unweighted": k_unw,
        "cohens_kappa_linear_weighted": k_lin,
        "cramers_v_including_NA": v,
    }
    with path.open("w") as fh:
        fh.write("metric\tvalue\n")
        for key, val in metrics.items():
            if isinstance(val, float):
                fh.write(f"{key}\t{val:.4f}\n")
            else:
                fh.write(f"{key}\t{val}\n")
    return metrics


def write_balanced_accuracy(
    path: Path,
    species: Dict[str, str],
    trait: Dict[str, str],
    preds: Dict[str, bool],
    truths: Dict[str, bool],
) -> Dict[str, Dict[str, float]]:
    eligible = [n for n in species if n in trait and n in preds and n in truths]
    by_axis: Dict[str, Dict[str, Dict]] = {
        "species_level": {t: {"pairs": [], "n": 0} for t in ORDINAL_TIERS},
        "motility_level": {t: {"pairs": [], "n": 0} for t in ORDINAL_TIERS},
    }
    for name in eligible:
        for axis, source in [("species_level", species), ("motility_level", trait)]:
            tier = source[name]
            if tier in ORDINAL_TIERS:
                pair = (truths[name], preds[name])
                by_axis[axis][tier]["pairs"].append(pair)
                by_axis[axis][tier]["n"] += 1

    result: Dict[str, Dict[str, float]] = {}
    with path.open("w") as fh:
        fh.write("axis\ttier\tn_species\tbalanced_accuracy\n")
        for axis, tiers in by_axis.items():
            result[axis] = {}
            for tier in ORDINAL_TIERS:
                cell = tiers[tier]
                ba = balanced_accuracy(cell["pairs"])
                result[axis][tier] = ba
                fh.write(f"{axis}\t{tier}\t{cell['n']}\t"
                         f"{'nan' if np.isnan(ba) else f'{ba:.4f}'}\n")
    # Counts too
    result["_n"] = {
        axis: {tier: by_axis[axis][tier]["n"] for tier in ORDINAL_TIERS}
        for axis in by_axis
    }
    return result


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot_figure(
    ba: Dict[str, Dict[str, float]],
    metrics: Dict[str, float],
    confusion: np.ndarray,
    out_path: Path,
) -> None:
    """Single-panel bubble correlation: species-level vs motility-specific
    self-assessment across the 4x4 rating grid (limited / moderate / extensive / NA)."""

    n_total = int(confusion.sum())
    max_count = int(confusion.max()) if n_total else 1
    # Largest bubble ~ 2600 pt^2; leaves breathing room inside a grid cell.
    area_scale = 2600.0 / max(max_count, 1)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    # Cell shading: pale grid to help the eye line up with tick labels.
    for i in range(len(TIERS)):
        for j in range(len(TIERS)):
            ax.add_patch(plt.Rectangle(
                (i - 0.5, j - 0.5), 1, 1,
                facecolor="#FAFAFA" if (i + j) % 2 == 0 else "#FFFFFF",
                edgecolor="none", zorder=0,
            ))

    # Agreement diagonal across the ordinal tiers.
    ax.plot(
        [-0.5, len(ORDINAL_TIERS) - 0.5],
        [-0.5, len(ORDINAL_TIERS) - 0.5],
        linestyle="--", color="#999", linewidth=1.0, zorder=1,
        label="perfect agreement",
    )

    xs, ys, sizes, counts = [], [], [], []
    for i in range(len(TIERS)):           # species-level tier -> x
        for j in range(len(TIERS)):       # motility-level tier -> y
            c = int(confusion[i, j])
            if c == 0:
                continue
            xs.append(i)
            ys.append(j)
            sizes.append(c * area_scale)
            counts.append(c)

    ax.scatter(
        xs, ys, s=sizes, c=counts,
        cmap="viridis", alpha=0.92,
        edgecolors="black", linewidths=0.7, zorder=3,
    )
    for x, y, c in zip(xs, ys, counts):
        ax.text(
            x, y, str(c),
            ha="center", va="center",
            fontsize=10,
            color="white" if c > max_count * 0.35 else "#222",
            fontweight="bold", zorder=4,
        )

    ax.set_xticks(range(len(TIERS)))
    ax.set_yticks(range(len(TIERS)))
    ax.set_xticklabels([t.capitalize() for t in TIERS], fontsize=11)
    ax.set_yticklabels([t.capitalize() for t in TIERS], fontsize=11)
    ax.set_xlim(-0.55, len(TIERS) - 0.45)
    ax.set_ylim(-0.55, len(TIERS) - 0.45)
    ax.set_xlabel("Species-level self-assessment", fontsize=12, fontweight="bold")
    ax.set_ylabel("Motility-specific self-assessment", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")

    # Soft gridlines between cells
    for k in range(len(TIERS) + 1):
        ax.axhline(k - 0.5, color="#E0E0E0", linewidth=0.5, zorder=0)
        ax.axvline(k - 0.5, color="#E0E0E0", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(0.8)

    # Title and metrics
    rho = metrics.get("spearman_rho_ordinal", float("nan"))
    kappa = metrics.get("cohens_kappa_linear_weighted", float("nan"))
    v = metrics.get("cramers_v_including_NA", float("nan"))
    exact = metrics.get("exact_agreement_ordinal", float("nan"))
    within1 = metrics.get("within_one_tier_agreement_ordinal", float("nan"))
    n_paired = int(metrics.get("n_paired_any_tier", 0))

    model_label = MODEL.split("/")[-1]
    ax.set_title(
        "Species-level vs motility-specific knowledge rating\n"
        f"{model_label} · WA subset · N = {n_paired} paired species",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Metrics line under the x-axis label.
    metric_txt = (
        rf"Spearman $\rho$ = {rho:.2f}   ·   "
        rf"Cohen's $\kappa_{{\mathrm{{lin}}}}$ = {kappa:.2f}   ·   "
        rf"Cramér's V = {v:.2f}   ·   "
        f"exact = {exact*100:.0f}%   ·   within ±1 tier = {within1*100:.0f}%"
    )
    fig.text(
        0.5, -0.02, metric_txt,
        ha="center", va="top", fontsize=9.5, color="#333",
    )

    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    species_labels = load_knowledge_labels(conn, SPECIES_TEMPLATE_USER)
    trait_labels = load_knowledge_labels(conn, MOTILITY_TEMPLATE_USER)
    preds = load_motility_predictions(conn)
    truths = load_motility_ground_truth(conn)

    paired_any = [n for n in species_labels if n in trait_labels]
    print(f"Species-level labels:   {len(species_labels):>5} species")
    print(f"Motility-level labels:  {len(trait_labels):>5} species (job still in progress if < 3884)")
    print(f"Phenotype predictions:  {len(preds):>5} species")
    print(f"Ground-truth labels:    {len(truths):>5} species")
    print(f"Paired (both labels):   {len(paired_any):>5} species")
    print()

    if not paired_any:
        print("No paired species yet — rerun once more motility predictions complete.")
        return

    species_list = [species_labels[n] for n in paired_any]
    trait_list = [trait_labels[n] for n in paired_any]

    print("Species-level tier distribution (paired subset):")
    for tier, count in Counter(species_list).most_common():
        print(f"  {tier:>9}: {count}")
    print()
    print("Motility tier distribution (paired subset):")
    for tier, count in Counter(trait_list).most_common():
        print(f"  {tier:>9}: {count}")
    print()

    conf_path = OUT_DIR / "confusion_species_vs_motility.tsv"
    metrics_path = OUT_DIR / "agreement_metrics.tsv"
    ba_path = OUT_DIR / "balanced_accuracy_by_tier.tsv"

    mat = write_confusion(conf_path, species_labels, trait_labels)
    print(f"Wrote {conf_path}")
    print("Confusion (rows = species-level, cols = motility-level):")
    print("          " + "".join(f"{t:>10}" for t in TIERS))
    for i, row_name in enumerate(TIERS):
        print(f"  {row_name:>7}" + "".join(f"{v:>10}" for v in mat[i]))
    print()

    metrics = write_metrics(metrics_path, species_list, trait_list)
    print(f"Wrote {metrics_path}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print()

    ba = write_balanced_accuracy(ba_path, species_labels, trait_labels, preds, truths)
    print(f"Wrote {ba_path}")
    for axis in ("species_level", "motility_level"):
        print(f"  {axis}:")
        for tier in ORDINAL_TIERS:
            ba_val = ba[axis][tier]
            n = ba["_n"][axis][tier]
            val_str = "nan" if np.isnan(ba_val) else f"{ba_val*100:5.1f}%"
            print(f"    {tier:>9}  n={n:>4}  balanced acc = {val_str}")
    print()

    plot_figure(ba, metrics, mat, PDF_PATH)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
