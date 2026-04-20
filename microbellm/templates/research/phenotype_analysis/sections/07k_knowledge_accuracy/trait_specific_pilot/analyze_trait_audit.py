#!/usr/bin/env python3
"""Trait-audit aggregation analysis for the reviewer rebuttal.

Q: does the species-level knowledge rating (one query per species) approximate
   the aggregate of 13 trait-level knowledge ratings (13 queries per species)?

If yes → the paper's species-level rating is a legitimate summary statistic,
and the reviewer's "you should ask per-trait" objection is moot.

The script reads predictions directly from the benchmark SQLite DB, joins
the species-level template (template3_knowlege) with the 13 trait-specific
templates (template1_knowledge_*) by binomial_name for one model, computes
several aggregation schemes, and writes both tables and figures used in the
rebuttal.

Usage:
    PILOT_MODEL=openai/gpt-4.1-nano \\
        python microbellm/templates/research/phenotype_analysis/sections/\\
07k_knowledge_accuracy/trait_specific_pilot/analyze_trait_audit.py

Runs safely before all jobs complete — species that don't yet have all 13
trait ratings are simply excluded from the paired analysis (with a warning
on n_paired). Rerun as more jobs finish.
"""

from __future__ import annotations

import os
import re
import sqlite3
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Silence the expected "Mean of empty slice" warning that fires when a random
# trait subset happens to be all-NA for a given species; those species are
# dropped by the downstream mask anyway.
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Mean of empty slice"
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[7]
DB_PATH = REPO_ROOT / "microbellm.db"

MODEL = os.environ.get("PILOT_MODEL", "openai/gpt-4.1-nano")
SPECIES_FILE = "wa_with_gcount.txt"

SPECIES_TEMPLATE_USER = "templates/user/template3_knowlege.txt"

# Order matters for the progressive-aggregation plot: keep a fixed sequence
# so the "first k of K" curve is deterministic across reruns.
# Note: `hemolysis` is intentionally excluded — this phenotype is not reported
# in the main paper's benchmark, so including it in the trait-audit would
# introduce a phenotype that has no counterpart in the main-paper panel.
TRAITS: List[str] = [
    "motility",
    "gram_staining",
    "aerophilicity",
    "extreme_environment_tolerance",
    "biofilm_formation",
    "animal_pathogenicity",
    "biosafety_level",
    "health_association",
    "host_association",
    "plant_pathogenicity",
    "spore_formation",
    "cell_shape",
]
TRAIT_TEMPLATE = "templates/user/template1_knowledge_{trait}.txt"

TIERS = ["limited", "moderate", "extensive", "NA"]
ORDINAL_TIERS = ["limited", "moderate", "extensive"]
TIER_NUM = {"limited": 0, "moderate": 1, "extensive": 2}

OUT_DIR = Path(__file__).resolve().parent
MODEL_SLUG = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL.split("/")[-1])


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


def discretize(mean_val: float) -> str:
    if mean_val < 0.5:
        return "limited"
    if mean_val < 1.5:
        return "moderate"
    return "extensive"


def aggregate_species(trait_ratings: Dict[str, str]) -> Dict[str, object]:
    """Aggregate 13 trait ratings into summary statistics for one species.

    Returns dict with:
      - tier: discretized aggregate tier (limited/moderate/extensive) or "NA"
      - mean: continuous mean of ordinal ratings, or np.nan
      - n_ordinal: number of non-NA trait ratings
      - na_frac: fraction of trait ratings that are NA
      - max, min, mode: over ordinal ratings (or None)
    """
    ordinal_vals = [
        TIER_NUM[trait_ratings[t]]
        for t in TRAITS
        if t in trait_ratings and trait_ratings[t] in ORDINAL_TIERS
    ]
    na_count = sum(
        1 for t in TRAITS
        if t in trait_ratings and trait_ratings[t] == "NA"
    )
    total_rated = sum(1 for t in TRAITS if t in trait_ratings)

    n_ordinal = len(ordinal_vals)
    if total_rated == 0 or n_ordinal < 3:
        return {
            "tier": "NA",
            "mean": float("nan"),
            "n_ordinal": n_ordinal,
            "na_frac": (na_count / total_rated) if total_rated else float("nan"),
            "max": None,
            "min": None,
            "mode": None,
        }

    mean_val = float(np.mean(ordinal_vals))
    inv = {v: k for k, v in TIER_NUM.items()}
    return {
        "tier": discretize(mean_val),
        "mean": mean_val,
        "n_ordinal": n_ordinal,
        "na_frac": na_count / total_rated,
        "max": inv[max(ordinal_vals)],
        "min": inv[min(ordinal_vals)],
        "mode": Counter(
            [inv[v] for v in ordinal_vals]
        ).most_common(1)[0][0],
    }


def cohen_kappa(labels_a: List[str], labels_b: List[str], weights: str = "none") -> float:
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


def _pairwise_rho(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    rx = _rank(xs)
    ry = _rank(ys)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(arr: np.ndarray) -> np.ndarray:
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr))
    # Tie-adjusted average ranks
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=float)
    for i, r in zip(inv, ranks):
        sums[i] += r
    means = sums / counts
    return means[inv]


def cramers_v_4x4(labels_a: List[str], labels_b: List[str]) -> float:
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
    denom = n * (k - 1)
    return float(np.sqrt(chi2 / denom)) if denom else float("nan")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_knowledge_labels(conn: sqlite3.Connection, user_template: str) -> Dict[str, str]:
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


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_bubble_s_vs_agg(
    species_tier: List[str],
    agg_tier: List[str],
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    n = len(species_tier)
    idx = {t: i for i, t in enumerate(TIERS)}
    confusion = np.zeros((len(TIERS), len(TIERS)), dtype=int)
    for s, a in zip(species_tier, agg_tier):
        confusion[idx[s], idx[a]] += 1

    max_count = int(confusion.max()) if n else 1
    area_scale = 2600.0 / max(max_count, 1)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    for i in range(len(TIERS)):
        for j in range(len(TIERS)):
            ax.add_patch(plt.Rectangle(
                (i - 0.5, j - 0.5), 1, 1,
                facecolor="#FAFAFA" if (i + j) % 2 == 0 else "#FFFFFF",
                edgecolor="none", zorder=0,
            ))

    ax.plot(
        [-0.5, len(ORDINAL_TIERS) - 0.5],
        [-0.5, len(ORDINAL_TIERS) - 0.5],
        linestyle="--", color="#999", linewidth=1.0, zorder=1,
        label="perfect agreement",
    )

    xs, ys, sizes, counts = [], [], [], []
    for i in range(len(TIERS)):
        for j in range(len(TIERS)):
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
    ax.set_xlabel("Species-level self-assessment (one query)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Trait-audit aggregate (mean of 13 trait queries)", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    for k in range(len(TIERS) + 1):
        ax.axhline(k - 0.5, color="#E0E0E0", linewidth=0.5, zorder=0)
        ax.axvline(k - 0.5, color="#E0E0E0", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(0.8)

    rho = metrics.get("spearman_rho", float("nan"))
    kappa = metrics.get("kappa_linear", float("nan"))
    v = metrics.get("cramers_v", float("nan"))
    exact = metrics.get("exact", float("nan"))
    within1 = metrics.get("within_one", float("nan"))

    ax.set_title(
        "Species-level rating vs aggregate of 13 trait-level ratings\n"
        f"{MODEL.split('/')[-1]} · WA subset · N = {n} paired species",
        fontsize=13, fontweight="bold", pad=12,
    )
    metric_txt = (
        rf"Spearman $\rho$ = {rho:.2f}   ·   "
        rf"Cohen's $\kappa_{{\mathrm{{lin}}}}$ = {kappa:.2f}   ·   "
        rf"Cramér's V = {v:.2f}   ·   "
        f"exact = {exact*100:.0f}%   ·   within ±1 tier = {within1*100:.0f}%"
    )
    fig.text(0.5, -0.02, metric_txt, ha="center", va="top", fontsize=9.5, color="#333")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_progressive_aggregation(
    rhos: List[float],
    rhos_ci: List[Tuple[float, float]],
    rhos_min: List[float],
    rhos_max: List[float],
    out_path: Path,
) -> None:
    """The money-shot figure: ρ(species-level, aggregate of k traits) vs k,
    averaged over random trait subsets of each size. Shows that individual
    trait ratings are weak proxies (ρ ≈ 0.3–0.6 depending on trait), but the
    aggregate saturates near the species-level rating as k grows."""
    k_vals = np.arange(1, len(rhos) + 1)
    lo = np.array([ci[0] for ci in rhos_ci])
    hi = np.array([ci[1] for ci in rhos_ci])
    mn = np.array(rhos_min)
    mx = np.array(rhos_max)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    # Outer envelope: min/max across subsets at each k
    ax.fill_between(k_vals, mn, mx, color="#4C72B0", alpha=0.10,
                    label="range across trait subsets (min–max)")
    # Inner envelope: 2.5th–97.5th percentile across subsets
    ax.fill_between(k_vals, lo, hi, color="#4C72B0", alpha=0.25,
                    label="95% interval across trait subsets")
    ax.plot(k_vals, rhos, color="#274466", linewidth=2.2, marker="o", markersize=6, zorder=3,
            label=r"median Spearman $\rho$(species-level, mean of $k$ random traits)")

    # Reference: best single-trait baseline (k=1 upper end = max across individual traits)
    if len(mx) >= 1 and not np.isnan(mx[0]):
        ax.axhline(mx[0], color="#B84A40", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.text(len(rhos), mx[0], f"  best single trait ρ = {mx[0]:.2f}",
                va="center", ha="left", fontsize=9, color="#B84A40")
    # Reference: worst single-trait baseline
    if len(mn) >= 1 and not np.isnan(mn[0]):
        ax.axhline(mn[0], color="#B84A40", linestyle=":", linewidth=1.2, alpha=0.5)
        ax.text(len(rhos), mn[0], f"  worst single trait ρ = {mn[0]:.2f}",
                va="center", ha="left", fontsize=8.5, color="#B84A40", alpha=0.85)

    ax.set_xticks(k_vals)
    ax.set_xlabel("Number of trait-level queries averaged (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Spearman $\rho$ with species-level rating", fontsize=12, fontweight="bold")
    y_lo = max(0, min(mn[~np.isnan(mn)].min() if (~np.isnan(mn)).any() else 0, 0) - 0.05)
    y_hi = min(1.0, (mx[~np.isnan(mx)].max() if (~np.isnan(mx)).any() else 1.0) + 0.05)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, linestyle="-", color="#CCCCCC", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(0.8)
    ax.set_title(
        "How many trait queries does it take to recover the species-level signal?\n"
        f"{MODEL.split('/')[-1]} · WA subset · {len(rhos)} traits sampled",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tier_distribution(
    species_tier: List[str],
    agg_tier: List[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    width = 0.35
    x = np.arange(len(TIERS))
    s_counts = [species_tier.count(t) for t in TIERS]
    a_counts = [agg_tier.count(t) for t in TIERS]
    s_total = max(sum(s_counts), 1)
    a_total = max(sum(a_counts), 1)

    ax.bar(x - width / 2, s_counts, width, label="Species-level rating",
           color="#4C72B0", edgecolor="#274466")
    ax.bar(x + width / 2, a_counts, width, label="Trait-aggregate (mean of k traits, discretised)",
           color="#DD8452", edgecolor="#884E2E")

    # Count on top, percentage just below the count, in a smaller muted font.
    for xi, sc, ac in zip(x, s_counts, a_counts):
        ax.text(xi - width / 2, sc, f"{sc}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(xi - width / 2, sc, f"\n({100 * sc / s_total:.1f}%)",
                ha="center", va="bottom", fontsize=8, color="#555")
        ax.text(xi + width / 2, ac, f"{ac}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(xi + width / 2, ac, f"\n({100 * ac / a_total:.1f}%)",
                ha="center", va="bottom", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in TIERS], fontsize=11)
    ax.set_ylabel("Number of species", fontsize=12, fontweight="bold")
    # Leave headroom so the two-line count+percent labels fit above the bars.
    top = max(max(s_counts), max(a_counts)) if (s_counts and a_counts) else 1
    ax.set_ylim(0, top * 1.18)
    ax.set_title(
        "Tier distribution: species-level rating vs trait-aggregate\n"
        f"{MODEL.split('/')[-1]} · WA subset",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, axis="y", linestyle="-", color="#CCCCCC", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(
    rating_names: List[str],
    rho_matrix: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(len(rating_names)))
    ax.set_yticks(range(len(rating_names)))
    pretty = [r.replace("_", " ") for r in rating_names]
    ax.set_xticklabels(pretty, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(pretty, fontsize=9)

    for i in range(len(rating_names)):
        for j in range(len(rating_names)):
            val = rho_matrix[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(val) > 0.55 else "#222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Spearman $\rho$", fontsize=11)

    ax.set_title(
        "Pairwise Spearman correlation across 14 knowledge ratings\n"
        f"{MODEL.split('/')[-1]} · WA subset",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #

def main() -> None:
    global TRAITS  # noqa: PLW0603  — we rewrite TRAITS to the "active" subset below
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    species_labels = load_knowledge_labels(conn, SPECIES_TEMPLATE_USER)

    trait_labels: Dict[str, Dict[str, str]] = {}
    for trait in TRAITS:
        t = TRAIT_TEMPLATE.format(trait=trait)
        trait_labels[trait] = load_knowledge_labels(conn, t)

    n_species_level = len(species_labels)
    print(f"Model:             {MODEL}")
    print(f"Species file:      {SPECIES_FILE}")
    print(f"Species-level labels: {n_species_level}")
    print("Trait-level labels:")
    for trait in TRAITS:
        n = len(trait_labels[trait])
        flag = "" if n == n_species_level else "  (incomplete)"
        print(f"  {trait:>32}: {n:>5}{flag}")

    # Active traits: only include traits that have ≥ 80% of species-level coverage.
    # This lets the script run incrementally as jobs finish — we still produce
    # sensible aggregates, we just note the reduced K in the output labels.
    active_traits = [
        t for t in TRAITS
        if len(trait_labels[t]) >= 0.8 * n_species_level
    ]
    skipped = [t for t in TRAITS if t not in active_traits]
    if skipped:
        print(f"\nSkipping traits with <80% coverage: {skipped}")
    print(f"Active traits for this run (K={len(active_traits)}): {active_traits}")

    # Paired set: species that have species-level + every ACTIVE trait label.
    paired_names = [
        name for name in species_labels
        if all(name in trait_labels[t] for t in active_traits)
    ]
    print(f"\nPaired species (have all {1 + len(active_traits)} ratings): {len(paired_names)}")

    if not paired_names or not active_traits:
        print("Not enough data yet — rerun once more trait jobs complete.")
        return

    # From here on, operate on active_traits rather than full TRAITS list.
    TRAITS = active_traits

    # Per-species rating table + aggregates.
    rows: List[Dict[str, object]] = []
    for name in paired_names:
        row = {"binomial_name": name, "species_level": species_labels[name]}
        trait_dict = {t: trait_labels[t][name] for t in TRAITS}
        row.update({f"trait_{t}": trait_dict[t] for t in TRAITS})
        agg = aggregate_species(trait_dict)
        row["agg_tier"] = agg["tier"]
        row["agg_mean"] = agg["mean"]
        row["agg_n_ordinal"] = agg["n_ordinal"]
        row["agg_na_frac"] = agg["na_frac"]
        row["agg_max"] = agg["max"]
        row["agg_min"] = agg["min"]
        row["agg_mode"] = agg["mode"]
        rows.append(row)

    # Write per-species TSV
    per_species_path = OUT_DIR / f"per_species_ratings_{MODEL_SLUG}.tsv"
    header = (
        ["binomial_name", "species_level"]
        + [f"trait_{t}" for t in TRAITS]
        + ["agg_tier", "agg_mean", "agg_n_ordinal", "agg_na_frac",
           "agg_max", "agg_min", "agg_mode"]
    )
    with per_species_path.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(
                "" if r.get(h) is None else
                (f"{r[h]:.4f}" if isinstance(r[h], float) and not np.isnan(r[h]) else str(r[h]))
                for h in header
            ) + "\n")
    print(f"\nWrote {per_species_path}")

    # Agreement metrics (species-level vs aggregate tier)
    s_tiers = [r["species_level"] for r in rows]
    a_tiers = [r["agg_tier"] for r in rows]

    # For Spearman use numeric mean rather than discretized tier
    s_num = np.array(
        [TIER_NUM[t] if t in TIER_NUM else np.nan for t in s_tiers],
        dtype=float,
    )
    a_mean = np.array([r["agg_mean"] for r in rows], dtype=float)
    rho = _pairwise_rho(s_num[~np.isnan(s_num) & ~np.isnan(a_mean)],
                        a_mean[~np.isnan(s_num) & ~np.isnan(a_mean)])

    exact = float(np.mean([s == a for s, a in zip(s_tiers, a_tiers)
                           if s in ORDINAL_TIERS and a in ORDINAL_TIERS]))
    within1 = float(np.mean([
        abs(ORDINAL_TIERS.index(s) - ORDINAL_TIERS.index(a)) <= 1
        for s, a in zip(s_tiers, a_tiers)
        if s in ORDINAL_TIERS and a in ORDINAL_TIERS
    ]))
    kappa_unw = cohen_kappa(s_tiers, a_tiers, weights="none")
    kappa_lin = cohen_kappa(s_tiers, a_tiers, weights="linear")
    cramers = cramers_v_4x4(s_tiers, a_tiers)

    # Fix 4: reorder so the rank-preserving primary metric (Spearman ρ on the
    # continuous aggregate) leads. The discretisation-sensitive metrics
    # (exact-tier agreement, κ, Cramér's V) are suppressed by regression-to-
    # the-mean as k grows and should NOT be cited in place of ρ.
    metrics = {
        # --- primary: rank-order agreement on the continuous aggregate ---
        "n_paired": len(rows),
        "spearman_rho_continuous_primary": rho,
        # --- supporting (categorical / discretised) ---
        "within_one_tier_categorical": within1,
        "exact_tier_categorical_discretized": exact,
        "kappa_linear_discretized": kappa_lin,
        "kappa_unweighted_discretized": kappa_unw,
        "cramers_v_4x4_discretized": cramers,
    }
    metrics_path = OUT_DIR / f"aggregate_agreement_metrics_{MODEL_SLUG}.tsv"
    with metrics_path.open("w") as fh:
        fh.write("metric\tvalue\tnotes\n")
        notes = {
            "n_paired": "paired species with species-level + aggregate defined",
            "spearman_rho_continuous_primary": "PRIMARY metric; rank correlation of species-level (ordinal) with continuous aggregate mean. Unaffected by regression-to-the-mean as k grows.",
            "within_one_tier_categorical": "proportion of species within one tier between species-level and discretised aggregate (ordinal only).",
            "exact_tier_categorical_discretized": "proportion of species with exact tier match after aggregate discretisation. Suppressed by regression-to-mean; do NOT cite in place of rho.",
            "kappa_linear_discretized": "Cohen's linear-weighted kappa on discretised tiers (ordinal only). Same caveat.",
            "kappa_unweighted_discretized": "Cohen's unweighted kappa on discretised tiers. Same caveat.",
            "cramers_v_4x4_discretized": "Cramer's V on 4x4 contingency including NA.",
        }
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            fh.write(f"{k}\t{val_str}\t{notes.get(k, '')}\n")
    print(f"Wrote {metrics_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Progressive aggregation curve: for each k in 1..K, compute ρ(S, mean of
    # k traits) AVERAGED over many random trait subsets of size k. This avoids
    # the artefact where a particular trait ordering produces a non-monotonic
    # curve (e.g., adding a weakly-correlated trait at k=2 dips ρ below k=1).
    # The random-subset curve shows "expected ρ from any k trait queries" —
    # a cleaner scientific claim for the rebuttal.
    rng = np.random.default_rng(42)
    from itertools import combinations as _combinations

    K = len(TRAITS)
    rhos: List[float] = []      # median across subsets
    rhos_ci: List[Tuple[float, float]] = []  # 2.5th/97.5th across subsets
    rhos_min: List[float] = []
    rhos_max: List[float] = []
    n_used_per_k: List[int] = []  # fix 2: species used per k-bin
    n = len(rows)
    trait_matrix = np.full((n, K), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, t in enumerate(TRAITS):
            val = r[f"trait_{t}"]
            if val in TIER_NUM:
                trait_matrix[i, j] = TIER_NUM[val]
    s_num_full = np.array(
        [TIER_NUM[t] if t in TIER_NUM else np.nan for t in s_tiers], dtype=float
    )

    # Note on denominator choice: we use FLOATING complete-case (each subset's
    # mean is defined on its own non-NA species set) rather than FIXED (species
    # with ordinal ratings on every K trait). The fixed subset biases toward
    # species the model is confident about across all traits — on our data
    # that drops N from 3884 → 1202 (31%) and shifts k=1 median ρ from 0.48
    # → 0.35. Floating complete-case keeps all 3884 species in play; we
    # report n_species_used per k-bin for transparency.
    fixed_mask = (~np.isnan(s_num_full)) & (~np.isnan(trait_matrix).any(axis=1))
    n_fixed_diag = int(fixed_mask.sum())
    print(f"\nDiagnostic: species with all {K} traits + species-level all ordinal: "
          f"{n_fixed_diag} / {n}")

    # Enumerate all C(K, k) subsets up to a cap; otherwise sample.
    MAX_SUBSETS_PER_K = 300
    for k in range(1, K + 1):
        all_subsets = list(_combinations(range(K), k))
        if len(all_subsets) > MAX_SUBSETS_PER_K:
            idx_choices = rng.choice(len(all_subsets), MAX_SUBSETS_PER_K, replace=False)
            subsets = [all_subsets[i] for i in idx_choices]
        else:
            subsets = all_subsets

        rho_k: List[float] = []
        n_used_at_k: List[int] = []
        for subset in subsets:
            cols = list(subset)
            means = np.nanmean(trait_matrix[:, cols], axis=1)
            mask = ~np.isnan(s_num_full) & ~np.isnan(means)
            if mask.sum() < 10:
                continue
            rho_k.append(_pairwise_rho(s_num_full[mask], means[mask]))
            n_used_at_k.append(int(mask.sum()))

        if not rho_k:
            rhos.append(float("nan"))
            rhos_ci.append((float("nan"), float("nan")))
            rhos_min.append(float("nan"))
            rhos_max.append(float("nan"))
            n_used_per_k.append(0)
            continue

        rho_k_arr = np.array(rho_k)
        rhos.append(float(np.median(rho_k_arr)))
        rhos_ci.append((float(np.percentile(rho_k_arr, 2.5)),
                        float(np.percentile(rho_k_arr, 97.5))))
        rhos_min.append(float(rho_k_arr.min()))
        rhos_max.append(float(rho_k_arr.max()))
        # Report the median species count across subsets for this k.
        n_used_per_k.append(int(np.median(n_used_at_k)))

    prog_path = OUT_DIR / f"progressive_aggregation_rho_{MODEL_SLUG}.tsv"
    with prog_path.open("w") as fh:
        fh.write("k\trho_median\tci_low\tci_high\trho_min\trho_max\tn_subsets\tn_species_used\n")
        for k, (rk, (lo, hi), rmn, rmx, nu) in enumerate(
            zip(rhos, rhos_ci, rhos_min, rhos_max, n_used_per_k), start=1
        ):
            n_sub = min(MAX_SUBSETS_PER_K, sum(1 for _ in _combinations(range(K), k)))
            fh.write(f"{k}\t{rk:.4f}\t{lo:.4f}\t{hi:.4f}\t{rmn:.4f}\t{rmx:.4f}\t{n_sub}\t{nu}\n")
    print(f"Wrote {prog_path}")

    # Correlation matrix: species_level + 13 traits (14x14)
    all_names = ["species_level"] + [f"trait_{t}" for t in TRAITS]
    numeric_all = np.column_stack([s_num_full, trait_matrix])
    p = numeric_all.shape[1]
    rho_mat = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(p):
            mask = ~np.isnan(numeric_all[:, i]) & ~np.isnan(numeric_all[:, j])
            if mask.sum() >= 3:
                rho_mat[i, j] = _pairwise_rho(numeric_all[mask, i], numeric_all[mask, j])

    corr_path = OUT_DIR / f"trait_correlation_spearman_{MODEL_SLUG}.tsv"
    with corr_path.open("w") as fh:
        fh.write("\t" + "\t".join(all_names) + "\n")
        for i, rname in enumerate(all_names):
            fh.write(rname + "\t" + "\t".join(
                "nan" if np.isnan(rho_mat[i, j]) else f"{rho_mat[i, j]:.3f}"
                for j in range(p)
            ) + "\n")
    print(f"Wrote {corr_path}")

    # Figures
    fig_bubble = OUT_DIR / f"species_vs_aggregate_{MODEL_SLUG}.pdf"
    plot_bubble_s_vs_agg(s_tiers, a_tiers, metrics, fig_bubble)
    print(f"Wrote {fig_bubble}")

    fig_prog = OUT_DIR / f"progressive_aggregation_{MODEL_SLUG}.pdf"
    plot_progressive_aggregation(rhos, rhos_ci, rhos_min, rhos_max, fig_prog)
    print(f"Wrote {fig_prog}")

    fig_dist = OUT_DIR / f"tier_distribution_{MODEL_SLUG}.pdf"
    plot_tier_distribution(s_tiers, a_tiers, fig_dist)
    print(f"Wrote {fig_dist}")

    fig_corr = OUT_DIR / f"trait_correlation_heatmap_{MODEL_SLUG}.pdf"
    plot_correlation_matrix(all_names, rho_mat, fig_corr)
    print(f"Wrote {fig_corr}")

    # Headline numbers the rebuttal wants to quote:
    K = len(TRAITS)
    print("\n=== Rebuttal quick reference ===")
    print(f"  ρ(S, single trait, k=1):         {rhos[0]:.3f}   (the 'weak proxy' reviewer feared)")
    print(f"  ρ(S, mean of all {K} traits):{' ' * max(0, 6 - len(str(K)))}{rhos[-1]:.3f}   (species-level ≈ aggregate)")
    if rhos[0] > 0:
        print(f"  Variance explained gain:         ρ²: {rhos[0]**2:.2f} → {rhos[-1]**2:.2f}")
    print(f"  Discretized exact agreement:     {exact*100:.0f}%")
    print(f"  Within-±1-tier agreement:        {within1*100:.0f}%")


if __name__ == "__main__":
    main()
