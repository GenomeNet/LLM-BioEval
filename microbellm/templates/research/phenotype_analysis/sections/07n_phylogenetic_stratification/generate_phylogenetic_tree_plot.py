#!/usr/bin/env python3
"""Circular phylogenetic tree of the WA benchmark species, coloured by
phylum with an outer ring encoding per-species mean LLM prediction
accuracy.

Supplementary figure for reviewer Q2 alongside the phylum heatmap in
``generate_phylogenetic_stratification_plot.py``. Pipeline:

1. Parse the FastTree/IQ-TREE Newick file at
   ``microbellm/static/data/wa_phylogeny.tre``.
2. Map leaf labels (assembly basenames) to WA species via
   ``supp_table_annot_taxon.tsv`` and prune the tree to WA tips.
3. Compute per-species mean balanced accuracy across the ten phenotypes
   and all evaluated models (mirroring the 07l/07n metric logic but
   keyed on species instead of phylum).
4. Render a circular phylogram (radial layout) with tip dots coloured
   by phylum and an outer heat ring encoding per-species accuracy.
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
from Bio import Phylo
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Wedge

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

API_URL = "http://localhost:5050"
DATASET_NAME = "WA_Test_Dataset"
SPECIES_FILE = "wa_with_gcount.txt"
TAXONOMY_PATH = Path("microbellm/static/data/supp_table_annot_taxon.tsv")
TREE_PATH = Path("microbellm/static/data/wa_phylogeny.tre")
OUTPUT_PATH = Path(
    "microbellm/templates/research/phenotype_analysis/sections/07n_phylogenetic_stratification/phylogenetic_tree.pdf"
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
CATEGORICAL_PHENOTYPES = {"cell_shape", "biosafety_level", "gram_staining"}
MISSING_TOKENS = {"n/a", "na", "null", "none", "nan", "undefined", "-", "unknown", "missing"}
PER_SPECIES_MIN_MODELS = 3  # species with fewer evaluated models -> grey tip

# Phylum colour palette — coordinated with the heatmap figure; unknown
# / merged-into-"Other" phyla get a neutral grey.
PHYLUM_COLORS = {
    "Pseudomonadota": "#1F77B4",
    "Actinomycetota": "#FF7F0F",
    "Bacillota": "#2BA02B",
    "Bacteroidota": "#9467BD",
    "Euryarchaeota": "#8C564C",
    "Campylobacterota": "#D62728",
    "Spirochaetota": "#BCBD21",
    "Myxococcota": "#17BECF",
}
OTHER_COLOR = "#9E9E9E"
KEPT_PHYLA = set(PHYLUM_COLORS.keys())


# ---------------------------------------------------------------------------
# API + normalisation helpers (shared with sister scripts)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Taxonomy loading: leaf_label -> (species_lower, phylum)
# ---------------------------------------------------------------------------

def load_leaf_metadata(path: Path) -> Tuple[Dict[str, Tuple[str, str, str]], Dict[str, str], Counter]:
    """Return (leaf_label -> (species_lower, phylum, order),
               species_lower -> phylum, phylum counts)."""
    leaf_meta: Dict[str, Tuple[str, str, str]] = {}
    species_phylum: Dict[str, str] = {}
    counts: Counter = Counter()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if (row.get("Member of WA subset") or "").strip().upper() != "TRUE":
                continue
            name = (row.get("Binomial name") or "").strip()
            phylum = (row.get("Phylum") or "").strip()
            order = (row.get("Order") or "").strip()
            fasta = (row.get("Fasta file") or "").strip()
            if not name or not phylum or phylum.upper() == "NA":
                continue
            species_phylum[name.lower()] = phylum
            counts[phylum] += 1
            if not order or order.upper() == "NA":
                order = f"{phylum} / unclassified"
            if fasta and fasta.upper() != "NA":
                leaf = fasta[:-6] if fasta.endswith(".fasta") else fasta
                leaf_meta[leaf] = (name.lower(), phylum, order)
    return leaf_meta, species_phylum, counts


# ---------------------------------------------------------------------------
# Per-species mean balanced accuracy
# ---------------------------------------------------------------------------

def compute_species_accuracy(
    predictions: Iterable[dict],
    gt_map: Dict[str, dict],
    species_set: set,
) -> Dict[str, Tuple[float, int]]:
    """Return species_lower -> (mean balanced accuracy across phenotypes
    and models, n_models_contributing)."""

    # Bucket predictions by (model, species).
    rows_by_model_species: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for rec in predictions:
        species = (rec.get("binomial_name") or "").strip().lower()
        if not species or species not in species_set:
            continue
        model = rec.get("model")
        if not model:
            continue
        rows_by_model_species[(model, species)].append(rec)

    # For each (model, species): compute a per-phenotype accuracy (0/1 per
    # comparison). Since a species has a single ground-truth value per
    # phenotype, we score each (model, species, phenotype) as 1 if the model's
    # normalised prediction matches the ground truth, 0 otherwise, NaN if
    # either side is missing. Average across phenotypes -> (model, species)
    # accuracy. Then average across models -> species accuracy.

    species_scores: Dict[str, List[float]] = defaultdict(list)
    for (model, species), rows in rows_by_model_species.items():
        gt = gt_map.get(species)
        if not gt:
            continue
        row = rows[-1]  # deterministic: most recent wins when duplicates appear
        pheno_scores: List[float] = []
        for phenotype in PHENOTYPES:
            if phenotype in CATEGORICAL_PHENOTYPES:
                tv = normalize_categorical(phenotype, gt.get(phenotype))
                pv = normalize_categorical(phenotype, row.get(phenotype))
            else:
                tv = normalize_boolean(gt.get(phenotype))
                pv = normalize_boolean(row.get(phenotype))
            if tv is None or pv is None:
                continue
            pheno_scores.append(1.0 if tv == pv else 0.0)
        if pheno_scores:
            species_scores[species].append(float(np.mean(pheno_scores)))

    out: Dict[str, Tuple[float, int]] = {}
    for species, scores in species_scores.items():
        if len(scores) < PER_SPECIES_MIN_MODELS:
            continue
        out[species] = (float(np.mean(scores)), len(scores))
    return out


# ---------------------------------------------------------------------------
# Tree pruning and radial layout
# ---------------------------------------------------------------------------

def prune_tree_to(tree, keep_leaves: set) -> int:
    """Prune leaves not in keep_leaves. Returns number of retained tips."""
    to_prune: List = []
    for leaf in tree.get_terminals():
        if leaf.name not in keep_leaves:
            to_prune.append(leaf)
    for leaf in to_prune:
        tree.prune(leaf)
    # Collapse resulting single-child internal nodes (they artefact the layout).
    _collapse_single_children(tree.root)
    return len(tree.get_terminals())


def _collapse_single_children(clade) -> None:
    """In-place collapse of degree-2 internal nodes (common after pruning)."""
    # Recurse first
    for child in list(clade.clades):
        _collapse_single_children(child)
    # If this node has exactly one child, merge their branch lengths.
    while len(clade.clades) == 1:
        only = clade.clades[0]
        if only.is_terminal():
            break
        clade.clades = only.clades
        if clade.branch_length is None:
            clade.branch_length = only.branch_length
        elif only.branch_length is not None:
            clade.branch_length += only.branch_length


def radial_layout(tree) -> Tuple[Dict[object, float], Dict[object, float]]:
    """Return (angle_of_node, radius_of_node) dicts.

    Angle: leaves evenly spaced in [0, 2π); internal nodes take the mean
    of their direct children's angles. Radius: cumulative branch length
    from the root (root at radius 0).
    """
    terminals = tree.get_terminals()
    n = len(terminals)
    angle = {}
    for i, leaf in enumerate(terminals):
        angle[leaf] = 2.0 * math.pi * i / n

    # Post-order traversal for internal angles
    def _set_internal(clade) -> float:
        if clade.is_terminal():
            return angle[clade]
        child_angles = [_set_internal(c) for c in clade.clades]
        angle[clade] = float(np.mean(child_angles))
        return angle[clade]

    _set_internal(tree.root)

    # Radii via cumulative branch length
    radius = {tree.root: 0.0}

    def _set_radius(clade) -> None:
        for c in clade.clades:
            bl = c.branch_length if c.branch_length is not None else 0.0
            # Guard against negative branch lengths occasionally produced by
            # FastTree / IQ-TREE after rooting.
            bl = max(bl, 0.0)
            radius[c] = radius[clade] + bl
            _set_radius(c)

    _set_radius(tree.root)
    return angle, radius


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_tree(
    tree,
    leaf_meta: Dict[str, Tuple[str, str]],
    species_accuracy: Dict[str, Tuple[float, int]],
    output_path: Path,
) -> None:
    angle, radius = radial_layout(tree)
    max_r = max(radius.values()) or 1.0

    # Normalise radii to [0, 1] to make the ring geometry easy.
    rnorm = {n: (radius[n] / max_r) for n in radius}

    # ---------- Build edge line segments ----------
    edge_segments: List[List[Tuple[float, float]]] = []
    arc_segments: List[List[Tuple[float, float]]] = []  # arc approximated by polyline

    def polar_to_xy(r: float, a: float) -> Tuple[float, float]:
        return (r * math.cos(a), r * math.sin(a))

    for clade in tree.find_clades(order="preorder"):
        for child in clade.clades:
            # Radial segment: from parent radius @ child angle -> child radius @ child angle
            p_r = rnorm[clade]
            c_r = rnorm[child]
            a = angle[child]
            edge_segments.append([polar_to_xy(p_r, a), polar_to_xy(c_r, a)])

        if clade.clades:
            # Arc at parent radius spanning children's angular extent.
            child_angles = [angle[c] for c in clade.clades]
            a0, a1 = min(child_angles), max(child_angles)
            steps = max(2, int(1 + 40 * (a1 - a0) / (2 * math.pi)))
            arc_points = [polar_to_xy(rnorm[clade], a)
                          for a in np.linspace(a0, a1, steps)]
            arc_segments.append(arc_points)

    # ---------- Figure ----------
    # Larger canvas (10.5") so that at ~3,270 tips around the circle each
    # tip bubble gets enough pixels to read as a distinct dot rather than
    # fusing into a band.
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.set_aspect("equal")
    ax.set_axis_off()

    terminals = tree.get_terminals()

    # ---------- Phylum background "pie" wedges ----------
    # Drawn first (zorder=0) so branches and tips sit on top. Consecutive
    # tips of the same phylum are grouped into one wedge that fills from
    # r=0 to just past the deepest tip — GraPhlAn-style clade shading.
    tips_with_meta = [leaf for leaf in terminals if leaf.name in leaf_meta]
    pie_outer = 1.05
    phylum_counts: Counter = Counter()
    if tips_with_meta:
        angle_phylum_pairs = sorted(
            ((angle[leaf], leaf_meta[leaf.name][1]) for leaf in tips_with_meta),
            key=lambda p: p[0],
        )
        angles_sorted = [p[0] for p in angle_phylum_pairs]
        phyla_sorted = [p[1] for p in angle_phylum_pairs]

        # Group contiguous same-phylum tips into angular runs.
        runs: List[Tuple[str, int, int]] = []
        run_start = 0
        for i in range(1, len(phyla_sorted)):
            if phyla_sorted[i] != phyla_sorted[run_start]:
                runs.append((phyla_sorted[run_start], run_start, i - 1))
                run_start = i
        runs.append((phyla_sorted[run_start], run_start, len(phyla_sorted) - 1))

        n_tips = len(angles_sorted)
        for phylum, s_idx, e_idx in runs:
            prev_a = angles_sorted[s_idx - 1] if s_idx > 0 else angles_sorted[-1] - 2 * math.pi
            next_a = angles_sorted[e_idx + 1] if e_idx < n_tips - 1 else angles_sorted[0] + 2 * math.pi
            lo = (prev_a + angles_sorted[s_idx]) / 2
            hi = (angles_sorted[e_idx] + next_a) / 2
            color = PHYLUM_COLORS.get(phylum, OTHER_COLOR)
            ax.add_patch(Wedge(
                (0, 0), pie_outer, math.degrees(lo), math.degrees(hi),
                width=pie_outer,  # full pie from r=0
                facecolor=color, edgecolor="none", alpha=0.18, zorder=0,
            ))

    edges = LineCollection(
        edge_segments + arc_segments,
        colors="#444444", linewidths=0.22, alpha=0.7, zorder=1,
    )
    ax.add_collection(edges)

    # ---------- Tip bubbles (GraPhlAn style: at actual branch-end radii) ----------
    tip_xs, tip_ys, tip_colors = [], [], []
    tip_acc_values: List[float] = []
    tip_acc_angles: List[float] = []
    accuracies_for_stats: List[float] = []

    for leaf in terminals:
        meta = leaf_meta.get(leaf.name)
        if not meta:
            continue
        species, phylum, _order = meta
        color = PHYLUM_COLORS.get(phylum, OTHER_COLOR)
        a = angle[leaf]
        # Place the tip at its true phylogram radius so bubbles scatter
        # across varying distances instead of packing into one ring.
        r = rnorm[leaf]
        tip_x, tip_y = polar_to_xy(r, a)
        tip_xs.append(tip_x)
        tip_ys.append(tip_y)
        tip_colors.append(color)
        phylum_counts[phylum] += 1

        acc = species_accuracy.get(species)
        if acc is None:
            tip_acc_values.append(float("nan"))
        else:
            tip_acc_values.append(acc[0])
            accuracies_for_stats.append(acc[0])
        tip_acc_angles.append(a)

    ax.scatter(
        tip_xs, tip_ys, s=22, c=tip_colors,
        edgecolors="#222222", linewidths=0.35, zorder=3,
    )

    # ---------- Outer accuracy ring (aggregated per Order) ----------
    # Phylum-level aggregation was too coarse (only ~9 arcs) and
    # per-species was too noisy; Order is a nice middle ground — it still
    # gives tens of distinct arcs but each one has enough species for a
    # stable mean. Background pie stays at phylum level, so the two
    # taxonomic levels are visible simultaneously.
    ring_inner = 1.08
    ring_outer = 1.18
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("#e6e6e6")
    norm = TwoSlopeNorm(vmin=0.35, vcenter=0.65, vmax=0.9)

    if tips_with_meta:
        # Build contiguous-same-Order runs along the angular layout.
        angle_order_pairs = sorted(
            ((angle[leaf], leaf_meta[leaf.name][2]) for leaf in tips_with_meta),
            key=lambda p: p[0],
        )
        order_angles_sorted = [p[0] for p in angle_order_pairs]
        orders_sorted = [p[1] for p in angle_order_pairs]
        order_runs: List[Tuple[str, int, int]] = []
        run_start = 0
        for i in range(1, len(orders_sorted)):
            if orders_sorted[i] != orders_sorted[run_start]:
                order_runs.append((orders_sorted[run_start], run_start, i - 1))
                run_start = i
        order_runs.append((orders_sorted[run_start], run_start, len(orders_sorted) - 1))
        n_order_tips = len(order_angles_sorted)

        # Per-Order mean of per-species accuracies.
        order_accs: Dict[str, List[float]] = defaultdict(list)
        for leaf in tips_with_meta:
            species = leaf_meta[leaf.name][0]
            order = leaf_meta[leaf.name][2]
            acc = species_accuracy.get(species)
            if acc is not None:
                order_accs[order].append(acc[0])
        order_mean_acc = {o: float(np.mean(vs)) for o, vs in order_accs.items() if vs}

        for order, s_idx, e_idx in order_runs:
            prev_a = order_angles_sorted[s_idx - 1] if s_idx > 0 else order_angles_sorted[-1] - 2 * math.pi
            next_a = order_angles_sorted[e_idx + 1] if e_idx < n_order_tips - 1 else order_angles_sorted[0] + 2 * math.pi
            lo = (prev_a + order_angles_sorted[s_idx]) / 2
            hi = (order_angles_sorted[e_idx] + next_a) / 2
            val = order_mean_acc.get(order, float("nan"))
            color = cmap(norm(val)) if math.isfinite(val) else "#e6e6e6"
            ax.add_patch(Wedge(
                (0, 0), ring_outer, math.degrees(lo), math.degrees(hi),
                width=ring_outer - ring_inner,
                facecolor=color, edgecolor="white", linewidth=0.25,
                zorder=2,
            ))

    # Subtle ring borders
    for r in (ring_inner, ring_outer):
        circle = plt.Circle((0, 0), r, fill=False, color="#444444", linewidth=0.3, zorder=4)
        ax.add_patch(circle)

    # ---------- Colorbar for the ring ----------
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes((0.88, 0.07, 0.015, 0.25))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Mean balanced accuracy\n(across models & phenotypes)",
                   fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ---------- Phylum legend ----------
    legend_entries: List[Tuple[str, str, int]] = []
    for phylum, color in PHYLUM_COLORS.items():
        if phylum_counts.get(phylum, 0) == 0:
            continue
        legend_entries.append((phylum, color, phylum_counts[phylum]))
    # Sort by count desc
    legend_entries.sort(key=lambda e: -e[2])
    other_count = sum(
        c for p, c in phylum_counts.items() if p not in PHYLUM_COLORS
    )
    if other_count:
        legend_entries.append(("Other", OTHER_COLOR, other_count))

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markeredgecolor="none",
                   markerfacecolor=color, markersize=5,
                   label=f"{name} (n={n})")
        for name, color, n in legend_entries
    ]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(-0.04, 1.02),
        fontsize=7, frameon=False, handletextpad=0.3, borderaxespad=0,
        title="Phylum (background)", title_fontsize=8,
    )

    ax.set_xlim(-1.24, 1.24)
    ax.set_ylim(-1.24, 1.24)

    # Title + stats footer
    n_tips = len(terminals)
    n_scored = sum(1 for v in tip_acc_values if math.isfinite(v))
    overall_mean = float(np.mean(accuracies_for_stats)) if accuracies_for_stats else float("nan")
    fig.suptitle(
        "Phylogenetic tree of WA benchmark species — "
        f"{n_tips} species, coloured by phylum; ring = mean LLM accuracy",
        fontsize=10, y=0.98,
    )
    fig.text(
        0.5, 0.015,
        f"Per-species accuracy computed over {n_scored} species with ≥{PER_SPECIES_MIN_MODELS} evaluated models   ·   overall mean = {overall_mean:.2f}",
        ha="center", fontsize=7, color="#444",
    )

    fig.savefig(
        output_path, facecolor="white", edgecolor="none",
        bbox_inches="tight", pad_inches=0.04,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--species-file", default=SPECIES_FILE)
    parser.add_argument("--taxonomy", default=str(TAXONOMY_PATH))
    parser.add_argument("--tree", default=str(TREE_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    leaf_meta, species_phylum, phylum_counts = load_leaf_metadata(Path(args.taxonomy))
    print(f"Taxonomy: {len(species_phylum)} WA species, "
          f"{len(leaf_meta)} with mappable Fasta file basenames.")

    print(f"Parsing tree from {args.tree} ...")
    tree = Phylo.read(args.tree, "newick")
    total_leaves = len(tree.get_terminals())
    print(f"Tree has {total_leaves} leaves before pruning.")

    keep = set(leaf_meta.keys())
    kept = prune_tree_to(tree, keep)
    print(f"Pruned to {kept} WA-matched leaves.")

    # API fetch for per-species accuracy
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
        sys.exit("Empty predictions or ground-truth payload.")

    gt_map = {
        rec["binomial_name"].lower(): rec
        for rec in gt_records if rec.get("binomial_name")
    }
    species_set = {meta[0] for meta in leaf_meta.values()}
    species_accuracy = compute_species_accuracy(predictions, gt_map, species_set)
    print(f"Per-species accuracy computed for {len(species_accuracy)} species.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_tree(tree, leaf_meta, species_accuracy, output_path)
    print(f"Saved tree figure to {output_path}")


if __name__ == "__main__":
    main()
