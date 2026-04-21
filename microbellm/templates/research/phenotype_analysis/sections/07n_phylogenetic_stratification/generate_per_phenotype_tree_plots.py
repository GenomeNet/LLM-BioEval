#!/usr/bin/env python3
"""Per-phenotype circular phylogenetic trees.

Produces one PDF per phenotype in the WA benchmark showing:

- Tip colour: ground-truth phenotype value (so phylogenetic signal in
  the label itself is visible).
- Inner ring: phylum (for clade context).
- Outer ring: fraction of evaluated LLMs that predict the phenotype
  correctly for each species.

Useful for reviewer Q2 to show where the phylogenetic signal lives and
whether LLM accuracy tracks that signal. Complements the pan-phenotype
tree in ``generate_phylogenetic_tree_plot.py``.
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
OUTPUT_DIR = Path(
    "microbellm/templates/research/phenotype_analysis/sections/07n_phylogenetic_stratification/per_phenotype_trees"
)

BOOLEAN_PHENOTYPES = (
    "motility",
    "extreme_environment_tolerance",
    "biofilm_formation",
    "animal_pathogenicity",
    "host_association",
    "plant_pathogenicity",
    "spore_formation",
)
CATEGORICAL_PHENOTYPES = ("cell_shape", "biosafety_level", "gram_staining")
ALL_PHENOTYPES = BOOLEAN_PHENOTYPES + CATEGORICAL_PHENOTYPES

PHENOTYPE_TITLES = {
    "motility": "Motility",
    "extreme_environment_tolerance": "Extreme environment tolerance",
    "biofilm_formation": "Biofilm formation",
    "animal_pathogenicity": "Animal pathogenicity",
    "host_association": "Host association",
    "plant_pathogenicity": "Plant pathogenicity",
    "spore_formation": "Spore formation",
    "cell_shape": "Cell shape",
    "biosafety_level": "Biosafety level",
    "gram_staining": "Gram staining",
}

# Tip palettes ---------------------------------------------------------------
# Boolean: True / False / missing
BOOLEAN_COLORS = {
    True: "#1a7f38",    # dark green
    False: "#c0c0c0",   # light grey
    None: "#ffffff",    # missing: white (invisible on background)
}
# Categorical phenotype palettes (keys match normalized values)
CATEGORICAL_COLORS = {
    "gram_staining": {
        "gram stain positive": "#6c4fa1",
        "gram stain negative": "#ff9800",
        "gram stain variable": "#00bcd4",
    },
    "biosafety_level": {
        "biosafety level 1": "#4caf50",
        "biosafety level 2": "#ff9800",
        "biosafety level 3": "#e53935",
    },
    # Cell shape has many values — we'll use a palette per-observed-level.
    "cell_shape": None,
}
CELL_SHAPE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Phylum ring colours (same palette as phylogenetic_tree plot) --------------
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
OTHER_PHYLUM_COLOR = "#9E9E9E"
MISSING_TOKENS = {"n/a", "na", "null", "none", "nan", "undefined", "-", "unknown", "missing"}


# ---------------------------------------------------------------------------
# API + normalisation helpers
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
# Taxonomy loading
# ---------------------------------------------------------------------------

def load_leaf_metadata(path: Path) -> Dict[str, Tuple[str, str, str]]:
    """leaf_label -> (species_lower, phylum, order)."""
    leaf_meta: Dict[str, Tuple[str, str, str]] = {}
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
            if not order or order.upper() == "NA":
                order = f"{phylum} / unclassified"
            if fasta and fasta.upper() != "NA":
                leaf = fasta[:-6] if fasta.endswith(".fasta") else fasta
                leaf_meta[leaf] = (name.lower(), phylum, order)
    return leaf_meta


# ---------------------------------------------------------------------------
# Per-species per-phenotype: ground truth + model accuracy
# ---------------------------------------------------------------------------

def compute_species_phenotype_stats(
    predictions: Iterable[dict],
    gt_map: Dict[str, dict],
    species_set: set,
    phenotype: str,
) -> Tuple[Dict[str, object], Dict[str, Tuple[float, int]]]:
    """Return (species_lower -> ground truth value,
               species_lower -> (accuracy across models, n_models))."""
    truth: Dict[str, object] = {}
    for species in species_set:
        gt = gt_map.get(species)
        if not gt:
            continue
        if phenotype in CATEGORICAL_PHENOTYPES:
            v = normalize_categorical(phenotype, gt.get(phenotype))
        else:
            v = normalize_boolean(gt.get(phenotype))
        if v is not None:
            truth[species] = v

    by_species: Dict[str, List[int]] = defaultdict(list)  # 1 correct / 0 wrong
    for rec in predictions:
        species = (rec.get("binomial_name") or "").strip().lower()
        if species not in truth:
            continue
        if phenotype in CATEGORICAL_PHENOTYPES:
            p = normalize_categorical(phenotype, rec.get(phenotype))
        else:
            p = normalize_boolean(rec.get(phenotype))
        if p is None:
            continue
        by_species[species].append(1 if p == truth[species] else 0)

    accuracy: Dict[str, Tuple[float, int]] = {}
    for species, hits in by_species.items():
        if len(hits) >= 3:
            accuracy[species] = (float(np.mean(hits)), len(hits))
    return truth, accuracy


# ---------------------------------------------------------------------------
# Tree helpers (adapted from generate_phylogenetic_tree_plot.py)
# ---------------------------------------------------------------------------

def prune_tree_to(tree, keep_leaves: set) -> int:
    to_prune = [leaf for leaf in tree.get_terminals() if leaf.name not in keep_leaves]
    for leaf in to_prune:
        tree.prune(leaf)
    _collapse_single_children(tree.root)
    return len(tree.get_terminals())


def _collapse_single_children(clade) -> None:
    for child in list(clade.clades):
        _collapse_single_children(child)
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
    terminals = tree.get_terminals()
    n = len(terminals)
    angle: Dict[object, float] = {}
    for i, leaf in enumerate(terminals):
        angle[leaf] = 2.0 * math.pi * i / n

    def _set_angle(clade) -> float:
        if clade.is_terminal():
            return angle[clade]
        child_angles = [_set_angle(c) for c in clade.clades]
        angle[clade] = float(np.mean(child_angles))
        return angle[clade]

    _set_angle(tree.root)

    radius = {tree.root: 0.0}

    def _set_radius(clade) -> None:
        for c in clade.clades:
            bl = max(c.branch_length or 0.0, 0.0)
            radius[c] = radius[clade] + bl
            _set_radius(c)

    _set_radius(tree.root)
    return angle, radius


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _polar_to_xy(r: float, a: float) -> Tuple[float, float]:
    return (r * math.cos(a), r * math.sin(a))


def _build_edge_segments(tree, angle, rnorm):
    edge_segments: List[List[Tuple[float, float]]] = []
    arc_segments: List[List[Tuple[float, float]]] = []
    for clade in tree.find_clades(order="preorder"):
        for child in clade.clades:
            a = angle[child]
            edge_segments.append([
                _polar_to_xy(rnorm[clade], a),
                _polar_to_xy(rnorm[child], a),
            ])
        if clade.clades:
            child_angles = [angle[c] for c in clade.clades]
            a0, a1 = min(child_angles), max(child_angles)
            steps = max(2, int(1 + 40 * (a1 - a0) / (2 * math.pi)))
            arc_segments.append([
                _polar_to_xy(rnorm[clade], a)
                for a in np.linspace(a0, a1, steps)
            ])
    return edge_segments, arc_segments


def _ring_wedges(angles: List[float], values: List, color_fn, inner: float, outer: float):
    """Yield Wedge patches covering angular slices around each tip."""
    pairs = sorted(zip(angles, values), key=lambda p: p[0])
    sorted_angles = [p[0] for p in pairs]
    sorted_vals = [p[1] for p in pairs]
    n = len(sorted_angles)
    for i, (a, v) in enumerate(zip(sorted_angles, sorted_vals)):
        prev_a = sorted_angles[i - 1] if i > 0 else sorted_angles[-1] - 2 * math.pi
        next_a = sorted_angles[i + 1] if i < n - 1 else sorted_angles[0] + 2 * math.pi
        lo = (prev_a + a) / 2
        hi = (a + next_a) / 2
        color = color_fn(v)
        if color is None:
            continue
        yield Wedge(
            (0, 0), outer, math.degrees(lo), math.degrees(hi),
            width=outer - inner, facecolor=color, edgecolor="none", zorder=2,
        )


def plot_phenotype_tree(
    tree,
    leaf_meta: Dict[str, Tuple[str, str]],
    phenotype: str,
    truth: Dict[str, object],
    accuracy: Dict[str, Tuple[float, int]],
    output_path: Path,
) -> None:
    angle, radius = radial_layout(tree)
    max_r = max(radius.values()) or 1.0
    rnorm = {n: (radius[n] / max_r) for n in radius}
    edge_segments, arc_segments = _build_edge_segments(tree, angle, rnorm)

    # Larger canvas (10.5") so individual tip circles have enough pixels to
    # read as distinct dots rather than fusing into a band.  At 3,270 tips
    # around a circle of circumference ~2π·r, each tip needs ≥ ~2 px to look
    # like its own circle; an 8" canvas only gives ~1 px per tip.
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.set_aspect("equal")
    ax.set_axis_off()

    terminals = tree.get_terminals()

    # ---- phylum background "pie" wedges ----
    # Drawn *before* the tree so branches and tips render on top. For each
    # contiguous run of same-phylum tips we emit one wedge covering r=0 to
    # just past the deepest tip. Transparency keeps the tree legible while
    # clearly marking clade territory.
    tips_with_meta = [leaf for leaf in terminals if leaf.name in leaf_meta]
    pie_outer = 1.05
    if tips_with_meta:
        angle_phylum_pairs = sorted(
            ((angle[leaf], leaf_meta[leaf.name][1]) for leaf in tips_with_meta),
            key=lambda p: p[0],
        )
        angles_sorted = [p[0] for p in angle_phylum_pairs]
        phyla_sorted = [p[1] for p in angle_phylum_pairs]

        # Group consecutive tips sharing a phylum into runs.
        runs: List[Tuple[str, int, int]] = []  # (phylum, start_idx, end_idx)
        run_start = 0
        for i in range(1, len(phyla_sorted)):
            if phyla_sorted[i] != phyla_sorted[run_start]:
                runs.append((phyla_sorted[run_start], run_start, i - 1))
                run_start = i
        runs.append((phyla_sorted[run_start], run_start, len(phyla_sorted) - 1))

        n_tips = len(angles_sorted)
        for phylum, s_idx, e_idx in runs:
            # Wedge half-way between this run's ends and the neighbouring
            # tips so wedges butt up against each other without gaps.
            prev_a = angles_sorted[s_idx - 1] if s_idx > 0 else angles_sorted[-1] - 2 * math.pi
            next_a = angles_sorted[e_idx + 1] if e_idx < n_tips - 1 else angles_sorted[0] + 2 * math.pi
            lo = (prev_a + angles_sorted[s_idx]) / 2
            hi = (angles_sorted[e_idx] + next_a) / 2
            color = PHYLUM_COLORS.get(phylum, OTHER_PHYLUM_COLOR)
            ax.add_patch(Wedge(
                (0, 0), pie_outer, math.degrees(lo), math.degrees(hi),
                width=pie_outer,            # full pie from r=0
                facecolor=color, edgecolor="none", alpha=0.18, zorder=0,
            ))

    ax.add_collection(LineCollection(
        edge_segments + arc_segments,
        colors="#444444", linewidths=0.22, alpha=0.7, zorder=1,
    ))

    # ---- tip truth value palette ----
    is_categorical = phenotype in CATEGORICAL_PHENOTYPES
    if is_categorical and CATEGORICAL_COLORS.get(phenotype) is None:
        # cell_shape: assign palette over observed values
        observed = sorted({v for v in truth.values() if isinstance(v, str)})
        cat_map = {v: CELL_SHAPE_PALETTE[i % len(CELL_SHAPE_PALETTE)]
                   for i, v in enumerate(observed)}
    elif is_categorical:
        cat_map = CATEGORICAL_COLORS[phenotype]
    else:
        cat_map = None

    # ---- gather per-tip data ----
    tip_colors: List[str] = []
    acc_angles: List[float] = []
    acc_vals: List[float] = []

    phylum_angles: List[float] = []
    phylum_values: List[str] = []

    truth_counter: Counter = Counter()
    acc_stats: List[float] = []

    for leaf in terminals:
        meta = leaf_meta.get(leaf.name)
        if not meta:
            continue
        species, phylum, _order = meta
        a = angle[leaf]

        # Phylum ring
        phylum_angles.append(a)
        phylum_values.append(phylum)

        # Tip colour = truth value
        tv = truth.get(species)
        if tv is None:
            tip_color = "#efefef"
        elif is_categorical:
            tip_color = cat_map.get(tv, "#efefef")
        else:
            tip_color = BOOLEAN_COLORS[bool(tv)]
        tip_colors.append(tip_color)
        if tv is not None:
            truth_counter[tv] += 1

        # Outer ring: accuracy on this phenotype
        acc = accuracy.get(species)
        acc_angles.append(a)
        if acc is None:
            acc_vals.append(float("nan"))
        else:
            acc_vals.append(acc[0])
            acc_stats.append(acc[0])

    # GraPhlAn-style layout (inside -> out):
    #   tree branches        -> phylogram, r = actual cumulative branch length
    #   tip markers          -> bubble at EACH tip's own branch-end radius
    #                           (spatially separates tips because branches
    #                            have different lengths; avoids the single
    #                            uniform-ring "band" effect).
    #   accuracy ring        -> r ∈ [1.06, 1.14]
    #   phylum ring          -> r ∈ [1.17, 1.21]
    #
    # This matches the GraPhlAn phylogram + annotation rings aesthetic.

    # ---- tip dots at each tip's actual radius ----
    mapped_terminals = [leaf for leaf in terminals if leaf.name in leaf_meta]
    tip_xs = [rnorm[leaf] * math.cos(angle[leaf]) for leaf in mapped_terminals]
    tip_ys = [rnorm[leaf] * math.sin(angle[leaf]) for leaf in mapped_terminals]
    ax.scatter(
        tip_xs, tip_ys, s=22, c=tip_colors,
        edgecolors="#222222", linewidths=0.25, zorder=4,
    )

    # ---- accuracy ring (aggregated per Order) ----
    # Phylum was too coarse (only ~9 arcs) and per-species too noisy;
    # aggregating at Order level gives a finer picture while keeping
    # each arc statistically meaningful. Background pie still shows the
    # Phylum level, so both taxonomic scales read at once.
    ring_acc_inner = 1.06
    ring_acc_outer = 1.14
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("#e6e6e6")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

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

        # Per-Order mean per-species accuracy on this phenotype.
        order_accs: Dict[str, List[float]] = defaultdict(list)
        for leaf in tips_with_meta:
            species = leaf_meta[leaf.name][0]
            order = leaf_meta[leaf.name][2]
            acc = accuracy.get(species)
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
                (0, 0), ring_acc_outer, math.degrees(lo), math.degrees(hi),
                width=ring_acc_outer - ring_acc_inner,
                facecolor=color, edgecolor="white", linewidth=0.3,
                zorder=2,
            ))

    # Phylum is now encoded as tinted background pie wedges (drawn before
    # the tree above); no separate ring needed.
    for r in (ring_acc_inner, ring_acc_outer):
        ax.add_patch(plt.Circle(
            (0, 0), r, fill=False, color="#444444",
            linewidth=0.25, zorder=5,
        ))

    # ---- colorbar ----
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes((0.90, 0.07, 0.015, 0.25))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Fraction of models correct\n(on this phenotype)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ---- truth-value legend (top-left) ----
    if is_categorical:
        truth_legend = []
        for val, color in (cat_map.items() if isinstance(cat_map, dict) else []):
            n_val = truth_counter.get(val, 0)
            if n_val == 0:
                continue
            truth_legend.append((val, color, n_val))
        truth_legend.sort(key=lambda e: -e[2])
        truth_handles = [
            plt.Line2D([], [], marker="o", linestyle="", markeredgecolor="none",
                       markerfacecolor=color, markersize=6,
                       label=f"{label} (n={n})")
            for label, color, n in truth_legend
        ]
    else:
        true_n = truth_counter.get(True, 0)
        false_n = truth_counter.get(False, 0)
        truth_handles = [
            plt.Line2D([], [], marker="o", linestyle="", markeredgecolor="none",
                       markerfacecolor=BOOLEAN_COLORS[True], markersize=6,
                       label=f"Positive (n={true_n})"),
            plt.Line2D([], [], marker="o", linestyle="", markeredgecolor="#888",
                       markerfacecolor=BOOLEAN_COLORS[False], markersize=6,
                       label=f"Negative (n={false_n})"),
        ]
    truth_legend_obj = ax.legend(
        handles=truth_handles, loc="upper left", bbox_to_anchor=(-0.04, 1.02),
        fontsize=7, frameon=False, handletextpad=0.3,
        title=f"Ground truth: {PHENOTYPE_TITLES.get(phenotype, phenotype)}",
        title_fontsize=8,
    )
    ax.add_artist(truth_legend_obj)

    # ---- phylum legend (bottom-left) ----
    phylum_counts = Counter(phylum_values)
    phylum_items = [
        (p, PHYLUM_COLORS[p], phylum_counts[p])
        for p in PHYLUM_COLORS if phylum_counts.get(p, 0) > 0
    ]
    phylum_items.sort(key=lambda e: -e[2])
    other_n = sum(c for p, c in phylum_counts.items() if p not in PHYLUM_COLORS)
    if other_n:
        phylum_items.append(("Other", OTHER_PHYLUM_COLOR, other_n))
    phylum_handles = [
        plt.Line2D([], [], marker="s", linestyle="", markeredgecolor="none",
                   markerfacecolor=color, markersize=6,
                   label=f"{name} (n={n})")
        for name, color, n in phylum_items
    ]
    ax.legend(
        handles=phylum_handles, loc="lower left", bbox_to_anchor=(-0.04, -0.02),
        fontsize=7, frameon=False, handletextpad=0.3,
        title="Phylum (background)", title_fontsize=8,
    )

    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-1.20, 1.20)

    n_tips = len(terminals)
    n_scored = sum(1 for v in acc_vals if math.isfinite(v))
    n_truth = len(truth)
    mean_acc = float(np.mean(acc_stats)) if acc_stats else float("nan")

    fig.suptitle(
        f"Phylogenetic distribution of {PHENOTYPE_TITLES.get(phenotype, phenotype)}",
        fontsize=11, y=0.995,
    )
    fig.text(
        0.5, 0.955,
        "Tips: ground-truth value   ·   Outer ring: fraction of models correct",
        ha="center", fontsize=8, color="#333",
    )
    fig.text(
        0.5, 0.02,
        f"{n_tips} tips   ·   {n_truth} with ground-truth label   ·   "
        f"{n_scored} scored by ≥3 models   ·   mean correct-rate = {mean_acc:.2f}",
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
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--phenotypes", nargs="*", default=list(ALL_PHENOTYPES),
        help="Subset of phenotypes to render (default: all 10).",
    )
    args = parser.parse_args()

    leaf_meta = load_leaf_metadata(Path(args.taxonomy))
    print(f"Taxonomy: {len(leaf_meta)} leaf-mappable WA species.")

    print(f"Parsing tree {args.tree} ...")
    tree_master = Phylo.read(args.tree, "newick")
    kept = prune_tree_to(tree_master, set(leaf_meta.keys()))
    print(f"Pruned to {kept} WA-matched leaves.")

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for phenotype in args.phenotypes:
        if phenotype not in ALL_PHENOTYPES:
            print(f"  ! Skipping unknown phenotype: {phenotype}", file=sys.stderr)
            continue
        print(f"Rendering {phenotype} ...")
        truth, accuracy = compute_species_phenotype_stats(
            predictions, gt_map, species_set, phenotype,
        )
        out_path = output_dir / f"tree_{phenotype}.pdf"
        # Re-parse the tree per phenotype isn't necessary; the layout only
        # depends on topology, which is unchanged.
        plot_phenotype_tree(
            tree_master, leaf_meta, phenotype, truth, accuracy, out_path,
        )
        print(f"  -> {out_path}  (n_truth={len(truth)}, n_scored={len(accuracy)})")


if __name__ == "__main__":
    main()
