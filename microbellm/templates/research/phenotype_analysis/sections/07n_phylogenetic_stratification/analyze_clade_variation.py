#!/usr/bin/env python3
"""Per-Order accuracy analysis grouped by Phylum.

Reuses the tree figure's per-species metric (mean per-phenotype correctness
averaged across models) and aggregates to NCBI Order, then summarises the
within-phylum distribution. Produces the Order-level numbers cited in the
reviewer-Q2 response:

* Range of Order-level accuracies across the 30 Orders with >=10 species
  (0.72 - 0.92).
* Best- and worst-performing Orders (overall and per Phylum).
* Phyla with the largest / smallest within-phylum range, i.e. the most
  variable and most consistent clades (Actinomycetota and Pseudomonadota
  are both 0.18; Bacteroidota is 0.05).
* The identity of the "pale-green block" in Actinomycetota
  (Kitasatosporales, BA 0.74, n=329, the largest Order in the phylum).

Inputs:
  microbellm/static/data/supp_table_annot_taxon.tsv   taxonomy + GT
  microbellm.db processing_results table              LLM predictions

Prints a plain-text summary to stdout. No figure output; the circular
phylogeny with the per-Order outer ring is produced by
``generate_phylogenetic_tree_plot.py`` in this same directory.

Run from the repo root:
    python microbellm/templates/research/phenotype_analysis/sections/\
07n_phylogenetic_stratification/analyze_clade_variation.py
"""

from __future__ import annotations

import csv
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

# Resolve the repo root from this file's location so the script works from
# any working directory and on any checkout.
REPO = Path(__file__).resolve().parents[6]
TAXONOMY = REPO / "microbellm" / "static" / "data" / "supp_table_annot_taxon.tsv"
DB = REPO / "microbellm.db"

# Phenotypes evaluated in the tree figure (matches generate_phylogenetic_tree_plot.py).
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
CATEGORICAL = {"cell_shape", "biosafety_level", "gram_staining"}

# Map DB column -> column in supp_table_annot_taxon.tsv that carries the
# ground-truth value for that phenotype.
TSV_COL = {
    "gram_staining": "Gram staining",
    "motility": "Motility",
    "extreme_environment_tolerance": "Extreme environment tolerance",
    "biofilm_formation": "Biofilm formation",
    "animal_pathogenicity": "Animal pathogenicity",
    "biosafety_level": "Biosafety level",
    "host_association": "Host association",
    "plant_pathogenicity": "Plant pathogenicity",
    "spore_formation": "Spore formation",
    "cell_shape": "Cell shape",
}

MISSING = {
    "n/a", "na", "null", "none", "nan", "undefined", "-", "unknown", "missing", "",
}
MIN_MODELS = 3  # species with fewer evaluated models are excluded (tree script default)
MIN_ORDER_SPECIES_FOR_TABLES = 10  # used for the "top 10 / bottom 10 Orders" tables


# --------------------------------------------------------------------------- #
# Normalisation helpers (kept identical to generate_phylogenetic_tree_plot.py
# so the numbers match the figure)
# --------------------------------------------------------------------------- #

def norm(v):
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in MISSING:
        return None
    return s.lower()


def norm_bool(v):
    s = norm(v)
    if s is None:
        return None
    if s in {"true", "1", "yes", "t", "y"}:
        return True
    if s in {"false", "0", "no", "f", "n"}:
        return False
    return None


def norm_cat(phen, v):
    s = norm(v)
    if s is None:
        return None
    if phen == "gram_staining":
        for k in ("positive", "negative", "variable"):
            if k in s:
                return f"gram stain {k}"
    if phen == "biosafety_level":
        for k in ("1", "2", "3"):
            if k in s:
                return f"biosafety level {k}"
    return s


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    # 1. Load taxonomy + GT from TSV (WA species only).
    gt_by_species = {}
    tax_by_species = {}  # species_lower -> (phylum, order)
    with TAXONOMY.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if (row.get("Member of WA subset") or "").strip().upper() != "TRUE":
                continue
            name = (row.get("Binomial name") or "").strip().lower()
            phy = (row.get("Phylum") or "").strip()
            order = (row.get("Order") or "").strip()
            if not name or not phy or phy.upper() == "NA":
                continue
            if not order or order.upper() == "NA":
                order = f"{phy} / unclassified"
            tax_by_species[name] = (phy, order)
            gt_by_species[name] = {p: row.get(TSV_COL[p]) for p in PHENOTYPES}

    print(f"WA species in taxonomy: {len(tax_by_species)}")

    # 2. Pull predictions from DB.
    conn = sqlite3.connect(DB)
    cols = ", ".join(PHENOTYPES)
    rows = conn.execute(
        f"""
        SELECT model, binomial_name, {cols}
        FROM processing_results
        WHERE user_template = 'templates/user/template1_phenotype.txt'
          AND species_file = 'wa_with_gcount.txt'
          AND status = 'completed'
        """
    ).fetchall()
    print(f"Prediction rows: {len(rows)}")

    # 3. Per-(model, species) mean per-phenotype correctness.
    per_pair = {}  # (model, species_lower) -> [0/1 per phenotype]
    for r in rows:
        model, sp = r[0], (r[1] or "").strip().lower()
        if sp not in gt_by_species:
            continue
        scores = []
        for i, phen in enumerate(PHENOTYPES):
            pred_raw = r[2 + i]
            gt_raw = gt_by_species[sp][phen]
            if phen in CATEGORICAL:
                gt, pr = norm_cat(phen, gt_raw), norm_cat(phen, pred_raw)
            else:
                gt, pr = norm_bool(gt_raw), norm_bool(pred_raw)
            if gt is None or pr is None:
                continue
            scores.append(1.0 if gt == pr else 0.0)
        if scores:
            per_pair.setdefault((model, sp), []).extend(scores)

    # 4. per-species mean (average of per-model means), require >= MIN_MODELS.
    model_mean_by_species = defaultdict(list)
    for (model, sp), scores in per_pair.items():
        model_mean_by_species[sp].append(sum(scores) / len(scores))

    species_acc = {
        sp: sum(ms) / len(ms)
        for sp, ms in model_mean_by_species.items()
        if len(ms) >= MIN_MODELS
    }
    print(f"Species with accuracy (>= {MIN_MODELS} models): {len(species_acc)}")

    # 5. Per-Order mean + Phylum lookup.
    order_species = defaultdict(list)
    order_phylum = {}
    for sp, acc in species_acc.items():
        phy, order = tax_by_species[sp]
        order_species[order].append(acc)
        order_phylum[order] = phy
    order_mean = {o: sum(v) / len(v) for o, v in order_species.items()}
    order_n = {o: len(v) for o, v in order_species.items()}

    # 6. Aggregate by Phylum: collect Order-level means for each Phylum.
    phylum_orders = defaultdict(list)  # phylum -> [(order, mean, n), ...]
    for o, m in order_mean.items():
        phylum_orders[order_phylum[o]].append((o, m, order_n[o]))

    global_mean = sum(species_acc.values()) / len(species_acc)
    print(f"\nGlobal per-species mean accuracy: {global_mean:.3f}\n")

    # Per-Phylum summary
    print(f"{'Phylum':<22} {'n_orders':>9} {'n_species':>10} {'min':>6} {'max':>6} {'range':>7} "
          f"{'best Order':<35} {'worst Order':<35}")
    print("-" * 150)
    phylum_stats = []
    for phy, lst in sorted(phylum_orders.items(), key=lambda p: -len(p[1])):
        # Filter to Orders with >=5 species for best/worst picks (tiny-n noise).
        lst_filt = [x for x in lst if x[2] >= 5] or lst
        means = [m for _, m, _ in lst_filt]
        best = max(lst_filt, key=lambda x: x[1])
        worst = min(lst_filt, key=lambda x: x[1])
        n_species = sum(n for _, _, n in lst)
        best_label = f"{best[0][:30]} ({best[2]})"
        worst_label = f"{worst[0][:30]} ({worst[2]})"
        print(f"{phy:<22} {len(lst):>9} {n_species:>10} "
              f"{min(means):>6.2f} {max(means):>6.2f} {max(means) - min(means):>7.2f} "
              f"{best_label:<35} {worst_label:<35}")
        phylum_stats.append(
            (phy, len(lst), n_species, min(means), max(means),
             max(means) - min(means), best, worst, means)
        )

    # Ranking by within-phylum range
    print("\n=== Phyla ranked by within-phylum range (most -> least variation) ===")
    for phy, n_ord, n_sp, mn, mx, rg, best, worst, means in sorted(
        phylum_stats, key=lambda r: -r[5]
    ):
        if len(means) >= 4:
            quantiles = statistics.quantiles(means, n=4)
            iqr = quantiles[2] - quantiles[0]
            iqr_str = f"{iqr:.2f}"
        else:
            iqr_str = "n/a"
        print(f"  {phy:<22} range={rg:.2f}  n_orders={n_ord}  IQR={iqr_str}")

    # Actinomycetota Orders sorted low-to-high (the PI's pale-green block).
    print("\n=== Actinomycetota Orders sorted by mean accuracy (low -> high) ===")
    acti = phylum_orders.get("Actinomycetota", [])
    acti.sort(key=lambda x: x[1])
    print(f"{'Order':<40} {'mean':>6} {'n_species':>10}")
    for o, m, n in acti:
        if n >= 3:
            print(f"  {o:<40} {m:>6.2f} {n:>10}")

    # Overall top / bottom Orders.
    print(f"\n=== Top 10 Orders overall (>= {MIN_ORDER_SPECIES_FOR_TABLES} species) ===")
    all_orders = [
        (o, m, order_n[o], order_phylum[o])
        for o, m in order_mean.items()
        if order_n[o] >= MIN_ORDER_SPECIES_FOR_TABLES
    ]
    for o, m, n, p in sorted(all_orders, key=lambda x: -x[1])[:10]:
        print(f"  {o:<32} {m:.2f}  n={n:>4}  [{p}]")

    print(f"\n=== Bottom 10 Orders overall (>= {MIN_ORDER_SPECIES_FOR_TABLES} species) ===")
    for o, m, n, p in sorted(all_orders, key=lambda x: x[1])[:10]:
        print(f"  {o:<32} {m:.2f}  n={n:>4}  [{p}]")


if __name__ == "__main__":
    main()
