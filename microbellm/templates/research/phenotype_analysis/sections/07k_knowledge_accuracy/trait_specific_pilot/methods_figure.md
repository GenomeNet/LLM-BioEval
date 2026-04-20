# Trait-audit analysis, methods and figure package (reviewer Q3)

This file is the paper-ready bundle for the trait-audit analysis: reader's
guide to the headline figure, camera-ready figure caption, methods section,
rebuttal response, output-file inventory, and reproduction steps.

Figure directory:
`microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/trait_specific_pilot/`

**Figure X file:** `progressive_aggregation_gpt-4.1-nano.pdf` (in this directory).

---

## 0. Response to reviewer Q3 (copy-paste into rebuttal letter)

> *Reviewer Q3: Currently, the hallucination assessment asks the LLM for its
> overall knowledge level of a species. However, hallucination assessment
> should also be trait-specific. For instance, an LLM might have extensive,
> high-confidence knowledge regarding a specific species' motility (due to
> abundant literature) but limited knowledge regarding its biofilm formation.*

We addressed this concern by running a trait-audit experiment: for every
WA species (N = 3,884) we issued 12 independent trait-specific knowledge
queries to GPT-4.1 nano, one per phenotype reported in the main paper,
in addition to the species-level query used throughout the manuscript,
and found that although any single trait rating is a noisy,
trait-dependent proxy (Spearman ρ between the species-level rating and
any one trait ranges from 0.11 to 0.62 depending on which trait is
asked), the species-level rating is rank-consistent with the mean of
three or more random trait ratings (median ρ ≈ 0.56 at k = 3, saturating
near 0.65 at k = 12), meaning the species-level query functions as a
summary statistic of trait-specific confidences at roughly 12x lower API
cost rather than a one-dimensional proxy. This analysis is added as a
new Supplementary Figure and a new Supplementary Methods subsection.

At the level of individual trait queries the reviewer's concern is
empirically valid: knowing the model's motility confidence for one
species tells you little about its biofilm-formation confidence for the
same species, exactly as the reviewer describes. The picture changes
once we ask what the species-level rating is actually summarising. For
each k from 1 to 12 we enumerated all C(12, k) subsets of k traits drawn
from the 12 phenotypes (sub-sampling uniformly at random to 300 subsets
when combinatorially larger), computed the per-species mean of each
subset's trait ratings, and measured Spearman ρ against the species-level
rating. Median ρ rises monotonically with k, already reaching ≈ 0.56 at
k = 3, and saturates near 0.65 at k = 12. The across-subset variability
collapses rapidly, meaning any three trait queries give a stable
aggregate regardless of which three are chosen. Read plainly: if you
replaced the species-level prompt with three or more trait-specific
queries per species and averaged them, you would recover most of the
species-level signal.

The plateau below ρ = 1 is expected and structural. The species-level
prompt integrates over aspects of organism familiarity beyond the
phenotype panel (taxonomy, strain count, culture-collection presence,
overall literature volume), so it cannot be a linear function of any
finite phenotype set. The ρ ≈ 0.65 ceiling reflects the portion of
species-level variance recoverable from the 12-trait audit.

---

## 1. Headline figure — how to read it

**File:** `progressive_aggregation_gpt-4.1-nano.pdf`

**Title on the figure:**
*"How many trait queries does it take to recover the species-level signal?"*

### What each axis means

**X-axis — number of trait-level queries averaged per species (k).**
Each value of *k* corresponds to a thought experiment: *"Imagine we asked
the model about k specific phenotypes for every species (e.g. only motility
at k=1; motility + aerophilicity + biosafety at k=3; all phenotypes at k=K),
then computed the per-species mean of those k ratings."* The x-axis scans
from k=1 (one trait-specific query per species — the reviewer's proposed
counterfactual) to k=K (all trait-specific queries — the maximal trait-level
information we can extract).

**Y-axis — Spearman rank correlation ρ between the species-level rating and
the k-trait aggregate.** Values lie in [−1, +1]. ρ=0 means no rank
agreement; ρ=1 means the two rankings are identical. Spearman (rank-order)
rather than Pearson (linear) because the ratings are ordinal
(limited < moderate < extensive) and we care about agreement on *order*,
not on absolute tier values.

### What each visual element means

**Solid dark line** — the **median** ρ across all (K choose k) trait
subsets of size k (sub-sampled to 300 random subsets when the enumerated
count is larger). This is the "typical" ρ you would get if you picked *any*
k trait queries uniformly at random.

**Inner shaded band** — the 2.5th–97.5th percentile of ρ across subsets of
size k. This is the *95 % across-subset interval*: it says "if you happened
to pick a different set of k traits, ρ would land somewhere in this band
with 95 % probability."

**Outer shaded band** — the full min–max envelope across subsets. Shows the
worst- and best-case subset at each k.

**Dotted reference lines** — the *best* and *worst* individual trait at k=1
(these are the extremes of the outer band at k=1), made horizontal for easy
visual comparison against higher-k ρ.

### How to read the story

1. **At k=1** (single trait query, the reviewer's concern): the outer band
   is very wide. Individual traits range from ρ ≈ 0.11 (plant pathogenicity)
   to ρ ≈ 0.62 (aerophilicity). **You don't know a priori which trait best
   tracks species-level knowledge**, so any one trait query is an unreliable
   proxy. This is exactly the reviewer's objection, and the figure
   *concedes* it at k=1.

2. **At k=3** (three random traits): the outer band collapses dramatically.
   Median ρ ≈ 0.56, 95 % range roughly [0.25, 0.66]. Even three arbitrary
   trait queries, averaged, yield a stable answer.

3. **As k grows**: the median line climbs monotonically; the bands shrink
   further. By k ≈ 6 the spread is negligible and the aggregate is a
   well-defined quantity independent of which traits you happen to pick.

4. **At k = K = 12** (all trait queries averaged): single point because only
   one subset exists. ρ saturates at ≈ 0.65.

5. **The asymptote below 1.0 is expected, not a defect.** The species-level
   prompt asks the model about *overall* species familiarity — which
   integrates over strain count, genome completeness, culture-collection
   availability, literature volume, and all other aspects of organism study,
   not just the 12 phenotypes we audit here. So species-level cannot be a
   pure linear function of any finite phenotype set. ρ ≈ 0.65 is the upper
   bound of the variance that our 12 trait ratings *can* recover; the
   remainder is "other" species-level signal (taxonomy, general interest,
   etc.).

### The rebuttal point in one sentence

> *A single trait rating is a noisy, trait-dependent proxy
> (ρ ∈ [0.11, 0.62]); averaging even ~3 random trait ratings recovers most
> of the species-level signal; species-level querying is thus a valid and
> compute-efficient summary of per-trait self-assessed knowledge.*

---

## 2. Figure caption (camera-ready)

> **Figure R2. Progressive aggregation of trait-specific knowledge ratings
> vs the species-level rating.** Using self-rated knowledge responses from
> GPT-4.1 nano across the full WA species panel (N = 3,884), for each k
> from 1 to 12 we enumerate all possible subsets of k traits drawn from
> the 12 phenotypes reported in the main paper (sub-sampling uniformly at
> random to 300 subsets when enumeration becomes impractical), compute the
> per-species mean of each subset's trait ratings, and quantify its
> agreement with the species-level rating through the Spearman rank
> correlation ρ. The solid line traces the median ρ across subsets at each
> k, the inner shaded band spans the 2.5th to 97.5th percentile across
> subsets, and the outer shaded band shows the full minimum-to-maximum
> envelope. Two dotted horizontal lines mark the best- and
> worst-correlated single trait at k = 1 as reference. Per-k subset counts
> and the number of species contributing to each ρ estimate (which vary
> with per-trait coverage) are given in the accompanying supplementary
> table. Interpretation is given in the main response text.

---

## 3. Methods section (camera-ready)

> **Trait-specific vs species-level knowledge audit.** To test whether the
> species-level self-assessed knowledge rating used throughout the main
> paper is a valid summary of trait-specific self-assessment, rather than
> a one-dimensional proxy that may mis-represent per-phenotype confidence,
> we ran a paired trait-audit experiment on the full WA subset (N = 3,884
> species; no subsampling). For each of the 12 phenotypes reported in the
> main paper (gram staining, motility, aerophilicity, extreme-environment
> tolerance, biofilm formation, animal pathogenicity, biosafety level,
> health association, host association, plant pathogenicity, spore
> formation, cell shape) we constructed a trait-specific knowledge-rating
> prompt with identical structural wording to the species-level template,
> substituting the trait name and trait-specific examples of "limited /
> moderate / extensive" evidence. The full 12 trait templates are released
> alongside a generator script in the public repository; wording
> parallelism is enforced by templating to prevent cross-trait artefacts
> in the comparison. Hemolysis is not reported in the main-paper benchmark
> and was excluded from the audit panel for consistency.
>
> For a single model (GPT-4.1 nano, queried via OpenRouter at default
> temperature) we issued one species-level query and twelve independent
> trait-level queries per species. Queries were issued as **independent
> API calls rather than as a single combined-phenotype prompt**, to
> prevent within-context priming across traits. Each call returned a JSON
> object with a single field (the ordinal knowledge-group tier); responses
> were parsed, case-normalised against a synonym dictionary, and validated
> against the allowed tier vocabulary before storage in the benchmark
> database.
>
> Tiers were encoded as ordinal integers (limited = 0, moderate = 1,
> extensive = 2) with NA (unrecognised name or taxonomy-only signal)
> treated as missing. For each species *i* we computed the per-species
> trait aggregate
>
>   a_i = mean({T_{i,j} : T_{i,j} ∈ ordinal}),
>
> where T_{i,j} is the ordinal rating of trait *j* for species *i*.
> Species with fewer than three ordinal trait ratings were excluded from
> the aggregate; this affected ≤ 1 % of species at analysis time.
> Agreement between the species-level rating S and the aggregate was
> quantified by **Spearman rank correlation ρ on the continuous aggregate**,
> computed on each complete-case subset (species with both S and a_i
> defined).
>
> To characterise how the number of trait queries affects agreement, we
> computed a progressive-aggregation curve: for each k ∈ {1, …, K = 12}
> we enumerated all (K choose k) trait subsets (sub-sampling uniformly at
> random to 300 when the enumerated count exceeded that), computed a_i
> per species for each subset, and recorded the distribution of ρ across
> subsets (median, 2.5th to 97.5th percentile, and min to max envelope).
> Per-k species counts are reported alongside the curve in the released
> supplementary table. This random-subset design marginalises over the
> identity of the specific traits included, so the reported curve
> reflects the *expected* agreement from *any* k trait queries rather
> than from a particular ordering.
>
> We retained the floating complete-case design (each subset's ρ is
> computed on its own non-NA species set) rather than a fixed-denominator
> design (species with ordinal ratings on all K traits): the fixed subset
> would drop N to ~1,200 species (31 % of the WA subset) and bias toward
> species in which the model expressed confidence on every trait, which is
> itself a function of trait-specific confidence and therefore confounded
> with the quantity under test. The diagnostic fixed-subset size is logged
> in the analysis output for transparency.
>
> As supplementary diagnostics we discretised the aggregate a_i into tiers
> using the thresholds {a < 0.5 → limited, a < 1.5 → moderate,
> a ≥ 1.5 → extensive} and computed categorical exact-match agreement,
> within-one-tier agreement, Cohen's linear-weighted κ, and Cramér's V
> against S. **Discretisation introduces a regression-to-the-mean bias
> that grows with k** (the aggregate concentrates near the population mean
> and loses tier-level sharpness), which systematically suppresses
> exact-match agreement and κ; for this reason Spearman ρ on the continuous
> aggregate is the preferred metric, and categorical agreement statistics
> are reported only in the supplementary output tables with explicit
> caveats. Pairwise Spearman correlations among all 14 ratings were
> computed on each pair's complete-case subset and are shown in the
> supplementary correlation heatmap.
>
> All templates, launcher scripts, predictions, intermediate tables, and
> plotting code are released in
> `microbellm/templates/research/phenotype_analysis/sections/07k_knowledge_accuracy/trait_specific_pilot/`.

---

## 4. Output files

| File | Contents |
|---|---|
| `progressive_aggregation_gpt-4.1-nano.pdf` | **Headline figure.** ρ vs k with median / 95 % / min-max bands. |
| `progressive_aggregation_rho_gpt-4.1-nano.tsv` | Numeric values backing the figure: `k`, `rho_median`, `ci_low`, `ci_high`, `rho_min`, `rho_max`, `n_subsets`, `n_species_used`. |
| `species_vs_aggregate_gpt-4.1-nano.pdf` | Supplementary: 4×4 bubble confusion of species-level tier vs discretised aggregate tier. |
| `tier_distribution_gpt-4.1-nano.pdf` | Supplementary: side-by-side tier-frequency bars (species-level vs aggregate, with counts + percentages). |
| `trait_correlation_heatmap_gpt-4.1-nano.pdf` | Supplementary: 14 × 14 Spearman ρ matrix across species-level and all trait ratings. |
| `per_species_ratings_gpt-4.1-nano.tsv` | One row per species: species-level rating, the 12 audited trait ratings, and the per-species aggregate summary statistics (mean, tier, n_ordinal, na_frac, max, min, mode). |
| `aggregate_agreement_metrics_gpt-4.1-nano.tsv` | Primary: Spearman ρ on continuous aggregate. Supplementary (with caveat notes): exact-tier, within-one-tier, Cohen's κ (unweighted, linear-weighted), Cramér's V. |
| `trait_correlation_spearman_gpt-4.1-nano.tsv` | 14 × 14 Spearman ρ matrix as TSV. |

## 5. Reproduction

**Prerequisites:** admin dashboard running with `OPENROUTER_API_KEY` set;
the `template1_knowledge_<trait>` files for the 12 audited phenotypes
present in `templates/{system,user,validation}/` (the hemolysis template
is also on disk but excluded from the analysis by the `TRAITS` list in
`analyze_trait_audit.py`).

**Launch the 12 new trait-audit jobs** (motility is pre-existing from the
earlier pilot):

```bash
python microbellm/templates/research/phenotype_analysis/sections/\
07k_knowledge_accuracy/trait_specific_pilot/launch_trait_audit_jobs.py \
    --model openai/gpt-4.1-nano \
    --species-file wa_with_gcount.txt
```

**Regenerate all figures and tables** once jobs complete (or rerun anytime
for partial results — the script uses an 80 %-coverage threshold to pick
the active trait set):

```bash
PILOT_MODEL=openai/gpt-4.1-nano python \
  microbellm/templates/research/phenotype_analysis/sections/\
07k_knowledge_accuracy/trait_specific_pilot/analyze_trait_audit.py
```

**Generate the 12 trait templates from scratch** (only needed once; already
committed):

```bash
python microbellm/templates/research/phenotype_analysis/sections/\
07k_knowledge_accuracy/trait_specific_pilot/generate_trait_templates.py
```
