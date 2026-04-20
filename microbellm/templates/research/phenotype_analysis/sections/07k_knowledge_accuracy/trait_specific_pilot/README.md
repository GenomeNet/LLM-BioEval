# Trait-specific vs species-level self-assessment (pilot)

This folder holds the artefacts for the rebuttal experiment that
addresses Reviewer Q3: "does species-level self-rated knowledge capture
trait-specific confidence?"

## Design

- **Model:** `x-ai/grok-3-mini` (single model, keeps inference cost bounded).
- **Species set:** `wa_with_gcount.txt` (same WA benchmark subset used
  elsewhere in Fig. 4 / knowledge-group analysis).
- **Phenotype:** motility (first pilot; other phenotypes can be cloned
  by duplicating the template files with a different phenotype name).
- **Two prediction runs** compared per species:
  1. *Species-level* knowledge rating — `template3_knowlege.txt`
     (already computed in the existing benchmark).
  2. *Trait-specific* knowledge rating — `template1_knowledge_motility.txt`
     (new, see `templates/user/` and `templates/system/`).
- Both runs emit `{"knowledge_group": "<limited|moderate|extensive|NA>"}`
  so all downstream normalisation and DB storage is reused unchanged.

## How to run

1. Start the admin dashboard:
   ```
   microbellm-admin --port 5051 --debug
   ```
2. Open http://localhost:5051/ → **Create Job** and submit:
   - species file: `wa_with_gcount.txt`
   - model: `x-ai/grok-3-mini`
   - system template: `template1_knowledge_motility.txt`
   - user template: `template1_knowledge_motility.txt`
3. Wait until job status is `completed` (progress shown on dashboard).

The species-level counterpart
(`x-ai/grok-3-mini` + `template3_knowlege.txt` + `wa_with_gcount.txt`)
should already exist from prior runs; verify on the dashboard.

## Analysis (to be written once data lands)

Planned script: `analyze_trait_vs_species_agreement.py`

1. Pull both prediction sets from `microbellm_jobs.db` keyed on species.
2. Join on `binomial_name`; drop species missing either rating.
3. Report:
   - Confusion matrix of trait-level × species-level `knowledge_group`.
   - Overall agreement rate and Cohen's κ.
   - Cramér's V for ordinal association.
   - Balanced accuracy on the motility phenotype, stratified by each
     rating axis, side-by-side (reproduces the figure sketched in the
     rebuttal).
4. If agreement is high → species-level rating is a usable proxy and the
   rebuttal statement stands. If low → motivates a broader trait-specific
   campaign (deferred for budget reasons).

## Adding more phenotypes later

Clone the three template files and replace every `motility` mention:

```
templates/system/template1_knowledge_<phenotype>.txt
templates/user/template1_knowledge_<phenotype>.txt
templates/validation/template1_knowledge_<phenotype>.json
```

No code changes required; the admin UI picks up new templates on restart.
