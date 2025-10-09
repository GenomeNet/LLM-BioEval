# LLM-BioEval

LLM-BioEval is an open toolkit for benchmarking large language models on structured microbial knowledge tasks. The codebase underpins the analyses in our manuscript, pairing curated BacDive/bugphyzz ground truth with automated inference pipelines, validation dashboards, and public reporting components.

## Project Highlights

- **End-to-end evaluation stack** – Deterministic template-based prompting, real-time admin dashboard for job orchestration, and a public web portal that serves the latest benchmark summaries from a shared SQLite database.
- **Curated microbial datasets** – 19k harmonized species records with 13 phenotype traits, plus synthetic taxa for hallucination stress tests and low-annotation cohorts for out-of-distribution evaluation.
- **Reproducible manuscript workflows** – Versioned scripts generate the knowledge calibration, hallucination, and phenotype accuracy figures reported in the paper, with outputs cached under the `microbellm/templates/research` directory.
- **Model-agnostic design** – Works with any model accessible through OpenRouter (300+ providers and releases), making it easy to compare frontier APIs with open checkpoints.

## Repository Structure

```
├─ microbellm/                 # Core package (Flask apps, job orchestration, utilities)
├─ microbellm/templates/       # Research dashboards, manuscript figures & scripts
├─ data/                       # Harmonized ground-truth exports and species cohorts
├─ scripts/                    # CLI utilities for summary tables and stats dumps
└─ tests/                      # Pytest suite covering API, database, and utils
```

## Quick Start

```bash
# clone and install (editable)
git clone https://github.com/GenomeNet/LLM-BioEval.git
cd LLM-BioEval
conda env create -f environment.yml
conda activate microbellm
pip install -e .
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

### Launch the dashboards

```bash
# Administrative inference dashboard (local-only)
microbellm-admin --debug --port 5051
# Public research portal
microbellm-web --debug --port 5050
```

Both services use the same SQLite database. The admin UI lets you queue model/species/template combinations, monitor jobs in real time, and review raw LLM outputs. The web portal renders all figures used in the manuscript and refreshes automatically when new inferences are written to the database.

### Run the automated tests

```bash
pytest
```

## Reproducing Manuscript Analyses

1. Generate or refresh the harmonized phenotype table (`data/original_data_generation/bugphyzz_corrected.R`).
2. Queue the desired model runs via the admin dashboard (or CLI wrappers in `microbellm/templates/research/phenotype_analysis/sections/07c_model_accuracy`).
3. Execute the analysis scripts for hallucination benchmarks, knowledge calibration, and phenotype accuracy (each section contains `generate_*.py` helpers).
4. Export CSV/JSON summaries through the dashboard or `scripts/export_model_accuracy_table.py`, and update the manuscript text via `generate_manuscript_stats.py` if needed.

Intermediate CSV/PDF outputs are intentionally Git-ignored; rerun the scripts to regenerate figures for publication or supplementary data.

## Data Notes

- **Ground truth**: Harmonized BacDive/bugphyzz export (`merged_data.rds`) filtered to single-valued phenotypes for the main benchmark.
- **Synthetic taxa**: 200 artificial binomial names grouped into four realism tiers for hallucination detection.
- **Species cohorts**: `wa_with_gcount.txt` (well-annotated) and `la.txt` (low-annotation) drive the phenotype benchmarks.

Please ensure you have permission to access model APIs and comply with provider usage policies when running large-scale evaluations.

## Citation

If you use LLM-BioEval or our benchmark results, please cite the accompanying manuscript and the Zenodo archive:

```
@software{llm_bioeval_zenodo,
  title        = {LLM-BioEval: Benchmarking Large Language Models on Microbial Knowledge},
  author       = {GenomeNet team},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.13839818}
}
```

## License

LLM-BioEval is released under the MIT License. Refer to `LICENSE` for details.
