# MicrobeLLM Database Architecture

## Overview

MicrobeLLM uses a dual-database architecture with SQLite for data persistence:

1. **`microbellm.db`** - Ground truth and reference data (static/research data)
2. **`microbellm_jobs.db`** - Job processing and results (operational data)

## Database Schemas

### microbellm.db (Reference Database)

This database contains the ground truth data and reference information used for evaluating LLM predictions.

#### Tables:

**`ground_truth`**
- Stores verified phenotypic characteristics of bacterial species
- Used as the reference standard for evaluating LLM predictions
- Schema:
  - `id`: Primary key
  - `dataset_name`: Name of the dataset source
  - `template_name`: Associated template identifier
  - `binomial_name`: Scientific name of the species
  - Phenotype columns: `gram_staining`, `motility`, `aerophilicity`, `extreme_environment_tolerance`, `biofilm_formation`, `animal_pathogenicity`, `biosafety_level`, `health_association`, `host_association`, `plant_pathogenicity`, `spore_formation`, `hemolysis`, `cell_shape`
  - `import_date`: Timestamp of data import
  - Unique constraint on `(dataset_name, binomial_name)`

**`ground_truth_datasets`**
- Metadata about imported ground truth datasets
- Schema:
  - `id`: Primary key
  - `dataset_name`: Unique dataset identifier
  - `description`: Dataset description
  - `source`: Data source information
  - `template_name`: Associated template
  - `species_count`: Number of species in dataset
  - `import_date`: Import timestamp
  - `validation_summary`: Validation results JSON

**`combinations`** (legacy)
- Original job configuration tracking
- Being phased out in favor of unified `processing_results` table

**`species_results`** (legacy)
- Original individual species results
- Being phased out in favor of unified `processing_results` table

### microbellm_jobs.db (Operational Database)

This is the primary operational database containing job processing data, results, and system configuration.

#### Core Tables:

**`processing_results`** (Primary unified table)
- Central table combining job configuration and species results
- Eliminates need for joins between combinations and species_results
- Schema includes:
  - Job identification: `job_id` (UUID), `job_status`, timestamps
  - Configuration: `species_file`, `model`, `system_template`, `user_template`
  - Species results: `binomial_name`, `status`, `result` (JSON), `error`
  - Parsed predictions: All phenotype columns for quick querying
- Indexes on `job_id`, job configuration, status fields, and `binomial_name`

**`job_summary`** (View)
- Aggregated view of job statistics
- Groups processing_results by job_id
- Provides counts: total, successful, failed, timeout, submitted species

**`managed_models`**
- Registry of LLM models available for testing
- Simple key-value structure with model identifier as primary key

**`managed_species_files`**
- Registry of species input files
- Tracks available CSV files for batch processing

**`template_metadata`**
- Template configuration and metadata
- Stores display names, descriptions, and template types
- Primary key on `(system_template, user_template)` pair

#### Legacy Tables (Still Present):

**`combinations`**
- Original job tracking table
- Contains job-level statistics and configuration
- Being maintained for backward compatibility

**`species_results`**
- Original per-species results linked to combinations
- Foreign key relationship to combinations table

**`results`**
- Flattened results table with parsed phenotypes
- Redundant with `processing_results` but kept for compatibility

**`ground_truth`** & **`ground_truth_datasets`**
- Duplicated from microbellm.db
- Allows admin interface to access ground truth without cross-database queries

## Data Flow

1. **Job Creation**: New job creates entry in `processing_results` with unique `job_id`
2. **Processing**: Each species gets a row with same `job_id`, individual status tracking
3. **Results Storage**: LLM responses stored as JSON in `result` column
4. **Parsing**: Phenotype predictions extracted and stored in dedicated columns
5. **Aggregation**: `job_summary` view provides real-time job statistics

## Migration Status

The system is transitioning from a normalized (combinations + species_results) to a denormalized (processing_results) structure for:
- Better query performance (no joins needed)
- Simpler data model
- Easier job management
- More efficient status tracking

Legacy tables are maintained for backward compatibility but new features use the unified `processing_results` table.

## Database Locations

Both databases reside in the project root:
- `microbellm.db` - Reference data (11 MB)
- `microbellm_jobs.db` - Operational data (131 MB, grows with usage)

## Technical Notes

- All databases use SQLite3 with WAL mode for concurrent access
- Timestamps stored in ISO format
- JSON data stored as TEXT in result columns
- Foreign key constraints enforced where applicable
- Indexes optimized for common query patterns (job lookups, status filtering)