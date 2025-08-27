# MicrobeLLM Database Architecture

## Overview

MicrobeLLM uses a single unified SQLite database for all data persistence:

**`microbellm.db`** - Contains all operational data, ground truth, and configuration

## Database Schema

### Core Tables

**`processing_results`** (Primary unified table)
- Central table for all job processing and results
- Combines job configuration with species-level results
- Schema includes:
  - Job identification: `job_id` (UUID), `job_status`, timestamps
  - Configuration: `species_file`, `model`, `system_template`, `user_template`
  - Species results: `binomial_name`, `status`, `result` (JSON), `error`
  - Parsed predictions: All phenotype columns for quick querying
    - `knowledge_group`, `gram_staining`, `motility`, `aerophilicity`
    - `extreme_environment_tolerance`, `biofilm_formation`, `animal_pathogenicity`
    - `biosafety_level`, `health_association`, `host_association`
    - `plant_pathogenicity`, `spore_formation`, `hemolysis`, `cell_shape`
- Indexes on `job_id`, job configuration, status fields, and `binomial_name`
- This table eliminates the need for joins between separate job and result tables

**`job_summary`** (View)
- Aggregated view of job statistics from `processing_results`
- Groups results by `job_id`
- Provides counts: total, successful, failed, timeout, submitted species
- Real-time statistics without additional computation

### Configuration Tables

**`managed_models`**
- Registry of available LLM models
- Simple structure with model identifier as primary key
- Used for model selection in UI

**`managed_species_files`**
- Registry of available species input files
- Tracks CSV files for batch processing
- Referenced in job creation

**`template_metadata`**
- Template configuration and metadata
- Stores display names, descriptions, and template types
- Primary key on `(system_template, user_template)` pair
- Used for template selection and validation

### Ground Truth Tables

**`ground_truth`**
- Verified phenotypic characteristics of bacterial species
- Reference standard for evaluating LLM predictions
- Schema:
  - `id`: Primary key
  - `dataset_name`: Dataset source identifier
  - `template_name`: Associated template
  - `binomial_name`: Scientific species name
  - Phenotype columns matching those in `processing_results`
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

### Legacy Tables (Maintained for Compatibility)

**`combinations`**
- Original job tracking table
- Contains job-level statistics and configuration
- Being phased out in favor of `processing_results`

**`species_results`**
- Original per-species results linked to combinations
- Foreign key relationship to combinations table
- Being phased out in favor of unified structure

**`results`**
- Flattened results table with parsed phenotypes
- Redundant with `processing_results` but kept for backward compatibility

## Data Flow

1. **Job Creation**: New job creates entries in `processing_results` with unique `job_id`
2. **Processing**: Each species gets a row with same `job_id`, individual status tracking
3. **Results Storage**: LLM responses stored as JSON in `result` column
4. **Parsing**: Phenotype predictions extracted and stored in dedicated columns
5. **Aggregation**: `job_summary` view provides real-time job statistics

## Benefits of Unified Architecture

- **Simplicity**: Single database file to manage
- **Performance**: No cross-database joins needed
- **Consistency**: Single source of truth for all data
- **Maintenance**: Easier backup, migration, and management
- **Efficiency**: Reduced query complexity

## Database Management

### Location
- Database file: `microbellm.db` in project root
- Size: ~126 MB (grows with usage)
- Format: SQLite3 with WAL mode for concurrent access

### Backup Strategy
- Regular backups recommended before major operations
- Simple file copy for backup: `cp microbellm.db microbellm.db.backup`

### Migration from Dual-Database System
The system previously used two databases:
- `microbellm.db` - Ground truth only (11 MB)
- `microbellm_jobs.db` - All operational data (126 MB)

These have been unified into a single `microbellm.db` containing all data.

## Technical Notes

- All timestamps stored in ISO format
- JSON data stored as TEXT in result columns
- Foreign key constraints enforced where applicable
- Indexes optimized for common query patterns:
  - Job lookups by ID
  - Status filtering
  - Species name searches
  - Configuration-based queries
- WAL mode enabled for better concurrent access
- VACUUM recommended periodically to optimize file size