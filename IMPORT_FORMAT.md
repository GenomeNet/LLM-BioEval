# MicrobeLLM Import Format Guide

## Overview
The import functionality at http://127.0.0.1:5051/import allows you to import prediction results from CSV files into the MicrobeLLM database.

## CSV Format Requirements

### Required Columns
These columns must be present in your CSV file:
- `binomial_name` - The species name (e.g., "Escherichia coli")
- `model` - The model used for prediction (e.g., "openai/gpt-4")
- `status` - The status of the prediction (typically "completed")

### Optional Columns

#### Metadata Columns
- `species_file` - The source species file (e.g., "artificial.txt", "wa_with_gcount.txt")
- `system_template` - System template path (e.g., "templates/system/template1_phenotype.txt")
- `user_template` - User template path (e.g., "templates/user/template1_phenotype.txt")
- `knowledge_group` - Knowledge classification (for knowledge templates)

#### Phenotype Prediction Columns
For phenotype predictions, you can include any of these columns:
- `gram_staining` - Values like "gram stain positive", "gram stain negative"
- `motility` - TRUE/FALSE or descriptive values
- `aerophilicity` - List format like "['aerobic']", "['anaerobic']", "['facultatively anaerobic']"
- `extreme_environment_tolerance` - TRUE/FALSE
- `biofilm_formation` - TRUE/FALSE
- `animal_pathogenicity` - TRUE/FALSE
- `biosafety_level` - e.g., "biosafety level 2"
- `health_association` - TRUE/FALSE
- `host_association` - TRUE/FALSE
- `plant_pathogenicity` - TRUE/FALSE
- `spore_formation` - TRUE/FALSE
- `hemolysis` - e.g., "alpha", "beta", "gamma"
- `cell_shape` - e.g., "coccus", "bacillus", "spirillum"

## Example CSV Files

### Example 1: Simple Phenotype Import
```csv
binomial_name,model,status,species_file,gram_staining,motility,aerophilicity,spore_formation
Escherichia coli,openai/gpt-4,completed,artificial.txt,gram stain negative,TRUE,['facultatively anaerobic'],FALSE
Bacillus subtilis,openai/gpt-4,completed,artificial.txt,gram stain positive,TRUE,['aerobic'],TRUE
Streptococcus pneumoniae,openai/gpt-4,completed,artificial.txt,gram stain positive,FALSE,['facultatively anaerobic'],FALSE
```

### Example 2: Knowledge Template Import
```csv
binomial_name,model,status,species_file,knowledge_group
Escherichia coli,openai/gpt-4,completed,artificial.txt,extensive
Bacillus subtilis,openai/gpt-4,completed,artificial.txt,extensive
Unknown bacterium,openai/gpt-4,completed,artificial.txt,limited
```

### Example 3: Full Phenotype Import with All Fields
```csv
binomial_name,model,status,species_file,system_template,user_template,gram_staining,motility,aerophilicity,extreme_environment_tolerance,biofilm_formation,animal_pathogenicity,biosafety_level,health_association,host_association,plant_pathogenicity,spore_formation,hemolysis,cell_shape
Streptococcus pneumoniae,x-ai/grok-3-mini,completed,wa_with_gcount.txt,templates/system/template1_phenotype.txt,templates/user/template1_phenotype.txt,gram stain positive,FALSE,['facultatively anaerobic'],FALSE,TRUE,TRUE,biosafety level 2,TRUE,TRUE,FALSE,FALSE,alpha,coccus
```

## Import Process

1. **Access the Import Page**: Navigate to http://127.0.0.1:5051/import

2. **Select Template Type**: Choose the template that matches your data:
   - Knowledge templates (template1_knowledge.txt, etc.)
   - Phenotype templates (template1_phenotype.txt, etc.)

3. **Review Format Guide**: The interface will show you the expected columns for your selected template

4. **Upload CSV File**: 
   - Click to select or drag-and-drop your CSV file
   - Option to overwrite existing entries if needed

5. **Validate & Import**: The system will:
   - Validate your CSV format
   - Report any validation errors
   - Import valid rows
   - Show statistics (imported, skipped, failed)

## Tips for Successful Import

1. **Match Existing Values**: Check existing data formats using the database viewer at http://127.0.0.1:5051/database

2. **Use Consistent Model Names**: Model names should match existing patterns (e.g., "openai/gpt-4", "anthropic/claude-3")

3. **Boolean Values**: Use TRUE/FALSE (uppercase) for boolean fields

4. **List Values**: For fields like aerophilicity, use Python list format: `['value']` or `['value1', 'value2']`

5. **Test Small First**: Start with a small test file to verify format before importing large datasets

6. **Check for Duplicates**: The system will skip duplicate entries unless you enable "Overwrite existing entries"

## Common Issues and Solutions

### Issue: Import fails with validation errors
**Solution**: Check that required columns (binomial_name, model, status) are present and have values

### Issue: Phenotype values not importing
**Solution**: Ensure values match expected formats (e.g., "gram stain positive" not "Gram positive")

### Issue: Duplicate entries being skipped
**Solution**: Enable "Overwrite existing entries" option if you want to update existing data

### Issue: Model or species_file not recognized
**Solution**: Use values that already exist in the database or they will be auto-created

## API Usage

You can also import via the API:

```bash
curl -X POST http://127.0.0.1:5051/api/import_csv_validated \
  -F "file=@your_data.csv" \
  -F "template=template1_phenotype.txt" \
  -F "template_type=phenotype" \
  -F "overwrite=false"
```

Response will include:
- `total_rows`: Number of rows in CSV
- `imported`: Successfully imported rows
- `skipped`: Duplicate rows skipped
- `updated`: Rows updated (if overwrite=true)
- `errors`: Failed rows count
- `validation_errors`: List of specific validation issues