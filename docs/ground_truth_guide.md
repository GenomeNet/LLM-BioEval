# Ground Truth Data Management Guide

## Overview

The Ground Truth Data Management system in MicrobeBench allows you to:

1. **Import validated ground truth data** for bacterial phenotypes
2. **Browse and search** through imported datasets
3. **Calculate model accuracy** by comparing predictions against ground truth
4. **Track validation errors** during import

## Ground Truth Data Format

### CSV Structure

Ground truth data must be provided in CSV format with the following columns:

```csv
binomial_name,gram_staining,motility,aerophilicity,extreme_environment_tolerance,biofilm_formation,animal_pathogenicity,biosafety_level,health_association,host_association,plant_pathogenicity,spore_formation,hemolysis,cell_shape
```

### Column Specifications

| Column | Type | Description | Allowed Values |
|--------|------|-------------|----------------|
| binomial_name | string | Scientific name of the species | Any valid species name |
| gram_staining | string | Gram stain result | gram stain positive, gram stain negative, gram stain variable |
| motility | string | Motility capability | TRUE, FALSE |
| aerophilicity | string | Oxygen requirements | aerobic, anaerobic, facultatively anaerobic, aerotolerant |
| extreme_environment_tolerance | string | Tolerance to extreme conditions | TRUE, FALSE |
| biofilm_formation | string | Ability to form biofilms | TRUE, FALSE |
| animal_pathogenicity | string | Pathogenic to animals | TRUE, FALSE |
| biosafety_level | string | Biosafety classification | biosafety level 1, biosafety level 2, biosafety level 3 |
| health_association | string | Association with human health | TRUE, FALSE |
| host_association | string | Lives in/on a host | TRUE, FALSE |
| plant_pathogenicity | string | Pathogenic to plants | TRUE, FALSE |
| spore_formation | string | Forms spores | TRUE, FALSE |
| hemolysis | string | Hemolytic activity | alpha, beta, gamma, non-hemolytic |
| cell_shape | string | Cell morphology | bacillus, coccus, spirillum, tail |

### Value Normalization

The system automatically normalizes values during import:

- **Case insensitive**: "TRUE", "true", "True" → "TRUE"
- **Whitespace trimmed**: " TRUE " → "TRUE"
- **Synonyms mapped**: 
  - "positive", "yes" → "TRUE"
  - "negative", "no" → "FALSE"
  - "gram+", "gram positive" → "gram stain positive"
  - "gram-", "gram negative" → "gram stain negative"

## Using the Ground Truth System

### 1. Importing Ground Truth Data

1. Navigate to **Ground Truth** in the main menu
2. In the "Import Ground Truth Data" section:
   - Select your CSV file
   - Provide a unique dataset name
   - Select the template that matches your data format
   - Add optional description and source information
3. Click "Import Data"

The system will:
- Validate all values against the template specifications
- Normalize values automatically
- Report any validation errors
- Store successfully imported records

### 2. Browsing Ground Truth Data

The Ground Truth viewer provides:
- **Dataset list**: Shows all imported datasets with metadata
- **Search functionality**: Filter by species name
- **Pagination**: Browse large datasets efficiently
- **Detail view**: Click any species to see all phenotype values

### 3. Model Accuracy Analysis

1. Navigate to **Model Accuracy** in the main menu
2. Select:
   - Ground truth dataset to use as reference
   - Model predictions file to evaluate
3. Click "Analyze Accuracy"

The analysis provides:
- **Overall accuracy metrics**: Total accuracy percentage across all phenotypes
- **Per-phenotype accuracy**: Breakdown by each phenotype characteristic
- **Confusion matrices**: Detailed prediction patterns for each phenotype
- **Missing prediction rates**: How often models fail to provide predictions

## Data Validation

### Validation Rules

Each phenotype field is validated according to its template specification:

1. **Allowed values check**: Values must match the allowed set
2. **Type validation**: Binary fields accept TRUE/FALSE, categorical fields accept specific values
3. **Format normalization**: Values are normalized to canonical forms

### Handling Validation Errors

During import, validation errors are:
- Logged with specific error messages
- Associated with the problematic species name
- Reported in the import summary
- Records with errors are still imported but marked

## Best Practices

### Data Preparation

1. **Verify column names** match exactly (case-sensitive)
2. **Use canonical values** when possible (e.g., "TRUE" not "yes")
3. **Leave unknown values empty** rather than using "unknown" or "?"
4. **Validate data** in a spreadsheet before import

### Dataset Management

1. **Use descriptive dataset names**: Include version, date, or source
2. **Document data sources**: Use the description field
3. **Keep original files**: The system stores normalized data
4. **Version your datasets**: Create new imports rather than updating

### Accuracy Analysis

1. **Match species names exactly**: Ground truth and predictions must use identical names
2. **Use consistent templates**: Ensure predictions use the same template as ground truth
3. **Review confusion matrices**: Identify systematic prediction errors
4. **Track improvements**: Compare accuracy across model versions

## Troubleshooting

### Common Issues

1. **Import fails completely**
   - Check CSV encoding (use UTF-8)
   - Verify column headers match exactly
   - Ensure no duplicate column names

2. **Many validation errors**
   - Review template specifications
   - Check for typos in values
   - Verify value normalization rules

3. **Low accuracy scores**
   - Confirm species name matching
   - Check template compatibility
   - Review missing prediction rates

### Getting Help

- Check validation error messages for specific issues
- Review template JSON files for exact specifications
- Export sample data to see expected formats 