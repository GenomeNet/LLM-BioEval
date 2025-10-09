#!/usr/bin/env python3
"""
Quick analysis of model performance for each phenotype
"""

import sqlite3
import pandas as pd
import numpy as np

# Connect to database
conn = sqlite3.connect('microbellm.db')

# Query to get model accuracies for each phenotype
query = """
WITH model_accuracies AS (
    SELECT
        pr.model,
        'spore_formation' as phenotype,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN LOWER(pr.spore_formation) = LOWER(gt.spore_formation) THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(100.0 * SUM(CASE WHEN LOWER(pr.spore_formation) = LOWER(gt.spore_formation) THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.status = 'completed'
        AND gt.spore_formation IS NOT NULL AND gt.spore_formation != ''
        AND pr.spore_formation IS NOT NULL AND pr.spore_formation != ''
    GROUP BY pr.model

    UNION ALL

    SELECT
        pr.model,
        'motility' as phenotype,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN LOWER(pr.motility) = LOWER(gt.motility) THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(100.0 * SUM(CASE WHEN LOWER(pr.motility) = LOWER(gt.motility) THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.status = 'completed'
        AND gt.motility IS NOT NULL AND gt.motility != ''
        AND pr.motility IS NOT NULL AND pr.motility != ''
    GROUP BY pr.model

    UNION ALL

    SELECT
        pr.model,
        'gram_staining' as phenotype,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN LOWER(pr.gram_staining) = LOWER(gt.gram_staining) THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(100.0 * SUM(CASE WHEN LOWER(pr.gram_staining) = LOWER(gt.gram_staining) THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.status = 'completed'
        AND gt.gram_staining IS NOT NULL AND gt.gram_staining != ''
        AND pr.gram_staining IS NOT NULL AND pr.gram_staining != ''
    GROUP BY pr.model

    UNION ALL

    SELECT
        pr.model,
        'aerophilicity' as phenotype,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN LOWER(pr.aerophilicity) = LOWER(gt.aerophilicity) THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(100.0 * SUM(CASE WHEN LOWER(pr.aerophilicity) = LOWER(gt.aerophilicity) THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.status = 'completed'
        AND gt.aerophilicity IS NOT NULL AND gt.aerophilicity != ''
        AND pr.aerophilicity IS NOT NULL AND pr.aerophilicity != ''
    GROUP BY pr.model

    UNION ALL

    SELECT
        pr.model,
        'cell_shape' as phenotype,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN LOWER(pr.cell_shape) = LOWER(gt.cell_shape) THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(100.0 * SUM(CASE WHEN LOWER(pr.cell_shape) = LOWER(gt.cell_shape) THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.status = 'completed'
        AND gt.cell_shape IS NOT NULL AND gt.cell_shape != ''
        AND pr.cell_shape IS NOT NULL AND pr.cell_shape != ''
    GROUP BY pr.model

    UNION ALL

    SELECT
        pr.model,
        'biofilm_formation' as phenotype,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN LOWER(pr.biofilm_formation) = LOWER(gt.biofilm_formation) THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(100.0 * SUM(CASE WHEN LOWER(pr.biofilm_formation) = LOWER(gt.biofilm_formation) THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.status = 'completed'
        AND gt.biofilm_formation IS NOT NULL AND gt.biofilm_formation != ''
        AND pr.biofilm_formation IS NOT NULL AND pr.biofilm_formation != ''
    GROUP BY pr.model

    UNION ALL

    SELECT
        pr.model,
        'hemolysis' as phenotype,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN LOWER(pr.hemolysis) = LOWER(gt.hemolysis) THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(100.0 * SUM(CASE WHEN LOWER(pr.hemolysis) = LOWER(gt.hemolysis) THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.status = 'completed'
        AND gt.hemolysis IS NOT NULL AND gt.hemolysis != ''
        AND pr.hemolysis IS NOT NULL AND pr.hemolysis != ''
    GROUP BY pr.model
)
SELECT
    phenotype,
    model,
    accuracy,
    total_predictions
FROM (
    SELECT
        phenotype,
        model,
        accuracy,
        total_predictions,
        ROW_NUMBER() OVER (PARTITION BY phenotype ORDER BY accuracy DESC, total_predictions DESC) as rn
    FROM model_accuracies
    WHERE total_predictions >= 20  -- Minimum sample size
)
WHERE rn = 1
ORDER BY accuracy DESC;
"""

print("Analyzing best model-phenotype combinations...")
print("="*70)

# Execute query and get results
df = pd.read_sql_query(query, conn)

# Create markdown table
md_table = "# Best Model-Phenotype Combinations\n\n"
md_table += "| Phenotype | Best Model | Accuracy | Sample Size |\n"
md_table += "|-----------|------------|----------|-------------|\n"

print(f"{'Phenotype':<25} {'Best Model':<35} {'Accuracy':<10} {'Samples':<10}")
print("-"*70)

for _, row in df.iterrows():
    phenotype_display = row['phenotype'].replace('_', ' ').title()
    model_short = row['model'].split('/')[-1] if '/' in row['model'] else row['model']

    md_table += f"| {phenotype_display} | {row['model']} | {row['accuracy']:.1f}% | {row['total_predictions']:,} |\n"
    print(f"{phenotype_display:<25} {row['model']:<35} {row['accuracy']:>6.1f}% {row['total_predictions']:>8,}")

# Save markdown file
with open('best_model_phenotype_combinations.md', 'w') as f:
    f.write(md_table)

print("\n" + "="*70)
print(f"Results saved to: best_model_phenotype_combinations.md")

# Now check Gemini-2.5-pro specifically for spore formation
print("\n" + "="*70)
print("GEMINI-2.5-PRO PERFORMANCE ON SPORE FORMATION")
print("="*70)

gemini_query = """
SELECT
    pr.model,
    COUNT(*) as total,
    SUM(CASE WHEN LOWER(pr.spore_formation) = LOWER(gt.spore_formation) THEN 1 ELSE 0 END) as correct,
    ROUND(100.0 * SUM(CASE WHEN LOWER(pr.spore_formation) = LOWER(gt.spore_formation) THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy
FROM processing_results pr
INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
WHERE pr.model = 'google/gemini-2.5-pro'
    AND pr.status = 'completed'
    AND gt.spore_formation IS NOT NULL AND gt.spore_formation != ''
    AND pr.spore_formation IS NOT NULL AND pr.spore_formation != ''
GROUP BY pr.model;
"""

gemini_df = pd.read_sql_query(gemini_query, conn)

if len(gemini_df) > 0:
    row = gemini_df.iloc[0]
    print(f"Model: {row['model']}")
    print(f"Total predictions: {row['total']:,}")
    print(f"Correct predictions: {row['correct']:,}")
    print(f"Accuracy: {row['accuracy']:.2f}%")
else:
    print("No data found for google/gemini-2.5-pro on spore formation")

# Get confusion matrix data
cm_query = """
SELECT
    gt.spore_formation as actual,
    pr.spore_formation as predicted,
    COUNT(*) as count
FROM processing_results pr
INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
WHERE pr.model = 'google/gemini-2.5-pro'
    AND pr.status = 'completed'
    AND gt.spore_formation IS NOT NULL AND gt.spore_formation != ''
    AND pr.spore_formation IS NOT NULL AND pr.spore_formation != ''
GROUP BY gt.spore_formation, pr.spore_formation
ORDER BY count DESC;
"""

print("\n" + "="*70)
print("CONFUSION MATRIX DATA (Top combinations)")
print("="*70)

cm_df = pd.read_sql_query(cm_query, conn)
if len(cm_df) > 0:
    print(f"{'Actual':<20} {'Predicted':<20} {'Count':<10}")
    print("-"*50)
    for _, row in cm_df.head(10).iterrows():
        print(f"{row['actual']:<20} {row['predicted']:<20} {row['count']:<10}")

conn.close()

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)