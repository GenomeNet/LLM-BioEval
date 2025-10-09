#!/usr/bin/env python3
"""
Analyze model performance across all phenotypes and generate:
1. Confusion matrix for Gemini-2.5-pro on spore formation
2. Table of best model-phenotype combinations
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Connect to database
conn = sqlite3.connect('microbellm.db')

# List of phenotypes to analyze
phenotypes = [
    'gram_staining',
    'motility',
    'aerophilicity',
    'extreme_environment_tolerance',
    'biofilm_formation',
    'animal_pathogenicity',
    'biosafety_level',
    'health_association',
    'host_association',
    'plant_pathogenicity',
    'spore_formation',
    'hemolysis',
    'cell_shape'
]

def standardize_value(value, phenotype):
    """Standardize phenotype values for comparison"""
    if pd.isna(value) or value == '' or value is None:
        return None

    value = str(value).lower().strip()

    # Handle common "no information" variations
    if value in ['unknown', 'na', 'n/a', 'not available', 'no information', 'null', 'none']:
        return 'unknown'

    # Binary phenotypes
    binary_phenotypes = ['motility', 'biofilm_formation', 'animal_pathogenicity',
                        'host_association', 'plant_pathogenicity', 'spore_formation']

    if phenotype in binary_phenotypes:
        if value in ['yes', 'true', '1', 'positive', 'motile', 'forms spores', 'spore-forming',
                     'pathogenic', 'pathogen', 'forms biofilm', 'biofilm-forming', 'associated']:
            return 'yes'
        elif value in ['no', 'false', '0', 'negative', 'non-motile', 'non motile', 'immotile',
                       'non-spore-forming', 'non spore forming', 'does not form spores',
                       'non-pathogenic', 'not pathogenic', 'does not form biofilm', 'not associated']:
            return 'no'

    # Keep original for categorical phenotypes
    return value

def calculate_accuracy_for_model_phenotype(model, phenotype):
    """Calculate accuracy for a specific model-phenotype combination"""
    query = f"""
    SELECT
        pr.binomial_name,
        pr.{phenotype} as predicted,
        gt.{phenotype} as ground_truth
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.model = ?
        AND pr.status = 'completed'
        AND gt.{phenotype} IS NOT NULL
        AND gt.{phenotype} != ''
        AND pr.{phenotype} IS NOT NULL
        AND pr.{phenotype} != ''
    """

    df = pd.read_sql_query(query, conn, params=[model])

    if len(df) == 0:
        return None, 0

    # Standardize values
    df['predicted_std'] = df['predicted'].apply(lambda x: standardize_value(x, phenotype))
    df['ground_truth_std'] = df['ground_truth'].apply(lambda x: standardize_value(x, phenotype))

    # Remove rows with None values
    df_clean = df.dropna(subset=['predicted_std', 'ground_truth_std'])

    if len(df_clean) == 0:
        return None, 0

    # Calculate accuracy
    accuracy = accuracy_score(df_clean['ground_truth_std'], df_clean['predicted_std'])

    return accuracy, len(df_clean)

# Get all models
models_query = "SELECT DISTINCT model FROM processing_results WHERE status = 'completed'"
models_df = pd.read_sql_query(models_query, conn)
models = models_df['model'].tolist()

print(f"Found {len(models)} models to analyze")
print(f"Analyzing {len(phenotypes)} phenotypes")
print("="*60)

# Calculate accuracy for all model-phenotype combinations
results = []
for phenotype in phenotypes:
    print(f"Analyzing {phenotype}...")
    best_accuracy = 0
    best_model = None
    best_n_samples = 0

    for model in models:
        accuracy, n_samples = calculate_accuracy_for_model_phenotype(model, phenotype)
        if accuracy is not None and n_samples >= 10:  # Minimum sample size threshold
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_n_samples = n_samples

    if best_model:
        results.append({
            'Phenotype': phenotype.replace('_', ' ').title(),
            'Best Model': best_model,
            'Accuracy': f"{best_accuracy:.2%}",
            'Sample Size': best_n_samples
        })

# Create DataFrame and sort by accuracy
results_df = pd.DataFrame(results)
results_df['Accuracy_numeric'] = results_df['Accuracy'].str.rstrip('%').astype(float) / 100
results_df = results_df.sort_values('Accuracy_numeric', ascending=False)
results_df = results_df.drop('Accuracy_numeric', axis=1)

# Save as markdown table
print("\n" + "="*60)
print("BEST MODEL-PHENOTYPE COMBINATIONS")
print("="*60)

# Generate markdown table
md_table = "# Best Model-Phenotype Combinations\n\n"
md_table += "| Phenotype | Best Model | Accuracy | Sample Size |\n"
md_table += "|-----------|------------|----------|-------------|\n"

for _, row in results_df.iterrows():
    md_table += f"| {row['Phenotype']} | {row['Best Model']} | {row['Accuracy']} | {row['Sample Size']:,} |\n"
    print(f"{row['Phenotype']:<30} {row['Best Model']:<30} {row['Accuracy']:<10} N={row['Sample Size']:,}")

# Save markdown file
with open('best_model_phenotype_combinations.md', 'w') as f:
    f.write(md_table)
    f.write("\n## Summary Statistics\n\n")
    f.write(f"- Total models analyzed: {len(models)}\n")
    f.write(f"- Total phenotypes analyzed: {len(phenotypes)}\n")
    f.write(f"- Average best accuracy: {results_df['Accuracy'].str.rstrip('%').astype(float).mean():.1f}%\n")

print(f"\nMarkdown table saved to: best_model_phenotype_combinations.md")

# Now generate confusion matrix for Gemini-2.5-pro on spore formation
print("\n" + "="*60)
print("GENERATING CONFUSION MATRIX FOR GEMINI-2.5-PRO - SPORE FORMATION")
print("="*60)

query = """
SELECT
    pr.binomial_name,
    pr.spore_formation as predicted,
    gt.spore_formation as ground_truth
FROM processing_results pr
INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
WHERE pr.model = 'google/gemini-2.5-pro'
    AND pr.status = 'completed'
    AND gt.spore_formation IS NOT NULL
    AND gt.spore_formation != ''
    AND pr.spore_formation IS NOT NULL
    AND pr.spore_formation != ''
"""

df_spore = pd.read_sql_query(query, conn)
conn.close()

if len(df_spore) > 0:
    # Standardize values
    df_spore['predicted_std'] = df_spore['predicted'].apply(lambda x: standardize_value(x, 'spore_formation'))
    df_spore['ground_truth_std'] = df_spore['ground_truth'].apply(lambda x: standardize_value(x, 'spore_formation'))

    # Remove None values
    df_spore_clean = df_spore.dropna(subset=['predicted_std', 'ground_truth_std'])

    print(f"Total predictions: {len(df_spore_clean)}")

    if len(df_spore_clean) > 0:
        # Get unique labels
        all_labels = sorted(list(set(df_spore_clean['predicted_std'].unique()) |
                                 set(df_spore_clean['ground_truth_std'].unique())))

        # Create confusion matrix
        cm = confusion_matrix(df_spore_clean['ground_truth_std'],
                            df_spore_clean['predicted_std'],
                            labels=all_labels)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_labels, yticklabels=all_labels,
                    cbar_kws={'label': 'Count'},
                    ax=ax)

        # Customize plot
        ax.set_xlabel('Predicted Spore Formation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Spore Formation (Ground Truth)', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix: Spore Formation Predictions\nModel: google/gemini-2.5-pro',
                    fontsize=14, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig('spore_formation_confusion_matrix_gemini25pro.png', dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: spore_formation_confusion_matrix_gemini25pro.png")

        # Calculate accuracy
        accuracy = accuracy_score(df_spore_clean['ground_truth_std'], df_spore_clean['predicted_std'])
        print(f"Accuracy for Gemini-2.5-pro on spore formation: {accuracy:.2%}")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(df_spore_clean['ground_truth_std'],
                                   df_spore_clean['predicted_std'],
                                   labels=all_labels, zero_division=0))
else:
    print("No data found for Gemini-2.5-pro spore formation predictions")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)