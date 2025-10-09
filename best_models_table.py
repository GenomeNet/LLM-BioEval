#!/usr/bin/env python3
import sqlite3
import pandas as pd

conn = sqlite3.connect('microbellm.db')

phenotypes = [
    'spore_formation',
    'motility',
    'gram_staining',
    'aerophilicity',
    'cell_shape',
    'biofilm_formation',
    'hemolysis'
]

results = []

for phenotype in phenotypes:
    query = f"""
    SELECT
        model,
        COUNT(*) as total,
        SUM(CASE WHEN LOWER(pr.{phenotype}) = LOWER(gt.{phenotype}) THEN 1 ELSE 0 END) as correct
    FROM processing_results pr
    INNER JOIN ground_truth gt ON pr.binomial_name = gt.binomial_name
    WHERE pr.status = 'completed'
        AND gt.{phenotype} IS NOT NULL
        AND LENGTH(gt.{phenotype}) > 0
        AND pr.{phenotype} IS NOT NULL
        AND LENGTH(pr.{phenotype}) > 0
    GROUP BY model
    HAVING total >= 50
    """

    df = pd.read_sql_query(query, conn)
    if len(df) > 0:
        df['accuracy'] = df['correct'] / df['total'] * 100
        best = df.nlargest(1, 'accuracy').iloc[0]
        results.append({
            'Phenotype': phenotype.replace('_', ' ').title(),
            'Best Model': best['model'],
            'Accuracy': f"{best['accuracy']:.1f}%",
            'Sample Size': int(best['total'])
        })
        print(f"Processed {phenotype}")

conn.close()

# Create table
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Accuracy', ascending=False)

# Print markdown table
print("\n# Best Model-Phenotype Combinations\n")
print("| Phenotype | Best Model | Accuracy | Sample Size |")
print("|-----------|------------|----------|-------------|")
for _, row in df_results.iterrows():
    print(f"| {row['Phenotype']} | {row['Best Model']} | {row['Accuracy']} | {row['Sample Size']:,} |")

# Save to file
with open('best_model_phenotype_combinations.md', 'w') as f:
    f.write("# Best Model-Phenotype Combinations\n\n")
    f.write("| Phenotype | Best Model | Accuracy | Sample Size |\n")
    f.write("|-----------|------------|----------|-------------|\n")
    for _, row in df_results.iterrows():
        f.write(f"| {row['Phenotype']} | {row['Best Model']} | {row['Accuracy']} | {row['Sample Size']:,} |\n")

print(f"\nSaved to: best_model_phenotype_combinations.md")