#!/usr/bin/env python3
"""
Generate confusion matrix for spore formation predictions by Gemini-2.5-pro model
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Connect to database
conn = sqlite3.connect('microbellm.db')

# Query to get spore formation predictions and ground truth for Gemini-2.5-pro
query = """
SELECT
    pr.binomial_name,
    pr.spore_formation as predicted,
    pr.raw_spore_formation as raw_predicted,
    s.spore_formation as ground_truth
FROM processing_results pr
INNER JOIN species s ON pr.binomial_name = s.binomial_name
WHERE pr.model = 'google/gemini-2.5-pro'
    AND pr.status = 'completed'
    AND s.spore_formation IS NOT NULL
    AND s.spore_formation != ''
    AND pr.spore_formation IS NOT NULL
    AND pr.spore_formation != ''
ORDER BY pr.binomial_name;
"""

# Load data
df = pd.read_sql_query(query, conn)
conn.close()

print(f"Total predictions found: {len(df)}")
print(f"\nUnique predicted values: {df['predicted'].unique()}")
print(f"Unique ground truth values: {df['ground_truth'].unique()}")

# Standardize values (handle case variations and common formats)
def standardize_spore_value(value):
    if pd.isna(value) or value == '':
        return None
    value = str(value).lower().strip()

    # Map common variations to standard values
    if value in ['yes', 'true', '1', 'spore-forming', 'spore forming', 'forms spores']:
        return 'yes'
    elif value in ['no', 'false', '0', 'non-spore-forming', 'non spore forming', 'does not form spores']:
        return 'no'
    elif value in ['unknown', 'na', 'n/a', 'not available', 'no information']:
        return 'unknown'
    else:
        # Keep original value if it doesn't match common patterns
        return value

# Apply standardization
df['predicted_std'] = df['predicted'].apply(standardize_spore_value)
df['ground_truth_std'] = df['ground_truth'].apply(standardize_spore_value)

# Remove rows with None values after standardization
df_clean = df.dropna(subset=['predicted_std', 'ground_truth_std'])

print(f"\nAfter standardization and cleaning: {len(df_clean)} samples")
print(f"Standardized predicted values: {df_clean['predicted_std'].unique()}")
print(f"Standardized ground truth values: {df_clean['ground_truth_std'].unique()}")

if len(df_clean) == 0:
    print("No valid data found for confusion matrix!")
    exit(1)

# Get unique labels for confusion matrix
all_labels = sorted(list(set(df_clean['predicted_std'].unique()) | set(df_clean['ground_truth_std'].unique())))

# Create confusion matrix
cm = confusion_matrix(df_clean['ground_truth_std'], df_clean['predicted_std'], labels=all_labels)

# Create figure with better styling
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

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Add grid for better readability
ax.set_facecolor('#f0f0f0')
for i in range(len(all_labels) + 1):
    ax.axhline(i, color='white', lw=2)
    ax.axvline(i, color='white', lw=2)

plt.tight_layout()

# Save figure
plt.savefig('spore_formation_confusion_matrix_gemini25pro.png', dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix saved to: spore_formation_confusion_matrix_gemini25pro.png")

# Show plot
plt.show()

# Print classification report
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(df_clean['ground_truth_std'], df_clean['predicted_std'],
                            labels=all_labels, zero_division=0))

# Calculate and print additional metrics
total_predictions = len(df_clean)
correct_predictions = sum(df_clean['predicted_std'] == df_clean['ground_truth_std'])
accuracy = correct_predictions / total_predictions * 100

print("\n" + "="*60)
print("Summary Statistics:")
print("="*60)
print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Overall accuracy: {accuracy:.2f}%")

# Print confusion matrix as DataFrame for better readability
print("\n" + "="*60)
print("Confusion Matrix Table:")
print("="*60)
cm_df = pd.DataFrame(cm, index=[f"Actual: {label}" for label in all_labels],
                     columns=[f"Predicted: {label}" for label in all_labels])
print(cm_df)

# Show distribution of predictions vs ground truth
print("\n" + "="*60)
print("Distribution of Values:")
print("="*60)
print("\nGround Truth Distribution:")
print(df_clean['ground_truth_std'].value_counts())
print("\nPredicted Distribution:")
print(df_clean['predicted_std'].value_counts())