#!/usr/bin/env python3
"""
Extract the actual names of species where all models agree (unanimous agreement)
and save them to a text file grouped by knowledge level.
"""

import sqlite3
from collections import defaultdict, Counter
import math

def load_all_species_predictions():
    """Load all predictions for real species from database."""
    print("Loading all species predictions from database...")

    # Connect to database
    db_path = '/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all predictions for real species from processing_results
    query = """
    SELECT binomial_name, knowledge_group, model
    FROM processing_results
    WHERE species_file = 'wa_with_gcount.txt'
    AND system_template = 'templates/system/template3_knowlege.txt'
    AND knowledge_group IN ('NA', 'limited', 'moderate', 'extensive')
    AND status = 'completed'
    ORDER BY binomial_name, model
    """

    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()

    # Group predictions by species
    species_predictions = defaultdict(list)
    for species, knowledge_group, model in results:
        if knowledge_group:
            species_predictions[species].append(knowledge_group.lower())

    print(f"Loaded predictions for {len(species_predictions)} species")
    print(f"Total predictions: {len(results)}")

    return species_predictions

def calculate_entropy(predictions):
    """Calculate Shannon entropy for a set of predictions."""
    if not predictions:
        return 0

    # Count occurrences
    counts = Counter(predictions)
    total = sum(counts.values())

    # Calculate entropy
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return entropy

def find_unanimous_species():
    """Find species where all models agree (entropy = 0)."""

    species_predictions = load_all_species_predictions()

    unanimous_species = defaultdict(list)
    species_with_entropy = []

    for species_name, predictions in species_predictions.items():
        # Skip species with too few predictions
        if len(predictions) < 10:
            continue

        # Calculate entropy
        entropy = calculate_entropy(predictions)
        species_with_entropy.append((species_name, entropy, predictions))

        # If entropy is 0 (or very close), all models agree
        if entropy < 0.001:
            unique_preds = list(set(predictions))
            if len(unique_preds) == 1:
                knowledge_group = unique_preds[0]
                unanimous_species[knowledge_group].append(species_name)

    # Sort species with low entropy to find near-unanimous ones
    species_with_entropy.sort(key=lambda x: x[1])

    print(f"\nFound {sum(len(v) for v in unanimous_species.values())} species with unanimous agreement")
    print("\nLowest entropy species (top 10):")
    for species, entropy, preds in species_with_entropy[:10]:
        unique = Counter(preds)
        print(f"  {species}: H={entropy:.4f}, {dict(unique)}")

    return unanimous_species

def save_unanimous_species_to_file(unanimous_species, output_file='unanimous_species_list.txt'):
    """Save the unanimous species to a text file grouped by knowledge level."""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPECIES WITH UNANIMOUS MODEL AGREEMENT\n")
        f.write("="*80 + "\n\n")

        total_count = sum(len(species_list) for species_list in unanimous_species.values())
        f.write(f"Total species with unanimous agreement: {total_count}\n\n")

        # Order for display
        knowledge_order = ['extensive', 'moderate', 'limited', 'na']

        for knowledge_group in knowledge_order:
            species_list = unanimous_species.get(knowledge_group, [])
            if species_list:
                f.write("-"*60 + "\n")
                f.write(f"{knowledge_group.upper()} KNOWLEDGE ({len(species_list)} species)\n")
                f.write("-"*60 + "\n")

                # Sort species alphabetically
                for species in sorted(species_list):
                    f.write(f"  - {species}\n")

                f.write("\n")

        # Summary statistics
        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n")

        for knowledge_group in knowledge_order:
            count = len(unanimous_species.get(knowledge_group, []))
            if total_count > 0:
                percentage = (count / total_count) * 100
                f.write(f"{knowledge_group.capitalize():12s}: {count:3d} species ({percentage:5.1f}%)\n")

    print(f"\nSaved unanimous species list to {output_file}")

def main():
    """Main function to extract and save unanimous species."""

    print("="*60)
    print("Extracting unanimous species from database")
    print("="*60)

    # Find unanimous species
    unanimous_species = find_unanimous_species()

    if not unanimous_species:
        print("No unanimous species found")
        return

    # Print summary
    print("\nSummary by knowledge group:")
    for knowledge_group in ['extensive', 'moderate', 'limited', 'na']:
        species_list = unanimous_species.get(knowledge_group, [])
        if species_list:
            print(f"  {knowledge_group.capitalize():12s}: {len(species_list):3d} species")
            # Show first 3 examples
            for species in species_list[:3]:
                print(f"    - {species}")
            if len(species_list) > 3:
                print(f"    ... and {len(species_list) - 3} more")

    # Save to file
    save_unanimous_species_to_file(unanimous_species)

    print("\n" + "="*60)
    print("Extraction complete!")

if __name__ == '__main__':
    main()