#!/usr/bin/env python3
"""
Analyze model agreement/disagreement on specific species knowledge classifications.
Finds species where models most often disagree and agree.
"""

import sqlite3
import json
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple
import argparse

def get_species_predictions(db_path: str = 'microbellm.db',
                           template: str = 'template3_knowlege') -> Dict:
    """
    Get all predictions for a specific template from the database.
    Returns: dict of species -> model -> knowledge_group
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query for template3_knowlege predictions from processing_results table
    query = """
    SELECT binomial_name, model, knowledge_group
    FROM processing_results
    WHERE user_template LIKE ?
    AND knowledge_group IS NOT NULL
    AND knowledge_group != ''
    AND status = 'completed'
    """

    cursor.execute(query, (f'%{template}%',))
    results = cursor.fetchall()

    # Organize by species
    species_predictions = defaultdict(dict)

    for species, model, knowledge_group in results:
        if species and model and knowledge_group:
            # Clean up model name
            model_short = model.split('/')[-1] if '/' in model else model
            species_predictions[species][model_short] = knowledge_group.lower()

    conn.close()

    print(f"Found {len(species_predictions)} species with predictions")
    print(f"Total predictions: {sum(len(m) for m in species_predictions.values())}")

    return species_predictions

def analyze_agreement(species_predictions: Dict) -> Tuple[List, List]:
    """
    Analyze agreement/disagreement for each species.
    Returns species with highest agreement and highest disagreement.
    """

    agreement_analysis = []

    for species, model_predictions in species_predictions.items():
        if len(model_predictions) < 5:  # Need sufficient models for meaningful analysis
            continue

        # Count classifications
        classification_counts = Counter(model_predictions.values())

        # Calculate metrics
        total_models = len(model_predictions)
        most_common_class = classification_counts.most_common(1)[0][0]
        most_common_count = classification_counts.most_common(1)[0][1]
        agreement_ratio = most_common_count / total_models

        # Calculate entropy for disagreement measure
        probs = [count/total_models for count in classification_counts.values()]
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)

        # Count how many different classifications
        num_different_classes = len(classification_counts)

        agreement_analysis.append({
            'species': species,
            'agreement_ratio': agreement_ratio,
            'entropy': entropy,
            'num_models': total_models,
            'num_classes': num_different_classes,
            'most_common': most_common_class,
            'most_common_count': most_common_count,
            'classifications': dict(classification_counts),
            'model_predictions': model_predictions
        })

    # Sort by agreement for high agreement species
    agreement_analysis.sort(key=lambda x: (-x['agreement_ratio'], -x['num_models']))
    high_agreement = agreement_analysis[:20]

    # Sort by entropy for high disagreement species
    agreement_analysis.sort(key=lambda x: (-x['entropy'], -x['num_models']))
    high_disagreement = [s for s in agreement_analysis if s['num_classes'] >= 3][:20]

    return high_agreement, high_disagreement

def print_agreement_analysis(high_agreement: List, high_disagreement: List):
    """Print detailed analysis of agreement and disagreement."""

    print("\n" + "="*80)
    print("SPECIES WITH HIGHEST MODEL AGREEMENT")
    print("="*80)

    for i, species_data in enumerate(high_agreement[:10], 1):
        print(f"\n{i}. {species_data['species']}")
        print(f"   Agreement: {species_data['agreement_ratio']:.1%} "
              f"({species_data['most_common_count']}/{species_data['num_models']} models)")
        print(f"   Consensus: {species_data['most_common'].upper()}")
        print(f"   Distribution: {species_data['classifications']}")

        # Show which models agree
        agreeing_models = [m for m, c in species_data['model_predictions'].items()
                          if c == species_data['most_common']]
        if len(agreeing_models) <= 10:
            print(f"   Agreeing models: {', '.join(agreeing_models[:5])}")
            if len(agreeing_models) > 5:
                print(f"                    {', '.join(agreeing_models[5:10])}")

    print("\n" + "="*80)
    print("SPECIES WITH HIGHEST MODEL DISAGREEMENT")
    print("="*80)

    for i, species_data in enumerate(high_disagreement[:10], 1):
        print(f"\n{i}. {species_data['species']}")
        print(f"   Entropy: {species_data['entropy']:.2f}")
        print(f"   {species_data['num_classes']} different classifications across "
              f"{species_data['num_models']} models")
        print(f"   Distribution: {species_data['classifications']}")

        # Show breakdown by classification
        for classification, count in sorted(species_data['classifications'].items(),
                                           key=lambda x: -x[1]):
            models_with_this = [m for m, c in species_data['model_predictions'].items()
                               if c == classification]
            print(f"   {classification.upper()} ({count} models): "
                  f"{', '.join(models_with_this[:3])}")
            if len(models_with_this) > 3:
                print(f"      + {len(models_with_this)-3} more...")

def export_results(high_agreement: List, high_disagreement: List,
                   output_file: str = 'model_agreement_analysis.json'):
    """Export results to JSON file."""

    # Prepare data for export
    export_data = {
        'high_agreement_species': [
            {
                'species': s['species'],
                'agreement_ratio': s['agreement_ratio'],
                'num_models': s['num_models'],
                'consensus': s['most_common'],
                'distribution': s['classifications']
            }
            for s in high_agreement
        ],
        'high_disagreement_species': [
            {
                'species': s['species'],
                'entropy': s['entropy'],
                'num_models': s['num_models'],
                'num_classes': s['num_classes'],
                'distribution': s['classifications']
            }
            for s in high_disagreement
        ],
        'summary': {
            'total_high_agreement': len(high_agreement),
            'total_high_disagreement': len(high_disagreement),
            'avg_agreement_ratio': np.mean([s['agreement_ratio'] for s in high_agreement]),
            'avg_disagreement_entropy': np.mean([s['entropy'] for s in high_disagreement])
        }
    }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nResults exported to {output_file}")

def get_model_statistics(species_predictions: Dict) -> Dict:
    """Get statistics about how often each model agrees with consensus."""

    model_agreement_stats = defaultdict(lambda: {'agree': 0, 'disagree': 0, 'total': 0})

    for species, model_predictions in species_predictions.items():
        if len(model_predictions) < 5:
            continue

        # Find consensus (most common classification)
        classification_counts = Counter(model_predictions.values())
        consensus = classification_counts.most_common(1)[0][0]

        # Check each model
        for model, classification in model_predictions.items():
            model_agreement_stats[model]['total'] += 1
            if classification == consensus:
                model_agreement_stats[model]['agree'] += 1
            else:
                model_agreement_stats[model]['disagree'] += 1

    # Calculate agreement percentages
    for model, stats in model_agreement_stats.items():
        stats['agreement_pct'] = stats['agree'] / stats['total'] * 100 if stats['total'] > 0 else 0

    return dict(model_agreement_stats)

def main():
    parser = argparse.ArgumentParser(
        description='Analyze model agreement on species knowledge classifications'
    )
    parser.add_argument('--db', default='microbellm.db',
                       help='Database path')
    parser.add_argument('--template', default='template3_knowlege',
                       help='Template name to analyze')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    print("Loading predictions from database...")
    species_predictions = get_species_predictions(args.db, args.template)

    if not species_predictions:
        print("No predictions found")
        return

    print("\nAnalyzing agreement/disagreement...")
    high_agreement, high_disagreement = analyze_agreement(species_predictions)

    # Print results
    print_agreement_analysis(high_agreement, high_disagreement)

    # Get model statistics
    print("\n" + "="*80)
    print("MODEL AGREEMENT WITH CONSENSUS")
    print("="*80)

    model_stats = get_model_statistics(species_predictions)

    # Sort by agreement percentage
    sorted_models = sorted(model_stats.items(),
                          key=lambda x: x[1]['agreement_pct'],
                          reverse=True)

    print("\nTop 10 models that most often agree with consensus:")
    for i, (model, stats) in enumerate(sorted_models[:10], 1):
        print(f"  {i:2d}. {model:30s} {stats['agreement_pct']:5.1f}% "
              f"({stats['agree']}/{stats['total']})")

    print("\nTop 10 models that most often disagree with consensus:")
    for i, (model, stats) in enumerate(sorted_models[-10:], 1):
        print(f"  {i:2d}. {model:30s} {stats['agreement_pct']:5.1f}% "
              f"({stats['agree']}/{stats['total']})")

    # Export if requested
    if args.output:
        export_results(high_agreement, high_disagreement, args.output)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total species analyzed: {len(species_predictions)}")
    print(f"Species with high agreement (>80%): {len([s for s in high_agreement if s['agreement_ratio'] > 0.8])}")
    print(f"Species with high disagreement (3+ classes): {len(high_disagreement)}")
    print(f"Total models in analysis: {len(model_stats)}")

if __name__ == '__main__':
    main()