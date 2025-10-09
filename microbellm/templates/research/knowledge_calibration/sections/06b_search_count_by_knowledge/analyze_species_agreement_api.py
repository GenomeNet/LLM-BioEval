#!/usr/bin/env python3
"""
Analyze model agreement/disagreement on specific species knowledge classifications.
Uses API endpoints instead of direct database access.
Finds species where models most often disagree and agree.
"""

import json
import urllib.request
import urllib.parse
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple
import argparse

def fetch_all_models(api_url: str = 'http://localhost:5050') -> List[str]:
    """Get list of all models from the API."""
    endpoint = f"{api_url}/api/knowledge_analysis_data"

    try:
        with urllib.request.urlopen(endpoint, timeout=30) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))

                all_models = set()
                for file_data in data['knowledge_analysis'].values():
                    if file_data.get('has_type_column'):
                        for templates in file_data.get('types', {}).values():
                            if 'template3_knowlege' in templates:
                                all_models.update(templates['template3_knowlege'].keys())

                return list(all_models)
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def fetch_model_predictions(api_url: str, model: str, species_file: str = 'wa_with_gcount.txt') -> Dict:
    """
    Fetch search count data for a model which includes knowledge classifications.
    """
    endpoint = f"{api_url}/api/search_count_by_knowledge"
    params = f"?species_file={species_file}&model={urllib.parse.quote(model)}"

    try:
        with urllib.request.urlopen(endpoint + params, timeout=30) as resp:
            if resp.status == 200:
                result = json.loads(resp.read().decode('utf-8'))
                if result.get('success'):
                    predictions = {}
                    raw_data = result.get('raw_data', {})

                    # Extract species and their knowledge classifications
                    for knowledge_level, species_list in raw_data.items():
                        if knowledge_level != 'NA':  # Use lowercase 'na'
                            level = knowledge_level.lower()
                        else:
                            level = 'na'

                        for item in species_list:
                            species = item.get('species')
                            if species:
                                predictions[species] = level

                    return predictions
    except:
        pass

    return {}

def get_all_species_predictions(api_url: str = 'http://localhost:5050') -> Dict:
    """
    Get all species predictions by fetching from each model.
    Returns: dict of species -> model -> knowledge_group
    """

    print("Fetching list of models...")
    all_models = fetch_all_models(api_url)

    if not all_models:
        print("No models found")
        return {}

    print(f"Found {len(all_models)} models to analyze")

    species_predictions = defaultdict(dict)

    # Fetch predictions for each model
    for i, model in enumerate(all_models, 1):
        if i % 10 == 0:
            print(f"  Processing model {i}/{len(all_models)}...")

        predictions = fetch_model_predictions(api_url, model)

        if predictions:
            model_short = model.split('/')[-1] if '/' in model else model
            for species, knowledge_level in predictions.items():
                species_predictions[species][model_short] = knowledge_level

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
                   output_file: str = 'model_agreement_analysis_api.json'):
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
            'avg_agreement_ratio': np.mean([s['agreement_ratio'] for s in high_agreement]) if high_agreement else 0,
            'avg_disagreement_entropy': np.mean([s['entropy'] for s in high_disagreement]) if high_disagreement else 0
        }
    }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nResults exported to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze model agreement on species knowledge classifications using API'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='API URL')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    print("Loading predictions from API...")
    print("This may take a few minutes as we fetch data for each model...")

    species_predictions = get_all_species_predictions(args.api_url)

    if not species_predictions:
        print("No predictions found")
        return

    print("\nAnalyzing agreement/disagreement...")
    high_agreement, high_disagreement = analyze_agreement(species_predictions)

    # Print results
    print_agreement_analysis(high_agreement, high_disagreement)

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

if __name__ == '__main__':
    main()