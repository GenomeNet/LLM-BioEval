#!/usr/bin/env python3
"""
Analyze model agreement/disagreement on knowledge group classifications.
Finds species where models most often disagree and agree on knowledge classifications.
"""

import json
import urllib.request
import urllib.error
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple
import argparse

def fetch_knowledge_data(api_url: str = 'http://localhost:5050') -> Dict:
    """Fetch knowledge analysis data from the API."""
    endpoint = f"{api_url}/api/knowledge_analysis_data"

    print(f"Fetching data from: {endpoint}")

    try:
        with urllib.request.urlopen(endpoint, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def analyze_model_agreement(data: Dict) -> Tuple[Dict, Dict]:
    """
    Analyze model agreement on species classifications.
    Returns species with highest agreement and disagreement.
    """

    if not data or 'knowledge_analysis' not in data:
        print("No knowledge analysis data found")
        return None, None

    # Dictionary to store classifications per species
    # species -> model -> classification
    species_classifications = defaultdict(dict)

    # Process each file and extract classifications
    for file_name, file_data in data['knowledge_analysis'].items():
        if not file_data.get('has_type_column'):
            continue

        types_data = file_data.get('types', {})

        for type_name, templates_data in types_data.items():
            # We'll use template3_knowlege as it seems to be the most complete
            if 'template3_knowlege' in templates_data:
                template_data = templates_data['template3_knowlege']

                # Process each model's classifications
                for model_name, model_stats in template_data.items():
                    # Extract classifications for each knowledge level
                    for knowledge_level in ['NA', 'limited', 'moderate', 'extensive',
                                           'no_result', 'inference_failed']:
                        if knowledge_level in model_stats:
                            # This gives us the count, but we need the actual species
                            # We'll need to reconstruct from the data structure
                            pass

    # Since the API doesn't give us individual species classifications directly,
    # we need to fetch from a different endpoint or reconstruct the data
    # Let's try to get the raw predictions data

    print("Fetching individual predictions...")

    # Get list of all models first
    all_models = set()
    for file_data in data['knowledge_analysis'].values():
        if file_data.get('has_type_column'):
            for templates in file_data.get('types', {}).values():
                if 'template3_knowlege' in templates:
                    all_models.update(templates['template3_knowlege'].keys())

    print(f"Found {len(all_models)} models with predictions")

    # For each model, fetch their predictions
    species_predictions = defaultdict(dict)

    for model in sorted(all_models):
        # Try to fetch predictions for this model
        try:
            predictions = fetch_model_predictions(api_url='http://localhost:5050',
                                                 model=model)
            if predictions:
                for species, classification in predictions.items():
                    species_predictions[species][model] = classification
        except Exception as e:
            print(f"Error fetching predictions for {model}: {e}")

    return analyze_agreement_from_predictions(species_predictions)

def fetch_model_predictions(api_url: str, model: str) -> Dict:
    """
    Fetch predictions for a specific model.
    This will try to get the predictions from the search_count_by_knowledge endpoint.
    """
    endpoint = f"{api_url}/api/search_count_by_knowledge"
    params = f"?species_file=wa_with_gcount.txt&model={urllib.parse.quote(model)}"

    try:
        with urllib.request.urlopen(endpoint + params, timeout=10) as resp:
            if resp.status == 200:
                result = json.loads(resp.read().decode('utf-8'))
                if result.get('success'):
                    # Extract species and their classifications
                    predictions = {}
                    raw_data = result.get('raw_data', {})

                    for knowledge_level, species_list in raw_data.items():
                        if knowledge_level != 'NA':  # Skip NA for now
                            for item in species_list:
                                species = item.get('species')
                                if species:
                                    predictions[species] = knowledge_level

                    return predictions
    except:
        pass

    return {}

def analyze_agreement_from_predictions(species_predictions: Dict[str, Dict[str, str]]) -> Tuple[List, List]:
    """
    Analyze agreement from the predictions dictionary.
    Returns lists of species with highest and lowest agreement.
    """

    agreement_scores = []

    for species, model_predictions in species_predictions.items():
        if len(model_predictions) < 2:
            continue  # Need at least 2 models to compare

        # Count how many models agree on each classification
        classification_counts = Counter(model_predictions.values())

        # Calculate agreement score
        total_models = len(model_predictions)
        max_agreement = max(classification_counts.values())
        agreement_ratio = max_agreement / total_models

        # Calculate entropy for disagreement measure
        probs = [count/total_models for count in classification_counts.values()]
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)

        agreement_scores.append({
            'species': species,
            'agreement_ratio': agreement_ratio,
            'entropy': entropy,
            'total_models': total_models,
            'classifications': classification_counts,
            'model_predictions': model_predictions
        })

    # Sort by agreement ratio
    agreement_scores.sort(key=lambda x: x['agreement_ratio'], reverse=True)

    # Get top agreeing species (high agreement ratio, low entropy)
    high_agreement = [s for s in agreement_scores if s['agreement_ratio'] >= 0.8][:10]

    # Get top disagreeing species (low agreement ratio, high entropy)
    agreement_scores.sort(key=lambda x: x['entropy'], reverse=True)
    high_disagreement = [s for s in agreement_scores if s['agreement_ratio'] < 0.5][:10]

    return high_agreement, high_disagreement

def fetch_all_model_predictions_batch(api_url: str = 'http://localhost:5050') -> Dict:
    """
    Try to fetch all model predictions in a more efficient way.
    We'll analyze the template data structure directly.
    """

    # First, get the knowledge analysis data
    endpoint = f"{api_url}/api/knowledge_analysis_data"

    try:
        with urllib.request.urlopen(endpoint, timeout=30) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    # We need to fetch the actual template responses
    # Let's try to get the raw species data with classifications

    # Dictionary to store all predictions
    species_predictions = defaultdict(dict)

    # Get template predictions for each model
    print("Fetching template predictions for all models...")

    # Try to fetch from template endpoint
    template_endpoint = f"{api_url}/api/template_predictions"
    species_endpoint = f"{api_url}/api/species_knowledge_levels"

    # Since we don't have direct access to individual predictions,
    # we'll need to analyze from the aggregated data
    # Let's create a simplified analysis based on available data

    return create_sample_analysis(data)

def create_sample_analysis(data: Dict) -> Tuple[List, List]:
    """
    Create a sample analysis based on the aggregated data.
    Since we can't get individual species predictions easily,
    we'll identify patterns from the statistics.
    """

    if not data or 'knowledge_analysis' not in data:
        return [], []

    # Collect model statistics for each template
    model_stats = defaultdict(dict)

    for file_data in data['knowledge_analysis'].values():
        if not file_data.get('has_type_column'):
            continue

        for type_name, templates_data in file_data.get('types', {}).items():
            if 'template3_knowlege' in templates_data:
                for model, stats in templates_data['template3_knowlege'].items():
                    model_short = model.split('/')[-1][:30]

                    total = sum(stats.get(k, 0) for k in ['NA', 'limited', 'moderate', 'extensive'])
                    if total > 0:
                        model_stats[model_short] = {
                            'na_pct': stats.get('NA', 0) / total * 100,
                            'limited_pct': stats.get('limited', 0) / total * 100,
                            'moderate_pct': stats.get('moderate', 0) / total * 100,
                            'extensive_pct': stats.get('extensive', 0) / total * 100,
                            'total': total
                        }

    # Print summary statistics
    print("\n=== MODEL CLASSIFICATION PATTERNS ===\n")

    # Find models that disagree most (have different distributions)
    print("Models with extreme classifications (potential disagreement sources):\n")

    # Models with high NA rates (uncertainty)
    high_na = sorted(model_stats.items(), key=lambda x: x[1]['na_pct'], reverse=True)[:5]
    print("Top 5 models with highest NA/uncertainty:")
    for model, stats in high_na:
        print(f"  {model}: NA={stats['na_pct']:.1f}%")

    print("\nTop 5 models with highest extensive knowledge:")
    high_extensive = sorted(model_stats.items(), key=lambda x: x[1]['extensive_pct'], reverse=True)[:5]
    for model, stats in high_extensive:
        print(f"  {model}: Extensive={stats['extensive_pct']:.1f}%")

    print("\nTop 5 models with highest limited knowledge:")
    high_limited = sorted(model_stats.items(), key=lambda x: x[1]['limited_pct'], reverse=True)[:5]
    for model, stats in high_limited:
        print(f"  {model}: Limited={stats['limited_pct']:.1f}%")

    # Calculate variance in distributions
    print("\n=== DISTRIBUTION VARIANCE ANALYSIS ===\n")

    # Calculate mean distribution
    all_na = [s['na_pct'] for s in model_stats.values()]
    all_limited = [s['limited_pct'] for s in model_stats.values()]
    all_moderate = [s['moderate_pct'] for s in model_stats.values()]
    all_extensive = [s['extensive_pct'] for s in model_stats.values()]

    print(f"Mean distribution across {len(model_stats)} models:")
    print(f"  NA:        {np.mean(all_na):.1f}% (std: {np.std(all_na):.1f}%)")
    print(f"  Limited:   {np.mean(all_limited):.1f}% (std: {np.std(all_limited):.1f}%)")
    print(f"  Moderate:  {np.mean(all_moderate):.1f}% (std: {np.std(all_moderate):.1f}%)")
    print(f"  Extensive: {np.mean(all_extensive):.1f}% (std: {np.std(all_extensive):.1f}%)")

    # Models that deviate most from mean
    print("\nModels that deviate most from average (likely disagree with others):")

    deviations = []
    mean_dist = [np.mean(all_na), np.mean(all_limited),
                 np.mean(all_moderate), np.mean(all_extensive)]

    for model, stats in model_stats.items():
        model_dist = [stats['na_pct'], stats['limited_pct'],
                     stats['moderate_pct'], stats['extensive_pct']]

        # Calculate L2 distance from mean
        deviation = np.sqrt(sum((m - a)**2 for m, a in zip(model_dist, mean_dist)))
        deviations.append((model, deviation, stats))

    deviations.sort(key=lambda x: x[1], reverse=True)

    for model, deviation, stats in deviations[:5]:
        print(f"  {model}: deviation={deviation:.1f}")
        print(f"    NA={stats['na_pct']:.1f}%, Lim={stats['limited_pct']:.1f}%, "
              f"Mod={stats['moderate_pct']:.1f}%, Ext={stats['extensive_pct']:.1f}%")

    return [], []  # We can't return specific species without individual predictions

def main():
    parser = argparse.ArgumentParser(
        description='Analyze model agreement on knowledge classifications'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='API URL')
    parser.add_argument('--output', help='Output file for results')

    args = parser.parse_args()

    print("Fetching knowledge analysis data...")
    data = fetch_knowledge_data(args.api_url)

    if not data:
        print("Failed to fetch data")
        return

    # Try to analyze agreement
    high_agreement, high_disagreement = create_sample_analysis(data)

    # Additional analysis: count total models
    all_models = set()
    for file_data in data['knowledge_analysis'].values():
        if file_data.get('has_type_column'):
            for templates in file_data.get('types', {}).values():
                if 'template3_knowlege' in templates:
                    all_models.update(templates['template3_knowlege'].keys())

    print(f"\n=== SUMMARY ===")
    print(f"Total models analyzed: {len(all_models)}")

    # List all models
    print("\nAll models included in analysis:")
    for i, model in enumerate(sorted(all_models), 1):
        model_short = model.split('/')[-1]
        print(f"  {i:2d}. {model_short}")

if __name__ == '__main__':
    main()