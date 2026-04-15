#!/usr/bin/env python3
"""Generate a comprehensive hallucination-rate figure covering every evaluated model.

This is intended as a supplementary figure summarising the Template 3
(NA-allowed) fabricated-name benchmark across all models tested. Each row
is one model, sorted by overall hallucination rate, with stacked segments
for correct rejection (NA / no_result / inference_failed) vs. the three
hallucination severities (limited / moderate / extensive).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

API_URL = "http://localhost:5050"
OUTPUT_PATH = Path(__file__).parent / "all_models_hallucination.pdf"


def fetch_knowledge_data(api_url: str) -> Dict:
    endpoint = api_url.rstrip('/') + '/api/knowledge_analysis_data'
    with urllib.request.urlopen(endpoint, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def aggregate_model_stats(knowledge_data: Dict) -> Dict[str, Dict[str, int]]:
    """Aggregate Template 3 counts per model across all input types."""
    model_stats: Dict[str, Dict[str, int]] = {}

    for _file, file_data in knowledge_data.items():
        if not file_data.get('has_type_column') or not file_data.get('types'):
            continue
        for input_type, template_data in file_data['types'].items():
            if input_type in ('UNCLASSIFIED', 'WA_WITH_GCOUNT'):
                continue
            for template_name, models in template_data.items():
                if 'template3' not in template_name.lower():
                    continue
                for model_name, stats in models.items():
                    bucket = model_stats.setdefault(model_name, {
                        'na': 0, 'limited': 0, 'moderate': 0,
                        'extensive': 0, 'no_result': 0,
                        'inference_failed': 0, 'total': 0,
                    })
                    for key in ('NA', 'limited', 'moderate', 'extensive',
                                'no_result', 'inference_failed'):
                        bucket[key.lower()] += stats.get(key, 0)
                    bucket['total'] += stats.get('total', 0)
    return model_stats


def build_model_rows(model_stats: Dict[str, Dict[str, int]]) -> List[Dict]:
    rows: List[Dict] = []
    for model, s in model_stats.items():
        total = s['total']
        if total <= 0:
            continue
        correct = s['na'] + s['no_result'] + s['inference_failed']
        hallucinations = s['limited'] + s['moderate'] + s['extensive']
        rows.append({
            'name': model.split('/')[-1][:32],
            'total': total,
            'correct_pct': 100.0 * correct / total,
            'limited_pct': 100.0 * s['limited'] / total,
            'moderate_pct': 100.0 * s['moderate'] / total,
            'extensive_pct': 100.0 * s['extensive'] / total,
            'hallucination_rate': 100.0 * hallucinations / total,
        })
    rows.sort(key=lambda r: r['hallucination_rate'])
    return rows


def plot_all_models(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        raise RuntimeError("No models to plot.")

    n = len(rows)
    y = np.arange(n)
    names = [r['name'] for r in rows]

    correct = np.array([r['correct_pct'] for r in rows])
    limited = np.array([r['limited_pct'] for r in rows])
    moderate = np.array([r['moderate_pct'] for r in rows])
    extensive = np.array([r['extensive_pct'] for r in rows])

    # Panel height scales with number of models; kept tight so the bars
    # touch each other rather than leaving visible stripes of whitespace.
    height = max(2.6, 0.13 * n + 0.6)
    fig, (ax_rate, ax_stack) = plt.subplots(
        1, 2, figsize=(8.5, height),
        gridspec_kw={'width_ratios': [1, 2.1], 'wspace': 0.06},
        sharey=True,
    )

    # Left panel: overall hallucination rate
    rate = limited + moderate + extensive
    ax_rate.barh(y, rate, color='#B22222', height=1.0,
                 edgecolor='white', linewidth=0.3)
    for yi, val in zip(y, rate):
        ax_rate.text(val + 1.2, yi, f"{val:.0f}%",
                     va='center', ha='left', fontsize=6.5)
    ax_rate.set_xlabel('Hallucination Rate (%)', fontsize=8)
    ax_rate.set_xlim(0, 105)
    ax_rate.set_yticks(y)
    ax_rate.set_yticklabels(names, fontsize=6.5)
    # Tight y-limits (bar_height/2 + tiny margin) so the last row sits flush
    # against the x-axis instead of leaving matplotlib's default half-unit gap.
    ax_rate.set_ylim(n - 0.5, -0.5)
    ax_rate.grid(axis='x', linestyle='-', linewidth=0.3,
                 color='#E5E5E5', zorder=0)
    ax_rate.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax_rate.spines[spine].set_visible(False)
    ax_rate.tick_params(axis='x', labelsize=7, length=0)
    ax_rate.tick_params(axis='y', length=0)

    # Right panel: stacked breakdown
    colors = {
        'correct': '#9ca3af',
        'limited': '#f9be24',
        'moderate': '#517abd',
        'extensive': '#45b75f',
    }
    ax_stack.barh(y, correct, color=colors['correct'], height=1.0,
                  edgecolor='white', linewidth=0.3, label='Correct rejection (NA)')
    ax_stack.barh(y, limited, left=correct, color=colors['limited'], height=1.0,
                  edgecolor='white', linewidth=0.3, label='Limited hallucination')
    ax_stack.barh(y, moderate, left=correct + limited, color=colors['moderate'],
                  height=1.0, edgecolor='white', linewidth=0.3,
                  label='Moderate hallucination')
    ax_stack.barh(y, extensive, left=correct + limited + moderate,
                  color=colors['extensive'], height=1.0,
                  edgecolor='white', linewidth=0.3,
                  label='Extensive hallucination')

    ax_stack.set_xlim(0, 100)
    ax_stack.set_xlabel('Response distribution on fabricated names (%)', fontsize=8)
    ax_stack.grid(axis='x', linestyle='-', linewidth=0.3,
                  color='#E5E5E5', zorder=0)
    ax_stack.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax_stack.spines[spine].set_visible(False)
    ax_stack.tick_params(axis='x', labelsize=7, length=0)
    ax_stack.tick_params(axis='y', length=0)
    ax_stack.legend(
        loc='lower center', bbox_to_anchor=(0.5, -0.14 - 0.4 / height),
        ncol=4, frameon=False, fontsize=7, handlelength=1.2,
        columnspacing=1.2, handletextpad=0.5,
    )

    fig.suptitle('Hallucination rates on fabricated names (Template 3, all models)',
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='pdf', facecolor='white', edgecolor='none')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-url', default=os.environ.get('MICROBELLM_API_URL', API_URL))
    parser.add_argument('--output', default=str(OUTPUT_PATH))
    args = parser.parse_args()

    try:
        payload = fetch_knowledge_data(args.api_url)
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(f"Failed to contact API at {args.api_url}: {exc}")

    knowledge_data = payload.get('knowledge_analysis', {})
    if not knowledge_data:
        sys.exit("No knowledge_analysis data returned from API.")

    stats = aggregate_model_stats(knowledge_data)
    rows = build_model_rows(stats)
    if not rows:
        sys.exit("No Template 3 results available to plot.")

    output_path = Path(args.output)
    plot_all_models(rows, output_path)
    print(f"Plotted {len(rows)} models -> {output_path}")


if __name__ == '__main__':
    main()
