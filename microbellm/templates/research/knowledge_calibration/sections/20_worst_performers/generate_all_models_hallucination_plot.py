#!/usr/bin/env python3
"""Generate the hallucination-rate figure covering every evaluated model.

This is the main-text figure summarising the Template 3 (NA-allowed)
fabricated-name benchmark across all evaluated models. Each row is one
model, sorted by overall hallucination rate. The left panel shows the
aggregate hallucination rate; the four right panels decompose the response
distribution separately for each of the four categories of fabricated
binomial names (random English words, pseudo-Latin inventions, and the
two half-real / half-invented hybrids).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

API_URL = "http://localhost:5050"
OUTPUT_PATH = Path(__file__).parent / "all_models_hallucination.pdf"

# The four artificial-name categories, in the order they appear as right-
# side sub-panels. The labels below go on each sub-panel title.
CATEGORY_ORDER = (
    'random_words',
    'latin_random_words',
    'real_genus_latin_strain',
    'latin_genus_real_strain',
)
CATEGORY_LABELS = {
    'random_words': 'English words\n(e.g. "Amber Field")',
    'latin_random_words': 'Pseudo-Latin\n(e.g. "Solispira lumina")',
    'real_genus_latin_strain': 'Real genus + Latin sp.\n(e.g. "Mycobacterium ferrum")',
    'latin_genus_real_strain': 'Latin genus + real sp.\n(e.g. "Temporibacter coli")',
}


def fetch_knowledge_data(api_url: str) -> Dict:
    endpoint = api_url.rstrip('/') + '/api/knowledge_analysis_data'
    with urllib.request.urlopen(endpoint, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def aggregate_model_stats(knowledge_data: Dict) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Return {model: {category: counts}} for Template 3.

    Unlike the earlier flat-aggregation, this preserves the per-category
    split so the right panels can show one stacked bar per category.
    """
    model_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

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
                    by_cat = model_stats.setdefault(model_name, {})
                    bucket = by_cat.setdefault(input_type, {
                        'na': 0, 'limited': 0, 'moderate': 0,
                        'extensive': 0, 'no_result': 0,
                        'inference_failed': 0, 'total': 0,
                    })
                    for key in ('NA', 'limited', 'moderate', 'extensive',
                                'no_result', 'inference_failed'):
                        bucket[key.lower()] += stats.get(key, 0)
                    bucket['total'] += stats.get('total', 0)
    return model_stats


def _per_cat_percentages(c: Optional[Dict[str, int]]) -> Optional[Dict[str, float]]:
    if not c:
        return None
    total = c.get('total', 0)
    if total <= 0:
        return None
    correct = c.get('na', 0) + c.get('no_result', 0) + c.get('inference_failed', 0)
    return {
        'correct_pct': 100.0 * correct / total,
        'limited_pct': 100.0 * c.get('limited', 0) / total,
        'moderate_pct': 100.0 * c.get('moderate', 0) / total,
        'extensive_pct': 100.0 * c.get('extensive', 0) / total,
    }


def build_model_rows(model_stats: Dict[str, Dict[str, Dict[str, int]]]) -> List[Dict]:
    rows: List[Dict] = []
    for model, by_cat in model_stats.items():
        # Overall counts aggregated over all categories (drive the sort + left panel).
        overall_total = sum(c['total'] for c in by_cat.values())
        if overall_total <= 0:
            continue
        overall_halluc = sum(
            c.get('limited', 0) + c.get('moderate', 0) + c.get('extensive', 0)
            for c in by_cat.values()
        )

        per_cat: Dict[str, Optional[Dict[str, float]]] = {}
        for cat in CATEGORY_ORDER:
            per_cat[cat] = _per_cat_percentages(by_cat.get(cat))

        rows.append({
            'name': model.split('/')[-1][:32],
            'total': overall_total,
            'hallucination_rate': 100.0 * overall_halluc / overall_total,
            'per_cat': per_cat,
        })
    rows.sort(key=lambda r: r['hallucination_rate'])
    return rows


def plot_all_models(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        raise RuntimeError("No models to plot.")

    n = len(rows)
    y = np.arange(n)
    names = [r['name'] for r in rows]

    # 5 columns: 1 aggregate rate + 4 stratified category panels.
    # Width scales with number of panels; height scales with number of models.
    height = max(3.0, 0.13 * n + 0.9)
    fig, axes = plt.subplots(
        1, 5, figsize=(13.5, height),
        gridspec_kw={
            'width_ratios': [1.1, 1.7, 1.7, 1.7, 1.7],
            'wspace': 0.10,
        },
        sharey=True,
    )
    ax_rate = axes[0]
    cat_axes = list(axes[1:])

    # --- Left panel: overall hallucination rate -------------------------- #
    rate = np.array([r['hallucination_rate'] for r in rows])
    ax_rate.barh(y, rate, color='#B22222', height=1.0,
                 edgecolor='white', linewidth=0.3)
    for yi, val in zip(y, rate):
        ax_rate.text(val + 1.2, yi, f"{val:.0f}%",
                     va='center', ha='left', fontsize=6.5)
    ax_rate.set_xlabel('Overall hallucination\nrate (%)', fontsize=7.5)
    ax_rate.set_xlim(0, 105)
    ax_rate.set_yticks(y)
    ax_rate.set_yticklabels(names, fontsize=6.5)
    ax_rate.set_ylim(n - 0.5, -0.5)
    ax_rate.grid(axis='x', linestyle='-', linewidth=0.3,
                 color='#E5E5E5', zorder=0)
    ax_rate.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax_rate.spines[spine].set_visible(False)
    ax_rate.tick_params(axis='x', labelsize=7, length=0)
    ax_rate.tick_params(axis='y', length=0)

    # --- Right panels: one stacked bar per category ---------------------- #
    colors = {
        'correct': '#9ca3af',
        'limited': '#f9be24',
        'moderate': '#517abd',
        'extensive': '#45b75f',
    }
    legend_handles = []
    legend_labels = [
        'Correct rejection (NA)',
        'Limited hallucination',
        'Moderate hallucination',
        'Extensive hallucination',
    ]

    for i, cat in enumerate(CATEGORY_ORDER):
        ax = cat_axes[i]

        correct = np.array([
            (r['per_cat'].get(cat) or {}).get('correct_pct', np.nan) for r in rows
        ])
        limited = np.array([
            (r['per_cat'].get(cat) or {}).get('limited_pct', 0.0) for r in rows
        ])
        moderate = np.array([
            (r['per_cat'].get(cat) or {}).get('moderate_pct', 0.0) for r in rows
        ])
        extensive = np.array([
            (r['per_cat'].get(cat) or {}).get('extensive_pct', 0.0) for r in rows
        ])

        mask = ~np.isnan(correct)
        cbar = ax.barh(y[mask], correct[mask], color=colors['correct'],
                       height=1.0, edgecolor='white', linewidth=0.3)
        lbar = ax.barh(y[mask], limited[mask], left=correct[mask],
                       color=colors['limited'], height=1.0,
                       edgecolor='white', linewidth=0.3)
        mbar = ax.barh(y[mask], moderate[mask],
                       left=correct[mask] + limited[mask],
                       color=colors['moderate'], height=1.0,
                       edgecolor='white', linewidth=0.3)
        ebar = ax.barh(y[mask], extensive[mask],
                       left=correct[mask] + limited[mask] + moderate[mask],
                       color=colors['extensive'], height=1.0,
                       edgecolor='white', linewidth=0.3)

        if i == 0:
            legend_handles = [cbar, lbar, mbar, ebar]

        ax.set_xlim(0, 100)
        ax.set_xlabel('% of responses', fontsize=7.5)
        ax.set_title(CATEGORY_LABELS[cat], fontsize=8)
        ax.grid(axis='x', linestyle='-', linewidth=0.3,
                color='#E5E5E5', zorder=0)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='x', labelsize=7, length=0)
        ax.tick_params(axis='y', length=0)

    # Shared legend at the bottom.
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.002),
        ncol=4, frameon=False, fontsize=7.5,
        handlelength=1.2, columnspacing=1.6, handletextpad=0.5,
    )

    fig.suptitle(
        'Hallucination rates on fabricated names, stratified by name category (Template 3)',
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
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
