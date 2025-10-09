#!/usr/bin/env python3
"""
Generate publication-quality vector plots for Knowledge Calibration Top Performers.

This script creates the same bar plots displayed on the web interface but as
publication-ready vector graphics (SVG/PDF) with proper styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pandas as pd
import sqlite3
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import os
import urllib.request
import urllib.error

# Define colors matching the web interface
COLORS = {
    'extensive': '#d4edda',  # Green gradient base
    'moderate': '#d1ecf1',   # Blue gradient base
    'limited': '#fff3cd',    # Yellow gradient base
    'na': '#e2e3e5',         # Gray gradient base
    'failed': '#f8d7da'      # Red gradient base (for inference_failed/no_result)
}

# Gradient colors for more sophisticated rendering
GRADIENT_COLORS = {
    'extensive': ('#d4edda', '#c3e6cb'),  # Green gradient
    'moderate': ('#d1ecf1', '#bee5eb'),   # Blue gradient
    'limited': ('#fff3cd', '#ffeaa7'),    # Yellow gradient
    'na': ('#e2e3e5', '#d6d8db'),        # Gray gradient
    'failed': ('#f8d7da', '#f5c6cb')     # Red gradient
}


class KnowledgeCalibrationPlotter:
    """Generate publication-quality plots for knowledge calibration results."""

    def __init__(self, db_path: str = "microbellm.db", api_url: Optional[str] = None, use_api: bool = True):
        """Initialize plotter with optional API usage and database connection.

        If use_api is True, the plotter will attempt to fetch data from the web API
        at api_url (or MICROBELLM_API_URL env var, or http://0.0.0.0:5000 by default),
        and fall back to SQLite if the API is unreachable.
        """
        self.db_path = db_path
        self.api_url = api_url
        self.use_api = use_api
        self.conn = sqlite3.connect(db_path)

    def fetch_knowledge_analysis_data(self) -> Dict:
        """Fetch knowledge analysis data via API if available, else from the database.

        API path: /api/knowledge_analysis_data
        - Base URL precedence: constructor api_url > MICROBELLM_API_URL env > http://0.0.0.0:5000
        - Returns a dict with key 'knowledge_analysis'
        """
        # Try API first if allowed
        if self.use_api:
            base_url = self.api_url or os.environ.get('MICROBELLM_API_URL') or 'http://0.0.0.0:5000'
            endpoint = base_url.rstrip('/') + '/api/knowledge_analysis_data'
            try:
                with urllib.request.urlopen(endpoint, timeout=10) as resp:
                    if resp.status == 200:
                        payload = json.loads(resp.read().decode('utf-8'))
                        if isinstance(payload, dict) and 'knowledge_analysis' in payload:
                            return {'knowledge_analysis': payload['knowledge_analysis']}
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
                # Fall back to DB if API not reachable or invalid
                pass

        # Fall back to querying the database, matching API logic
        query = """
        SELECT
            species_file as file_name,
            user_template as query_template,
            model as model_name,
            knowledge_group as knowledge_level,
            status,
            COUNT(*) as count
        FROM processing_results
        WHERE (knowledge_group IS NOT NULL AND status = 'completed')
           OR status = 'failed'
        GROUP BY species_file, user_template, model, knowledge_group, status
        ORDER BY species_file, user_template, model
        """

        df = pd.read_sql_query(query, self.conn)

        # Transform to nested dictionary structure
        result = {}
        for _, row in df.iterrows():
            file_name = row['file_name'] or 'default'
            template = row['query_template']
            model = row['model_name']
            level = row['knowledge_level']
            status = row['status']
            count = row['count']

            if file_name not in result:
                result[file_name] = {'has_type_column': True, 'types': {'default': {}}}

            type_col = 'default'

            if template not in result[file_name]['types'][type_col]:
                result[file_name]['types'][type_col][template] = {}
            if model not in result[file_name]['types'][type_col][template]:
                result[file_name]['types'][type_col][template][model] = {
                    'NA': 0, 'limited': 0, 'moderate': 0, 'extensive': 0,
                    'no_result': 0, 'inference_failed': 0, 'total': 0
                }

            if status == 'failed':
                result[file_name]['types'][type_col][template][model]['inference_failed'] = count
            elif level:
                normalized_level = str(level).lower().strip()
                if normalized_level in ['limited', 'minimal', 'basic', 'low']:
                    result[file_name]['types'][type_col][template][model]['limited'] = count
                elif normalized_level in ['moderate', 'medium', 'intermediate']:
                    result[file_name]['types'][type_col][template][model]['moderate'] = count
                elif normalized_level in ['extensive', 'comprehensive', 'detailed', 'high', 'full']:
                    result[file_name]['types'][type_col][template][model]['extensive'] = count
                elif normalized_level in ['na', 'n/a', 'n.a.', 'not available', 'not applicable', 'unknown', 'none']:
                    result[file_name]['types'][type_col][template][model]['NA'] = count
                else:
                    result[file_name]['types'][type_col][template][model]['no_result'] = count

            result[file_name]['types'][type_col][template][model]['total'] += count

        return {'knowledge_analysis': result}

    def calculate_model_scores(self, data: Dict) -> List[Dict]:
        """Calculate overall model scores and rankings."""
        model_data = {}

        # Process knowledge analysis data
        for file_name, file_data in data['knowledge_analysis'].items():
            if not file_data.get('has_type_column') or not file_data.get('types'):
                continue

            for input_type, template_data in file_data['types'].items():
                for template_name, models in template_data.items():
                    for model_name, stats in models.items():
                        if model_name not in model_data:
                            model_data[model_name] = {
                                'total_score': 0,
                                'total_samples': 0,
                                'distribution': {
                                    'NA': 0, 'limited': 0, 'moderate': 0,
                                    'extensive': 0, 'no_result': 0,
                                    'inference_failed': 0, 'total': 0
                                }
                            }

                        # Calculate weighted score (higher is worse)
                        na_failed = (stats.get('NA', 0) +
                                   stats.get('no_result', 0) +
                                   stats.get('inference_failed', 0))
                        score = (na_failed * 3 +
                                stats.get('limited', 0) * 2 +
                                stats.get('moderate', 0) * 1 +
                                stats.get('extensive', 0) * 0)

                        model_data[model_name]['total_score'] += score
                        model_data[model_name]['total_samples'] += stats.get('total', 0)

                        # Update distribution
                        for key in ['NA', 'limited', 'moderate', 'extensive',
                                  'no_result', 'inference_failed', 'total']:
                            model_data[model_name]['distribution'][key] += stats.get(key, 0)

        # Calculate average scores and sort
        results = []
        for model_name, data in model_data.items():
            if data['total_samples'] > 0:
                avg_score = data['total_score'] / data['total_samples']
                results.append({
                    'model_name': model_name,
                    'display_name': model_name.split('/')[-1],
                    'average_quality_score': avg_score,
                    'total_samples': data['total_samples'],
                    'distribution': data['distribution']
                })

        # Sort by average quality score (higher is better for knowledge calibration)
        # Higher score = more NA/Failed = better at detecting fake/unknown species
        results.sort(key=lambda x: x['average_quality_score'], reverse=True)
        return results[:4]  # Top 4 performers

    def create_stacked_bar(self, ax, distribution: Dict, title: str,
                           score: float, rank: int, show_legend: bool = False):
        """Create a single stacked horizontal bar chart."""
        total = distribution['total']
        if total == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return

        # Combine failed categories with NA
        segments = [
            ('extensive', distribution.get('extensive', 0), COLORS['extensive']),
            ('moderate', distribution.get('moderate', 0), COLORS['moderate']),
            ('limited', distribution.get('limited', 0), COLORS['limited']),
            ('NA/Failed', (distribution.get('NA', 0) +
                          distribution.get('no_result', 0) +
                          distribution.get('inference_failed', 0)), COLORS['na'])
        ]

        # Create stacked bar
        left = 0
        bar_height = 0.6
        bar_y = 0.2

        for label, count, color in segments:
            if count > 0:
                width = count / total
                rect = Rectangle((left, bar_y), width, bar_height,
                               facecolor=color, edgecolor='white', linewidth=1)
                ax.add_patch(rect)

                # Add percentage labels if segment is large enough
                pct = (count / total) * 100
                if pct > 10:  # Only show label if > 10%
                    ax.text(left + width/2, bar_y + bar_height/2,
                           f'{count}\n({pct:.1f}%)',
                           ha='center', va='center', fontsize=9,
                           fontweight='normal')

                left += width

        # Add rank circle
        circle = plt.Circle((0.05, 0.9), 0.04, color='#7e57c2', zorder=10)
        ax.add_patch(circle)
        ax.text(0.05, 0.9, str(rank), ha='center', va='center',
               color='white', fontsize=11, fontweight='bold', zorder=11)

        # Add title and score
        ax.text(0.12, 0.9, title, fontsize=11, fontweight='bold',
               va='center', color='#333')
        ax.text(0.12, 0.75, f'Avg. Quality Score: {score:.2f}',
               fontsize=9, va='center', color='#666')

        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add legend to first plot only
        if show_legend:
            legend_elements = [
                mpatches.Patch(color=COLORS['extensive'], label='Extensive'),
                mpatches.Patch(color=COLORS['moderate'], label='Moderate'),
                mpatches.Patch(color=COLORS['limited'], label='Limited'),
                mpatches.Patch(color=COLORS['na'], label='NA/Failed')
            ]
            ax.legend(handles=legend_elements, loc='upper right',
                     frameon=False, fontsize=8, ncol=4,
                     bbox_to_anchor=(1, 0.15))

    def generate_top_performers_plot(self, output_path: str = None,
                                    format: str = 'svg', dpi: int = 300):
        """Generate the complete top performers visualization."""
        # Fetch and process data
        data = self.fetch_knowledge_analysis_data()
        top_models = self.calculate_model_scores(data)

        if not top_models:
            print("No model data available")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        fig.suptitle('Top Performing Models - Hallucination Benchmark',
                    fontsize=16, fontweight='bold', y=0.98)

        # Create bar chart for each model
        for i, model in enumerate(top_models):
            self.create_stacked_bar(
                axes[i],
                model['distribution'],
                model['display_name'],
                model['average_quality_score'],
                i + 1,
                show_legend=(i == 0)  # Only show legend on first chart
            )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        if output_path is None:
            output_path = f'top_performers.{format}'

        plt.savefig(output_path, format=format, dpi=dpi,
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved to {output_path}")
        plt.close(fig)

    def generate_comparison_plot(self, output_path: str = None,
                                format: str = 'svg', dpi: int = 300):
        """Generate a simplified comparison plot suitable for papers."""
        # Fetch and process data
        data = self.fetch_knowledge_analysis_data()
        top_models = self.calculate_model_scores(data)

        if not top_models:
            print("No model data available")
            return

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 5))

        models = [m['display_name'] for m in top_models]
        scores = [m['average_quality_score'] for m in top_models]

        # Create bars
        bars = ax.barh(models, scores, color='#7e57c2', alpha=0.8, height=0.6)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', va='center', fontsize=10)

        # Styling
        ax.set_xlabel('Average Quality Score (higher is better)', fontsize=12)
        ax.set_title('Top Performing Models - Hallucination Benchmark',
                    fontsize=14, fontweight='bold', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = f'top_performers_comparison.{format}'

        plt.savefig(output_path, format=format, dpi=dpi,
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Comparison plot saved to {output_path}")
        plt.close(fig)

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate publication-quality plots for Knowledge Calibration Top Performers'
    )
    parser.add_argument('--db', default='microbellm.db',
                       help='Path to the database file')
    parser.add_argument('--api-url', default=None,
                       help='Base URL for the running web API (e.g., http://0.0.0.0:5000)')
    parser.add_argument('--no-api', action='store_true',
                       help='Do not call the web API; query the database directly')
    parser.add_argument('--output', '-o', default='plots/top_performers.pdf',
                       help='Output file path')
    parser.add_argument('--format', '-f', choices=['pdf', 'png'],
                       default='pdf', help='Output format')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for raster formats')
    parser.add_argument('--comparison', action='store_true',
                       help='Also generate simplified comparison plot')

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plotter = KnowledgeCalibrationPlotter(db_path=args.db, api_url=args.api_url, use_api=(not args.no_api))
    try:
        plotter.generate_top_performers_plot(
            str(output_path),
            format=args.format,
            dpi=args.dpi
        )

        if args.comparison:
            comp_path = str(output_path).replace(f'.{args.format}',
                                                f'_comparison.{args.format}')
            plotter.generate_comparison_plot(comp_path,
                                           format=args.format,
                                           dpi=args.dpi)
    finally:
        plotter.close()


if __name__ == '__main__':
    main()