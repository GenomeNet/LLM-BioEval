#!/usr/bin/env python3
"""
Generate publication-quality plots for Top Models - Web Alignment (Knowledge-Web Correlation).
Creates visualizations showing models with highest correlation between knowledge levels and web presence.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pandas as pd
import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import os
import urllib.request
import urllib.error

# Define colors matching the web interface
COLORS = {
    'extensive': '#d4edda',  # Green gradient base
    'moderate': '#d1ecf1',   # Blue gradient base  
    'limited': '#fff3cd',    # Yellow gradient base
    'na': '#e2e3e5',         # Gray gradient base
    'failed': '#f8d7da'      # Red gradient base
}

class WebAlignmentPlotter:
    """Generate publication-quality plots for web alignment analysis."""
    
    def __init__(self, api_url: Optional[str] = None, use_api: bool = True):
        """Initialize plotter with API configuration."""
        self.api_url = api_url
        self.use_api = use_api
        
    def fetch_correlation_data(self) -> Dict:
        """Fetch correlation data from the API.
        
        API path: /api/search_count_correlation
        Returns correlation coefficients between knowledge levels and web search counts.
        """
        if self.use_api:
            base_url = self.api_url or os.environ.get('MICROBELLM_API_URL') or 'http://localhost:5050'
            endpoint = base_url.rstrip('/') + '/api/search_count_correlation'
            
            try:
                with urllib.request.urlopen(endpoint, timeout=10) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode('utf-8'))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
                print(f"Error fetching from API: {e}")
                return None
        
        return None
    
    def calculate_top_models(self, data: Dict) -> List[Dict]:
        """Extract and rank models by correlation coefficient."""
        if not data or 'correlation_data' not in data:
            return []
            
        model_results = []
        
        # Process correlation data from all files
        for file_name, file_data in data['correlation_data'].items():
            for template_name, models in file_data.items():
                for model_name, stats in models.items():
                    if stats.get('species_count', 0) < 2:  # Skip if too few data points
                        continue
                        
                    model_results.append({
                        'model_name': model_name,
                        'display_name': model_name.split('/')[-1],
                        'template_name': template_name,
                        'correlation': stats.get('correlation_coefficient', 0),
                        'species_count': stats.get('species_count', 0),
                        'knowledge_distribution': stats.get('knowledge_distribution', {}),
                        'data_points': stats.get('data_points', [])
                    })
        
        # Sort by correlation (descending - higher correlation is better)
        model_results.sort(key=lambda x: x['correlation'], reverse=True)
        return model_results[:4]  # Top 4 performers
    
    def create_correlation_bar(self, ax, model: Dict, rank: int, show_legend: bool = False):
        """Create a single bar visualization for correlation performance."""
        dist = model['knowledge_distribution']
        total = model['species_count']
        
        if total == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return
        
        # Create segments for knowledge distribution
        segments = [
            ('extensive', dist.get('extensive', 0), COLORS['extensive']),
            ('moderate', dist.get('moderate', 0), COLORS['moderate']),
            ('limited', dist.get('limited', 0), COLORS['limited']),
            ('NA', dist.get('NA', 0) + dist.get('no_result', 0) + dist.get('inference_failed', 0), COLORS['na'])
        ]
        
        # Create stacked bar
        left = 0
        bar_height = 0.6
        bar_y = 0.2
        
        for label, count, color in segments:
            if count > 0:
                width = count / total
                rect = Rectangle((left, bar_y), width, bar_height,
                               facecolor=color, edgecolor='#cccccc', linewidth=0.5)
                ax.add_patch(rect)
                
                # Add percentage labels if segment is large enough
                pct = (count / total) * 100
                if pct > 10:  # Only show label if > 10%
                    ax.text(left + width/2, bar_y + bar_height/2,
                           f'{count}\n({pct:.1f}%)',
                           ha='center', va='center', fontsize=9)
                
                left += width
        
        # Add rank circle (blue for correlation alignment)
        circle = plt.Circle((0.05, 0.9), 0.04, color='#3b82f6', zorder=10)
        ax.add_patch(circle)
        ax.text(0.05, 0.9, str(rank), ha='center', va='center',
               color='white', fontsize=11, fontweight='bold', zorder=11)
        
        # Add title and correlation score
        ax.text(0.12, 0.9, model['display_name'], fontsize=11, fontweight='bold',
               va='center', color='#333')
        ax.text(0.12, 0.75, f'Correlation: {model["correlation"]:.3f}',
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
                mpatches.Patch(color=COLORS['na'], label='NA')
            ]
            ax.legend(handles=legend_elements, loc='upper right',
                     frameon=False, fontsize=8, ncol=4,
                     bbox_to_anchor=(1, 0.15))
    
    def generate_web_alignment_plot(self, output_path: str = None,
                                   format: str = 'pdf', dpi: int = 300):
        """Generate the complete web alignment visualization."""
        # Fetch and process data
        data = self.fetch_correlation_data()
        
        if not data:
            print("No correlation data available")
            return
            
        top_models = self.calculate_top_models(data)
        
        if not top_models:
            print("No model correlation data available")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        fig.suptitle('Top Models - Knowledge-Web Alignment',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Create bar chart for each model
        for i, model in enumerate(top_models):
            self.create_correlation_bar(
                axes[i],
                model,
                i + 1,
                show_legend=(i == 0)
            )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        if output_path is None:
            output_path = 'web_alignment_top_models.pdf'
        
        plt.savefig(output_path, format=format, dpi=dpi,
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved to {output_path}")
        plt.close(fig)
    
    def generate_scatter_plot(self, output_path: str = None,
                            format: str = 'pdf', dpi: int = 300):
        """Generate scatter plot showing correlation between knowledge and web presence."""
        data = self.fetch_correlation_data()
        
        if not data:
            print("No correlation data available")
            return
        
        top_models = self.calculate_top_models(data)
        
        if not top_models:
            print("No model correlation data available")  
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Knowledge Level vs Web Presence Correlation',
                    fontsize=16, fontweight='bold')
        
        for idx, (ax, model) in enumerate(zip(axes.flat, top_models[:4])):
            if not model.get('data_points'):
                ax.text(0.5, 0.5, 'No data points', ha='center', va='center')
                ax.set_title(model['display_name'])
                continue
                
            # Extract data points
            data_points = model['data_points']
            search_counts = [p['search_count'] for p in data_points]
            knowledge_scores = [p['knowledge_score'] for p in data_points]
            
            # Create scatter plot
            ax.scatter(search_counts, knowledge_scores, alpha=0.6, s=50, c='#3b82f6')
            
            # Add trend line if enough points
            if len(search_counts) > 1:
                z = np.polyfit(search_counts, knowledge_scores, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(search_counts), max(search_counts), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2)
            
            # Labels and title
            ax.set_xlabel('Web Search Count (log scale)', fontsize=10)
            ax.set_ylabel('Knowledge Score', fontsize=10)
            ax.set_title(f'{model["display_name"]} (r={model["correlation"]:.3f})',
                        fontsize=11, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        if output_path is None:
            output_path = 'web_alignment_scatter.pdf'
            
        plt.savefig(output_path, format=format, dpi=dpi,
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Scatter plot saved to {output_path}")
        plt.close(fig)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate publication-quality plots for Web Alignment Top Models'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='Base URL for the running web API')
    parser.add_argument('--no-api', action='store_true',
                       help='Skip API (will fail as this requires API data)')
    parser.add_argument('--output', '-o', default='plots/web_alignment_top_models.pdf',
                       help='Output file path')
    parser.add_argument('--format', '-f', choices=['pdf', 'png'],
                       default='pdf', help='Output format')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for raster formats')
    parser.add_argument('--scatter', action='store_true',
                       help='Also generate scatter plots')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plotter = WebAlignmentPlotter(api_url=args.api_url, use_api=(not args.no_api))
    
    plotter.generate_web_alignment_plot(
        str(output_path),
        format=args.format,
        dpi=args.dpi
    )
    
    if args.scatter:
        scatter_path = str(output_path).replace('.pdf', '_scatter.pdf')
        plotter.generate_scatter_plot(
            scatter_path,
            format=args.format,
            dpi=args.dpi
        )

if __name__ == '__main__':
    main()