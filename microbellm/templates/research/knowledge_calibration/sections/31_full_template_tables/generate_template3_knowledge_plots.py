#!/usr/bin/env python3
"""
Generate publication-quality stacked bar plots for Template 3 Knowledge distribution.
Fetches data from the full_template_tables API endpoint.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

def fetch_template_data(api_url: str = None, template: str = 'template3_knowledge') -> Dict:
    """Fetch template table data from the API."""
    base_url = api_url or os.environ.get('MICROBELLM_API_URL') or 'http://localhost:5050'
    endpoint = base_url.rstrip('/') + '/components/knowledge_calibration/full_template_tables'

    print(f"Fetching data from: {endpoint}")

    try:
        with urllib.request.urlopen(endpoint, timeout=30) as resp:
            if resp.status == 200:
                html_content = resp.read().decode('utf-8')
                # Extract the data from the HTML response
                # The API returns HTML with embedded data, we need to parse it
                return extract_data_from_html(html_content, template)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"Error fetching from API: {e}")
        return None

    return None

def extract_data_from_html(html_content: str, template: str) -> Dict:
    """Extract data from the HTML response for a specific template."""
    # Look for the specific template section
    import re

    # Find the template3_knowledge section
    template_pattern = f'id="{template}"'
    if template_pattern not in html_content:
        print(f"Template {template} not found in response")
        return None

    # Extract model data from the table rows
    # This is a simplified extraction - you may need to adjust based on actual HTML structure
    model_data = {}

    # Pattern to find table rows with model data
    row_pattern = r'<tr[^>]*>(.*?)</tr>'
    rows = re.findall(row_pattern, html_content, re.DOTALL)

    for row in rows:
        if 'bar-chart-segments' in row:
            # Extract model name
            model_match = re.search(r'<td[^>]*>([^<]+)</td>', row)
            if model_match:
                model_name = model_match.group(1).strip()

                # Extract percentages from bar segments
                na_match = re.search(r'class="bar-segment na"[^>]*>([0-9.]+)%', row)
                limited_match = re.search(r'class="bar-segment limited"[^>]*>([0-9.]+)%', row)
                moderate_match = re.search(r'class="bar-segment moderate"[^>]*>([0-9.]+)%', row)
                extensive_match = re.search(r'class="bar-segment extensive"[^>]*>([0-9.]+)%', row)

                # Extract sample size
                sample_match = re.search(r'<td[^>]*>(\d+)</td>\s*</tr>', row)

                if model_name and model_name not in ['Model', 'Total']:
                    model_data[model_name] = {
                        'na': float(na_match.group(1)) if na_match else 0,
                        'limited': float(limited_match.group(1)) if limited_match else 0,
                        'moderate': float(moderate_match.group(1)) if moderate_match else 0,
                        'extensive': float(extensive_match.group(1)) if extensive_match else 0,
                        'sample_size': int(sample_match.group(1)) if sample_match else 0
                    }

    return model_data

def fetch_template_data_api(api_url: str = None) -> Dict:
    """Fetch template data directly from the knowledge analysis API endpoint."""
    base_url = api_url or os.environ.get('MICROBELLM_API_URL') or 'http://localhost:5050'
    endpoint = base_url.rstrip('/') + '/api/knowledge_analysis_data'

    print(f"Fetching data from: {endpoint}")

    try:
        with urllib.request.urlopen(endpoint, timeout=30) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                return process_knowledge_analysis_data(data)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
        print(f"Error fetching from API: {e}")
        return None

    return None

def process_knowledge_analysis_data(data: Dict) -> Dict:
    """Process knowledge analysis data from API to extract template3_knowlege."""
    if not data or 'knowledge_analysis' not in data:
        print("No knowledge_analysis in API response")
        return None

    knowledge_analysis = data['knowledge_analysis']

    # Aggregate data across all files and types
    aggregated_model_data = {}

    # Process each file
    for file_name, file_data in knowledge_analysis.items():
        if not file_data.get('has_type_column'):
            continue

        types_data = file_data.get('types', {})

        # Process each type (e.g., latin_genus_real_strain)
        for type_name, templates_data in types_data.items():
            # Look for template3_knowlege
            if 'template3_knowlege' not in templates_data:
                continue

            template3_data = templates_data['template3_knowlege']

            # Aggregate data for each model
            for model_name, model_stats in template3_data.items():
                if model_name not in aggregated_model_data:
                    aggregated_model_data[model_name] = {
                        'na': 0,
                        'limited': 0,
                        'moderate': 0,
                        'extensive': 0,
                        'no_result': 0,
                        'inference_failed': 0
                    }

                # Add counts
                for key in ['NA', 'limited', 'moderate', 'extensive', 'no_result', 'inference_failed']:
                    if key in model_stats:
                        # Handle both 'NA' and 'na' keys
                        agg_key = key.lower() if key != 'NA' else 'na'
                        aggregated_model_data[model_name][agg_key] += model_stats[key]

    # Build the result structure
    result = {
        'template3_knowlege': {
            'rows': []
        }
    }

    # Process aggregated data
    for model_name, counts in aggregated_model_data.items():
        total_na = counts['na'] + counts.get('no_result', 0) + counts.get('inference_failed', 0)
        total_limited = counts['limited']
        total_moderate = counts['moderate']
        total_extensive = counts['extensive']
        total_samples = total_na + total_limited + total_moderate + total_extensive

        if total_samples > 0:
            row_data = {
                'model': model_name.split('/')[-1][:40],  # Shorten model names
                'sample_size': total_samples,
                'distribution': {
                    'na_pct': (total_na / total_samples) * 100,
                    'limited_pct': (total_limited / total_samples) * 100,
                    'moderate_pct': (total_moderate / total_samples) * 100,
                    'extensive_pct': (total_extensive / total_samples) * 100,
                    'na': total_na,
                    'limited': total_limited,
                    'moderate': total_moderate,
                    'extensive': total_extensive,
                }
            }
            result['template3_knowlege']['rows'].append(row_data)

    print(f"Processed {len(result['template3_knowlege']['rows'])} models")
    return result

def parse_template3_from_html(html_content: str) -> Dict:
    """Parse template3_knowlege data from HTML content."""
    import re

    # Look for template3_knowlege (note the typo in 'knowlege')
    if 'template3_knowlege' not in html_content:
        print("template3_knowlege not found in HTML")
        return None

    # Find the template3 section
    template3_start = html_content.find('template3_knowlege')

    # Find the next table after template3_knowlege
    table_start = html_content.find('<table', template3_start)
    if table_start == -1:
        print("No table found for template3")
        return None

    table_end = html_content.find('</table>', table_start) + 8
    table_html = html_content[table_start:table_end]

    # Extract rows
    row_pattern = r'<tr[^>]*>(.*?)</tr>'
    rows = re.findall(row_pattern, table_html, re.DOTALL)

    template_data = {
        'template3_knowlege': {
            'rows': []
        }
    }

    for row in rows[1:]:  # Skip header row
        # Extract model name
        model_match = re.search(r'<td[^>]*>([^<]+)</td>', row)
        if not model_match:
            continue

        model_name = model_match.group(1).strip()

        if model_name in ['Model', 'Total', '']:
            continue

        # Extract percentages from bar segments
        na_match = re.search(r'class="bar-segment na"[^>]*>([0-9.]+)%', row)
        limited_match = re.search(r'class="bar-segment limited"[^>]*>([0-9.]+)%', row)
        moderate_match = re.search(r'class="bar-segment moderate"[^>]*>([0-9.]+)%', row)
        extensive_match = re.search(r'class="bar-segment extensive"[^>]*>([0-9.]+)%', row)

        # Extract sample size - it's the last <td> in the row
        td_matches = re.findall(r'<td[^>]*>([^<]+)</td>', row)
        sample_size = 0
        if len(td_matches) >= 3:
            try:
                sample_size = int(td_matches[-1].strip())
            except:
                pass

        # Build row data
        row_data = {
            'model': model_name,
            'sample_size': sample_size,
            'distribution': {
                'na_pct': float(na_match.group(1)) if na_match else 0,
                'limited_pct': float(limited_match.group(1)) if limited_match else 0,
                'moderate_pct': float(moderate_match.group(1)) if moderate_match else 0,
                'extensive_pct': float(extensive_match.group(1)) if extensive_match else 0,
                'na': int(sample_size * float(na_match.group(1)) / 100) if na_match else 0,
                'limited': int(sample_size * float(limited_match.group(1)) / 100) if limited_match else 0,
                'moderate': int(sample_size * float(moderate_match.group(1)) / 100) if moderate_match else 0,
                'extensive': int(sample_size * float(extensive_match.group(1)) / 100) if extensive_match else 0,
            }
        }

        template_data['template3_knowlege']['rows'].append(row_data)

    return template_data

def create_template3_stacked_plot(api_url: str, output_path: str = 'template3_knowledge_stacked.pdf',
                                  top_n: int = None, sort_by: str = 'extensive',
                                  min_samples: int = 0, clean_version: bool = False):
    """Create a stacked bar plot for Template 3 Knowledge distribution.

    Args:
        api_url: API URL for fetching data
        output_path: Output file path
        top_n: Number of models to display (None = all models)
        sort_by: How to sort models ('extensive', 'na', 'total', 'name')
        min_samples: Minimum sample size to include model
        clean_version: If True, creates a clean version without numbers and legend
    """

    # Fetch data from the template tables API
    data = fetch_template_data_api(api_url)
    if not data:
        print("No data available from API")
        return

    # Look for template3_knowlege in the data (note the typo)
    template3_data = None
    for template_id, template_info in data.items():
        if 'template3_knowlege' in template_id.lower() or 'template3' in template_id.lower():
            template3_data = template_info
            break

    if not template3_data or 'rows' not in template3_data:
        print("No template3_knowlege data found")
        print(f"Available templates: {list(data.keys())}")
        return

    # Process the data
    model_results = []

    for row in template3_data['rows']:
        if row.get('is_total'):
            continue

        model_name = row.get('model', '')
        if not model_name:
            continue

        # Get short name for display
        short_name = model_name.split('/')[-1][:30]  # Truncate long names

        # Get distribution data
        dist = row.get('distribution', {})
        sample_size = row.get('sample_size', 0)

        if sample_size > 0:
            model_results.append({
                'name': short_name,
                'na_pct': dist.get('na_pct', 0),
                'limited_pct': dist.get('limited_pct', 0),
                'moderate_pct': dist.get('moderate_pct', 0),
                'extensive_pct': dist.get('extensive_pct', 0),
                'na_count': dist.get('na', 0),
                'limited_count': dist.get('limited', 0),
                'moderate_count': dist.get('moderate', 0),
                'extensive_count': dist.get('extensive', 0),
                'total': sample_size
            })

    if not model_results:
        print("No model data to plot")
        return

    # Filter by minimum sample size
    if min_samples > 0:
        model_results = [m for m in model_results if m['total'] >= min_samples]
        print(f"Filtered to {len(model_results)} models with >= {min_samples} samples")

    # Sort models based on criteria
    if sort_by == 'extensive':
        model_results.sort(key=lambda x: x['extensive_pct'], reverse=True)
        sort_desc = "extensive knowledge %"
    elif sort_by == 'na':
        model_results.sort(key=lambda x: x['na_pct'], reverse=True)
        sort_desc = "NA/Unknown %"
    elif sort_by == 'total':
        model_results.sort(key=lambda x: x['total'], reverse=True)
        sort_desc = "total sample size"
    elif sort_by == 'name':
        model_results.sort(key=lambda x: x['name'].lower())
        sort_desc = "alphabetical order"
    else:
        model_results.sort(key=lambda x: x['extensive_pct'], reverse=True)
        sort_desc = "extensive knowledge %"

    # Take specified number of models or all
    if top_n is not None:
        top_models = model_results[:top_n]
        print(f"\nShowing top {min(top_n, len(top_models))} models sorted by {sort_desc}")
    else:
        top_models = model_results
        print(f"\nShowing all {len(top_models)} models sorted by {sort_desc}")

    print(f"First 5 models:")
    for i, model in enumerate(top_models[:5], 1):
        print(f"{i}. {model['name']}: Extensive={model['extensive_pct']:.1f}%, NA={model['na_pct']:.1f}%, n={model['total']}")

    # Prepare data for plotting
    model_names = [m['name'] for m in top_models]
    na_values = [m['na_pct'] for m in top_models]
    limited_values = [m['limited_pct'] for m in top_models]
    moderate_values = [m['moderate_pct'] for m in top_models]
    extensive_values = [m['extensive_pct'] for m in top_models]

    # Create the plot - adjust size based on number of models
    if clean_version:
        # 3cm x 3cm (approximately 1.18 inches each)
        fig_width = 1.18  # 3cm in inches
        fig_height = 1.18  # 3cm in inches
    else:
        if len(top_models) > 30:
            fig_height = 12 + (len(top_models) - 30) * 0.2
        elif len(top_models) > 15:
            fig_height = 10
        else:
            fig_height = 8
        fig_width = 14

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Define bar parameters - make bars wider for clean version
    if clean_version:
        bar_height = 1.0  # No spacing between bars
    else:
        bar_height = 0.7  # Original spacing
    y_positions = np.arange(len(model_names))

    # Define colors matching the web interface
    colors = {
        'na': '#9CA3AF',      # Gray
        'limited': '#FBBF24',  # Yellow
        'moderate': '#60A5FA', # Blue
        'extensive': '#34D399' # Green
    }

    # Create stacked horizontal bars
    # Remove edge lines for clean version
    edge_color = 'none' if clean_version else 'white'
    edge_width = 0 if clean_version else 1

    p1 = ax.barh(y_positions, na_values, bar_height,
                 label='NA/Unknown', color=colors['na'], edgecolor=edge_color, linewidth=edge_width)

    p2 = ax.barh(y_positions, limited_values, bar_height, left=na_values,
                 label='Limited', color=colors['limited'], edgecolor=edge_color, linewidth=edge_width)

    # Calculate cumulative positions
    left_moderate = [na + lim for na, lim in zip(na_values, limited_values)]
    p3 = ax.barh(y_positions, moderate_values, bar_height, left=left_moderate,
                 label='Moderate', color=colors['moderate'], edgecolor=edge_color, linewidth=edge_width)

    left_extensive = [na + lim + mod for na, lim, mod in
                     zip(na_values, limited_values, moderate_values)]
    p4 = ax.barh(y_positions, extensive_values, bar_height, left=left_extensive,
                 label='Extensive', color=colors['extensive'], edgecolor=edge_color, linewidth=edge_width)

    # Add percentage labels on segments (only if not clean version)
    if not clean_version:
        for i in range(len(model_names)):
            cumulative = 0
            segments = [
                (na_values[i], top_models[i]['na_count'], 'NA'),
                (limited_values[i], top_models[i]['limited_count'], 'Limited'),
                (moderate_values[i], top_models[i]['moderate_count'], 'Moderate'),
                (extensive_values[i], top_models[i]['extensive_count'], 'Extensive')
            ]

            for pct_value, count, label in segments:
                if pct_value > 5:  # Only show if > 5%
                    text_label = f'{pct_value:.0f}%'
                    # Add count for larger segments
                    if pct_value > 15:
                        text_label = f'{pct_value:.0f}%\n({count})'

                    text_color = 'white' if label in ['NA', 'Extensive'] else 'black'
                    ax.text(cumulative + pct_value/2, i, text_label,
                           ha='center', va='center', fontsize=8,
                           fontweight='bold', color=text_color)
                cumulative += pct_value

        # Add sample size annotations on the right
        for i, model in enumerate(top_models):
            ax.text(102, i, f'n={model["total"]}', fontsize=9,
                   va='center', ha='left', color='#666666')

    # Customize the plot
    if clean_version:
        # Remove y-axis labels for tiny plot
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names, fontsize=10)

    if not clean_version:
        ax.set_xlabel('Knowledge Distribution (%)', fontsize=12, fontweight='bold')
        ax.set_title('Template 3: Knowledge Distribution by Model',
                    fontsize=14, fontweight='bold', pad=20)
    else:
        # Remove axis labels for clean version
        ax.set_xlabel('')
        ax.set_title('')

    # Set xlim based on clean version
    if clean_version:
        ax.set_xlim(0, 100)  # Exact 100% for clean version
    else:
        ax.set_xlim(0, 108)  # Extended to show sample size

    # Add legend (only if not clean version)
    if not clean_version:
        ax.legend(loc='lower right', frameon=True, fontsize=10,
                 title='Knowledge Level', title_fontsize=11,
                 fancybox=True, shadow=True)

    # Add grid (only if not clean version)
    if not clean_version:
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    # Style the spines
    if clean_version:
        # Remove all spines and ticks for clean version
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])  # Remove x-axis ticks
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

    # Add a subtitle with template information (only if not clean version)
    if not clean_version:
        template_desc = "Question: 'What is the natural habitat of {species}?'"
        fig.text(0.5, 0.02, template_desc, ha='center', fontsize=10,
                style='italic', color='#666666')

    if clean_version:
        # Remove all padding for tiny plot
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save the plot
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0 if clean_version else 0.1)
    print(f"\nStacked bar plot saved to {output_path}")

    # Skip detailed plot for clean version
    if clean_version:
        plt.close('all')
        return

    # Create a second plot focusing on top 5 models with more detail
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    top5 = top_models[:5]
    model_names_top5 = [m['name'] for m in top5]
    y_pos = np.arange(len(model_names_top5))

    # Create grouped bars for better comparison
    bar_width = 0.2

    na_vals = [m['na_pct'] for m in top5]
    lim_vals = [m['limited_pct'] for m in top5]
    mod_vals = [m['moderate_pct'] for m in top5]
    ext_vals = [m['extensive_pct'] for m in top5]

    ax2.barh(y_pos - bar_width*1.5, na_vals, bar_width,
            label='NA/Unknown', color=colors['na'])
    ax2.barh(y_pos - bar_width*0.5, lim_vals, bar_width,
            label='Limited', color=colors['limited'])
    ax2.barh(y_pos + bar_width*0.5, mod_vals, bar_width,
            label='Moderate', color=colors['moderate'])
    ax2.barh(y_pos + bar_width*1.5, ext_vals, bar_width,
            label='Extensive', color=colors['extensive'])

    # Add value labels
    for i, values in enumerate([na_vals, lim_vals, mod_vals, ext_vals]):
        offset = -bar_width*1.5 + i*bar_width
        for j, val in enumerate(values):
            if val > 0:
                ax2.text(val + 1, j + offset, f'{val:.1f}%',
                        va='center', fontsize=9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names_top5, fontsize=11)
    ax2.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Top 5 Models - Template 3 Knowledge Distribution (Detailed View)',
                 fontsize=13, fontweight='bold', pad=20)

    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    output_detailed = output_path.replace('.pdf', '_top5_detailed.pdf')
    plt.tight_layout()
    plt.savefig(output_detailed, format='pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Detailed top 5 plot saved to {output_detailed}")

    plt.close('all')

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate Template 3 Knowledge distribution plots'
    )
    parser.add_argument('--api-url', default='http://localhost:5050',
                       help='Base URL for the running web API')
    parser.add_argument('--output', '-o',
                       default='template3_knowledge_stacked.pdf',
                       help='Output file path')
    parser.add_argument('--top-n', type=int, default=None,
                       help='Number of models to display (default: all models)')
    parser.add_argument('--sort-by', choices=['extensive', 'na', 'total', 'name'],
                       default='extensive',
                       help='How to sort models (default: extensive)')
    parser.add_argument('--min-samples', type=int, default=0,
                       help='Minimum sample size to include model (default: 0 = all)')
    parser.add_argument('--all-models', action='store_true',
                       help='Show all models (overrides --top-n)')
    parser.add_argument('--clean', action='store_true',
                       help='Create clean version without numbers and legend')

    args = parser.parse_args()

    # Handle --all-models flag
    if args.all_models:
        args.top_n = None

    print(f"Generating Template 3 Knowledge distribution plot")
    print(f"- API URL: {args.api_url}")
    if args.top_n is not None:
        print(f"- Top N models: {args.top_n}")
    else:
        print(f"- Showing: ALL models")
    print(f"- Sort by: {args.sort_by}")
    if args.min_samples > 0:
        print(f"- Min samples: {args.min_samples}")
    if args.clean:
        print(f"- Clean version: YES (no numbers/legend)")

    create_template3_stacked_plot(args.api_url, args.output, args.top_n,
                                args.sort_by, args.min_samples, args.clean)

if __name__ == '__main__':
    main()