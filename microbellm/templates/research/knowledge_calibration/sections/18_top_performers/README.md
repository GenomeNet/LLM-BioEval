# Knowledge Calibration Top Performers - Plot Generation

This directory contains the plot generation script for the Knowledge Calibration Top Performers section, along with the HTML component.

## Scripts

### generate_top_performer_plots.py

Generates the bar chart visualizations showing model performance distributions across different knowledge levels.

#### Features

- Creates stacked horizontal bar charts matching the web interface design
- Generates both combined and individual model plots
- Outputs publication-ready PDF format (default)
- Maintains consistent color scheme with the web interface
- Includes proper legends and labels for publication use
- Uses web API for data consistency with the HTML interface

#### Usage

Basic usage:
```bash
# Generate PDF plots (default)
python generate_top_performer_plots.py --api-url http://localhost:5050

# Specify output path
python generate_top_performer_plots.py -o output/top_models.pdf

# Generate additional comparison plot
python generate_top_performer_plots.py --comparison

# Use database directly instead of API
python generate_top_performer_plots.py --no-api --db /path/to/microbellm.db
```

#### Command Line Options

- `--api-url URL`: Base URL for the web API (e.g., http://localhost:5050)
- `--no-api`: Use database directly instead of API
- `--db PATH`: Path to the microbellm.db database (default: microbellm.db)
- `--output PATH`, `-o PATH`: Output file path (default: plots/top_performers.pdf)
- `--format FORMAT`, `-f FORMAT`: Output format - pdf or png (default: pdf)
- `--dpi DPI`: DPI for raster formats (default: 300)
- `--comparison`: Generate additional simplified comparison plot

#### Output Files

The script generates multiple PDF files:
1. `top_performers.pdf` - Combined plot with all 4 top models
2. `top_performers_model_1.pdf` - Individual plot for rank 1 model
3. `top_performers_model_2.pdf` - Individual plot for rank 2 model
4. `top_performers_model_3.pdf` - Individual plot for rank 3 model
5. `top_performers_model_4.pdf` - Individual plot for rank 4 model
6. `top_performers_comparison.pdf` - Simplified bar chart (if --comparison used)

#### Color Scheme

The plots use the same color scheme as the web interface:
- **Extensive** (Green): #d4edda - Model provides in-depth, reference-level information
- **Moderate** (Blue): #d1ecf1 - Model gives several specific details
- **Limited** (Yellow): #fff3cd - Model knows only basic facts
- **NA/Failed** (Gray): #e2e3e5 - No information or inference failed

#### Requirements

- matplotlib
- numpy
- pandas
- sqlite3 (included with Python)

Install requirements:
```bash
pip install matplotlib numpy pandas
```

## Directory Structure

```
microbellm/templates/research/knowledge_calibration/sections/18_top_performers/
├── README.md                          # This file
├── generate_top_performer_plots.py    # Main plotting script
└── output/                            # Default output directory (created automatically)
```

## Important Notes on Scoring

For Knowledge Calibration (artificial/untrue data detection):
- **Higher average quality score = better performance**
- Higher score means more NA/Failed responses, which is desired for fake species
- The web interface sorts by descending score (highest first)
- This is opposite to phenotype analysis where lower scores are better

## Integration with Paper/Publication

The generated PDF files can be directly included in LaTeX documents:

```latex
\usepackage{graphicx}
\includegraphics[width=\textwidth]{figures/top_performers.pdf}
```

The PDF format ensures:
- Vector quality at any scale
- Perfect compatibility with LaTeX
- Consistent rendering across platforms
- Publication-ready output

## Data Source

The script reads directly from the `microbellm.db` SQLite database, specifically from the `predictions` table. It calculates the same metrics as the web interface to ensure consistency.