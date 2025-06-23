# MicrobeLLM

[![DOI](https://zenodo.org/badge/851077612.svg)](https://zenodo.org/doi/10.5281/zenodo.13839818)

MicrobeLLM is a Python tool designed to evaluate Large Language Models (LLMs) on their ability to predict microbial phenotypes. This tool helps researchers assess how well different LLMs perform on microbiological tasks by querying them with bacterial species names and comparing their predictions against known characteristics.

## Key Features

- **Model Evaluation**: Test multiple LLMs on microbial phenotype prediction tasks
- **Flexible Templates**: Customizable system and user message templates for different evaluation scenarios  
- **Batch Processing**: Evaluate models on large datasets with parallel processing
- **OpenRouter Integration**: Access to a wide variety of LLMs through OpenRouter API
- **Web Interface**: Real-time monitoring and management of prediction jobs through a web dashboard
- **Job Management**: Pause, resume, and track progress of multiple concurrent jobs

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/GenomeNet/microbeLLM.git
cd microbeLLM

# Run the installation script
./install.sh
```

Or manually:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate microbellm

# Install the package
pip install -e .
```

### 2. Set up API Key

You need an OpenRouter API key to use the models:

```bash
export OPENROUTER_API_KEY='your-openrouter-api-key'
```

### 3. Quick Test

Test with a single bacterial species:

```bash
microbellm single \
  --binomial_name "Escherichia coli" \
  --system_template templates/system/template1.txt \
  --user_template templates/user/template1.txt \
  --output_csv test_result.csv \
  --verbose
```

### 4. Web Interface (Recommended)

Launch the web interface for easier job management:

```bash
microbellm-web
```

Then open your browser to `http://localhost:5000` to access the dashboard.

## Usage

### Single Species Prediction

```bash
microbellm single \
  --binomial_name "Bacillus subtilis" \
  --system_template templates/system/template1.txt \
  --user_template templates/user/template1.txt \
  --output_csv single_result.csv \
  --model "anthropic/claude-3.5-sonnet" \
  --temperature 0.0 \
  --verbose
```

### Batch Processing

```bash
microbellm batch \
  --input_csv data/test_species.csv \
  --output_csv batch_results.csv \
  --system_template templates/system/template1.txt \
  --user_template templates/user/template1.txt \
  --model "anthropic/claude-3.5-sonnet" \
  --threads 4 \
  --verbose
```

### Web Interface Usage

The web interface provides a comprehensive dashboard for managing large-scale prediction jobs:

#### Features:
- **Real-time Progress Tracking**: Monitor job progress with live updates
- **Multi-Model Support**: Run predictions across 20+ different LLMs simultaneously
- **Template Management**: Use different prompt templates for comprehensive evaluation
- **Job Control**: Start, pause, and resume jobs with a single click
- **Error Tracking**: View detailed error logs and statistics
- **Resume Capability**: Jobs automatically skip completed tasks when restarted

#### Creating a Job:
1. Click "Create New Job" on the dashboard
2. Enter a descriptive job name
3. Add species names (one per line)
4. Add model names (e.g., `anthropic/claude-3.5-sonnet`)
5. Specify template file paths
6. Click "Create Job" to configure the job
7. Use "Start Job" to begin processing

#### Managing Jobs:
- **Dashboard**: View all jobs and their current status
- **Job Details**: Click on any job to see detailed progress and configuration
- **Real-time Updates**: Progress bars and statistics update automatically
- **Control Buttons**: Start, pause, or resume jobs as needed

## Available Models

MicrobeLLM supports any model available through OpenRouter. Popular choices include:

- `anthropic/claude-3.5-sonnet` (default, excellent reasoning)
- `openai/gpt-4o`
- `google/gemini-pro-1.5`
- `meta-llama/llama-3.1-405b-instruct`

See the [OpenRouter models page](https://openrouter.ai/models) for the full list.

## Input Format

Your input CSV file should have a `Binomial.name` column:

```csv
Binomial.name
Escherichia coli
Bacillus subtilis
Staphylococcus aureus
```

## Output Format

The tool outputs a CSV file with predictions for various microbial characteristics:

- `oxygen_requirements`: Aerobic, anaerobic, facultative, etc.
- `gram_positive`: TRUE/FALSE for Gram staining
- `spore_former`: Whether the organism forms spores
- `motile`: Whether the organism is motile
- `biosafety_level`: Biosafety classification
- `cell_shape`: Morphological characteristics
- And more...

## Template Customization

You can customize the prompts by editing the template files:

- `templates/system/template1.txt`: System message that sets the context
- `templates/user/template1.txt`: User message template with `{binomial_name}` placeholder

## Development

```bash
# Install in development mode
pip install -e .

# Run the web interface
microbellm-web

# Or run with custom settings
microbellm-web --host 0.0.0.0 --port 8080 --debug

# Run tests
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use MicrobeLLM in your research, please cite:

```bibtex
@software{microbellm2024,
  title={MicrobeLLM: Evaluating Large Language Models on Microbial Phenotype Prediction},
  author={GenomeNet Team},
  year={2024},
  url={https://github.com/GenomeNet/microbeLLM}
}
```
