# MicrobeLLM

[![DOI](https://zenodo.org/badge/851077612.svg)](https://zenodo.org/doi/10.5281/zenodo.13839818)

MicrobeLLM is a Python tool designed to evaluate Large Language Models (LLMs) on their ability to predict microbial phenotypes. This tool helps researchers assess how well different LLMs perform on microbiological tasks by querying them with bacterial species names and comparing their predictions against known characteristics.

## Key Features

- **Model Evaluation**: Test multiple LLMs on microbial phenotype prediction tasks
- **Web Interface**: Real-time monitoring and management of prediction jobs
- **Batch Processing**: Evaluate models on large datasets with parallel processing
- **OpenRouter Integration**: Access to 20+ LLMs through a single API
- **Job Management**: Pause, resume, and track progress of multiple concurrent jobs

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/GenomeNet/microbeLLM.git
cd microbeLLM

# Create conda environment and install
conda env create -f environment.yml
conda activate microbellm
pip install -e .
```

### 2. Set up API Key

```bash
export OPENROUTER_API_KEY='your-openrouter-api-key'
```

### 3. Launch Web Interface

```bash
microbellm-web
```

Open your browser to `http://localhost:5000`

## Using the Web Interface

### Creating a Job
1. Click "Create New Job" on the dashboard
2. Enter a job name
3. Add species names (one per line)
4. Add model names (e.g., `anthropic/claude-3.5-sonnet`)
5. Specify template file paths
6. Click "Create Job" then "Start Job"

### Features
- **Real-time Progress**: Monitor job progress with live updates
- **Multi-Model Support**: Run predictions across different LLMs simultaneously
- **Error Tracking**: View detailed error logs and statistics
- **Resume Capability**: Jobs automatically skip completed tasks when restarted

## Admin Dashboard

View and manage results:

```bash
microbellm-admin --debug
```

Access at `http://localhost:5050`

## Testing

Run admin functionality tests:

```bash
python test_admin.py
```

This tests database operations, API endpoints, and UI functionality.


## Input/Output

**Input CSV format:**
```csv
Binomial.name
Escherichia coli
Bacillus subtilis
Staphylococcus aureus
```

**Output includes predictions for:**
- Oxygen requirements
- Gram staining
- Spore formation
- Motility
- Biosafety level
- Cell shape
- And more...

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{microbellm2024,
  title={MicrobeLLM: Evaluating Large Language Models on Microbial Phenotype Prediction},
  author={GenomeNet Team},
  year={2024},
  url={https://github.com/GenomeNet/microbeLLM}
}
```