#!/bin/bash

echo "Setting up MicrobeLLM..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml

echo "Activating environment..."
conda activate microbellm

# Install the package
echo "Installing MicrobeLLM..."
pip install -e .

echo ""
echo "Installation complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate microbellm"
echo ""
echo "To test the installation, run:"
echo "  microbellm single --binomial_name 'Escherichia coli' --system_template templates/system/template1.txt --user_template templates/user/template1.txt --output test_results.csv --verbose"
echo ""
echo "Don't forget to set your API key:"
echo "  export OPENROUTER_API_KEY='your_api_key_here'"
echo "" 