import os

# --- General Configuration ---
# Path to the SQLite database file
DATABASE_PATH = 'microbellm.db'

# --- API Configuration ---
# To use the application, you need an OpenRouter API key.
# You can get one here: https://openrouter.ai/
# It's recommended to set this as an environment variable.
# Example: export OPENROUTER_API_KEY='your_api_key'
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# --- Model Configuration ---
# List of popular or recommended models to feature in the UI.
# You can find more models on https://openrouter.ai/
# Format: 'provider/model_name'
POPULAR_MODELS = [
    'anthropic/claude-3.5-sonnet',
    'anthropic/claude-3-opus',
    'google/gemini-pro-1.5',
    'mistralai/mistral-large',
    'openai/gpt-4o',
    'openai/gpt-4-turbo',
]

# --- Directory Configuration ---
# Paths to directories for species files and templates
SPECIES_DIR = 'data'
TEMPLATES_DIR = 'templates'
SYSTEM_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, 'system')
USER_TEMPLATES_DIR = os.path.join(TEMPLATES_DIR, 'user') 