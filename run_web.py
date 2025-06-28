#!/usr/bin/env python3
"""
Simple script to run the MicrobeBench web interface
"""

import os
import sys
from microbellm.web_app import main

if __name__ == '__main__':
    # Check if we can import required modules
    try:
        import flask
        import flask_socketio
    except ImportError as e:
        print(f"Error: Missing required module: {e}")
        print("Please install dependencies with: pip install flask flask-socketio")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("=" * 50)
        print("WARNING: OPENROUTER_API_KEY not set!")
        print("=" * 50)
        print("You need to set your OpenRouter API key to use the prediction features.")
        print("Get your key from: https://openrouter.ai/keys")
        print("Then set it with: export OPENROUTER_API_KEY='your-api-key'")
        print("=" * 50)
        print()
    
    print("Starting MicrobeBench Web Interface...")
    print("Access the dashboard at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Run the web app
    main() 